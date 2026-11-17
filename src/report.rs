//! What scry tells besides the matches: progress, downloads, warnings, the end, and errors. People
//! get text on stderr. Programs get JSON Lines on stdout, one object per line, next to the matches.

use std::{
    borrow::Cow,
    cell::Cell,
    io::{self, Read, Write},
    os::unix::ffi::OsStrExt,
    path::Path,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant, SystemTime},
};

use base64::{Engine as _, prelude::BASE64_STANDARD};
use indicatif::{HumanDuration, ProgressBar, ProgressStyle};
use jiff::{Timestamp, civil::DateTime};
use serde::Serialize;

use crate::{
    decode, embed,
    output::{self, clean, clean_path},
    search::{Hit, Progress, Snapshot, Stage},
    walk::Skipped,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Format {
    Text,
    Json,
}

static FORMAT: OnceLock<Format> = OnceLock::new();

/// Sets the format for the whole run, before scry prints anything.
pub fn init(format: Format) {
    FORMAT.set(format).expect("the format is set once");
}

pub fn json() -> bool {
    FORMAT.get() == Some(&Format::Json)
}

/// The schema of `--json`. Programs ignore fields and types that they do not know, so new ones
/// can come in any version.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Event<'a> {
    Download {
        label: &'a str,
        done: u64,
        total: u64,
    },
    Progress(Snapshot),
    Match(Match<'a>),
    Warning {
        code: Code,
        message: &'a str,
        #[serde(flatten)]
        path: Option<PathField<'a>>,
    },
    End {
        matched: usize,
        truncated: bool,
        found: usize,
        unsupported: usize,
        unreadable: usize,
    },
    Error {
        code: Fault,
        message: &'a str,
    },
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Code {
    Unsupported,
    Unreadable,
    GpuFallback,
    NoImages,
    NoCache,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Fault {
    Usage,
    Download,
    Other,
}

/// JSON text must be UTF-8. A path that is not keeps its bytes in `path_bytes`, and `path` is
/// only for display.
#[derive(Serialize)]
struct PathField<'a> {
    path: Cow<'a, str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    path_bytes: Option<String>,
}

impl<'a> PathField<'a> {
    fn new(path: &'a Path) -> Self {
        let bytes =
            (path.to_str().is_none()).then(|| BASE64_STANDARD.encode(path.as_os_str().as_bytes()));
        Self {
            path: path.to_string_lossy(),
            path_bytes: bytes,
        }
    }
}

/// Data fields are always present, and null when unknown. Fields that depend on an option are
/// present only with it.
#[derive(Serialize)]
struct Match<'a> {
    #[serde(flatten)]
    path: PathField<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    score: Option<f32>,
    date: Option<String>,
    taken: Option<String>,
    modified: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    camera: Option<&'a str>,
    gps: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    matches: Option<&'a [String]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
}

impl<'a> Match<'a> {
    fn new(hit: &'a Hit, matches: bool) -> Self {
        let file = hit.file.as_ref();
        let meta = file.and_then(|f| f.meta.as_ref());
        Self {
            path: PathField::new(&hit.path),
            score: hit.score,
            date: file.and_then(|f| f.date()).map(civil),
            taken: meta.and_then(|m| m.exif.taken).map(civil),
            modified: file.and_then(|f| utc(f.modified)),
            width: meta.map(|m| m.width),
            height: meta.map(|m| m.height),
            camera: meta.and_then(|m| m.exif.camera.as_deref()),
            gps: meta.map(|m| m.exif.gps),
            matches: matches.then_some(&hit.matches),
            text: hit.text.as_deref(),
        }
    }
}

/// Wall-clock time without an offset, to the second. EXIF dates have no offset.
fn civil(time: DateTime) -> String {
    time.strftime("%Y-%m-%dT%H:%M:%S").to_string()
}

fn utc(time: SystemTime) -> Option<String> {
    let time = Timestamp::try_from(time).ok()?;
    Some(time.strftime("%Y-%m-%dT%H:%M:%SZ").to_string())
}

/// Writes one line and flushes it. False once nobody reads stdout.
fn emit(event: &Event) -> io::Result<bool> {
    let mut line = serde_json::to_vec(event).expect("an event is valid JSON");
    line.push(b'\n');
    let mut out = io::stdout().lock();
    output::has_reader(out.write_all(&line).and_then(|()| out.flush()))
}

/// `matches`: the search has text patterns.
pub fn hit(hit: &Hit, matches: bool) -> io::Result<bool> {
    emit(&Event::Match(Match::new(hit, matches)))
}

const TICK: Duration = Duration::from_millis(100);

struct Status {
    spinner: Option<ProgressBar>,
    /// The last progress line of `--json`.
    sent: Option<(Snapshot, Instant)>,
}

static STATUS: Mutex<Status> = Mutex::new(Status {
    spinner: None,
    sent: None,
});

/// Shows how far the search is. JSON gets a line when the progress changes: at most 10 a second,
/// and each new stage at once.
pub fn progress(progress: &Progress) {
    let now = progress.snapshot();
    let mut status = STATUS.lock().unwrap();
    if json() {
        let skip = status.sent.is_some_and(|(sent, at)| {
            sent == now || sent.stage == now.stage && at.elapsed() < TICK
        });
        if skip {
            return;
        }
        status.sent = Some((now, Instant::now()));
        let _ = emit(&Event::Progress(now));
        return;
    }
    let spinner = status.spinner.get_or_insert_with(|| {
        let spinner = ProgressBar::new_spinner();
        spinner.enable_steady_tick(TICK);
        spinner
    });
    let text = match now.stage {
        Stage::Load => "loading the models".into(),
        Stage::Check => format!("{}/{} checked", now.done, now.total),
        Stage::Score => format!("{}/{} scored", now.done, now.total),
        Stage::Text => format!("{}/{} read", now.done, now.total),
    };
    spinner.set_message(format!("{text}{}", time_left(spinner, &now)));
}

/// Removes the progress line, so that the terminal shows a match or the picker instead.
pub fn clear() {
    if let Some(spinner) = STATUS.lock().unwrap().spinner.take() {
        spinner.finish_and_clear();
    }
}

/// Feeds the progress to the estimate in `bar`, and returns text like ", about 3 minutes left"
/// once the estimate has enough data.
pub fn time_left(bar: &ProgressBar, now: &Snapshot) -> String {
    bar.set_length(now.total as u64);
    bar.set_position(now.done as u64);
    if now.stage == Stage::Load
        || now.done == 0
        || now.done == now.total
        || bar.elapsed() < Duration::from_secs(3)
    {
        return String::new();
    }
    format!(", about {} left", HumanDuration(bar.eta()))
}

/// The progress of one model download. Used on one thread.
pub struct Download {
    label: &'static str,
    total: u64,
    done: Cell<u64>,
    /// Hidden for `--json`.
    bar: ProgressBar,
    sent: Cell<Option<Instant>>,
}

impl Download {
    pub fn new(label: &'static str, total: u64) -> Self {
        clear();
        let bar = if json() {
            ProgressBar::hidden()
        } else {
            let style = ProgressStyle::with_template(
                "{msg} {bar:30} {decimal_bytes}/{decimal_total_bytes} {eta}",
            )
            .expect("the template is valid");
            ProgressBar::new(total)
                .with_style(style)
                .with_message(label)
        };
        Self {
            label,
            total,
            done: Cell::new(0),
            bar,
            sent: Cell::new(None),
        }
    }

    pub fn set(&self, done: u64) {
        self.done.set(done);
        self.bar.set_position(done);
        if json() && self.sent.get().is_none_or(|at| at.elapsed() >= TICK) {
            self.sent.set(Some(Instant::now()));
            self.send();
        }
    }

    fn send(&self) {
        let _ = emit(&Event::Download {
            label: self.label,
            done: self.done.get(),
            total: self.total,
        });
    }

    /// Counts the bytes that go through `inner`.
    pub fn read<R: Read>(&self, inner: R) -> impl Read {
        Counted {
            inner,
            download: self,
        }
    }

    /// A line for people. It shows also without a terminal, where the bar does not, so a log
    /// tells why a run is slow.
    pub fn note(&self, message: &str) {
        if !json() {
            self.bar.suspend(|| eprintln!("scry: {message}"));
        }
    }

    pub fn finish(self) {
        self.bar.finish_and_clear();
        if json() {
            self.send();
        }
    }
}

struct Counted<'a, R> {
    inner: R,
    download: &'a Download,
}

impl<R: Read> Read for Counted<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.download.set(self.download.done.get() + n as u64);
        Ok(n)
    }
}

/// A note that does not stop the search.
pub fn warn(code: Code, message: &str) {
    if json() {
        let _ = emit(&Event::Warning {
            code,
            message,
            path: None,
        });
    } else {
        eprintln!("scry: {message}");
    }
}

pub fn error(code: Fault, message: &str) {
    if json() {
        let _ = emit(&Event::Error { code, message });
    } else {
        eprintln!("scry: {message}");
    }
}

/// How a search ended, for the notes or the `end` line.
pub struct Summary<'a> {
    pub progress: &'a Progress,
    pub matched: usize,
    pub skipped: Option<Skipped>,
    /// The default limit that cut a list of best matches, which a note mentions.
    pub cut: Option<usize>,
}

/// Tells why images are missing from the output.
pub fn end(summary: &Summary) {
    if json() {
        end_json(summary);
    } else {
        // stderr is unbuffered: one write keeps the lines whole next to stdout.
        let _ = io::stderr().write_all(notes(summary).as_bytes());
    }
}

fn end_json(summary: &Summary) {
    let progress = summary.progress;
    for failure in progress.unreadable.first().iter() {
        let _ = emit(&Event::Warning {
            code: Code::Unreadable,
            message: &failure.reason,
            path: failure.path.as_deref().map(PathField::new),
        });
    }
    let warnings = [
        (Code::Unsupported, unsupported(progress)),
        (Code::GpuFallback, gpu_fallback().map(|(e, _)| e)),
        (Code::NoImages, no_images(summary).map(|(e, _)| e)),
    ];
    for (code, message) in warnings {
        if let Some(message) = message {
            let _ = emit(&Event::Warning {
                code,
                message: &message,
                path: None,
            });
        }
    }
    let _ = emit(&Event::End {
        matched: summary.matched,
        truncated: progress.truncated(),
        found: progress.found(),
        unsupported: progress.unsupported(),
        unreadable: progress.unreadable.count(),
    });
}

fn notes(summary: &Summary) -> String {
    let progress = summary.progress;
    let mut notes = String::new();
    if let Some(cut) = summary.cut.filter(|_| progress.truncated()) {
        notes += &format!("scry: showed the best {cut}. Use -n N or -n all for more.\n");
    }
    if let Some(message) = unsupported(progress) {
        notes += &format!("scry: {message}\n");
    }
    let first = progress.unreadable.first();
    for failure in first.iter() {
        let reason = clean(&failure.reason);
        notes += &match &failure.path {
            Some(path) => format!("scry: cannot read {}: {reason}\n", clean_path(path)),
            None => format!("scry: {reason}\n"),
        };
    }
    let more = progress.unreadable.count() - first.len();
    if more > 0 {
        notes += &format!("scry: skipped {more} more unreadable {}\n", files(more));
    }
    for (message, tip) in [gpu_fallback(), no_images(summary)].into_iter().flatten() {
        notes += &format!("scry: {message}\n");
        if let Some(tip) = tip {
            notes += &format!("tip: {tip}\n");
        }
    }
    notes
}

fn files(n: usize) -> &'static str {
    if n == 1 { "file" } else { "files" }
}

fn unsupported(progress: &Progress) -> Option<String> {
    let n = progress.unsupported();
    (n > 0).then(|| {
        format!(
            "skipped {n} {} {}: scry cannot read these formats yet",
            decode::UNSUPPORTED,
            files(n)
        )
    })
}

fn gpu_fallback() -> Option<(String, Option<String>)> {
    let e = embed::gpu::failure()?;
    Some((
        format!("used the CPU, because the GPU failed: {e}"),
        Some("set SCRY_DEVICE=cpu to skip the GPU.".into()),
    ))
}

fn no_images(summary: &Summary) -> Option<(String, Option<String>)> {
    let progress = summary.progress;
    if progress.found() + progress.unsupported() > 0 {
        return None;
    }
    let tip = summary.skipped.map(|skipped| {
        let (what, flags) = match skipped {
            Skipped::Hidden => ("hidden files", "--hidden"),
            Skipped::Ignored => ("the paths in .gitignore files", "--no-ignore"),
            Skipped::Both => (
                "hidden files and the paths in .gitignore files",
                "--hidden and --no-ignore",
            ),
        };
        format!("scry skips {what}. Use {flags} to search them.")
    });
    Some(("found no image files".into(), tip))
}

#[cfg(test)]
mod tests {
    use std::{ffi::OsStr, path::PathBuf};

    use super::*;
    use crate::search::Hit;

    fn line(event: &Event) -> String {
        serde_json::to_string(event).unwrap()
    }

    #[test]
    fn paths_that_are_not_utf8_keep_their_bytes() {
        let hit = Hit::new(PathBuf::from(OsStr::from_bytes(b"a\xffb.jpg")));
        let json = line(&Event::Match(Match::new(&hit, false)));
        assert!(
            json.starts_with(r#"{"type":"match","path":"a�b.jpg","path_bytes":"Yf9iLmpwZw==""#),
            "{json}"
        );
        let hit = Hit::new("ok.jpg".into());
        let json = line(&Event::Match(Match::new(&hit, true)));
        assert_eq!(
            json,
            r#"{"type":"match","path":"ok.jpg","date":null,"taken":null,"modified":null,"width":null,"height":null,"camera":null,"gps":null,"matches":[]}"#
        );
    }

    #[test]
    fn warnings_name_their_file() {
        let path = Path::new("x.png");
        let json = line(&Event::Warning {
            code: Code::Unreadable,
            message: "bad",
            path: Some(PathField::new(path)),
        });
        assert_eq!(
            json,
            r#"{"type":"warning","code":"unreadable","message":"bad","path":"x.png"}"#
        );
        let json = line(&Event::Warning {
            code: Code::NoCache,
            message: "m",
            path: None,
        });
        assert_eq!(
            json,
            r#"{"type":"warning","code":"no_cache","message":"m"}"#
        );
    }
}
