use std::{
    borrow::Cow,
    fmt::{self, Write as _},
    fs::OpenOptions,
    io::{self, BufWriter, IsTerminal, Write},
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering::Relaxed},
        mpsc::{Receiver, RecvTimeoutError, TryRecvError},
    },
    time::Duration,
};

use anyhow::Result;
use indicatif::{HumanDuration, ProgressBar};
use ratatui::{
    backend::IntoCrossterm,
    buffer::Buffer,
    crossterm::{
        cursor::{MoveDown, MoveRight, MoveUp, RestorePosition, SavePosition},
        queue,
        style::{Colors, Print, ResetColor, SetColors},
        terminal,
    },
    layout::{Position, Rect, Size},
    widgets::Widget,
};
use ratatui_image::{
    FontSize, Image, Resize,
    picker::{Picker, cap_parser::QueryStdioOptions},
};
use rustix::{
    event::{PollFd, PollFlags, Timespec},
    termios::{self, OptionalActions, Termios},
};

use crate::{
    decode,
    search::{Hit, Progress, Search},
    walk::Skipped,
};

pub struct ListOpts {
    pub score: bool,
    pub preview: bool,
    pub end: End,
}

/// What ends each path on stdin and stdout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum End {
    Newline,
    Nul,
}

impl End {
    pub fn byte(self) -> u8 {
        match self {
            End::Newline => b'\n',
            End::Nul => b'\0',
        }
    }
}

const THUMB_ROWS: u16 = 10;
const TICK: Duration = Duration::from_millis(100);

pub fn list(search: Search, opts: &ListOpts) -> Result<usize> {
    let mut out = Printer::stdout(opts.end);
    let preview = (opts.preview && out.tty).then(Preview::new).transpose()?;
    // A sorted search sees all images before the first hit, so progress shows until then.
    let spinner = ProgressBar::new_spinner();
    spinner.enable_steady_tick(TICK);
    let mut hit = next(
        &search.hits,
        || Ok(true),
        || {
            let left = time_left(&spinner, &search.progress);
            spinner.set_message(format!("{}{left}", search.status()));
        },
    )?;
    spinner.finish_and_clear();

    let mut printed = 0;
    while let Some(h) = hit {
        if let Some(preview) = &preview {
            preview.show(&mut out.out, &h.path)?;
        }
        if !out.record(h.score.filter(|_| opts.score), &h.path, &h.text)? {
            return Ok(printed);
        }
        printed += 1;
        hit = next(&search.hits, || out.flush(), || {})?;
    }
    out.flush()?;
    Ok(printed)
}

pub fn paths(paths: &[PathBuf], end: End) -> io::Result<()> {
    let mut out = Printer::stdout(end);
    for path in paths {
        if !out.record(None, path, &[])? {
            return Ok(());
        }
    }
    out.flush().map(drop)
}

/// The next hit, or None at the end or once nobody reads stdout. Runs `flush` before it waits,
/// so each hit shows as soon as it comes, and `tick` while it waits.
fn next(
    hits: &Receiver<Result<Hit>>,
    flush: impl FnOnce() -> io::Result<bool>,
    mut tick: impl FnMut(),
) -> Result<Option<Hit>> {
    match hits.try_recv() {
        Ok(hit) => return hit.map(Some),
        Err(TryRecvError::Disconnected) => return Ok(None),
        Err(TryRecvError::Empty) => {}
    }
    if !flush()? {
        return Ok(None);
    }
    loop {
        match hits.recv_timeout(TICK) {
            Ok(hit) => return hit.map(Some),
            Err(RecvTimeoutError::Disconnected) => return Ok(None),
            Err(RecvTimeoutError::Timeout) if stdout_closed() => return Ok(None),
            Err(RecvTimeoutError::Timeout) => tick(),
        }
    }
}

/// Linux flags POLLERR on a pipe once its reader is gone, before a write fails. Without this
/// check, `scry -t x | head -1` would read every image after head exits.
fn stdout_closed() -> bool {
    let stdout = io::stdout();
    let mut fds = [PollFd::new(&stdout, PollFlags::empty())];
    rustix::event::poll(&mut fds, Some(&Timespec::default()))
        .is_ok_and(|_| fds[0].revents().intersects(PollFlags::ERR | PollFlags::HUP))
}

/// Tells why images are missing from the output.
pub fn notes(progress: &Progress, skipped: Option<Skipped>) {
    let count = |n: &AtomicUsize| n.load(Relaxed);
    let files = |n| if n == 1 { "file" } else { "files" };
    let (found, unsupported, errors) = (
        count(&progress.found),
        count(&progress.unsupported),
        count(&progress.errors),
    );
    let mut notes = String::new();
    if unsupported > 0 {
        notes += &format!(
            "scry: skipped {unsupported} HEIC, AVIF, JPEG XL or RAW {}: scry cannot read these \
             formats yet\n",
            files(unsupported)
        );
    }
    if errors > 0 {
        notes += &format!("scry: skipped {errors} unreadable {}\n", files(errors));
    }
    if found + unsupported == 0 {
        notes += "scry: found no image files\n";
        if let Some(skipped) = skipped {
            notes += &skip_tip(skipped);
        }
    }
    // stderr is unbuffered: one write keeps the lines whole next to stdout.
    let _ = io::stderr().write_all(notes.as_bytes());
}

fn skip_tip(skipped: Skipped) -> String {
    let (what, flags) = match skipped {
        Skipped::Hidden => ("hidden files", "--hidden"),
        Skipped::Ignored => ("the paths in .gitignore files", "--no-ignore"),
        Skipped::Both => (
            "hidden files and the paths in .gitignore files",
            "--hidden and --no-ignore",
        ),
    };
    format!("tip: scry skips {what}. Use {flags} to search them.\n")
}

/// Feeds the progress to the estimate in `bar`, and returns text like ", about 3 minutes left"
/// once the estimate has enough data.
pub fn time_left(bar: &ProgressBar, progress: &Progress) -> String {
    let (done, found) = (progress.done.load(Relaxed), progress.found.load(Relaxed));
    bar.set_length(found as u64);
    bar.set_position(done as u64);
    if done == 0 || done == found || bar.elapsed() < Duration::from_secs(3) {
        return String::new();
    }
    format!(", about {} left", HumanDuration(bar.eta()))
}

/// Asks the terminal which image protocol it supports. Kitty can read images from shared memory,
/// so the pixels do not go through the terminal and tmux as text.
pub fn picker() -> Result<Picker> {
    let options = QueryStdioOptions {
        kitty_shared_memory_object: QueryStdioOptions::probe_kitty_smo(),
        ..Default::default()
    };
    Ok(Picker::from_query_stdio_with_options(options)?)
}

/// Writes one record per hit: the score, the path and the text, separated by tabs. Pipes get
/// raw bytes. A terminal gets clean text, and paths without the `./` that the walk puts before
/// paths in the current directory. In a pipe, the `./` keeps a name like `-x.jpg` from passing
/// as an option.
struct Printer<W> {
    out: W,
    tty: bool,
    end: End,
}

impl Printer<BufWriter<io::StdoutLock<'static>>> {
    fn stdout(end: End) -> Self {
        Self {
            out: BufWriter::new(io::stdout().lock()),
            tty: io::stdout().is_terminal(),
            end,
        }
    }
}

impl<W: Write> Printer<W> {
    /// False once nobody reads the output.
    fn record(&mut self, score: Option<f32>, path: &Path, text: &[String]) -> io::Result<bool> {
        has_reader(self.write(score, path, text))
    }

    fn flush(&mut self) -> io::Result<bool> {
        has_reader(self.out.flush())
    }

    fn write(&mut self, score: Option<f32>, path: &Path, text: &[String]) -> io::Result<()> {
        if let Some(score) = score {
            write!(self.out, "{}\t", format_score(score))?;
        }
        if self.tty {
            write!(
                self.out,
                "{}",
                clean_path(path.strip_prefix("./").unwrap_or(path))
            )?;
        } else {
            self.out.write_all(path.as_os_str().as_bytes())?;
        }
        for field in text {
            self.out.write_all(b"\t")?;
            if self.tty {
                write!(self.out, "{}", clean(field))?;
            } else {
                self.out.write_all(field.as_bytes())?;
            }
        }
        self.out.write_all(&[self.end.byte()])
    }
}

/// A broken pipe is the normal end of `scry ... | head`.
fn has_reader(written: io::Result<()>) -> io::Result<bool> {
    match written {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(false),
        Err(e) => Err(e),
    }
}

/// A file name or other untrusted text, made safe to show in a terminal.
pub struct Clean<'a>(Cow<'a, str>);

pub fn clean(s: &str) -> Clean<'_> {
    Clean(Cow::Borrowed(s))
}

pub fn clean_path(path: &Path) -> Clean<'_> {
    Clean(path.to_string_lossy())
}

impl fmt::Display for Clean<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, part) in self.0.split(control).enumerate() {
            if i > 0 {
                f.write_char('?')?;
            }
            f.write_str(part)?;
        }
        Ok(())
    }
}

/// A terminal acts on control characters, so a name could send escape sequences. Bidi
/// controls can make the text around them read in another order.
pub fn control(c: char) -> bool {
    c.is_control() || matches!(c, '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
}

/// Like printf's `%.3g`: 3 significant digits, with an exponent below 0.0001. Most `-q` scores
/// are tiny, and a fixed number of decimals prints them all as 0.
fn format_score(score: f32) -> String {
    // The exponent after rounding to 3 digits, so 0.99999 prints as 1.
    let sci = format!("{score:.2e}");
    let (mantissa, exp) = sci.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("{:e} has an integer exponent");
    if exp < -4 {
        return format!("{}e{exp}", trim_zeros(mantissa));
    }
    trim_zeros(&format!("{score:.*}", (2 - exp).max(0) as usize)).into()
}

fn trim_zeros(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s
    }
}

struct Preview {
    picker: Picker,
    _mode: TtyMode,
}

impl Preview {
    fn new() -> Result<Self> {
        // The terminal answers capability queries on stdin, which may be the piped path list.
        // The walker reads that list from its own duplicate of fd 0.
        if !io::stdin().is_terminal() {
            let tty = OpenOptions::new().read(true).write(true).open("/dev/tty")?;
            rustix::stdio::dup2_stdin(&tty)?;
        }
        let mode = TtyMode(termios::tcgetattr(io::stdin())?);
        Ok(Self {
            picker: picker()?,
            _mode: mode,
        })
    }

    /// An image that fails to decode gets no thumbnail.
    fn show(&self, out: &mut impl Write, path: &Path) -> Result<()> {
        let cells = Size::new(terminal::size()?.0, THUMB_ROWS);
        let side = pixel_side(cells, self.picker.font_size());
        let Ok(img) = decode::load(path, side) else {
            return Ok(());
        };
        let thumb = self.picker.new_protocol(img, cells, Resize::Fit(None))?;
        let mut buf = Buffer::empty(Rect::from((Position::ORIGIN, thumb.size())));
        Image::new(&thumb).render(buf.area, &mut buf);
        Ok(draw(out, &buf)?)
    }
}

/// The capability query turns off echo and line editing. If the terminal does not answer in
/// time, the query returns before it turns them on again. This puts the saved mode back.
struct TtyMode(Termios);

impl Drop for TtyMode {
    fn drop(&mut self) {
        let _ = termios::tcsetattr(io::stdin(), OptionalActions::Now, &self.0);
    }
}

/// An image shrunk to fit a square of this side still fills `cells` at `font`, in either
/// orientation.
pub fn pixel_side(cells: Size, font: FontSize) -> u32 {
    (u32::from(cells.width) * u32::from(font.width))
        .max(u32::from(cells.height) * u32::from(font.height))
}

/// Draws `buf` below the cursor without knowing the cursor's screen position, so no terminal
/// round trip is needed. Image protocols put their escape sequence in the first cell and mark the
/// cells it covers as skipped. The buffer diff leaves those out.
fn draw(out: &mut impl Write, buf: &Buffer) -> io::Result<()> {
    let rows = buf.area.height;
    // Reserve the rows first so that drawing never scrolls.
    queue!(
        out,
        Print("\n".repeat(rows.into())),
        MoveUp(rows),
        SavePosition
    )?;
    let mut next = None;
    for (x, y, cell) in Buffer::empty(buf.area).diff(buf) {
        if next != Some((x, y)) {
            queue!(out, RestorePosition)?;
            if y > 0 {
                queue!(out, MoveDown(y))?;
            }
            if x > 0 {
                queue!(out, MoveRight(x))?;
            }
        }
        next = Some((x + 1, y));
        let colors = Colors::new(cell.fg.into_crossterm(), cell.bg.into_crossterm());
        queue!(out, SetColors(colors), Print(cell.symbol()))?;
    }
    queue!(out, ResetColor, RestorePosition, MoveDown(rows))
}

#[cfg(test)]
mod tests {
    use std::os::unix::ffi::OsStringExt;

    use super::*;

    fn printed(tty: bool, end: End, score: Option<f32>, path: &[u8], text: &[String]) -> Vec<u8> {
        let mut printer = Printer {
            out: Vec::new(),
            tty,
            end,
        };
        let path = PathBuf::from(std::ffi::OsString::from_vec(path.to_vec()));
        assert!(printer.record(score, &path, text).unwrap());
        printer.out
    }

    fn written(path: &[u8], score: Option<f32>, tty: bool, end: End) -> Vec<u8> {
        printed(tty, end, score, path, &[])
    }

    #[test]
    fn terminal_output_is_clean() {
        assert_eq!(
            written(b"a\x1b[2Jb.jpg", None, true, End::Newline),
            b"a?[2Jb.jpg\n"
        );
        assert_eq!(
            written("x\u{9b}y.png".as_bytes(), None, true, End::Nul),
            b"x?y.png\0"
        );
        assert_eq!(
            written(b"a\x1b\xff.jpg", None, false, End::Nul),
            b"a\x1b\xff.jpg\0"
        );
        assert_eq!(
            written("ok/čaj.jpg".as_bytes(), None, true, End::Newline),
            "ok/čaj.jpg\n".as_bytes()
        );
        assert_eq!(
            written("evil\u{202e}gpj.exe".as_bytes(), None, true, End::Newline),
            b"evil?gpj.exe\n"
        );
        assert_eq!(clean("tab\there\u{2066}").to_string(), "tab?here?");
    }

    #[test]
    fn dot_slash_stays_in_pipes() {
        assert_eq!(
            written(b"./-x.jpg", None, false, End::Newline),
            b"./-x.jpg\n"
        );
        assert_eq!(written(b"./-x.jpg", None, true, End::Newline), b"-x.jpg\n");
        assert_eq!(
            written(b"../a.jpg", None, true, End::Newline),
            b"../a.jpg\n"
        );
    }

    #[test]
    fn scores() {
        assert_eq!(
            written(b"a.jpg", Some(0.00012), false, End::Newline),
            b"0.00012\ta.jpg\n"
        );
        assert_eq!(
            written(b"a.jpg", Some(-0.5), false, End::Nul),
            b"-0.5\ta.jpg\0"
        );
        let cases = [
            (0.1287, "0.129"),
            (0.1284, "0.128"),
            (0.005812, "0.00581"),
            (0.9296, "0.93"),
            (1.0, "1"),
            (0.99999, "1"),
            (0.0, "0"),
            (-0.0512, "-0.0512"),
            (0.0001, "0.0001"),
            (1.234e-5, "1.23e-5"),
            (2.0e-9, "2e-9"),
            (2.6e-37, "2.6e-37"),
        ];
        for (score, want) in cases {
            assert_eq!(format_score(score), want, "{score}");
            let back: f32 = want.parse().unwrap();
            assert!(
                (back - score).abs() <= score.abs() * 0.005,
                "{want} parses back"
            );
        }
    }

    #[test]
    fn text_fields_follow_the_path() {
        let text = ["INV-2041".to_string(), "Total \x1b[2J".to_string()];
        assert_eq!(
            printed(true, End::Newline, Some(0.5), b"a.png", &text),
            b"0.5\ta.png\tINV-2041\tTotal ?[2J\n"
        );
    }

    #[test]
    fn broken_pipe_ends_the_output() {
        let broken = io::Error::from(io::ErrorKind::BrokenPipe);
        assert!(!has_reader(Err(broken)).unwrap());
        assert!(has_reader(Err(io::ErrorKind::Other.into())).is_err());
    }
}
