use std::{
    borrow::Cow,
    fmt::{self, Write as _},
    fs::File,
    io::{self, BufWriter, IsTerminal, Write},
    os::{
        fd::{AsFd, BorrowedFd, OwnedFd},
        unix::ffi::OsStrExt,
    },
    path::{Path, PathBuf},
    sync::{
        LazyLock,
        mpsc::{Receiver, RecvTimeoutError, TryRecvError},
    },
    time::Duration,
};

use anyhow::Result;
use ratatui::{
    backend::IntoCrossterm,
    buffer::Buffer,
    crossterm::{
        cursor::{MoveDown, MoveRight, MoveUp, RestorePosition, SavePosition},
        queue,
        style::{Colors, Print, ResetColor, SetColors, available_color_count},
        terminal,
    },
    layout::{Position, Rect, Size},
    style::Color,
    widgets::Widget,
};
use ratatui_image::{
    FontSize, Image, Resize,
    picker::{Picker, ProtocolType, cap_parser::QueryStdioOptions},
};
use rustix::{
    event::{PollFd, PollFlags, Timespec},
    fs::{OFlags, fcntl_getfl},
    stdio,
    termios::{self, OptionalActions, Termios},
};

use crate::{
    decode, report,
    search::{FileInfo, Hit, Search},
};

pub struct ListOpts {
    pub out: Out,
    /// The default limit, if it applies. The notes tell when it cuts the list.
    pub cut: Option<usize>,
}

pub enum Out {
    Lines {
        score: bool,
        preview: bool,
        shown: Shown,
        end: End,
    },
    /// `matches`: the search has text patterns.
    Json { matches: bool },
}

/// What follows the path in a line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shown {
    Nothing,
    Text,
    Matches,
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

/// Returns the number of printed matches.
pub fn list(search: Search, out: &Out) -> Result<usize> {
    match *out {
        Out::Lines {
            score,
            preview,
            shown,
            end,
        } => lines(search, score, preview, shown, end),
        Out::Json { matches } => json(search, matches),
    }
}

fn lines(search: Search, score: bool, preview: bool, shown: Shown, end: End) -> Result<usize> {
    let mut out = Printer::stdout(end);
    let preview = (preview && out.tty).then(Preview::new).transpose()?;
    // A sorted search sees all images before the first hit, so progress shows until then.
    let mut hit = next(
        &search.hits,
        || Ok(true),
        || report::progress(&search.progress),
    )?;
    report::clear();
    let mut printed = 0;
    while let Some(h) = hit {
        if let Some(preview) = &preview {
            preview.show(&mut out.out, &h.path)?;
        }
        let fields = match shown {
            Shown::Nothing => &[],
            Shown::Text => h.text.as_slice(),
            Shown::Matches => h.matches.as_slice(),
        };
        if !out.record(h.score.filter(|_| score), &h.path, fields)? {
            return Ok(printed);
        }
        printed += 1;
        hit = next(
            &search.hits,
            || out.flush(),
            || report::watch(&search.progress),
        )?;
    }
    out.flush()?;
    Ok(printed)
}

fn json(search: Search, matches: bool) -> Result<usize> {
    let tick = || report::progress(&search.progress);
    let mut printed = 0;
    let mut hit = next(&search.hits, || Ok(true), tick)?;
    while let Some(h) = hit {
        tick();
        if !report::hit(&h, matches)? {
            return Ok(printed);
        }
        printed += 1;
        hit = next(&search.hits, || Ok(true), tick)?;
    }
    Ok(printed)
}

/// The images that a picker returns.
pub fn paths(paths: &[PathBuf], end: End) -> io::Result<()> {
    if report::json() {
        for path in paths {
            let hit = Hit {
                file: FileInfo::read(path).ok(),
                ..Hit::new(path.clone())
            };
            if !report::hit(&hit, false)? {
                return Ok(());
            }
        }
        return Ok(());
    }
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
            Err(RecvTimeoutError::Timeout) if closed(io::stdout()) => return Ok(None),
            Err(RecvTimeoutError::Timeout) => tick(),
        }
    }
}

/// The kernel flags a pipe once its reader is gone, before a write fails. macOS flags it only
/// for a poll that asks for POLLOUT. Without this check, `scry -t x | head -1` would read every
/// image after head exits.
fn closed(out: impl AsFd) -> bool {
    let mut fds = [PollFd::new(&out, PollFlags::OUT)];
    rustix::event::poll(&mut fds, Some(&Timespec::default()))
        .is_ok_and(|_| fds[0].revents().intersects(PollFlags::ERR | PollFlags::HUP))
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

/// Gives half blocks the nearest xterm colors if the terminal has no 24-bit color, as Terminal.app
/// before macOS 26. A terminal with 24-bit color tells it in COLORTERM.
pub fn fit_colors(buf: &mut Buffer, area: Rect) {
    static TRUECOLOR: LazyLock<bool> = LazyLock::new(|| available_color_count() == u16::MAX);
    if *TRUECOLOR {
        return;
    }
    for pos in area.intersection(buf.area).positions() {
        let cell = &mut buf[pos];
        cell.fg = xterm(cell.fg);
        cell.bg = xterm(cell.bg);
    }
}

/// The nearest color in the 6 x 6 x 6 cube of xterm, whose levels are 0, 95, 135, 175, 215 and 255.
fn xterm(color: Color) -> Color {
    let Color::Rgb(r, g, b) = color else {
        return color;
    };
    let level = |v: u8| match v {
        0..48 => 0,
        48..115 => 1,
        _ => (v - 35) / 40,
    };
    Color::Indexed(16 + 36 * level(r) + 6 * level(g) + level(b))
}

/// Where `tty` looks, for messages.
pub const TERMINAL: &str = if cfg!(target_os = "macos") {
    "a terminal on stdin, stdout or stderr"
} else {
    "a terminal"
};

/// The controlling terminal. A copy of a stdio fd comes first, because macOS cannot poll an open
/// /dev/tty. A copy also needs no access to the device file, which after `su` belongs to another
/// user.
pub fn tty() -> io::Result<OwnedFd> {
    let stdio = [stdio::stdin(), stdio::stdout(), stdio::stderr()];
    if let Some(fd) = stdio.into_iter().find(|&fd| controls(fd)) {
        return fd.try_clone_to_owned();
    }
    if cfg!(target_os = "macos") {
        return Err(io::Error::other(format!("found no {TERMINAL}")));
    }
    let tty = File::options().read(true).write(true).open("/dev/tty")?;
    Ok(tty.into())
}

/// True if `fd` is the controlling terminal, open to read and write. Not tcgetsid: macOS returns
/// a session for a terminal that is not the controlling terminal.
fn controls(fd: BorrowedFd) -> bool {
    termios::tcgetpgrp(fd).is_ok()
        && fcntl_getfl(fd).is_ok_and(|flags| flags & OFlags::RWMODE == OFlags::RDWR)
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
pub fn has_reader(written: io::Result<()>) -> io::Result<bool> {
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
            stdio::dup2_stdin(tty()?)?;
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
        let area = Rect::from((Position::ORIGIN, thumb.size()));
        let mut buf = Buffer::empty(area);
        Image::new(&thumb).render(area, &mut buf);
        if self.picker.protocol_type() == ProtocolType::Halfblocks {
            fit_colors(&mut buf, area);
        }
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
    fn xterm_colors_are_nearest() {
        let cube = |r: u8, g: u8, b: u8| Color::Indexed(16 + 36 * r + 6 * g + b);
        assert_eq!(xterm(Color::Rgb(0, 0, 0)), cube(0, 0, 0));
        assert_eq!(xterm(Color::Rgb(255, 0, 0)), cube(5, 0, 0));
        assert_eq!(xterm(Color::Rgb(47, 48, 114)), cube(0, 1, 1));
        assert_eq!(xterm(Color::Rgb(115, 154, 255)), cube(2, 2, 5));
        assert_eq!(xterm(Color::Yellow), Color::Yellow);
    }

    #[test]
    fn pipe_without_reader_is_closed() {
        let (reader, writer) = io::pipe().unwrap();
        assert!(!closed(&writer));
        drop(reader);
        assert!(closed(&writer));
    }

    #[test]
    fn broken_pipe_ends_the_output() {
        let broken = io::Error::from(io::ErrorKind::BrokenPipe);
        assert!(!has_reader(Err(broken)).unwrap());
        assert!(has_reader(Err(io::ErrorKind::Other.into())).is_err());
    }
}
