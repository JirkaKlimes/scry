mod grid;
mod kitty;
mod prefetch;
mod scatter;

use std::{
    cell::RefCell,
    fs::File,
    io::{self, Write},
    mem,
    ops::ControlFlow::{self, Break, Continue},
    os::fd::OwnedFd,
    panic,
    path::{Path, PathBuf},
    sync::{atomic::Ordering::Relaxed, mpsc::TryRecvError},
    task::Poll,
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use image::Rgba;
use indicatif::ProgressBar;
use ratatui::{
    DefaultTerminal, Frame, Terminal,
    backend::CrosstermBackend,
    buffer::CellDiffOption,
    crossterm::{
        event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
        execute,
        terminal::{EnterAlternateScreen, enable_raw_mode},
    },
    layout::{Constraint, Rect, Size},
    style::{Color, Style, Stylize},
    symbols::border,
    text::{Line, Span},
    widgets::Block,
};
use ratatui_image::{
    Image, Resize,
    picker::{Picker, ProtocolType},
    protocol::Protocol,
};
use rustix::{
    io::dup,
    stdio::{dup2_stdin, dup2_stdout},
};

use self::{
    grid::Grid,
    prefetch::{Prefetch, Shape, Thumb, Ticket},
    scatter::Scatter,
};
use crate::{
    output,
    search::{Hit, Order, Progress, Search},
};

/// How a picker shows the hits.
#[derive(Clone, Copy)]
pub enum View {
    /// Images land one at a time on a pile.
    Scatter,
    /// Pages of thumbnails in the search's order.
    Grid,
}

pub enum Exit {
    Picked(Vec<PathBuf>),
    /// The search found nothing to show.
    Empty,
    Aborted,
}

type Outcome = ControlFlow<Exit>;

/// How the terminal keeps the images it shows.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Graphics {
    /// kitty: each image lives in cells of placeholder text.
    Cells,
    /// Sixel and iTerm2: an image stays only where the terminal last drew it.
    Layered,
    /// Half blocks: each image is colored text.
    Text,
}

impl Graphics {
    /// The color of the line drawn into the edge of each image, so images that touch or overlap
    /// stay apart. kitty shows the cell background through transparent pixels, which lets the
    /// line change color without a new image.
    fn edge(self) -> Rgba<u8> {
        let alpha = if self == Self::Cells { 0 } else { u8::MAX };
        Rgba([128, 128, 128, alpha])
    }
}

/// One half block on each side of an image drawn in half blocks.
const HALF_BLOCK_EDGE: border::Set = border::Set {
    top_left: "█",
    top_right: "█",
    bottom_left: "█",
    bottom_right: "█",
    vertical_left: "█",
    vertical_right: "█",
    horizontal_top: "▀",
    horizontal_bottom: "▄",
};

const HINTS: &str = "asdfghjklqwertyuiopzxcvbnm";
const FRAME: Duration = Duration::from_millis(33);

pub fn run(search: Search, view: View) -> Result<Exit> {
    let name = match view {
        View::Scatter => "scatter",
        View::Grid => "grid",
    };
    let _stdio = TtyStdio::swap().with_context(|| format!("{name} mode needs a terminal"))?;
    restore_on_panic();
    let _restore = Restore;
    let mut terminal = init()?;
    let picker = output::picker().unwrap_or_else(|_| Picker::halfblocks());
    let core = Core::new(search, picker)?;
    match view {
        View::Scatter => App::new(Scatter::new(&core.search.order), core).run(&mut terminal),
        View::Grid => App::new(Grid::default(), core).run(&mut terminal),
    }
}

/// Like `ratatui::init`, but without its panic hook, which restores the terminal on a panic in
/// any thread.
fn init() -> io::Result<DefaultTerminal> {
    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen)?;
    Terminal::new(CrosstermBackend::new(io::stdout()))
}

/// Restores the terminal on a panic in the UI thread. A worker thread catches its own panics,
/// and their messages would only garble the screen.
fn restore_on_panic() {
    let ui = thread::current().id();
    let hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        if thread::current().id() == ui {
            ratatui::restore();
            hook(info);
        }
    }));
}

/// Points fd 0 and 1 at /dev/tty for the UI, like fzf, so the real stdout stays clean for the
/// selection. Drop puts the original fds back.
struct TtyStdio {
    stdin: OwnedFd,
    stdout: OwnedFd,
}

impl TtyStdio {
    fn swap() -> io::Result<Self> {
        let tty = File::options().read(true).write(true).open("/dev/tty")?;
        let saved = Self {
            stdin: dup(io::stdin())?,
            stdout: dup(io::stdout())?,
        };
        dup2_stdin(&tty)?;
        dup2_stdout(&tty)?;
        Ok(saved)
    }
}

impl Drop for TtyStdio {
    fn drop(&mut self) {
        let _ = io::stdout().flush();
        let _ = dup2_stdin(&self.stdin);
        let _ = dup2_stdout(&self.stdout);
    }
}

struct Restore;

impl Drop for Restore {
    fn drop(&mut self) {
        ratatui::restore();
    }
}

#[derive(Clone, Copy)]
enum Step {
    Back,
    Forward,
}

struct Zoom {
    path: PathBuf,
    /// The room that the image was shrunk for.
    room: Size,
    image: Loading<Protocol>,
}

impl Zoom {
    fn open(prefetch: &Prefetch, path: PathBuf, room: Size) -> Self {
        let area = f32::from(room.width) * f32::from(room.height);
        Self {
            image: Loading::Pending(prefetch.thumb(path.clone(), Shape::Fit { area, room })),
            path,
            room,
        }
    }
}

enum Loading<T> {
    Pending(Ticket),
    Ready(T),
    Failed,
}

impl<T> Loading<T> {
    /// Takes the thumbnail once a worker has it, and turns it into `T` with `make`.
    fn poll(&mut self, make: impl FnOnce(Thumb) -> Option<T>) {
        let Self::Pending(ticket) = self else {
            return;
        };
        let Poll::Ready(thumb) = ticket.poll() else {
            return;
        };
        *self = thumb.and_then(make).map_or(Self::Failed, Self::Ready);
    }

    fn pending(&self) -> bool {
        matches!(self, Self::Pending(_))
    }
}

/// What scatter and grid mode share: the search, the terminal, the marks and the zoom.
struct Core {
    search: Search,
    /// False once all hits have arrived.
    searching: bool,
    prefetch: Prefetch,
    /// Estimates the time left while a sorted search sees all images.
    estimate: ProgressBar,
    picker: Picker,
    graphics: Graphics,
    /// The area for images, above the status line.
    bounds: Rect,
    marks: Vec<PathBuf>,
    zoom: Option<Zoom>,
    /// The kitty images that the frame in progress sends.
    transmits: RefCell<Vec<kitty::Transmit>>,
}

impl Core {
    fn new(search: Search, picker: Picker) -> Result<Self> {
        let graphics = match picker.protocol_type() {
            ProtocolType::Kitty => Graphics::Cells,
            ProtocolType::Sixel | ProtocolType::Iterm2 => Graphics::Layered,
            ProtocolType::Halfblocks => Graphics::Text,
        };
        Ok(Self {
            prefetch: Prefetch::new(picker.font_size(), graphics.edge(), &search.progress)?,
            estimate: ProgressBar::hidden(),
            graphics,
            picker,
            search,
            searching: true,
            bounds: Rect::ZERO,
            marks: Vec::new(),
            zoom: None,
            transmits: RefCell::default(),
        })
    }

    /// Gives each new hit to `push`.
    fn drain(&mut self, mut push: impl FnMut(Hit)) -> Result<()> {
        while self.searching {
            match self.search.hits.try_recv() {
                Ok(hit) => push(hit?),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => self.searching = false,
            }
        }
        Ok(())
    }

    /// The area for one image full screen.
    fn room(&self) -> Size {
        self.bounds.as_size()
    }

    fn load_zoom(&mut self) {
        let room = self.room();
        let Some(zoom) = &mut self.zoom else {
            return;
        };
        if zoom.room != room {
            // Shrunk for another terminal size, it would not fit.
            *zoom = Zoom::open(&self.prefetch, zoom.path.clone(), room);
            return;
        }
        zoom.image
            .poll(|thumb| protocol(&self.picker, &self.search.progress, thumb));
    }

    fn toggle_mark(&mut self, path: PathBuf) {
        match self.marks.iter().position(|m| *m == path) {
            Some(i) => _ = self.marks.remove(i),
            None => self.marks.push(path),
        }
    }

    fn marked(&self, path: &Path) -> bool {
        self.marks.iter().any(|m| m == path)
    }

    /// Draws `image` in `area`. All images go through here, so that a label drawn later in the
    /// frame cannot lose a kitty image.
    fn draw_image(&self, frame: &mut Frame, image: &Protocol, area: Rect) {
        frame.render_widget(Image::new(image), area);
        if self.graphics == Graphics::Cells {
            kitty::anchor(frame.buffer_mut(), area);
            let transmit = kitty::transmit(frame.buffer_mut(), area);
            self.transmits.borrow_mut().extend(transmit);
        }
    }

    /// The color for the edge of an image.
    fn edge(&self, path: &Path, selected: bool) -> Color {
        match (selected, self.marked(path)) {
            (true, _) => Color::Yellow,
            (_, true) => Color::Green,
            _ => Color::DarkGray,
        }
    }

    /// Colors the edge of an image by its state. kitty shows the cell background through the
    /// transparent edge in each image. Half blocks are text, so the outer half blocks take the
    /// color. A sixel or iTerm2 image cannot change without a new image.
    fn tint(&self, frame: &mut Frame, area: Rect, color: Color) {
        match self.graphics {
            Graphics::Cells => frame.buffer_mut().set_style(area, Style::new().bg(color)),
            Graphics::Text => {
                let edge = Block::bordered()
                    .border_set(HALF_BLOCK_EDGE)
                    .border_style(color);
                frame.render_widget(edge, area);
            }
            Graphics::Layered => {}
        }
    }

    fn draw_zoom(&self, frame: &mut Frame, zoom: &Zoom) {
        let message = match &zoom.image {
            Loading::Ready(image) => {
                let size = image.size();
                let rect = self.bounds.centered(
                    Constraint::Length(size.width),
                    Constraint::Length(size.height),
                );
                self.draw_image(frame, image, rect);
                self.tint(frame, rect, Color::Yellow);
                return;
            }
            Loading::Pending(_) => "loading",
            Loading::Failed => "cannot open this image",
        };
        self.note(frame, message);
    }

    /// A sorted search sends no hit before it has seen all images, so it tells why the screen
    /// is empty.
    fn draw_waiting(&self, frame: &mut Frame) {
        let waiting = match self.search.order {
            Order::Score(()) => "scoring all images to show the best first",
            Order::Date(_) => "sorting all images by date",
            Order::Found => return,
        };
        let left = output::time_left(&self.estimate, &self.search.progress);
        self.note(frame, &format!("{waiting}{left}"));
    }

    fn note(&self, frame: &mut Frame, text: &str) {
        let row = self.bounds.centered_vertically(Constraint::Length(1));
        frame.render_widget(Line::from(text).centered().dim(), row);
    }
}

/// What scatter and grid mode do in their own way: where the images go and which one is next.
/// `App` handles the keys that they share.
trait Board {
    /// Takes the new hits and thumbnails, and moves on by `elapsed`.
    fn tick(&mut self, core: &mut Core, elapsed: Duration) -> Result<Outcome>;

    /// True when only a key or a resize can change the board.
    fn idle(&self) -> bool;

    /// A key that `App` does not handle.
    fn key(&mut self, core: &mut Core, code: KeyCode);

    fn selected(&self) -> Option<&Path>;

    /// Selects the image with the hint letter at `hint` in `HINTS`. False if no image has it.
    fn select(&mut self, hint: usize) -> bool;

    /// Selects the image before or after the selected one, for the zoom. False at the end.
    fn step(&mut self, step: Step) -> bool;

    /// True until the board has something to show: a hit in grid mode, an image in scatter mode.
    fn is_empty(&self) -> bool;

    /// Runs right before each frame. A board can make the images of the frame here, and clear or
    /// redraw the sixel and iTerm2 images that must go.
    fn prepare(&mut self, terminal: &mut DefaultTerminal, core: &Core) -> io::Result<()>;

    fn draw(&self, frame: &mut Frame, core: &Core);

    /// Facts for the status line, before the marks.
    fn status(&self) -> Vec<Span<'static>>;

    /// The keys for the status line.
    fn keys(&self) -> Span<'static>;
}

struct App<B> {
    core: Core,
    board: B,
}

impl<B: Board> App<B> {
    fn new(board: B, core: Core) -> Self {
        Self { core, board }
    }

    fn run(mut self, terminal: &mut DefaultTerminal) -> Result<Exit> {
        let mut last = Instant::now();
        loop {
            let area = terminal.size()?;
            let now = Instant::now();
            self.core.bounds = Rect::new(0, 0, area.width, area.height.saturating_sub(1));
            if let Break(exit) = self.board.tick(&mut self.core, now - last)? {
                return Ok(exit);
            }
            last = now;
            self.core.load_zoom();
            self.draw(terminal)?;
            // Only a key or a resize can change an idle screen, so wait for one.
            if (self.idle() || event::poll(FRAME)?)
                && let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
                && let Break(exit) = self.key(key)
            {
                return Ok(exit);
            }
        }
    }

    fn idle(&self) -> bool {
        !self.core.searching
            && !(self.core.zoom.as_ref()).is_some_and(|zoom| zoom.image.pending())
            && self.board.idle()
    }

    fn key(&mut self, key: KeyEvent) -> Outcome {
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return Break(Exit::Aborted);
        }
        if key
            .modifiers
            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
        {
            return Continue(());
        }
        if let Some(zoom) = self.core.zoom.take() {
            return self.zoom_key(key.code, zoom);
        }
        match key.code {
            KeyCode::Esc => return Break(Exit::Aborted),
            KeyCode::Enter => match self.picked() {
                Some(paths) => return Break(Exit::Picked(paths)),
                None => self.board.key(&mut self.core, KeyCode::Enter),
            },
            KeyCode::Tab => {
                if let Some(path) = self.board.selected() {
                    self.core.toggle_mark(path.to_owned());
                }
            }
            KeyCode::Char(c) if c.is_ascii_alphabetic() => {
                let zoom = c.is_ascii_uppercase() || key.modifiers.contains(KeyModifiers::SHIFT);
                if let Some(hint) = HINTS.find(c.to_ascii_lowercase())
                    && self.board.select(hint)
                    && zoom
                {
                    self.zoom();
                }
            }
            code => self.board.key(&mut self.core, code),
        }
        Continue(())
    }

    fn zoom_key(&mut self, code: KeyCode, zoom: Zoom) -> Outcome {
        match code {
            KeyCode::Enter => {
                if !self.core.marked(&zoom.path) {
                    self.core.marks.push(zoom.path);
                }
                return Break(Exit::Picked(mem::take(&mut self.core.marks)));
            }
            KeyCode::Tab => {
                self.core.toggle_mark(zoom.path.clone());
                self.core.zoom = Some(zoom);
            }
            KeyCode::Left | KeyCode::Right => {
                let step = match code {
                    KeyCode::Left => Step::Back,
                    _ => Step::Forward,
                };
                if self.board.step(step) {
                    self.zoom();
                } else {
                    self.core.zoom = Some(zoom);
                }
            }
            _ => {}
        }
        Continue(())
    }

    /// The marked images in mark order, else the selected image.
    fn picked(&mut self) -> Option<Vec<PathBuf>> {
        if !self.core.marks.is_empty() {
            return Some(mem::take(&mut self.core.marks));
        }
        self.board.selected().map(|path| vec![path.to_owned()])
    }

    /// Shows the selected image full screen.
    fn zoom(&mut self) {
        if let Some(path) = self.board.selected() {
            let zoom = Zoom::open(&self.core.prefetch, path.to_owned(), self.core.room());
            self.core.zoom = Some(zoom);
        }
    }

    fn draw(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        self.board.prepare(terminal, &self.core)?;
        terminal.draw(|frame| {
            self.draw_frame(frame);
            kitty::keep(frame.buffer_mut(), self.core.transmits.take());
        })?;
        Ok(())
    }

    fn draw_frame(&self, frame: &mut Frame) {
        if let Some(row) = frame.area().rows().next_back() {
            frame.render_widget(self.status(), row);
        }
        if let Some(zoom) = &self.core.zoom {
            self.core.draw_zoom(frame, zoom);
            return;
        }
        self.board.draw(frame, &self.core);
        if self.core.searching && self.board.is_empty() {
            self.core.draw_waiting(frame);
        }
    }

    fn status(&self) -> Line<'static> {
        let mut spans = vec![Span::raw(format!(" {}  ", self.core.search.status()))];
        spans.extend(self.board.status());
        if !self.core.marks.is_empty() {
            spans.push(format!("{} marked  ", self.core.marks.len()).green());
        }
        let shown = match &self.core.zoom {
            Some(zoom) => Some(zoom.path.as_path()),
            None => self.board.selected(),
        };
        if let Some(name) = shown.and_then(Path::file_name) {
            let name = output::clean_path(Path::new(name));
            spans.push(format!("{name}  ").yellow().bold());
        }
        spans.push(match self.core.zoom {
            Some(_) => "ZOOM  ←/→ step · tab mark · enter pick · any key back".yellow(),
            None => self.board.keys(),
        });
        Line::from(spans)
    }
}

/// A label that fits `width` cells, or `None` if no cell is free.
fn tag(text: &str, width: u16) -> Option<String> {
    match width {
        0 => None,
        1 | 2 => Some(text.to_owned()),
        _ => Some(format!(" {text} ")),
    }
}

/// The hint letter at `hint` in `HINTS`. The selected image gets other colors, because a sixel
/// or iTerm2 image cannot show the selection on its edge.
fn draw_hint(frame: &mut Frame, area: Rect, hint: usize, selected: bool) {
    let style = if selected {
        Style::new().black().on_white()
    } else {
        Style::new().black().on_yellow()
    };
    draw_label(frame, area, &HINTS[hint..=hint], style.bold());
}

/// The check mark of a marked image.
fn draw_check(frame: &mut Frame, area: Rect) {
    draw_label(frame, area, "✓", Style::new().black().on_green().bold());
}

/// `text` in as much of `area` as it needs. ratatui-image tells ratatui to skip the cells of a
/// sixel or iTerm2 image, and a label keeps that option. So the label cells get the default
/// option back, or the label never reaches the screen.
fn draw_label(frame: &mut Frame, area: Rect, text: &str, style: Style) {
    let Some(label) = tag(text, area.width) else {
        return;
    };
    let line = Line::from(label).style(style);
    let width = line.width() as u16;
    frame.render_widget(line, area);
    for x in area.x..area.x.saturating_add(width) {
        if let Some(cell) = frame.buffer_mut().cell_mut((x, area.y)) {
            cell.set_diff_option(CellDiffOption::None);
        }
    }
}

/// Call only right before a draw that shows the image. A kitty image waits in shared memory
/// until the terminal reads it, and only the terminal frees it.
fn protocol(picker: &Picker, progress: &Progress, thumb: Thumb) -> Option<Protocol> {
    picker
        .new_protocol(thumb.image, thumb.cells, Resize::Fit(None))
        .inspect_err(|_| {
            progress.errors.fetch_add(1, Relaxed);
        })
        .ok()
}

#[cfg(test)]
mod tests {
    use ratatui::backend::TestBackend;

    use super::*;

    #[test]
    fn labels_show_over_skipped_image_cells() {
        let mut terminal = Terminal::new(TestBackend::new(6, 1)).unwrap();
        terminal
            .draw(|frame| {
                for x in 0..6 {
                    frame.buffer_mut()[(x, 0)]
                        .set_symbol("x")
                        .set_diff_option(CellDiffOption::Skip);
                }
                draw_check(frame, frame.area());
            })
            .unwrap();
        let shown = terminal.backend().buffer();
        assert_eq!(shown[(1, 0)].symbol(), "✓");
        assert_eq!(shown[(4, 0)].symbol(), " ", "the image cells stay skipped");
    }
}
