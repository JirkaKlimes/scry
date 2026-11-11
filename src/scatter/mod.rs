mod feed;
mod kitty;
mod layout;
mod prefetch;

use std::{
    fs::File,
    hash::{BuildHasher, RandomState},
    io::{self, Write},
    mem,
    ops::ControlFlow::{self, Break, Continue},
    os::fd::OwnedFd,
    panic,
    path::{Path, PathBuf},
    sync::{
        atomic::Ordering::Relaxed,
        mpsc::{Receiver, TryRecvError},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use image::Rgba;
use indicatif::ProgressBar;
use ratatui::{
    DefaultTerminal, Frame, Terminal,
    backend::CrosstermBackend,
    crossterm::{
        event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
        execute,
        terminal::{EnterAlternateScreen, enable_raw_mode},
    },
    layout::{Constraint, Position, Rect, Size},
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
    feed::Feed,
    layout::{Layout, Tile},
    prefetch::{Prefetch, Thumb},
};
use crate::{
    output,
    search::{Hit, Order, Progress, Search},
};

pub enum Exit {
    Picked(Vec<PathBuf>),
    /// No image could be shown.
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
    /// The color of the line drawn into the edge of each image, so images that overlap stay
    /// apart. kitty shows the cell background through transparent pixels, which lets the line
    /// change color without a new image.
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
const INTERVAL: Duration = Duration::from_millis(200);
const MIN_INTERVAL: Duration = Duration::from_millis(50);
const MAX_INTERVAL: Duration = Duration::from_secs(2);
/// Share of the screen that one image covers, before the boost for top scores.
const SIZE: f32 = 1.0 / 6.0;
const MIN_SIZE: f32 = 0.02;
const MAX_SIZE: f32 = 0.5;
/// Top scores get up to this much more width and height.
const BOOST: f32 = 0.3;

pub fn run(search: Search) -> Result<Exit> {
    let _stdio = TtyStdio::swap().context("scatter mode needs a terminal")?;
    restore_on_panic();
    let _restore = Restore;
    let mut terminal = init()?;
    let picker = output::picker().unwrap_or_else(|_| Picker::halfblocks());
    App::new(search, picker)?.run(&mut terminal)
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

enum State {
    Running,
    Paused(Pause),
}

struct Pause {
    selected: Option<u32>,
    /// Tile ids with a hint letter, in `HINTS` order. Fixed until the pile changes, so the
    /// letters do not move while you read them.
    hints: Vec<u32>,
    zoom: Option<Zoom>,
}

struct Zoom {
    path: PathBuf,
    /// The room that the image was shrunk for.
    room: Size,
    image: Loading,
}

enum Loading {
    Pending(Receiver<Option<Thumb>>),
    Ready(Protocol),
    Failed,
}

struct Card {
    hit: Hit,
    image: Protocol,
}

/// What a frame shows, for sixel and iTerm2. Their images are pixels, not cells: text drawn
/// over an image stays after the text goes away, and an image comes back on top only when the
/// terminal gets it again.
#[derive(Default)]
struct Scene {
    /// Ids of the tiles in view, in draw order.
    tiles: Vec<u32>,
    bounds: Rect,
    paused: bool,
    selected: Option<u32>,
    zoom: bool,
}

impl Scene {
    /// True when the terminal can draw this scene over `shown`: they are equal, or this one
    /// only adds a tile on top.
    fn follows(&self, shown: &Scene) -> bool {
        let same = |s: &Scene| (s.bounds, s.paused, s.selected, s.zoom);
        if same(self) != same(shown) {
            return false;
        }
        match self.tiles.split_last() {
            _ if self.tiles == shown.tiles => true,
            Some((top, below)) if !shown.tiles.contains(top) => {
                let mut old = shown.tiles.iter();
                below.iter().all(|id| old.any(|o| o == id))
            }
            _ => false,
        }
    }
}

struct App {
    search: Search,
    /// False once all hits have arrived.
    searching: bool,
    feed: Feed,
    prefetch: Prefetch,
    /// Estimates the time left while a sorted search sees all images.
    estimate: ProgressBar,
    picker: Picker,
    graphics: Graphics,
    shown: Scene,
    layout: Layout<Card>,
    clock: Duration,
    next_spawn: Duration,
    interval: Duration,
    size: f32,
    state: State,
    marks: Vec<PathBuf>,
}

impl App {
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
            shown: Scene::default(),
            picker,
            feed: match search.order {
                Order::Date(_) => Feed::in_order(),
                Order::Found | Order::Score(()) => Feed::shuffled(),
            },
            search,
            searching: true,
            layout: Layout::new(RandomState::new().hash_one(())),
            clock: Duration::ZERO,
            next_spawn: Duration::ZERO,
            interval: INTERVAL,
            size: SIZE,
            state: State::Running,
            marks: Vec::new(),
        })
    }

    fn run(mut self, terminal: &mut DefaultTerminal) -> Result<Exit> {
        let mut last = Instant::now();
        loop {
            let area = terminal.size()?;
            let now = Instant::now();
            let bounds = Rect::new(0, 0, area.width, area.height.saturating_sub(1));
            if let Break(exit) = self.tick(now - last, bounds)? {
                return Ok(exit);
            }
            last = now;
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

    fn tick(&mut self, elapsed: Duration, bounds: Rect) -> Result<Outcome> {
        if !self.paused() {
            self.clock += elapsed;
        }
        self.drain()?;
        for card in self.layout.fit(bounds) {
            self.feed.retry(card.hit);
        }
        if !self.searching && self.feed.is_empty() && self.prefetch.idle() {
            if self.layout.landed() == 0 {
                return Ok(Break(Exit::Empty));
            }
            // Before the next landing, so a new image never gets its first frame under hints.
            if self.layout.ahead() == 0 {
                self.pause();
            }
        }
        self.load_zoom();
        self.request_thumbs();
        if !self.paused() && self.clock >= self.next_spawn && self.advance() {
            self.next_spawn = self.clock + self.interval;
        }
        Ok(Continue(()))
    }

    fn drain(&mut self) -> Result<()> {
        while self.searching {
            match self.search.hits.try_recv() {
                Ok(hit) => self.feed.push(hit?),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => self.searching = false,
            }
        }
        Ok(())
    }

    fn request_thumbs(&mut self) {
        let room = self.room();
        let screen = f32::from(room.width) * f32::from(room.height);
        while screen > 0.0
            && self.prefetch.wants()
            && let Some((hit, emphasis)) = self.feed.pop()
        {
            let area = screen * self.size * (1.0 + BOOST * emphasis).powi(2);
            self.prefetch.request(hit, emphasis, area, room);
        }
    }

    /// Puts the next image on top: one that `back` took off, or else a new one. False when
    /// none is ready.
    fn advance(&mut self) -> bool {
        self.layout.forward() || self.land()
    }

    fn land(&mut self) -> bool {
        let Some(prepared) = self.prefetch.take() else {
            return false;
        };
        let cells = prepared.thumb.cells;
        let Some(rect) = self.layout.place(cells) else {
            // Prepared before the terminal shrank: prepare it again at the current size.
            self.feed.retry(prepared.hit);
            return false;
        };
        let Some(image) = protocol(&self.picker, &self.search.progress, prepared.thumb) else {
            return false;
        };
        let card = Card {
            hit: prepared.hit,
            image,
        };
        self.layout.push(rect, prepared.emphasis, card);
        true
    }

    fn load_zoom(&mut self) {
        let room = self.room();
        let State::Paused(Pause {
            zoom: Some(zoom), ..
        }) = &mut self.state
        else {
            return;
        };
        if zoom.room != room {
            // Shrunk for another terminal size, it would not fit.
            *zoom = Zoom::open(&self.prefetch, zoom.path.clone(), room);
            return;
        }
        let Loading::Pending(rx) = &zoom.image else {
            return;
        };
        zoom.image = match rx.try_recv() {
            Err(TryRecvError::Empty) => return,
            Ok(Some(thumb)) => protocol(&self.picker, &self.search.progress, thumb)
                .map_or(Loading::Failed, Loading::Ready),
            Ok(None) | Err(TryRecvError::Disconnected) => Loading::Failed,
        };
    }

    /// True when only a key or a resize can change the screen.
    fn idle(&self) -> bool {
        !self.searching
            && match &self.state {
                State::Paused(pause) => !matches!(
                    pause.zoom,
                    Some(Zoom {
                        image: Loading::Pending(_),
                        ..
                    })
                ),
                State::Running => false,
            }
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
        if let State::Paused(pause) = &mut self.state
            && let Some(zoom) = pause.zoom.take()
        {
            return self.zoom_key(key, zoom);
        }
        match key.code {
            KeyCode::Esc => return Break(Exit::Aborted),
            KeyCode::Enter => return self.enter(),
            KeyCode::Char(' ') if self.paused() => self.state = State::Running,
            KeyCode::Char(' ') => self.pause(),
            KeyCode::Left => self.back(),
            KeyCode::Right => self.forward(),
            KeyCode::Char('+' | '=') => self.speed(0.8),
            KeyCode::Char('-') => self.speed(1.25),
            KeyCode::Char(']') => self.scale(1.25),
            KeyCode::Char('[') => self.scale(0.8),
            KeyCode::Tab => {
                if let Some(path) = self.selected_path() {
                    self.toggle_mark(path);
                }
            }
            KeyCode::Char(c)
                if c.is_ascii_uppercase()
                    || c.is_ascii_lowercase() && key.modifiers.contains(KeyModifiers::SHIFT) =>
            {
                if let Some(id) = self.hinted(c.to_ascii_lowercase()) {
                    self.select(id);
                    self.zoom();
                }
            }
            KeyCode::Char(c) => {
                if let Some(id) = self.hinted(c) {
                    self.select(id);
                }
            }
            _ => {}
        }
        Continue(())
    }

    fn zoom_key(&mut self, key: KeyEvent, zoom: Zoom) -> Outcome {
        match key.code {
            KeyCode::Enter => {
                if !self.marks.contains(&zoom.path) {
                    self.marks.push(zoom.path);
                }
                return Break(Exit::Picked(mem::take(&mut self.marks)));
            }
            KeyCode::Tab => {
                self.toggle_mark(zoom.path.clone());
                self.set_zoom(zoom);
            }
            // Right stops at the newest image: a new kitty image must show in a tile first.
            KeyCode::Left | KeyCode::Right => {
                let moved = match key.code {
                    KeyCode::Left => self.layout.back(),
                    _ => self.layout.forward(),
                };
                if moved {
                    self.select_top();
                    self.zoom();
                } else {
                    self.set_zoom(zoom);
                }
            }
            _ => {}
        }
        Continue(())
    }

    fn enter(&mut self) -> Outcome {
        if !self.marks.is_empty() {
            return Break(Exit::Picked(mem::take(&mut self.marks)));
        }
        if let Some(tile) = self.selected_tile() {
            return Break(Exit::Picked(vec![tile.item.hit.path.clone()]));
        }
        self.pause();
        let tiles = self.layout.tiles();
        let score = |t: &&Tile<Card>| t.item.hit.score.unwrap_or(0.0);
        // max_by keeps the last maximum, so ties go to the newest image.
        let best = (self.layout.views().iter())
            .map(|v| &tiles[v.index])
            .max_by(|a, b| score(a).total_cmp(&score(b)))
            .map(|t| t.id);
        if let Some(id) = best {
            self.select(id);
        }
        Continue(())
    }

    /// Pauses and shows the screen as it was before the newest image landed.
    fn back(&mut self) {
        self.pause();
        if self.layout.back() {
            self.select_top();
        }
    }

    /// Pauses and shows the next image.
    fn forward(&mut self) {
        self.pause();
        if self.advance() {
            self.select_top();
        }
    }

    fn speed(&mut self, factor: f32) {
        self.interval = self
            .interval
            .mul_f32(factor)
            .clamp(MIN_INTERVAL, MAX_INTERVAL);
    }

    fn scale(&mut self, factor: f32) {
        self.size = (self.size * factor).clamp(MIN_SIZE, MAX_SIZE);
    }

    /// Shows the selected image full screen.
    fn zoom(&mut self) {
        if let Some(path) = self.selected_path() {
            let zoom = Zoom::open(&self.prefetch, path, self.room());
            self.set_zoom(zoom);
        }
    }

    fn set_zoom(&mut self, zoom: Zoom) {
        if let State::Paused(pause) = &mut self.state {
            pause.zoom = Some(zoom);
        }
    }

    fn toggle_mark(&mut self, path: PathBuf) {
        match self.marks.iter().position(|m| *m == path) {
            Some(i) => _ = self.marks.remove(i),
            None => self.marks.push(path),
        }
    }

    fn paused(&self) -> bool {
        matches!(self.state, State::Paused(_))
    }

    fn pause(&mut self) {
        if let State::Running = self.state {
            self.state = State::Paused(Pause {
                selected: None,
                hints: Vec::new(),
                zoom: None,
            });
            self.rehint();
        }
    }

    /// Gives a hint letter to each image in view, the newest first.
    fn rehint(&mut self) {
        let tiles = self.layout.tiles();
        let hints = (self.layout.views().iter().rev())
            .take(HINTS.len())
            .map(|v| tiles[v.index].id)
            .collect();
        if let State::Paused(pause) = &mut self.state {
            pause.hints = hints;
        }
    }

    fn select_top(&mut self) {
        self.rehint();
        if let Some(id) = self.layout.top().map(|t| t.id) {
            self.select(id);
        }
    }

    fn select(&mut self, id: u32) {
        if let State::Paused(pause) = &mut self.state {
            pause.selected = Some(id);
        }
    }

    fn selected(&self) -> Option<u32> {
        match &self.state {
            State::Paused(pause) => pause.selected,
            State::Running => None,
        }
    }

    fn hinted(&self, hint: char) -> Option<u32> {
        let State::Paused(pause) = &self.state else {
            return None;
        };
        HINTS.find(hint).and_then(|i| pause.hints.get(i)).copied()
    }

    fn selected_tile(&self) -> Option<&Tile<Card>> {
        let id = self.selected()?;
        self.layout.tiles().iter().find(|t| t.id == id)
    }

    fn selected_path(&self) -> Option<PathBuf> {
        self.selected_tile().map(|t| t.item.hit.path.clone())
    }

    /// The area for one image full screen.
    fn room(&self) -> Size {
        self.layout.bounds().as_size()
    }

    /// Indices of the tiles to draw, bottom to top, with the selected one last.
    fn draw_order(&self) -> Vec<usize> {
        let tiles = self.layout.tiles();
        let selected = (self.selected()).and_then(|id| tiles.iter().position(|t| t.id == id));
        let mut order: Vec<usize> = (self.layout.visible().into_iter())
            .filter(|&i| Some(i) != selected)
            .collect();
        order.extend(selected);
        order
    }

    fn scene(&self, order: &[usize]) -> Scene {
        Scene {
            tiles: order.iter().map(|&i| self.layout.tiles()[i].id).collect(),
            bounds: self.layout.bounds(),
            paused: self.paused(),
            selected: self.selected(),
            zoom: matches!(&self.state, State::Paused(Pause { zoom: Some(_), .. })),
        }
    }

    fn draw(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        let order = self.draw_order();
        if self.graphics == Graphics::Layered {
            let scene = self.scene(&order);
            if !scene.follows(&self.shown) {
                terminal.clear()?;
                // One frame per tile, so each image covers the ones below it.
                if !scene.zoom {
                    for n in 1..order.len() {
                        terminal.draw(|frame| self.draw_tiles(frame, &order[..n]))?;
                    }
                }
            }
            self.shown = scene;
        }
        terminal.draw(|frame| self.draw_frame(frame, &order))?;
        Ok(())
    }

    fn draw_frame(&self, frame: &mut Frame, order: &[usize]) {
        if let Some(row) = frame.area().rows().next_back() {
            frame.render_widget(self.status(), row);
        }
        let pause = match &self.state {
            State::Paused(pause) => Some(pause),
            State::Running => None,
        };
        if let Some(zoom) = pause.and_then(|p| p.zoom.as_ref()) {
            self.draw_zoom(frame, zoom);
            return;
        }
        self.draw_tiles(frame, order);
        self.draw_marks(frame);
        if let Some(pause) = pause {
            self.draw_hints(frame, pause);
        }
        if self.searching && self.layout.tiles().is_empty() {
            let waiting = match self.search.order {
                Order::Score(()) => Some("scoring all images to show the best first"),
                Order::Date(_) => Some("sorting all images by date"),
                Order::Found => None,
            };
            if let Some(waiting) = waiting {
                let left = output::time_left(&self.estimate, &self.search.progress);
                self.note(frame, &format!("{waiting}{left}"));
            }
        }
    }

    fn draw_tiles(&self, frame: &mut Frame, order: &[usize]) {
        let selected = self.selected();
        for &i in order {
            let tile = &self.layout.tiles()[i];
            frame.render_widget(Image::new(&tile.item.image), tile.rect);
            if self.graphics == Graphics::Cells {
                kitty::anchor(frame.buffer_mut(), tile.rect);
            }
            let color = match (selected == Some(tile.id), self.marked(&tile.item.hit.path)) {
                (true, _) => Color::Yellow,
                (_, true) => Color::Green,
                _ => Color::DarkGray,
            };
            self.tint(frame, tile.rect, color);
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

    /// A check mark at the top right of each marked image in view.
    fn draw_marks(&self, frame: &mut Frame) {
        for view in self.layout.views() {
            let tile = &self.layout.tiles()[view.index];
            if !self.marked(&tile.item.hit.path) {
                continue;
            }
            let at = Position::new(tile.rect.right().saturating_sub(3), tile.rect.y);
            let area = self.layout.own_cells(view.index, at, 3);
            if let Some(label) = tag("✓", area.width) {
                frame.render_widget(Line::from(label).black().on_green().bold(), area);
            }
        }
    }

    /// Hint letters go on top of everything, at the top left of the part of each image in view.
    /// A letter covers only cells of its own image: the first cell of a new kitty image carries
    /// its pixels.
    fn draw_hints(&self, frame: &mut Frame, pause: &Pause) {
        for view in self.layout.views() {
            let tile = &self.layout.tiles()[view.index];
            let Some(i) = pause.hints.iter().position(|&id| id == tile.id) else {
                continue;
            };
            let area = if pause.selected == Some(tile.id) {
                let at = tile.rect.as_position();
                Rect::new(at.x, at.y, 3, 1).intersection(tile.rect)
            } else {
                self.layout.own_cells(view.index, view.anchor, 3)
            };
            if let Some(label) = tag(&HINTS[i..=i], area.width) {
                frame.render_widget(Line::from(label).black().on_yellow().bold(), area);
            }
        }
    }

    fn draw_zoom(&self, frame: &mut Frame, zoom: &Zoom) {
        let bounds = self.layout.bounds();
        let message = match &zoom.image {
            Loading::Ready(image) => {
                let size = image.size();
                let rect = bounds.centered(
                    Constraint::Length(size.width),
                    Constraint::Length(size.height),
                );
                frame.render_widget(Image::new(image), rect);
                self.tint(frame, rect, Color::Yellow);
                return;
            }
            Loading::Pending(_) => "loading",
            Loading::Failed => "cannot open this image",
        };
        self.note(frame, message);
    }

    fn note(&self, frame: &mut Frame, text: &str) {
        let row = self
            .layout
            .bounds()
            .centered_vertically(Constraint::Length(1));
        frame.render_widget(Line::from(text).centered().dim(), row);
    }

    fn marked(&self, path: &Path) -> bool {
        self.marks.iter().any(|m| m == path)
    }

    fn status(&self) -> Line<'static> {
        let rate = 1.0 / self.interval.as_secs_f32();
        let size = self.size * 100.0;
        let mut spans = vec![Span::raw(format!(
            " {}  {rate:.1}/s  size {size:.0}%  ",
            self.search.status()
        ))];
        if self.layout.ahead() > 0 {
            spans.push(format!("{} back  ", self.layout.ahead()).yellow());
        }
        if !self.marks.is_empty() {
            spans.push(format!("{} marked  ", self.marks.len()).green());
        }
        let shown = match &self.state {
            State::Paused(Pause {
                zoom: Some(zoom), ..
            }) => Some(zoom.path.clone()),
            State::Paused(_) => self.selected_path(),
            State::Running => None,
        };
        if let Some(name) = shown.as_deref().and_then(Path::file_name) {
            let name = output::clean_path(Path::new(name));
            spans.push(format!("{name}  ").yellow().bold());
        }
        spans.push(match &self.state {
            State::Paused(Pause { zoom: Some(_), .. }) => {
                "ZOOM  ←/→ step · tab mark · enter pick · any key back".yellow()
            }
            State::Paused(_) => {
                "PAUSED  ←/→ step · a-z select · A-Z zoom · tab mark · enter pick".yellow()
            }
            State::Running => {
                "space pause · ← back · enter pick · +/- speed · [/] size · esc quit".dim()
            }
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

impl Zoom {
    fn open(prefetch: &Prefetch, path: PathBuf, room: Size) -> Self {
        let area = f32::from(room.width) * f32::from(room.height);
        Self {
            image: Loading::Pending(prefetch.thumb(path.clone(), area, room)),
            path,
            room,
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
