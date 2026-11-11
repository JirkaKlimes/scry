mod feed;
mod layout;
mod queue;

use std::{
    hash::{BuildHasher, RandomState},
    io,
    ops::ControlFlow::{Break, Continue},
    path::Path,
    time::Duration,
};

use anyhow::Result;
use ratatui::{
    DefaultTerminal, Frame,
    crossterm::event::KeyCode,
    layout::{Position, Rect},
    style::Stylize,
    text::Span,
};
use ratatui_image::protocol::Protocol;

use self::{
    feed::Feed,
    layout::{Layout, Tile},
    queue::Queue,
};
use super::{
    Board, Core, Exit, Graphics, HINTS, Outcome, Step, draw_check, draw_hint, prefetch::Shape,
    protocol,
};
use crate::search::{Hit, Order};

const INTERVAL: Duration = Duration::from_millis(200);
const MIN_INTERVAL: Duration = Duration::from_millis(50);
const MAX_INTERVAL: Duration = Duration::from_secs(2);
/// Share of the screen that one image covers, before the boost for top scores.
const SIZE: f32 = 1.0 / 6.0;
const MIN_SIZE: f32 = 0.02;
const MAX_SIZE: f32 = 0.5;
/// Top scores get up to this much more width and height.
const BOOST: f32 = 0.3;

enum State {
    Running,
    Paused(Pause),
}

struct Pause {
    selected: Option<u32>,
    /// Tile ids with a hint letter, in `HINTS` order. Fixed until the pile changes, so the
    /// letters do not move while you read them.
    hints: Vec<u32>,
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
    /// Ids of the marked tiles in view. A check mark that goes away needs a new frame.
    marked: Vec<u32>,
}

impl Scene {
    /// True when the terminal can draw this scene over `shown`: they are equal, or this one
    /// only adds a tile on top.
    fn follows(&self, shown: &Scene) -> bool {
        let same = |s: &Scene| (s.bounds, s.paused, s.selected, s.zoom);
        if same(self) != same(shown) || self.marked != shown.marked {
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

/// Images land one at a time on a pile, at a steady pace.
pub struct Scatter {
    feed: Feed,
    queue: Queue,
    shown: Scene,
    layout: Layout<Card>,
    clock: Duration,
    next_spawn: Duration,
    interval: Duration,
    size: f32,
    state: State,
}

impl Scatter {
    pub fn new(order: &Order<()>) -> Self {
        Self {
            feed: match order {
                Order::Date(_) => Feed::in_order(),
                Order::Found | Order::Score(()) => Feed::shuffled(),
            },
            queue: Queue::default(),
            shown: Scene::default(),
            layout: Layout::new(RandomState::new().hash_one(())),
            clock: Duration::ZERO,
            next_spawn: Duration::ZERO,
            interval: INTERVAL,
            size: SIZE,
            state: State::Running,
        }
    }

    fn request_thumbs(&mut self, core: &mut Core) {
        let room = core.room();
        let screen = f32::from(room.width) * f32::from(room.height);
        while screen > 0.0
            && self.queue.wants()
            && let Some((hit, emphasis)) = self.feed.pop()
        {
            let area = screen * self.size * (1.0 + BOOST * emphasis).powi(2);
            let shape = Shape::Fit { area, room };
            self.queue.request(&core.prefetch, hit, emphasis, shape);
        }
    }

    /// Puts the next image on top: one that `back` took off, or else a new one. False when
    /// none is ready.
    fn advance(&mut self, core: &mut Core) -> bool {
        self.layout.forward() || self.land(core)
    }

    fn land(&mut self, core: &mut Core) -> bool {
        let Some(prepared) = self.queue.take() else {
            return false;
        };
        let cells = prepared.thumb.cells;
        let Some(rect) = self.layout.place(cells) else {
            // Prepared before the terminal shrank: prepare it again at the current size.
            self.feed.retry(prepared.hit);
            return false;
        };
        let Some(image) = protocol(&core.picker, &core.search.progress, prepared.thumb) else {
            return false;
        };
        let card = Card {
            hit: prepared.hit,
            image,
        };
        self.layout.push(rect, prepared.emphasis, card);
        true
    }

    /// Pauses and selects the best image in view.
    fn select_best(&mut self) {
        self.pause();
        let tiles = self.layout.tiles();
        let score = |t: &&Tile<Card>| t.item.hit.score.unwrap_or(0.0);
        // max_by keeps the last maximum, so ties go to the newest image.
        let best = (self.layout.views().iter())
            .map(|v| &tiles[v.index])
            .max_by(|a, b| score(a).total_cmp(&score(b)))
            .map(|t| t.id);
        if let Some(id) = best {
            self.select_id(id);
        }
    }

    /// Pauses and shows the screen as it was before the newest image landed.
    fn back(&mut self) {
        self.pause();
        if self.layout.back() {
            self.select_top();
        }
    }

    /// Pauses and shows the next image.
    fn forward(&mut self, core: &mut Core) {
        self.pause();
        if self.advance(core) {
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

    fn paused(&self) -> bool {
        matches!(self.state, State::Paused(_))
    }

    fn pause(&mut self) {
        if let State::Running = self.state {
            self.state = State::Paused(Pause {
                selected: None,
                hints: Vec::new(),
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
            self.select_id(id);
        }
    }

    fn select_id(&mut self, id: u32) {
        if let State::Paused(pause) = &mut self.state {
            pause.selected = Some(id);
        }
    }

    fn selected_id(&self) -> Option<u32> {
        match &self.state {
            State::Paused(pause) => pause.selected,
            State::Running => None,
        }
    }

    fn selected_tile(&self) -> Option<&Tile<Card>> {
        let id = self.selected_id()?;
        self.layout.tiles().iter().find(|t| t.id == id)
    }

    /// Indices of the tiles to draw, bottom to top, with the selected one last.
    fn draw_order(&self) -> Vec<usize> {
        let tiles = self.layout.tiles();
        let selected = (self.selected_id()).and_then(|id| tiles.iter().position(|t| t.id == id));
        let mut order: Vec<usize> = (self.layout.visible().into_iter())
            .filter(|&i| Some(i) != selected)
            .collect();
        order.extend(selected);
        order
    }

    fn scene(&self, order: &[usize], core: &Core) -> Scene {
        let tiles = self.layout.tiles();
        Scene {
            tiles: order.iter().map(|&i| tiles[i].id).collect(),
            bounds: self.layout.bounds(),
            paused: self.paused(),
            selected: self.selected_id(),
            zoom: core.zoom.is_some(),
            marked: (self.layout.views().iter())
                .map(|v| &tiles[v.index])
                .filter(|t| core.marked(&t.item.hit.path))
                .map(|t| t.id)
                .collect(),
        }
    }

    fn draw_tiles(&self, frame: &mut Frame, core: &Core, order: &[usize]) {
        let selected = self.selected_id();
        for &i in order {
            let tile = &self.layout.tiles()[i];
            core.draw_image(frame, &tile.item.image, tile.rect);
            let color = core.edge(&tile.item.hit.path, selected == Some(tile.id));
            core.tint(frame, tile.rect, color);
        }
    }

    /// A check mark at the top right of each marked image in view.
    fn draw_marks(&self, frame: &mut Frame, core: &Core) {
        for view in self.layout.views() {
            let tile = &self.layout.tiles()[view.index];
            if core.marked(&tile.item.hit.path) {
                let at = Position::new(tile.rect.right().saturating_sub(3), tile.rect.y);
                draw_check(frame, self.layout.own_cells(view.index, at, 3));
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
            let selected = pause.selected == Some(tile.id);
            let area = if selected {
                let at = tile.rect.as_position();
                Rect::new(at.x, at.y, 3, 1).intersection(tile.rect)
            } else {
                self.layout.own_cells(view.index, view.anchor, 3)
            };
            draw_hint(frame, area, i, selected);
        }
    }
}

impl Board for Scatter {
    fn tick(&mut self, core: &mut Core, elapsed: Duration) -> Result<Outcome> {
        if !self.paused() {
            self.clock += elapsed;
        }
        core.drain(|hit| self.feed.push(hit))?;
        for card in self.layout.fit(core.bounds) {
            self.feed.retry(card.hit);
        }
        if !core.searching && self.feed.is_empty() && self.queue.is_empty() {
            if self.layout.landed() == 0 {
                return Ok(Break(Exit::Empty));
            }
            // Before the next landing, so a new image never gets its first frame under hints.
            if self.layout.ahead() == 0 {
                self.pause();
            }
        }
        self.request_thumbs(core);
        if !self.paused() && self.clock >= self.next_spawn && self.advance(core) {
            self.next_spawn = self.clock + self.interval;
        }
        Ok(Continue(()))
    }

    fn idle(&self) -> bool {
        self.paused()
    }

    fn key(&mut self, core: &mut Core, code: KeyCode) {
        match code {
            KeyCode::Enter => self.select_best(),
            KeyCode::Char(' ') if self.paused() => self.state = State::Running,
            KeyCode::Char(' ') => self.pause(),
            KeyCode::Left => self.back(),
            KeyCode::Right => self.forward(core),
            KeyCode::Char('+' | '=') => self.speed(0.8),
            KeyCode::Char('-') => self.speed(1.25),
            KeyCode::Char(']') => self.scale(1.25),
            KeyCode::Char('[') => self.scale(0.8),
            _ => {}
        }
    }

    fn selected(&self) -> Option<&Path> {
        self.selected_tile().map(|t| t.item.hit.path.as_path())
    }

    fn select(&mut self, hint: usize) -> bool {
        let State::Paused(pause) = &mut self.state else {
            return false;
        };
        let Some(&id) = pause.hints.get(hint) else {
            return false;
        };
        pause.selected = Some(id);
        true
    }

    // Forward stops at the newest image, because a new kitty image must show in a tile first.
    fn step(&mut self, step: Step) -> bool {
        let moved = match step {
            Step::Back => self.layout.back(),
            Step::Forward => self.layout.forward(),
        };
        if moved {
            self.select_top();
        }
        moved
    }

    fn is_empty(&self) -> bool {
        self.layout.tiles().is_empty()
    }

    fn prepare(&mut self, terminal: &mut DefaultTerminal, core: &Core) -> io::Result<()> {
        if core.graphics != Graphics::Layered {
            return Ok(());
        }
        let order = self.draw_order();
        let scene = self.scene(&order, core);
        if !scene.follows(&self.shown) {
            terminal.clear()?;
            // One frame per tile, so each image covers the ones below it. The top image gets its
            // own frame too, because a hint over its first cell would lose its pixels.
            if !scene.zoom {
                for n in 1..=order.len() {
                    terminal.draw(|frame| self.draw_tiles(frame, core, &order[..n]))?;
                }
            }
        }
        self.shown = scene;
        Ok(())
    }

    fn draw(&self, frame: &mut Frame, core: &Core) {
        self.draw_tiles(frame, core, &self.draw_order());
        self.draw_marks(frame, core);
        if let State::Paused(pause) = &self.state {
            self.draw_hints(frame, pause);
        }
    }

    fn status(&self) -> Vec<Span<'static>> {
        let rate = 1.0 / self.interval.as_secs_f32();
        let size = self.size * 100.0;
        let mut spans = vec![Span::raw(format!("{rate:.1}/s  size {size:.0}%  "))];
        if self.layout.ahead() > 0 {
            spans.push(format!("{} back  ", self.layout.ahead()).yellow());
        }
        spans
    }

    fn keys(&self) -> Span<'static> {
        match self.state {
            State::Paused(_) => {
                "PAUSED  ←/→ step · a-z select · A-Z zoom · tab mark · enter pick".yellow()
            }
            State::Running => {
                "space pause · ← back · enter pick · +/- speed · [/] size · esc quit".dim()
            }
        }
    }
}
