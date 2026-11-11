use std::{
    collections::{HashMap, hash_map::Entry},
    io,
    ops::{
        ControlFlow::{Break, Continue},
        Range,
    },
    path::Path,
    time::Duration,
};

use anyhow::Result;
use ratatui::{
    DefaultTerminal, Frame,
    crossterm::event::KeyCode,
    layout::{Constraint, Rect, Size},
    style::Stylize,
    text::{Line, Span},
    widgets::Block,
};
use ratatui_image::{FontSize, protocol::Protocol};

use super::{
    Board, Core, Exit, Graphics, HINTS, Loading, Outcome, Step, draw_check, draw_hint,
    prefetch::{Prefetch, Shape, Thumb},
    protocol,
};
use crate::search::Hit;

/// The aspect of most photos.
const PHOTO: f32 = 3.0 / 2.0;
/// Thumbnails that wait for a worker at a time. The workers take them in the order they come,
/// so with few in line, the page in view goes first after a page turn. A dropped ticket skips
/// its decode.
const LOADING: usize = 4;
/// Labels must miss the first cell of a tile, because that cell sends a new kitty image. So a
/// tile needs two rows and two columns.
const MIN_TILE: u16 = 2;

/// Pages of thumbnails in the order of the search. Each image on a page has a hint letter.
#[derive(Default)]
pub struct Grid {
    hits: Vec<Hit>,
    /// The index of the selected hit. The page that holds it is in view.
    selected: usize,
    /// `None` while no tile fits the terminal.
    geometry: Option<Geometry>,
    /// Thumbnails for the page in view and the pages next to it, by hit index. A failed one
    /// stays, so a broken image decodes only once.
    thumbs: HashMap<usize, Loading<Thumb>>,
    /// The images on screen, by hit index.
    images: HashMap<usize, Protocol>,
    shown: Scene,
}

/// What a frame shows, for sixel and iTerm2. Their images stay where the terminal drew them,
/// so a new page, a closed zoom, and a check mark that goes away need a clear screen.
#[derive(Default, PartialEq)]
struct Scene {
    bounds: Rect,
    page: usize,
    zoom: bool,
    /// The marked hits on the page.
    marked: Vec<usize>,
}

impl Grid {
    /// The hits on the page in view.
    fn page(&self, geometry: Geometry) -> Range<usize> {
        let start = self.selected / geometry.len() * geometry.len();
        start..(start + geometry.len()).min(self.hits.len())
    }

    /// Selects hit `i`. False if there is no such hit.
    fn select_hit(&mut self, i: Option<usize>) -> bool {
        let found = i.filter(|&i| i < self.hits.len());
        if let Some(i) = found {
            self.selected = i;
        }
        found.is_some()
    }

    /// Shows the page before or after, and selects its first image.
    fn turn(&mut self, step: Step) {
        let Some(geometry) = self.geometry else {
            return;
        };
        let start = self.page(geometry).start;
        self.select_hit(match step {
            Step::Back => start.checked_sub(geometry.len()),
            Step::Forward => Some(start + geometry.len()),
        });
    }

    /// Asks for the thumbnails of the page in view and the pages next to it, a few at a time,
    /// and drops the others.
    fn request(&mut self, prefetch: &Prefetch, geometry: Geometry) {
        let page = self.page(geometry);
        let near = page.start.saturating_sub(geometry.len())..page.start + 2 * geometry.len();
        // The page in view first, then the next page, then the page before.
        let next: Vec<usize> = (page.start..near.end)
            .chain(near.start..page.start)
            .filter(|&i| i < self.hits.len())
            .filter(|i| {
                !matches!(
                    self.thumbs.get(i),
                    Some(Loading::Ready(_) | Loading::Failed)
                )
            })
            .take(LOADING)
            .collect();
        self.thumbs.retain(|i, thumb| match thumb {
            Loading::Pending(_) => next.contains(i),
            Loading::Ready(_) => near.contains(i),
            Loading::Failed => true,
        });
        for i in next {
            if let Entry::Vacant(entry) = self.thumbs.entry(i) {
                let path = self.hits[i].path.clone();
                let thumb = prefetch.thumb(path, Shape::Fill(geometry.tile));
                entry.insert(Loading::Pending(thumb));
            }
        }
    }

    fn scene(&self, core: &Core, geometry: Geometry) -> Scene {
        let page = self.page(geometry);
        Scene {
            bounds: core.bounds,
            page: page.start / geometry.len(),
            zoom: core.zoom.is_some(),
            marked: page.filter(|&i| core.marked(&self.hits[i].path)).collect(),
        }
    }
}

impl Board for Grid {
    fn tick(&mut self, core: &mut Core, _: Duration) -> Result<Outcome> {
        core.drain(|hit| self.hits.push(hit))?;
        if !core.searching && self.hits.is_empty() {
            return Ok(Break(Exit::Empty));
        }
        // More hits can come until the search ends.
        let images = if core.searching {
            HINTS.len()
        } else {
            self.hits.len()
        };
        let geometry = Geometry::new(core.room(), core.picker.font_size(), images);
        if geometry != self.geometry {
            // Each thumbnail fills its tile exactly, so new tiles need new thumbnails.
            self.thumbs
                .retain(|_, thumb| matches!(thumb, Loading::Failed));
            self.images.clear();
            self.geometry = geometry;
        }
        let Some(geometry) = geometry else {
            return Ok(Continue(()));
        };
        let page = self.page(geometry);
        self.images.retain(|i, _| page.contains(i));
        for thumb in self.thumbs.values_mut() {
            thumb.poll(Some);
        }
        self.request(&core.prefetch, geometry);
        Ok(Continue(()))
    }

    fn idle(&self) -> bool {
        !self.thumbs.values().any(Loading::pending)
    }

    fn key(&mut self, _: &mut Core, code: KeyCode) {
        match code {
            KeyCode::Right | KeyCode::PageDown | KeyCode::Char(' ') => self.turn(Step::Forward),
            KeyCode::Left | KeyCode::PageUp => self.turn(Step::Back),
            _ => {}
        }
    }

    fn selected(&self) -> Option<&Path> {
        self.hits.get(self.selected).map(|hit| hit.path.as_path())
    }

    fn select(&mut self, hint: usize) -> bool {
        let Some(geometry) = self.geometry else {
            return false;
        };
        let page = self.page(geometry);
        self.select_hit(Some(page.start + hint).filter(|i| page.contains(i)))
    }

    fn step(&mut self, step: Step) -> bool {
        self.select_hit(match step {
            Step::Back => self.selected.checked_sub(1),
            Step::Forward => Some(self.selected + 1),
        })
    }

    fn is_empty(&self) -> bool {
        self.hits.is_empty()
    }

    fn prepare(&mut self, terminal: &mut DefaultTerminal, core: &Core) -> io::Result<()> {
        let Some(geometry) = self.geometry else {
            return Ok(());
        };
        if core.zoom.is_none() {
            for i in self.page(geometry) {
                let Some(Loading::Ready(thumb)) = self.thumbs.get(&i) else {
                    continue;
                };
                if self.images.contains_key(&i) {
                    continue;
                }
                match protocol(&core.picker, &core.search.progress, thumb.clone()) {
                    Some(image) => _ = self.images.insert(i, image),
                    None => _ = self.thumbs.insert(i, Loading::Failed),
                }
            }
        }
        if core.graphics == Graphics::Layered {
            let scene = self.scene(core, geometry);
            if scene != self.shown {
                terminal.clear()?;
                self.shown = scene;
            }
        }
        Ok(())
    }

    fn draw(&self, frame: &mut Frame, core: &Core) {
        let Some(geometry) = self.geometry else {
            return;
        };
        for (n, i) in self.page(geometry).enumerate() {
            let rect = geometry.tile(core.bounds, n);
            let path = &self.hits[i].path;
            let color = core.edge(path, i == self.selected);
            if let Some(image) = self.images.get(&i) {
                core.draw_image(frame, image, rect);
                core.tint(frame, rect, color);
            } else if matches!(self.thumbs.get(&i), Some(Loading::Failed)) {
                frame.render_widget(Block::bordered().border_style(color), rect);
                let row = rect.centered_vertically(Constraint::Length(1));
                frame.render_widget(Line::from("cannot open").centered().dim(), row);
            } else {
                continue;
            }
            // The labels stay off the first cell, which sends a new kitty image.
            let hint = Rect::new(rect.x, rect.bottom() - 1, 3, 1).intersection(rect);
            draw_hint(frame, hint, n, i == self.selected);
            if core.marked(path) {
                let x = rect.right().saturating_sub(3).max(rect.x + 1);
                draw_check(frame, Rect::new(x, rect.y, rect.right() - x, 1));
            }
        }
    }

    fn status(&self) -> Vec<Span<'static>> {
        let page = match self.geometry {
            Some(geometry) if !self.hits.is_empty() => {
                let pages = self.hits.len().div_ceil(geometry.len());
                format!("page {}/{pages}  ", self.selected / geometry.len() + 1)
            }
            _ => String::new(),
        };
        let n = self.hits.len();
        let images = if n == 1 { "image" } else { "images" };
        vec![Span::raw(format!("{page}{n} {images}  "))]
    }

    fn keys(&self) -> Span<'static> {
        "a-z select · A-Z zoom · tab mark · enter pick · ←/→ page · esc quit".dim()
    }
}

/// How a page splits the screen into tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Geometry {
    cols: u16,
    rows: u16,
    /// The cells of one tile.
    tile: Size,
}

impl Geometry {
    /// The grid that shows photos best in `bounds`, or `None` if no tile fits. It has the
    /// highest count² × the area of a 3:2 photo in one tile, where count is the number of images
    /// on a page, at most `images`. Count × area is the part of the screen that photos cover, and
    /// there one tile gets the same score as four or sixteen. So the count counts twice: more
    /// images on a page means fewer pages.
    fn new(bounds: Size, font: FontSize, images: usize) -> Option<Self> {
        let most = HINTS.len() as u16;
        (1..=most)
            .flat_map(|cols| (1..=most / cols).map(move |rows| (cols, rows)))
            .map(|(cols, rows)| Self {
                cols,
                rows,
                tile: Size::new(bounds.width / cols, bounds.height / rows),
            })
            .filter(|g| g.tile.width >= MIN_TILE && g.tile.height >= MIN_TILE)
            .max_by(|a, b| a.score(font, images).total_cmp(&b.score(font, images)))
    }

    fn score(self, font: FontSize, images: usize) -> f32 {
        let width = f32::from(self.tile.width) * f32::from(font.width);
        let height = f32::from(self.tile.height) * f32::from(font.height);
        let photo = width.min(height * PHOTO);
        let count = self.len().min(images) as f32;
        count * count * photo * photo / PHOTO
    }

    fn len(self) -> usize {
        usize::from(self.cols * self.rows)
    }

    /// Tile `n` of a page, in reading order. The tiles touch, and the grid is in the middle of
    /// `bounds`.
    fn tile(self, bounds: Rect, n: usize) -> Rect {
        let (col, row) = (n as u16 % self.cols, n as u16 / self.cols);
        let (width, height) = (self.tile.width, self.tile.height);
        Rect::new(
            bounds.x + (bounds.width - self.cols * width) / 2 + col * width,
            bounds.y + (bounds.height - self.rows * height) / 2 + row * height,
            width,
            height,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    const FONT: FontSize = FontSize::new(10, 20);

    fn shape(width: u16, height: u16) -> (u16, u16, Size) {
        let g = Geometry::new(Size::new(width, height), FONT, HINTS.len()).unwrap();
        (g.cols, g.rows, g.tile)
    }

    #[test]
    fn tiles_get_the_shape_of_photos() {
        assert_eq!(shape(200, 49), (6, 4, Size::new(33, 12)));
        assert_eq!(shape(80, 23), (6, 4, Size::new(13, 5)));
        assert_eq!(shape(200, 30), (8, 3, Size::new(25, 10)));
        assert_eq!(shape(60, 60), (3, 8, Size::new(20, 7)));
    }

    #[test]
    fn a_screen_shaped_like_a_photo_gets_more_than_one_tile() {
        assert_eq!(shape(160, 49), (5, 5, Size::new(32, 9)));
    }

    #[test]
    fn a_few_images_get_large_tiles_on_one_page() {
        let shape = |images| {
            let g = Geometry::new(Size::new(200, 49), FONT, images).unwrap();
            (g.cols, g.rows)
        };
        assert_eq!(shape(1), (1, 1));
        assert_eq!(shape(2), (2, 1));
        assert_eq!(shape(3), (2, 2));
        assert_eq!(shape(24), (6, 4));
        assert_eq!(shape(500), (6, 4));
    }

    #[test]
    fn tiles_fit_and_get_one_hint_each() {
        for width in (1..400).step_by(7) {
            for height in (1..120).step_by(5) {
                let Some(g) = Geometry::new(Size::new(width, height), FONT, HINTS.len()) else {
                    assert!(width < MIN_TILE || height < MIN_TILE);
                    continue;
                };
                assert!(g.len() <= HINTS.len());
                assert!(g.tile.width >= MIN_TILE && g.tile.height >= MIN_TILE);
                assert!(g.cols * g.tile.width <= width && g.rows * g.tile.height <= height);
            }
        }
    }

    #[test]
    fn tiles_touch_in_reading_order_in_the_middle() {
        let g = Geometry {
            cols: 3,
            rows: 2,
            tile: Size::new(10, 5),
        };
        let bounds = Rect::new(0, 0, 33, 11);
        assert_eq!(g.tile(bounds, 0), Rect::new(1, 0, 10, 5));
        assert_eq!(g.tile(bounds, 1), Rect::new(11, 0, 10, 5));
        assert_eq!(g.tile(bounds, 3), Rect::new(1, 5, 10, 5));
        assert_eq!(g.tile(bounds, 5), Rect::new(21, 5, 10, 5));
    }

    #[test]
    fn pages_and_steps() {
        let mut grid = Grid {
            hits: (0..30)
                .map(|i| Hit::new(PathBuf::from(i.to_string())))
                .collect(),
            geometry: Some(Geometry {
                cols: 4,
                rows: 3,
                tile: Size::new(10, 5),
            }),
            ..Grid::default()
        };
        assert!(grid.select(11));
        assert!(!grid.select(12), "the next page has no hints yet");
        grid.turn(Step::Back);
        assert_eq!(grid.selected, 11, "no page before the first");
        grid.turn(Step::Forward);
        assert_eq!(grid.selected, 12);
        grid.turn(Step::Forward);
        assert_eq!(grid.selected(), Some(Path::new("24")));
        assert!(!grid.select(6), "the last page holds 24 to 29");
        grid.turn(Step::Forward);
        assert_eq!(grid.selected, 24, "no page after the last");
        assert!(grid.step(Step::Back), "the zoom steps across pages");
        assert_eq!(grid.selected, 23);
        grid.selected = 29;
        assert!(!grid.step(Step::Forward));
        grid.turn(Step::Back);
        assert_eq!(grid.selected, 12);
    }
}
