use std::mem;

use ratatui::layout::{Position, Rect, Size};

const CANDIDATES: usize = 40;
/// Landings after which a tile counts half as much when the next tile picks where to land.
const HALF_LIFE: f32 = 4.0;
/// Top scores age up to this much slower, so they stay in view longer.
const KEEP_BOOST: f32 = 1.0;
/// Tiles with a smaller share of their area in view get no hint and cannot be selected.
const MIN_SHARE: f32 = 0.15;
/// The newest tiles stay in the pile when they are fully covered, so `back` can show them again.
const HISTORY: usize = 64;

pub struct Tile<T> {
    pub rect: Rect,
    /// Unique, in landing order.
    pub id: u32,
    emphasis: f32,
    pub item: T,
}

/// A tile with enough of its area in view to select.
pub struct Selectable {
    pub index: usize,
    /// The top left cell of the part in view.
    pub anchor: Position,
}

/// A pile of tiles, bottom to top. A new tile goes on top and covers older ones. No tile moves,
/// and no tile disappears while any part of it is in view, because a disappearing image pulls
/// the eye. Only a smaller terminal drops the tiles that no longer fit.
pub struct Layout<T> {
    tiles: Vec<Tile<T>>,
    /// The tiles that `back` took off the top. The last one comes back first.
    ahead: Vec<Tile<T>>,
    bounds: Rect,
    /// For each cell of `bounds`, row by row, the index of the tile on top.
    cover: Vec<Option<usize>>,
    landed: u32,
    rng: Rng,
}

impl<T> Layout<T> {
    pub fn new(seed: u64) -> Self {
        Self {
            tiles: Vec::new(),
            ahead: Vec::new(),
            bounds: Rect::ZERO,
            cover: Vec::new(),
            landed: 0,
            rng: Rng(seed | 1),
        }
    }

    pub fn tiles(&self) -> &[Tile<T>] {
        &self.tiles
    }

    pub fn bounds(&self) -> Rect {
        self.bounds
    }

    pub fn top(&self) -> Option<&Tile<T>> {
        self.tiles.last()
    }

    pub fn landed(&self) -> u32 {
        self.landed
    }

    /// How many tiles `forward` can put back.
    pub fn ahead(&self) -> usize {
        self.ahead.len()
    }

    /// Sets the area of the pile and drops the tiles that no longer fit. Gives back the items of
    /// the dropped tiles that `back` took off, so the feed can show them again.
    pub fn fit(&mut self, bounds: Rect) -> Vec<T> {
        if bounds == self.bounds {
            return Vec::new();
        }
        self.bounds = bounds;
        let fits = |t: &Tile<T>| bounds.intersection(t.rect) == t.rect;
        self.tiles.retain(fits);
        let (ahead, gone): (Vec<_>, Vec<_>) =
            mem::take(&mut self.ahead).into_iter().partition(fits);
        self.ahead = ahead;
        self.prune();
        gone.into_iter().map(|t| t.item).collect()
    }

    /// Of a few random spots, the one that hides the content shown longest ago. Empty cells
    /// cost nothing, so the screen fills first. `None` if `size` does not fit.
    pub fn place(&mut self, size: Size) -> Option<Rect> {
        let bounds = self.bounds;
        let free_w = bounds.width.checked_sub(size.width)?;
        let free_h = bounds.height.checked_sub(size.height)?;
        let weights: Vec<f32> = self.tiles.iter().map(|t| self.weight(t)).collect();
        let cost: Vec<f32> = (self.cover.iter())
            .map(|top| top.map_or(0.0, |i| weights[i]))
            .collect();
        let width = usize::from(bounds.width);
        let hidden = |r: Rect| -> f32 {
            (r.y..r.bottom())
                .map(|y| {
                    let start = usize::from(y - bounds.y) * width + usize::from(r.x - bounds.x);
                    cost[start..start + usize::from(r.width)]
                        .iter()
                        .sum::<f32>()
                })
                .sum()
        };
        (0..CANDIDATES)
            .map(|_| Rect {
                x: bounds.x + self.rng.edgy(free_w),
                y: bounds.y + self.rng.edgy(free_h),
                width: size.width,
                height: size.height,
            })
            .map(|r| (r, hidden(r)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(r, _)| r)
    }

    /// Puts a new tile on top. Call it only when no tile is ahead, so the landing order holds.
    pub fn push(&mut self, rect: Rect, emphasis: f32, item: T) {
        debug_assert!(
            self.ahead.is_empty(),
            "a new tile lands under the tiles ahead"
        );
        self.tiles.push(Tile {
            rect,
            id: self.landed,
            emphasis,
            item,
        });
        self.landed += 1;
        self.prune();
    }

    /// Takes the top tile off, as it was before that tile landed. False when one tile is left,
    /// or at the end of the history.
    pub fn back(&mut self) -> bool {
        if self.tiles.len() < 2 || self.ahead.len() >= HISTORY {
            return false;
        }
        self.ahead.extend(self.tiles.pop());
        self.cover = cover(&self.tiles, self.bounds);
        true
    }

    /// Puts back the tile that `back` took last. False when no tile is ahead.
    pub fn forward(&mut self) -> bool {
        let Some(tile) = self.ahead.pop() else {
            return false;
        };
        self.tiles.push(tile);
        self.cover = cover(&self.tiles, self.bounds);
        true
    }

    /// Drops fully covered tiles, except the newest ones.
    /// Drops the tiles that `back` can never uncover: those under the newest few that are also
    /// covered by older tiles.
    fn prune(&mut self) {
        let mut keep = vec![false; self.tiles.len()];
        let base = keep.len().saturating_sub(HISTORY);
        keep[base..].fill(true);
        for &i in cover(&self.tiles[..base], self.bounds).iter().flatten() {
            keep[i] = true;
        }
        if keep.contains(&false) {
            let mut keep = keep.into_iter();
            self.tiles.retain(|_| keep.next() == Some(true));
        }
        self.cover = cover(&self.tiles, self.bounds);
    }

    /// Indices of the tiles with any cell in view, bottom to top.
    pub fn visible(&self) -> Vec<usize> {
        let mut seen = vec![false; self.tiles.len()];
        for &i in self.cover.iter().flatten() {
            seen[i] = true;
        }
        (seen.iter().enumerate())
            .filter_map(|(i, &seen)| seen.then_some(i))
            .collect()
    }

    /// Tiles with enough of their area in view, bottom to top.
    pub fn views(&self) -> Vec<Selectable> {
        struct Part {
            cells: usize,
            anchor: Option<Position>,
        }
        let mut parts: Vec<Part> = (0..self.tiles.len())
            .map(|_| Part {
                cells: 0,
                anchor: None,
            })
            .collect();
        for (cell, top) in self.cover.iter().enumerate() {
            let Some(i) = *top else { continue };
            parts[i].cells += 1;
            parts[i].anchor.get_or_insert(self.position(cell));
        }
        (parts.into_iter().enumerate())
            .filter_map(|(index, part)| {
                let rect = self.tiles[index].rect;
                let area = f32::from(rect.width) * f32::from(rect.height);
                let anchor = part
                    .anchor
                    .filter(|_| part.cells as f32 >= MIN_SHARE * area)?;
                Some(Selectable { index, anchor })
            })
            .collect()
    }

    /// The cells from `at` to the right, up to `len`, where tile `index` is on top.
    pub fn own_cells(&self, index: usize, at: Position, len: u16) -> Rect {
        let width = (0..len)
            .take_while(|&dx| {
                self.top_at(Position::new(at.x.saturating_add(dx), at.y)) == Some(index)
            })
            .count();
        Rect::new(at.x, at.y, width as u16, 1)
    }

    fn top_at(&self, at: Position) -> Option<usize> {
        if !self.bounds.contains(at) {
            return None;
        }
        let x = usize::from(at.x - self.bounds.x);
        let y = usize::from(at.y - self.bounds.y);
        self.cover[y * usize::from(self.bounds.width) + x]
    }

    fn position(&self, cell: usize) -> Position {
        let width = usize::from(self.bounds.width);
        Position::new(
            self.bounds.x + (cell % width) as u16,
            self.bounds.y + (cell / width) as u16,
        )
    }

    fn weight(&self, tile: &Tile<T>) -> f32 {
        let age = (self.landed - tile.id) as f32 / (1.0 + KEEP_BOOST * tile.emphasis);
        0.5f32.powf(age / HALF_LIFE)
    }
}

fn cover<T>(tiles: &[Tile<T>], bounds: Rect) -> Vec<Option<usize>> {
    let width = usize::from(bounds.width);
    let mut top = vec![None; width * usize::from(bounds.height)];
    for (i, tile) in tiles.iter().enumerate() {
        let r = tile.rect.intersection(bounds);
        for y in r.y..r.bottom() {
            let start = usize::from(y - bounds.y) * width + usize::from(r.x - bounds.x);
            top[start..start + usize::from(r.width)].fill(Some(i));
        }
    }
    top
}

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u16) -> u16 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % u64::from(n)) as u16
    }

    /// A value in `0..=n` that hits both ends more often than the inside, so tiles also cover
    /// the screen edges instead of leaving old slivers there.
    fn edgy(&mut self, n: u16) -> u16 {
        let margin = n / 8;
        self.below(n + 2 * margin + 1).saturating_sub(margin).min(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOUNDS: Rect = Rect::new(0, 0, 200, 50);

    fn pile(rects: &[Rect]) -> Layout<usize> {
        let mut layout = Layout::new(1);
        layout.fit(BOUNDS);
        for (i, r) in rects.iter().enumerate() {
            layout.push(*r, 0.0, i);
        }
        layout
    }

    fn items(layout: &Layout<usize>) -> Vec<usize> {
        layout.tiles.iter().map(|t| t.item).collect()
    }

    #[test]
    fn fills_empty_space_first() {
        let mut layout = pile(&[Rect::new(0, 0, 100, 50)]);
        let rect = layout.place(Size::new(40, 20)).unwrap();
        assert!(rect.x >= 100, "{rect:?}");
    }

    #[test]
    fn covers_the_oldest_tile() {
        let mut layout = pile(&[Rect::new(0, 0, 100, 50), Rect::new(100, 0, 100, 50)]);
        let rect = layout.place(Size::new(40, 20)).unwrap();
        assert!(rect.right() <= 100, "{rect:?}");
    }

    #[test]
    fn top_scores_age_slower() {
        let mut layout = Layout::new(1);
        layout.push(Rect::new(0, 0, 10, 10), 1.0, ());
        layout.push(Rect::new(20, 0, 10, 10), 0.0, ());
        layout.landed = 10;
        assert!(layout.weight(&layout.tiles[0]) > layout.weight(&layout.tiles[1]));
    }

    #[test]
    fn placements_stay_in_bounds() {
        for seed in 0..20 {
            let mut layout = Layout::new(seed);
            layout.fit(BOUNDS);
            for i in 0..50 {
                let rect = layout.place(Size::new(60, 20)).unwrap();
                assert_eq!(BOUNDS.intersection(rect), rect);
                layout.push(rect, 0.0, i);
            }
        }
    }

    #[test]
    fn too_big_does_not_fit() {
        let mut layout = Layout::<()>::new(1);
        layout.fit(BOUNDS);
        assert!(layout.place(Size::new(201, 10)).is_none());
        assert!(layout.place(Size::new(200, 50)).is_some());
    }

    #[test]
    fn keeps_covered_tiles_only_for_history() {
        let full: Vec<Rect> = vec![BOUNDS; HISTORY + 3];
        let mut layout = pile(&full);
        assert_eq!(items(&layout), (2..HISTORY + 3).collect::<Vec<_>>());
        assert_eq!(layout.visible(), [HISTORY]);
        while layout.back() {}
        assert_eq!(layout.ahead(), HISTORY);
        assert_eq!(items(&layout), [2]);
    }

    #[test]
    fn back_shows_the_pile_as_it_was() {
        let mut layout = pile(&[Rect::new(0, 0, 100, 50), Rect::new(50, 0, 100, 50), BOUNDS]);
        assert!(layout.back());
        assert_eq!(layout.visible(), [0, 1]);
        assert!(layout.back());
        assert_eq!(layout.top().map(|t| t.item), Some(0));
        assert!(!layout.back(), "keeps the last tile");
        assert_eq!(layout.ahead(), 2);
        assert!(layout.forward());
        assert!(layout.forward());
        assert!(!layout.forward());
        assert_eq!(items(&layout), [0, 1, 2]);
        assert_eq!(layout.visible(), [2]);
    }

    #[test]
    fn fit_drops_what_no_longer_fits() {
        let mut layout = pile(&[
            Rect::new(0, 0, 20, 10),
            Rect::new(150, 0, 20, 10),
            Rect::new(10, 0, 20, 10),
            Rect::new(160, 0, 20, 10),
        ]);
        layout.back();
        layout.back();
        let gone = layout.fit(Rect::new(0, 0, 100, 50));
        assert_eq!(gone, [3]);
        assert_eq!(items(&layout), [0]);
        assert!(layout.forward());
        assert_eq!(items(&layout), [0, 2]);
    }

    #[test]
    fn views_skip_mostly_covered_tiles() {
        let layout = pile(&[
            Rect::new(0, 0, 100, 50),
            Rect::new(0, 0, 95, 50),
            Rect::new(150, 0, 20, 10),
        ]);
        let views: Vec<usize> = layout.views().iter().map(|v| v.index).collect();
        assert_eq!(views, [1, 2]);
    }

    #[test]
    fn anchor_is_the_top_left_of_the_part_in_view() {
        let layout = pile(&[Rect::new(0, 0, 100, 50), Rect::new(0, 0, 50, 25)]);
        let views = layout.views();
        assert_eq!(views[0].anchor, Position::new(50, 0));
        assert_eq!(views[1].anchor, Position::new(0, 0));
    }

    #[test]
    fn own_cells_stop_at_another_tile() {
        let layout = pile(&[Rect::new(0, 0, 100, 50), Rect::new(2, 0, 50, 25)]);
        assert_eq!(layout.own_cells(0, Position::new(0, 0), 3).width, 2);
        assert_eq!(layout.own_cells(1, Position::new(2, 0), 3).width, 3);
        assert_eq!(layout.own_cells(0, Position::new(199, 49), 3).width, 0);
    }
}
