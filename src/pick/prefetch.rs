use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Weak,
        atomic::Ordering::Relaxed,
        mpsc::{self, Receiver, TryRecvError},
    },
    task::Poll,
};

use anyhow::Result;
use image::{DynamicImage, Rgba, imageops::FilterType};
use ratatui::layout::Size;
use ratatui_image::FontSize;
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::{decode, output, search::Progress};

const WORKERS: usize = 2;

/// An image that fills `cells` exactly at the terminal's font size.
#[derive(Clone)]
pub struct Thumb {
    pub image: DynamicImage,
    pub cells: Size,
}

/// The cells that a thumbnail fills.
#[derive(Clone, Copy)]
pub enum Shape {
    /// The image's own aspect, about `area` cells, and no larger than `room`.
    Fit { area: f32, room: Size },
    /// Exactly these cells. The image is cropped to their aspect.
    Fill(Size),
}

impl Shape {
    /// The side of the square that the image shrinks to fit before the crop.
    fn side(self, font: FontSize) -> u32 {
        match self {
            Shape::Fit { room, .. } => output::pixel_side(room, font),
            // Twice the long side, so a photo turned the other way still covers the cells.
            Shape::Fill(cells) => 2 * output::pixel_side(cells, font),
        }
    }

    fn cells(self, image: (u32, u32), font: FontSize) -> Size {
        match self {
            Shape::Fit { area, room } => image_cells(image, font, area, room),
            Shape::Fill(cells) => cells,
        }
    }
}

/// A thumbnail on its way from a worker. Dropping the ticket cancels the work that has not
/// started yet, so the images in view do not wait behind images that are gone.
pub struct Ticket {
    thumb: Receiver<Option<Thumb>>,
    /// The worker checks that this is alive before each slow step.
    _wanted: Arc<()>,
    progress: Arc<Progress>,
}

impl Ticket {
    /// The thumbnail once the worker is done, or `None` if the image cannot be shown. A failure
    /// counts as an unreadable file. Poll until the ticket is ready, then drop it.
    pub fn poll(&self) -> Poll<Option<Thumb>> {
        let thumb = match self.thumb.try_recv() {
            Err(TryRecvError::Empty) => return Poll::Pending,
            Ok(thumb) => thumb,
            Err(TryRecvError::Disconnected) => None,
        };
        if thumb.is_none() {
            self.progress.errors.fetch_add(1, Relaxed);
        }
        Poll::Ready(thumb)
    }
}

/// Decodes and shrinks images on worker threads so the UI thread never decodes.
pub struct Prefetch {
    pool: ThreadPool,
    font: FontSize,
    edge: Rgba<u8>,
    progress: Arc<Progress>,
}

impl Prefetch {
    pub fn new(font: FontSize, edge: Rgba<u8>, progress: &Arc<Progress>) -> Result<Self> {
        let pool = ThreadPoolBuilder::new()
            .num_threads(WORKERS)
            // A panic drops the sender, so the receiver sees a failed image.
            .panic_handler(|_| {})
            .build()?;
        Ok(Self {
            pool,
            font,
            edge,
            progress: progress.clone(),
        })
    }

    /// Makes a thumbnail of the image at `path` on a worker thread.
    pub fn thumb(&self, path: PathBuf, shape: Shape) -> Ticket {
        let (tx, rx) = mpsc::sync_channel(1);
        let wanted = Arc::new(());
        let (font, edge, weak) = (self.font, self.edge, Arc::downgrade(&wanted));
        self.pool
            .spawn(move || _ = tx.send(thumbnail(&path, shape, font, edge, &weak)));
        Ticket {
            thumb: rx,
            _wanted: wanted,
            progress: self.progress.clone(),
        }
    }
}

/// `None` if the image cannot be shown, or once nobody wants it.
fn thumbnail(
    path: &Path,
    shape: Shape,
    font: FontSize,
    edge: Rgba<u8>,
    wanted: &Weak<()>,
) -> Option<Thumb> {
    let alive = || wanted.strong_count() > 0;
    if !alive() {
        return None;
    }
    let image = decode::load(path, shape.side(font)).ok()?;
    if !alive() {
        return None;
    }
    let cells = shape.cells((image.width(), image.height()), font);
    // Whole cells rarely match the image's aspect. A crop lets the image touch its border on all
    // sides.
    let width = u32::from(cells.width) * u32::from(font.width);
    let height = u32::from(cells.height) * u32::from(font.height);
    let mut image = image
        .resize_to_fill(width, height, FilterType::Triangle)
        .into_rgba8();
    // A line drawn in cells takes a full cell, and the rest of that cell hides the images below.
    // A line in the image's own pixels touches the image and hides nothing.
    let line = u32::from(font.width / 5).max(1);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        if x < line || y < line || x >= width - line || y >= height - line {
            *pixel = edge;
        }
    }
    Some(Thumb {
        image: image.into(),
        cells,
    })
}

/// Size in cells for an image of `image` pixels in cells of `font` pixels: the image's own
/// aspect, about `area` cells, and no larger than `room`.
fn image_cells(image: (u32, u32), font: FontSize, area: f32, room: Size) -> Size {
    let aspect = image.0 as f32 / image.1 as f32 * f32::from(font.height) / f32::from(font.width);
    let height = (area / aspect).sqrt();
    let width = height * aspect;
    let fit = (f32::from(room.width) / width)
        .min(f32::from(room.height) / height)
        .min(1.0);
    let cells = |n: f32, max: u16| (n * fit).round().clamp(1.0, f32::from(max.max(1))) as u16;
    Size::new(cells(width, room.width), cells(height, room.height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_cells_keep_aspect_and_area() {
        let font = FontSize::new(10, 20);
        let landscape = image_cells((1600, 1200), font, 1200.0, Size::new(200, 50));
        assert_eq!(landscape, Size::new(57, 21));
        let portrait = image_cells((1200, 1600), font, 1200.0, Size::new(200, 50));
        assert_eq!(portrait, Size::new(42, 28));
    }

    #[test]
    fn image_cells_fit_the_room() {
        let font = FontSize::new(10, 20);
        let room = Size::new(100, 30);
        assert_eq!(
            image_cells((1600, 1200), font, 1e6, room),
            Size::new(80, 30)
        );
        assert_eq!(
            image_cells((1600, 1200), font, 1e6, Size::new(1, 1)),
            Size::new(1, 1)
        );
    }

    #[test]
    fn fill_takes_the_cells() {
        let font = FontSize::new(10, 20);
        let shape = Shape::Fill(Size::new(30, 10));
        assert_eq!(shape.cells((4000, 1000), font), Size::new(30, 10));
        assert_eq!(shape.side(font), 600);
    }
}
