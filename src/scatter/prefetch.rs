use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::Ordering::Relaxed,
        mpsc::{self, Receiver, TryRecvError},
    },
};

use anyhow::Result;
use image::{DynamicImage, Rgba, imageops::FilterType};
use ratatui::layout::Size;
use ratatui_image::FontSize;
use rayon::{ThreadPool, ThreadPoolBuilder};

use super::layout;
use crate::{
    decode, output,
    search::{Hit, Progress},
};

const READY: usize = 4;
const WORKERS: usize = 2;

/// An image that fills `cells` exactly at the terminal's font size.
pub struct Thumb {
    pub image: DynamicImage,
    pub cells: Size,
}

pub struct Prepared {
    pub hit: Hit,
    pub emphasis: f32,
    pub thumb: Thumb,
}

struct Pending {
    hit: Hit,
    emphasis: f32,
    thumb: Receiver<Option<Thumb>>,
}

/// Decodes and shrinks images on worker threads so the UI thread never decodes.
pub struct Prefetch {
    pool: ThreadPool,
    font: FontSize,
    edge: Rgba<u8>,
    /// In request order, so the feed order holds when a later image is ready first.
    queue: VecDeque<Pending>,
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
            queue: VecDeque::new(),
            progress: progress.clone(),
        })
    }

    /// Shrinks the image at `path` to about `area` cells, no larger than `room`.
    pub fn thumb(&self, path: PathBuf, area: f32, room: Size) -> Receiver<Option<Thumb>> {
        let (tx, rx) = mpsc::sync_channel(1);
        let (font, edge) = (self.font, self.edge);
        self.pool
            .spawn(move || _ = tx.send(thumbnail(&path, area, room, font, edge)));
        rx
    }

    pub fn wants(&self) -> bool {
        self.queue.len() < READY
    }

    pub fn request(&mut self, hit: Hit, emphasis: f32, area: f32, room: Size) {
        let thumb = self.thumb(hit.path.clone(), area, room);
        self.queue.push_back(Pending {
            hit,
            emphasis,
            thumb,
        });
    }

    /// The next image in request order, once it is ready.
    pub fn take(&mut self) -> Option<Prepared> {
        loop {
            let pending = self.queue.pop_front()?;
            match pending.thumb.try_recv() {
                Err(TryRecvError::Empty) => {
                    self.queue.push_front(pending);
                    return None;
                }
                Ok(Some(thumb)) => {
                    return Some(Prepared {
                        hit: pending.hit,
                        emphasis: pending.emphasis,
                        thumb,
                    });
                }
                Ok(None) | Err(TryRecvError::Disconnected) => {
                    self.progress.errors.fetch_add(1, Relaxed);
                }
            }
        }
    }

    pub fn idle(&self) -> bool {
        self.queue.is_empty()
    }
}

fn thumbnail(path: &Path, area: f32, room: Size, font: FontSize, edge: Rgba<u8>) -> Option<Thumb> {
    let image = decode::load(path, output::pixel_side(room, font)).ok()?;
    let cell = Size::new(font.width, font.height);
    let cells = layout::image_cells((image.width(), image.height()), cell, area, room);
    // Whole cells rarely match the image's aspect. A crop of less than one cell lets the image
    // touch its border on all sides.
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
