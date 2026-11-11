use std::{collections::VecDeque, task::Poll};

use crate::{
    pick::prefetch::{Prefetch, Shape, Thumb, Ticket},
    search::Hit,
};

/// Images that the queue prepares ahead of their landing.
const READY: usize = 4;

pub struct Prepared {
    pub hit: Hit,
    pub emphasis: f32,
    pub thumb: Thumb,
}

struct Pending {
    hit: Hit,
    emphasis: f32,
    thumb: Ticket,
}

/// The next images, in request order, so the feed order holds when a later image is ready first.
#[derive(Default)]
pub struct Queue(VecDeque<Pending>);

impl Queue {
    pub fn wants(&self) -> bool {
        self.0.len() < READY
    }

    pub fn request(&mut self, prefetch: &Prefetch, hit: Hit, emphasis: f32, shape: Shape) {
        let thumb = prefetch.thumb(hit.path.clone(), shape);
        self.0.push_back(Pending {
            hit,
            emphasis,
            thumb,
        });
    }

    /// The next image in request order, once it is ready.
    pub fn take(&mut self) -> Option<Prepared> {
        loop {
            let Poll::Ready(thumb) = self.0.front()?.thumb.poll() else {
                return None;
            };
            let pending = self.0.pop_front()?;
            if let Some(thumb) = thumb {
                return Some(Prepared {
                    hit: pending.hit,
                    emphasis: pending.emphasis,
                    thumb,
                });
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
