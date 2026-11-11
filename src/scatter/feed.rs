use std::{
    cmp::Ordering,
    collections::{BinaryHeap, VecDeque},
    hash::{BuildHasher, RandomState},
    sync::Arc,
};

use crate::{embed::cos, search::Hit};

const TOP: usize = 8;
const RECENT: usize = 12;
const LAMBDA: f32 = 0.5;

pub struct Feed {
    heap: BinaryHeap<Entry>,
    recent: VecDeque<Arc<[f32]>>,
    seq: u64,
    /// None keeps the order in which the hits come.
    shuffle: Option<RandomState>,
    lo: f32,
    hi: f32,
}

impl Feed {
    /// Equal scores, and all hits of an unsorted search, come out in random order: walk order
    /// would show a folder at a time, and burst shots side by side.
    pub fn shuffled() -> Self {
        Self::new(Some(RandomState::new()))
    }

    /// Hits come out in the order they came in, for a search sorted by date.
    pub fn in_order() -> Self {
        Self::new(None)
    }

    fn new(shuffle: Option<RandomState>) -> Self {
        Self {
            heap: BinaryHeap::new(),
            recent: VecDeque::with_capacity(RECENT),
            seq: 0,
            shuffle,
            lo: f32::INFINITY,
            hi: f32::NEG_INFINITY,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    pub fn push(&mut self, hit: Hit) {
        let key = match &self.shuffle {
            Some(shuffle) => shuffle.hash_one(self.seq),
            // Below the key of `retry`.
            None => u64::MAX - 1 - self.seq,
        };
        self.seq += 1;
        self.insert(key, hit);
    }

    /// Puts back a hit whose image no longer fits the terminal, ahead of the hits with an equal
    /// score. It no longer counts as recently picked.
    pub fn retry(&mut self, hit: Hit) {
        if let Some(e) = &hit.embedding {
            self.recent.retain(|r| !Arc::ptr_eq(r, e));
        }
        self.insert(u64::MAX, hit);
    }

    fn insert(&mut self, key: u64, hit: Hit) {
        if let Some(s) = hit.score {
            self.lo = self.lo.min(s);
            self.hi = self.hi.max(s);
        }
        self.heap.push(Entry { key, hit });
    }

    /// Next hit to show and its emphasis in 0..=1: the best of the top few after an MMR
    /// penalty for looking like recently picked hits.
    pub fn pop(&mut self) -> Option<(Hit, f32)> {
        let mut top: Vec<Entry> = (0..TOP).map_while(|_| self.heap.pop()).collect();
        let values: Vec<f32> = top.iter().map(|e| self.value(&e.hit)).collect();
        // rev: max_by keeps the last maximum, so ties go to the best heap entry.
        let best = (0..top.len())
            .rev()
            .max_by(|&a, &b| values[a].total_cmp(&values[b]))?;
        let hit = top.swap_remove(best).hit;
        self.heap.extend(top);
        if let Some(e) = &hit.embedding {
            if self.recent.len() == RECENT {
                self.recent.pop_front();
            }
            self.recent.push_back(e.clone());
        }
        let emphasis = self.emphasis(&hit);
        Some((hit, emphasis))
    }

    fn emphasis(&self, hit: &Hit) -> f32 {
        hit.score.map_or(0.0, |s| {
            if self.hi > self.lo {
                (s - self.lo) / (self.hi - self.lo)
            } else {
                1.0
            }
        })
    }

    fn value(&self, hit: &Hit) -> f32 {
        let redundancy = hit.embedding.as_ref().map_or(0.0, |e| {
            self.recent.iter().map(|r| cos(e, r)).fold(0.0, f32::max)
        });
        self.emphasis(hit) - LAMBDA * redundancy
    }
}

/// The higher score comes first. For equal scores, the higher key comes first.
struct Entry {
    key: u64,
    hit: Hit,
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        let score = |e: &Self| e.hit.score.unwrap_or(0.0);
        score(self)
            .total_cmp(&score(other))
            .then(self.key.cmp(&other.key))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Entry {}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn hit(name: &str, score: Option<f32>, embedding: Option<&[f32]>) -> Hit {
        Hit {
            score,
            embedding: embedding.map(Arc::from),
            ..Hit::new(PathBuf::from(name))
        }
    }

    fn order(feed: &mut Feed) -> Vec<String> {
        std::iter::from_fn(|| feed.pop())
            .map(|(h, _)| h.path.display().to_string())
            .collect()
    }

    #[test]
    fn unranked_is_shuffled() {
        let mut feed = Feed::shuffled();
        let names: Vec<String> = (0..100).map(|i| i.to_string()).collect();
        for name in &names {
            feed.push(hit(name, None, None));
        }
        let mut shown = order(&mut feed);
        assert_ne!(shown, names);
        shown.sort_by_key(|n| n.parse::<u32>().unwrap());
        assert_eq!(shown, names);
    }

    #[test]
    fn in_order_keeps_order_and_retries_first() {
        let mut feed = Feed::in_order();
        let names: Vec<String> = (0..20).map(|i| i.to_string()).collect();
        for name in &names {
            feed.push(hit(name, None, None));
        }
        let (first, _) = feed.pop().unwrap();
        assert_eq!(first.path, PathBuf::from("0"));
        feed.retry(first);
        assert_eq!(order(&mut feed), names);

        let mut feed = Feed::shuffled();
        (0..20).for_each(|i| feed.push(hit(&i.to_string(), None, None)));
        feed.retry(hit("again", None, None));
        assert_eq!(order(&mut feed)[0], "again");
    }

    #[test]
    fn retry_is_not_penalized_by_itself() {
        let mut feed = Feed::shuffled();
        feed.push(hit("a", Some(1.0), Some(&[1.0, 0.0])));
        feed.push(hit("b", Some(0.95), Some(&[0.0, 1.0])));
        feed.push(hit("floor", Some(0.0), None));
        let (a, _) = feed.pop().unwrap();
        feed.retry(a);
        assert_eq!(order(&mut feed), ["a", "b", "floor"]);
    }

    #[test]
    fn ranked_without_embeddings_is_by_score() {
        let mut feed = Feed::shuffled();
        feed.push(hit("low", Some(0.1), None));
        feed.push(hit("high", Some(0.9), None));
        feed.push(hit("mid", Some(0.5), None));
        assert_eq!(order(&mut feed), ["high", "mid", "low"]);
    }

    #[test]
    fn mmr_defers_near_duplicates() {
        let burst: &[f32] = &[1.0, 0.0];
        let other: &[f32] = &[0.0, 1.0];
        let mut feed = Feed::shuffled();
        feed.push(hit("burst1", Some(1.0), Some(burst)));
        feed.push(hit("burst2", Some(0.95), Some(burst)));
        feed.push(hit("other", Some(0.8), Some(other)));
        feed.push(hit("floor", Some(0.0), None));
        assert_eq!(order(&mut feed), ["burst1", "other", "burst2", "floor"]);
    }

    #[test]
    fn mmr_only_looks_at_the_top() {
        let same: &[f32] = &[1.0, 0.0];
        let mut feed = Feed::shuffled();
        feed.push(hit("first", Some(1.0), Some(same)));
        for i in 0..TOP {
            feed.push(hit(&format!("dup{i}"), Some(0.9), Some(same)));
        }
        feed.push(hit("diverse", Some(0.0), Some(&[0.0, 1.0])));
        let order = order(&mut feed);
        assert!(order[1].starts_with("dup"), "{order:?}");
        assert_eq!(order.last().unwrap(), "diverse");
    }

    #[test]
    fn emphasis_is_relative_to_seen_scores() {
        let mut feed = Feed::shuffled();
        feed.push(hit("a", Some(0.25), None));
        feed.push(hit("b", Some(0.75), None));
        feed.push(hit("c", Some(0.5), None));
        let emphasis: Vec<f32> = std::iter::from_fn(|| feed.pop()).map(|(_, e)| e).collect();
        assert_eq!(emphasis, [1.0, 0.5, 0.0]);
    }
}
