use std::{
    cmp::Ordering,
    collections::HashSet,
    fs, iter,
    num::NonZeroUsize,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering::Relaxed},
        mpsc::{self, Receiver, Sender},
    },
    thread,
};

use anyhow::{Context, Result};
use clap::ValueEnum;
use image::DynamicImage;
use jiff::civil::DateTime;
use rayon::prelude::*;
use regex::Regex;

use crate::{
    cache::{Cache, Key, View},
    catch, decode,
    embed::{self, ImageEmbedder, Prompt, TextEmbedder},
    meta::{self, MetaFilter},
    ocr::Ocr,
    output::clean_path,
    walk::{self, Found, Skipped, Source, WalkOpts},
};

pub struct Query {
    pub source: Source,
    pub walk: WalkOpts,
    pub meta: MetaFilter,
    /// All must match the text in the image.
    pub text: Vec<Regex>,
    /// Hits carry the text in the image, or only the parts that match `text`.
    pub show_text: bool,
    pub order: Order,
    pub limit: Option<NonZeroUsize>,
}

/// A sorted order sends the first hit only after it has seen all images. `S` is what a score
/// needs at each stage: the query, the loaded model, or nothing.
pub enum Order<S = Scoring> {
    /// The walk order, or the order of the paths on stdin.
    Found,
    /// Best first.
    Score(S),
    Date(DateOrder),
}

pub struct Scoring {
    pub rank: Rank,
    pub min: Option<f32>,
    /// The best hits carry their image embedding, so scatter mode can spread out look-alikes.
    pub embeddings: bool,
}

pub enum Rank {
    Text { query: Prompt, not: Vec<Prompt> },
    Like(PathBuf),
}

#[derive(Clone, Copy, ValueEnum)]
pub enum DateOrder {
    Newest,
    Oldest,
}

pub struct Hit {
    pub path: PathBuf,
    pub score: Option<f32>,
    pub embedding: Option<Arc<[f32]>>,
    /// With `show_text`: the text in the image, or the parts that match.
    pub text: Vec<String>,
}

impl Hit {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            score: None,
            embedding: None,
            text: Vec::new(),
        }
    }
}

/// In a sorted search, `done` counts the images that have a score or a date. OCR comes after.
#[derive(Default)]
pub struct Progress {
    pub found: AtomicUsize,
    pub done: AtomicUsize,
    pub errors: AtomicUsize,
    pub unsupported: AtomicUsize,
}

pub struct Search {
    /// An error ends the search.
    pub hits: Receiver<Result<Hit>>,
    pub progress: Arc<Progress>,
    pub order: Order<()>,
    pub skipped: Option<Skipped>,
    cache: Arc<Cache>,
}

impl Search {
    /// Progress for people, like "120/4000 scored".
    pub fn status(&self) -> String {
        let verb = match self.order {
            Order::Score(()) => "scored",
            Order::Found | Order::Date(_) => "checked",
        };
        let (done, found) = (&self.progress.done, &self.progress.found);
        format!("{}/{} {verb}", done.load(Relaxed), found.load(Relaxed))
    }
}

/// The caller can stop before the search ends, so store the results so far.
impl Drop for Search {
    fn drop(&mut self) {
        self.cache.flush();
    }
}

/// Limits the decoded images in memory and the delay before the first hit.
const CHUNK: usize = 32;

/// Images shrink to fit this square before embedding, so a chunk of large images stays small.
const EMBED_SIDE: u32 = 1024;

/// Scatter mode spreads out look-alikes only among the best hits, so only those get their
/// embedding.
const EMBEDDED: usize = 2048;

/// Device and inode.
type FileId = (u64, u64);

pub fn run(query: Query) -> Result<Search> {
    let progress = Arc::new(Progress::default());
    // The walk starts first, so it runs while the models load.
    let walk = walk::candidates(query.source, query.walk)?;
    let paths = counted(walk.found, progress.clone());
    let cache = Arc::new(Cache::open());
    let (order, sort) = match query.order {
        Order::Found => (Order::Found, Order::Found),
        Order::Score(scoring) => (
            Order::Score(()),
            Order::Score(Ranker::new(scoring, &cache)?),
        ),
        Order::Date(date) => (Order::Date(date), Order::Date(date)),
    };
    let ocr = (query.show_text || !query.text.is_empty())
        .then(Ocr::load)
        .transpose()?;
    let (tx, hits) = mpsc::channel();
    let mut sink = Sink::new(tx, query.limit);
    let engine = Engine {
        cache: cache.clone(),
        filter: query.meta,
        ocr,
        text: query.text,
        show_text: query.show_text,
        order: sort,
        progress: progress.clone(),
    };
    thread::spawn(move || {
        // Decoding runs on rten's pool too, so it never competes with inference for cores.
        let run = catch(|| rten::thread_pool().run(|| engine.run(paths, &mut sink)));
        if let Err(e) = run {
            let _ = sink.tx.send(Err(e));
        }
    });
    Ok(Search {
        hits,
        progress,
        order,
        skipped: walk.skipped,
        cache,
    })
}

fn counted(found: Receiver<Found>, progress: Arc<Progress>) -> Receiver<PathBuf> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for item in found {
            let path = match item {
                Found::Image(path) => path,
                Found::Unsupported => {
                    progress.unsupported.fetch_add(1, Relaxed);
                    continue;
                }
                Found::Error => {
                    progress.errors.fetch_add(1, Relaxed);
                    continue;
                }
            };
            progress.found.fetch_add(1, Relaxed);
            if tx.send(path).is_err() {
                break;
            }
        }
    });
    rx
}

struct Ranker {
    embedder: ImageEmbedder,
    target: Target,
    min: Option<f32>,
    embeddings: bool,
}

enum Target {
    Text { query: Vec<f32>, not: Vec<Vec<f32>> },
    Like { image: Vec<f32>, id: FileId },
}

impl Ranker {
    fn new(
        Scoring {
            rank,
            min,
            embeddings,
        }: Scoring,
        cache: &Cache,
    ) -> Result<Self> {
        let embedder = ImageEmbedder::load()?;
        let target = match rank {
            Rank::Text { query, not } => {
                // Drops at the end of this block, with the 300 MB of the text model.
                let model = TextEmbedder::default();
                let embed = |prompt: &Prompt| cache.text_embedding(prompt, || model.embed(prompt));
                Target::Text {
                    query: embed(&query)?,
                    not: not.iter().map(embed).collect::<Result<_>>()?,
                }
            }
            Rank::Like(path) => {
                let (image, id) = reference(&embedder, cache, &path)
                    .with_context(|| format!("--like {}", clean_path(&path)))?;
                Target::Like { image, id }
            }
        };
        Ok(Self {
            embedder,
            target,
            min,
            embeddings,
        })
    }

    /// None when the image is dropped.
    fn score(&self, image: &[f32]) -> Option<f32> {
        let score = self.target.score(image)?;
        self.min.is_none_or(|min| score >= min).then_some(score)
    }
}

impl Target {
    /// None when the image is closer to a negative than to the query.
    fn score(&self, image: &[f32]) -> Option<f32> {
        match self {
            Target::Text { query, not } => {
                let cos = embed::cos(image, query);
                not.iter()
                    .all(|n| embed::cos(image, n) <= cos)
                    .then(|| embed::score(cos))
            }
            Target::Like {
                image: reference, ..
            } => Some(embed::cos(image, reference)),
        }
    }

    /// The --like image does not rank against itself.
    fn excludes(&self, id: FileId) -> bool {
        matches!(self, Target::Like { id: reference, .. } if *reference == id)
    }
}

fn reference(embedder: &ImageEmbedder, cache: &Cache, path: &Path) -> Result<(Vec<f32>, FileId)> {
    let md = fs::metadata(path)?;
    decode::check(&md)?;
    let image = embeddings(embedder, cache, &[&Key::new(path, &md)])?.remove(0)?;
    Ok((image, (md.dev(), md.ino())))
}

enum Input {
    Cached(Vec<f32>),
    Image(DynamicImage),
}

/// One embedding for each key, from the cache or from the image. An image that cannot be read
/// gets its own error. A model error fails the whole call, because the next images would fail
/// the same way.
fn embeddings(
    embedder: &ImageEmbedder,
    cache: &Cache,
    keys: &[&Key],
) -> Result<Vec<Result<Vec<f32>>>> {
    let view = cache.view();
    let inputs: Vec<Result<Input>> = keys
        .par_iter()
        .map(|key| match view.embedding(key) {
            Some(embedding) => Ok(Input::Cached(embedding)),
            None => decode::load(key.path(), EMBED_SIDE).map(Input::Image),
        })
        .collect();
    drop(view);
    let images: Vec<&DynamicImage> = (inputs.iter())
        .filter_map(|input| match input {
            Ok(Input::Image(image)) => Some(image),
            _ => None,
        })
        .collect();
    let mut computed = embedder.images(&images)?.into_iter();
    let embeddings = inputs.into_iter().zip(keys).map(|(input, key)| {
        input.map(|input| match input {
            Input::Cached(embedding) => embedding,
            Input::Image(_) => {
                let embedding = computed.next().expect("one embedding for each image");
                cache.put_embedding(key, &embedding);
                embedding
            }
        })
    });
    Ok(embeddings.collect())
}

struct Engine {
    cache: Arc<Cache>,
    filter: MetaFilter,
    ocr: Option<Ocr>,
    text: Vec<Regex>,
    show_text: bool,
    order: Order<Ranker>,
    progress: Arc<Progress>,
}

struct Cand {
    key: Key,
    id: FileId,
    hit: Hit,
    /// The date, when admission read the metadata.
    date: Option<DateTime>,
}

impl Engine {
    fn run(&self, paths: Receiver<PathBuf>, sink: &mut Sink) -> Result<()> {
        let mut seen = HashSet::new();
        match &self.order {
            Order::Found => {
                for chunk in ready_chunks(&paths, self.text_chunk()) {
                    let n = chunk.len();
                    let cands = self.read_text(self.admit(chunk, false, &mut seen));
                    self.progress.done.fetch_add(n, Relaxed);
                    if !sink.send(cands.into_iter().map(|c| c.hit)) {
                        break;
                    }
                }
            }
            Order::Score(ranker) => {
                let scored = self.gather(&paths, |chunk| {
                    self.rank(ranker, self.admit(chunk, false, &mut seen))
                })?;
                let mut best = best_first(scored);
                if ranker.embeddings {
                    self.embed_best(&mut best);
                }
                self.send_sorted(best, sink);
            }
            Order::Date(order) => {
                let mut cands =
                    self.gather(&paths, |chunk| Ok(self.admit(chunk, true, &mut seen)))?;
                cands.sort_by(|a, b| by_date(*order, a, b));
                self.send_sorted(cands, sink);
            }
        }
        Ok(())
    }

    /// OCR is slow, so its chunks only fill the pool once: the first hit comes sooner.
    fn text_chunk(&self) -> usize {
        match self.ocr {
            Some(_) => rayon::current_num_threads(),
            None => CHUNK,
        }
    }

    fn gather<T>(
        &self,
        paths: &Receiver<PathBuf>,
        mut stage: impl FnMut(Vec<PathBuf>) -> Result<Vec<T>>,
    ) -> Result<Vec<T>> {
        let mut all = Vec::new();
        for chunk in chunks(paths.iter(), CHUNK) {
            let n = chunk.len();
            all.extend(stage(chunk)?);
            self.progress.done.fetch_add(n, Relaxed);
        }
        Ok(all)
    }

    /// Sends sorted candidates in order, so a limit cuts OCR short.
    fn send_sorted(&self, cands: impl IntoIterator<Item = Cand>, sink: &mut Sink) {
        for chunk in chunks(cands.into_iter(), self.text_chunk()) {
            if !sink.send(self.read_text(chunk).into_iter().map(|c| c.hit)) {
                break;
            }
        }
    }

    /// `dated` reads the metadata for a date even without a metadata filter.
    fn admit(&self, paths: Vec<PathBuf>, dated: bool, seen: &mut HashSet<FileId>) -> Vec<Cand> {
        let view = self.cache.view();
        let mut cands: Vec<Cand> = paths
            .into_par_iter()
            .filter_map(|path| self.ok(self.admit_one(&view, path, dated)).flatten())
            .collect();
        // Overlapping paths find a file twice. The first path wins, so threads do not decide.
        cands.retain(|c| seen.insert(c.id));
        cands
    }

    fn admit_one(&self, view: &View, path: PathBuf, dated: bool) -> Result<Option<Cand>> {
        let md = fs::metadata(&path)?;
        decode::check(&md)?;
        let key = Key::new(&path, &md);
        let mut date = None;
        if dated || self.filter.needs_meta() {
            let meta = view.meta(&key, || meta::read(&path))?;
            let modified = md.modified()?;
            if !self.filter.check(&meta, modified) {
                return Ok(None);
            }
            date = meta.date(modified);
        }
        Ok(Some(Cand {
            key,
            id: (md.dev(), md.ino()),
            hit: Hit::new(path),
            date,
        }))
    }

    fn rank(&self, ranker: &Ranker, mut cands: Vec<Cand>) -> Result<Vec<(f32, Cand)>> {
        cands.retain(|c| !ranker.target.excludes(c.id));
        let keys: Vec<&Key> = cands.iter().map(|c| &c.key).collect();
        let embeddings = embeddings(&ranker.embedder, &self.cache, &keys)?;
        let scored = cands
            .into_iter()
            .zip(embeddings)
            .filter_map(|(c, embedding)| {
                let score = ranker.score(&self.ok(embedding)?)?;
                Some((score, c))
            });
        Ok(scored.collect())
    }

    /// Reads the embeddings back from the cache, so no other hit holds one until the sort. The
    /// embeddings of this run can still wait in the write queue.
    fn embed_best(&self, best: &mut [Cand]) {
        self.cache.flush();
        let view = self.cache.view();
        for c in best.iter_mut().take(EMBEDDED) {
            c.hit.embedding = view.embedding(&c.key).map(Arc::from);
        }
    }

    fn read_text(&self, cands: Vec<Cand>) -> Vec<Cand> {
        let Some(ocr) = &self.ocr else {
            return cands;
        };
        let view = self.cache.view();
        cands
            .into_par_iter()
            .filter_map(|mut c| {
                let text = self.ok(view.ocr(&c.key, || ocr.read(&c.hit.path)))?;
                if !self.text.iter().all(|re| re.is_match(&text)) {
                    return None;
                }
                if self.show_text {
                    c.hit.text = self.shown(text);
                }
                Some(c)
            })
            .collect()
    }

    /// All the text, or each match of each pattern. Empty matches show nothing, as in grep -o.
    fn shown(&self, text: String) -> Vec<String> {
        if self.text.is_empty() {
            return vec![text];
        }
        (self.text.iter())
            .flat_map(|re| re.find_iter(&text))
            .filter(|m| !m.is_empty())
            .map(|m| m.as_str().to_owned())
            .collect()
    }

    /// Like `Result::ok`, but counts the error.
    fn ok<T>(&self, result: Result<T>) -> Option<T> {
        result
            .inspect_err(|_| {
                self.progress.errors.fetch_add(1, Relaxed);
            })
            .ok()
    }
}

/// Equal scores go by path, so the output is the same in each run.
fn best_first(mut scored: Vec<(f32, Cand)>) -> Vec<Cand> {
    scored.sort_by(|(a, x), (b, y)| b.total_cmp(a).then_with(|| x.hit.path.cmp(&y.hit.path)));
    (scored.into_iter())
        .map(|(score, mut c)| {
            c.hit.score = Some(score);
            c
        })
        .collect()
}

fn by_date(order: DateOrder, a: &Cand, b: &Cand) -> Ordering {
    let oldest = a.date.cmp(&b.date);
    let first = match order {
        DateOrder::Oldest => oldest,
        DateOrder::Newest => oldest.reverse(),
    };
    first.then_with(|| a.hit.path.cmp(&b.hit.path))
}

struct Sink {
    tx: Sender<Result<Hit>>,
    left: Option<usize>,
}

impl Sink {
    fn new(tx: Sender<Result<Hit>>, limit: Option<NonZeroUsize>) -> Self {
        Self {
            tx,
            left: limit.map(NonZeroUsize::get),
        }
    }

    /// False once no more hits are wanted.
    fn send(&mut self, hits: impl IntoIterator<Item = Hit>) -> bool {
        for hit in hits {
            if self.left == Some(0) || self.tx.send(Ok(hit)).is_err() {
                return false;
            }
            self.left = self.left.map(|n| n - 1);
        }
        self.left != Some(0)
    }
}

fn chunks<T>(mut items: impl Iterator<Item = T>, size: usize) -> impl Iterator<Item = Vec<T>> {
    iter::from_fn(move || {
        Some(items.by_ref().take(size).collect::<Vec<_>>()).filter(|c| !c.is_empty())
    })
}

/// Waits only for the first item of each chunk, so items that trickle in, like paths from
/// `inotifywait`, go out at once.
fn ready_chunks<T>(rx: &Receiver<T>, size: usize) -> impl Iterator<Item = Vec<T>> {
    iter::from_fn(move || {
        let first = rx.recv().ok()?;
        Some(
            iter::once(first)
                .chain(rx.try_iter().take(size - 1))
                .collect(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Temp, encode};

    fn hit(name: &str) -> Hit {
        Hit::new(name.into())
    }

    #[test]
    fn negatives_drop_closer_images() {
        let target = Target::Text {
            query: vec![1.0, 0.0],
            not: vec![vec![0.0, 1.0]],
        };
        assert_eq!(target.score(&[0.8, 0.6]), Some(embed::score(0.8)));
        assert_eq!(target.score(&[0.6, 0.8]), None);
    }

    fn cand(name: &str) -> Cand {
        let md = fs::metadata(".").unwrap();
        Cand {
            key: Key::new(Path::new(name), &md),
            id: (0, 0),
            hit: hit(name),
            date: None,
        }
    }

    #[test]
    fn best_first_ties_by_path() {
        let scored = vec![(0.1, cand("d")), (0.9, cand("b")), (0.1, cand("c"))];
        let order: Vec<_> = (best_first(scored).into_iter())
            .map(|c| (c.hit.path, c.hit.score))
            .collect();
        let want = [("b", 0.9), ("c", 0.1), ("d", 0.1)].map(|(p, s)| (PathBuf::from(p), Some(s)));
        assert_eq!(order, want);
    }

    #[test]
    fn sink_stops_at_limit() {
        let (tx, rx) = mpsc::channel();
        let mut sink = Sink::new(tx, NonZeroUsize::new(3));
        assert!(sink.send([hit("a"), hit("b")]));
        assert!(!sink.send([hit("c"), hit("d")]));
        drop(sink);
        let got: Vec<PathBuf> = rx.iter().map(|h| h.unwrap().path).collect();
        assert_eq!(got, ["a", "b", "c"].map(PathBuf::from));
    }

    #[test]
    fn sink_without_limit_takes_all() {
        let (tx, rx) = mpsc::channel();
        let mut sink = Sink::new(tx, None);
        assert!(sink.send((0..100).map(|i| hit(&i.to_string()))));
        drop(sink);
        assert_eq!(rx.iter().count(), 100);
    }

    #[test]
    fn chunks_cover_all_items() {
        let got: Vec<Vec<i32>> = chunks(0..5, 2).collect();
        assert_eq!(got, [vec![0, 1], vec![2, 3], vec![4]]);
        assert_eq!(chunks(0..0, 2).count(), 0);
    }

    #[test]
    fn ready_chunks_do_not_wait_to_fill() {
        let (tx, rx) = mpsc::channel();
        tx.send(1).unwrap();
        let mut chunks = ready_chunks(&rx, 4);
        assert_eq!(chunks.next(), Some(vec![1]));
        (2..=7).for_each(|i| tx.send(i).unwrap());
        drop(tx);
        assert_eq!(chunks.collect::<Vec<_>>(), [vec![2, 3, 4, 5], vec![6, 7]]);
    }

    fn engine(order: Order<Ranker>, text: &[&str]) -> Engine {
        Engine {
            cache: Arc::new(Cache::disabled()),
            filter: MetaFilter::default(),
            ocr: None,
            text: text.iter().map(|t| Regex::new(t).unwrap()).collect(),
            show_text: true,
            order,
            progress: Arc::default(),
        }
    }

    #[test]
    fn overlapping_paths_give_each_file_once() {
        let dir = Temp::dir("search");
        let png = encode(&DynamicImage::new_rgb8(2, 2), image::ImageFormat::Png);
        for name in ["a.png", "b.png"] {
            fs::write(dir.join(name), &png).unwrap();
        }
        let (tx, rx) = mpsc::channel();
        for path in [dir.join("a.png"), dir.join("b.png"), dir.join("./a.png")] {
            tx.send(path).unwrap();
        }
        drop(tx);
        let (hits, got) = mpsc::channel();
        let mut sink = Sink::new(hits, None);
        let engine = engine(Order::Date(DateOrder::Oldest), &[]);
        engine.run(rx, &mut sink).unwrap();
        drop(sink);
        let names: Vec<PathBuf> = got.iter().map(|h| h.unwrap().path).collect();
        assert_eq!(names, [dir.join("a.png"), dir.join("b.png")]);
    }

    #[test]
    fn empty_matches_show_nothing() {
        let engine = engine(Order::Found, &["x*", "b+"]);
        assert_eq!(engine.shown("abba".into()), ["bb"]);
    }
}
