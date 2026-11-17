use std::{
    cmp::Ordering,
    collections::HashSet,
    fmt,
    fs::{self, Metadata},
    io, iter,
    num::NonZeroUsize,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering::Relaxed},
        mpsc::{self, Receiver, Sender},
    },
    thread,
    time::SystemTime,
};

use anyhow::{Context, Result};
use image::DynamicImage;
use jiff::civil::DateTime;
use rayon::prelude::*;
use regex::Regex;
use serde::Serialize;

use crate::{
    cache::{Cache, Key, View},
    catch, decode,
    embed::{self, ImageEmbedder, Prompt, TextEmbedder},
    meta::{self, Meta, MetaFilter},
    ocr::Ocr,
    output::clean_path,
    pick, report,
    walk::{self, Found, Skipped, Source, WalkOpts, Watch},
};

pub struct Query {
    pub source: Source,
    pub walk: WalkOpts,
    pub meta: MetaFilter,
    /// All must match the text in the image.
    pub text: Vec<Regex>,
    /// Hits carry all the text in the image.
    pub show_text: bool,
    /// Hits carry what the file tells about the image.
    pub describe: bool,
    pub order: Order,
    pub limit: Option<NonZeroUsize>,
    pub watch: Option<Watch>,
}

/// A sorted order sends the first hit only after it has seen all images. `S` is what a score
/// needs at each stage: the query, the loaded model, or nothing.
pub enum Order<S = Scoring> {
    /// The walk order, or the order of the paths on stdin.
    Found,
    Sorted(Field, Dir),
    /// The best images by score, sorted again by `By`. The limit or the minimum score chooses
    /// the set.
    Ranked(S, By, Dir),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Field {
    Date,
    Path,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum By {
    Score,
    Field(Field),
}

/// Each key has its natural direction, as in `ls`: score best first, date newest first, path
/// A to Z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dir {
    Natural,
    Reverse,
}

pub struct Scoring {
    pub rank: Rank,
    pub min: Option<f32>,
    /// Hits carry their image embedding, so a picker can rank them again and spread out
    /// look-alikes.
    pub embeddings: bool,
}

pub enum Rank {
    Text { query: Prompt, not: Vec<Prompt> },
    Like(PathBuf),
}

impl Rank {
    /// The option that gives the rank, for messages.
    pub fn flag(&self) -> &'static str {
        match self {
            Rank::Text { .. } => "-q",
            Rank::Like(_) => "-l",
        }
    }
}

#[derive(Clone)]
pub struct Hit {
    pub path: PathBuf,
    pub score: Option<f32>,
    pub embedding: Option<Arc<[f32]>>,
    /// With `show_text`.
    pub text: Option<String>,
    /// With text patterns: each match of each one. Empty matches are left out, as in grep -o.
    pub matches: Option<Vec<String>>,
    /// With `describe`.
    pub file: Option<FileInfo>,
    /// --watch found it after the first pass.
    pub live: bool,
}

impl Hit {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            score: None,
            embedding: None,
            text: None,
            matches: None,
            file: None,
            live: false,
        }
    }
}

#[derive(Clone)]
pub struct FileInfo {
    /// None if scry cannot read the header.
    pub meta: Option<Meta>,
    pub modified: SystemTime,
}

impl FileInfo {
    /// The date that --date and --sort date use.
    pub fn date(&self) -> Option<DateTime> {
        meta::date_of(self.meta.as_ref(), self.modified)
    }
}

/// `done` counts the images of the current stage.
#[derive(Default)]
pub struct Progress {
    found: AtomicUsize,
    done: AtomicUsize,
    unsupported: AtomicUsize,
    pub unreadable: Unreadable,
    stage: AtomicU8,
    /// The candidates of the text stage after a sort.
    listed: AtomicUsize,
    walked: AtomicBool,
    truncated: AtomicBool,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum Stage {
    Check,
    Load,
    Score,
    Text,
    /// --watch, after the first pass.
    Watch,
}

/// How far a search is, at one moment.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Snapshot {
    pub stage: Stage,
    pub done: usize,
    pub total: usize,
    /// The walk still finds images, so `total` can grow.
    pub walking: bool,
}

impl fmt::Display for Snapshot {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let verb = match self.stage {
            Stage::Load => return f.write_str("loading the models"),
            Stage::Check => "checked",
            Stage::Score => "scored",
            Stage::Text => "read",
            Stage::Watch => return write!(f, "watching, {} new", self.total),
        };
        write!(f, "{}/{} {verb}", self.done, self.total)
    }
}

impl Progress {
    pub fn found(&self) -> usize {
        self.found.load(Relaxed)
    }

    pub fn unsupported(&self) -> usize {
        self.unsupported.load(Relaxed)
    }

    /// The limit stopped the search before it sent or checked all images.
    pub fn truncated(&self) -> bool {
        self.truncated.load(Relaxed)
    }

    pub fn snapshot(&self) -> Snapshot {
        let stage = match self.stage.load(Relaxed) {
            1 => Stage::Load,
            2 => Stage::Score,
            3 => Stage::Text,
            4 => Stage::Watch,
            _ => Stage::Check,
        };
        let listed = self.listed.load(Relaxed);
        let total = match stage {
            Stage::Text => listed,
            Stage::Watch => self.done.load(Relaxed),
            _ => self.found(),
        };
        Snapshot {
            stage,
            done: self.done.load(Relaxed),
            total,
            walking: !self.walked.load(Relaxed),
        }
    }

    pub fn live(&self) -> bool {
        self.stage.load(Relaxed) == Stage::Watch as u8
    }

    fn fail(&self, path: Option<&Path>, reason: &dyn fmt::Display) {
        if self.tells() {
            report::unreadable(path, &format!("{reason:#}"));
        }
        self.unreadable.add(path, reason);
    }

    /// An image in a format that scry cannot read.
    fn skip(&self, path: &Path) {
        if self.tells() {
            let message = format!(
                "skipped {}: scry cannot read this format yet",
                clean_path(path)
            );
            report::warn(report::Code::Unsupported, &message);
        }
        self.unsupported.fetch_add(1, Relaxed);
    }

    /// A --watch run tells at once, because it ends only when it is stopped. The picker owns
    /// the terminal, so it gets the notes at the end.
    fn tells(&self) -> bool {
        self.live() && !pick::active()
    }

    /// `listed` counts the candidates of the text stage.
    fn enter(&self, stage: Stage, listed: usize) {
        self.listed.store(listed, Relaxed);
        self.done.store(0, Relaxed);
        self.stage.store(stage as u8, Relaxed);
    }
}

/// Files that scry cannot read. The first few keep the path and the reason, for the notes at the
/// end.
#[derive(Default)]
pub struct Unreadable {
    count: AtomicUsize,
    first: Mutex<Vec<Failure>>,
}

pub struct Failure {
    /// None when the error does not name a file, as a failed read of stdin.
    pub path: Option<PathBuf>,
    pub reason: String,
}

impl Unreadable {
    const KEPT: usize = 5;

    pub fn add(&self, path: Option<&Path>, reason: impl fmt::Display) {
        if self.count.fetch_add(1, Relaxed) < Self::KEPT {
            self.first.lock().unwrap().push(Failure {
                path: path.map(Path::to_path_buf),
                reason: format!("{reason:#}"),
            });
        }
    }

    pub fn count(&self) -> usize {
        self.count.load(Relaxed)
    }

    pub fn first(&self) -> MutexGuard<'_, Vec<Failure>> {
        self.first.lock().unwrap()
    }
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
    /// Embeds images on a background thread, for a picker that ranks its hits again. Each answer
    /// carries the number that came with its path, and `None` for an image that cannot be read.
    /// An error ends the thread: the next images would fail the same way.
    pub fn embed(&self, embedder: ImageEmbedder) -> (Sender<(usize, PathBuf)>, Receiver<Embedded>) {
        let (tx, paths) = mpsc::channel::<(usize, PathBuf)>();
        let (done, rx) = mpsc::channel();
        let (cache, progress) = (self.cache.clone(), self.progress.clone());
        thread::spawn(move || {
            // False once nobody waits for the answers.
            let answer = |i, path: &Path, embedding: Result<Vec<f32>>| {
                let embedding = embedding
                    .inspect_err(|e| progress.unreadable.add(Some(path), e))
                    .ok();
                done.send(Ok((i, embedding.map(Arc::from)))).is_ok()
            };
            for chunk in ready_chunks(&paths, CHUNK) {
                let mut keys = Vec::new();
                for (i, path) in chunk {
                    match fs::metadata(&path) {
                        Ok(md) => keys.push((i, Key::new(&path, &md))),
                        Err(e) => _ = answer(i, &path, Err(e.into())),
                    }
                }
                let readable: Vec<&Key> = keys.iter().map(|(_, key)| key).collect();
                // Only the work runs on rten's pool, not the wait for paths. A worker that waits
                // in a join runs other jobs, and a job that waits for the picker would stall the
                // search on that worker.
                let embedded = rten::thread_pool().run(|| embeddings(&embedder, &cache, &readable));
                let embedded = match embedded {
                    Ok(embedded) => embedded,
                    Err(e) => {
                        let _ = done.send(Err(e));
                        return;
                    }
                };
                for ((i, key), embedding) in keys.iter().zip(embedded) {
                    if !answer(*i, key.path(), embedding) {
                        return;
                    }
                }
            }
        });
        (tx, rx)
    }
}

/// An answer of `Search::embed`.
pub type Embedded = Result<(usize, Option<Arc<[f32]>>)>;

#[cfg(test)]
impl Search {
    /// A search that found nothing.
    pub fn empty() -> Self {
        Self {
            hits: mpsc::channel().1,
            progress: Arc::default(),
            order: Order::Found,
            skipped: None,
            cache: Arc::new(Cache::disabled()),
        }
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

/// Device and inode.
type FileId = (u64, u64);

/// A file and the time it was made, which tells apart a file that reuses a deleted inode.
type Born = (FileId, Option<SystemTime>);

pub fn run(query: Query) -> Result<Search> {
    let progress = Arc::new(Progress::default());
    // The walk starts first, so it runs while the models load.
    let walk = walk::candidates(query.source, query.walk, query.watch)?;
    let paths = counted(walk.found, progress.clone(), false);
    let live = (walk.live).map(|live| counted(live, progress.clone(), true));
    let cache = Arc::new(Cache::open()?);
    crate::stop_on_signals(cache.clone())?;
    let reads_text = query.show_text || !query.text.is_empty();
    if reads_text || matches!(query.order, Order::Ranked(..)) {
        progress.enter(Stage::Load, 0);
        report::progress(&progress);
    }
    let (order, sort) = match query.order {
        Order::Found => (Order::Found, Order::Found),
        Order::Sorted(field, dir) => (Order::Sorted(field, dir), Order::Sorted(field, dir)),
        Order::Ranked(scoring, by, dir) => (
            Order::Ranked((), by, dir),
            Order::Ranked(Ranker::new(scoring, &cache)?, by, dir),
        ),
    };
    let ocr = reads_text.then(Ocr::load).transpose()?;
    let (tx, hits) = mpsc::channel();
    let mut sink = Sink::new(tx, query.limit);
    let engine = Engine {
        cache: cache.clone(),
        filter: query.meta,
        ocr,
        text: query.text,
        show_text: query.show_text,
        describe: query.describe,
        order: sort,
        progress: progress.clone(),
        emitted: live.as_ref().map(|_| Mutex::default()),
    };
    thread::spawn(move || {
        // Decoding runs on rten's pool too, so it never competes with inference for cores.
        let run = catch(|| rten::thread_pool().run(|| engine.run(paths, live, &mut sink)));
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

/// `live`: the images of --watch after the walk.
fn counted(found: Receiver<Found>, progress: Arc<Progress>, live: bool) -> Receiver<PathBuf> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for item in found {
            let path = match item {
                Found::Image(path) => path,
                Found::Unsupported(path) => {
                    progress.skip(&path);
                    continue;
                }
                Found::Error { path, reason } => {
                    progress.fail(path.as_deref(), &reason);
                    continue;
                }
            };
            progress.found.fetch_add(1, Relaxed);
            if tx.send(path).is_err() {
                break;
            }
        }
        if !live {
            progress.walked.store(true, Relaxed);
        }
    });
    rx
}

struct Ranker {
    embedder: ImageEmbedder,
    target: Target,
    /// The --like image, which does not rank against itself.
    exclude: Option<FileId>,
    min: Option<f32>,
    embeddings: bool,
}

/// What a score compares each image embedding with.
pub enum Target {
    Text { query: Vec<f32>, not: Vec<Vec<f32>> },
    Like(Vec<f32>),
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
        let (target, exclude) = match rank {
            Rank::Text { query, not } => {
                // Drops at the end of this block, with the 300 MB of the text model.
                let model = TextEmbedder::default();
                let embed = |prompt: &Prompt| cache.text_embedding(prompt, || model.embed(prompt));
                let target = Target::Text {
                    query: embed(&query)?,
                    not: not.iter().map(embed).collect::<Result<_>>()?,
                };
                (target, None)
            }
            Rank::Like(path) => {
                let (image, id) = reference(&embedder, cache, &path)
                    .with_context(|| format!("--like {}", clean_path(&path)))?;
                (Target::Like(image), Some(id))
            }
        };
        Ok(Self {
            embedder,
            target,
            exclude,
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
    pub fn score(&self, image: &[f32]) -> Option<f32> {
        match self {
            Target::Text { query, not } => {
                let cos = embed::cos(image, query);
                not.iter()
                    .all(|n| embed::cos(image, n) <= cos)
                    .then(|| embed::score(cos))
            }
            Target::Like(reference) => Some(embed::cos(image, reference)),
        }
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
    describe: bool,
    order: Order<Ranker>,
    progress: Arc<Progress>,
    /// With --watch: the files that went out, so a change does not send them twice.
    emitted: Option<Mutex<HashSet<Born>>>,
}

struct Cand {
    key: Key,
    id: FileId,
    born: Option<SystemTime>,
    hit: Hit,
    modified: SystemTime,
    /// When admission read it.
    meta: Option<Meta>,
}

impl Cand {
    fn date(&self) -> Option<DateTime> {
        meta::date_of(self.meta.as_ref(), self.modified)
    }
}

impl Engine {
    fn run(
        &self,
        paths: Receiver<PathBuf>,
        live: Option<Receiver<PathBuf>>,
        sink: &mut Sink,
    ) -> Result<()> {
        let mut seen = HashSet::new();
        self.progress.enter(Stage::Check, 0);
        match &self.order {
            Order::Found => {
                for chunk in ready_chunks(&paths, self.text_chunk()) {
                    let n = chunk.len();
                    let cands = self.read_text(self.admit(chunk, false, Some(&mut seen)));
                    self.progress.done.fetch_add(n, Relaxed);
                    if !self.emit(cands, sink) {
                        // An image that waits in the walk is not checked. A wait for the next one
                        // would hold up the exit when stdin sends paths slowly.
                        let full = sink.left == Some(0);
                        let left = sink.refused || paths.try_recv().is_ok();
                        self.stopped(sink, full && left);
                        break;
                    }
                }
            }
            Order::Sorted(field, dir) => {
                let dated = *field == Field::Date;
                let mut cands = self.gather(&paths, |chunk| {
                    Ok(self.admit(chunk, dated, Some(&mut seen)))
                })?;
                cands.sort_by(|a, b| compare(By::Field(*field), *dir, a, b));
                self.send_sorted(cands, sink);
            }
            Order::Ranked(ranker, by, dir) => {
                self.progress.enter(Stage::Score, 0);
                let dated = *by == By::Field(Field::Date);
                let scored = self.gather(&paths, |chunk| {
                    self.rank(ranker, self.admit(chunk, dated, Some(&mut seen)))
                })?;
                let best = best_first(scored);
                if (*by, *dir) == (By::Score, Dir::Natural) {
                    self.send_sorted(best, sink);
                } else {
                    let (mut set, left) = self.take(best, sink.left);
                    set.sort_by(|a, b| compare(*by, *dir, a, b));
                    self.emit(set, sink);
                    self.stopped(sink, left);
                }
            }
        }
        match live {
            Some(live) if sink.left != Some(0) => self.stream(&live, sink),
            _ => Ok(()),
        }
    }

    /// --watch: each new image as it comes, in that order.
    fn stream(&self, live: &Receiver<PathBuf>, sink: &mut Sink) -> Result<()> {
        self.progress.enter(Stage::Watch, 0);
        // A new image waits in the cache's memory until the next commit.
        self.cache.flush();
        let size = self.text_chunk();
        for chunk in ready_chunks(live, size) {
            let n = chunk.len();
            let mut cands = self.admit(chunk, false, None);
            if let Some(emitted) = &self.emitted {
                // Events can send a path twice, also in one chunk.
                let emitted = emitted.lock().unwrap();
                let mut chunk = HashSet::new();
                cands.retain(|c| {
                    let file = (c.id, c.born);
                    !emitted.contains(&file) && chunk.insert(file)
                });
            }
            self.progress.done.fetch_add(cands.len(), Relaxed);
            if let Order::Ranked(ranker, ..) = &self.order {
                cands = (self.rank(ranker, cands)?.into_iter())
                    .map(|(score, mut c)| {
                        c.hit.score = Some(score);
                        c
                    })
                    .collect();
            }
            let mut passed = self.read_text(cands);
            passed.iter_mut().for_each(|c| c.hit.live = true);
            if !self.emit(passed, sink) {
                break;
            }
            if n < size {
                self.cache.flush();
            }
        }
        Ok(())
    }

    fn emit(&self, mut cands: Vec<Cand>, sink: &mut Sink) -> bool {
        // The limit refuses the rest, so they need no metadata.
        if let Some(left) = sink.left.filter(|&left| cands.len() > left) {
            cands.truncate(left);
            sink.refused = true;
        }
        self.describe(&mut cands);
        if let Some(emitted) = &self.emitted {
            emitted
                .lock()
                .unwrap()
                .extend(cands.iter().map(|c| (c.id, c.born)));
        }
        sink.send(cands.into_iter().map(|c| c.hit))
    }

    /// `left`: candidates were left when the search stopped.
    fn stopped(&self, sink: &Sink, left: bool) {
        if sink.left == Some(0) && (left || sink.refused) {
            self.progress.truncated.store(true, Relaxed);
        }
    }

    fn describe(&self, cands: &mut [Cand]) {
        if !self.describe {
            return;
        }
        let view = self.cache.view();
        cands.par_iter_mut().for_each(|c| {
            let meta =
                (c.meta.take()).or_else(|| view.meta(&c.key, || meta::read(&c.hit.path)).ok());
            c.hit.file = Some(FileInfo {
                meta,
                modified: c.modified,
            });
        });
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

    /// The first `limit` candidates that pass the text filters, and whether others were left.
    /// OCR stops there.
    fn take(&self, cands: Vec<Cand>, limit: Option<usize>) -> (Vec<Cand>, bool) {
        self.enter_text(cands.len());
        let mut set = Vec::new();
        let mut chunks = chunks(cands.into_iter(), self.text_chunk()).peekable();
        while let Some(chunk) = chunks.next() {
            set.extend(self.read_listed(chunk));
            if let Some(n) = limit.filter(|&n| set.len() >= n) {
                let left = set.len() > n || chunks.peek().is_some();
                set.truncate(n);
                return (set, left);
            }
        }
        (set, false)
    }

    /// Sends sorted candidates in order, so a limit cuts OCR short.
    fn send_sorted(&self, cands: Vec<Cand>, sink: &mut Sink) {
        self.enter_text(cands.len());
        let mut chunks = chunks(cands.into_iter(), self.text_chunk()).peekable();
        while let Some(chunk) = chunks.next() {
            let passed = self.read_listed(chunk);
            if !self.emit(passed, sink) {
                self.stopped(sink, chunks.peek().is_some());
                break;
            }
        }
    }

    /// A sorted list goes through OCR after the sort, as a stage of its own.
    fn enter_text(&self, listed: usize) {
        if self.ocr.is_some() {
            self.progress.enter(Stage::Text, listed);
        }
    }

    fn read_listed(&self, chunk: Vec<Cand>) -> Vec<Cand> {
        let n = chunk.len();
        let passed = self.read_text(chunk);
        if self.ocr.is_some() {
            self.progress.done.fetch_add(n, Relaxed);
        }
        passed
    }

    /// `dated` reads the metadata for a date even without a metadata filter.
    /// `seen` drops the files that came before: overlapping paths find a file twice. It comes
    /// before any check that can fail, so a failure counts once, and the first path wins. None
    /// is the live phase of --watch, where a file that is gone or empty is not ready yet: a
    /// writer that fills it sends it again.
    fn admit(
        &self,
        paths: Vec<PathBuf>,
        dated: bool,
        seen: Option<&mut HashSet<FileId>>,
    ) -> Vec<Cand> {
        let live = seen.is_none();
        let mut files: Vec<(PathBuf, Metadata)> = (paths.into_par_iter())
            .filter_map(|path| match fs::metadata(&path) {
                Ok(md) if live && md.len() == 0 => None,
                Err(e) if live && e.kind() == io::ErrorKind::NotFound => None,
                md => {
                    let md = self.ok(&path, md.map_err(Into::into))?;
                    Some((path, md))
                }
            })
            .collect();
        if let Some(seen) = seen {
            let found = files.len();
            files.retain(|(_, md)| seen.insert((md.dev(), md.ino())));
            self.progress.found.fetch_sub(found - files.len(), Relaxed);
        }
        let view = self.cache.view();
        (files.into_par_iter())
            .filter_map(|(path, md)| {
                self.ok(&path, self.admit_one(&view, &path, &md, dated))
                    .flatten()
            })
            .collect()
    }

    fn admit_one(
        &self,
        view: &View,
        path: &Path,
        md: &Metadata,
        dated: bool,
    ) -> Result<Option<Cand>> {
        decode::check(md)?;
        let key = Key::new(path, md);
        let modified = md.modified()?;
        let filtered = self.filter.needs_meta();
        let meta = match (dated || filtered).then(|| view.meta(&key, || meta::read(path))) {
            None => None,
            Some(Ok(read)) => Some(read),
            // A sort by date takes the time the file changed, as the output does.
            Some(Err(_)) if !filtered => None,
            Some(Err(e)) => return Err(e),
        };
        if let Some(read) = &meta
            && !self.filter.check(read, modified)
        {
            return Ok(None);
        }
        Ok(Some(Cand {
            key,
            id: (md.dev(), md.ino()),
            born: md.created().ok(),
            hit: Hit::new(path.to_path_buf()),
            modified,
            meta,
        }))
    }

    fn rank(&self, ranker: &Ranker, mut cands: Vec<Cand>) -> Result<Vec<(f32, Cand)>> {
        cands.retain(|c| ranker.exclude != Some(c.id));
        let keys: Vec<&Key> = cands.iter().map(|c| &c.key).collect();
        let embeddings = embeddings(&ranker.embedder, &self.cache, &keys)?;
        let scored = cands
            .into_iter()
            .zip(embeddings)
            .filter_map(|(mut c, embedding)| {
                let embedding = self.ok(&c.hit.path, embedding)?;
                let score = ranker.score(&embedding)?;
                if ranker.embeddings {
                    c.hit.embedding = Some(embedding.into());
                }
                Some((score, c))
            });
        Ok(scored.collect())
    }

    fn read_text(&self, cands: Vec<Cand>) -> Vec<Cand> {
        let Some(ocr) = &self.ocr else {
            return cands;
        };
        let view = self.cache.view();
        cands
            .into_par_iter()
            .filter_map(|mut c| {
                let text = self.ok(&c.hit.path, view.ocr(&c.key, || ocr.read(&c.hit.path)))?;
                if !self.text.iter().all(|re| re.is_match(&text)) {
                    return None;
                }
                c.hit.matches = (!self.text.is_empty()).then(|| self.matches(&text));
                if self.show_text {
                    c.hit.text = Some(text);
                }
                Some(c)
            })
            .collect()
    }

    fn matches(&self, text: &str) -> Vec<String> {
        (self.text.iter())
            .flat_map(|re| re.find_iter(text))
            .filter(|m| !m.is_empty())
            .map(|m| m.as_str().to_owned())
            .collect()
    }

    /// Like `Result::ok`, but records the error.
    fn ok<T>(&self, path: &Path, result: Result<T>) -> Option<T> {
        result
            .inspect_err(|e| self.progress.fail(Some(path), e))
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

fn compare(by: By, dir: Dir, a: &Cand, b: &Cand) -> Ordering {
    let score = |c: &Cand| c.hit.score.unwrap_or(f32::NEG_INFINITY);
    let natural = match by {
        By::Score => score(b).total_cmp(&score(a)),
        By::Field(Field::Date) => b.date().cmp(&a.date()),
        By::Field(Field::Path) => a.hit.path.cmp(&b.hit.path),
    };
    let order = match dir {
        Dir::Natural => natural,
        Dir::Reverse => natural.reverse(),
    };
    order.then_with(|| a.hit.path.cmp(&b.hit.path))
}

struct Sink {
    tx: Sender<Result<Hit>>,
    left: Option<usize>,
    /// The limit refused a hit.
    refused: bool,
}

impl Sink {
    fn new(tx: Sender<Result<Hit>>, limit: Option<NonZeroUsize>) -> Self {
        Self {
            tx,
            left: limit.map(NonZeroUsize::get),
            refused: false,
        }
    }

    /// False once no more hits are wanted.
    fn send(&mut self, hits: impl IntoIterator<Item = Hit>) -> bool {
        for hit in hits {
            if self.left == Some(0) {
                self.refused = true;
                return false;
            }
            if self.tx.send(Ok(hit)).is_err() {
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
            born: None,
            hit: hit(name),
            modified: SystemTime::UNIX_EPOCH,
            meta: None,
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
            describe: false,
            order,
            progress: Arc::default(),
            emitted: None,
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
        let engine = engine(Order::Sorted(Field::Path, Dir::Natural), &[]);
        engine.run(rx, None, &mut sink).unwrap();
        drop(sink);
        let names: Vec<PathBuf> = got.iter().map(|h| h.unwrap().path).collect();
        assert_eq!(names, [dir.join("a.png"), dir.join("b.png")]);
    }

    #[test]
    fn empty_matches_show_nothing() {
        let engine = engine(Order::Found, &["x*", "b+"]);
        assert_eq!(engine.matches("abba"), ["bb"]);
    }
}
