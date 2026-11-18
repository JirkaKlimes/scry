use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, Sender},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use ratatui::{
    crossterm::event::{KeyCode, KeyEvent, KeyModifiers},
    style::Stylize,
    text::Span,
};

use crate::{
    embed::{ImageEmbedder, Prompt, TextEmbedder},
    guard,
    output::{clean, clean_path},
    search::{Embedded, Hit, Order, Search, Target},
};

/// While you type, new embeddings join the ranking at most this often, so the images do not
/// move all the time.
const REFRESH: Duration = Duration::from_millis(500);

const NO_MODEL: &str = "this needs the SigLIP 2 model: run a scry -q search once to download it";

/// What the hits are ranked by.
#[derive(Clone, PartialEq)]
enum Probe {
    Text(String),
    Like(PathBuf),
}

enum Model {
    Unloaded,
    Loading(Receiver<Result<Box<TextEmbedder>>>),
    Ready(Box<TextEmbedder>),
}

/// Embeds the hits that came without an embedding.
struct Images {
    paths: Sender<(usize, PathBuf)>,
    done: Receiver<Embedded>,
    /// Sent and not answered yet.
    pending: usize,
}

impl Images {
    fn send(&mut self, i: usize, path: &Path) {
        if self.paths.send((i, path.to_owned())).is_ok() {
            self.pending += 1;
        }
    }
}

/// Ranks the hits again inside a picker, by a text that you type or by an image. The search
/// still chooses the hits: a ranking only changes their order.
pub struct Ranking {
    /// All hits, with the scores that the search gave them, in the order they came. An index
    /// names a hit for good.
    hits: Vec<Hit>,
    /// The indexes of `hits` in the order of the search. New images of --watch come after the
    /// sort, so each hit goes to its place.
    order: Vec<usize>,
    by_path: HashMap<PathBuf, usize>,
    /// The probes that you gave, the newest last. Backspace goes back one.
    history: Vec<Probe>,
    /// The query line, while you type.
    typing: Option<String>,
    text: Model,
    images: Option<Images>,
    /// The board must show a new order.
    stale: bool,
    /// Embeddings came since the last ranking.
    fresh: bool,
    ranked: Instant,
}

impl Ranking {
    pub fn new() -> Self {
        Self {
            hits: Vec::new(),
            order: Vec::new(),
            by_path: HashMap::new(),
            history: Vec::new(),
            typing: None,
            text: Model::Unloaded,
            images: None,
            stale: false,
            fresh: false,
            ranked: Instant::now(),
        }
    }

    /// Takes a new hit from the search, in `order`. Returns its place on the board: its place
    /// in the search order, or last on a board that a probe ranks, until the next ranking.
    pub fn add(&mut self, hit: Hit, order: &Order<()>) -> usize {
        let i = self.hits.len();
        if hit.embedding.is_none()
            && let Some(images) = &mut self.images
        {
            images.send(i, &hit.path);
        }
        let at = (self.order).partition_point(|&j| order.before(&self.hits[j], &hit));
        self.order.insert(at, i);
        self.by_path.insert(hit.path.clone(), i);
        self.hits.push(hit);
        if self.probing() {
            self.fresh = true;
            i
        } else {
            at
        }
    }

    /// The hits at `paths`, as the search gave them.
    pub fn hits(&self, paths: &[PathBuf]) -> Vec<Hit> {
        (paths.iter())
            .map(|path| self.hits[self.by_path[path]].clone())
            .collect()
    }

    pub fn typing(&self) -> bool {
        self.typing.is_some()
    }

    /// True when the hits are in an order of their own, not in the order of the search.
    pub fn probing(&self) -> bool {
        self.probe().is_some()
    }

    /// True when only a key can change the ranking.
    pub fn idle(&self) -> bool {
        !self.stale && !matches!(self.text, Model::Loading(_)) && self.pending() == 0
    }

    /// Opens the query line with the text query in use.
    pub fn open(&mut self, search: &Search) -> Result<()> {
        ensure!(TextEmbedder::is_downloaded(), NO_MODEL);
        self.embed_missing(search)?;
        if let Model::Unloaded = self.text {
            let (tx, rx) = mpsc::channel();
            thread::spawn(move || tx.send(guard(TextEmbedder::load).map(Box::new)));
            self.text = Model::Loading(rx);
        }
        self.typing = Some(match self.history.last() {
            Some(Probe::Text(text)) => text.clone(),
            _ => String::new(),
        });
        Ok(())
    }

    /// Ranks by how similar the images look to the one at `path`.
    pub fn like(&mut self, path: &Path, search: &Search) -> Result<()> {
        self.embed_missing(search)?;
        self.push(Probe::Like(path.to_owned()));
        self.stale = true;
        Ok(())
    }

    /// Goes back to the ranking before.
    pub fn back(&mut self) {
        self.stale |= self.history.pop().is_some();
    }

    /// A key while the query line is open. Each change ranks again.
    pub fn key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter => match self.typing.take() {
                // The board already shows this ranking.
                Some(text) if !text.trim().is_empty() => self.push(Probe::Text(text)),
                _ => self.stale = true,
            },
            KeyCode::Esc => {
                self.typing = None;
                self.stale = true;
            }
            code => {
                let Some(text) = &mut self.typing else {
                    return;
                };
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                match code {
                    KeyCode::Backspace => _ = text.pop(),
                    KeyCode::Char('u') if ctrl => text.clear(),
                    KeyCode::Char(c) if !ctrl => text.push(c),
                    _ => return,
                }
                self.stale = true;
            }
        }
    }

    /// The hits in a new order, when the board must show one.
    pub fn poll(&mut self) -> Result<Option<Vec<Hit>>> {
        self.receive()?;
        let refresh = self.fresh && self.typing.is_some() && self.ranked.elapsed() >= REFRESH;
        if !self.stale && !refresh {
            return Ok(None);
        }
        let target = match self.probe() {
            None => None,
            Some(probe) => match self.target(&probe) {
                Ok(Some(target)) => Some(target),
                Ok(None) => return Ok(None),
                Err(e) => {
                    // The next poll shows the ranking before.
                    self.typing = None;
                    self.history.retain(|p| *p != probe);
                    return Err(e);
                }
            },
        };
        self.stale = false;
        self.fresh = false;
        self.ranked = Instant::now();
        let hits = self.order.iter().map(|&i| &self.hits[i]);
        Ok(Some(match target {
            Some(target) => rank(hits, &target),
            // A board in a new order takes each image as one of the others.
            None => hits
                .map(|hit| Hit {
                    live: false,
                    ..hit.clone()
                })
                .collect(),
        }))
    }

    /// Facts for the status line.
    pub fn status(&self) -> Vec<Span<'static>> {
        let mut spans = Vec::new();
        match (&self.typing, self.history.last()) {
            (Some(text), _) => spans.push(format!("/{}▏ ", clean(text)).yellow().bold()),
            (None, Some(Probe::Text(text))) => spans.push(format!("/{}  ", clean(text)).yellow()),
            (None, Some(Probe::Like(path))) => {
                let name = path.file_name().map_or(path.as_path(), Path::new);
                spans.push(format!("*{}  ", clean_path(name)).yellow());
            }
            (None, None) => {}
        }
        if self.typing.is_some() && matches!(self.text, Model::Loading(_)) {
            spans.push("loading the text model  ".dim());
        }
        if self.pending() > 0 {
            spans.push(format!("{} not ranked yet  ", self.pending()).dim());
        }
        spans
    }

    fn probe(&self) -> Option<Probe> {
        match &self.typing {
            Some(text) if !text.trim().is_empty() => Some(Probe::Text(text.clone())),
            _ => self.history.last().cloned(),
        }
    }

    /// The target of `probe`, or `None` while its embedding is not ready.
    fn target(&self, probe: &Probe) -> Result<Option<Target>> {
        Ok(match probe {
            Probe::Text(text) => match &self.text {
                Model::Ready(model) => Some(Target::Text {
                    query: model.embed(&Prompt::new(text))?,
                    not: Vec::new(),
                }),
                _ => None,
            },
            Probe::Like(path) => match self.embedding(path) {
                Some(image) => Some(Target::Like(image)),
                None if self.pending() > 0 => None,
                None => bail!("cannot read {}", clean_path(path)),
            },
        })
    }

    fn push(&mut self, probe: Probe) {
        if self.history.last() != Some(&probe) {
            self.history.push(probe);
        }
    }

    fn embedding(&self, path: &Path) -> Option<Vec<f32>> {
        let hit = &self.hits[*self.by_path.get(path)?];
        hit.embedding.as_deref().map(<[f32]>::to_vec)
    }

    fn pending(&self) -> usize {
        self.images.as_ref().map_or(0, |images| images.pending)
    }

    /// Starts to embed the hits that came without an embedding, so that a ranking can take them
    /// in.
    fn embed_missing(&mut self, search: &Search) -> Result<()> {
        if self.images.is_some() || self.hits.iter().all(|h| h.embedding.is_some()) {
            return Ok(());
        }
        ensure!(ImageEmbedder::is_downloaded(), NO_MODEL);
        let (paths, done) = search.embed(ImageEmbedder::load()?);
        let mut images = Images {
            paths,
            done,
            pending: 0,
        };
        for (i, hit) in self.hits.iter().enumerate() {
            if hit.embedding.is_none() {
                images.send(i, &hit.path);
            }
        }
        self.images = Some(images);
        Ok(())
    }

    /// Takes the text model and the embeddings that are ready.
    fn receive(&mut self) -> Result<()> {
        if let Model::Loading(rx) = &self.text
            && let Ok(loaded) = rx.try_recv()
        {
            match loaded {
                Ok(model) => self.text = Model::Ready(model),
                Err(e) => {
                    self.text = Model::Unloaded;
                    self.typing = None;
                    self.history.retain(|p| !matches!(p, Probe::Text(_)));
                    self.stale = true;
                    return Err(e).context("cannot load the text model");
                }
            }
        }
        let Some(images) = &mut self.images else {
            return Ok(());
        };
        for answer in images.done.try_iter().collect::<Vec<_>>() {
            let Ok((i, embedding)) = answer else {
                self.images = None;
                return answer.map(drop).context("cannot embed the images");
            };
            images.pending -= 1;
            self.hits[i].embedding = embedding;
            self.fresh = true;
        }
        Ok(())
    }
}

/// The hits by `target`, best first. Equal scores go by path. Hits without an embedding keep
/// their order, at the end.
fn rank<'a>(hits: impl Iterator<Item = &'a Hit>, target: &Target) -> Vec<Hit> {
    let mut ranked: Vec<Hit> = hits
        .map(|hit| Hit {
            score: hit.embedding.as_deref().and_then(|e| target.score(e)),
            // A board in a new order takes each image as one of the others.
            live: false,
            ..hit.clone()
        })
        .collect();
    ranked.sort_by(|a, b| match (a.score, b.score) {
        (Some(x), Some(y)) => y.total_cmp(&x).then_with(|| a.path.cmp(&b.path)),
        (x, y) => y.is_some().cmp(&x.is_some()),
    });
    ranked
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn hit(name: &str, embedding: Option<[f32; 2]>) -> Hit {
        Hit {
            embedding: embedding.map(|e| Arc::from(e.as_slice())),
            ..Hit::new(name.into())
        }
    }

    fn names(hits: Option<Vec<Hit>>) -> Vec<String> {
        let hits = hits.expect("a new order");
        hits.iter().map(|h| h.path.display().to_string()).collect()
    }

    fn press(ranking: &mut Ranking, code: KeyCode) {
        ranking.key(KeyEvent::from(code));
    }

    #[test]
    fn rank_puts_the_best_first_and_the_unknown_last() {
        let hits = [
            hit("z", None),
            hit("b", Some([0.0, 1.0])),
            hit("a", None),
            hit("d", Some([1.0, 0.0])),
            hit("c", Some([1.0, 0.0])),
        ];
        let ranked = rank(hits.iter(), &Target::Like(vec![1.0, 0.0]));
        let order: Vec<_> = (ranked.iter())
            .map(|h| (h.path.to_str().unwrap(), h.score))
            .collect();
        let want = [
            ("c", Some(1.0)),
            ("d", Some(1.0)),
            ("b", Some(0.0)),
            ("z", None),
            ("a", None),
        ];
        assert_eq!(order, want);
    }

    #[test]
    fn like_ranks_and_back_restores_the_search_order() {
        let search = Search::empty();
        let mut ranking = Ranking::new();
        for h in [
            hit("a", Some([1.0, 0.0])),
            hit("b", Some([0.0, 1.0])),
            hit("c", Some([0.6, 0.8])),
        ] {
            ranking.add(h, &Order::Found);
        }
        assert!(ranking.poll().unwrap().is_none());
        ranking.like(Path::new("b"), &search).unwrap();
        assert!(ranking.probing());
        assert_eq!(names(ranking.poll().unwrap()), ["b", "c", "a"]);
        assert!(ranking.idle());
        ranking.back();
        assert!(!ranking.probing());
        let restored = ranking.poll().unwrap().unwrap();
        assert!(restored.iter().all(|h| h.score.is_none()));
        assert_eq!(names(Some(restored)), ["a", "b", "c"]);
    }

    #[test]
    fn a_new_image_goes_to_its_place_in_the_search_order() {
        use std::time::{Duration, SystemTime};

        use crate::search::{Dir, Field, FileInfo};

        let dated = |name: &str, secs: u64| Hit {
            file: Some(FileInfo {
                meta: None,
                modified: SystemTime::UNIX_EPOCH + Duration::from_secs(secs),
            }),
            ..hit(name, Some([1.0, 0.0]))
        };
        let newest = Order::Sorted(Field::Date, Dir::Natural);
        let mut ranking = Ranking::new();
        assert_eq!(ranking.add(dated("b", 200), &newest), 0);
        assert_eq!(ranking.add(dated("a", 100), &newest), 1);
        let live = Hit {
            live: true,
            ..dated("new", 300)
        };
        assert_eq!(ranking.add(live, &newest), 0, "the newest image goes first");
        assert_eq!(ranking.add(dated("mid", 150), &newest), 2);

        ranking.history.push(Probe::Like("a".into()));
        assert_eq!(
            ranking.add(dated("late", 400), &newest),
            4,
            "a probe board takes it last"
        );
        ranking.back();
        let restored = ranking.poll().unwrap();
        assert_eq!(names(restored.clone()), ["late", "new", "b", "mid", "a"]);
        assert!(restored.unwrap().iter().all(|h| !h.live));
        assert_eq!(names(Some(ranking.hits(&["mid".into()]))), ["mid"]);
    }

    #[test]
    fn the_query_line_edits_and_keeps_or_drops_the_text() {
        let mut ranking = Ranking::new();
        ranking.typing = Some(String::new());
        for c in "cats".chars() {
            press(&mut ranking, KeyCode::Char(c));
        }
        press(&mut ranking, KeyCode::Backspace);
        assert!(ranking.probe() == Some(Probe::Text("cat".into())));
        ranking.key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL));
        assert!(ranking.probe().is_none());
        press(&mut ranking, KeyCode::Char('x'));
        press(&mut ranking, KeyCode::Esc);
        assert!(!ranking.typing() && ranking.history.is_empty());

        ranking.typing = Some("dog".into());
        press(&mut ranking, KeyCode::Enter);
        assert!(ranking.history == [Probe::Text("dog".into())]);
        ranking.typing = Some(" ".into());
        press(&mut ranking, KeyCode::Enter);
        assert_eq!(ranking.history.len(), 1, "a blank query keeps the ranking");
    }
}
