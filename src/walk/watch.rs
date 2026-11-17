//! For --watch: after the walk, scry keeps watching the directories and sends each new image.

use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering::Relaxed},
        mpsc::{self, Receiver, RecvTimeoutError, Sender},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::Result;
use clap::ValueEnum;
use ignore::IncrementalIgnore;
use notify::{
    Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
    event::{AccessKind, AccessMode, ModifyKind, RenameMode},
};

use super::{Found, WalkOpts, allowed, builder, entry};
use crate::{
    output::clean_path,
    report::{self, Code},
};

/// All images, then each new one, or only the new ones.
#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
pub enum Watch {
    All,
    New,
}

/// A writer that does not close its file, or a system that does not tell, gets this much quiet
/// before scry reads the file.
const SETTLE: Duration = Duration::from_millis(500);

/// Watches each directory that the walk enters. On macOS a watch covers the whole tree below
/// it, so only the roots get one.
#[derive(Clone)]
pub struct Dirs {
    watcher: Arc<Mutex<RecommendedWatcher>>,
    /// Only the first failure shows: past the system limit, every directory fails.
    failed: Arc<AtomicBool>,
}

impl Dirs {
    pub fn new() -> Result<(Self, Receiver<notify::Result<Event>>)> {
        let (tx, events) = mpsc::channel();
        let watcher = RecommendedWatcher::new(tx, notify::Config::default())?;
        let dirs = Self {
            watcher: Arc::new(Mutex::new(watcher)),
            failed: Arc::default(),
        };
        Ok((dirs, events))
    }

    pub fn root(&self, dir: &Path) {
        let mode = if cfg!(target_os = "macos") {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };
        self.watch(dir, mode);
    }

    /// A directory below a root.
    pub fn add(&self, dir: &Path) {
        if !cfg!(target_os = "macos") {
            self.watch(dir, RecursiveMode::NonRecursive);
        }
    }

    fn watch(&self, dir: &Path, mode: RecursiveMode) {
        let Err(e) = self.watcher.lock().unwrap().watch(dir, mode) else {
            return;
        };
        if self.failed.swap(true, Relaxed) {
            return;
        }
        let why = match e.kind {
            notify::ErrorKind::MaxFilesWatch => {
                "the system limit is reached. Raise fs.inotify.max_user_watches.".into()
            }
            _ => e.to_string(),
        };
        let message = format!("cannot watch {} for new images: {why}", clean_path(dir));
        report::warn(Code::Unwatched, &message);
    }
}

/// A walk root, and the forms in which a watch can report its paths.
struct Root {
    given: PathBuf,
    absolute: PathBuf,
    canonical: Option<PathBuf>,
    matcher: IncrementalIgnore,
}

/// Turns events into the paths of images that are ready to read, and sends them in the form
/// the walk prints.
pub fn follow(
    events: Receiver<notify::Result<Event>>,
    roots: &[PathBuf],
    opts: WalkOpts,
    dirs: Dirs,
    tx: Sender<Found>,
) -> Result<()> {
    let cwd = env::current_dir()?;
    let roots = (roots.iter())
        .map(|root| Root {
            given: root.clone(),
            absolute: cwd.join(root),
            canonical: fs::canonicalize(root).ok(),
            matcher: builder(root, Path::new(""), &opts, None)
                .build_matchers()
                .remove(0),
        })
        .collect();
    let mut follower = Follower {
        roots,
        opts,
        dirs,
        tx,
    };
    thread::spawn(move || follower.run(&events));
    Ok(())
}

struct Follower {
    roots: Vec<Root>,
    opts: WalkOpts,
    dirs: Dirs,
    tx: Sender<Found>,
}

impl Follower {
    /// Ends once nobody takes the images.
    fn run(&mut self, events: &Receiver<notify::Result<Event>>) {
        let mut due: HashMap<PathBuf, (usize, Instant)> = HashMap::new();
        loop {
            let next = due.values().map(|&(_, at)| at).min();
            let wait = next.map_or(Duration::MAX, |at| {
                at.saturating_duration_since(Instant::now())
            });
            match events.recv_timeout(wait) {
                Ok(Ok(event)) if event.need_rescan() => {
                    // The system dropped events. The engine skips the images that it sent.
                    for i in 0..self.roots.len() {
                        let root = self.roots[i].given.clone();
                        if !self.scan(&root, Path::new("")) {
                            return;
                        }
                    }
                }
                Ok(Ok(event)) => {
                    let Some(after) = settle(&event.kind) else {
                        continue;
                    };
                    let at = Instant::now() + after;
                    for path in &event.paths {
                        if let Some((root, path)) = self.local(path) {
                            due.insert(path, (root, at));
                        }
                    }
                }
                Ok(Err(_)) | Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return,
            }
            let now = Instant::now();
            let ready: Vec<_> = due.extract_if(|_, &mut (_, at)| at <= now).collect();
            for (path, (root, _)) in ready {
                if !self.handle(root, &path) {
                    return;
                }
            }
        }
    }

    /// The root of an event path, and the path as the walk prints it.
    fn local(&self, path: &Path) -> Option<(usize, PathBuf)> {
        self.roots.iter().enumerate().find_map(|(i, root)| {
            let bases = [
                Some(&root.given),
                Some(&root.absolute),
                root.canonical.as_ref(),
            ];
            let below = bases
                .into_iter()
                .flatten()
                .find_map(|base| path.strip_prefix(base).ok())?;
            Some((i, root.given.join(below)))
        })
    }

    /// False once nobody takes the images.
    fn handle(&mut self, root: usize, path: &Path) -> bool {
        // Gone, as a temporary file that was renamed.
        let Ok(md) = fs::metadata(path) else {
            return true;
        };
        let below = (path.strip_prefix(&self.roots[root].given))
            .expect("local paths start with their root")
            .to_path_buf();
        if !self.admits(root, &below, md.is_dir()) {
            return true;
        }
        if md.is_dir() {
            // Watch first, so the scan sees each file that comes before the watch.
            self.dirs.add(path);
            return self.scan(path, &below);
        }
        // An empty file is a placeholder that a writer fills later.
        if !md.is_file() || md.len() == 0 {
            return true;
        }
        Found::new(path).is_none_or(|found| self.tx.send(found).is_ok())
    }

    /// The walk rules: hidden files, ignore files, globs, and each directory on the way.
    fn admits(&mut self, root: usize, below: &Path, is_dir: bool) -> bool {
        let globs = &self.opts.globs;
        let on_the_way = (below.ancestors().skip(1))
            .filter(|dir| !dir.as_os_str().is_empty())
            .all(|dir| allowed(globs, dir, true));
        on_the_way
            && allowed(globs, below, is_dir)
            && !self.roots[root].matcher.matched(below, is_dir).is_ignore()
    }

    /// Walks a directory that came after the walk, or a root again.
    fn scan(&self, dir: &Path, below: &Path) -> bool {
        let walk = builder(dir, below, &self.opts, Some(&self.dirs)).build();
        walk.filter_map(entry)
            .all(|found| self.tx.send(found).is_ok())
    }
}

/// How long a path waits before scry reads it. A writer that closes the file, or a rename into
/// place, means it is ready. Other changes wait for quiet. scry's own reads open files too.
fn settle(kind: &EventKind) -> Option<Duration> {
    match kind {
        EventKind::Access(AccessKind::Close(AccessMode::Write))
        | EventKind::Modify(ModifyKind::Name(RenameMode::To | RenameMode::Both)) => {
            Some(Duration::ZERO)
        }
        EventKind::Create(_)
        | EventKind::Modify(
            ModifyKind::Any | ModifyKind::Data(_) | ModifyKind::Other | ModifyKind::Name(_),
        ) => Some(SETTLE),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::{
        testing::Temp,
        walk::{Source, candidates, globs},
    };

    #[test]
    fn sends_new_images_once_they_are_ready() {
        let root = Temp::dir("watch");
        fs::write(root.join("old.png"), b"x").unwrap();
        let opts = WalkOpts {
            hidden: false,
            no_ignore: false,
            globs: globs([]).unwrap(),
        };
        let source = Source {
            paths: vec![root.to_path_buf()],
            stdin: None,
        };
        let walk = candidates(source, opts, Some(Watch::New)).unwrap();
        assert_eq!(
            walk.found.iter().count(),
            0,
            "=new skips the images that exist"
        );
        fs::create_dir(root.join("sub")).unwrap();
        fs::write(root.join("sub/new.jpg"), b"x").unwrap();
        fs::write(root.join("b.png"), b"x").unwrap();
        fs::write(root.join(".hidden.png"), b"x").unwrap();
        fs::write(root.join("notes.txt"), b"x").unwrap();
        fs::write(root.join("empty.png"), b"").unwrap();
        let live = walk.live.unwrap();
        let want = BTreeSet::from([root.join("b.png"), root.join("sub/new.jpg")]);
        let mut got = BTreeSet::new();
        while got != want {
            match live.recv_timeout(Duration::from_secs(5)) {
                Ok(Found::Image(path)) => _ = got.insert(path),
                Ok(_) => {}
                Err(_) => break,
            }
        }
        assert_eq!(got, want);
        assert!(live.recv_timeout(SETTLE * 3).is_err(), "nothing else comes");
    }
}
