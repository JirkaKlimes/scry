//! For --watch: after the walk, scry keeps watching the directories and sends each new image.

use std::{
    collections::{HashMap, HashSet},
    env, fs,
    io::ErrorKind::{NotFound, PermissionDenied},
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering::Relaxed},
        mpsc::{self, Receiver, RecvTimeoutError, Sender},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
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
/// On Linux, a writer that closes the file tells so. A write waits for that, but not longer than
/// this, for a writer that keeps the file open.
const WRITING: Duration = Duration::from_secs(10);

/// Watches each directory that the walk enters. On macOS a watch covers the whole tree below
/// it, so only the roots get one.
#[derive(Clone)]
pub struct Dirs {
    watcher: Arc<Mutex<RecommendedWatcher>>,
    /// The system limit of watches is reached. The warning shows once, because every directory
    /// after it fails too.
    full: Arc<AtomicBool>,
    /// macOS also reports a directory when a file in it changes. Only a directory that is not
    /// here yet is new.
    known: Arc<Mutex<HashSet<PathBuf>>>,
}

impl Dirs {
    pub fn new() -> Result<(Self, Receiver<notify::Result<Event>>)> {
        let (tx, events) = mpsc::channel();
        let watcher = RecommendedWatcher::new(tx, notify::Config::default())?;
        let dirs = Self {
            watcher: Arc::new(Mutex::new(watcher)),
            full: Arc::default(),
            known: Arc::default(),
        };
        Ok((dirs, events))
    }

    pub fn root(&self, dir: &Path) {
        self.known.lock().unwrap().insert(dir.to_path_buf());
        let mode = if cfg!(target_os = "macos") {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };
        self.watch(dir, mode);
    }

    /// A directory below a root. False if it has a watch already.
    pub fn add(&self, dir: &Path) -> bool {
        if !self.known.lock().unwrap().insert(dir.to_path_buf()) {
            return false;
        }
        if !cfg!(target_os = "macos") {
            self.watch(dir, RecursiveMode::NonRecursive);
        }
        true
    }

    /// A directory that is gone, with the ones below it. A new one at its path needs a watch.
    fn forget(&self, dir: &Path) {
        self.known
            .lock()
            .unwrap()
            .retain(|known| !known.starts_with(dir));
    }

    fn watch(&self, dir: &Path, mode: RecursiveMode) {
        let Err(e) = self.watcher.lock().unwrap().watch(dir, mode) else {
            return;
        };
        let why = match e.kind {
            // The walk tells about a directory that it cannot read. One that is gone needs no
            // watch.
            notify::ErrorKind::PathNotFound | notify::ErrorKind::WatchNotFound => return,
            notify::ErrorKind::Io(e) if matches!(e.kind(), NotFound | PermissionDenied) => return,
            notify::ErrorKind::MaxFilesWatch if self.full.swap(true, Relaxed) => return,
            notify::ErrorKind::MaxFilesWatch => {
                "the system limit is reached. Raise fs.inotify.max_user_watches.".into()
            }
            notify::ErrorKind::Io(e) => e.to_string(),
            notify::ErrorKind::Generic(text) => text,
            notify::ErrorKind::InvalidConfig(config) => format!("{config:?}"),
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

/// What the follower of the events needs.
pub struct Follow<'a> {
    pub roots: &'a [PathBuf],
    pub opts: WalkOpts,
    pub dirs: Dirs,
    /// When the watches went up.
    pub since: SystemTime,
    pub mode: Watch,
}

impl Follow<'_> {
    /// Turns events into the paths of images that are ready to read, and sends them in the form
    /// the walk prints.
    pub fn start(self, events: Receiver<notify::Result<Event>>, tx: Sender<Found>) -> Result<()> {
        let cwd = env::current_dir()?;
        let roots = (self.roots.iter())
            .map(|root| Root {
                given: root.clone(),
                absolute: cwd.join(root),
                canonical: fs::canonicalize(root).ok(),
                matcher: builder(root, Path::new(""), &self.opts, None)
                    .build_matchers()
                    .remove(0),
            })
            .collect();
        let mut follower = Follower {
            roots,
            opts: self.opts,
            dirs: self.dirs,
            since: self.since,
            mode: self.mode,
            tx,
        };
        thread::spawn(move || follower.run(&events));
        Ok(())
    }
}

struct Follower {
    roots: Vec<Root>,
    opts: WalkOpts,
    dirs: Dirs,
    since: SystemTime,
    mode: Watch,
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
            let first = match events.recv_timeout(wait) {
                Ok(event) => Some(event),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => return,
            };
            // All waiting events first, so the paths of one burst go out once.
            for event in first.into_iter().chain(events.try_iter()) {
                match event {
                    Ok(event) if event.need_rescan() => {
                        if !self.rescan() {
                            return;
                        }
                    }
                    Ok(event) => self.note(&event, &mut due),
                    Err(_) => {}
                }
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

    fn note(&self, event: &Event, due: &mut HashMap<PathBuf, (usize, Instant)>) {
        if let EventKind::Remove(_) | EventKind::Modify(ModifyKind::Name(RenameMode::From)) =
            event.kind
        {
            for path in &event.paths {
                if let Some((_, path)) = self.local(path) {
                    self.dirs.forget(&path);
                }
            }
        }
        let Some(after) = settle(&event.kind) else {
            return;
        };
        let at = Instant::now() + after;
        for path in &event.paths {
            if let Some((root, path)) = self.local(path) {
                due.insert(path, (root, at));
            }
        }
    }

    /// The system dropped events, so the roots are walked again for the files that changed since
    /// the watches went up. The engine skips the images that it sent.
    fn rescan(&self) -> bool {
        (self.roots.iter()).all(|root| self.scan(&root.given, Path::new(""), true))
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
        let Ok(md) = fs::symlink_metadata(path) else {
            return true;
        };
        // A new link to an old file is new.
        let changed = changed_since(&md, self.since);
        // As in the walk, a link to a file counts, and a link to a directory is not followed.
        let md = match md.is_symlink() {
            true => match fs::metadata(path) {
                Ok(target) if target.is_file() => target,
                _ => return true,
            },
            false => md,
        };
        let below = (path.strip_prefix(&self.roots[root].given))
            .expect("local paths start with their root")
            .to_path_buf();
        if !self.admits(root, &below, md.is_dir()) {
            return true;
        }
        if md.is_dir() {
            // Watch first, so the scan sees each file that comes before the watch. A directory
            // that moves in keeps the change times of its files, so they all count. The files of
            // a known directory tell about themselves.
            return !self.dirs.add(path) || self.scan(path, &below, false);
        }
        // An empty file is a placeholder that a writer fills later.
        if !md.is_file() || md.len() == 0 {
            return true;
        }
        if self.mode == Watch::New && !changed {
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
    /// `changed`: only the files that changed since the watches went up.
    fn scan(&self, dir: &Path, below: &Path, changed: bool) -> bool {
        let walk = builder(dir, below, &self.opts, Some(&self.dirs)).build();
        (walk.filter_map(entry))
            .filter(|found| !changed || self.changed(found))
            .all(|found| self.tx.send(found).is_ok())
    }

    fn changed(&self, found: &Found) -> bool {
        match found {
            Found::Image(path) | Found::Unsupported(path) => {
                fs::metadata(path).is_ok_and(|md| changed_since(&md, self.since))
            }
            Found::Error { .. } => false,
        }
    }
}

/// A write, a rename or a link moves the change time. An open with no write leaves it.
fn changed_since(md: &fs::Metadata, since: SystemTime) -> bool {
    let changed = u64::try_from(md.ctime())
        .map(|secs| UNIX_EPOCH + Duration::new(secs, md.ctime_nsec() as u32));
    changed.is_ok_and(|changed| changed >= since)
}

/// How long a path waits before scry reads it. A writer that closes the file, or a rename into
/// place, means it is ready. Other changes wait for quiet. scry's own reads open files too.
fn settle(kind: &EventKind) -> Option<Duration> {
    match kind {
        EventKind::Access(AccessKind::Close(AccessMode::Write))
        | EventKind::Modify(ModifyKind::Name(RenameMode::To)) => Some(Duration::ZERO),
        EventKind::Modify(ModifyKind::Any | ModifyKind::Data(_)) if cfg!(target_os = "linux") => {
            Some(WRITING)
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
        // macOS can report a path again after it settles. The engine drops the repeats.
        while let Ok(found) = live.recv_timeout(SETTLE * 3) {
            assert!(
                matches!(&found, Found::Image(path) if want.contains(path)),
                "nothing else comes"
            );
        }
    }
}
