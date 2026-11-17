use std::{
    ffi::OsString,
    fs::{self, File},
    io::{self, BufRead, BufReader},
    iter,
    os::{fd::AsFd, unix::ffi::OsStringExt},
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

use anyhow::{Context, Result};
use ignore::{
    DirEntry, WalkBuilder, WalkState,
    overrides::{Override, OverrideBuilder},
};

use crate::{
    decode,
    output::{End, clean_path},
};

pub struct Source {
    pub paths: Vec<PathBuf>,
    /// Paths on stdin, each ended by this byte.
    pub stdin: Option<End>,
}

#[derive(Clone)]
pub struct WalkOpts {
    pub hidden: bool,
    pub no_ignore: bool,
    pub globs: Override,
}

pub enum Found {
    Image(PathBuf),
    /// An image in a format that scry cannot read.
    Unsupported,
    /// A file or directory that scry cannot read.
    Error {
        path: Option<PathBuf>,
        reason: String,
    },
}

impl Found {
    fn new(path: &Path) -> Option<Self> {
        if decode::is_image(path) {
            Some(Found::Image(path.to_path_buf()))
        } else if decode::is_unsupported(path) {
            Some(Found::Unsupported)
        } else {
            None
        }
    }
}

pub struct Walk {
    pub found: Receiver<Found>,
    /// What a directory walk left out. None without a directory walk or filters.
    pub skipped: Option<Skipped>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Skipped {
    Hidden,
    /// Paths in .gitignore and other ignore files.
    Ignored,
    Both,
}

/// Globs filter every path. File arguments and stdin paths skip the hidden and ignore filters.
pub fn candidates(source: Source, opts: WalkOpts) -> Result<Walk> {
    let (tx, found) = mpsc::channel();
    let mut dirs = Vec::new();
    for path in source.paths {
        let context = || clean_path(&path).to_string();
        if fs::metadata(&path).with_context(context)?.is_dir() {
            // A directory that scry cannot list, as macOS guards ~/Documents, stops scry here
            // with the reason.
            fs::read_dir(&path).with_context(context)?;
            dirs.push(path);
        } else if let Some(found) = wanted(&path, &opts.globs) {
            tx.send(found).expect("the receiver is in this function");
        }
    }
    let skipped = match (opts.hidden, opts.no_ignore) {
        _ if dirs.is_empty() => None,
        (false, false) => Some(Skipped::Both),
        (false, true) => Some(Skipped::Hidden),
        (true, false) => Some(Skipped::Ignored),
        (true, true) => None,
    };
    if let Some(end) = source.stdin {
        // Read a duplicate so that fd 0 can later be pointed at the terminal.
        let input = File::from(io::stdin().as_fd().try_clone_to_owned()?);
        let (opts, tx) = (opts.clone(), tx.clone());
        thread::spawn(move || read_list(BufReader::new(input), end, &opts, &tx));
    }
    walk(&dirs, opts, tx);
    Ok(Walk { found, skipped })
}

/// Globs ignore case, and a leading ! excludes. A glob with a slash matches a walked path from
/// below its walk root, and a file argument or stdin path as given.
pub fn globs<'a>(patterns: impl IntoIterator<Item = &'a str>) -> Result<Override, ignore::Error> {
    let mut builder = OverrideBuilder::new(".");
    builder.case_insensitive(true)?;
    for glob in patterns {
        builder.add(glob)?;
    }
    builder.build()
}

/// A directory in the list is walked there, one at a time: a list of many directories would
/// start too many parallel walks.
fn read_list(input: impl BufRead, end: End, opts: &WalkOpts, tx: &Sender<Found>) {
    for line in input.split(end.byte()) {
        let mut line = match line {
            Ok(line) => line,
            Err(e) => {
                let _ = tx.send(Found::Error {
                    path: None,
                    reason: format!("cannot read the paths on stdin: {e}"),
                });
                return;
            }
        };
        // A list from a Windows tool ends its lines with CRLF.
        if end == End::Newline && line.last() == Some(&b'\r') {
            line.pop();
        }
        let path = PathBuf::from(OsString::from_vec(line));
        let mut found: Box<dyn Iterator<Item = Found>> = match wanted(&path, &opts.globs) {
            Some(found) => Box::new(iter::once(found)),
            None if path.is_dir() => Box::new(builder(&path, opts).build().filter_map(entry)),
            None => continue,
        };
        if !found.all(|found| tx.send(found).is_ok()) {
            return;
        }
    }
}

fn walk(dirs: &[PathBuf], opts: WalkOpts, tx: Sender<Found>) {
    let Some((first, rest)) = dirs.split_first() else {
        return;
    };
    let mut builder = builder(first, &opts);
    for dir in rest {
        builder.add(dir);
    }
    let walker = builder.build_parallel();
    thread::spawn(move || {
        walker.run(|| {
            let tx = tx.clone();
            Box::new(move |e| match entry(e).map(|found| tx.send(found)) {
                Some(Err(_)) => WalkState::Quit,
                _ => WalkState::Continue,
            })
        })
    });
}

fn builder(dir: &Path, opts: &WalkOpts) -> WalkBuilder {
    let globs = opts.globs.clone();
    let mut builder = WalkBuilder::new(dir);
    // As walker overrides, an include glob would also admit hidden and ignored files.
    // Globs here only narrow the walk.
    builder
        .standard_filters(!opts.no_ignore)
        .hidden(!opts.hidden)
        .filter_entry(move |e| {
            let is_dir = e.file_type().is_some_and(|t| t.is_dir());
            // The Photos library of macOS holds several copies of each photo, and macOS guards
            // it. As a walk root, it is still searched.
            let photos = is_dir
                && (e.path().extension()).is_some_and(|x| x.eq_ignore_ascii_case("photoslibrary"));
            !photos && !globs.matched(relative(e), is_dir).is_ignore()
        });
    builder
}

/// An image, or an error. Other entries give None.
fn entry(entry: Result<DirEntry, ignore::Error>) -> Option<Found> {
    match entry {
        Ok(e) => Found::new(e.path()).filter(|_| is_file(&e)),
        Err(e) => Some(walk_error(e)),
    }
}

fn walk_error(e: ignore::Error) -> Found {
    match e {
        ignore::Error::WithPath { path, err } => Found::Error {
            path: Some(path),
            reason: err.to_string(),
        },
        ignore::Error::WithDepth { err, .. } => walk_error(*err),
        e => Found::Error {
            path: None,
            reason: e.to_string(),
        },
    }
}

/// The path below the root that the walk started from. A glob like `Screenshots/*` matches
/// against it, not against the whole path.
fn relative(entry: &DirEntry) -> &Path {
    let mut parts = entry.path().components();
    let root = parts.clone().count().saturating_sub(entry.depth());
    parts.by_ref().take(root).for_each(drop);
    parts.as_path()
}

/// Links to files count. Links to directories are not followed.
fn is_file(entry: &DirEntry) -> bool {
    entry.file_type().is_some_and(|t| t.is_file())
        || entry.path_is_symlink() && entry.path().is_file()
}

fn wanted(path: &Path, globs: &Override) -> Option<Found> {
    Found::new(path).filter(|_| !globs.matched(path, false).is_ignore())
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, io::Cursor, os::unix::fs::PermissionsExt};

    use super::*;
    use crate::testing::Temp;

    fn tree() -> Temp {
        let root = Temp::dir("walk");
        for f in [
            "a.jpg",
            "b.PNG",
            "notes.txt",
            "sub/c.webp",
            "sub/skip.jpg",
            ".hidden/d.gif",
            ".e.jpg",
            "ignored/f.bmp",
            "IMG_1.CR2",
            "Photos Library.photoslibrary/originals/g.jpg",
            "sub/raw.nef",
        ] {
            let path = root.join(f);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, b"").unwrap();
        }
        fs::write(root.join(".ignore"), "ignored/\n").unwrap();
        root
    }

    fn paths(paths: Vec<PathBuf>) -> Source {
        Source { paths, stdin: None }
    }

    fn found(source: Source, opts: WalkOpts) -> Vec<Found> {
        candidates(source, opts).unwrap().found.iter().collect()
    }

    fn image(found: Found) -> Option<PathBuf> {
        match found {
            Found::Image(path) => Some(path),
            _ => None,
        }
    }

    fn names(source: Source, opts: WalkOpts, root: &Path) -> BTreeSet<String> {
        found(source, opts)
            .into_iter()
            .filter_map(image)
            .map(|p| {
                p.strip_prefix(root)
                    .unwrap_or(&p)
                    .to_string_lossy()
                    .into_owned()
            })
            .collect()
    }

    fn opts(patterns: &[&str]) -> WalkOpts {
        WalkOpts {
            hidden: false,
            no_ignore: false,
            globs: globs(patterns.iter().copied()).unwrap(),
        }
    }

    fn set(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn walks_images_with_filters() {
        let root = tree();
        let walk = || paths(vec![root.to_path_buf()]);
        assert_eq!(
            names(walk(), opts(&[]), &root),
            set(&["a.jpg", "b.PNG", "sub/c.webp", "sub/skip.jpg"])
        );
        assert_eq!(
            names(walk(), opts(&["*.jpg"]), &root),
            set(&["a.jpg", "sub/skip.jpg"])
        );
        assert_eq!(
            names(walk(), opts(&["*.png"]), &root),
            set(&["b.PNG"]),
            "globs ignore case"
        );
        assert_eq!(
            names(walk(), opts(&["!skip.*", "!c.webp"]), &root),
            set(&["a.jpg", "b.PNG"])
        );
        assert_eq!(
            names(walk(), opts(&["!sub/"]), &root),
            set(&["a.jpg", "b.PNG"])
        );
        assert_eq!(
            names(walk(), opts(&["sub/*"]), &root),
            set(&["sub/c.webp", "sub/skip.jpg"]),
            "slash globs match below the walk root"
        );
        assert_eq!(
            names(paths(vec![root.join("sub")]), opts(&["sub/*"]), &root),
            set(&[]),
            "the root is not part of the matched path"
        );
        assert_eq!(
            names(walk(), opts(&["**/sub/*"]), &root),
            set(&["sub/c.webp", "sub/skip.jpg"])
        );
        let all = WalkOpts {
            hidden: true,
            no_ignore: true,
            ..opts(&[])
        };
        assert_eq!(
            names(walk(), all, &root),
            set(&[
                "a.jpg",
                "b.PNG",
                "sub/c.webp",
                "sub/skip.jpg",
                ".hidden/d.gif",
                ".e.jpg",
                "ignored/f.bmp"
            ])
        );
        let hidden = WalkOpts {
            hidden: true,
            ..opts(&[])
        };
        assert!(!names(walk(), hidden, &root).contains("ignored/f.bmp"));
    }

    #[test]
    fn counts_images_it_cannot_read() {
        let root = tree();
        let unsupported = |source, globs: &[&str]| {
            let found = found(source, opts(globs));
            found
                .iter()
                .filter(|f| matches!(f, Found::Unsupported))
                .count()
        };
        assert_eq!(unsupported(paths(vec![root.to_path_buf()]), &[]), 2);
        assert_eq!(unsupported(paths(vec![root.to_path_buf()]), &["*.jpg"]), 0);
        assert_eq!(unsupported(paths(vec![root.join("IMG_1.CR2")]), &[]), 1);
    }

    #[test]
    fn explicit_files_skip_walk_filters_but_not_globs() {
        let root = tree();
        let source = || {
            paths(vec![
                root.join("sub/skip.jpg"),
                root.join(".e.jpg"),
                root.join("ignored/f.bmp"),
                root.join("notes.txt"),
                root.join("sub"),
            ])
        };
        assert_eq!(
            names(source(), opts(&[]), &root),
            set(&["sub/skip.jpg", ".e.jpg", "ignored/f.bmp", "sub/c.webp"])
        );
        assert_eq!(
            names(source(), opts(&["!skip.jpg", "!*.bmp"]), &root),
            set(&[".e.jpg", "sub/c.webp"])
        );
        assert!(candidates(paths(vec![root.join("missing.jpg")]), opts(&[])).is_err());
        assert!(globs(["a[b"]).is_err());
    }

    #[test]
    fn follows_links_to_files_only() {
        let root = tree();
        let target = root.join("ignored");
        std::os::unix::fs::symlink(target.join("f.bmp"), root.join("link.bmp")).unwrap();
        std::os::unix::fs::symlink(&target, root.join("album")).unwrap();
        let names = names(paths(vec![root.to_path_buf()]), opts(&["*.bmp"]), &root);
        assert_eq!(names, set(&["link.bmp"]));
    }

    #[test]
    fn reports_unreadable_directories() {
        let root = tree();
        let locked = root.join("sub");
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();
        let readable = fs::read_dir(&locked).is_ok();
        let found = found(paths(vec![root.to_path_buf()]), opts(&[]));
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).unwrap();
        // Root can read the directory anyway.
        if !readable {
            let errors = found
                .iter()
                .filter(|f| matches!(f, Found::Error { path: Some(_), .. }));
            assert_eq!(errors.count(), 1);
            assert_eq!(found.into_iter().filter_map(image).count(), 2);
        }
    }

    #[test]
    fn reads_path_lists() {
        let collect = |input: &[u8], end, patterns: &[&str]| {
            let (tx, rx) = mpsc::channel();
            read_list(Cursor::new(input.to_vec()), end, &opts(patterns), &tx);
            drop(tx);
            rx.iter().filter_map(image).collect::<Vec<_>>()
        };
        let p = |s: &str| PathBuf::from(s);
        assert_eq!(
            collect(b"a.jpg\nb.txt\n\nc d.PNG\n", End::Newline, &[]),
            [p("a.jpg"), p("c d.PNG")]
        );
        assert_eq!(
            collect(b"a\nb.jpg\0c.gif\r", End::Nul, &[]),
            [p("a\nb.jpg")]
        );
        assert_eq!(
            collect(b"\xff.jpg\n", End::Newline, &[]),
            [PathBuf::from(OsString::from_vec(b"\xff.jpg".to_vec()))]
        );
        assert_eq!(
            collect(b"x/a.jpg\nb.png\nc.PNG\n", End::Newline, &["*.png"]),
            [p("b.png"), p("c.PNG")]
        );
        assert_eq!(
            collect(b"a.jpg\r\nb c.png\r\n", End::Newline, &[]),
            [p("a.jpg"), p("b c.png")]
        );
        assert_eq!(
            collect(b"pics/sub/a.jpg\nsub/b.jpg\n", End::Newline, &["sub/*"]),
            [p("sub/b.jpg")],
            "stdin paths match as given"
        );
        let root = tree();
        let list = format!("{}\n", root.join("sub").display());
        let mut walked = collect(list.as_bytes(), End::Newline, &[]);
        walked.sort();
        assert_eq!(
            walked,
            [root.join("sub/c.webp"), root.join("sub/skip.jpg")],
            "directories in the list are walked"
        );
    }

    #[test]
    fn only_directory_walks_filter() {
        let root = tree();
        let skipped = |dirs, hidden, no_ignore| {
            let opts = WalkOpts {
                hidden,
                no_ignore,
                ..opts(&[])
            };
            candidates(paths(dirs), opts).unwrap().skipped
        };
        let dir = || vec![root.to_path_buf()];
        assert_eq!(skipped(dir(), false, false), Some(Skipped::Both));
        assert_eq!(skipped(dir(), true, false), Some(Skipped::Ignored));
        assert_eq!(skipped(dir(), false, true), Some(Skipped::Hidden));
        assert_eq!(skipped(dir(), true, true), None);
        assert_eq!(skipped(vec![root.join("a.jpg")], false, false), None);
    }
}
