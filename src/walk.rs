use std::{
    ffi::OsString,
    fs::{self, File},
    io::{self, BufRead, BufReader},
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

pub enum Source {
    Paths(Vec<PathBuf>),
    Stdin(End),
}

pub struct WalkOpts {
    pub hidden: bool,
    pub no_ignore: bool,
    pub globs: Override,
}

pub enum Found {
    Image(PathBuf),
    /// A HEIC, RAW or other image that scry cannot read yet.
    Unsupported,
    /// A file or directory that scry cannot read.
    Error,
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
    let skipped = match source {
        Source::Stdin(end) => {
            // Read a duplicate so that fd 0 can later be pointed at the terminal.
            let input = File::from(io::stdin().as_fd().try_clone_to_owned()?);
            let globs = opts.globs;
            thread::spawn(move || read_list(BufReader::new(input), end, &globs, &tx));
            None
        }
        Source::Paths(paths) => {
            let mut dirs = Vec::new();
            for path in paths {
                if fs::metadata(&path)
                    .with_context(|| clean_path(&path).to_string())?
                    .is_dir()
                {
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
            walk(&dirs, opts, tx);
            skipped
        }
    };
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

fn read_list(input: impl BufRead, end: End, globs: &Override, tx: &Sender<Found>) {
    for line in input.split(end.byte()) {
        let Ok(mut line) = line else {
            let _ = tx.send(Found::Error);
            return;
        };
        // A list from a Windows tool ends its lines with CRLF.
        if end == End::Newline && line.last() == Some(&b'\r') {
            line.pop();
        }
        let path = PathBuf::from(OsString::from_vec(line));
        if let Some(found) = wanted(&path, globs)
            && tx.send(found).is_err()
        {
            return;
        }
    }
}

fn walk(dirs: &[PathBuf], opts: WalkOpts, tx: Sender<Found>) {
    let Some((first, rest)) = dirs.split_first() else {
        return;
    };
    let mut builder = WalkBuilder::new(first);
    for dir in rest {
        builder.add(dir);
    }
    let globs = opts.globs;
    // As walker overrides, an include glob would also admit hidden and ignored files.
    // Globs here only narrow the walk.
    let walker = builder
        .standard_filters(!opts.no_ignore)
        .hidden(!opts.hidden)
        .filter_entry(move |e| {
            let is_dir = e.file_type().is_some_and(|t| t.is_dir());
            !globs.matched(relative(e), is_dir).is_ignore()
        })
        .build_parallel();
    thread::spawn(move || {
        walker.run(|| {
            let tx = tx.clone();
            Box::new(move |entry| {
                let found = match entry {
                    Ok(e) => match Found::new(e.path()) {
                        Some(found) if is_file(&e) => found,
                        _ => return WalkState::Continue,
                    },
                    Err(_) => Found::Error,
                };
                match tx.send(found) {
                    Ok(()) => WalkState::Continue,
                    Err(_) => WalkState::Quit,
                }
            })
        })
    });
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
            "IMG_1.HEIC",
            "sub/raw.nef",
        ] {
            let path = root.join(f);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, b"").unwrap();
        }
        fs::write(root.join(".ignore"), "ignored/\n").unwrap();
        root
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
        let paths = || Source::Paths(vec![root.to_path_buf()]);
        assert_eq!(
            names(paths(), opts(&[]), &root),
            set(&["a.jpg", "b.PNG", "sub/c.webp", "sub/skip.jpg"])
        );
        assert_eq!(
            names(paths(), opts(&["*.jpg"]), &root),
            set(&["a.jpg", "sub/skip.jpg"])
        );
        assert_eq!(
            names(paths(), opts(&["*.png"]), &root),
            set(&["b.PNG"]),
            "globs ignore case"
        );
        assert_eq!(
            names(paths(), opts(&["!skip.*", "!c.webp"]), &root),
            set(&["a.jpg", "b.PNG"])
        );
        assert_eq!(
            names(paths(), opts(&["!sub/"]), &root),
            set(&["a.jpg", "b.PNG"])
        );
        assert_eq!(
            names(paths(), opts(&["sub/*"]), &root),
            set(&["sub/c.webp", "sub/skip.jpg"]),
            "slash globs match below the walk root"
        );
        assert_eq!(
            names(
                Source::Paths(vec![root.join("sub")]),
                opts(&["sub/*"]),
                &root
            ),
            set(&[]),
            "the root is not part of the matched path"
        );
        assert_eq!(
            names(paths(), opts(&["**/sub/*"]), &root),
            set(&["sub/c.webp", "sub/skip.jpg"])
        );
        let all = WalkOpts {
            hidden: true,
            no_ignore: true,
            ..opts(&[])
        };
        assert_eq!(
            names(paths(), all, &root),
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
        assert!(!names(paths(), hidden, &root).contains("ignored/f.bmp"));
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
        assert_eq!(unsupported(Source::Paths(vec![root.to_path_buf()]), &[]), 2);
        assert_eq!(
            unsupported(Source::Paths(vec![root.to_path_buf()]), &["*.jpg"]),
            0
        );
        assert_eq!(
            unsupported(Source::Paths(vec![root.join("IMG_1.HEIC")]), &[]),
            1
        );
    }

    #[test]
    fn explicit_files_skip_walk_filters_but_not_globs() {
        let root = tree();
        let source = || {
            Source::Paths(vec![
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
        assert!(candidates(Source::Paths(vec![root.join("missing.jpg")]), opts(&[])).is_err());
        assert!(globs(["a[b"]).is_err());
    }

    #[test]
    fn follows_links_to_files_only() {
        let root = tree();
        let target = root.join("ignored");
        std::os::unix::fs::symlink(target.join("f.bmp"), root.join("link.bmp")).unwrap();
        std::os::unix::fs::symlink(&target, root.join("album")).unwrap();
        let names = names(
            Source::Paths(vec![root.to_path_buf()]),
            opts(&["*.bmp"]),
            &root,
        );
        assert_eq!(names, set(&["link.bmp"]));
    }

    #[test]
    fn reports_unreadable_directories() {
        let root = tree();
        let locked = root.join("sub");
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();
        let readable = fs::read_dir(&locked).is_ok();
        let found = found(Source::Paths(vec![root.to_path_buf()]), opts(&[]));
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).unwrap();
        // Root can read the directory anyway.
        if !readable {
            let errors = found.iter().filter(|f| matches!(f, Found::Error));
            assert_eq!(errors.count(), 1);
            assert_eq!(found.into_iter().filter_map(image).count(), 2);
        }
    }

    #[test]
    fn reads_path_lists() {
        let collect = |input: &[u8], end, patterns: &[&str]| {
            let (tx, rx) = mpsc::channel();
            let globs = globs(patterns.iter().copied()).unwrap();
            read_list(Cursor::new(input.to_vec()), end, &globs, &tx);
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
    }

    #[test]
    fn only_directory_walks_filter() {
        let root = tree();
        let skipped = |paths, hidden, no_ignore| {
            let opts = WalkOpts {
                hidden,
                no_ignore,
                ..opts(&[])
            };
            candidates(Source::Paths(paths), opts).unwrap().skipped
        };
        let dir = || vec![root.to_path_buf()];
        assert_eq!(skipped(dir(), false, false), Some(Skipped::Both));
        assert_eq!(skipped(dir(), true, false), Some(Skipped::Ignored));
        assert_eq!(skipped(dir(), false, true), Some(Skipped::Hidden));
        assert_eq!(skipped(dir(), true, true), None);
        assert_eq!(skipped(vec![root.join("a.jpg")], false, false), None);
    }
}
