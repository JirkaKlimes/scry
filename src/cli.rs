use std::{
    convert::Infallible,
    fs, io,
    num::NonZeroUsize,
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
};

use clap::{ArgGroup, CommandFactory, Parser, error::ErrorKind};
use ignore::overrides::Override;
use jiff::civil::DateTime;
use regex::{Regex, RegexBuilder};
use regex_syntax::ast::{Ast, ClassSetItem, Visitor, parse::Parser as RegexParser, visit};
use rustix::fs::FileType;

use crate::{
    decode,
    embed::Prompt,
    meta::{self, MetaFilter, Range},
    output::{self, End, ListOpts, clean_path},
    search::{DateOrder, Order, Query, Rank, Scoring},
    walk::{self, Source, WalkOpts},
};

pub enum Mode {
    List(ListOpts),
    Scatter(End),
}

const DEFAULT_LIMIT: NonZeroUsize = NonZeroUsize::new(20).unwrap();

#[derive(Parser)]
#[command(
    name = "scry",
    version,
    about = "Find images by what they show, the text in them, or their metadata.",
    group(ArgGroup::new("rank").args(["query", "like"])),
    after_help = "\
RANGE is A..B, A.., ..B or A. Both ends are inclusive. A date (2024, 2024-06, 2024-06-15) covers
the whole period. An age (12h, 7d, 2w, 3mo, 1y) counts back from now: --date 7d.. is the last 7
days, ..1y is older than a year.

Examples:
  scry -q 'a dog in the snow' ~/Pictures
  scry -t 'INV-\\d{4}' ~/Documents
  scry -s --date 2024-07 ~/Pictures
  scry --order newest -n 1 -t error ~/Pictures/Screenshots

scry keeps models and a cache in ~/.cache/scry. Set SCRY_CACHE to use another directory.
Exit status: 0 if scry printed a path, 1 if not, 2 on error, 130 if you quit scatter mode."
)]
struct Args {
    /// Files or directories [default: paths on stdin if it is a pipe or a file, else .]
    #[arg(value_name = "PATH")]
    paths: Vec<PathBuf>,

    /// Rank by how well TEXT describes the image, in any language
    #[arg(short, long, value_name = "TEXT", help_heading = "Meaning")]
    query: Option<String>,

    /// Rank by how similar the image looks to IMAGE
    #[arg(short, long, value_name = "IMAGE", help_heading = "Meaning")]
    like: Option<PathBuf>,

    /// Drop images that are closer to TEXT than to the query (repeatable)
    #[arg(
        short = 'x',
        long = "not",
        value_name = "TEXT",
        requires = "query",
        conflicts_with = "like",
        help_heading = "Meaning"
    )]
    not: Vec<String>,

    /// Drop images that score below SCORE (see -S)
    #[arg(
        long,
        value_name = "SCORE",
        requires = "rank",
        allow_negative_numbers = true,
        help_heading = "Meaning"
    )]
    min: Option<f32>,

    /// Keep images whose text matches REGEX (repeatable, all must match)
    #[arg(short, long, value_name = "REGEX", help_heading = "Text in images")]
    text: Vec<String>,

    /// Ignore case in REGEX [default: only if it has no capital letter]
    #[arg(short, long, requires = "text", help_heading = "Text in images")]
    ignore_case: bool,

    /// Print the text after each path, or only the parts that match -t
    #[arg(
        short = 'o',
        long,
        conflicts_with = "scatter",
        help_heading = "Text in images"
    )]
    show_text: bool,

    /// Width in pixels
    #[arg(long, value_name = "RANGE", value_parser = meta::pixel_range, help_heading = "Metadata")]
    width: Option<Range<u32>>,

    /// Height in pixels
    #[arg(long, value_name = "RANGE", value_parser = meta::pixel_range, help_heading = "Metadata")]
    height: Option<Range<u32>>,

    /// Date the photo was taken (EXIF), else the time the file changed
    #[arg(long, value_name = "RANGE", value_parser = meta::time_range, help_heading = "Metadata")]
    date: Option<Range<DateTime>>,

    /// Camera make and model, for example 'fujifilm x-t4'
    #[arg(long, value_name = "REGEX", value_parser = camera, help_heading = "Metadata")]
    camera: Option<Regex>,

    /// Only images with a GPS location
    #[arg(long, conflicts_with = "no_gps", help_heading = "Metadata")]
    gps: bool,

    /// Only images without a GPS location
    #[arg(long, help_heading = "Metadata")]
    no_gps: bool,

    /// Include paths that match GLOB, or exclude them with a leading ! (repeatable)
    #[arg(short, long, value_name = "GLOB", help_heading = "Files")]
    glob: Vec<String>,

    /// Search hidden files and directories
    #[arg(long, help_heading = "Files")]
    hidden: bool,

    /// Do not read .gitignore and other ignore files
    #[arg(long, help_heading = "Files")]
    no_ignore: bool,

    /// Show the images one at a time over the terminal and pick by eye
    #[arg(short, long, help_heading = "Output")]
    scatter: bool,

    /// Sort by date, newest or oldest first
    #[arg(long, value_name = "ORDER", help_heading = "Output")]
    order: Option<DateOrder>,

    /// Print at most N paths, or all [default: 20 with -q or -l, unless --min or -s]
    #[arg(short = 'n', long, value_name = "N", value_parser = limit, help_heading = "Output")]
    limit: Option<Limit>,

    /// Print the score and a tab before each path
    #[arg(
        short = 'S',
        long,
        requires = "rank",
        conflicts_with = "scatter",
        help_heading = "Output"
    )]
    score: bool,

    /// Show a thumbnail above each path in the terminal
    #[arg(short, long, conflicts_with = "scatter", help_heading = "Output")]
    preview: bool,

    /// Separate paths with NUL instead of newline, on stdin and stdout
    #[arg(short = '0', long, help_heading = "Output")]
    null: bool,
}

#[derive(Clone, Copy)]
enum Limit {
    All,
    First(NonZeroUsize),
}

pub fn parse() -> (Query, Mode) {
    Args::parse().split().unwrap_or_else(|e| e.exit())
}

impl Args {
    fn split(self) -> Result<(Query, Mode), clap::Error> {
        check_paths(&self.paths, self.query.as_deref())?;
        if let Some(like) = &self.like {
            check_like(like)?;
        }
        let text = regexes(&self.text, self.ignore_case)?;
        let globs = globs(&self.glob)?;
        let rank = rank(self.query, self.not, self.like);
        let order = order(rank, self.order, self.min, self.scatter)?;
        let end = if self.null { End::Nul } else { End::Newline };
        let query = Query {
            source: source(self.paths, end),
            walk: WalkOpts {
                hidden: self.hidden,
                no_ignore: self.no_ignore,
                globs,
            },
            meta: MetaFilter {
                width: self.width,
                height: self.height,
                date: self.date,
                camera: self.camera,
                gps: (self.gps || self.no_gps).then_some(self.gps),
            },
            text,
            show_text: self.show_text,
            limit: limit_or_default(self.limit, &order, self.scatter),
            order,
        };
        let mode = if self.scatter {
            Mode::Scatter(end)
        } else {
            Mode::List(ListOpts {
                score: self.score,
                preview: self.preview,
                end,
            })
        };
        Ok((query, mode))
    }
}

fn regexes(patterns: &[String], ignore_case: bool) -> Result<Vec<Regex>, clap::Error> {
    (patterns.iter())
        .map(|re| {
            RegexBuilder::new(re)
                .case_insensitive(ignore_case || !has_capital(re))
                .build()
                .map_err(|e| invalid(&format!("invalid value '{re}' for '--text <REGEX>': {e}")))
        })
        .collect()
}

fn globs(patterns: &[String]) -> Result<Override, clap::Error> {
    walk::globs(patterns.iter().map(String::as_str)).map_err(|e| match e {
        ignore::Error::Glob {
            glob: Some(glob),
            err,
        } => invalid(&format!(
            "invalid value '{glob}' for '--glob <GLOB>': {err}"
        )),
        e => invalid(&format!("invalid '--glob <GLOB>': {e}")),
    })
}

/// Clap allows -x only with -q, and -q only without -l.
fn rank(query: Option<String>, not: Vec<String>, like: Option<PathBuf>) -> Option<Rank> {
    match query {
        Some(query) => Some(Rank::Text {
            query: Prompt::new(&query),
            not: not.iter().map(|n| Prompt::new(n)).collect(),
        }),
        None => like.map(Rank::Like),
    }
}

/// `scatter` makes the best hits carry their embedding.
fn order(
    rank: Option<Rank>,
    date: Option<DateOrder>,
    min: Option<f32>,
    scatter: bool,
) -> Result<Order, clap::Error> {
    match (rank, date) {
        (Some(rank), None) => {
            if let Some(min) = min {
                check_min(&rank, min)?;
            }
            Ok(Order::Score(Scoring {
                rank,
                min,
                embeddings: scatter,
            }))
        }
        (Some(_), Some(_)) => Err(Args::command().error(
            ErrorKind::ArgumentConflict,
            "--order cannot sort the matches of -q or -l\n\n  \
             tip: choose the matches first, then sort them in a second scry:\n       \
             scry -q 'a dog' --min 0.01 ~/Pictures | scry --order newest",
        )),
        (None, Some(date)) => Ok(Order::Date(date)),
        (None, None) => Ok(Order::Found),
    }
}

fn source(paths: Vec<PathBuf>, end: End) -> Source {
    if !paths.is_empty() {
        Source::Paths(paths)
    } else if stdin_piped() {
        Source::Stdin(end)
    } else {
        Source::Paths(vec![".".into()])
    }
}

/// A ranked search prints the best few, unless it has a cutoff or shows a scatter.
fn limit_or_default(limit: Option<Limit>, order: &Order, scatter: bool) -> Option<NonZeroUsize> {
    match limit {
        Some(Limit::All) => None,
        Some(Limit::First(n)) => Some(n),
        None => {
            matches!(order, Order::Score(s) if s.min.is_none() && !scatter).then_some(DEFAULT_LIMIT)
        }
    }
}

fn invalid(msg: &str) -> clap::Error {
    Args::command().error(ErrorKind::ValueValidation, msg)
}

fn limit(s: &str) -> Result<Limit, String> {
    if s == "all" {
        return Ok(Limit::All);
    }
    s.parse()
        .map(Limit::First)
        .map_err(|_| "expected a number from 1 up, or all".into())
}

fn camera(re: &str) -> Result<Regex, regex::Error> {
    RegexBuilder::new(re).case_insensitive(true).build()
}

/// A score out of range drops every image, with no hint why.
fn check_min(rank: &Rank, min: f32) -> Result<(), clap::Error> {
    let (lo, flag) = match rank {
        Rank::Text { .. } => (0.0, "-q"),
        Rank::Like(_) => (-1.0, "-l"),
    };
    if (lo..=1.0).contains(&min) {
        return Ok(());
    }
    Err(invalid(&format!(
        "invalid value '{min}' for '--min <SCORE>': {flag} scores are from {lo} to 1\n\n  \
         tip: run the search once with -S to see the scores"
    )))
}

/// People who know grep type the query as the first argument. When the missing paths look like
/// words, the error shows the command they meant.
fn check_paths(paths: &[PathBuf], query: Option<&str>) -> Result<(), clap::Error> {
    let missing: Vec<&PathBuf> = paths
        .iter()
        .filter(|p| fs::metadata(p).is_err_and(|e| e.kind() == io::ErrorKind::NotFound))
        .collect();
    let Some(first) = missing.first() else {
        return Ok(());
    };
    let mut msg = format!("no such file or directory: '{}'", clean_path(first));
    if let Some(cmd) = meant(paths, &missing, query) {
        msg += &format!(
            "\n\n  tip: to search by meaning, put the words in quotes after -q:\n       {cmd}"
        );
    }
    Err(Args::command().error(ErrorKind::Io, msg))
}

/// None unless all missing paths look like words, and plain shell quotes can show each part.
fn meant(paths: &[PathBuf], missing: &[&PathBuf], query: Option<&str>) -> Option<String> {
    let word = |p: &&PathBuf| !p.as_os_str().as_bytes().contains(&b'/') && !decode::is_image(p);
    if !missing.iter().all(word) {
        return None;
    }
    let words: Vec<&str> = (query.into_iter().map(Some))
        .chain(missing.iter().map(|p| p.to_str()))
        .map(|word| word.filter(shows))
        .collect::<Option<_>>()?;
    let mut cmd = format!("scry -q {}", quoted(&words.join(" ")));
    for path in paths.iter().filter(|p| !missing.contains(p)) {
        cmd.push(' ');
        cmd.push_str(&quote(path.to_str().filter(shows)?));
    }
    Some(cmd)
}

/// True if the text shows in a terminal as it is.
fn shows(s: &&str) -> bool {
    !s.contains(output::control)
}

/// Checks the --like image now, not after the model download.
fn check_like(path: &Path) -> Result<(), clap::Error> {
    let problem = match fs::metadata(path) {
        Err(e) => e.to_string(),
        Ok(md) if md.is_dir() => "it is a directory".into(),
        Ok(md) if !md.is_file() => "it is not a regular file".into(),
        Ok(_) if decode::is_unsupported(path) => "scry cannot read this format yet".into(),
        Ok(_) if !decode::is_image(path) => "it is not an image file".into(),
        Ok(_) => return Ok(()),
    };
    Err(invalid(&format!(
        "invalid value '{}' for '--like <IMAGE>': {problem}",
        clean_path(path)
    )))
}

/// Quotes a path for a POSIX shell, unless it needs no quotes.
fn quote(s: &str) -> String {
    let plain = |c: char| c.is_alphanumeric() || "._-/+=:,@%".contains(c);
    if !s.is_empty() && s.chars().all(plain) {
        s.into()
    } else {
        quoted(s)
    }
}

fn quoted(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// Smart case, as in ripgrep and fd: a pattern ignores case unless it has a capital letter.
/// Escapes like `\S` and `\W` do not count.
fn has_capital(re: &str) -> bool {
    struct Capital(bool);

    impl Visitor for Capital {
        type Output = bool;
        type Err = Infallible;

        fn finish(self) -> Result<bool, Infallible> {
            Ok(self.0)
        }

        fn visit_pre(&mut self, ast: &Ast) -> Result<(), Infallible> {
            if let Ast::Literal(lit) = ast {
                self.0 |= lit.c.is_uppercase();
            }
            Ok(())
        }

        fn visit_class_set_item_pre(&mut self, item: &ClassSetItem) -> Result<(), Infallible> {
            match item {
                ClassSetItem::Literal(lit) => self.0 |= lit.c.is_uppercase(),
                ClassSetItem::Range(r) => {
                    self.0 |= r.start.c.is_uppercase() || r.end.c.is_uppercase()
                }
                _ => {}
            }
            Ok(())
        }
    }

    RegexParser::new()
        .parse(re)
        .is_ok_and(|ast| visit(&ast, Capital(false)) == Ok(true))
}

/// A terminal or /dev/null on stdin means that nothing was piped.
fn stdin_piped() -> bool {
    rustix::fs::fstat(io::stdin()).is_ok_and(|st| {
        matches!(
            FileType::from_raw_mode(st.st_mode),
            FileType::Fifo | FileType::Socket | FileType::RegularFile
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::Temp;

    fn parse(args: &[&str]) -> Result<(Query, Mode), clap::Error> {
        Args::try_parse_from([&["scry"], args].concat())?.split()
    }

    fn err(args: &[&str]) -> ErrorKind {
        parse(args).err().expect("should fail").kind()
    }

    #[test]
    fn rules() {
        let file = Temp::file("like.jpg", b"");
        let like = file.to_str().unwrap();
        assert_eq!(
            err(&["-q", "cat", "-l", "a.jpg"]),
            ErrorKind::ArgumentConflict
        );
        assert_eq!(err(&["-x", "dog"]), ErrorKind::MissingRequiredArgument);
        assert_eq!(
            err(&["-l", "a.jpg", "-x", "dog"]),
            ErrorKind::ArgumentConflict
        );
        assert_eq!(err(&["--min", "0.5"]), ErrorKind::MissingRequiredArgument);
        assert_eq!(err(&["-i"]), ErrorKind::MissingRequiredArgument);
        assert_eq!(err(&["-S"]), ErrorKind::MissingRequiredArgument);
        assert_eq!(err(&["--gps", "--no-gps"]), ErrorKind::ArgumentConflict);
        assert_eq!(err(&["-s", "-p"]), ErrorKind::ArgumentConflict);
        assert_eq!(err(&["-s", "-o"]), ErrorKind::ArgumentConflict);
        assert_eq!(err(&["-s", "-S", "-q", "cat"]), ErrorKind::ArgumentConflict);
        assert_eq!(err(&["--width", "big"]), ErrorKind::ValueValidation);
        assert_eq!(err(&["--date", "7d"]), ErrorKind::ValueValidation);
        assert_eq!(err(&["-t", "a[b"]), ErrorKind::ValueValidation);
        assert_eq!(err(&["--camera", "("]), ErrorKind::ValueValidation);
        assert_eq!(err(&["--order", "new"]), ErrorKind::InvalidValue);
        assert_eq!(err(&["-g", "a[b"]), ErrorKind::ValueValidation);
        assert!(parse(&["-q", "cat", "-x", "dog", "-x", "fox", "--min", "0.1"]).is_ok());
        assert!(parse(&["-l", like, "-g", "Screenshots/*"]).is_ok());
    }

    #[test]
    fn like_is_checked_before_the_search() {
        let message = |args: &[&str]| parse(args).err().unwrap().to_string();
        assert!(message(&["-l", "typo.jpg"]).contains("for '--like <IMAGE>': No such file"));
        assert!(message(&["-l", "src"]).contains("it is a directory"));
        assert!(message(&["-l", "Cargo.toml"]).contains("not an image file"));
        assert_eq!(err(&["-l", "Cargo.toml"]), ErrorKind::ValueValidation);
    }

    #[test]
    fn order_is_one_per_run() {
        let file = Temp::file("like.jpg", b"");
        let like = file.to_str().unwrap();
        let order = |args: &[&str]| parse(args).unwrap().0.order;
        assert!(matches!(order(&[]), Order::Found));
        assert!(matches!(order(&["-t", "x"]), Order::Found));
        assert!(matches!(
            order(&["--order", "newest"]),
            Order::Date(DateOrder::Newest)
        ));
        assert!(matches!(
            order(&["--order", "oldest", "-s"]),
            Order::Date(DateOrder::Oldest)
        ));
        assert!(matches!(order(&["-q", "cat"]), Order::Score(_)));
        assert_eq!(
            err(&["-q", "cat", "--order", "newest"]),
            ErrorKind::ArgumentConflict
        );
        assert_eq!(
            err(&["-l", like, "--order", "oldest"]),
            ErrorKind::ArgumentConflict
        );
    }

    #[test]
    fn min_is_in_range() {
        let file = Temp::file("like.jpg", b"");
        let like = file.to_str().unwrap();
        assert_eq!(
            err(&["-q", "cat", "--min", "5"]),
            ErrorKind::ValueValidation
        );
        assert_eq!(
            err(&["-q", "cat", "--min", "-0.1"]),
            ErrorKind::ValueValidation
        );
        assert_eq!(
            err(&["-l", like, "--min", "-2"]),
            ErrorKind::ValueValidation
        );
        assert!(parse(&["-l", like, "--min", "-0.5", "-S"]).is_ok());
        assert!(parse(&["-q", "cat", "--min", "5.8e-3"]).is_ok());
        assert!(parse(&["-q", "cat", "--min", "1"]).is_ok());
    }

    fn limit(args: &[&str]) -> Option<usize> {
        parse(args).unwrap().0.limit.map(NonZeroUsize::get)
    }

    #[test]
    fn default_limit() {
        let file = Temp::file("like.jpg", b"");
        let like = file.to_str().unwrap();
        assert_eq!(limit(&["-q", "cat"]), Some(20));
        assert_eq!(limit(&["-l", like]), Some(20));
        assert_eq!(limit(&["-q", "cat", "-n", "5"]), Some(5));
        assert_eq!(limit(&["-q", "cat", "-n", "all"]), None);
        assert_eq!(limit(&["-q", "cat", "--min", "0.2"]), None);
        assert_eq!(limit(&["-q", "cat", "-s"]), None);
        assert_eq!(limit(&["-t", "x"]), None);
        assert_eq!(limit(&["--order", "newest"]), None);
        assert_eq!(limit(&["-n", "3"]), Some(3));
        assert_eq!(err(&["-n", "0"]), ErrorKind::ValueValidation);
        assert_eq!(err(&["-n", "some"]), ErrorKind::ValueValidation);
    }

    #[test]
    fn builds_query() {
        let (q, mode) = parse(&[
            "-t",
            "Total",
            "-i",
            "--camera",
            "canon",
            "--no-gps",
            "-g",
            "!*.png",
            "-0",
            "-S",
            "-q",
            "Receipt",
            "src",
            "Cargo.toml",
        ])
        .unwrap();
        assert!(q.text[0].is_match("TOTAL: 5"));
        assert!(q.meta.camera.as_ref().unwrap().is_match("CANON EOS"));
        assert_eq!(q.meta.gps, Some(false));
        assert!(q.meta.needs_meta());
        assert!(q.walk.globs.matched("x/a.PNG", false).is_ignore());
        assert!(
            matches!(q.source, Source::Paths(p) if p == [PathBuf::from("src"), "Cargo.toml".into()])
        );
        assert!(matches!(
            q.order,
            Order::Score(Scoring { rank: Rank::Text { query, not }, min: None, embeddings: false })
                if query.as_str() == "receipt" && not.is_empty()
        ));
        assert!(matches!(mode, Mode::List(o) if o.end == End::Nul && o.score && !o.preview));

        let (q, mode) = parse(&["-t", "Total", "-s", "-0", "src"]).unwrap();
        assert!(!q.text[0].is_match("TOTAL"));
        assert_eq!(q.meta.gps, None);
        assert!(matches!(q.order, Order::Found) && !q.meta.needs_meta());
        assert!(matches!(mode, Mode::Scatter(End::Nul)));
        let (q, _) = parse(&["-q", "cat", "-s"]).unwrap();
        assert!(matches!(
            q.order,
            Order::Score(Scoring {
                embeddings: true,
                ..
            })
        ));
    }

    #[test]
    fn smart_case() {
        let (q, _) = parse(&["-t", "total", "-t", r"\S+\W\D", "-t", "Total"]).unwrap();
        assert!(q.text[0].is_match("TOTAL"));
        assert!(q.text[1].is_match("AB.C"));
        assert!(!q.text[2].is_match("TOTAL"));
        for (re, capital) in [
            ("invoice", false),
            (r"inv-\d{4}", false),
            (r"\p{Lu}", false),
            ("(?P<Name>x)", false),
            ("Invoice", true),
            ("[A-Z]", true),
            ("[xY]", true),
            ("ČAJ", true),
        ] {
            assert_eq!(has_capital(re), capital, "{re}");
        }
    }

    #[test]
    fn missing_paths_teach_q() {
        let message = |args: &[&str]| parse(args).err().unwrap().to_string();
        let tip = message(&["dog", "src"]);
        assert!(tip.contains("no such file or directory: 'dog'"), "{tip}");
        assert!(tip.contains("scry -q 'dog' src"), "{tip}");
        assert!(message(&["-q", "red", "car", "src"]).contains("scry -q 'red car' src"));
        assert!(message(&["it's"]).contains(r"scry -q 'it'\''s'"));
        assert!(!message(&["missing/dir"]).contains("tip:"));
        assert!(!message(&["photo.jpg"]).contains("tip:"));
        assert_eq!(err(&["dog"]), ErrorKind::Io);
        let escape = message(&["\x1b]52;c;eA==\x07"]);
        assert!(
            escape.contains("'?]52;c;eA==?'") && !escape.contains("tip:"),
            "{escape}"
        );
    }

    #[test]
    fn help_renders() {
        Args::command().debug_assert();
    }
}
