use std::{
    convert::Infallible,
    fs, io,
    num::NonZeroUsize,
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
};

use clap::{ArgGroup, CommandFactory, Parser, ValueEnum, error::ErrorKind};
use ignore::overrides::Override;
use jiff::civil::DateTime;
use regex::{Regex, RegexBuilder};
use regex_syntax::ast::{Ast, ClassSetItem, Visitor, parse::Parser as RegexParser, visit};
use rustix::fs::FileType;

use crate::{
    cache, decode,
    embed::{self, Prompt},
    meta::{self, MetaFilter, Range},
    output::{self, End, ListOpts, Out, Shown, clean_path},
    pick::{Bind, View},
    search::{By, Dir, Field, Order, Query, Rank, Scoring},
    walk::{self, Source, WalkOpts, Watch},
};

pub enum Mode {
    List(ListOpts),
    Pick {
        view: View,
        binds: Vec<Bind>,
        end: End,
    },
}

const DEFAULT_LIMIT: NonZeroUsize = NonZeroUsize::new(20).unwrap();

#[derive(Parser)]
#[command(
    name = "scry",
    version,
    about = "Find images by what they show, the text in them, or their metadata.",
    max_term_width = 100,
    group(ArgGroup::new("rank").args(["query", "like"])),
    group(ArgGroup::new("view").args(["scatter", "grid"])),
    after_help = format!("\
RANGE is A..B, A.., ..B or A. Both ends are inclusive. A date covers its last part: 2024 is a \
year, 2024-06-15T14 is an hour. An offset (Z, +02:00) turns a date into local time. An age (15m, \
1h30m, 7d, 3mo, 1y) counts back from now: --date 7d is the last 7 days, ..1y is older than a \
year. The words today and yesterday mean calendar days.

Environment:
  SCRY_CACHE   Directory for the models and the cache
               [default: {cache}]
  SCRY_DEVICE  Where the image model runs [default: auto]
               auto: {gpu} ({api}), else the CPU
               gpu: any GPU, also an integrated one. An error if there is none
               cpu: only the CPU

Exit status: 0 if scry printed a path, else 1, and with --json 0 if the run ends. 2 on error. 130 \
if you quit scatter or grid mode. Files that scry cannot read do not change it.

Examples:
  scry -q 'a dog in the snow' ~/Pictures
  scry -t 'INV-\\d{{4}}' ~/Documents
  scry --grid --date 2024-07 ~/Pictures
  scry --grid --bind 'ctrl-d:cp -- \"$@\" ~/picked/' ~/Pictures
  scry --sort date -n 1 -t error ~/Pictures/Screenshots
  scry --watch=new -t error ~/Pictures/Screenshots",
        cache = cache::DEFAULT_DIR,
        gpu = embed::gpu::AUTO,
        api = embed::gpu::API,
    )
)]
struct Args {
    /// Files or directories, or - for a list of paths on stdin [default: the list on stdin if it
    /// is a pipe or a file, else .]
    #[arg(value_name = "PATH")]
    paths: Vec<PathBuf>,

    /// Rank by how well TEXT describes the image, in any language
    #[arg(short, long, value_name = "TEXT", value_parser = words, help_heading = "Meaning")]
    query: Option<String>,

    /// Rank by how similar the image looks to IMAGE
    #[arg(short, long, value_name = "IMAGE", help_heading = "Meaning")]
    like: Option<PathBuf>,

    /// Drop images that are closer to TEXT than to the query (repeatable)
    #[arg(
        short = 'x',
        long = "not",
        value_name = "TEXT",
        value_parser = words,
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

    /// Ignore case in -t [default: only if the pattern has no capital letter]
    #[arg(short, long, requires = "text", help_heading = "Text in images")]
    ignore_case: bool,

    /// Print the text after each path, or only the parts that match -t
    #[arg(
        short = 'o',
        long,
        conflicts_with = "view",
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

    /// Keep watching the directories, and search each new image. With =new, skip the images
    /// that exist now
    #[arg(
        long,
        value_name = "new",
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "all",
        hide_possible_values = true,
        help_heading = "Files"
    )]
    watch: Option<Watch>,

    /// Show the images one at a time over the terminal and pick by eye
    #[arg(short, long, help_heading = "Output")]
    scatter: bool,

    /// Pick images by eye from pages of thumbnails
    #[arg(long, help_heading = "Output")]
    grid: bool,

    /// In scatter or grid mode, run COMMAND when you press KEY (ctrl-a to ctrl-z, alt-a to alt-z,
    /// f1 to f12). COMMAND gets the marked images, or else the selected one, as "$@" (repeatable)
    #[arg(
        long,
        value_name = "KEY:COMMAND",
        requires = "view",
        help_heading = "Output"
    )]
    bind: Vec<Bind>,

    /// Sort by KEY: score (best first), date (newest first) or path (A to Z) [default: score with -q
    /// or -l, else the order scry finds the images]
    #[arg(long, value_name = "KEY", hide_possible_values = true, help_heading = "Output")]
    sort: Option<SortKey>,

    /// Reverse the order
    #[arg(short, long, help_heading = "Output")]
    reverse: bool,

    /// Print at most N paths (a number, or all) [default: 20 with -q or -l, unless --min, -s,
    /// --grid or --watch]
    #[arg(short = 'n', long, value_name = "N", value_parser = limit, help_heading = "Output")]
    limit: Option<Limit>,

    /// Print the score and a tab before each path
    #[arg(
        short = 'S',
        long,
        requires = "rank",
        conflicts_with = "view",
        help_heading = "Output"
    )]
    score: bool,

    /// Show a thumbnail above each path in the terminal
    #[arg(short, long, conflicts_with = "view", help_heading = "Output")]
    preview: bool,

    /// Separate paths with NUL instead of newline, on stdin and stdout
    #[arg(short = '0', long, help_heading = "Output")]
    null: bool,

    /// Print JSON Lines for programs: the progress, each image with its metadata, and the end
    #[arg(long, conflicts_with_all = ["preview", "score"], help_heading = "Output")]
    json: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum SortKey {
    Score,
    Date,
    Path,
}

#[derive(Clone, Copy)]
enum Limit {
    All,
    First(NonZeroUsize),
}

pub fn parse() -> Result<(Query, Mode), clap::Error> {
    Args::try_parse()?.split()
}

/// The message of a usage error, without the usage and help hints that clap adds. clap also
/// tells to put a value that starts with - after `--`, but scry reads the arguments after `--` as
/// paths.
pub fn message(e: &clap::Error) -> String {
    let text = e.to_string();
    let text = text.strip_prefix("error: ").unwrap_or(&text);
    (text.split("\n\n"))
        .take_while(|part| !part.starts_with("Usage:") && !part.starts_with("For more information"))
        .filter(|part| !part.starts_with("  tip: to pass '"))
        .collect::<Vec<_>>()
        .join("\n\n")
        .trim_end()
        .to_owned()
}

impl Args {
    fn split(self) -> Result<(Query, Mode), clap::Error> {
        // A program that starts scry often leaves stdin open, and scry would wait for it.
        if self.json && self.paths.is_empty() {
            return Err(Args::command().error(
                ErrorKind::MissingRequiredArgument,
                "--json needs a PATH, or - to read the paths on stdin",
            ));
        }
        check_paths(&self.paths, self.query.as_deref())?;
        if let Some(like) = &self.like {
            check_like(like)?;
        }
        let text = regexes(&self.text, self.ignore_case)?;
        let globs = globs(&self.glob)?;
        let view = match (self.scatter, self.grid) {
            (true, _) => Some(View::Scatter),
            (_, true) => Some(View::Grid),
            _ => None,
        };
        let rank = rank(self.query, self.not, self.like);
        if let (Some(_), Some(rank), None) = (self.watch, &rank, self.min) {
            return Err(Args::command().error(
                ErrorKind::MissingRequiredArgument,
                format!(
                    "--watch with {} needs --min: a new image goes out when its score is high enough",
                    rank.flag()
                ),
            ));
        }
        let order = order(rank, self.sort, self.reverse, self.min, self.limit, view)?;
        let end = if self.null { End::Nul } else { End::Newline };
        let source = source(self.paths, end);
        if self.watch.is_some() {
            check_watch(&source)?;
        }
        let query = Query {
            source,
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
            show_text: self.show_text,
            describe: self.json,
            limit: limit_or_default(self.limit, &order, view.is_some() || self.watch.is_some()),
            watch: self.watch,
            order,
            text,
        };
        let out = if self.json {
            Out::Json {
                matches: !query.text.is_empty(),
            }
        } else {
            let shown = match (self.show_text, query.text.is_empty()) {
                (false, _) => Shown::Nothing,
                (true, true) => Shown::Text,
                (true, false) => Shown::Matches,
            };
            Out::Lines {
                score: self.score,
                preview: self.preview,
                shown,
                end,
            }
        };
        let mode = match view {
            Some(view) => Mode::Pick {
                view,
                binds: self.bind,
                end,
            },
            None => Mode::List(ListOpts {
                out,
                cut: query
                    .limit
                    .filter(|_| self.limit.is_none())
                    .map(NonZeroUsize::get),
            }),
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

/// A picker makes the hits carry their embedding, so that it can rank them again.
fn order(
    rank: Option<Rank>,
    sort: Option<SortKey>,
    reverse: bool,
    min: Option<f32>,
    limit: Option<Limit>,
    view: Option<View>,
) -> Result<Order, clap::Error> {
    let dir = if reverse { Dir::Reverse } else { Dir::Natural };
    let Some(rank) = rank else {
        return match sort {
            Some(SortKey::Score) => Err(conflict("--sort score needs -q or -l")),
            Some(SortKey::Date) => Ok(Order::Sorted(Field::Date, dir)),
            Some(SortKey::Path) => Ok(Order::Sorted(Field::Path, dir)),
            None if reverse => Err(conflict(
                "-r reverses the order of --sort, -q or -l\n\n  \
                 tip: scry always searches subdirectories",
            )),
            None => Ok(Order::Found),
        };
    };
    if let Some(min) = min {
        check_min(&rank, min)?;
    }
    let flag = rank.flag();
    let by = match sort {
        None | Some(SortKey::Score) => By::Score,
        Some(SortKey::Date) => By::Field(Field::Date),
        Some(SortKey::Path) => By::Field(Field::Path),
    };
    // A list without -n prints the best 20. Pickers and -n all show every image.
    let counted = match limit {
        Some(Limit::First(_)) => true,
        Some(Limit::All) => false,
        None => view.is_none(),
    };
    if by != By::Score && min.is_none() && !counted {
        return Err(conflict(&format!(
            "--sort sorts the best matches of {flag}, so it needs a count: -n N or --min SCORE"
        )));
    }
    if matches!(view, Some(View::Scatter)) && by == By::Score && reverse {
        return Err(conflict(&format!(
            "scatter mode shows the matches of {flag} in a random order, so -r does nothing"
        )));
    }
    let scoring = Scoring {
        rank,
        min,
        embeddings: view.is_some(),
    };
    Ok(Order::Ranked(scoring, by, dir))
}

fn source(paths: Vec<PathBuf>, end: End) -> Source {
    let (dash, mut paths): (Vec<_>, Vec<_>) = paths.into_iter().partition(|p| is_stdin(p));
    let stdin = !dash.is_empty() || paths.is_empty() && stdin_piped();
    if paths.is_empty() && !stdin {
        paths.push(".".into());
    }
    Source {
        paths,
        stdin: stdin.then_some(end),
    }
}

fn is_stdin(path: &Path) -> bool {
    path.as_os_str() == "-"
}

/// --watch reads directories, and new images come only from there.
fn check_watch(source: &Source) -> Result<(), clap::Error> {
    if source.stdin.is_some() {
        return Err(conflict(
            "--watch reads new images from directories, not from stdin",
        ));
    }
    if !source.paths.iter().any(|p| p.is_dir()) {
        return Err(Args::command().error(
            ErrorKind::MissingRequiredArgument,
            "--watch needs a directory",
        ));
    }
    Ok(())
}

/// A ranked search prints the best few, unless it has a cutoff, you pick by eye, or it watches.
fn limit_or_default(limit: Option<Limit>, order: &Order, open: bool) -> Option<NonZeroUsize> {
    match limit {
        Some(Limit::All) => None,
        Some(Limit::First(n)) => Some(n),
        None => matches!(order, Order::Ranked(s, ..) if s.min.is_none() && !open)
            .then_some(DEFAULT_LIMIT),
    }
}

fn invalid(msg: &str) -> clap::Error {
    Args::command().error(ErrorKind::ValueValidation, msg)
}

fn conflict(msg: &str) -> clap::Error {
    Args::command().error(ErrorKind::ArgumentConflict, msg)
}

/// An empty text ranks by nothing, with no hint why.
fn words(s: &str) -> Result<String, String> {
    if s.trim().is_empty() {
        return Err("the text is empty".into());
    }
    Ok(s.into())
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
    let lo = match rank {
        Rank::Text { .. } => 0.0,
        Rank::Like(_) => -1.0,
    };
    let flag = rank.flag();
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
    let missing: Vec<&PathBuf> = (paths.iter())
        .filter(|p| !is_stdin(p))
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
        Err(e) if e.kind() == io::ErrorKind::NotFound => "no such file or directory".into(),
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
        assert_eq!(err(&["--grid", "-s"]), ErrorKind::ArgumentConflict);
        assert_eq!(
            err(&["--bind", "ctrl-d:x"]),
            ErrorKind::MissingRequiredArgument
        );
        assert_eq!(err(&["-s", "--bind", "d:x"]), ErrorKind::ValueValidation);
        assert_eq!(err(&["--grid", "-p"]), ErrorKind::ArgumentConflict);
        assert_eq!(err(&["--grid", "-o"]), ErrorKind::ArgumentConflict);
        assert_eq!(
            err(&["--grid", "-S", "-q", "cat"]),
            ErrorKind::ArgumentConflict
        );
        assert_eq!(err(&["-q", " "]), ErrorKind::ValueValidation);
        assert_eq!(err(&["-q", "cat", "-x", ""]), ErrorKind::ValueValidation);
        assert_eq!(err(&["--width", "big"]), ErrorKind::ValueValidation);
        assert_eq!(err(&["--date", "7"]), ErrorKind::ValueValidation);
        assert!(parse(&["--date", "7d"]).is_ok());
        assert_eq!(err(&["-t", "a[b"]), ErrorKind::ValueValidation);
        assert_eq!(err(&["--camera", "("]), ErrorKind::ValueValidation);
        assert_eq!(err(&["-g", "a[b"]), ErrorKind::ValueValidation);
        assert!(parse(&["-q", "cat", "-x", "dog", "-x", "fox", "--min", "0.1"]).is_ok());
        assert!(parse(&["-l", like, "-g", "Screenshots/*"]).is_ok());
    }

    #[test]
    fn like_is_checked_before_the_search() {
        let message = |args: &[&str]| parse(args).err().unwrap().to_string();
        assert!(message(&["-l", "typo.jpg"]).contains("for '--like <IMAGE>': no such file"));
        assert!(message(&["-l", "src"]).contains("it is a directory"));
        assert!(message(&["-l", "Cargo.toml"]).contains("not an image file"));
        assert_eq!(err(&["-l", "Cargo.toml"]), ErrorKind::ValueValidation);
    }

    #[test]
    fn sort_rules() {
        let file = Temp::file("like.jpg", b"");
        let like = file.to_str().unwrap();
        let order = |args: &[&str]| parse(args).unwrap().0.order;
        assert!(matches!(order(&[]), Order::Found));
        assert!(matches!(order(&["-t", "x"]), Order::Found));
        assert!(matches!(
            order(&["--sort", "date"]),
            Order::Sorted(Field::Date, Dir::Natural)
        ));
        assert!(matches!(
            order(&["--sort", "path", "-r", "-s"]),
            Order::Sorted(Field::Path, Dir::Reverse)
        ));
        assert!(matches!(
            order(&["-q", "cat"]),
            Order::Ranked(_, By::Score, Dir::Natural)
        ));
        assert!(matches!(
            order(&["-q", "cat", "-r"]),
            Order::Ranked(_, By::Score, Dir::Reverse)
        ));
        assert!(matches!(
            order(&["-q", "cat", "--sort", "date"]),
            Order::Ranked(_, By::Field(Field::Date), Dir::Natural)
        ));
        assert!(matches!(
            order(&["-l", like, "--sort", "path", "--grid", "-n", "9"]),
            Order::Ranked(_, By::Field(Field::Path), _)
        ));
        assert!(parse(&["-q", "cat", "--sort", "date", "--grid", "--min", "0.1"]).is_ok());
        assert_eq!(err(&["--sort", "score"]), ErrorKind::ArgumentConflict);
        assert_eq!(err(&["-r"]), ErrorKind::ArgumentConflict);
        assert_eq!(err(&["-r", "-t", "x"]), ErrorKind::ArgumentConflict);
        assert_eq!(
            err(&["-q", "cat", "--sort", "date", "-n", "all"]),
            ErrorKind::ArgumentConflict
        );
        assert_eq!(
            err(&["-q", "cat", "--sort", "date", "--grid"]),
            ErrorKind::ArgumentConflict
        );
        assert_eq!(err(&["-q", "cat", "-s", "-r"]), ErrorKind::ArgumentConflict);
        assert_eq!(err(&["--sort", "newest"]), ErrorKind::InvalidValue);
        assert_eq!(err(&["--order", "newest"]), ErrorKind::UnknownArgument);
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
        assert_eq!(limit(&["-q", "cat", "--grid"]), None);
        assert_eq!(limit(&["-t", "x"]), None);
        assert_eq!(limit(&["--sort", "date"]), None);
        assert_eq!(limit(&["-q", "cat", "--sort", "date"]), Some(20));
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
            q.source.paths == [PathBuf::from("src"), "Cargo.toml".into()]
                && q.source.stdin.is_none()
        );
        assert!(matches!(
            q.order,
            Order::Ranked(Scoring { rank: Rank::Text { query, not }, min: None, embeddings: false }, ..)
                if query.as_str() == "receipt" && not.is_empty()
        ));
        assert!(matches!(
            mode,
            Mode::List(ListOpts {
                out: Out::Lines {
                    score: true,
                    preview: false,
                    shown: Shown::Nothing,
                    end: End::Nul
                },
                cut: Some(20),
            })
        ));

        let (q, mode) = parse(&["-t", "Total", "-s", "-0", "src"]).unwrap();
        assert!(!q.text[0].is_match("TOTAL"));
        assert_eq!(q.meta.gps, None);
        assert!(matches!(q.order, Order::Found) && !q.meta.needs_meta());
        assert!(matches!(
            mode,
            Mode::Pick { view: View::Scatter, binds, end: End::Nul } if binds.is_empty()
        ));
        let (q, _) = parse(&["-q", "cat", "-s"]).unwrap();
        assert!(matches!(
            q.order,
            Order::Ranked(
                Scoring {
                    embeddings: true,
                    ..
                },
                ..
            )
        ));
        let (q, mode) = parse(&["-q", "cat", "--grid", "--bind", "ctrl-d:rm \"$@\""]).unwrap();
        assert!(matches!(
            mode,
            Mode::Pick { view: View::Grid, binds, end: End::Newline } if binds[0].name == "ctrl-d"
        ));
        assert!(matches!(
            q.order,
            Order::Ranked(
                Scoring {
                    embeddings: true,
                    ..
                },
                ..
            )
        ));
        let (q, _) = parse(&["-q", "cat"]).unwrap();
        assert!(matches!(
            q.order,
            Order::Ranked(
                Scoring {
                    embeddings: false,
                    ..
                },
                ..
            )
        ));
    }

    #[test]
    fn json_output() {
        let (q, mode) = parse(&["--json", "-t", "x", "-o", "src"]).unwrap();
        assert!(q.describe && q.show_text);
        assert!(matches!(
            mode,
            Mode::List(ListOpts {
                out: Out::Json { matches: true },
                cut: None
            })
        ));
        assert_eq!(err(&["--json"]), ErrorKind::MissingRequiredArgument);
        assert!(parse(&["--json", "-"]).is_ok());
        assert_eq!(err(&["--json", "-p", "src"]), ErrorKind::ArgumentConflict);
        let e = Args::try_parse_from(["scry", "--order", "newest"])
            .err()
            .unwrap();
        assert_eq!(message(&e), "unexpected argument '--order' found");
        let e = Args::try_parse_from(["scry", "-n", "0"]).err().unwrap();
        assert_eq!(
            message(&e),
            "invalid value '0' for '--limit <N>': expected a number from 1 up, or all"
        );
        let e = Args::try_parse_from(["scry", "--hiden"]).err().unwrap();
        assert_eq!(
            message(&e),
            "unexpected argument '--hiden' found\n\n  tip: a similar argument exists: '--hidden'"
        );
    }

    #[test]
    fn watch_rules() {
        let (q, _) = parse(&["--watch", "src"]).unwrap();
        assert_eq!(q.watch, Some(Watch::All));
        let (q, _) = parse(&["--watch=new", "-q", "cat", "--min", "0.1", "src"]).unwrap();
        assert_eq!((q.watch, q.limit), (Some(Watch::New), None));
        assert_eq!(
            err(&["--watch", "-q", "cat", "src"]),
            ErrorKind::MissingRequiredArgument
        );
        assert_eq!(
            err(&["--watch", "Cargo.toml"]),
            ErrorKind::MissingRequiredArgument
        );
        assert_eq!(err(&["--watch", "-", "src"]), ErrorKind::ArgumentConflict);
        assert_eq!(err(&["--watch=old", "src"]), ErrorKind::InvalidValue);
        let (q, _) = parse(&["--watch", "src", "Cargo.toml"]).unwrap();
        assert_eq!(q.source.paths.len(), 2);
    }

    #[test]
    fn dash_reads_stdin() {
        let source = |args: &[&str]| parse(args).unwrap().0.source;
        let s = source(&["-", "src"]);
        assert_eq!(
            (s.paths, s.stdin),
            (vec![PathBuf::from("src")], Some(End::Newline))
        );
        let s = source(&["-0", "-"]);
        assert_eq!((s.paths, s.stdin), (vec![], Some(End::Nul)));
        assert!(source(&["src"]).stdin.is_none());
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

    #[test]
    fn help_names_only_this_platform() {
        let help = Args::command().render_long_help().to_string();
        let other = if cfg!(target_os = "macos") {
            ["Linux", "Vulkan", "XDG"]
        } else {
            ["macOS", "Metal", "Library"]
        };
        for word in other {
            assert!(!help.contains(word), "{word} in the help");
        }
    }
}
