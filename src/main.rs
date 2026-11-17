mod cache;
mod cli;
mod decode;
mod embed;
mod meta;
mod models;
mod ocr;
mod output;
mod pick;
mod report;
mod search;
#[cfg(test)]
mod testing;
mod walk;

use std::{
    any::Any,
    cell::Cell,
    env, io,
    panic::{self, AssertUnwindSafe},
    process::ExitCode,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering::Relaxed},
    },
    thread,
};

use anyhow::{Result, anyhow};
use signal_hook::{
    consts::{SIGHUP, SIGINT, SIGTERM},
    iterator::Signals,
    low_level::emulate_default_handler,
};

use cache::Cache;
use cli::Mode;
use pick::Exit;
use report::{Fault, Format, Summary};

fn main() -> ExitCode {
    report::init(if wants_json() {
        Format::Json
    } else {
        Format::Text
    });
    let code = match run() {
        Ok(code) => code,
        Err(e) => {
            let fault = match e.downcast_ref::<models::Failed>() {
                Some(_) => Fault::Download,
                None => Fault::Other,
            };
            report::error(fault, &format!("{e:#}"));
            ExitCode::from(2)
        }
    };
    embed::gpu::stop();
    code
}

/// Before clap parses the arguments, so that its errors also come as JSON.
fn wants_json() -> bool {
    (env::args_os().skip(1))
        .take_while(|arg| arg != "--")
        .any(|arg| arg == "--json")
}

fn run() -> Result<ExitCode> {
    let (query, mode) = match cli::parse() {
        Ok(parsed) => parsed,
        // Help and version.
        Err(e) if !e.use_stderr() => e.exit(),
        Err(e) => {
            report::error(Fault::Usage, &cli::message(&e));
            return Ok(ExitCode::from(2));
        }
    };
    hide_guarded_panics();
    let search = search::run(query)?;
    let (progress, skipped) = (search.progress.clone(), search.skipped);
    let (matched, cut, quit) = match mode {
        Mode::List(opts) => (output::list(search, &opts.out)?, opts.cut, false),
        Mode::Pick { view, binds, end } => match pick::run(search, view, binds)? {
            Exit::Picked(hits) => {
                output::hits(&hits, end)?;
                (hits.len(), None, false)
            }
            Exit::Empty => (0, None, false),
            Exit::Aborted => (0, None, true),
        },
    };
    report::end(&Summary {
        progress: &progress,
        matched,
        skipped,
        cut,
    });
    Ok(match (quit, matched) {
        (true, _) => ExitCode::from(130),
        // A program reads the count in the end line.
        (false, 0) if !report::json() => ExitCode::FAILURE,
        _ => ExitCode::SUCCESS,
    })
}

/// A stopped scry keeps the work since the last cache commit and leaves the terminal as it was.
/// Then it dies by the same signal, so that a shell loop that gets Ctrl-C also stops.
pub fn stop_on_signals(cache: Arc<Cache>) -> Result<()> {
    let mut signals = Signals::new([SIGINT, SIGTERM, SIGHUP])?;
    thread::spawn(move || {
        if let Some(signal) = signals.forever().next() {
            // The picker draws nothing more while this holds stdout. A list holds stdout for the
            // whole run, so only the picker gets this lock.
            let _picker = pick::active().then(|| io::stdout().lock());
            cache.flush();
            pick::restore();
            let _ = emulate_default_handler(signal);
        }
    });
    Ok(())
}

thread_local! {
    static GUARDS: Cell<usize> = const { Cell::new(0) };
}

/// Guards that run now, in all threads.
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

/// Turns a panic in `f` into a quiet error, so a bug in a decoder or a model costs one file, not
/// the search.
pub fn guard<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    GUARDS.set(GUARDS.get() + 1);
    ACTIVE.fetch_add(1, Relaxed);
    let result = catch(f);
    ACTIVE.fetch_sub(1, Relaxed);
    GUARDS.set(GUARDS.get() - 1);
    result
}

/// Turns a panic in `f` into an error. The panic hook still prints it.
pub fn catch<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    panic::catch_unwind(AssertUnwindSafe(f))
        .unwrap_or_else(|payload| Err(anyhow!("crashed: {}", message(&*payload))))
}

fn message(payload: &(dyn Any + Send)) -> &str {
    (payload.downcast_ref::<&str>().copied())
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic")
}

/// Keeps the panics that `guard` catches off the terminal. Other panics print as before.
fn hide_guarded_panics() {
    let show = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        if !guarded() {
            show(info);
        }
    }));
}

/// True in a panic hook when `guard` catches the panic. A panic in a rayon job runs the hook on
/// the worker thread, not on the thread that waits in `guard`, so workers count as guarded while
/// any guard runs.
pub fn guarded() -> bool {
    GUARDS.get() > 0 || rayon::current_thread_index().is_some() && ACTIVE.load(Relaxed) > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_turns_panics_into_errors() {
        assert_eq!(guard(|| Ok(1)).unwrap(), 1);
        let err = guard::<()>(|| panic!("bad {}", "pixels")).unwrap_err();
        assert_eq!(err.to_string(), "crashed: bad pixels");
        assert!(!guarded());
    }
}
