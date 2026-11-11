use std::{
    fs::{self, File, TryLockError},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, ensure};
use indicatif::{HumanBytes, ProgressBar, ProgressStyle};
use rten::Model;
use sha2::{Digest, Sha256};

use crate::{cache, output::clean_path};

pub struct Asset {
    /// For people, like "SigLIP 2 image model".
    pub label: &'static str,
    pub name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
}

pub fn fetch(asset: &Asset) -> Result<PathBuf> {
    let dir = cache::dir()?.join("models");
    let path = dir.join(asset.name);
    if path.exists() {
        return Ok(path);
    }
    fs::create_dir_all(&dir)?;
    // The lock makes a concurrent scry wait for this download. An interrupted download leaves
    // only this file, and the next run starts it again.
    let part = dir.join(format!("{}.part", asset.name));
    let mut file = File::options()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&part)?;
    match file.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => {
            eprintln!(
                "scry: waiting for another scry to download the {}",
                asset.label
            );
            file.lock()?;
        }
        Err(TryLockError::Error(e)) => return Err(e.into()),
    }
    if path.exists() {
        let _ = fs::remove_file(&part);
    } else {
        file.set_len(0)?;
        download(asset, &mut file, &dir)
            .with_context(|| format!("cannot download the {}", asset.label))?;
        fs::rename(&part, &path)?;
    }
    Ok(path)
}

/// A model that fails to load was most likely damaged after its download.
pub fn load(path: &Path) -> Result<Model> {
    Model::load_file(path).with_context(|| cannot_load(path))
}

pub fn cannot_load(path: &Path) -> String {
    format!(
        "cannot load {} (delete it, and scry downloads it again)",
        clean_path(path)
    )
}

fn download(asset: &Asset, file: &mut File, dir: &Path) -> Result<()> {
    let mut resp = agent().get(asset.url).call().context(asset.url)?;
    let bar = progress(asset, resp.body().content_length(), dir)?;
    let sha256 = copy_hashed(bar.wrap_read(resp.body_mut().as_reader()), file)?;
    file.sync_all()?;
    bar.finish_and_clear();
    ensure!(
        sha256 == asset.sha256,
        "checksum mismatch for {}: expected {}, got {sha256}",
        asset.name,
        asset.sha256
    );
    Ok(())
}

/// Without timeouts, a stalled connection hangs scry and every scry that waits for the lock.
fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .https_only(true)
        .timeout_resolve(Some(Duration::from_secs(30)))
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_per_read(Some(Duration::from_secs(60)))
        .build()
        .into()
}

/// Also prints a line, because without a terminal the bar does not show. A log then tells why
/// the run is slow.
fn progress(asset: &Asset, size: Option<u64>, dir: &Path) -> Result<ProgressBar> {
    let size_text = size
        .map(|n| format!(" ({})", HumanBytes(n)))
        .unwrap_or_default();
    eprintln!(
        "scry: downloading the {}{size_text} to {}. scry does this only once.",
        asset.label,
        clean_path(dir)
    );
    let style = ProgressStyle::with_template("{msg} {bar:30} {bytes}/{total_bytes} {eta}")?;
    Ok(ProgressBar::new(size.unwrap_or(0))
        .with_style(style)
        .with_message(asset.label))
}

fn copy_hashed(mut from: impl Read, to: &mut impl Write) -> io::Result<String> {
    let mut hasher = Sha256::new();
    let mut buf = vec![0; 1 << 16];
    loop {
        let n = from.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        to.write_all(&buf[..n])?;
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}
