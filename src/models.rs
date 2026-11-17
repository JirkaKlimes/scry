use std::{
    fmt,
    fs::{self, File, TryLockError},
    io::{self, Read, Seek, Write},
    path::{Path, PathBuf},
    thread,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use indicatif::DecimalBytes;
use rten::Model;
use sha2::{Digest, Sha256};
use ureq::http::StatusCode;

use crate::{cache, output::clean_path, report::Download};

pub struct Asset {
    /// For people, like "SigLIP 2 image model".
    pub label: &'static str,
    pub name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    /// In bytes. The progress shows it before the server answers.
    pub size: u64,
}

pub fn fetch(asset: &Asset) -> Result<PathBuf> {
    let dir = dir()?;
    let path = dir.join(asset.name);
    if path.exists() {
        return Ok(path);
    }
    fs::create_dir_all(&dir)?;
    // The lock makes a concurrent scry wait for this download. An interrupted download leaves
    // this file, and the next run continues it.
    let part = dir.join(format!("{}.part", asset.name));
    let mut file = File::options()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&part)?;
    let progress = Download::new(asset.label, asset.size);
    match file.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => {
            progress.note(&format!(
                "waiting for another scry to download the {}",
                asset.label
            ));
            wait_for_lock(&file, &progress)?;
        }
        Err(TryLockError::Error(e)) => return Err(e.into()),
    }
    if path.exists() {
        let _ = fs::remove_file(&part);
    } else {
        progress.note(&format!(
            "downloading the {} ({}) to {}. Later runs use this copy.",
            asset.label,
            DecimalBytes(asset.size),
            clean_path(&dir)
        ));
        download(asset, &mut file, &progress).context(Failed(asset.label))?;
        fs::rename(&part, &path)?;
    }
    progress.finish();
    Ok(path)
}

/// A failed download, which the JSON error line tells apart.
#[derive(Debug)]
pub struct Failed(&'static str);

impl fmt::Display for Failed {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "cannot download the {}", self.0)
    }
}

/// Shows the progress of the scry that holds the lock, from the size of its file.
fn wait_for_lock(file: &File, progress: &Download) -> Result<()> {
    loop {
        progress.set(file.metadata()?.len());
        thread::sleep(Duration::from_millis(100));
        match file.try_lock() {
            Ok(()) => return Ok(()),
            Err(TryLockError::WouldBlock) => {}
            Err(TryLockError::Error(e)) => return Err(e.into()),
        }
    }
}

/// True when `fetch` needs no download.
pub fn is_downloaded(asset: &Asset) -> bool {
    dir().is_ok_and(|dir| dir.join(asset.name).exists())
}

fn dir() -> Result<PathBuf> {
    Ok(cache::dir()?.join("models"))
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

/// Continues from the bytes that an earlier run wrote, if the server can send the rest.
fn download(asset: &Asset, file: &mut File, progress: &Download) -> Result<()> {
    let mut hasher = Sha256::new();
    let mut done = copy_hashed(&*file, io::sink(), &mut hasher)?;
    if done < asset.size {
        let mut resp = (agent().get(asset.url))
            .header("Range", format!("bytes={done}-"))
            .call()
            .context(asset.url)?;
        if resp.status() != StatusCode::PARTIAL_CONTENT {
            file.set_len(0)?;
            file.rewind()?;
            hasher = Sha256::new();
            done = 0;
        }
        progress.set(done);
        copy_hashed(
            progress.read(resp.body_mut().as_reader()),
            &*file,
            &mut hasher,
        )?;
        file.sync_all()?;
    }
    let sha256 = hex(&hasher.finalize());
    if sha256 != asset.sha256 {
        // The next run starts again from zero.
        file.set_len(0)?;
        bail!(
            "checksum mismatch for {}: expected {}, got {sha256}",
            asset.name,
            asset.sha256
        );
    }
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

/// Returns the number of bytes.
fn copy_hashed(mut from: impl Read, mut to: impl Write, hasher: &mut Sha256) -> io::Result<u64> {
    let mut buf = vec![0; 1 << 16];
    let mut total = 0;
    loop {
        let n = from.read(&mut buf)?;
        if n == 0 {
            return Ok(total);
        }
        hasher.update(&buf[..n]);
        to.write_all(&buf[..n])?;
        total += n as u64;
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
