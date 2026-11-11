use std::{
    env,
    ffi::OsString,
    fs::{self, DirBuilder, File, Permissions},
    io, mem,
    os::unix::{
        ffi::OsStringExt,
        fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt},
    },
    path::{Path, PathBuf},
    sync::{Mutex, MutexGuard, OnceLock},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use half::f16;
use redb::{
    ConcurrencyMode, Database, DatabaseError, ReadOnlyTable, ReadTransaction, ReadableDatabase,
    StorageError, TableDefinition, TableError, Value, WriteTransaction, backends::FileBackend,
};

use crate::{
    embed::{self, Prompt},
    meta::{Exif, Meta},
    ocr,
    output::clean_path,
};

const VERSION: u32 = 4;
const SCHEMA: TableDefinition<(), u32> = TableDefinition::new("schema");
const META: Def<MetaRow> = TableDefinition::new("meta");
const OCR: Def<String> = TableDefinition::new(ocr::ID);
const IMAGES: Def<Embedding> = TableDefinition::new(embed::IMAGE_ID);
/// Embeddings of search texts, by the prompt that the model reads.
const TEXTS: TableDefinition<&str, Embedding> = TableDefinition::new(embed::TEXT_ID);

/// redb keeps the pages it reads in memory, by default up to 1 GiB. A search reads each page
/// about once, so a large cache only costs memory.
const CACHE_SIZE: usize = 32 << 20;
/// A commit syncs the file twice, so writes wait for this much data or time.
const FLUSH_BYTES: usize = 8 << 20;
const FLUSH_AFTER: Duration = Duration::from_secs(2);

/// Size, mtime and ctime in nanoseconds, and inode. `cp -p` or `exiftool -P` keep the size
/// and mtime of a changed file, but nobody can set its ctime.
type Stamp = (u64, i128, i128, u64);
type Row<V> = (Stamp, V);
type Def<V> = TableDefinition<'static, &'static [u8], Row<V>>;
type Table<V> = ReadOnlyTable<&'static [u8], Row<V>>;
type MetaRow = (u32, u32, Option<String>, Option<String>, bool);
/// The bits of f16 values. Embeddings are unit vectors, so f16 changes a cosine by only about
/// 1e-5.
type Embedding = Vec<u16>;
type Op = Box<dyn FnOnce(&WriteTransaction) -> Result<(), redb::Error> + Send>;

/// A file in the cache. The stamp changes when the file changes, so a stale entry is never read.
pub struct Key {
    path: PathBuf,
    stamp: Stamp,
    /// The real path, so `a.jpg` and `./a.jpg` share entries. realpath costs a syscall for each
    /// part of the path, so it runs only when a table needs the key.
    real: OnceLock<Option<Vec<u8>>>,
}

impl Key {
    pub fn new(path: &Path, md: &fs::Metadata) -> Self {
        let ns = |s: i64, n: i64| i128::from(s) * 1_000_000_000 + i128::from(n);
        Self {
            path: path.to_owned(),
            stamp: (
                md.size(),
                ns(md.mtime(), md.mtime_nsec()),
                ns(md.ctime(), md.ctime_nsec()),
                md.ino(),
            ),
            real: OnceLock::new(),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn real(&self) -> Option<&[u8]> {
        let real = || {
            Some(
                fs::canonicalize(&self.path)
                    .ok()?
                    .into_os_string()
                    .into_vec(),
            )
        };
        self.real.get_or_init(real).as_deref()
    }
}

/// Concurrent scry processes share the cache. Commits are batched, so an interrupted run loses
/// little.
pub struct Cache {
    /// None if the cache did not open. Then nothing is stored.
    db: Option<Database>,
    pending: Mutex<Pending>,
}

struct Pending {
    ops: Vec<Op>,
    bytes: usize,
    committed: Instant,
}

/// One read snapshot for a chunk of lookups from many threads. Each lookup in its own
/// transaction would wait on locks. A key holds the stamp, so an old snapshot can only miss.
pub struct View<'a> {
    cache: &'a Cache,
    meta: Option<Table<MetaRow>>,
    ocr: Option<Table<String>>,
    images: Option<Table<Embedding>>,
}

impl Cache {
    /// Without a cache the search is slower, and scatter mode cannot spread out look-alikes.
    /// Neither is an error.
    pub fn open() -> Self {
        let db = dir().and_then(|dir| {
            let path = dir.join("cache.redb");
            open_or_replace(&path).with_context(|| clean_path(&path).to_string())
        });
        let db = db
            .inspect_err(|e| eprintln!("scry: running without a cache: {e:#}"))
            .ok();
        Self::new(db)
    }

    fn new(db: Option<Database>) -> Self {
        Self {
            db,
            pending: Mutex::new(Pending {
                ops: Vec::new(),
                bytes: 0,
                committed: Instant::now(),
            }),
        }
    }

    #[cfg(test)]
    pub fn disabled() -> Self {
        Self::new(None)
    }

    pub fn view(&self) -> View<'_> {
        let tx = self.db.as_ref().and_then(|db| db.begin_read().ok());
        View {
            cache: self,
            meta: table(&tx, META),
            ocr: table(&tx, OCR),
            images: table(&tx, IMAGES),
        }
    }

    pub fn put_embedding(&self, key: &Key, embedding: &[f32]) {
        self.put(IMAGES, key, to_f16(embedding));
    }

    /// The embedding of a search text, so a repeated search needs no text model.
    pub fn text_embedding(
        &self,
        prompt: &Prompt,
        embed: impl FnOnce() -> Result<Vec<f32>>,
    ) -> Result<Vec<f32>> {
        if let Some(bits) = self.text(prompt.as_str()) {
            return Ok(from_f16(bits));
        }
        let embedding = embed()?;
        let (text, bits) = (prompt.as_str().to_owned(), to_f16(&embedding));
        self.queue(
            text.len() + 2 * bits.len(),
            Box::new(move |tx| {
                tx.open_table(TEXTS)?.insert(text.as_str(), bits)?;
                Ok(())
            }),
        );
        Ok(embedding)
    }

    pub fn flush(&self) {
        self.commit_pending(self.pending.lock().unwrap());
    }

    fn text(&self, text: &str) -> Option<Embedding> {
        let tx = self.db.as_ref()?.begin_read().ok()?;
        Some(tx.open_table(TEXTS).ok()?.get(text).ok()??.value())
    }

    fn put<V>(&self, def: Def<V>, key: &Key, value: V)
    where
        V: for<'a> Value<SelfType<'a> = V> + Send + 'static,
    {
        let Some(path) = key.real() else {
            return;
        };
        let (path, stamp) = (path.to_vec(), key.stamp);
        let bytes = path.len() + V::as_bytes(&value).as_ref().len();
        self.queue(
            bytes,
            Box::new(move |tx| {
                tx.open_table(def)?
                    .insert(path.as_slice(), (stamp, value))?;
                Ok(())
            }),
        );
    }

    fn queue(&self, bytes: usize, op: Op) {
        if self.db.is_none() {
            return;
        }
        let mut pending = self.pending.lock().unwrap();
        pending.ops.push(op);
        pending.bytes += bytes;
        if pending.bytes >= FLUSH_BYTES || pending.committed.elapsed() >= FLUSH_AFTER {
            self.commit_pending(pending);
        }
    }

    /// Release the lock before the commit, so other threads can queue writes during it.
    fn commit_pending(&self, mut pending: MutexGuard<Pending>) {
        let ops = mem::take(&mut pending.ops);
        pending.bytes = 0;
        pending.committed = Instant::now();
        drop(pending);
        if let Some(db) = &self.db
            && !ops.is_empty()
        {
            // A lost batch only costs a recompute in a later run.
            let _ = commit(db, ops);
        }
    }
}

impl Drop for Cache {
    fn drop(&mut self) {
        self.flush();
    }
}

impl View<'_> {
    pub fn meta(&self, key: &Key, read: impl FnOnce() -> Result<Meta>) -> Result<Meta> {
        let (width, height, taken, camera, gps) = self.cached(&self.meta, META, key, || {
            let m = read()?;
            let taken = m.exif.taken.map(|t| t.to_string());
            Ok((m.width, m.height, taken, m.exif.camera, m.exif.gps))
        })?;
        Ok(Meta {
            width,
            height,
            exif: Exif {
                taken: taken.and_then(|t| t.parse().ok()),
                camera,
                gps,
            },
        })
    }

    pub fn ocr(&self, key: &Key, read: impl FnOnce() -> Result<String>) -> Result<String> {
        self.cached(&self.ocr, OCR, key, read)
    }

    pub fn embedding(&self, key: &Key) -> Option<Vec<f32>> {
        get(&self.images, key).map(from_f16)
    }

    fn cached<V>(
        &self,
        table: &Option<Table<V>>,
        def: Def<V>,
        key: &Key,
        compute: impl FnOnce() -> Result<V>,
    ) -> Result<V>
    where
        V: for<'a> Value<SelfType<'a> = V> + Clone + Send + 'static,
    {
        if let Some(value) = get(table, key) {
            return Ok(value);
        }
        let value = compute()?;
        self.cache.put(def, key, value.clone());
        Ok(value)
    }
}

fn table<V: Value + 'static>(tx: &Option<ReadTransaction>, def: Def<V>) -> Option<Table<V>> {
    tx.as_ref()?.open_table(def).ok()
}

/// A failed read is a miss: the cache is disposable.
fn get<V>(table: &Option<Table<V>>, key: &Key) -> Option<V>
where
    V: for<'a> Value<SelfType<'a> = V> + 'static,
{
    let (stamp, value) = table.as_ref()?.get(key.real()?).ok()??.value();
    (stamp == key.stamp).then_some(value)
}

fn to_f16(embedding: &[f32]) -> Embedding {
    embedding
        .iter()
        .map(|&x| f16::from_f32(x).to_bits())
        .collect()
}

fn from_f16(bits: Embedding) -> Vec<f32> {
    bits.into_iter()
        .map(|b| f16::from_bits(b).to_f32())
        .collect()
}

/// The cache directory. Only its owner can open it: the cache holds file paths and the text in
/// images.
pub fn dir() -> Result<PathBuf> {
    let dir = location(env::var_os("SCRY_CACHE"))?;
    DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(&dir)
        .with_context(|| clean_path(&dir).to_string())?;
    Ok(dir)
}

/// SCRY_CACHE, else the cache directory of the system. An empty SCRY_CACHE counts as unset.
/// A relative one would put models where scry runs, and trust a model file found there.
fn location(var: Option<OsString>) -> Result<PathBuf> {
    match var.filter(|v| !v.is_empty()).map(PathBuf::from) {
        Some(dir) => {
            ensure!(dir.is_absolute(), "SCRY_CACHE must be an absolute path");
            Ok(dir)
        }
        None => Ok(dirs::cache_dir()
            .context("cannot find a cache directory, set SCRY_CACHE")?
            .join("scry")),
    }
}

/// Replaces a file with another schema, another redb format, or no redb database.
fn open_or_replace(path: &Path) -> Result<Database> {
    if let Some(db) = open_db(path)? {
        return Ok(db);
    }
    fs::remove_file(path)?;
    open_db(path)?.context("cache was replaced by another scry version")
}

fn open_db(path: &Path) -> Result<Option<Database>> {
    let file = File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(path)?;
    // The mode above applies only to a new file.
    file.set_permissions(Permissions::from_mode(0o600))?;
    let opened = Database::builder()
        .set_cache_size(CACHE_SIZE)
        .set_concurrency_mode(ConcurrencyMode::MultiWriter)
        .create_with_backend(FileBackend::new(file)?);
    let db = match opened {
        Ok(db) => db,
        Err(e) if unusable(&e) => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let version = match db.begin_read()?.open_table(SCHEMA) {
        Ok(table) => table.get(())?.map(|v| v.value()),
        Err(TableError::TableDoesNotExist(_)) => None,
        Err(e) => return Err(e.into()),
    };
    match version {
        Some(VERSION) => {}
        Some(_) => return Ok(None),
        None => {
            let tx = db.begin_write()?;
            tx.open_table(SCHEMA)?.insert((), VERSION)?;
            tx.commit()?;
        }
    }
    Ok(Some(db))
}

/// A bad file signature or a truncated header comes back as an I/O error.
fn unusable(e: &DatabaseError) -> bool {
    match e {
        DatabaseError::UpgradeRequired(_) | DatabaseError::Storage(StorageError::Corrupted(_)) => {
            true
        }
        DatabaseError::Storage(StorageError::Io(e)) => matches!(
            e.kind(),
            io::ErrorKind::InvalidData | io::ErrorKind::UnexpectedEof
        ),
        _ => false,
    }
}

fn commit(db: &Database, ops: Vec<Op>) -> Result<(), redb::Error> {
    let tx = db.begin_write()?;
    for op in ops {
        op(&tx)?;
    }
    Ok(tx.commit()?)
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, fs::FileTimes};

    use jiff::civil::date;

    use super::*;
    use crate::testing::Temp;

    fn file(dir: &Path, name: &str, content: &str) -> Key {
        let path = dir.join(name);
        fs::write(&path, content).unwrap();
        Key::new(&path, &fs::metadata(&path).unwrap())
    }

    fn db(dir: &Path) -> PathBuf {
        dir.join("cache.redb")
    }

    fn open(dir: &Path) -> Cache {
        Cache::new(Some(open_or_replace(&db(dir)).unwrap()))
    }

    fn sample_meta() -> Meta {
        Meta {
            width: 640,
            height: 480,
            exif: Exif {
                taken: Some(date(2024, 6, 1).at(12, 30, 5, 0)),
                camera: Some("Canon EOS 5D".into()),
                gps: true,
            },
        }
    }

    #[test]
    fn values_survive_reopen() {
        let dir = Temp::dir("roundtrip");
        let key = file(&dir, "a.jpg", "a");
        {
            let cache = open(&dir);
            let view = cache.view();
            view.meta(&key, || Ok(sample_meta())).unwrap();
            view.ocr(&key, || Ok("INV-2041".into())).unwrap();
            cache.put_embedding(&key, &[0.5, -1.0, 2.0]);
        }
        let cache = open(&dir);
        let view = cache.view();
        let meta = view.meta(&key, || panic!("meta recomputed")).unwrap();
        assert_eq!((meta.width, meta.height), (640, 480));
        assert_eq!(meta.exif.taken, sample_meta().exif.taken);
        assert_eq!(meta.exif.camera.as_deref(), Some("Canon EOS 5D"));
        assert!(meta.exif.gps);
        assert_eq!(
            view.ocr(&key, || panic!("ocr recomputed")).unwrap(),
            "INV-2041"
        );
        assert_eq!(view.embedding(&key), Some(vec![0.5, -1.0, 2.0]));
    }

    #[test]
    fn rewritten_file_is_recomputed() {
        let dir = Temp::dir("stale");
        let old = file(&dir, "a.jpg", "a");
        {
            let cache = open(&dir);
            cache.view().ocr(&old, || Ok("old".into())).unwrap();
            cache.put_embedding(&old, &[1.0]);
        }
        let new = file(&dir, "a.jpg", "longer");
        let cache = open(&dir);
        let view = cache.view();
        assert_eq!(view.embedding(&new), None);
        let calls = Cell::new(0);
        let text = view.ocr(&new, || {
            calls.set(calls.get() + 1);
            Ok("new".into())
        });
        assert_eq!((text.unwrap(), calls.get()), ("new".into(), 1));
    }

    #[test]
    fn kept_size_and_mtime_do_not_hide_a_change() {
        let dir = Temp::dir("ctime");
        let old = file(&dir, "a.jpg", "a");
        let path = dir.join("a.jpg");
        let mtime = fs::metadata(&path).unwrap().modified().unwrap();
        {
            let cache = open(&dir);
            cache.view().ocr(&old, || Ok("old".into())).unwrap();
        }
        // Like `cp -p`: same size and mtime, new content.
        std::thread::sleep(Duration::from_millis(20));
        fs::write(&path, "b").unwrap();
        let file = File::options().write(true).open(&path).unwrap();
        file.set_times(FileTimes::new().set_modified(mtime))
            .unwrap();
        let new = Key::new(&path, &fs::metadata(&path).unwrap());
        assert_eq!((new.stamp.0, new.stamp.1), (old.stamp.0, old.stamp.1));
        let cache = open(&dir);
        assert_eq!(cache.view().ocr(&new, || Ok("new".into())).unwrap(), "new");
    }

    #[test]
    fn failed_compute_is_not_cached() {
        let dir = Temp::dir("failed");
        let key = file(&dir, "a.jpg", "a");
        let cache = open(&dir);
        let view = cache.view();
        assert!(view.ocr(&key, || anyhow::bail!("corrupt")).is_err());
        assert_eq!(view.ocr(&key, || Ok("ok".into())).unwrap(), "ok");
    }

    #[test]
    fn other_schema_is_replaced() {
        let dir = Temp::dir("schema");
        let key = file(&dir, "a.jpg", "a");
        {
            let cache = open(&dir);
            cache.view().ocr(&key, || Ok("old".into())).unwrap();
            let tx = cache.db.as_ref().unwrap().begin_write().unwrap();
            tx.open_table(SCHEMA)
                .unwrap()
                .insert((), VERSION + 1)
                .unwrap();
            tx.commit().unwrap();
        }
        let cache = open(&dir);
        assert_eq!(cache.view().ocr(&key, || Ok("new".into())).unwrap(), "new");
        let tx = cache.db.as_ref().unwrap().begin_read().unwrap();
        let version = tx
            .open_table(SCHEMA)
            .unwrap()
            .get(())
            .unwrap()
            .unwrap()
            .value();
        assert_eq!(version, VERSION);
    }

    #[test]
    fn writes_wait_for_a_batch_or_flush() {
        let dir = Temp::dir("flush");
        let key = file(&dir, "a.jpg", "a");
        let cache = open(&dir);
        cache.put_embedding(&key, &[1.0]);
        assert_eq!(cache.view().embedding(&key), None);
        cache.flush();
        assert_eq!(cache.view().embedding(&key), Some(vec![1.0]));
    }

    #[test]
    fn text_embeddings_are_kept() {
        let dir = Temp::dir("texts");
        let cache = open(&dir);
        let cat = cache.text_embedding(&Prompt::new("cat"), || Ok(vec![0.25]));
        cache.flush();
        let capital = cache.text_embedding(&Prompt::new("Cat"), || panic!("embedded again"));
        assert_eq!((cat.unwrap(), capital.unwrap()), (vec![0.25], vec![0.25]));
        let dog = cache.text_embedding(&Prompt::new("dog"), || Ok(vec![-1.0]));
        assert_eq!(dog.unwrap(), [-1.0]);
    }

    #[test]
    fn foreign_files_are_replaced() {
        let dir = Temp::dir("foreign");
        let key = file(&dir, "a.jpg", "a");
        open(&dir).view().ocr(&key, || Ok("a".into())).unwrap();
        let valid = fs::read(db(&dir)).unwrap();
        for junk in [b"not a database".as_slice(), &valid[..100]] {
            fs::write(db(&dir), junk).unwrap();
            let cache = open(&dir);
            assert_eq!(cache.view().ocr(&key, || Ok("new".into())).unwrap(), "new");
        }
    }

    #[test]
    fn only_the_owner_can_read_the_cache() {
        let dir = Temp::dir("private");
        fs::write(db(&dir), "").unwrap();
        fs::set_permissions(db(&dir), Permissions::from_mode(0o644)).unwrap();
        drop(open(&dir));
        let mode = fs::metadata(db(&dir)).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    #[test]
    fn cache_location() {
        let default = location(None).unwrap();
        assert!(default.ends_with("scry") && default.is_absolute());
        assert_eq!(location(Some("".into())).unwrap(), default);
        assert_eq!(location(Some("/x/y".into())).unwrap(), Path::new("/x/y"));
        assert!(location(Some("cache".into())).is_err());
    }

    #[test]
    fn missing_database_stores_nothing() {
        let dir = Temp::dir("none");
        let key = file(&dir, "a.jpg", "a");
        let cache = Cache::new(None);
        cache.put_embedding(&key, &[1.0]);
        cache.flush();
        assert_eq!(cache.view().embedding(&key), None);
        assert_eq!(cache.view().ocr(&key, || Ok("x".into())).unwrap(), "x");
    }
}
