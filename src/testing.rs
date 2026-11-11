use std::{
    env, fs,
    io::Cursor,
    ops::Deref,
    path::{Path, PathBuf},
    process,
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

use exif::{Field, experimental::Writer};
use image::{DynamicImage, ImageFormat};

/// A path in the temporary directory, removed on drop. Each path is new, so tests that run at
/// the same time never share one.
pub struct Temp(PathBuf);

impl Temp {
    /// A path with nothing at it yet. It ends with `name`, so it keeps the extension.
    pub fn path(name: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Relaxed);
        Self(env::temp_dir().join(format!("scry-{}-{n}-{name}", process::id())))
    }

    pub fn file(name: &str, bytes: &[u8]) -> Self {
        let temp = Self::path(name);
        fs::write(&temp.0, bytes).unwrap();
        temp
    }

    pub fn dir(name: &str) -> Self {
        let temp = Self::path(name);
        fs::create_dir(&temp.0).unwrap();
        temp
    }
}

impl Deref for Temp {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for Temp {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = match fs::symlink_metadata(&self.0) {
            Ok(md) if md.is_dir() => fs::remove_dir_all(&self.0),
            _ => fs::remove_file(&self.0),
        };
    }
}

pub fn encode(image: &DynamicImage, format: ImageFormat) -> Vec<u8> {
    let mut bytes = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut bytes), format)
        .unwrap();
    bytes
}

/// A black JPEG of `width` x `height`, with an APP1 EXIF segment that holds `fields`.
pub fn jpeg_with_exif(width: u32, height: u32, fields: &[Field]) -> Vec<u8> {
    let mut tiff = Cursor::new(Vec::new());
    let mut writer = Writer::new();
    fields.iter().for_each(|f| writer.push_field(f));
    writer.write(&mut tiff, false).unwrap();
    let jpeg = encode(&DynamicImage::new_rgb8(width, height), ImageFormat::Jpeg);
    let app1 = [b"Exif\0\0".as_slice(), tiff.get_ref()].concat();
    let len = u16::try_from(app1.len() + 2).unwrap().to_be_bytes();
    [&jpeg[..2], &[0xFF, 0xE1], &len, &app1, &jpeg[2..]].concat()
}
