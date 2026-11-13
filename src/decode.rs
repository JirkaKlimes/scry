use std::{
    fs::{self, File},
    io::{self, BufReader, Cursor, Read, Seek, SeekFrom},
    os::unix::fs::FileExt,
    path::Path,
    sync::{Condvar, Mutex},
};

use anyhow::{Result, ensure};
use exif::{In, Tag};
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, metadata::Orientation};
use rustix::fs::{Mode, OFlags};

use crate::guard;

#[cfg(target_os = "macos")]
mod imageio;

const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff"];

/// HEIC, AVIF and JPEG XL, which scry reads through ImageIO on macOS.
const IMAGEIO_EXTS: &[&str] = &["heic", "heif", "avif", "jxl"];

/// Camera RAW images, which scry cannot decode yet.
const RAW_EXTS: &[&str] = &["dng", "cr2", "cr3", "nef", "arw", "raf", "orf", "rw2"];

const IMAGEIO: bool = cfg!(target_os = "macos");

/// The formats that `is_unsupported` finds, for messages.
pub const UNSUPPORTED: &str = if IMAGEIO {
    "camera RAW"
} else {
    "HEIC, AVIF, JPEG XL or camera RAW"
};

pub fn is_image(path: &Path) -> bool {
    has_ext(path, IMAGE_EXTS) || IMAGEIO && has_ext(path, IMAGEIO_EXTS)
}

pub fn is_unsupported(path: &Path) -> bool {
    has_ext(path, RAW_EXTS) || !IMAGEIO && has_ext(path, IMAGEIO_EXTS)
}

fn has_ext(path: &Path, exts: &[&str]) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| exts.iter().any(|x| x.eq_ignore_ascii_case(e)))
}

/// Larger images are skipped, because a small file can hold a huge blank image.
const MAX_DECODED: u64 = 256 << 20;

/// An image file larger than its pixels holds mostly other data, so this limit costs no real
/// image. It stops a sparse or endless file before it fills the memory.
const MAX_FILE: u64 = MAX_DECODED;

/// Most headers fit in this many bytes, so a header read does not read the whole file.
const PREFIX: u64 = 128 << 10;

/// Decoders in all threads together hold at most this many bytes, unless one image is larger
/// and decodes alone.
static BUDGET: Budget = Budget::new(2 << 30);

/// Only regular files of a sane size. A link to /dev/zero, a FIFO, or a sparse 12 GiB file
/// would hang a thread or fill the memory.
pub fn check(md: &fs::Metadata) -> Result<u64> {
    ensure!(md.is_file(), "not a regular file");
    ensure!(md.len() > 0, "empty file");
    ensure!(md.len() <= MAX_FILE, "too large: {} MiB", md.len() >> 20);
    Ok(md.len())
}

/// What the start of an image file tells.
pub struct Header {
    width: u32,
    height: u32,
    orientation: Orientation,
    format: Format,
    decoded: u64,
    pub exif: Option<exif::Exif>,
}

enum Format {
    Image(ImageFormat),
    #[cfg(target_os = "macos")]
    ImageIo,
}

impl Header {
    /// Width and height as viewers show the image.
    pub fn size(&self) -> (u32, u32) {
        match self.orientation {
            Orientation::Rotate90
            | Orientation::Rotate270
            | Orientation::Rotate90FlipH
            | Orientation::Rotate270FlipH => (self.height, self.width),
            _ => (self.width, self.height),
        }
    }

    /// The most memory that a decoder needs. zune-jpeg holds the whole file, and for a
    /// progressive JPEG also 16-bit coefficients for each sample. Other decoders need about one
    /// more copy of the pixels.
    fn peak(&self, file: u64) -> u64 {
        match self.format {
            Format::Image(ImageFormat::Jpeg) => file + 3 * self.decoded,
            _ => 2 * self.decoded,
        }
    }
}

/// Reads the size and the EXIF, but not the pixels.
pub fn header(path: &Path) -> Result<Header> {
    guard(|| read_header(&mut open(path)?, path))
}

/// Decodes the image at `path`, shrunk to fit a `side` square, the right way up.
pub fn load(path: &Path, side: u32) -> Result<DynamicImage> {
    guard(|| {
        let mut file = open(path)?;
        let header = read_header(&mut file, path)?;
        ensure!(
            header.decoded <= MAX_DECODED,
            "too large: {} MiB decoded",
            header.decoded >> 20
        );
        let charge = BUDGET.charge(header.peak(file.len));
        file.rewind()?;
        let image = match header.format {
            Format::Image(format) => {
                let orientation = header.orientation;
                // The EXIF of a TIFF holds a copy of the file.
                drop(header);
                let decoder =
                    ImageReader::with_format(BufReader::new(file), format).into_decoder()?;
                // Turn the small copy: a square fit gives the same result before and after a turn.
                let mut image = shrink(DynamicImage::from_decoder(decoder)?, side);
                image.apply_orientation(orientation);
                image
            }
            #[cfg(target_os = "macos")]
            Format::ImageIo => {
                let mut bytes = Vec::new();
                file.read_to_end(&mut bytes)?;
                imageio::load(&bytes, side.min(header.width.max(header.height)))?
            }
        };
        drop(charge);
        Ok(image)
    })
}

fn shrink(image: DynamicImage, side: u32) -> DynamicImage {
    if image.width().max(image.height()) > side {
        image.thumbnail(side, side)
    } else {
        image
    }
}

/// TIFF can put its header at the end, and WebP puts EXIF after the pixels. So when a parse
/// of the first bytes fails, the whole file gets a second try.
fn read_header(file: &mut Bounded, path: &Path) -> Result<Header> {
    let mut bytes = Vec::new();
    file.by_ref().take(PREFIX).read_to_end(&mut bytes)?;
    if bytes.len() as u64 == file.len {
        return parse(&bytes, path, true);
    }
    parse(&bytes, path, false).or_else(|_| {
        let _charge = BUDGET.charge(file.len);
        file.read_to_end(&mut bytes)?;
        parse(&bytes, path, true)
    })
}

/// `whole` tells whether `bytes` is the whole file, so that broken EXIF is final.
fn parse(bytes: &[u8], path: &Path, whole: bool) -> Result<Header> {
    let format = image::guess_format(bytes).or_else(|_| ImageFormat::from_path(path));
    #[cfg(target_os = "macos")]
    if !format.as_ref().is_ok_and(ImageFormat::reading_enabled) {
        return parse_imageio(bytes, whole);
    }
    let format = format?;
    let mut decoder = ImageReader::with_format(Cursor::new(bytes), format).into_decoder()?;
    let (width, height) = decoder.dimensions();
    let exif = match read_exif(&mut decoder, format, bytes) {
        Ok(exif) => exif,
        // Broken EXIF only costs the EXIF filters.
        Err(_) if whole => None,
        Err(e) => return Err(e),
    };
    Ok(Header {
        width,
        height,
        orientation: orientation(exif.as_ref()),
        format: Format::Image(format),
        decoded: decoder.total_bytes(),
        exif,
    })
}

/// ImageIO and the HEIF parser of the EXIF reader need the whole file.
#[cfg(target_os = "macos")]
fn parse_imageio(bytes: &[u8], whole: bool) -> Result<Header> {
    ensure!(whole, "not the whole file");
    let (width, height, orientation) = imageio::info(bytes)?;
    Ok(Header {
        width,
        height,
        orientation,
        format: Format::ImageIo,
        decoded: 4 * u64::from(width) * u64::from(height),
        // Broken EXIF only costs the EXIF filters.
        exif: exif::Reader::new()
            .read_from_container(&mut Cursor::new(bytes))
            .ok(),
    })
}

fn orientation(exif: Option<&exif::Exif>) -> Orientation {
    exif.and_then(|e| {
        e.get_field(Tag::Orientation, In::PRIMARY)?
            .value
            .get_uint(0)
    })
    .and_then(|o| Orientation::from_exif(o.try_into().ok()?))
    .unwrap_or(Orientation::NoTransforms)
}

/// A TIFF file is itself the structure that EXIF uses.
fn read_exif(
    decoder: &mut impl ImageDecoder,
    format: ImageFormat,
    bytes: &[u8],
) -> Result<Option<exif::Exif>> {
    let raw = match format {
        ImageFormat::Tiff => Some(bytes.to_vec()),
        _ => decoder.exif_metadata()?,
    };
    Ok(raw
        .map(|raw| exif::Reader::new().read_raw(raw))
        .transpose()?)
}

/// O_NONBLOCK keeps the open of a FIFO from waiting for a writer. The check then rejects it.
fn open(path: &Path) -> Result<Bounded> {
    let flags = OFlags::RDONLY | OFlags::NONBLOCK | OFlags::NOCTTY | OFlags::CLOEXEC;
    let file = File::from(rustix::fs::open(path, flags, Mode::empty())?);
    let len = check(&file.metadata()?)?;
    Ok(Bounded { file, pos: 0, len })
}

/// A file that ends at the size that fstat saw, so a file that grows cannot make a read endless.
struct Bounded {
    file: File,
    pos: u64,
    len: u64,
}

impl Read for Bounded {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let left = usize::try_from(self.len.saturating_sub(self.pos)).unwrap_or(usize::MAX);
        let n = buf.len().min(left);
        let n = self.file.read_at(&mut buf[..n], self.pos)?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for Bounded {
    fn seek(&mut self, to: SeekFrom) -> io::Result<u64> {
        let pos = match to {
            SeekFrom::Start(pos) => Some(pos),
            SeekFrom::End(delta) => self.len.checked_add_signed(delta),
            SeekFrom::Current(delta) => self.pos.checked_add_signed(delta),
        };
        self.pos = pos.ok_or(io::ErrorKind::InvalidInput)?;
        Ok(self.pos)
    }
}

/// Bytes of decoder memory in use over all threads. Without a limit, each core could hold a
/// 256 MiB image and its copies at the same time.
struct Budget {
    used: Mutex<u64>,
    freed: Condvar,
    limit: u64,
}

impl Budget {
    const fn new(limit: u64) -> Self {
        Self {
            used: Mutex::new(0),
            freed: Condvar::new(),
            limit,
        }
    }

    /// Waits until `bytes` fit. Bytes over the whole limit wait until nothing else is charged.
    fn charge(&self, bytes: u64) -> Charge<'_> {
        let mut used = self.used.lock().unwrap();
        while *used > 0 && *used + bytes > self.limit {
            used = self.freed.wait(used).unwrap();
        }
        *used += bytes;
        Charge {
            budget: self,
            bytes,
        }
    }
}

struct Charge<'a> {
    budget: &'a Budget,
    bytes: u64,
}

impl Drop for Charge<'_> {
    fn drop(&mut self) {
        *self.budget.used.lock().unwrap() -= self.bytes;
        self.budget.freed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use std::{os::unix::fs::symlink, process::Command, sync::mpsc, thread, time::Duration};

    use exif::{Field, Value};

    use super::*;
    use crate::testing::{Temp, encode, jpeg_with_exif};

    /// A JPEG of `width` x `height` whose EXIF says to turn it by 90 degrees.
    fn turned_jpeg(width: u32, height: u32) -> Vec<u8> {
        let orientation = Field {
            tag: Tag::Orientation,
            ifd_num: In::PRIMARY,
            value: Value::Short(vec![6]),
        };
        jpeg_with_exif(width, height, &[orientation])
    }

    #[test]
    fn only_macos_reads_heic() {
        let heic = Path::new("IMG_1.HEIC");
        assert_eq!(is_image(heic), cfg!(target_os = "macos"));
        assert_eq!(is_unsupported(heic), !cfg!(target_os = "macos"));
        assert!(is_unsupported(Path::new("IMG_2.CR2")));
    }

    #[test]
    fn load_turns_the_image() {
        let file = Temp::file("turned.jpg", &turned_jpeg(3, 2));
        let image = load(&file, 1024).unwrap();
        assert_eq!((image.width(), image.height()), (2, 3));
        assert_eq!(header(&file).unwrap().size(), (2, 3));
    }

    #[test]
    fn load_shrinks_then_turns() {
        let file = Temp::file("shrunk.jpg", &turned_jpeg(300, 200));
        let image = load(&file, 30).unwrap();
        assert_eq!((image.width(), image.height()), (20, 30));
    }

    #[test]
    fn load_refuses_huge_images() {
        // A BMP header for a 20000x20000 RGB image, without the pixels.
        let header = [
            b"BM".as_slice(),
            &54u32.to_le_bytes(),
            &[0; 4],
            &54u32.to_le_bytes(),
            &40u32.to_le_bytes(),
            &20_000i32.to_le_bytes(),
            &20_000i32.to_le_bytes(),
            &1u16.to_le_bytes(),
            &24u16.to_le_bytes(),
            &[0; 24],
        ]
        .concat();
        let file = Temp::file("huge.bmp", &header);
        let err = load(&file, 1024).unwrap_err();
        assert!(err.to_string().starts_with("too large"), "{err}");
    }

    #[test]
    fn header_at_the_end_is_found() {
        // The TIFF encoder writes the directory after the pixels, past the prefix.
        let tiff = encode(&DynamicImage::new_rgb8(300, 300), ImageFormat::Tiff);
        assert!(tiff.len() as u64 > PREFIX);
        let file = Temp::file("late.tif", &tiff);
        assert_eq!(header(&file).unwrap().size(), (300, 300));
    }

    #[test]
    fn only_regular_files_open() {
        let empty = Temp::file("empty.jpg", b"");
        assert_eq!(header(&empty).err().unwrap().to_string(), "empty file");
        let zero = Temp::path("zero.jpg");
        symlink("/dev/zero", &zero).unwrap();
        assert_eq!(
            load(&zero, 64).unwrap_err().to_string(),
            "not a regular file"
        );
        let fifo = Temp::path("fifo.png");
        // rustix has no mkfifo for macOS.
        let mkfifo = Command::new("mkfifo").arg(&*fifo).status().unwrap();
        assert!(mkfifo.success());
        assert_eq!(
            header(&fifo).err().unwrap().to_string(),
            "not a regular file"
        );
    }

    #[test]
    fn bounded_file_ends_at_its_size() {
        let file = Temp::file("bounded.bin", b"0123456789");
        let mut bounded = Bounded {
            file: File::open(&file).unwrap(),
            pos: 0,
            len: 4,
        };
        let mut all = Vec::new();
        bounded.read_to_end(&mut all).unwrap();
        assert_eq!(all, b"0123");
        assert_eq!(bounded.seek(SeekFrom::End(-1)).unwrap(), 3);
        let mut last = Vec::new();
        bounded.read_to_end(&mut last).unwrap();
        assert_eq!(last, b"3");
        assert!(bounded.seek(SeekFrom::Current(-9)).is_err());
    }

    #[test]
    fn budget_waits_for_room() {
        let budget = Budget::new(10);
        let big = budget.charge(25);
        let (tx, rx) = mpsc::channel();
        thread::scope(|s| {
            s.spawn(|| {
                let _small = budget.charge(1);
                tx.send(()).unwrap();
            });
            assert!(rx.recv_timeout(Duration::from_millis(50)).is_err());
            drop(big);
            rx.recv_timeout(Duration::from_secs(5)).unwrap();
        });
    }
}
