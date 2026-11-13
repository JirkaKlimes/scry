//! HEIC, AVIF and JPEG XL through ImageIO, the image decoder of macOS.

use anyhow::{Context, Result};
use image::{DynamicImage, RgbaImage, metadata::Orientation};
use objc2::rc::autoreleasepool;
use objc2_core_foundation::{
    CFBoolean, CFData, CFDictionary, CFNumber, CFRetained, CFString, CFType, CGPoint, CGRect,
    CGSize,
};
use objc2_core_graphics::{
    CGBitmapContextCreate, CGColorSpace, CGContext, CGImage, CGImageAlphaInfo, kCGColorSpaceSRGB,
};
use objc2_image_io::{
    CGImageSource, kCGImagePropertyOrientation, kCGImagePropertyPixelHeight,
    kCGImagePropertyPixelWidth, kCGImageSourceCreateThumbnailFromImageAlways,
    kCGImageSourceCreateThumbnailWithTransform, kCGImageSourceThumbnailMaxPixelSize,
};

const UNREADABLE: &str = "macOS cannot read this image";

// ImageIO can autorelease objects. A rayon thread has no pool that would release them, so each
// call below has its own.

/// The width, height and orientation of the first image in `bytes`.
pub fn info(bytes: &[u8]) -> Result<(u32, u32, Orientation)> {
    autoreleasepool(|_| {
        // SAFETY: There are no options.
        let properties = unsafe { source(bytes)?.properties_at_index(0, None) };
        // SAFETY: ImageIO keys the properties with strings.
        let properties: CFRetained<CFDictionary<CFString, CFType>> =
            unsafe { CFRetained::cast_unchecked(properties.context(UNREADABLE)?) };
        let number = |key| properties.get(key)?.downcast_ref::<CFNumber>()?.as_i64();
        let size = |key| {
            number(key)
                .and_then(|n| n.try_into().ok())
                .context(UNREADABLE)
        };
        // SAFETY: ImageIO defines these constants.
        let (width, height, orientation) = unsafe {
            (
                kCGImagePropertyPixelWidth,
                kCGImagePropertyPixelHeight,
                kCGImagePropertyOrientation,
            )
        };
        let orientation = number(orientation)
            .and_then(|o| Orientation::from_exif(o.try_into().ok()?))
            .unwrap_or(Orientation::NoTransforms);
        Ok((size(width)?, size(height)?, orientation))
    })
}

/// Decodes the first image in `bytes`, the right way up and shrunk to fit a `side` square.
pub fn load(bytes: &[u8], side: u32) -> Result<DynamicImage> {
    autoreleasepool(|_| {
        let side = CFNumber::new_i64(side.into());
        let yes = CFBoolean::new(true);
        // SAFETY: ImageIO defines these constants.
        let keys = unsafe {
            [
                kCGImageSourceCreateThumbnailFromImageAlways,
                kCGImageSourceCreateThumbnailWithTransform,
                kCGImageSourceThumbnailMaxPixelSize,
            ]
        };
        let values: [&CFType; 3] = [yes, yes, &side];
        let options = CFDictionary::from_slices(&keys, &values);
        // SAFETY: The options have the types that ImageIO expects.
        let image = unsafe { source(bytes)?.thumbnail_at_index(0, Some(options.as_opaque())) }
            .context(UNREADABLE)?;
        draw(&image)
    })
}

fn source(bytes: &[u8]) -> Result<CFRetained<CGImageSource>> {
    // SAFETY: There are no options.
    unsafe { CGImageSource::with_data(&CFData::from_bytes(bytes), None) }.context(UNREADABLE)
}

/// Draws `image` into 8-bit sRGB pixels, which converts all pixel formats and color spaces.
/// CoreGraphics draws RGBA only with premultiplied alpha. For an opaque photo, that is the same.
fn draw(image: &CGImage) -> Result<DynamicImage> {
    let (width, height) = (CGImage::width(Some(image)), CGImage::height(Some(image)));
    let mut pixels = vec![0; width * height * 4];
    // SAFETY: ImageIO defines this constant.
    let srgb = CGColorSpace::with_name(Some(unsafe { kCGColorSpaceSRGB }));
    // SAFETY: `pixels` holds `height` rows of `width * 4` bytes, and outlives the context.
    let context = unsafe {
        CGBitmapContextCreate(
            pixels.as_mut_ptr().cast(),
            width,
            height,
            8,
            width * 4,
            srgb.as_deref(),
            CGImageAlphaInfo::PremultipliedLast.0,
        )
    }
    .context(UNREADABLE)?;
    let rect = CGRect::new(CGPoint::ZERO, CGSize::new(width as f64, height as f64));
    CGContext::draw_image(Some(&context), rect, Some(image));
    drop(context);
    let image = RgbaImage::from_raw(width.try_into()?, height.try_into()?, pixels);
    Ok(image.context(UNREADABLE)?.into())
}

/// heif-enc of libheif and cjxl of libjxl made the test images from a PNG.
#[cfg(test)]
mod tests {
    use image::RgbaImage;
    use jiff::civil::date;

    use crate::{
        decode::{header, load},
        meta::{self, Exif, Meta},
        testing::Temp,
    };

    const RED: [u8; 3] = [255, 0, 0];
    const BLUE: [u8; 3] = [0, 0, 255];

    /// True if the pixel at `x`, `y` is near `rgb`. The test images are lossy.
    fn near(image: &RgbaImage, x: u32, y: u32, rgb: [u8; 3]) -> bool {
        let pixel = image.get_pixel(x, y).0;
        (0..3).all(|c| pixel[c].abs_diff(rgb[c]) < 40)
    }

    /// The HEIC holds a 128 x 64 image, red on the left and blue on the right. It is turned by 90
    /// degrees, and its EXIF tells the date and the camera, as a phone writes them.
    #[test]
    fn reads_heic() {
        let file = Temp::file("turned.heic", include_bytes!("testdata/turned.heic"));
        let exif = Exif {
            taken: Some(date(2024, 6, 15).at(12, 0, 0, 0)),
            camera: Some("Apple iPhone 15".into()),
            gps: false,
        };
        let meta = meta::read(&file).unwrap();
        assert_eq!(
            meta,
            Meta {
                width: 64,
                height: 128,
                exif
            }
        );
        let image = load(&file, 32).unwrap().into_rgba8();
        assert_eq!(image.dimensions(), (16, 32));
        assert!(near(&image, 8, 4, RED) && near(&image, 8, 28, BLUE));
    }

    /// Each holds a 128 x 64 image, red on the left and blue on the right.
    #[test]
    fn reads_avif_and_jpeg_xl() {
        let files: [(&str, &[u8]); 2] = [
            ("plain.avif", include_bytes!("testdata/plain.avif")),
            ("plain.jxl", include_bytes!("testdata/plain.jxl")),
        ];
        for (name, bytes) in files {
            let file = Temp::file(name, bytes);
            assert_eq!(header(&file).unwrap().size(), (128, 64), "{name}");
            let image = load(&file, 512).unwrap().into_rgba8();
            assert_eq!(image.dimensions(), (128, 64), "{name}");
            assert!(
                near(&image, 16, 32, RED) && near(&image, 112, 32, BLUE),
                "{name}"
            );
        }
    }
}
