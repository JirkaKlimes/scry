use std::path::Path;

use anyhow::Result;
use image::{DynamicImage, Rgb, RgbImage, Rgba};
use ocrs::{ImageSource, OcrEngine, OcrEngineParams};

use crate::{
    decode, guard,
    models::{self, Asset},
};

// concat! takes only literals, so the model file names are a macro.
macro_rules! name {
    (detection) => {
        "ocrs-text-detection-a917b23d.onnx"
    };
    (recognition) => {
        "ocrs-text-recognition-86c145c2.onnx"
    };
}

/// The cache names OCR text after both model files. A new model needs a new file name, so the
/// cache never gives the text of another model.
pub const ID: &str = concat!(name!(detection), "+", name!(recognition));

const DETECTION: Asset = Asset {
    label: "OCR text detection model",
    name: name!(detection),
    url: "https://ocrs-models.s3-accelerate.amazonaws.com/text-detection.onnx",
    sha256: "a917b23dbd9524b465df7e922641b3ff2981623df4ded5a0234004ef2fee7cfe",
};

const RECOGNITION: Asset = Asset {
    label: "OCR text recognition model",
    name: name!(recognition),
    url: "https://ocrs-models.s3-accelerate.amazonaws.com/text-recognition.onnx",
    sha256: "86c145c2edb96c8caed5b1ebb8f44d706408922211451c309c625157dd6061c5",
};

/// Recognition cuts each text line from this copy, so a larger copy reads smaller text. It also
/// costs more memory and time.
const MAX_SIDE: u32 = 4096;

pub struct Ocr(OcrEngine);

impl Ocr {
    pub fn load() -> Result<Self> {
        Ok(Self(OcrEngine::new(OcrEngineParams {
            detection_model: Some(models::load(&models::fetch(&DETECTION)?)?),
            recognition_model: Some(models::load(&models::fetch(&RECOGNITION)?)?),
            ..Default::default()
        })?))
    }

    pub fn read(&self, path: &Path) -> Result<String> {
        guard(|| self.text(decode::load(path, MAX_SIDE)?))
    }

    /// Line breaks rarely matter to a search. Single spaces between all words let a phrase match
    /// when it wraps.
    fn text(&self, image: DynamicImage) -> Result<String> {
        let rgb = on_white(image).into_rgb8();
        let source = ImageSource::from_bytes(rgb.as_raw(), rgb.dimensions())?;
        let text = self.0.get_text(&self.0.prepare_input(source)?)?;
        Ok(text.split_whitespace().collect::<Vec<_>>().join(" "))
    }
}

/// Transparent pixels often store black, which would hide dark text. Viewers show them on white.
fn on_white(image: DynamicImage) -> DynamicImage {
    if !image.color().has_alpha() {
        return image;
    }
    let rgba = image.into_rgba8();
    let blend = |c: u8, a: u8| {
        let (c, a) = (u16::from(c), u16::from(a));
        ((c * a + 255 * (255 - a)) / 255) as u8
    };
    DynamicImage::ImageRgb8(RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
        let Rgba([r, g, b, a]) = *rgba.get_pixel(x, y);
        Rgb([r, g, b].map(|c| blend(c, a)))
    }))
}

#[cfg(test)]
mod tests {
    use image::RgbaImage;

    use super::*;

    #[test]
    fn transparency_becomes_white() {
        let pixels = [
            [0, 0, 0, 0],
            [0, 0, 0, 255],
            [200, 100, 0, 128],
            [9, 9, 9, 255],
        ];
        let image = RgbaImage::from_fn(4, 1, |x, _| Rgba(pixels[x as usize]));
        let flat = on_white(DynamicImage::ImageRgba8(image)).into_rgb8();
        let got: Vec<[u8; 3]> = flat.pixels().map(|p| p.0).collect();
        assert_eq!(got, [[255; 3], [0; 3], [227, 177, 127], [9; 3]]);
    }

    #[test]
    #[ignore = "downloads the OCR models"]
    fn blank_image_has_no_text() {
        let blank = DynamicImage::ImageRgb8(RgbImage::from_pixel(400, 300, [255; 3].into()));
        assert_eq!(Ocr::load().unwrap().text(blank).unwrap(), "");
    }
}
