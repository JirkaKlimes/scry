use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
};

use anyhow::{Context, Result};
use half::f16;
use image::{DynamicImage, imageops::FilterType};
use rayon::prelude::*;
use rten::{Model, NodeId, Value};
use tokenizers::{
    DecoderWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper, TokenizerImpl,
    TruncationParams, models::bpe::BPE,
};

use crate::{
    guard,
    models::{self, Asset},
};

macro_rules! hf {
    ($file:literal) => {
        concat!(
            "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/",
            "ba1f3b0843f24bc5417d38e19c37b287d719b2f4/",
            $file
        )
    };
}

/// The cache names embeddings after the model file. A new model needs a new file name, or scry
/// would keep the old file, so the cache never gives embeddings of another model.
pub const IMAGE_ID: &str = VISION.name;
/// The tokenizer comes from the same revision as the text model.
pub const TEXT_ID: &str = TEXT.name;

// The int8 vision tower ranks clearly worse than fp32. The int8 text tower ranks almost as well
// as fp32, at half the size of fp16.
const VISION: Asset = Asset {
    label: "SigLIP 2 image model",
    name: "siglip2-base-224-ba1f3b08-vision-fp16.onnx",
    url: hf!("onnx/vision_model_fp16.onnx"),
    sha256: "a1959f7bd3993a607e48839f6d01e25b876fe76afda301b028b78eef68aabd95",
};
const TEXT: Asset = Asset {
    label: "SigLIP 2 text model",
    name: "siglip2-base-224-ba1f3b08-text-int8.onnx",
    url: hf!("onnx/text_model_int8.onnx"),
    sha256: "3a0603d3a00c05a80a6ded4743c16aaac7b1e62cdcc7e362e7ce418659b96400",
};
const TOKENIZER: Asset = Asset {
    label: "SigLIP 2 tokenizer",
    name: "siglip2-base-224-ba1f3b08-tokenizer.json",
    url: hf!("tokenizer.json"),
    sha256: "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322",
};

// exp(logit_scale) and logit_bias of google/siglip2-base-patch16-224 at revision 75de2d55.
const SCALE: f32 = 112.668_9;
const BIAS: f32 = -16.771_725;

const SIZE: usize = 224;
const TOKENS: usize = 64;
const BATCH: usize = 8;

// With the generic `Tokenizer`, serde buffers the 256k vocab through untagged enums, which
// makes loading about twice as slow.
type Tokenizer = TokenizerImpl<
    BPE,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

/// The image tower. It loads on first use: a search with cached embeddings does not need it.
pub struct ImageEmbedder {
    path: PathBuf,
    tower: OnceLock<Tower>,
}

impl ImageEmbedder {
    pub fn load() -> Result<Self> {
        // Download now, so a progress bar never cuts into the scatter UI later.
        Ok(Self {
            path: models::fetch(&VISION)?,
            tower: OnceLock::new(),
        })
    }

    pub fn images(&self, images: &[&DynamicImage]) -> Result<Vec<Vec<f32>>> {
        guard(|| {
            let mut out = Vec::with_capacity(images.len());
            for batch in images.chunks(BATCH) {
                let tower = get_or_load(&self.tower, || Tower::load(&self.path, "pixel_values"))?;
                let pixels = batch.par_iter().flat_map_iter(|i| pixels(i)).collect();
                out.extend(tower.run(Value::from_shape([batch.len(), 3, SIZE, SIZE], pixels)?)?);
            }
            Ok(out)
        })
    }
}

/// A search text in the form that the text model reads. SigLIP 2 was trained on lowercased
/// text. The cache keys text embeddings by this form, so `Cat` finds the embedding of `cat`.
pub struct Prompt(String);

impl Prompt {
    pub fn new(text: &str) -> Self {
        Self(text.to_lowercase())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// The text tower and the tokenizer, loaded on first use. They take about 300 MB, so drop
/// this once the texts are embedded.
#[derive(Default)]
pub struct TextEmbedder(OnceLock<(Tokenizer, Tower)>);

impl TextEmbedder {
    /// One text per call: the int8 tower quantizes activations over the whole batch, so a text
    /// embedded together with others would get a slightly different embedding than alone.
    pub fn embed(&self, prompt: &Prompt) -> Result<Vec<f32>> {
        guard(|| {
            let (tokenizer, tower) = get_or_load(&self.0, load_text)?;
            let encoding = tokenizer
                .encode(prompt.as_str(), true)
                .map_err(anyhow::Error::msg)?;
            let ids = encoding.get_ids().iter().map(|&id| id as i32).collect();
            Ok(tower.run(Value::from_shape([1, TOKENS], ids)?)?.remove(0))
        })
    }
}

fn load_text() -> Result<(Tokenizer, Tower)> {
    // One download at a time, so the progress bars do not overlap.
    let tokenizer = models::fetch(&TOKENIZER)?;
    let tower = models::fetch(&TEXT)?;
    let (tokenizer, tower) = rayon::join(
        || load_tokenizer(&tokenizer),
        || Tower::load(&tower, "input_ids"),
    );
    Ok((tokenizer?, tower?))
}

/// The probability that a text describes an image, from the cosine of their embeddings.
pub fn score(cos: f32) -> f32 {
    1.0 / (1.0 + (-(SCALE * cos + BIAS)).exp())
}

/// The cosine of two unit vectors, which is their dot product.
pub fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

struct Tower {
    model: Model,
    input: NodeId,
    output: NodeId,
}

impl Tower {
    fn load(path: &Path, input: &str) -> Result<Self> {
        let model = models::load(path)?;
        let node = |name| {
            model
                .node_id(name)
                .with_context(|| models::cannot_load(path))
        };
        Ok(Self {
            input: node(input)?,
            output: node("pooler_output")?,
            model,
        })
    }

    fn run(&self, input: Value) -> Result<Vec<Vec<f32>>> {
        let [out] = self
            .model
            .run_n(vec![(self.input, input.into())], [self.output], None)?;
        let ([_, dim], data) = out.into_shape_vec::<f32, 2>()?;
        Ok(data.chunks(dim).map(normalize).collect())
    }
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(path)
        .map_err(anyhow::Error::msg)
        .with_context(|| models::cannot_load(path))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: TOKENS,
            ..Default::default()
        }))
        .map_err(anyhow::Error::msg)?;
    Ok(tokenizer)
}

/// Stable stand-in for `OnceLock::get_or_try_init`.
fn get_or_load<T>(cell: &OnceLock<T>, load: impl FnOnce() -> Result<T>) -> Result<&T> {
    if let Some(value) = cell.get() {
        return Ok(value);
    }
    let value = load()?;
    Ok(cell.get_or_init(|| value))
}

/// Same as SiglipImageProcessor: bilinear resize to a square, then planar RGB in [-1, 1].
fn pixels(image: &DynamicImage) -> Vec<f32> {
    let side = SIZE as u32;
    let rgb = image
        .resize_exact(side, side, FilterType::Triangle)
        .into_rgb8();
    (0..3)
        .flat_map(|c| rgb.pixels().map(move |p| f32::from(p[c]) / 127.5 - 1.0))
        .collect()
}

/// The cache stores f16. Rounding here too gives the same scores whether an embedding comes
/// from the cache or from the model.
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| f16::from_f32(x / norm).to_f32()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_gives_unit_length_in_f16() {
        let v = normalize(&[3.0, -4.0]);
        assert_eq!(v, [0.6, -0.8].map(|x| f16::from_f32(x).to_f32()));
        assert!((cos(&v, &v) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn score_is_a_monotonic_probability() {
        let scores: Vec<f32> = (-20..=25).map(|i| score(i as f32 / 100.0)).collect();
        assert!(scores.windows(2).all(|w| w[0] < w[1]));
        assert!(score(-1.0) >= 0.0 && score(1.0) <= 1.0);
        assert!((score(-BIAS / SCALE) - 0.5).abs() < 1e-3);
    }

    #[test]
    fn pixels_are_planar_and_scaled() {
        let image = image::RgbImage::from_pixel(7, 3, image::Rgb([255, 0, 51]));
        let p = pixels(&DynamicImage::ImageRgb8(image));
        assert_eq!(p.len(), 3 * SIZE * SIZE);
        for (plane, want) in p.chunks(SIZE * SIZE).zip([1.0, -1.0, -0.6]) {
            assert!(plane.iter().all(|x| (x - want).abs() < 1e-6));
        }
    }

    #[test]
    #[ignore = "downloads the SigLIP2 model (about 500 MB) and needs SCRY_TEST_CORPUS"]
    fn ranks_test_corpus() {
        let dir = std::env::var("SCRY_TEST_CORPUS").expect("SCRY_TEST_CORPUS is not set");
        let files = [
            "photos/000000000139.jpg",
            "photos/000000000285.jpg",
            "photos/000000000632.jpg",
            "photos/000000000724.jpg",
            "photos/000000000776.jpg",
            "photos/000000000785.jpg",
            "photos/000000000802.jpg",
            "photos/000000000872.jpg",
            "photos/000000000885.jpg",
            "photos/000000001000.jpg",
            "photos/000000001268.jpg",
            "photos/000000001296.jpg",
            "photos/000000001353.jpg",
            "photos/000000001425.jpg",
            "photos/000000001490.jpg",
            "photos/000000039769.jpg",
            "text/code_screenshot.png",
            "text/invoice.png",
            "text/meme.jpg",
            "text/notes.png",
        ];
        let english = [
            ("a living room with a television", 0),
            ("a bear", 1),
            ("a bedroom", 2),
            ("a stop sign", 3),
            ("teddy bears", 4),
            ("skiing in the snow", 5),
            ("a kitchen with a refrigerator", 6),
            ("baseball game", 7),
            ("tennis", 8),
            ("a group of children", 9),
            ("a bridge over a river", 10),
            ("a woman using a smartphone", 11),
            ("amusement park ride", 12),
            ("a black and white photo of food", 13),
            ("stand up paddle board", 14),
            ("cats on a couch", 15),
            ("a screenshot of source code", 16),
        ];
        let czech = [("medvěd", 1), ("kočky na gauči", 15), ("lyžování", 5)];

        let e = ImageEmbedder::load().unwrap();
        let t = TextEmbedder::default();
        let embed = |text: &str| t.embed(&Prompt::new(text)).unwrap();
        let images: Vec<_> = files
            .iter()
            .map(|f| image::open(Path::new(&dir).join(f)).unwrap())
            .collect();
        let images = e.images(&images.iter().collect::<Vec<_>>()).unwrap();
        let best = |query: &[f32], skip: Option<usize>| {
            (0..images.len())
                .filter(|&i| Some(i) != skip)
                .max_by(|&a, &b| cos(query, &images[a]).total_cmp(&cos(query, &images[b])))
                .unwrap()
        };
        let hits = |queries: &[(&str, usize)]| {
            queries
                .iter()
                .filter(|(text, want)| best(&embed(text), None) == *want)
                .count()
        };

        assert!(hits(&english) * 5 >= english.len() * 4);
        assert_eq!(hits(&czech), czech.len());
        assert_eq!(files[best(&images[3], Some(3))], "text/meme.jpg");

        let (bear, teddy) = (embed("bear"), embed("teddy bear"));
        assert!(cos(&images[4], &teddy) > cos(&images[4], &bear));
        assert!(cos(&images[1], &teddy) <= cos(&images[1], &bear));
    }
}
