//! The image tower on a GPU, through tract on wgpu: Vulkan on Linux, Metal on macOS. wgpu loads
//! the GPU driver at run time, so scry also runs on a computer without one.

use std::{
    env::{self, VarError},
    path::Path,
    sync::OnceLock,
};

use anyhow::{Context, Result, bail, ensure};
use image::{DynamicImage, RgbImage};
use rten::Value;
use tract::prelude::*;
use wgpu::{AdapterInfo, Backend, DeviceType};

use super::{BATCH, SIZE, Tower, normalize, pixels};
use crate::{catch, models};

/// Where the image tower runs, from SCRY_DEVICE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    /// A GPU that is faster than the CPU, else the CPU.
    Auto,
    /// Any GPU that tract can use, also a slow one. An error if there is none.
    Gpu,
    Cpu,
}

impl Device {
    /// An empty SCRY_DEVICE counts as unset, as with SCRY_CACHE.
    pub fn from_env() -> Result<Self> {
        match env::var("SCRY_DEVICE").as_deref() {
            Err(VarError::NotPresent) | Ok("" | "auto") => Ok(Self::Auto),
            Ok("gpu") => Ok(Self::Gpu),
            Ok("cpu") => Ok(Self::Cpu),
            _ => bail!("SCRY_DEVICE must be auto, gpu or cpu"),
        }
    }
}

/// Why auto used the CPU on a GPU that looked fast. A message at that time would cut into a
/// picker, so scry tells it at exit.
static FAILURE: OnceLock<String> = OnceLock::new();

pub fn failure() -> Option<&'static str> {
    FAILURE.get().map(String::as_str)
}

pub struct Vision {
    runnable: Runnable,
    output: usize,
}

impl Vision {
    /// The tower on a GPU, or None if the CPU must run it.
    pub fn open(path: &Path, device: Device, cpu: &Tower) -> Result<Option<Self>> {
        if device == Device::Cpu || !adapter().is_some_and(|a| usable(&a, device)) {
            ensure!(
                device != Device::Gpu,
                "SCRY_DEVICE is gpu, but scry found no GPU that it can use"
            );
            return Ok(None);
        }
        match catch(|| Self::load(path, cpu)) {
            Ok(vision) => Ok(Some(vision)),
            Err(e) if device == Device::Auto => {
                let _ = FAILURE.set(format!("{e:#}"));
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    fn load(path: &Path, cpu: &Tower) -> Result<Self> {
        // With the value_info of the fp16 export, tract cannot analyse its input casts.
        let mut options = OnnxOptions::new();
        options.ignore_value_info = true;
        let mut model = tract::onnx()?.load_with_options(path, options)?;
        // tract 0.23.10 cannot run the pooling head with a symbolic batch size.
        model.set_input_fact(0, format!("{BATCH},3,{SIZE},{SIZE},f32").as_str())?;
        let mut model = model.into_model()?;
        // tract renames the outputs. pooler_output is the only one with two dimensions.
        let output = (0..model.output_count()?)
            .find(|&i| {
                model
                    .output_fact(i)
                    .and_then(|f| f.rank())
                    .is_ok_and(|r| r == 2)
            })
            .with_context(|| models::cannot_load(path))?;
        // In f32, each GPU gives the embeddings of rten and needs no f16 shaders.
        model.transform(FloatPrecision::new(DatumType::F16, DatumType::F32))?;
        let vision = Self {
            runnable: tract::runtime_for_name("wgpu")?.prepare(model)?,
            output,
        };
        vision.check(cpu)?;
        Ok(vision)
    }

    /// The embeddings of up to BATCH images. Zeros fill the batch, because its size is fixed.
    pub fn run(&self, mut pixels: Vec<f32>) -> Result<Vec<Vec<f32>>> {
        let n = pixels.len() / (3 * SIZE * SIZE);
        pixels.resize(BATCH * 3 * SIZE * SIZE, 0.0);
        let out = self
            .runnable
            .run([Tensor::from_slice(&[BATCH, 3, SIZE, SIZE], &pixels)?])?;
        let rows = out[self.output].as_slice::<f32>()?;
        Ok(rows
            .chunks(rows.len() / BATCH)
            .take(n)
            .map(normalize)
            .collect())
    }

    /// A bad kernel or driver gives wrong embeddings with no error, and the cache would keep
    /// them. So the GPU must agree with the CPU on a test image.
    fn check(&self, cpu: &Tower) -> Result<()> {
        let pixels = pixels(&pattern(64, 64, 4));
        let want = cpu.run(Value::from_shape([1, 3, SIZE, SIZE], pixels.clone())?)?;
        let d = distance(&want[0], &self.run(pixels)?[0]);
        ensure!(d < 1e-3, "the GPU gives wrong embeddings (off by {d})");
        Ok(())
    }
}

fn pattern(width: u32, height: u32, step: u32) -> DynamicImage {
    let image = RgbImage::from_fn(width, height, |x, y| {
        [x * step, y * 7, x ^ y].map(|c| c as u8).into()
    });
    DynamicImage::ImageRgb8(image)
}

/// The largest difference in one dimension. The rounding to f16 alone gives up to about 6e-5.
fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The adapter that tract-wgpu 0.23.10 takes, because it makes the same request.
fn adapter() -> Option<AdapterInfo> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    };
    let adapter = pollster::block_on(instance.request_adapter(&options)).ok()?;
    Some(adapter.get_info())
}

/// tract-wgpu needs Vulkan or Metal. Measured: a discrete GPU embeds about 10 times as fast as
/// the CPU, and an Intel UHD 3 times as slow.
fn usable(info: &AdapterInfo, device: Device) -> bool {
    let fast = match info.device_type {
        DeviceType::DiscreteGpu => true,
        // wgpu calls Apple silicon integrated, because the GPU shares memory with the CPU.
        DeviceType::IntegratedGpu => cfg!(all(target_os = "macos", target_arch = "aarch64")),
        DeviceType::Other | DeviceType::VirtualGpu | DeviceType::Cpu => false,
    };
    matches!(info.backend, Backend::Vulkan | Backend::Metal) && (fast || device == Device::Gpu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::VISION;

    #[test]
    fn auto_takes_only_fast_gpus() {
        let apple = cfg!(all(target_os = "macos", target_arch = "aarch64"));
        for (kind, backend, device, want) in [
            (DeviceType::DiscreteGpu, Backend::Vulkan, Device::Auto, true),
            (DeviceType::DiscreteGpu, Backend::Metal, Device::Auto, true),
            (
                DeviceType::IntegratedGpu,
                Backend::Metal,
                Device::Auto,
                apple,
            ),
            (
                DeviceType::IntegratedGpu,
                Backend::Vulkan,
                Device::Gpu,
                true,
            ),
            (DeviceType::Cpu, Backend::Vulkan, Device::Auto, false),
            (DeviceType::Cpu, Backend::Vulkan, Device::Gpu, true),
            (DeviceType::DiscreteGpu, Backend::Gl, Device::Gpu, false),
        ] {
            let info = AdapterInfo::new(kind, backend);
            assert_eq!(
                usable(&info, device),
                want,
                "{kind:?} {backend:?} {device:?}"
            );
        }
    }

    #[test]
    #[ignore = "downloads the SigLIP 2 image model and needs Vulkan or Metal (lavapipe works)"]
    fn gpu_matches_cpu() {
        let path = models::fetch(&VISION).unwrap();
        let cpu = Tower::load(&path, "pixel_values").unwrap();
        let gpu = Vision::open(&path, Device::Gpu, &cpu).unwrap().unwrap();
        // One full batch, and one that zeros fill.
        let pixels: Vec<f32> = (0..BATCH as u32 + 3)
            .flat_map(|i| pixels(&pattern(40 + i, 30, i)))
            .collect();
        for batch in pixels.chunks(BATCH * 3 * SIZE * SIZE) {
            let n = batch.len() / (3 * SIZE * SIZE);
            let want = cpu
                .run(Value::from_shape([n, 3, SIZE, SIZE], batch.to_vec()).unwrap())
                .unwrap();
            let got = gpu.run(batch.to_vec()).unwrap();
            assert_eq!(got.len(), n);
            assert!(want.iter().zip(&got).all(|(w, g)| distance(w, g) < 1e-3));
        }
    }
}
