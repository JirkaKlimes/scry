//! The image tower on a GPU, through tract on wgpu: Vulkan on Linux, Metal on macOS. wgpu loads
//! the GPU driver at run time, so scry also runs on a computer without one.

use std::{
    env::{self, VarError},
    mem,
    path::Path,
    sync::{
        Mutex, MutexGuard, OnceLock, PoisonError,
        mpsc::{self, Sender},
    },
    thread,
};

use anyhow::{Context, Result, bail, ensure};
use image::{DynamicImage, RgbImage};
use rten::Value;
use tract::prelude::*;
use wgpu::{AdapterInfo, Backend, DeviceType};

use super::{BATCH, SIZE, Tower, normalize, pixels};
use crate::{catch, guard, models};

/// Where the image tower runs, from SCRY_DEVICE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Device {
    /// A GPU that is faster than the CPU, else the CPU.
    Auto,
    /// Any GPU that tract can use, also a slow one. An error if there is none.
    Gpu,
    Cpu,
}

impl Device {
    /// An empty SCRY_DEVICE counts as unset, as with SCRY_CACHE.
    pub(super) fn from_env() -> Result<Self> {
        match env::var("SCRY_DEVICE").as_deref() {
            Err(VarError::NotPresent) | Ok("" | "auto") => Ok(Self::Auto),
            Ok("gpu") => Ok(Self::Gpu),
            Ok("cpu") => Ok(Self::Cpu),
            _ => bail!("SCRY_DEVICE must be auto, gpu or cpu"),
        }
    }
}

/// The GPU API of this platform, for messages.
pub const API: &str = if cfg!(target_os = "macos") {
    "Metal"
} else {
    "Vulkan"
};

/// wgpu calls the GPU of Apple silicon integrated, because it shares memory with the CPU. It is
/// fast all the same, unlike most other integrated GPUs.
const APPLE_SILICON: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

/// The GPU that auto takes, for the help.
pub const AUTO: &str = if APPLE_SILICON {
    "the GPU"
} else {
    "a discrete GPU"
};

/// Why auto used the CPU on a GPU that looked fast. A message at that time would cut into a
/// picker, so scry tells it at exit.
static FAILURE: OnceLock<String> = OnceLock::new();

pub fn failure() -> Option<&'static str> {
    FAILURE.get().map(String::as_str)
}

/// Held while scry calls the GPU driver. A driver can crash scry if it unloads at exit while
/// another thread runs it.
static BUSY: Mutex<()> = Mutex::new(());

fn busy() -> MutexGuard<'static, ()> {
    BUSY.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Waits for the driver call that runs, and keeps the GPU idle until scry exits.
pub fn stop() {
    mem::forget(busy());
}

/// The largest distance from the CPU embedding that the check accepts. Measured on NVIDIA, Intel
/// and lavapipe: up to 7e-5, the f16 rounding of the cache. At 1e-3, the logit of a `-q` score
/// moves by at most 0.11.
const LIMIT: f32 = 1e-3;

type Job = (Vec<f32>, Sender<Result<Vec<Vec<f32>>>>);

const STOPPED: &str = "the GPU thread stopped";

pub(super) struct Vision {
    jobs: Sender<Job>,
}

impl Vision {
    /// The tower on a GPU, or None if the CPU must run it.
    pub(super) fn open(path: &Path, device: Device, cpu: &Tower) -> Result<Option<Self>> {
        let open = || {
            if !adapter().is_some_and(|a| usable(&a, device)) {
                ensure!(
                    device != Device::Gpu,
                    "SCRY_DEVICE is gpu, but scry found no {API} GPU"
                );
                return Ok(None);
            }
            Self::load(path, cpu).map(Some)
        };
        match catch(open) {
            Err(e) if device == Device::Auto => {
                let _ = FAILURE.set(format!("{e:#}"));
                Ok(None)
            }
            result => result,
        }
    }

    /// tract-wgpu keeps GPU state for each thread. Measured: if one thread prepares the model and
    /// another thread runs it first, every image gets the same embedding. So one thread prepares
    /// the model and runs all batches, whichever thread asks.
    fn load(path: &Path, cpu: &Tower) -> Result<Self> {
        let (jobs, queue) = mpsc::channel::<Job>();
        let (ready, started) = mpsc::channel();
        let path = path.to_owned();
        thread::spawn(move || {
            let model = {
                let _busy = busy();
                guard(|| prepare(&path))
            };
            let (runnable, output) = match model {
                Ok(model) => model,
                Err(e) => {
                    let _ = ready.send(Err(e));
                    return;
                }
            };
            let _ = ready.send(Ok(()));
            for (pixels, reply) in queue {
                let _busy = busy();
                let _ = reply.send(guard(|| embed(&runnable, output, pixels)));
            }
            let _busy = busy();
            drop(runnable);
        });
        started.recv().context(STOPPED)??;
        let vision = Self { jobs };
        vision.check(cpu)?;
        Ok(vision)
    }

    /// The embeddings of up to BATCH images.
    pub(super) fn run(&self, pixels: Vec<f32>) -> Result<Vec<Vec<f32>>> {
        let (reply, answer) = mpsc::channel();
        self.jobs.send((pixels, reply)).ok().context(STOPPED)?;
        answer.recv().context(STOPPED)?
    }

    /// A bad kernel or driver gives wrong embeddings with no error, and the cache would keep
    /// them. So the GPU must agree with the CPU on two test images, which also checks how it
    /// lays out the batch.
    fn check(&self, cpu: &Tower) -> Result<()> {
        let pixels: Vec<f32> = [pattern(64, 64, 4), pattern(48, 80, 9)]
            .iter()
            .flat_map(pixels)
            .collect();
        let want = cpu.run(Value::from_shape([2, 3, SIZE, SIZE], pixels.clone())?)?;
        for (want, got) in want.iter().zip(self.run(pixels)?) {
            let d = distance(want, &got);
            ensure!(d < LIMIT, "the GPU gives wrong embeddings (off by {d})");
        }
        Ok(())
    }
}

/// The model on the GPU, and the index of pooler_output in its outputs.
fn prepare(path: &Path) -> Result<(Runnable, usize)> {
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
    Ok((tract::runtime_for_name("wgpu")?.prepare(model)?, output))
}

/// Zeros fill the batch, because its size is fixed.
fn embed(runnable: &Runnable, output: usize, mut pixels: Vec<f32>) -> Result<Vec<Vec<f32>>> {
    let n = pixels.len() / (3 * SIZE * SIZE);
    pixels.resize(BATCH * 3 * SIZE * SIZE, 0.0);
    let out = runnable.run([Tensor::from_slice(&[BATCH, 3, SIZE, SIZE], &pixels)?])?;
    let rows = out[output].as_slice::<f32>()?;
    Ok(rows
        .chunks(rows.len() / BATCH)
        .take(n)
        .map(normalize)
        .collect())
}

fn pattern(width: u32, height: u32, step: u32) -> DynamicImage {
    let image = RgbImage::from_fn(width, height, |x, y| {
        [x * step, y * 7, x ^ y].map(|c| c as u8).into()
    });
    DynamicImage::ImageRgb8(image)
}

/// The L2 distance. A GPU that computes nothing gives NaN, which fails the check.
fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// The adapter that tract-wgpu 0.23.10 takes, because it makes the same request.
fn adapter() -> Option<AdapterInfo> {
    let _busy = busy();
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
        DeviceType::IntegratedGpu => APPLE_SILICON,
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
        use Backend::{Gl, Metal, Vulkan};
        use Device::{Auto, Gpu};
        use DeviceType::{Cpu as Software, DiscreteGpu as Discrete, IntegratedGpu as Integrated};
        for (kind, backend, device, want) in [
            (Discrete, Vulkan, Auto, true),
            (Discrete, Metal, Auto, true),
            (Integrated, Metal, Auto, APPLE_SILICON),
            (Integrated, Vulkan, Gpu, true),
            (Software, Vulkan, Auto, false),
            (Software, Vulkan, Gpu, true),
            (Discrete, Gl, Gpu, false),
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
    fn distance_fails_on_nan() {
        assert_eq!(distance(&[3.0, 4.0], &[0.0, 0.0]), 5.0);
        assert!(distance(&[f32::NAN, 0.0], &[0.0, 0.0]).is_nan());
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
            assert!(want.iter().zip(&got).all(|(w, g)| distance(w, g) < LIMIT));
        }
    }
}
