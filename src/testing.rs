//! Debug/testing utilities.
//!
//! When `debug-export` feature is enabled, provides numpy (.npy) file I/O
//! for comparing intermediate tensors with the Python reference.
//! Otherwise, `save_npy` compiles to a no-op so the rest of the code
//! does not need feature-gated call sites.

use anyhow::Result;

// ── Real implementations (feature = debug-export) ──────────────────────────

#[cfg(feature = "debug-export")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "debug-export")]
use ndarray_npy::{ReadNpyExt, WriteNpyExt};

/// Load a float32 .npy file into a Candle Tensor.
#[cfg(feature = "debug-export")]
pub fn load_npy_f32<P: AsRef<std::path::Path>>(path: P, device: &Device) -> Result<Tensor> {
    let file = std::fs::File::open(path)?;
    let arr = ndarray::ArrayD::<f32>::read_npy(file)?;
    let shape: Vec<usize> = arr.shape().to_vec();
    let data: Vec<f32> = arr.into_raw_vec();
    let tensor = Tensor::from_slice(&data, shape, device)?;
    Ok(tensor)
}

/// Load an int64 .npy file into a Candle Tensor.
#[cfg(feature = "debug-export")]
pub fn load_npy_i64<P: AsRef<std::path::Path>>(path: P, device: &Device) -> Result<Tensor> {
    let file = std::fs::File::open(path)?;
    let arr = ndarray::ArrayD::<i64>::read_npy(file)?;
    let shape: Vec<usize> = arr.shape().to_vec();
    let data: Vec<i64> = arr.into_raw_vec();
    let tensor = Tensor::from_slice(&data, shape, device)?;
    Ok(tensor)
}

/// Save a Tensor to a .npy file (real implementation).
#[cfg(feature = "debug-export")]
pub fn save_npy<P: AsRef<std::path::Path>>(path: P, tensor: &Tensor) -> Result<()> {
    let dtype = tensor.dtype();
    match dtype {
        DType::F32 => {
            let data = tensor.flatten_all()?.to_vec1::<f32>()?;
            let shape: Vec<usize> = tensor.dims().to_vec();
            let arr = ndarray::ArrayD::from_shape_vec(shape, data)?;
            arr.write_npy(std::fs::File::create(path)?)?;
        }
        DType::F64 => {
            let data = tensor.flatten_all()?.to_vec1::<f64>()?;
            let shape: Vec<usize> = tensor.dims().to_vec();
            let arr = ndarray::ArrayD::from_shape_vec(shape, data)?;
            arr.write_npy(std::fs::File::create(path)?)?;
        }
        DType::I64 => {
            let data = tensor.flatten_all()?.to_vec1::<i64>()?;
            let shape: Vec<usize> = tensor.dims().to_vec();
            let arr = ndarray::ArrayD::from_shape_vec(shape, data)?;
            arr.write_npy(std::fs::File::create(path)?)?;
        }
        _ => anyhow::bail!("Unsupported dtype for npy save: {:?}", tensor.dtype()),
    }
    Ok(())
}

/// Assert mean squared error between two tensors is below threshold.
#[cfg(feature = "debug-export")]
pub fn assert_mse(a: &Tensor, b: &Tensor, max_mse: f64) -> Result<()> {
    let mse = (a - b)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    if mse > max_mse as f32 {
        anyhow::bail!("MSE {} exceeds threshold {}", mse, max_mse);
    }
    Ok(())
}

// ── No-op stubs (feature NOT enabled) ───────────────────────────────────────

/// No-op save when debug-export feature is disabled.
#[cfg(not(feature = "debug-export"))]
pub fn save_npy<P: AsRef<std::path::Path>>(_path: P, _tensor: &candle_core::Tensor) -> Result<()> {
    Ok(())
}
