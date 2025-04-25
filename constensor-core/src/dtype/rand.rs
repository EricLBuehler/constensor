use super::DType;
use crate::Result;
// Bring in the CUDA RNG and slice types when using CUDA
#[cfg(feature = "cuda")]
use crate::cuda_backend::error::WrapErr;
#[cfg(feature = "cuda")]
use cudarc::curand::CudaRng;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;
// Optional half-precision types
#[cfg(feature = "bfloat")]
use half::bf16;
#[cfg(feature = "half")]
use half::f16;

/// Dispatch random fills based on the data type (CUDA backend).
pub trait RandDispatch {
    /// Fill the slice with uniform random values on the GPU.
    #[cfg(feature = "cuda")]
    fn cuda_fill_with_uniform(rng: &CudaRng, slice: &mut CudaSlice<Self>) -> Result<()>
    where
        Self: Sized;

    /// Fill the slice with normal (Gaussian) random values on the GPU.
    #[cfg(feature = "cuda")]
    fn cuda_fill_with_normal(
        rng: &CudaRng,
        slice: &mut CudaSlice<Self>,
        mean: Self,
        std: Self,
    ) -> Result<()>
    where
        Self: Sized;
}

// f32: support both uniform and normal
#[cfg(feature = "cuda")]
impl RandDispatch for f32 {
    fn cuda_fill_with_uniform(rng: &CudaRng, slice: &mut CudaSlice<Self>) -> Result<()> {
        rng.fill_with_uniform(slice).w()
    }
    fn cuda_fill_with_normal(
        rng: &CudaRng,
        slice: &mut CudaSlice<Self>,
        mean: Self,
        std: Self,
    ) -> Result<()> {
        rng.fill_with_normal(slice, mean, std).w()
    }
}

// f64: support both uniform and normal
#[cfg(feature = "cuda")]
impl RandDispatch for f64 {
    fn cuda_fill_with_uniform(rng: &CudaRng, slice: &mut CudaSlice<Self>) -> Result<()> {
        rng.fill_with_uniform(slice).w()
    }
    fn cuda_fill_with_normal(
        rng: &CudaRng,
        slice: &mut CudaSlice<Self>,
        mean: Self,
        std: Self,
    ) -> Result<()> {
        rng.fill_with_normal(slice, mean, std).w()
    }
}

// u32: uniform only
#[cfg(feature = "cuda")]
impl RandDispatch for u32 {
    fn cuda_fill_with_uniform(rng: &CudaRng, slice: &mut CudaSlice<Self>) -> Result<()> {
        rng.fill_with_uniform(slice).w()
    }
    fn cuda_fill_with_normal(
        _rng: &CudaRng,
        _slice: &mut CudaSlice<Self>,
        _mean: Self,
        _std: Self,
    ) -> Result<()> {
        crate::bail!(
            "Normal random fill is not supported for dtype {}",
            Self::C_NAME
        )
    }
}

// All other integral or half types: unsupported
#[cfg(feature = "cuda")]
impl RandDispatch for u8 {
    fn cuda_fill_with_uniform(_rng: &CudaRng, _slice: &mut CudaSlice<Self>) -> Result<()> {
        crate::bail!(
            "Uniform random fill is not supported for dtype {}",
            Self::C_NAME
        )
    }
    fn cuda_fill_with_normal(
        _rng: &CudaRng,
        _slice: &mut CudaSlice<Self>,
        _mean: Self,
        _std: Self,
    ) -> Result<()> {
        crate::bail!(
            "Normal random fill is not supported for dtype {}",
            Self::C_NAME
        )
    }
}
#[cfg(feature = "cuda")]
impl RandDispatch for i32 {
    fn cuda_fill_with_uniform(_rng: &CudaRng, _slice: &mut CudaSlice<Self>) -> Result<()> {
        crate::bail!(
            "Uniform random fill is not supported for dtype {}",
            Self::C_NAME
        )
    }
    fn cuda_fill_with_normal(
        _rng: &CudaRng,
        _slice: &mut CudaSlice<Self>,
        _mean: Self,
        _std: Self,
    ) -> Result<()> {
        crate::bail!(
            "Normal random fill is not supported for dtype {}",
            Self::C_NAME
        )
    }
}
#[cfg(feature = "cuda")]
impl RandDispatch for i64 {
    fn cuda_fill_with_uniform(_rng: &CudaRng, _slice: &mut CudaSlice<Self>) -> Result<()> {
        crate::bail!(
            "Uniform random fill is not supported for dtype {}",
            Self::C_NAME
        )
    }
    fn cuda_fill_with_normal(
        _rng: &CudaRng,
        _slice: &mut CudaSlice<Self>,
        _mean: Self,
        _std: Self,
    ) -> Result<()> {
        crate::bail!(
            "Normal random fill is not supported for dtype {}",
            Self::C_NAME
        )
    }
}
#[cfg(all(feature = "cuda", feature = "half"))]
impl RandDispatch for f16 {
    fn cuda_fill_with_uniform(_rng: &CudaRng, _slice: &mut CudaSlice<Self>) -> Result<()> {
        crate::bail!(
            "Uniform random fill is not supported for dtype {}",
            Self::C_NAME
        )
    }
    fn cuda_fill_with_normal(
        _rng: &CudaRng,
        _slice: &mut CudaSlice<Self>,
        _mean: Self,
        _std: Self,
    ) -> Result<()> {
        crate::bail!(
            "Normal random fill is not supported for dtype {}",
            Self::C_NAME
        )
    }
}
#[cfg(all(feature = "cuda", feature = "bfloat"))]
impl RandDispatch for bf16 {
    fn cuda_fill_with_uniform(_rng: &CudaRng, _slice: &mut CudaSlice<Self>) -> Result<()> {
        crate::bail!(
            "Uniform random fill is not supported for dtype {}",
            Self::C_NAME
        )
    }
    fn cuda_fill_with_normal(
        _rng: &CudaRng,
        _slice: &mut CudaSlice<Self>,
        _mean: Self,
        _std: Self,
    ) -> Result<()> {
        crate::bail!(
            "Normal random fill is not supported for dtype {}",
            Self::C_NAME
        )
    }
}
