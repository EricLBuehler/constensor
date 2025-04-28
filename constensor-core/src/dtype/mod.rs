use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
};

#[cfg(feature = "bfloat")]
use half::bf16;
#[cfg(feature = "half")]
use half::f16;

#[cfg(feature = "cuda")]
use cudarc::driver::DeviceRepr;

use gemm::GemmDispatch;
use rand::RandDispatch;
use simd_ops::SimdSupported;

mod gemm;
mod rand;
mod simd_ops;

/// Type which can be square-rooted.
/// If self<0 and Self is integral, then te output is 0
pub trait Sqrtable {
    fn sqrt(&self) -> Self
    where
        Self: Sized;
}

impl Sqrtable for f32 {
    fn sqrt(&self) -> Self
    where
        Self: Sized,
    {
        f32::sqrt(*self)
    }
}

impl Sqrtable for f64 {
    fn sqrt(&self) -> Self
    where
        Self: Sized,
    {
        f64::sqrt(*self)
    }
}

#[cfg(feature = "bfloat")]
impl Sqrtable for bf16 {
    fn sqrt(&self) -> Self
    where
        Self: Sized,
    {
        bf16::from_f64_const(self.to_f64_const().sqrt())
    }
}

#[cfg(feature = "half")]
impl Sqrtable for f16 {
    fn sqrt(&self) -> Self
    where
        Self: Sized,
    {
        f16::from_f64_const(self.to_f64_const().sqrt())
    }
}

macro_rules! sqrt_integral {
    ($t:ty) => {
        impl Sqrtable for $t {
            fn sqrt(&self) -> Self
            where
                Self: Sized,
            {
                (*self as f64).sqrt() as $t
            }
        }
    };
}

sqrt_integral!(u8);
sqrt_integral!(u32);
sqrt_integral!(i32);
sqrt_integral!(i64);

pub trait Expable {
    fn exp(&self) -> Self
    where
        Self: Sized;
    fn exp2(&self) -> Self
    where
        Self: Sized;
}

impl Expable for f32 {
    fn exp(&self) -> Self
    where
        Self: Sized,
    {
        f32::exp(*self)
    }
    fn exp2(&self) -> Self
    where
        Self: Sized,
    {
        f32::exp2(*self)
    }
}

impl Expable for f64 {
    fn exp(&self) -> Self
    where
        Self: Sized,
    {
        f64::exp(*self)
    }
    fn exp2(&self) -> Self
    where
        Self: Sized,
    {
        f64::exp2(*self)
    }
}

macro_rules! exp_integral {
    ($t:ty) => {
        impl Expable for $t {
            fn exp(&self) -> Self
            where
                Self: Sized,
            {
                (*self as f64).exp() as $t
            }
            fn exp2(&self) -> Self
            where
                Self: Sized,
            {
                (*self as f64).exp2() as $t
            }
        }
    };
}

exp_integral!(u8);
exp_integral!(u32);
exp_integral!(i32);
exp_integral!(i64);

#[cfg(feature = "bfloat")]
impl Expable for bf16 {
    fn exp(&self) -> Self
    where
        Self: Sized,
    {
        bf16::from_f64_const(self.to_f64_const().exp())
    }
    fn exp2(&self) -> Self
    where
        Self: Sized,
    {
        bf16::from_f64_const(self.to_f64_const().exp2())
    }
}

#[cfg(feature = "half")]
impl Expable for f16 {
    fn exp(&self) -> Self
    where
        Self: Sized,
    {
        f16::from_f64_const(self.to_f64_const().exp())
    }
    fn exp2(&self) -> Self
    where
        Self: Sized,
    {
        f16::from_f64_const(self.to_f64_const().exp2())
    }
}

pub trait Loggable {
    fn log(&self) -> Self
    where
        Self: Sized;
    fn log1p(&self) -> Self
    where
        Self: Sized;
}

impl Loggable for f32 {
    fn log(&self) -> Self
    where
        Self: Sized,
    {
        f32::ln(*self)
    }
    fn log1p(&self) -> Self
    where
        Self: Sized,
    {
        f32::ln_1p(*self)
    }
}

impl Loggable for f64 {
    fn log(&self) -> Self
    where
        Self: Sized,
    {
        f64::ln(*self)
    }
    fn log1p(&self) -> Self
    where
        Self: Sized,
    {
        f64::ln_1p(*self)
    }
}

macro_rules! log_integral {
    ($t:ty) => {
        impl Loggable for $t {
            fn log(&self) -> Self
            where
                Self: Sized,
            {
                (*self as f64).ln() as $t
            }
            fn log1p(&self) -> Self
            where
                Self: Sized,
            {
                (*self as f64).ln_1p() as $t
            }
        }
    };
}

log_integral!(u8);
log_integral!(u32);
log_integral!(i32);
log_integral!(i64);

#[cfg(feature = "bfloat")]
impl Loggable for bf16 {
    fn log(&self) -> Self
    where
        Self: Sized,
    {
        bf16::from_f64_const(self.to_f64_const().ln())
    }
    fn log1p(&self) -> Self
    where
        Self: Sized,
    {
        bf16::from_f64_const(self.to_f64_const().ln_1p())
    }
}

#[cfg(feature = "half")]
impl Loggable for f16 {
    fn log(&self) -> Self
    where
        Self: Sized,
    {
        f16::from_f64_const(self.to_f64_const().ln())
    }
    fn log1p(&self) -> Self
    where
        Self: Sized,
    {
        f16::from_f64_const(self.to_f64_const().ln_1p())
    }
}

pub trait DTypeOps:
    Copy
    + Add<Output = Self>
    + Div<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Sqrtable
    + Expable
    + Loggable
    + SimdSupported
    + GemmDispatch
    + RandDispatch
{
}

#[cfg(feature = "cuda")]
pub trait DeviceReprLike: DeviceRepr {}

#[cfg(not(feature = "cuda"))]
pub trait DeviceReprLike {}

impl DeviceReprLike for u8 {}
impl DeviceReprLike for i32 {}
impl DeviceReprLike for u32 {}
impl DeviceReprLike for i64 {}
impl DeviceReprLike for f32 {}
impl DeviceReprLike for f64 {}

pub trait MaybeNeg {
    const NAME: &'static str;

    /// A fallible version of `neg` that panics on an unsupported type.
    fn maybe_neg(self) -> Self;
}

macro_rules! maybe_neg_failing {
    ($rt:ident) => {
        impl MaybeNeg for $rt {
            const NAME: &'static str = stringify!($rt);

            fn maybe_neg(self) -> Self {
                panic!("This type does not support ")
            }
        }
    };
}

macro_rules! maybe_neg {
    ($rt:ident) => {
        impl MaybeNeg for $rt {
            const NAME: &'static str = stringify!($rt);

            fn maybe_neg(self) -> Self {
                -self
            }
        }
    };
}

maybe_neg_failing!(u8);
maybe_neg_failing!(u32);
maybe_neg!(i32);
maybe_neg!(i64);
maybe_neg!(f32);
maybe_neg!(f64);

/// Marker trait for tensor datatypes.
pub trait DType:
    Debug + Clone + DTypeOps + Send + Sync + MaybeNeg + DeviceReprLike + 'static
{
    const ZERO: Self;
    const ONE: Self;
    const C_NAME: &'static str;
    const C_DEP: Option<&'static str>;
    const INTEGRAL: bool;

    fn to_f64(&self) -> f64;
    fn from_f64(x: f64) -> Self;
}

macro_rules! dtype {
    ($rt:ident, $zero:expr, $one:expr, $c_repr:expr, $integral:expr) => {
        impl DTypeOps for $rt {}
        impl DType for $rt {
            const ZERO: $rt = $zero;
            const ONE: $rt = $one;
            const C_NAME: &'static str = $c_repr;
            const C_DEP: Option<&'static str> = None;
            const INTEGRAL: bool = $integral;

            fn to_f64(&self) -> f64 {
                *self as f64
            }
            fn from_f64(x: f64) -> Self {
                x as $rt
            }
        }
    };
}

dtype!(u8, 0u8, 1u8, "uint8_t", true);
dtype!(u32, 0u32, 1u32, "uint32_t", true);
dtype!(i32, 0i32, 1i32, "int", true);
dtype!(i64, 0i64, 1i64, "int64_t", true);
dtype!(f32, 0f32, 1f32, "float", false);
dtype!(f64, 0f64, 1f64, "double", false);

#[cfg(feature = "half")]
impl DTypeOps for f16 {}
#[cfg(feature = "half")]
impl DeviceReprLike for f16 {}
#[cfg(feature = "half")]
maybe_neg!(f16);
#[cfg(feature = "half")]
impl DType for f16 {
    const ZERO: f16 = f16::from_f64_const(0.0);
    const ONE: f16 = f16::from_f64_const(1.0);
    const C_NAME: &'static str = "__half";
    const C_DEP: Option<&'static str> = Some("#include \"cuda_fp16.h\"");
    const INTEGRAL: bool = false;

    fn to_f64(&self) -> f64 {
        self.to_f64_const()
    }
    fn from_f64(x: f64) -> Self {
        Self::from_f64_const(x)
    }
}
#[cfg(feature = "bfloat")]
impl DTypeOps for bf16 {}
#[cfg(feature = "bfloat")]
impl DeviceReprLike for bf16 {}
#[cfg(feature = "bfloat")]
maybe_neg!(bf16);
#[cfg(feature = "bfloat")]
impl DType for bf16 {
    const ZERO: bf16 = bf16::from_f64_const(0.0);
    const ONE: bf16 = bf16::from_f64_const(1.0);
    const C_NAME: &'static str = "__nv_bfloat16";
    const C_DEP: Option<&'static str> = Some("#include \"cuda_bf16.h\"");
    const INTEGRAL: bool = false;

    fn to_f64(&self) -> f64 {
        self.to_f64_const()
    }
    fn from_f64(x: f64) -> Self {
        Self::from_f64_const(x)
    }
}
