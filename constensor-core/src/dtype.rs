use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Neg, Sub},
};

#[cfg(feature = "bfloat")]
use half::bf16;
#[cfg(feature = "half")]
use half::f16;

#[cfg(feature = "cuda")]
use cudarc::driver::DeviceRepr;

/// Marker trait for signed datatypes.
pub trait SignedDType: Neg<Output = Self> {}

impl SignedDType for i32 {}
impl SignedDType for i64 {}
impl SignedDType for f32 {}
impl SignedDType for f64 {}
#[cfg(feature = "bfloat")]
impl SignedDType for bf16 {}
#[cfg(feature = "half")]
impl SignedDType for f16 {}

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
    fn sqrt(&self) -> Option<Self>
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

pub trait DTypeOps:
    Copy + Add<Output = Self> + Div<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Sqrtable
{
}

#[cfg(feature = "cuda")]
/// Marker trait for tensor datatypes.
pub trait DType: Debug + DeviceRepr + Clone + DTypeOps {
    const ZERO: Self;
    const ONE: Self;
    const NAME: &'static str;
    const C_NAME: &'static str;
    const C_DEP: Option<&'static str>;
    const INTEGRAL: bool;

    /// Offset i by start and step using the formula `i*step + start`
    fn offset(i: usize, start: Self, step: Self) -> Self;
}

#[cfg(not(feature = "cuda"))]
/// Marker trait for tensor datatypes.
pub trait DType: Debug + Clone + DTypeOps {
    const ZERO: Self;
    const ONE: Self;
    const NAME: &'static str;
    const C_NAME: &'static str;
    const C_DEP: Option<&'static str>;
    const INTEGRAL: bool;

    /// Offset i by start and step using the formula `i*step + start`
    fn offset(i: usize, start: Self, step: Self) -> Self;
}

macro_rules! dtype {
    ($rt:ident, $zero:expr, $one:expr, $repr:expr, $c_repr:expr, $integral:expr) => {
        impl DTypeOps for $rt {}
        impl DType for $rt {
            const ZERO: $rt = $zero;
            const ONE: $rt = $one;
            const NAME: &'static str = $repr;
            const C_NAME: &'static str = $c_repr;
            const C_DEP: Option<&'static str> = None;
            const INTEGRAL: bool = $integral;

            fn offset(i: usize, start: Self, step: Self) -> Self {
                (i as $rt) * step + start
            }
        }
    };
}

dtype!(u8, 0u8, 1u8, "u8", "uint8_t", true);
dtype!(u32, 0u32, 1u32, "u32", "uint32_t", true);
dtype!(i32, 0i32, 1i32, "i32", "int", true);
dtype!(i64, 0i64, 1i64, "i64", "int64_t", true);
dtype!(f32, 0f32, 1f32, "f32", "float", false);
dtype!(f64, 0f64, 1f64, "f64", "double", false);

#[cfg(feature = "half")]
impl DTypeOps for f16 {}
#[cfg(feature = "half")]
impl DType for f16 {
    const ZERO: f16 = f16::from_f32_const(0.0);
    const ONE: f16 = f16::from_f32_const(1.0);
    const NAME: &'static str = "f16";
    const C_NAME: &'static str = "__half";
    const C_DEP: Option<&'static str> = Some("#include \"cuda_fp16.h\"");
    const INTEGRAL: bool = false;

    fn offset(i: usize, start: Self, step: Self) -> Self {
        f16::from_f64_const(i as f64) * step + start
    }
}
#[cfg(feature = "bfloat")]
impl DTypeOps for bf16 {}
#[cfg(feature = "bfloat")]
impl DType for bf16 {
    const ZERO: bf16 = bf16::from_f32_const(0.0);
    const ONE: bf16 = bf16::from_f32_const(1.0);
    const NAME: &'static str = "bf16";
    const C_NAME: &'static str = "__nv_bfloat16";
    const C_DEP: Option<&'static str> = Some("#include \"cuda_bf16.h\"");
    const INTEGRAL: bool = false;

    fn offset(i: usize, start: Self, step: Self) -> Self {
        bf16::from_f64_const(i as f64) * step + start
    }
}
