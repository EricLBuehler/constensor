use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
};

#[cfg(feature = "cuda")]
use cudarc::driver::DeviceRepr;

#[cfg(feature = "cuda")]
/// Marker trait for tensor datatypes.
pub trait DType:
    Debug
    + DeviceRepr
    + Clone
    + Copy
    + Add<Output = Self>
    + Div<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    const NAME: &'static str;

    fn offset(i: usize, start: Self, step: Self) -> Self;
}

#[cfg(not(feature = "cuda"))]
/// Marker trait for tensor datatypes.
pub trait DType: Debug + Clone + Copy {
    const ZERO: Self;
    const ONE: Self;
    const NAME: &'static str;
}

macro_rules! dtype {
    ($rt:ident, $zero:expr, $one:expr, $repr:expr) => {
        impl DType for $rt {
            const ZERO: $rt = $zero;
            const ONE: $rt = $one;
            const NAME: &'static str = $repr;

            fn offset(i: usize, start: Self, step: Self) -> Self {
                (i as $rt) * step + start
            }
        }
    };
}

dtype!(u8, 0u8, 1u8, "u8");
dtype!(u32, 0u32, 1u32, "u32");
dtype!(i64, 0i64, 1i64, "i64");
dtype!(f32, 0f32, 1f32, "f32");
dtype!(f64, 0f64, 1f64, "f64");

#[cfg(feature = "bfloat")]
use half::bf16;
#[cfg(feature = "half")]
use half::f16;
#[cfg(feature = "half")]
impl DType for f16 {
    const ZERO: f16 = f16::from_f32_const(0.0);
    const ONE: f16 = f16::from_f32_const(1.0);
    const NAME: &'static str = "f16";

    fn offset(i: usize, start: Self, step: Self) -> Self {
        f16::from_f64_const(i as f64) * step + start
    }
}
#[cfg(feature = "bfloat")]
impl DType for bf16 {
    const ZERO: bf16 = bf16::from_f32_const(0.0);
    const ONE: bf16 = bf16::from_f32_const(1.0);
    const NAME: &'static str = "bf16";

    fn offset(i: usize, start: Self, step: Self) -> Self {
        bf16::from_f64_const(i as f64) * step + start
    }
}
