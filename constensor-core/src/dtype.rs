use std::fmt::Debug;

#[cfg(feature = "cuda")]
use cudarc::driver::DeviceRepr;

#[cfg(feature = "cuda")]
/// Marker trait for tensor datatypes.
pub trait DType: Debug + DeviceRepr + Clone + Copy {
    const ZERO: Self;
    const ONE: Self;
    const NAME: &'static str;
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
dtype!(
    f16,
    f16::from_f32_const(0.0),
    f16::from_f32_const(1.0),
    "f16"
);
#[cfg(feature = "bfloat")]
dtype!(
    bf16,
    bf16::from_f32_const(0.0),
    bf16::from_f32_const(1.0),
    "bf16"
);
