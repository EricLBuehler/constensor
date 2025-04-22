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

use crate::graph::BinaryOpType;

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

pub trait DTypeOps:
    Copy
    + Add<Output = Self>
    + Div<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Sqrtable
    + SimdSupported
{
}

#[cfg(feature = "cuda")]
/// Marker trait for tensor datatypes.
pub trait DType: Debug + DeviceRepr + Clone + DTypeOps + Send + Sync {
    const ZERO: Self;
    const ONE: Self;
    const NAME: &'static str;
    const C_NAME: &'static str;
    const C_DEP: Option<&'static str>;
    const INTEGRAL: bool;

    fn to_f64(&self) -> f32;
    fn from_f64(x: f32) -> Self;
}

#[cfg(not(feature = "cuda"))]
/// Marker trait for tensor datatypes.
pub trait DType: Debug + Clone + DTypeOps + Send + Sync {
    const ZERO: Self;
    const ONE: Self;
    const NAME: &'static str;
    const C_NAME: &'static str;
    const C_DEP: Option<&'static str>;
    const INTEGRAL: bool;

    fn to_f64(&self) -> f64;
    fn from_f64(x: f64) -> Self;
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

            fn to_f64(&self) -> f64 {
                *self as f64
            }
            fn from_f64(x: f64) -> Self {
                x as $rt
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
    const ZERO: f16 = f16::from_f64_const(0.0);
    const ONE: f16 = f16::from_f64_const(1.0);
    const NAME: &'static str = "f16";
    const C_NAME: &'static str = "__half";
    const C_DEP: Option<&'static str> = Some("#include \"cuda_fp16.h\"");
    const INTEGRAL: bool = false;
}
#[cfg(feature = "bfloat")]
impl DTypeOps for bf16 {}
#[cfg(feature = "bfloat")]
impl DType for bf16 {
    const ZERO: bf16 = bf16::from_f64_const(0.0);
    const ONE: bf16 = bf16::from_f64_const(1.0);
    const NAME: &'static str = "bf16";
    const C_NAME: &'static str = "__nv_bfloat16";
    const C_DEP: Option<&'static str> = Some("#include \"cuda_bf16.h\"");
    const INTEGRAL: bool = false;
}

pub trait SimdSupported {
    // In bytes, this is also the lane count in bytes
    const BLOCK_SIZE: usize = 8;

    fn binary_simd_op(lhs: &mut Vec<Self>, rhs: Vec<Self>, op: BinaryOpType)
    where
        Self: Sized;

    fn fma_op(lhs: &mut Vec<Self>, a: Vec<Self>, b: Vec<Self>)
    where
        Self: Sized;
}

macro_rules! simd_supported {
    ($t:ident BINARY_INTERNAL) => {
        fn binary_simd_op(lhs: &mut Vec<Self>, mut rhs: Vec<Self>, op: BinaryOpType)
        where
            Self: Sized,
        {
            // Pad to zeros
            let pad_count = lhs.len() % Self::BLOCK_SIZE;
            lhs.extend(vec![$t::ONE; Self::BLOCK_SIZE - pad_count]);
            rhs.extend(vec![$t::ONE; Self::BLOCK_SIZE - pad_count]);
            let n_blocks = lhs.len() / Self::BLOCK_SIZE;

            let mut lhs_simd: Vec<std::simd::Simd<$t, { Self::BLOCK_SIZE }>> =
                Vec::with_capacity(n_blocks);
            let mut rhs_simd: Vec<std::simd::Simd<$t, { Self::BLOCK_SIZE }>> =
                Vec::with_capacity(n_blocks);
            let l_blocks = lhs.chunks(Self::BLOCK_SIZE).collect::<Vec<_>>();
            let r_blocks = rhs.chunks(Self::BLOCK_SIZE).collect::<Vec<_>>();

            for (l_blk, r_blk) in l_blocks.iter().zip(&r_blocks) {
                lhs_simd.push(std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(
                    l_blk,
                ));
                rhs_simd.push(std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(
                    r_blk,
                ));
            }
            *lhs = lhs_simd
                .into_iter()
                .zip(rhs_simd)
                .map(|(lhs, rhs)| match op {
                    BinaryOpType::Add => lhs + rhs,
                    BinaryOpType::Mul => lhs * rhs,
                    BinaryOpType::Sub => lhs - rhs,
                    BinaryOpType::Div => lhs / rhs,
                })
                .enumerate()
                .flat_map(
                    |(i, x): (usize, std::simd::Simd<$t, { Self::BLOCK_SIZE }>)| {
                        // Handle undoing the padding
                        if i != n_blocks - 1 {
                            x.as_array().to_vec()
                        } else {
                            x.as_array()[0..pad_count].to_vec()
                        }
                    },
                )
                .collect::<Vec<_>>();
        }
    };

    ($t:ident FMA) => {
        impl SimdSupported for $t {
            simd_supported!($t BINARY_INTERNAL);

            fn fma_op(lhs: &mut Vec<Self>, mut a: Vec<Self>, mut b: Vec<Self>)
            where
                Self: Sized,
            {
                // Pad to zeros
                let pad_count = lhs.len() % Self::BLOCK_SIZE;
                lhs.extend(vec![$t::ONE; Self::BLOCK_SIZE - pad_count]);
                a.extend(vec![$t::ONE; Self::BLOCK_SIZE - pad_count]);
                b.extend(vec![$t::ONE; Self::BLOCK_SIZE - pad_count]);
                let n_blocks = lhs.len() / Self::BLOCK_SIZE;

                let mut lhs_simd: Vec<std::simd::Simd<$t, { Self::BLOCK_SIZE }>> =
                    Vec::with_capacity(n_blocks);
                let mut a_simd: Vec<std::simd::Simd<$t, { Self::BLOCK_SIZE }>> =
                    Vec::with_capacity(n_blocks);
                let mut b_simd: Vec<std::simd::Simd<$t, { Self::BLOCK_SIZE }>> =
                    Vec::with_capacity(n_blocks);
                let l_blocks = lhs.chunks(Self::BLOCK_SIZE).collect::<Vec<_>>();
                let a_blocks = a.chunks(Self::BLOCK_SIZE).collect::<Vec<_>>();
                let b_blocks = b.chunks(Self::BLOCK_SIZE).collect::<Vec<_>>();

                for (l_blk, (a_blk, b_blk)) in l_blocks
                    .into_iter()
                    .zip(a_blocks.into_iter().zip(b_blocks.into_iter()))
                {
                    lhs_simd.push(std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(
                        l_blk,
                    ));
                    a_simd.push(std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(
                        a_blk,
                    ));
                    b_simd.push(std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(
                        b_blk,
                    ));
                }
                use std::simd::StdFloat;
                *lhs = lhs_simd
                    .into_iter()
                    .zip(a_simd.into_iter().zip(b_simd))
                    .map(|(lhs, (a, b))| lhs.mul_add(a, b))
                    .enumerate()
                    .flat_map(
                        |(i, x): (usize, std::simd::Simd<$t, { Self::BLOCK_SIZE }>)| {
                            // Handle undoing the padding
                            if i != n_blocks - 1 {
                                x.as_array().to_vec()
                            } else {
                                x.as_array()[0..pad_count].to_vec()
                            }
                        },
                    )
                    .collect::<Vec<_>>();
            }
        }
    };
    ($t:ident NOFMA) => {
        impl SimdSupported for $t {
            simd_supported!($t BINARY_INTERNAL);

            fn fma_op(lhs: &mut Vec<Self>, mut a: Vec<Self>, mut b: Vec<Self>)
            where
                Self: Sized,
            {
                // Pad to zeros
                let pad_count = lhs.len() % Self::BLOCK_SIZE;
                lhs.extend(vec![$t::ONE; Self::BLOCK_SIZE - pad_count]);
                a.extend(vec![$t::ONE; Self::BLOCK_SIZE - pad_count]);
                b.extend(vec![$t::ONE; Self::BLOCK_SIZE - pad_count]);
                let n_blocks = lhs.len() / Self::BLOCK_SIZE;

                let mut lhs_simd: Vec<std::simd::Simd<$t, { Self::BLOCK_SIZE }>> =
                    Vec::with_capacity(n_blocks);
                let mut a_simd: Vec<std::simd::Simd<$t, { Self::BLOCK_SIZE }>> =
                    Vec::with_capacity(n_blocks);
                let mut b_simd: Vec<std::simd::Simd<$t, { Self::BLOCK_SIZE }>> =
                    Vec::with_capacity(n_blocks);
                let l_blocks = lhs.chunks(Self::BLOCK_SIZE).collect::<Vec<_>>();
                let a_blocks = a.chunks(Self::BLOCK_SIZE).collect::<Vec<_>>();
                let b_blocks = b.chunks(Self::BLOCK_SIZE).collect::<Vec<_>>();

                for (l_blk, (a_blk, b_blk)) in l_blocks
                    .into_iter()
                    .zip(a_blocks.into_iter().zip(b_blocks.into_iter()))
                {
                    lhs_simd.push(std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(
                        l_blk,
                    ));
                    a_simd.push(std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(
                        a_blk,
                    ));
                    b_simd.push(std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(
                        b_blk,
                    ));
                }

                *lhs = lhs_simd
                    .into_iter()
                    .zip(a_simd.into_iter().zip(b_simd))
                    .map(|(lhs, (a, b))| lhs * a + b)
                    .enumerate()
                    .flat_map(
                        |(i, x): (usize, std::simd::Simd<$t, { Self::BLOCK_SIZE }>)| {
                            // Handle undoing the padding
                            if i != n_blocks - 1 {
                                x.as_array().to_vec()
                            } else {
                                x.as_array()[0..pad_count].to_vec()
                            }
                        },
                    )
                    .collect::<Vec<_>>();
            }
        }
    };
    ($t:ident NOSIMD) => {
        impl SimdSupported for $t {
            fn binary_simd_op(lhs: &mut Vec<Self>, rhs: Vec<Self>, op: BinaryOpType)
            where
                Self: Sized,
            {
                *lhs = lhs
                    .into_iter()
                    .zip(rhs)
                    .map(|(lhs, rhs)| match op {
                        BinaryOpType::Add => *lhs + rhs,
                        BinaryOpType::Mul => *lhs * rhs,
                        BinaryOpType::Sub => *lhs - rhs,
                        BinaryOpType::Div => *lhs / rhs,
                    })
                    .collect::<Vec<_>>();
            }

            fn fma_op(lhs: &mut Vec<Self>, a: Vec<Self>, b: Vec<Self>)
            where
                Self: Sized,
            {
                *lhs = lhs
                    .into_iter()
                    .zip(a.into_iter().zip(b))
                    .map(|(lhs, (a, b))| *lhs * a + b)
                    .collect::<Vec<_>>();
            }
        }
    };
}

simd_supported!(f32 FMA);
simd_supported!(f64 FMA);
simd_supported!(u8 NOFMA);
simd_supported!(u32 NOFMA);
simd_supported!(i32 NOFMA);
simd_supported!(i64 NOFMA);

#[cfg(feature = "half")]
simd_supported!(f16 NOSIMD);
#[cfg(feature = "bfloat")]
simd_supported!(bf16 NOSIMD);
