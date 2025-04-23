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

use crate::graph::BinaryOpType;

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

#[cfg(not(feature = "cuda"))]
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

pub trait SimdSupported {
    // In bytes, this is also the lane count in bytes
    const BLOCK_SIZE: usize = 8;

    fn binary_simd_op(a: &[Self], b: &[Self], out: &mut Vec<Self>, op: BinaryOpType)
    where
        Self: Sized;

    fn binary_simd_op_inplace_lhs(a: &mut [Self], b: &[Self], op: BinaryOpType)
    where
        Self: Sized;

    fn binary_simd_op_inplace_rhs(a: &[Self], b: &mut [Self], op: BinaryOpType)
    where
        Self: Sized;

    fn fma_op(a: &[Self], b: &[Self], c: &[Self], out: &mut Vec<Self>)
    where
        Self: Sized;

    fn fma_op_inplace_a(a: &mut [Self], b: &[Self], c: &[Self])
    where
        Self: Sized;

    fn fma_op_inplace_b(a: &[Self], b: &mut [Self], c: &[Self])
    where
        Self: Sized;

    fn fma_op_inplace_c(a: &[Self], b: &[Self], c: &mut [Self])
    where
        Self: Sized;
}

macro_rules! simd_supported {
    ($t:ident BINARY_INTERNAL) => {
        fn binary_simd_op(a: &[Self], b: &[Self], out: &mut Vec<Self>, op: BinaryOpType)
        where
            Self: Sized,
        {
            let len = a.len();
            let n_blocks = len / Self::BLOCK_SIZE;

            // Define SIMD and scalar operations based on the chosen operation
            let simd_op = |l: std::simd::Simd<$t, { Self::BLOCK_SIZE }>,
                           r: std::simd::Simd<$t, { Self::BLOCK_SIZE }>| {
                match op {
                    BinaryOpType::Add => l + r,
                    BinaryOpType::Mul => l * r,
                    BinaryOpType::Sub => l - r,
                    BinaryOpType::Div => l / r,
                }
            };
            let scalar_op = |l: Self, r: Self| {
                match op {
                    BinaryOpType::Add => l + r,
                    BinaryOpType::Mul => l * r,
                    BinaryOpType::Sub => l - r,
                    BinaryOpType::Div => l / r,
                }
            };

            // Vectorized loop
            for i in 0..n_blocks {
                let off = i * Self::BLOCK_SIZE;
                let l_chunk = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&a[off..off + Self::BLOCK_SIZE]);
                let r_chunk = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&b[off..off + Self::BLOCK_SIZE]);
                let res = simd_op(l_chunk, r_chunk);
                out[off..(off + Self::BLOCK_SIZE).min(len)].copy_from_slice(res.as_array());
            }
            // Scalar fallback for remainder
            for i in n_blocks * Self::BLOCK_SIZE..len {
                out[i] = scalar_op(a[i], b[i]);
            }
        }

        fn binary_simd_op_inplace_lhs(a: &mut [Self], b: &[Self], op: BinaryOpType)
        where
            Self: Sized,
        {
            let len = a.len();
            let n_blocks = len / Self::BLOCK_SIZE;

            // Define SIMD and scalar operations based on the chosen operation
            let simd_op = |l: std::simd::Simd<$t, { Self::BLOCK_SIZE }>,
                           r: std::simd::Simd<$t, { Self::BLOCK_SIZE }>| {
                match op {
                    BinaryOpType::Add => l + r,
                    BinaryOpType::Mul => l * r,
                    BinaryOpType::Sub => l - r,
                    BinaryOpType::Div => l / r,
                }
            };
            let scalar_op = |l: Self, r: Self| {
                match op {
                    BinaryOpType::Add => l + r,
                    BinaryOpType::Mul => l * r,
                    BinaryOpType::Sub => l - r,
                    BinaryOpType::Div => l / r,
                }
            };

            // Vectorized loop
            for i in 0..n_blocks {
                let off = i * Self::BLOCK_SIZE;
                let l_chunk = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&a[off..off + Self::BLOCK_SIZE]);
                let r_chunk = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&b[off..off + Self::BLOCK_SIZE]);
                let res = simd_op(l_chunk, r_chunk);
                a[off..(off + Self::BLOCK_SIZE).min(len)].copy_from_slice(res.as_array());
            }
            // Scalar fallback for remainder
            for i in n_blocks * Self::BLOCK_SIZE..len {
                a[i] = scalar_op(a[i], b[i]);
            }
        }

        fn binary_simd_op_inplace_rhs(a: &[Self], b: &mut [Self], op: BinaryOpType)
        where
            Self: Sized,
        {
            let len = a.len();
            let n_blocks = len / Self::BLOCK_SIZE;

            // Define SIMD and scalar operations based on the chosen operation
            let simd_op = |l: std::simd::Simd<$t, { Self::BLOCK_SIZE }>,
                           r: std::simd::Simd<$t, { Self::BLOCK_SIZE }>| {
                match op {
                    BinaryOpType::Add => l + r,
                    BinaryOpType::Mul => l * r,
                    BinaryOpType::Sub => l - r,
                    BinaryOpType::Div => l / r,
                }
            };
            let scalar_op = |l: Self, r: Self| {
                match op {
                    BinaryOpType::Add => l + r,
                    BinaryOpType::Mul => l * r,
                    BinaryOpType::Sub => l - r,
                    BinaryOpType::Div => l / r,
                }
            };

            // Vectorized loop
            for i in 0..n_blocks {
                let off = i * Self::BLOCK_SIZE;
                let l_chunk = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&a[off..off + Self::BLOCK_SIZE]);
                let r_chunk = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&b[off..off + Self::BLOCK_SIZE]);
                let res = simd_op(l_chunk, r_chunk);
                b[off..(off + Self::BLOCK_SIZE).min(len)].copy_from_slice(res.as_array());
            }
            // Scalar fallback for remainder
            for i in n_blocks * Self::BLOCK_SIZE..len {
                b[i] = scalar_op(a[i], b[i]);
            }
        }
    };

    ($t:ident FMA) => {
        impl SimdSupported for $t {
            simd_supported!($t BINARY_INTERNAL);

            fn fma_op(a: &[Self], b: &[Self], c: &[Self], out: &mut Vec<Self>)
            where
                Self: Sized,
            {
                let len = a.len();
                let n_blocks = len / Self::BLOCK_SIZE;

                use std::simd::StdFloat;
                for i in 0..n_blocks {
                    let off = i * Self::BLOCK_SIZE;
                    let a = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&a[off..off + Self::BLOCK_SIZE]);
                    let b = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&b[off..off + Self::BLOCK_SIZE]);
                    let c = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&c[off..off + Self::BLOCK_SIZE]);
                    let res = a.mul_add(b, c);
                    out[off..off + Self::BLOCK_SIZE].copy_from_slice(res.as_array());
                }
                for i in n_blocks * Self::BLOCK_SIZE..len {
                    out[i] = a[i].mul_add(b[i], c[i]);
                }
            }

            fn fma_op_inplace_a(a: &mut [Self], b: &[Self], c: &[Self])
            where
                Self: Sized,
            {
                let len = a.len();
                let n_blocks = len / Self::BLOCK_SIZE;

                use std::simd::StdFloat;
                for i in 0..n_blocks {
                    let off = i * Self::BLOCK_SIZE;
                    let ax = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&a[off..off + Self::BLOCK_SIZE]);
                    let b = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&b[off..off + Self::BLOCK_SIZE]);
                    let c = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&c[off..off + Self::BLOCK_SIZE]);
                    let res = ax.mul_add(b, c);
                    a[off..off + Self::BLOCK_SIZE].copy_from_slice(res.as_array());
                }
                for i in n_blocks * Self::BLOCK_SIZE..len {
                    a[i] = a[i].mul_add(b[i], c[i]);
                }
            }

            fn fma_op_inplace_b(a: &[Self], b: &mut [Self], c: &[Self])
            where
                Self: Sized,
            {
                let len = a.len();
                let n_blocks = len / Self::BLOCK_SIZE;

                use std::simd::StdFloat;
                for i in 0..n_blocks {
                    let off = i * Self::BLOCK_SIZE;
                    let a = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&a[off..off + Self::BLOCK_SIZE]);
                    let bx = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&b[off..off + Self::BLOCK_SIZE]);
                    let c = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&c[off..off + Self::BLOCK_SIZE]);
                    let res = a.mul_add(bx, c);
                    b[off..off + Self::BLOCK_SIZE].copy_from_slice(res.as_array());
                }
                for i in n_blocks * Self::BLOCK_SIZE..len {
                    b[i] = a[i].mul_add(b[i], c[i]);
                }
            }

            fn fma_op_inplace_c(a: &[Self], b: &[Self], c: &mut [Self])
            where
                Self: Sized,
            {
                let len = a.len();
                let n_blocks = len / Self::BLOCK_SIZE;

                use std::simd::StdFloat;
                for i in 0..n_blocks {
                    let off = i * Self::BLOCK_SIZE;
                    let a = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&a[off..off + Self::BLOCK_SIZE]);
                    let b = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&b[off..off + Self::BLOCK_SIZE]);
                    let cx = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&c[off..off + Self::BLOCK_SIZE]);
                    let res = a.mul_add(b, cx);
                    c[off..off + Self::BLOCK_SIZE].copy_from_slice(res.as_array());
                }
                for i in n_blocks * Self::BLOCK_SIZE..len {
                    c[i] = c[i].mul_add(b[i], c[i]);
                }
            }
        }
    };
    ($t:ident NOFMA) => {
        impl SimdSupported for $t {
            simd_supported!($t BINARY_INTERNAL);

            fn fma_op(a: &[Self], b: &[Self], c: &[Self], out: &mut Vec<Self>)
            where
                Self: Sized,
            {
                let len = a.len();
                let n_blocks = len / Self::BLOCK_SIZE;

                for i in 0..n_blocks {
                    let off = i * Self::BLOCK_SIZE;
                    let a = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&a[off..off + Self::BLOCK_SIZE]);
                    let b = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&b[off..off + Self::BLOCK_SIZE]);
                    let c = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&c[off..off + Self::BLOCK_SIZE]);
                    let res = a * b + c;
                    out[off..off + Self::BLOCK_SIZE].copy_from_slice(res.as_array());
                }
                for i in n_blocks * Self::BLOCK_SIZE..len {
                    out[i] = a[i] * b[i] + c[i];
                }
            }

            fn fma_op_inplace_a(a: &mut [Self], b: &[Self], c: &[Self])
            where
                Self: Sized,
            {
                let len = a.len();
                let n_blocks = len / Self::BLOCK_SIZE;

                for i in 0..n_blocks {
                    let off = i * Self::BLOCK_SIZE;
                    let ax = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&a[off..off + Self::BLOCK_SIZE]);
                    let b = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&b[off..off + Self::BLOCK_SIZE]);
                    let c = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&c[off..off + Self::BLOCK_SIZE]);
                    let res = ax * b + c;
                    a[off..off + Self::BLOCK_SIZE].copy_from_slice(res.as_array());
                }
                for i in n_blocks * Self::BLOCK_SIZE..len {
                    a[i] = a[i] * b[i] + c[i];
                }
            }

            fn fma_op_inplace_b(a: &[Self], b: &mut [Self], c: &[Self])
            where
                Self: Sized,
            {
                let len = a.len();
                let n_blocks = len / Self::BLOCK_SIZE;

                for i in 0..n_blocks {
                    let off = i * Self::BLOCK_SIZE;
                    let a = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&a[off..off + Self::BLOCK_SIZE]);
                    let bx = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&b[off..off + Self::BLOCK_SIZE]);
                    let c = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&c[off..off + Self::BLOCK_SIZE]);
                    let res = a * bx + c;
                    b[off..off + Self::BLOCK_SIZE].copy_from_slice(res.as_array());
                }
                for i in n_blocks * Self::BLOCK_SIZE..len {
                    b[i] = a[i] * b[i] + c[i];
                }
            }

            fn fma_op_inplace_c(a: &[Self], b: &[Self], c: &mut [Self])
            where
                Self: Sized,
            {
                let len = a.len();
                let n_blocks = len / Self::BLOCK_SIZE;

                for i in 0..n_blocks {
                    let off = i * Self::BLOCK_SIZE;
                    let a = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&a[off..off + Self::BLOCK_SIZE]);
                    let b = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&b[off..off + Self::BLOCK_SIZE]);
                    let cx = std::simd::Simd::<$t, { Self::BLOCK_SIZE }>::from_slice(&c[off..off + Self::BLOCK_SIZE]);
                    let res = a * b + cx;
                    c[off..off + Self::BLOCK_SIZE].copy_from_slice(res.as_array());
                }
                for i in n_blocks * Self::BLOCK_SIZE..len {
                    c[i] += a[i] * b[i];
                }
            }
        }
    };
    ($t:ident NOSIMD) => {
        impl SimdSupported for $t {
            fn binary_simd_op(a: &[Self], b: &[Self], out: &mut Vec<Self>, op: BinaryOpType)
            where
                Self: Sized,
            {
                use rayon::prelude::*;

                out.par_iter_mut()
                    .zip(a.par_iter().zip(b))
                    .for_each(|(out, (lhs, rhs))| *out = match op {
                        BinaryOpType::Add => *lhs + rhs,
                        BinaryOpType::Mul => *lhs * rhs,
                        BinaryOpType::Sub => *lhs - rhs,
                        BinaryOpType::Div => *lhs / rhs,
                    });
            }

            fn binary_simd_op_inplace_lhs(a: &mut [Self], b: &[Self], op: BinaryOpType)
            where
                Self: Sized,
            {
                use rayon::prelude::*;

                a.par_iter_mut().zip(b)
                    .for_each(|(lhs, rhs)| *lhs = match op {
                        BinaryOpType::Add => *lhs + rhs,
                        BinaryOpType::Mul => *lhs * rhs,
                        BinaryOpType::Sub => *lhs - rhs,
                        BinaryOpType::Div => *lhs / rhs,
                    });
            }

            fn binary_simd_op_inplace_rhs(a: &[Self], b: &mut[Self], op: BinaryOpType)
            where
                Self: Sized,
            {
                use rayon::prelude::*;

                b.par_iter_mut().zip(a)
                    .for_each(|(rhs, lhs)| *rhs = match op {
                        BinaryOpType::Add => *lhs + *rhs,
                        BinaryOpType::Mul => *lhs * *rhs,
                        BinaryOpType::Sub => *lhs - *rhs,
                        BinaryOpType::Div => *lhs / *rhs,
                    });
            }

            fn fma_op(a: &[Self], b: &[Self], c: &[Self], out: &mut Vec<Self>)
            where
                Self: Sized,
            {
                use rayon::prelude::*;

                out.par_iter_mut()
                    .zip(a.par_iter().zip(b.par_iter().zip(c)))
                    .for_each(|(out, (a, (b, c)))| *out = *a * *b + *c);
            }

            fn fma_op_inplace_a(a: &mut [Self], b: &[Self], c: &[Self])
            where
                Self: Sized,
            {
                use rayon::prelude::*;

                a.par_iter_mut()
                    .zip(b.par_iter().zip(c))
                    .for_each(|(a, (b, c))| *a = *a * *b + *c);
            }

            fn fma_op_inplace_b(a: &[Self], b: &mut [Self], c: &[Self])
            where
                Self: Sized,
            {
                use rayon::prelude::*;

                b.par_iter_mut()
                    .zip(a.par_iter().zip(c))
                    .for_each(|(b, (a, c))| *b = *a * *b + *c);
            }

            fn fma_op_inplace_c(a: &[Self], b: &[Self], c: &mut [Self])
            where
                Self: Sized,
            {
                use rayon::prelude::*;

                c.par_iter_mut()
                    .zip(a.par_iter().zip(b))
                    .for_each(|(c, (a, b))| *c = *a * *b + *c);
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
