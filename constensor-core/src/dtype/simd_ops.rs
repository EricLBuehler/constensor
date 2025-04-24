#[cfg(feature = "bfloat")]
use half::bf16;
#[cfg(feature = "half")]
use half::f16;

use crate::graph::BinaryOpType;

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
                // SAFETY: the invariant is upheld with the loop condition
                let l_chunk: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                    std::ptr::read_unaligned(a.as_ptr().add(off) as *const _)
                };
                let r_chunk: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                    std::ptr::read_unaligned(b.as_ptr().add(off) as *const _)
                };
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
                // SAFETY: the invariant is upheld with the loop condition
                let l_chunk: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                    std::ptr::read_unaligned(a.as_ptr().add(off) as *const _)
                };
                let r_chunk: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                    std::ptr::read_unaligned(b.as_ptr().add(off) as *const _)
                };
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
                // SAFETY: the invariant is upheld with the loop condition
                let l_chunk: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                    std::ptr::read_unaligned(a.as_ptr().add(off) as *const _)
                };
                let r_chunk: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                    std::ptr::read_unaligned(b.as_ptr().add(off) as *const _)
                };
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
                    // SAFETY: the invariant is upheld with the loop condition
                    let a: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(a.as_ptr().add(off) as *const _)
                    };
                    let b: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(b.as_ptr().add(off) as *const _)
                    };
                    let c: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(c.as_ptr().add(off) as *const _)
                    };
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
                    // SAFETY: the invariant is upheld with the loop condition
                    let ax: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(a.as_ptr().add(off) as *const _)
                    };
                    let b: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(b.as_ptr().add(off) as *const _)
                    };
                    let c: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(c.as_ptr().add(off) as *const _)
                    };
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
                    // SAFETY: the invariant is upheld with the loop condition
                    let a: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(a.as_ptr().add(off) as *const _)
                    };
                    let bx: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(b.as_ptr().add(off) as *const _)
                    };
                    let c: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(c.as_ptr().add(off) as *const _)
                    };
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
                    // SAFETY: the invariant is upheld with the loop condition
                    let a: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(a.as_ptr().add(off) as *const _)
                    };
                    let b: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(b.as_ptr().add(off) as *const _)
                    };
                    let cx: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(c.as_ptr().add(off) as *const _)
                    };
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
                    // SAFETY: the invariant is upheld with the loop condition
                    let a: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(a.as_ptr().add(off) as *const _)
                    };
                    let b: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(b.as_ptr().add(off) as *const _)
                    };
                    let c: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(c.as_ptr().add(off) as *const _)
                    };
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
                    // SAFETY: the invariant is upheld with the loop condition
                    let ax: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(a.as_ptr().add(off) as *const _)
                    };
                    let b: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(b.as_ptr().add(off) as *const _)
                    };
                    let c: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(c.as_ptr().add(off) as *const _)
                    };
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
                    // SAFETY: the invariant is upheld with the loop condition
                    let a: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(a.as_ptr().add(off) as *const _)
                    };
                    let bx: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(b.as_ptr().add(off) as *const _)
                    };
                    let c: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(c.as_ptr().add(off) as *const _)
                    };
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
                    // SAFETY: the invariant is upheld with the loop condition
                    let a: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(a.as_ptr().add(off) as *const _)
                    };
                    let b: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(b.as_ptr().add(off) as *const _)
                    };
                    let cx: std::simd::Simd<$t, { Self::BLOCK_SIZE }> = unsafe {
                        std::ptr::read_unaligned(c.as_ptr().add(off) as *const _)
                    };
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
