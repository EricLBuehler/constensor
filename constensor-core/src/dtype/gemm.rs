use gemm::{gemm, Parallelism};

#[cfg(feature = "bfloat")]
use half::bf16;
#[cfg(feature = "half")]
use half::f16;

pub trait GemmDispatch {
    // In bytes, this is also the lane count in bytes
    const BLOCK_SIZE: usize = 8;

    #[allow(clippy::too_many_arguments)]
    // Matrix multiplication: (B x M x K) * (B x K x N) = (B x M x N)
    fn launch_gemm(
        lhs: &[Self],
        rhs: &[Self],
        b: usize,
        m: usize,
        n: usize,
        k: usize,
        out: &mut Vec<Self>,
        alpha: Self,
        beta: Self,
    ) where
        Self: Sized;
}

macro_rules! instantiate_gemm {
    ($rt:ident, $init:expr, NAIVE) => {
        impl GemmDispatch for $rt {
            fn launch_gemm(
                lhs: &[Self],
                rhs: &[Self],
                b: usize,
                m: usize,
                n: usize,
                k: usize,
                out: &mut Vec<Self>,
                alpha: Self,
                beta: Self,
            ) where
                Self: Sized,
            {
                for b in 0..b {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = $init;
                            for p in 0..k {
                                sum +=
                                    beta * lhs[b * m * k + i * k + p] * rhs[b * k * n + p * n + j];
                            }
                            out[b * m * n + i * n + j] = alpha * out[b * m * n + i * n + j] + sum;
                        }
                    }
                }
            }
        }
    };

    ($rt:ident, $zero:expr,  GEMM) => {
        impl GemmDispatch for $rt {
            fn launch_gemm(
                lhs: &[Self],
                rhs: &[Self],
                b: usize,
                m: usize,
                n: usize,
                k: usize,
                out: &mut Vec<Self>,
                alpha: Self,
                beta: Self,
            ) where
                Self: Sized,
            {
                let num_threads = num_cpus::get();
                let parallelism = if num_threads > 1 {
                    Parallelism::Rayon(num_threads)
                } else {
                    Parallelism::None
                };

                // cs = stride[-1], rs = stride[-2]
                let dst_cs = 1;
                let dst_rs = n;

                let lhs_cs = 1;
                let lhs_rs = k;

                let rhs_cs = 1;
                let rhs_rs = n;

                let read_dst = alpha != $zero;

                for b in 0..b {
                    let lhs_p = &lhs[b * m * k..];
                    let rhs_p = &rhs[b * k * n..];
                    let out_p = &mut out[b * m * n..];

                    unsafe {
                        gemm(
                            /* m: usize = */ m,
                            /* n: usize = */ n,
                            /* k: usize = */ k,
                            /* dst: *mut T = */ out_p.as_mut_ptr(),
                            /* dst_cs: isize = */ dst_cs as isize,
                            /* dst_rs: isize = */ dst_rs as isize,
                            /* read_dst: bool = */ read_dst,
                            /* lhs: *const T = */ lhs_p.as_ptr(),
                            /* lhs_cs: isize = */ lhs_cs as isize,
                            /* lhs_rs: isize = */ lhs_rs as isize,
                            /* rhs: *const T = */ rhs_p.as_ptr(),
                            /* rhs_cs: isize = */ rhs_cs as isize,
                            /* rhs_rs: isize = */ rhs_rs as isize,
                            /* alpha: T = */ alpha,
                            /* beta: T = */ beta,
                            /* conj_dst: bool = */ false,
                            /* conj_lhs: bool = */ false,
                            /* conj_rhs: bool = */ false,
                            parallelism,
                        )
                    }
                }
            }
        }
    };
    // SIMD-accelerated gemm using SimdSupported for vectorized operations along 'n' dimension
    ($rt:ident, $init:expr, SIMD) => {
        impl GemmDispatch for $rt {
            fn launch_gemm(
                lhs: &[Self],
                rhs: &[Self],
                b: usize,
                m: usize,
                n: usize,
                k: usize,
                out: &mut Vec<Self>,
                alpha: Self,
                beta: Self,
            ) where
                Self: Sized,
            {
                use crate::dtype::SimdSupported;
                use crate::graph::BinaryOpType;
                // number of lanes for vectorization
                const BLOCK_SIZE: usize = <$rt as SimdSupported>::BLOCK_SIZE;
                let n_blocks = n / BLOCK_SIZE;
                let rem = n % BLOCK_SIZE;

                // SAFETY: for all the unsafe blocks below, we are checking the bounds here.
                let lhs_len = lhs.len();
                let rhs_len = rhs.len();
                let out_len = out.len();
                debug_assert_eq!(lhs_len, b * m * k);
                debug_assert_eq!(rhs_len, b * k * n);
                debug_assert_eq!(out_len, b * m * n);

                for batch in 0..b {
                    let lhs_p = unsafe {
                        let out_ptr = lhs.as_ptr().add(batch * m * k);
                        std::slice::from_raw_parts(out_ptr, lhs_len - batch * m * k)
                    };
                    let rhs_p = unsafe {
                        let out_ptr = rhs.as_ptr().add(batch * k * n);
                        std::slice::from_raw_parts(out_ptr, rhs_len - batch * k * n)
                    };
                    let out_p = unsafe {
                        let out_ptr = out.as_mut_ptr().add(batch * m * n);
                        std::slice::from_raw_parts_mut(out_ptr, out_len - batch * m * n)
                    };

                    for i in 0..m {
                        // mutable slice for current output row
                        let out_row = unsafe {
                            let out_ptr = out_p.as_mut_ptr().add(i * n);
                            std::slice::from_raw_parts_mut(out_ptr, n)
                        };
                        // process full vector blocks
                        for block in 0..n_blocks {
                            let off = block * BLOCK_SIZE;
                            // initialize or scale existing output
                            let out_chunk = unsafe {
                                let out_ptr = out_row.as_mut_ptr().add(off);
                                std::slice::from_raw_parts_mut(out_ptr, BLOCK_SIZE)
                            };
                            if beta != $init {
                                // scale by alpha: out = alpha * out
                                let alpha_arr = [alpha; BLOCK_SIZE];
                                <Self as SimdSupported>::binary_simd_op_inplace_lhs(
                                    out_chunk,
                                    &alpha_arr,
                                    BinaryOpType::Mul,
                                );
                            } else {
                                // initialize to zero
                                for x in out_chunk.iter_mut() {
                                    *x = $init;
                                }
                            }
                            // accumulate dot-product contributions
                            for p in 0..k {
                                let a_val = lhs_p[i * k + p];
                                let a_arr = [a_val; BLOCK_SIZE];
                                let b_chunk = unsafe {
                                    let out_ptr = rhs_p.as_ptr().add(p * n + off);
                                    std::slice::from_raw_parts(out_ptr, BLOCK_SIZE)
                                };
                                <Self as SimdSupported>::fma_op_inplace_c(
                                    &a_arr, b_chunk, out_chunk,
                                );
                            }
                        }
                        // handle remainder elements
                        if rem > 0 {
                            let off = n_blocks * BLOCK_SIZE;
                            let out_chunk = unsafe {
                                let out_ptr = out_row.as_mut_ptr().add(off);
                                std::slice::from_raw_parts_mut(out_ptr, rem)
                            };
                            if beta != $init {
                                for x in out_chunk.iter_mut() {
                                    *x *= alpha;
                                }
                            } else {
                                for x in out_chunk.iter_mut() {
                                    *x = $init;
                                }
                            }
                            for p in 0..k {
                                let a_val = lhs_p[i * k + p];
                                for j in 0..rem {
                                    out_chunk[j] += a_val * rhs_p[p * n + off + j];
                                }
                            }
                        }
                    }
                }
            }
        }
    };
}

instantiate_gemm!(u8, 0, SIMD);
instantiate_gemm!(u32, 0, SIMD);
instantiate_gemm!(i32, 0, SIMD);
instantiate_gemm!(i64, 0, SIMD);
instantiate_gemm!(f32, 0., GEMM);
instantiate_gemm!(f64, 0., GEMM);
#[cfg(feature = "bfloat")]
instantiate_gemm!(bf16, bf16::from_f32(0.), SIMD);
#[cfg(feature = "half")]
instantiate_gemm!(f16, f16::from_f32(0.), GEMM);
