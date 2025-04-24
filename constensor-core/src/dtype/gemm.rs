use gemm::{gemm, Parallelism};

#[cfg(feature = "bfloat")]
use half::bf16;
#[cfg(feature = "half")]
use half::f16;

pub trait GemmDispatch {
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

    ($rt:ident, GEMM) => {
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
                            /* read_dst: bool = */ false,
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
}

instantiate_gemm!(u8, 0, NAIVE);
instantiate_gemm!(u32, 0, NAIVE);
instantiate_gemm!(i32, 0, NAIVE);
instantiate_gemm!(i64, 0, NAIVE);
instantiate_gemm!(f32, GEMM);
instantiate_gemm!(f64, GEMM);
#[cfg(feature = "bfloat")]
instantiate_gemm!(bf16, bf16::from_f32(0.), NAIVE);
#[cfg(feature = "half")]
instantiate_gemm!(f16, GEMM);
