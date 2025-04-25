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
        lhs_stride: &[usize],
        rhs: &[Self],
        rhs_stride: &[usize],
        b: usize,
        m: usize,
        n: usize,
        k: usize,
        out: &mut Vec<Self>,
        out_stride: &[usize],
        alpha: Self,
        beta: Self,
    ) where
        Self: Sized;

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    // Matrix multiplication: (B x M x K) * (B x K x N) = (B x M x N)
    fn launch_gemm_cuda(
        cublas: &cudarc::cublas::CudaBlas,
        lhs: &cudarc::driver::CudaSlice<Self>,
        rhs: &cudarc::driver::CudaSlice<Self>,
        b: usize,
        m: usize,
        n: usize,
        k: usize,
        out: &mut cudarc::driver::CudaSlice<Self>,
        alpha: Self,
        beta: Self,
    ) -> crate::Result<()>
    where
        Self: Sized;
}

macro_rules! instantiate_gemm_cuda {
    (u8) => {
        instantiate_gemm_cuda!(__instantiate_fail);
    };
    (u32) => {
        instantiate_gemm_cuda!(__instantiate_fail);
    };
    (i32) => {
        instantiate_gemm_cuda!(__instantiate_fail);
    };
    (i64) => {
        instantiate_gemm_cuda!(__instantiate_fail);
    };

    (__instantiate_fail) => {
        #[cfg(feature = "cuda")]
        fn launch_gemm_cuda(
            _cublas: &cudarc::cublas::CudaBlas,
            _lhs: &cudarc::driver::CudaSlice<Self>,
            _rhs: &cudarc::driver::CudaSlice<Self>,
            _b: usize,
            _m: usize,
            _n: usize,
            _k: usize,
            _out: &mut cudarc::driver::CudaSlice<Self>,
            _alpha: Self,
            _beta: Self,
        ) -> crate::Result<()>
        where
            Self: Sized,
        {
            panic!("`launch_gemm_cuda` called with invalid configuration (w/o CUDA, dtype)")
        }
    };

    ($rt:ident) => {
        #[cfg(feature = "cuda")]
        fn launch_gemm_cuda(
            cublas: &cudarc::cublas::CudaBlas,
            lhs: &cudarc::driver::CudaSlice<$rt>,
            rhs: &cudarc::driver::CudaSlice<$rt>,
            b: usize,
            m: usize,
            n: usize,
            k: usize,
            out: &mut cudarc::driver::CudaSlice<$rt>,
            alpha: $rt,
            beta: $rt,
        ) -> crate::Result<()> {
            use crate::cuda_backend::error::WrapErr;
            use cudarc::cublas::Gemm;

            let gemm_cfg = crate::cuda_backend::util::gemm_config(alpha, beta, (b, m, n, k))?;

            unsafe {
                cublas
                    .gemm_strided_batched(
                        gemm_cfg,
                        &lhs.as_view(),
                        &rhs.as_view(),
                        &mut out.as_view_mut(),
                    )
                    .w()?;
            }

            Ok(())
        }
    };
}

macro_rules! instantiate_gemm {
    ($rt:ident, $init:expr, NAIVE) => {
        impl GemmDispatch for $rt {
            fn launch_gemm(
                lhs: &[Self],
                lhs_stride: &[usize],
                rhs: &[Self],
                rhs_stride: &[usize],
                b: usize,
                m: usize,
                n: usize,
                k: usize,
                out: &mut Vec<Self>,
                out_stride: &[usize],
                alpha: Self,
                beta: Self,
            ) where
                Self: Sized,
            {
                let lhs_bs = lhs_stride[0];
                let lhs_rs = lhs_stride[1];
                let lhs_cs = lhs_stride[2];

                let rhs_bs = rhs_stride[0];
                let rhs_rs = rhs_stride[1];
                let rhs_cs = rhs_stride[2];

                let out_bs = out_stride[0];
                let out_rs = out_stride[1];
                let out_cs = out_stride[2];

                for batch_idx in 0..b {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = $init;
                            for p in 0..k {
                                let lhs_val = lhs[batch_idx * lhs_bs + i * lhs_rs + p * lhs_cs];
                                let rhs_val = rhs[batch_idx * rhs_bs + p * rhs_rs + j * rhs_cs];
                                sum += beta * lhs_val * rhs_val;
                            }
                            let out_idx = batch_idx * out_bs + i * out_rs + j * out_cs;
                            out[out_idx] = alpha * out[out_idx] + sum;
                        }
                    }
                }
            }

            instantiate_gemm_cuda!($rt);
        }
    };

    ($rt:ident, $zero:expr,  GEMM) => {
        impl GemmDispatch for $rt {
            fn launch_gemm(
                lhs: &[Self],
                lhs_stride: &[usize],
                rhs: &[Self],
                rhs_stride: &[usize],
                b: usize,
                m: usize,
                n: usize,
                k: usize,
                out: &mut Vec<Self>,
                out_stride: &[usize],
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

                debug_assert_eq!(lhs.len(), b * m * k);
                debug_assert_eq!(lhs_stride.len(), 3);
                debug_assert_eq!(rhs.len(), b * k * n);
                debug_assert_eq!(rhs_stride.len(), 3);
                debug_assert_eq!(out.len(), b * m * n);
                debug_assert_eq!(out_stride.len(), 3);

                // cs = stride[-1], rs = stride[-2]
                let dst_cs = out_stride[2];
                let dst_rs = out_stride[1];

                let lhs_cs = lhs_stride[2];
                let lhs_rs = lhs_stride[1];

                let rhs_cs = rhs_stride[2];
                let rhs_rs = rhs_stride[1];

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

            instantiate_gemm_cuda!($rt);
        }
    };
    // SIMD-accelerated gemm using SimdSupported for vectorized operations along 'n' dimension
    ($rt:ident, $init:expr, SIMD) => {
        impl GemmDispatch for $rt {
            fn launch_gemm(
                lhs: &[Self],
                lhs_stride: &[usize],
                rhs: &[Self],
                rhs_stride: &[usize],
                b: usize,
                m: usize,
                n: usize,
                k: usize,
                out: &mut Vec<Self>,
                out_stride: &[usize],
                alpha: Self,
                beta: Self,
            ) where
                Self: Sized,
            {
                use crate::dtype::SimdSupported;
                use crate::graph::BinaryOpType;
                const BLOCK_SIZE: usize = <$rt as SimdSupported>::BLOCK_SIZE;
                let n_blocks = n / BLOCK_SIZE;
                let rem = n % BLOCK_SIZE;

                let lhs_bs = lhs_stride[0];
                let lhs_rs = lhs_stride[1];
                let lhs_cs = lhs_stride[2];

                let rhs_bs = rhs_stride[0];
                let rhs_rs = rhs_stride[1];
                let rhs_cs = rhs_stride[2];

                let out_bs = out_stride[0];
                let out_rs = out_stride[1];
                let out_cs = out_stride[2];

                debug_assert_eq!(lhs.len(), b * m * k);
                debug_assert_eq!(lhs_stride.len(), 3);
                debug_assert_eq!(rhs.len(), b * k * n);
                debug_assert_eq!(rhs_stride.len(), 3);
                debug_assert_eq!(out.len(), b * m * n);
                debug_assert_eq!(out_stride.len(), 3);

                for batch in 0..b {
                    // Compute base pointers once per batch
                    let lhs_base = unsafe { lhs.as_ptr().add(batch * lhs_bs) };
                    let rhs_base = unsafe { rhs.as_ptr().add(batch * rhs_bs) };
                    let out_base = unsafe { out.as_mut_ptr().add(batch * out_bs) };

                    for i in 0..m {
                        // Pointer to the start of the current output row
                        let out_row_ptr = unsafe { out_base.add(i * out_rs) };

                        // Process full SIMD blocks
                        for block in 0..n_blocks {
                            let off = block * BLOCK_SIZE;
                            let out_ptr = unsafe { out_row_ptr.add(off * out_cs) };
                            let out_chunk =
                                unsafe { std::slice::from_raw_parts_mut(out_ptr, BLOCK_SIZE) };

                            if beta != $init {
                                let alpha_arr = [alpha; BLOCK_SIZE];
                                <Self as SimdSupported>::binary_simd_op_inplace_lhs(
                                    out_chunk,
                                    &alpha_arr,
                                    BinaryOpType::Mul,
                                );
                            } else {
                                for x in out_chunk.iter_mut() {
                                    *x = $init;
                                }
                            }

                            for p in 0..k {
                                let a_val = unsafe { *lhs_base.add(i * lhs_rs + p * lhs_cs) };
                                let a_arr = [a_val; BLOCK_SIZE];
                                let b_ptr = unsafe { rhs_base.add(p * rhs_rs + off * rhs_cs) };
                                let b_chunk =
                                    unsafe { std::slice::from_raw_parts(b_ptr, BLOCK_SIZE) };
                                <Self as SimdSupported>::fma_op_inplace_c(
                                    &a_arr, b_chunk, out_chunk,
                                );
                            }
                        }

                        // Handle remainder elements
                        if rem > 0 {
                            let off = n_blocks * BLOCK_SIZE;
                            let out_ptr = unsafe { out_row_ptr.add(off * out_cs) };
                            let out_chunk = unsafe { std::slice::from_raw_parts_mut(out_ptr, rem) };

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
                                let a_val = unsafe { *lhs_base.add(i * lhs_rs + p * lhs_cs) };
                                for j in 0..rem {
                                    let b_val =
                                        unsafe { *rhs_base.add(p * rhs_rs + (off + j) * rhs_cs) };
                                    out_chunk[j] += a_val * b_val;
                                }
                            }
                        }
                    }
                }
            }

            instantiate_gemm_cuda!($rt);
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
// Use naive implementation for bf16 to avoid CPU SIMD half-precision assembly requirements
instantiate_gemm!(bf16, bf16::from_f32(0.), NAIVE);
#[cfg(feature = "half")]
// Use naive implementation for f16 to avoid CPU SIMD half-precision assembly requirements
instantiate_gemm!(f16, f16::from_f32(0.), NAIVE);
