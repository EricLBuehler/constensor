use cudarc::cublas::{GemmConfig, StridedBatchedConfig};

use crate::{Error, Result};

pub(crate) fn gemm_config<T>(
    alpha: T,
    beta: T,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    rhs_stride: &[usize],
    out_stride: &[usize],
) -> Result<StridedBatchedConfig<T>> {
    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm
    use cudarc::cublas::sys::cublasOperation_t;

    let lhs_dims = [b, m, k];
    let rhs_dims = [b, k, n];

    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    // The a tensor has dims batching, k, n (rhs)
    // We also allow for the case where the stride on the minor dimension is not as expected but
    // there is a single element.
    let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        Err(Error::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            out_stride: out_stride.to_vec(),
            mnk: (m, n, k),
        })?
    };
    // The b tensor has dims batching, m, k (lhs)
    // We also allow for the case where the stride on the minor dimension is not as expected but
    // there is a single element.
    let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        Err(Error::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            out_stride: out_stride.to_vec(),
            mnk: (m, n, k),
        })?
    };
    // The setup below was copied from:
    // https://github.com/lebedov/scikit-cuda/blob/7e7300474286019c917a6c8a4bca59405c64fbce/tests/test_cublas.py#L531
    let gemm = GemmConfig {
        alpha,
        beta,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda,
        ldb,
        ldc: n as i32,
        transa,
        transb,
    };

    let stride_a: usize = match lhs_stride[..lhs_stride.len() - 2] {
        [s1, stride] if s1 == stride * lhs_dims[1] => stride,
        [_, stride] if lhs_dims[0] == 1 => stride,
        [stride, _] if lhs_dims[1] == 1 => stride,
        [stride] => stride,
        [] => m * k,
        _ => Err(Error::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            out_stride: out_stride.to_vec(),
            mnk: (m, n, k),
        })?,
    };
    let stride_b: usize = match rhs_stride[..rhs_stride.len() - 2] {
        [s1, stride] if s1 == stride * rhs_dims[1] => stride,
        [_, stride] if rhs_dims[0] == 1 => stride,
        [stride, _] if rhs_dims[1] == 1 => stride,
        [stride] => stride,
        [] => n * k,
        _ => Err(Error::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            out_stride: out_stride.to_vec(),
            mnk: (m, n, k),
        })?,
    };
    let stride_c: usize = match out_stride[..out_stride.len() - 2] {
        [s1, stride] if s1 == stride * rhs_dims[1] => stride,
        [_, stride] if rhs_dims[0] == 1 => stride,
        [stride, _] if rhs_dims[1] == 1 => stride,
        [stride] => stride,
        [] => m * n,
        _ => Err(Error::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            out_stride: out_stride.to_vec(),
            mnk: (m, n, k),
        })?,
    };

    Ok(StridedBatchedConfig {
        batch_size: b as i32,
        gemm,
        stride_a: stride_a as i64,
        stride_b: stride_b as i64,
        stride_c: stride_c as i64,
    })
}
