//! Instantiate cubecl kernels for operations on all supported dtypes.

use crate::{
    dtype::DTypeOps,
    graph::{BinaryOpType, UnaryOpType},
};
use cubecl::{channel::MutexComputeChannel, cube, prelude::*, wgpu::WgpuServer};

use super::RT;

pub trait UnaryKernelLaunch: CubeType + CubePrimitive + Send + Sync + Cast {
    fn launch(
        client: &ComputeClient<WgpuServer, MutexComputeChannel<WgpuServer>>,
        count: CubeCount,
        dim: CubeDim,
        a: ArrayArg<'_, RT>,
        out: ArrayArg<'_, RT>,
        numel: u32,
        ops: Sequence<UnaryOpType>,
    );
}

// Integer impl → forwards to unary_int
macro_rules! impl_launch_int {
    ($($t:ty),* $(,)?) => {
        $(
            impl UnaryKernelLaunch for $t {
                #[inline(always)]
                fn launch(
                    client: &ComputeClient<WgpuServer, MutexComputeChannel<WgpuServer>>,
                    count: CubeCount,
                    dim: CubeDim,
                    a: ArrayArg<'_, RT>,
                    out: ArrayArg<'_, RT>,
                    numel: u32,
                    ops: Sequence<UnaryOpType>,
                ) {
                    unsafe { unary_int::launch_unchecked::<Self, RT>(client, count, dim, a, out, numel, ops) };
                }
            }
        )*
    };
}

impl_launch_int!(i32, i64, u8, u32);

impl UnaryKernelLaunch for f32 {
    #[inline(always)]
    fn launch(
        client: &ComputeClient<WgpuServer, MutexComputeChannel<WgpuServer>>,
        count: CubeCount,
        dim: CubeDim,
        a: ArrayArg<'_, RT>,
        out: ArrayArg<'_, RT>,
        numel: u32,
        ops: Sequence<UnaryOpType>,
    ) {
        unsafe {
            unary_float::launch_unchecked::<Self, RT>(client, count, dim, a, out, numel, ops)
        };
    }
}

impl UnaryKernelLaunch for f64 {
    #[inline(always)]
    fn launch(
        client: &ComputeClient<WgpuServer, MutexComputeChannel<WgpuServer>>,
        count: CubeCount,
        dim: CubeDim,
        a: ArrayArg<'_, RT>,
        out: ArrayArg<'_, RT>,
        numel: u32,
        ops: Sequence<UnaryOpType>,
    ) {
        unsafe {
            unary_float::launch_unchecked::<Self, RT>(client, count, dim, a, out, numel, ops)
        };
    }
}

#[cfg(feature = "half")]
impl UnaryKernelLaunch for half::f16 {
    #[inline(always)]
    fn launch(
        client: &ComputeClient<WgpuServer, MutexComputeChannel<WgpuServer>>,
        count: CubeCount,
        dim: CubeDim,
        a: ArrayArg<'_, RT>,
        out: ArrayArg<'_, RT>,
        numel: u32,
        ops: Sequence<UnaryOpType>,
    ) {
        unsafe {
            unary_float::launch_unchecked::<Self, RT>(client, count, dim, a, out, numel, ops)
        };
    }
}

#[cfg(feature = "bfloat")]
impl UnaryKernelLaunch for half::bf16 {
    #[inline(always)]
    fn launch(
        client: &ComputeClient<WgpuServer, MutexComputeChannel<WgpuServer>>,
        count: CubeCount,
        dim: CubeDim,
        a: ArrayArg<'_, RT>,
        out: ArrayArg<'_, RT>,
        numel: u32,
        ops: Sequence<UnaryOpType>,
    ) {
        unsafe {
            unary_float::launch_unchecked::<Self, RT>(client, count, dim, a, out, numel, ops)
        };
    }
}

/// Convenience wrapper that launches the *right* kernel for `T`.
///
/// `numel` and `ops` are compile‑time parameters exactly like the inner
/// kernels, so you call this with the same `cube!` launch macro you
/// already use.
///
/// ```ignore
/// cube!(
///     ctx,
///     unary_auto::<u32>( &a, &mut out, comptime numel, comptime ops_seq )
/// );
/// ```
pub(super) fn unary_auto<T: UnaryKernelLaunch>(
    client: &ComputeClient<WgpuServer, MutexComputeChannel<WgpuServer>>,
    count: CubeCount,
    dim: CubeDim,
    a: ArrayArg<'_, RT>,
    out: ArrayArg<'_, RT>,
    numel: u32,
    ops: Sequence<UnaryOpType>,
) {
    T::launch(client, count, dim, a, out, numel, ops);
}

#[cube(launch_unchecked)]
pub(super) fn binary<T: CubeType + CubePrimitive + Send + Sync + DTypeOps>(
    a: &Array<Line<T>>,
    b: &Sequence<Array<Line<T>>>,
    out: &mut Array<Line<T>>,
    #[comptime] numel: u32,
    #[comptime] ops: Sequence<BinaryOpType>,
) {
    if ABSOLUTE_POS < numel {
        let op = comptime! { ops.index(0) };
        let bv = b.index(0);
        match op {
            BinaryOpType::Add => out[ABSOLUTE_POS] = a[ABSOLUTE_POS] + bv[ABSOLUTE_POS],
            BinaryOpType::Sub => out[ABSOLUTE_POS] = a[ABSOLUTE_POS] - bv[ABSOLUTE_POS],
            BinaryOpType::Mul => out[ABSOLUTE_POS] = a[ABSOLUTE_POS] * bv[ABSOLUTE_POS],
            BinaryOpType::Div => out[ABSOLUTE_POS] = a[ABSOLUTE_POS] / bv[ABSOLUTE_POS],
        }

        #[unroll]
        for index in 1..ops.len() {
            let op = comptime! { ops.index(index.clone()) };
            let bv = b.index(index);
            match op {
                BinaryOpType::Add => out[ABSOLUTE_POS] = out[ABSOLUTE_POS] + bv[ABSOLUTE_POS],
                BinaryOpType::Sub => out[ABSOLUTE_POS] = out[ABSOLUTE_POS] - bv[ABSOLUTE_POS],
                BinaryOpType::Mul => out[ABSOLUTE_POS] = out[ABSOLUTE_POS] * bv[ABSOLUTE_POS],
                BinaryOpType::Div => out[ABSOLUTE_POS] = out[ABSOLUTE_POS] / bv[ABSOLUTE_POS],
            }
        }
    }
}

#[cube(launch_unchecked)]
pub(super) fn unary_float<F: Float>(
    a: &Array<F>,
    out: &mut Array<F>,
    #[comptime] numel: u32,
    #[comptime] ops: Sequence<UnaryOpType>,
) {
    if ABSOLUTE_POS < numel {
        let op = comptime! { ops.index(0) };
        match op {
            UnaryOpType::Neg => out[ABSOLUTE_POS] = -a[ABSOLUTE_POS],
            UnaryOpType::Sqrt => out[ABSOLUTE_POS] = F::sqrt(a[ABSOLUTE_POS]),
            UnaryOpType::Exp => out[ABSOLUTE_POS] = F::exp(a[ABSOLUTE_POS]),
            UnaryOpType::Exp2 => todo!(),
        }

        #[unroll]
        for index in 1..ops.len() {
            let op = comptime! { ops.index(index.clone()) };
            match op {
                UnaryOpType::Neg => out[ABSOLUTE_POS] = -out[ABSOLUTE_POS],
                UnaryOpType::Sqrt => out[ABSOLUTE_POS] = F::sqrt(out[ABSOLUTE_POS]),
                UnaryOpType::Exp => out[ABSOLUTE_POS] = F::exp(out[ABSOLUTE_POS]),
                UnaryOpType::Exp2 => todo!(),
            }
        }
    }
}

#[cube(launch_unchecked)]
pub(super) fn unary_int<I: CubeType + CubePrimitive + Send + Sync + DTypeOps + Cast>(
    a: &Array<I>,
    out: &mut Array<I>,
    #[comptime] numel: u32,
    #[comptime] ops: Sequence<UnaryOpType>,
) {
    if ABSOLUTE_POS < numel {
        // ---- first op -----------------------------------------------------
        let op = comptime! { ops.index(0) };
        let mut tmp: f32 = f32::cast_from(a[ABSOLUTE_POS]);

        match op {
            UnaryOpType::Neg => tmp = -tmp,
            UnaryOpType::Sqrt => tmp = f32::sqrt(tmp),
            UnaryOpType::Exp => tmp = f32::exp(tmp),
            UnaryOpType::Exp2 => todo!(),
        }

        out[ABSOLUTE_POS] = I::cast_from(tmp);

        // ---- remaining ops (if any) --------------------------------------
        #[unroll]
        for index in 1..ops.len() {
            let op = comptime! { ops.index(index.clone()) };
            let mut tmp: f32 = f32::cast_from(out[ABSOLUTE_POS]);

            match op {
                UnaryOpType::Neg => tmp = -tmp,
                UnaryOpType::Sqrt => tmp = f32::sqrt(tmp),
                UnaryOpType::Exp => tmp = f32::exp(tmp),
                UnaryOpType::Exp2 => todo!(),
            }

            out[ABSOLUTE_POS] = I::cast_from(tmp);
        }
    }
}

#[cube(launch_unchecked)]
pub(super) fn fill<
    T: CubeType + CubePrimitive + Send + Sync + LaunchArgExpand + Numeric + DTypeOps,
>(
    out: &mut Array<T>,
    value: T,
    #[comptime] numel: u32,
) {
    if ABSOLUTE_POS < numel {
        out[ABSOLUTE_POS] = value;
    }
}
