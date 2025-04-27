//! Instantiate cubecl kernels for binary operations on all supported dtypes.

use crate::{dtype::DTypeOps, graph::BinaryOpType};
use cubecl::{cube, prelude::*};

#[cube(launch_unchecked)]
pub(super) fn binary<T: CubeType + CubePrimitive + Send + Sync + DTypeOps>(
    a: &Array<T>,
    b: &Sequence<Array<T>>,
    out: &mut Array<T>,
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
