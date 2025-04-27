//! Instantiate cubecl kernels for binary operations on all supported dtypes.

use crate::{dtype::DTypeOps, graph::BinaryOpType};
use cubecl::{cube, prelude::*};

#[cube(launch_unchecked)]
pub(super) fn binary<T: CubeType + CubePrimitive + Send + Sync + DTypeOps>(
    a: &Sequence<Array<T>>,
    b: &Sequence<Array<T>>,
    out: &mut Sequence<Array<T>>,
    #[comptime] numel: u32,
    #[comptime] ops: Sequence<BinaryOpType>,
) {
    if ABSOLUTE_POS < numel {
        #[unroll]
        for index in 0..ops.len() {
            let op = comptime! { ops.index(index.clone()) };
            let av = a.index(index);
            let bv = b.index(index);
            let ov = out.index_mut(index);
            match op {
                BinaryOpType::Add => ov[ABSOLUTE_POS] = av[ABSOLUTE_POS] + bv[ABSOLUTE_POS],
                BinaryOpType::Sub => ov[ABSOLUTE_POS] = av[ABSOLUTE_POS] - bv[ABSOLUTE_POS],
                BinaryOpType::Mul => ov[ABSOLUTE_POS] = av[ABSOLUTE_POS] * bv[ABSOLUTE_POS],
                BinaryOpType::Div => ov[ABSOLUTE_POS] = av[ABSOLUTE_POS] / bv[ABSOLUTE_POS],
            }
        }
    }
}
