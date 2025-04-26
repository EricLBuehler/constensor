//! Instantiate cubecl kernels for binary operations on all supported dtypes.

use crate::graph::BinaryOpType;
use cubecl::{cube, prelude::*};
#[cfg(feature = "bfloat")]
use half::bf16;
#[cfg(feature = "half")]
use half::f16;

macro_rules! define_binary_kernel {
    ($name:ident, $ty:ty) => {
        #[cube(launch_unchecked)]
        fn $name(
            a: &Sequence<Array<$ty>>,
            b: &Sequence<Array<$ty>>,
            out: &mut Sequence<Array<$ty>>,
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
    };
}

// Integer types
define_binary_kernel!(binary_u8, u8);
define_binary_kernel!(binary_u32, u32);
define_binary_kernel!(binary_i32, i32);
define_binary_kernel!(binary_i64, i64);

// Floating-point types
define_binary_kernel!(binary_f32, f32);
define_binary_kernel!(binary_f64, f64);
#[cfg(feature = "half")]
define_binary_kernel!(binary_f16, f16);
#[cfg(feature = "bfloat")]
define_binary_kernel!(binary_bf16, bf16);
