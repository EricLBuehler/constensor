// clang-format -i  constensor-cuda-kernels/src/*.cu

pub const FILL: &str = include_str!(concat!(env!("OUT_DIR"), "/fill.ptx"));
pub const ARANGE: &str = include_str!(concat!(env!("OUT_DIR"), "/arange.ptx"));
