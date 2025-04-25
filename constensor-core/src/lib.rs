#![feature(portable_simd)]

mod cpu_storage;
#[cfg(feature = "cuda")]
mod cuda_backend;
mod device;
mod dtype;
mod error;
mod graph;
mod shape;
mod storage;
mod tensor;

#[cfg(feature = "cuda")]
pub use device::Cuda;
pub use device::{BestDevice, Cpu};
pub use dtype::DType;
pub use error::{Context, Error, Result};
pub use graph::{CompiledGraph, Graph, GraphNode, Op};
pub use shape::{Shape, R1, R2, R3, R4, R5, R6};
pub use tensor::{GraphTensor, Tensor};
