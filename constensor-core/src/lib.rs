#![feature(portable_simd)]

mod cpu_storage;
#[cfg(feature = "cuda")]
mod cuda_backend;
mod device;
mod dtype;
mod error;
mod graph;
mod graphtensor;
mod shape;
mod storage;
mod tensor;

pub use device::Cpu;
#[cfg(feature = "cuda")]
pub use device::Cuda;
pub use dtype::{DType, SignedDType};
pub use error::{Error, Result};
pub use graph::{Graph, Op};
pub use graphtensor::GraphTensor;
pub use shape::{Shape, R1, R2, R3, R4, R5, R6};
pub use tensor::Tensor;
