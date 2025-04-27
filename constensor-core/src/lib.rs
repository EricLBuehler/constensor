#![feature(portable_simd)]

//! Constensor is an experimental ML framework featuring a graph-based JIT compiler.
//!
//! It's designed with clarity and efficiency in mind, so besides compile-time checks for shape, dtype, and device,
//! it also is based around a graph design.
//!
//! This means that all the code you write is entered into a graph and then analyzed when you call [`Graph::optimize`]!
//! The optimization step fuses operations, allows for automatic and seamless inplacing, constant folding, and other features.
//!
//! Then, by precompliling the graph with [`Graph::compile`], we can make intelligent decisions about when we run certain operations.
//! For instance, on CUDA, streams are automatically used where possible to parallelize execution.
//!
//! Currently, only CUDA and CPU are supported but Metal and [cubecl](https://github.com/tracel-ai/cubecl) is support coming very soon.
//!
//! ## A quick guide
//! - First, create a [`Graph`]. This will hold all the operations and do optimization and compilation.
//! - Tensors are modelled with a [`GraphTensor`]. These represent the operation but do not perform any computation.
//! - Be sure to optimize the graph using [`Graph::optimize`] after all operations are complete!
//! - Compile the graph using [`Graph::compile`], which will insert the device-specific optimizations. This returns a [`CompiledGraph`].
//! - Run using [`CompiledGraph::run`]. This returns a concrete [`Tensor`].
//!
//! ## What can you do with it?
//! ```
//! use constensor_core::{Cpu, Graph, GraphTensor, Tensor, R1, R2};
//!
//! let mut graph: Graph<f32> = Graph::empty();
//! let _arange = GraphTensor::<R1<10>, f32, Cpu>::arange(&mut graph, 0., 1.);
//! let a = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(&mut graph, 1.0);
//! let b = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(&mut graph, 2.0);
//! let c = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(&mut graph, 3.0);
//! let d = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(&mut graph, 4.0);
//! let res = a * b + c;
//! let _out = res + d;
//!
//! graph.optimize();
//!
//! let compiled: constensor_core::CompiledGraph<R2<3, 4>, f32, Cpu> = graph.compile().unwrap();
//! let res = compiled.run().unwrap();
//!
//! let tensor: Tensor<R2<3, 4>, f32, Cpu> = res;
//!
//! assert_eq!(tensor.data().unwrap().to_vec(), vec![vec![9.0; 4]; 3],);
//! ```

mod backends;
use backends::*;
mod device;
mod dtype;
mod error;
mod graph;
mod shape;
mod storage;
mod tensor;

#[cfg(feature = "cuda")]
pub use device::Cuda;
pub use device::{BestDevice, Cpu, Wgpu};
pub use dtype::DType;
pub use error::{Context, Error, Result};
pub use graph::{CompiledGraph, Graph, GraphNode, Op};
pub use shape::{Shape, R1, R2, R3, R4, R5, R6};
pub use tensor::{GraphTensor, Tensor};
