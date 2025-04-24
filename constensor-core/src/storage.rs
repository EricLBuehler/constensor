use std::borrow::Cow;

#[cfg(feature = "cuda")]
use crate::cuda_backend::CudaStorage;
use crate::{cpu_storage::CpuStorage, device::Dev, CompiledGraph, DType, GraphNode, Result, Shape};

pub enum Storage<T: DType> {
    #[cfg(feature = "cuda")]
    Cuda(CudaStorage<T>),
    Cpu(CpuStorage<T>),
}

impl<T: DType> Storage<T> {
    pub(crate) fn to_cpu_storage(&self) -> Result<Cow<CpuStorage<T>>> {
        match self {
            Self::Cpu(cpu) => cpu.to_cpu_storage(),
            #[cfg(feature = "cuda")]
            Self::Cuda(cuda) => cuda.to_cpu_storage(),
        }
    }
}

pub trait BackendStorage<T: DType> {
    fn to_cpu_storage(&self) -> Result<Cow<CpuStorage<T>>>;
}

pub trait BackendDevice {
    type Storage<X: DType>: BackendStorage<X>;

    fn compile<S: Shape, T: DType, D: Dev>(
        &self,
        graph: Vec<GraphNode<T>>,
    ) -> Result<CompiledGraph<S, T, D>>;
    fn run_graph<S: Shape, T: DType, D: Dev>(
        &self,
        graph: &CompiledGraph<S, T, D>,
    ) -> Result<Self::Storage<T>>;
}
