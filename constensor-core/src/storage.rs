use std::borrow::Cow;

#[cfg(feature = "cuda")]
use crate::cuda_backend::CudaStorage;
use crate::{cpu_storage::CpuStorage, DType, Op, Result, Shape, SignedDType};

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

    fn compile_and_run_graph_signed<S: Shape, T: DType + SignedDType>(
        &self,
        graph: &[Op<T>],
    ) -> Result<Self::Storage<T>>;

    fn compile_and_run_graph<S: Shape, T: DType>(
        &self,
        graph: &[Op<T>],
    ) -> Result<Self::Storage<T>>;
}
