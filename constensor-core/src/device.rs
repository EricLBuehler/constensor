#[cfg(feature = "cuda")]
use crate::cuda_backend::CudaDevice;
use crate::{
    cpu_storage::CpuDevice,
    storage::{BackendDevice, Storage},
    DType, GraphNode, Result,
};

/// Marker trait for devices
pub trait Dev: Clone {
    fn resolve() -> Result<Device>;
}

#[derive(Clone)]
pub struct Cpu;

impl Dev for Cpu {
    fn resolve() -> Result<Device> {
        Ok(Device::Cpu)
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct Cuda<const ORD: usize>;

#[cfg(feature = "cuda")]
macro_rules! cuda_device {
    ($ord:expr) => {
        impl Dev for Cuda<$ord> {
            fn resolve() -> Result<Device> {
                Ok(Device::Cuda(CudaDevice::new($ord)?))
            }
        }
    };
}

// NOTE: Support up to 10 ordinals
#[cfg(feature = "cuda")]
cuda_device!(0);
#[cfg(feature = "cuda")]
cuda_device!(1);
#[cfg(feature = "cuda")]
cuda_device!(2);
#[cfg(feature = "cuda")]
cuda_device!(3);
#[cfg(feature = "cuda")]
cuda_device!(4);
#[cfg(feature = "cuda")]
cuda_device!(5);
#[cfg(feature = "cuda")]
cuda_device!(6);
#[cfg(feature = "cuda")]
cuda_device!(7);
#[cfg(feature = "cuda")]
cuda_device!(8);
#[cfg(feature = "cuda")]
cuda_device!(9);

/// A concrete device.
#[derive(Clone)]
pub enum Device {
    #[cfg(feature = "cuda")]
    Cuda(CudaDevice),
    Cpu,
}

impl Device {
    pub fn compile_and_run_graph<T: DType>(&self, graph: &[GraphNode<T>]) -> Result<Storage<T>> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(cuda) => Ok(Storage::Cuda(cuda.compile_and_run_graph::<S, T>(graph)?)),
            Self::Cpu => Ok(Storage::Cpu(CpuDevice.compile_and_run_graph::<T>(graph)?)),
        }
    }
}
