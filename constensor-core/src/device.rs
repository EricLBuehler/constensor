#[cfg(feature = "cuda")]
use crate::cuda_backend::CudaDevice;
use crate::{
    backends::wgpu_backend::WgpuDevice,
    cpu_backend::CpuDevice,
    storage::{BackendDevice, Storage},
    CompiledGraph, DType, GraphNode, Result, Shape,
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
#[derive(Clone)]
pub struct Wgpu;

impl Dev for Wgpu {
    fn resolve() -> Result<Device> {
        Ok(Device::Wgpu)
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

#[cfg(feature = "cuda")]
pub type BestDevice<const ORD: usize> = Cuda<ORD>;
#[cfg(not(feature = "cuda"))]
pub type BestDevice<const ORD: usize> = Cpu;

/// A concrete device.
#[derive(Clone)]
pub enum Device {
    #[cfg(feature = "cuda")]
    Cuda(CudaDevice),
    Cpu,
    Wgpu,
}

impl Device {
    pub fn run_graph<S: Shape, T: DType, D: Dev>(
        &self,
        graph: &CompiledGraph<S, T, D>,
    ) -> Result<Storage<T>> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(cuda) => Ok(Storage::Cuda(cuda.run_graph::<S, T, D>(graph)?)),
            Self::Cpu => Ok(Storage::Cpu(CpuDevice.run_graph::<S, T, D>(graph)?)),
            Self::Wgpu => Ok(Storage::Wgpu(WgpuDevice.run_graph::<S, T, D>(graph)?)),
        }
    }

    pub fn compile<S: Shape, T: DType, D: Dev>(
        &self,
        graph: Vec<GraphNode<T>>,
    ) -> Result<CompiledGraph<S, T, D>> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(cuda) => cuda.compile::<S, T, D>(graph),
            Self::Cpu => CpuDevice.compile::<S, T, D>(graph),
            Self::Wgpu => WgpuDevice.compile::<S, T, D>(graph),
        }
    }
}
