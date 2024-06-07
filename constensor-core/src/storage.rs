use std::borrow::Cow;

#[cfg(feature = "cuda")]
use crate::cuda_backend::CudaStorage;
use crate::{cpu_storage::CpuStorage, DType, Result, Shape};

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

pub trait Offsetable: DType {
    fn offset(i: usize, start: Self, step: Self) -> Self;
}

macro_rules! offsetable {
    ($ty:ty) => {
        impl Offsetable for $ty {
            fn offset(i: usize, start: Self, step: Self) -> Self {
                (i as $ty) * step + start
            }
        }
    };
}

offsetable!(u8);
offsetable!(u32);
offsetable!(i64);
offsetable!(f32);
offsetable!(f64);

#[cfg(feature = "bfloat")]
use half::bf16;
#[cfg(feature = "half")]
use half::f16;
#[cfg(feature = "half")]
impl Offsetable for f16 {
    fn offset(i: usize, start: Self, step: Self) -> Self {
        f16::from_f64_const(i as f64) * step + start
    }
}
#[cfg(feature = "bfloat")]
impl Offsetable for bf16 {
    fn offset(i: usize, start: Self, step: Self) -> Self {
        bf16::from_f64_const(i as f64) * step + start
    }
}

pub trait BackendDevice {
    type Storage<X: DType>: BackendStorage<X>;

    fn const_impl<S: Shape, T: DType>(&self, v: T) -> Result<Self::Storage<T>>;
    fn arange_impl<S: Shape, O: Offsetable>(&self, start: O, step: O) -> Result<Self::Storage<O>>;
}
