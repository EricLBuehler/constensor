use std::{borrow::Cow, ops::Deref, sync::Arc};
mod error;
use cudarc::driver::{CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};
use error::{CudaError, WrapErr};

use crate::{
    cpu_storage::CpuStorage,
    storage::{BackendDevice, BackendStorage},
    DType, Offsetable, Result,
};

#[derive(Clone)]
pub struct CudaDevice {
    device: Arc<cudarc::driver::CudaDevice>,
}

impl CudaDevice {
    pub(crate) fn new(ordinal: usize) -> Result<Self> {
        Ok(Self {
            device: cudarc::driver::CudaDevice::new(ordinal).w()?,
        })
    }

    pub(crate) fn get_or_load_func<T: DType>(
        &self,
        module_name: &str,
        ptx: &'static str,
    ) -> Result<CudaFunction> {
        let module_name = format!("{module_name}_{}", T::NAME);
        let module_name = &module_name;
        if !self.has_func(module_name, module_name) {
            // Leaking the string here is a bit sad but we need a &'static str and this is only
            // done once per kernel name.
            let static_module_name = Box::leak(module_name.to_string().into_boxed_str());
            self.load_ptx(ptx.into(), module_name, &[static_module_name])
                .map_err(|cuda| CudaError::Load {
                    cuda,
                    module_name: module_name.to_string(),
                })
                .w()?;
        }
        self.get_func(module_name, module_name)
            // Clippy recommends this `ok_or` rather than `ok_or_else` so hopefully the compiler is
            // able to only build the error value if needed.
            .ok_or(CudaError::MissingKernel {
                module_name: module_name.to_string(),
            })
            .w()
    }
}

impl Deref for CudaDevice {
    type Target = Arc<cudarc::driver::CudaDevice>;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

pub struct CudaStorage<T: DType> {
    slice: CudaSlice<T>,
    device: CudaDevice,
}

impl<T: DType> BackendStorage<T> for CudaStorage<T> {
    fn to_cpu_storage(&self) -> Result<Cow<CpuStorage<T>>> {
        let data = self.device.dtoh_sync_copy(&self.slice).w()?;
        Ok(Cow::Owned(CpuStorage(data)))
    }
}

impl BackendDevice for CudaDevice {
    type Storage<X: DType> = CudaStorage<X>;

    fn const_impl<S: crate::Shape, T: DType>(&self, v: T) -> Result<Self::Storage<T>> {
        let n_elems = S::element_count();
        let data = unsafe { self.device.alloc::<T>(n_elems) }.w()?;
        let func = self.get_or_load_func::<T>("fill", constensor_cuda_kernels::FILL)?;
        let params = (&data, v, n_elems);
        let cfg = LaunchConfig::for_num_elems(n_elems as u32);
        unsafe { func.launch(cfg, params) }.w()?;
        Ok(CudaStorage {
            slice: data,
            device: self.clone(),
        })
    }

    fn arange_impl<S: crate::Shape, O: Offsetable>(
        &self,
        start: O,
        step: O,
    ) -> Result<Self::Storage<O>> {
        let n_elems = S::element_count();
        let data = unsafe { self.device.alloc::<O>(n_elems) }.w()?;
        let func = self.get_or_load_func::<O>("arange", constensor_cuda_kernels::ARANGE)?;
        let params = (&data, start, step, n_elems);
        let cfg = LaunchConfig::for_num_elems(n_elems as u32);
        unsafe { func.launch(cfg, params) }.w()?;
        Ok(CudaStorage {
            slice: data,
            device: self.clone(),
        })
    }
}
