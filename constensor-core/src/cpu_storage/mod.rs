use std::borrow::Cow;

use crate::{
    storage::{BackendDevice, BackendStorage, Offsetable},
    DType, Result,
};

pub struct CpuDevice;

#[derive(Clone)]
pub struct CpuStorage<T: DType>(pub(crate) Vec<T>);

impl<T: DType> BackendStorage<T> for CpuStorage<T> {
    fn to_cpu_storage(&self) -> Result<std::borrow::Cow<CpuStorage<T>>> {
        // Note: copying all data here.
        Ok(Cow::Owned(self.clone()))
    }
}

impl BackendDevice for CpuDevice {
    type Storage<X: DType> = CpuStorage<X>;

    fn const_impl<S: crate::Shape, T: DType>(&self, v: T) -> Result<Self::Storage<T>> {
        Ok(CpuStorage(vec![v; S::element_count()]))
    }

    fn arange_impl<S: crate::Shape, O: Offsetable>(
        &self,
        start: O,
        step: O,
    ) -> Result<Self::Storage<O>> {
        let mut accum = Vec::with_capacity(S::element_count());
        for i in 0..S::element_count() {
            accum.push(O::offset(i, start, step));
        }
        Ok(CpuStorage(accum))
    }
}
