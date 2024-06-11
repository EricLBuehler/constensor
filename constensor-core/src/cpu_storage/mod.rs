use std::borrow::Cow;

use crate::{
    storage::{BackendDevice, BackendStorage},
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

    fn compile_and_run_graph<S: crate::Shape, T: DType>(
        &self,
        graph: &[crate::Op<T>],
    ) -> Result<Self::Storage<T>> {
        dbg!(&graph);
        todo!("No CPU implementation for `compile_and_run_graph`")
    }
}
