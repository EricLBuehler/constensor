use crate::{
    device::{Cpu, Dev},
    storage::Storage,
    DType, Result, Shape, R1, R2, R3,
};

#[cfg(feature = "cuda")]
use crate::device::Cuda;

use std::{borrow::Cow, marker::PhantomData, ops::Deref, sync::Arc};

use super::contiguous_strides;

#[derive(Clone)]
pub struct Tensor_<S: Shape, T: DType, D: Dev> {
    storage: Arc<Storage<T>>,
    strides: Vec<usize>,
    _ghost: PhantomData<(S, T, D)>,
}

/// Tensors are n dimensional arrays. Only functions which allocate, copy
/// data, or do operations return `Result`s.
#[derive(Clone)]
pub struct Tensor<S: Shape, T: DType, D: Dev>(Arc<Tensor_<S, T, D>>);

impl<S: Shape, T: DType, D: Dev> Deref for Tensor<S, T, D> {
    type Target = Tensor_<S, T, D>;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

/// Create a Tensor from storage with its default (contiguous) strides.
pub(crate) fn from_storage<S: Shape, T: DType, D: Dev>(
    storage: Arc<Storage<T>>,
) -> Tensor<S, T, D> {
    let shape = S::shape();
    let strides = contiguous_strides(&shape);
    Tensor(Arc::new(Tensor_ {
        storage,
        strides,
        _ghost: PhantomData,
    }))
}

/// Create a Tensor from storage with explicit strides (for views/transposes).
fn from_storage_strided<S: Shape, T: DType, D: Dev>(
    storage: Arc<Storage<T>>,
    strides: Vec<usize>,
) -> Tensor<S, T, D> {
    Tensor(Arc::new(Tensor_ {
        storage,
        strides,
        _ghost: PhantomData,
    }))
}

macro_rules! tensor_api {
    ($device:ty) => {
        impl<T: DType, const A: usize> Tensor<R1<A>, T, $device> {
            /// Get data for a vector.
            pub fn data(&self) -> Result<Cow<Vec<T>>> {
                let data = self.storage.to_cpu_storage()?;
                Ok(Cow::Owned(data.into_owned().0))
            }
        }

        impl<T: DType, const A: usize, const B: usize> Tensor<R2<A, B>, T, $device> {
            /// Get data for a matrix, respecting strides (supports views/transposes).
            pub fn data(&self) -> Result<Cow<Vec<Vec<T>>>> {
                let data = self.storage.to_cpu_storage()?;
                let mut rows = Vec::with_capacity(A);
                for i in 0..A {
                    let base = i * self.strides[0];
                    let row = (0..B)
                        .map(|j| data.as_ref().0[base + j * self.strides[1]])
                        .collect();
                    rows.push(row);
                }
                Ok(Cow::Owned(rows))
            }
        }

        impl<T: DType, const A: usize, const B: usize, const C: usize>
            Tensor<R3<A, B, C>, T, $device>
        {
            /// Get data for a 3 dimensional tensor, respecting strides (supports views/transposes).
            pub fn data(&self) -> Result<Cow<Vec<Vec<Vec<T>>>>> {
                let data = self.storage.to_cpu_storage()?;
                let mut top_rows = Vec::with_capacity(A);
                for i in 0..A {
                    let off_i = i * self.strides[0];
                    let mut rows = Vec::with_capacity(B);
                    for j in 0..B {
                        let off_j = off_i + j * self.strides[1];
                        let row = (0..C)
                            .map(|k| data.as_ref().0[off_j + k * self.strides[2]])
                            .collect();
                        rows.push(row);
                    }
                    top_rows.push(rows);
                }
                Ok(Cow::Owned(top_rows))
            }
        }
    };
}

tensor_api!(Cpu);
#[cfg(feature = "cuda")]
tensor_api!(Cuda<0>);

impl<S: Shape, T: DType, D: Dev> Tensor<S, T, D> {
    /// Cast this tensor to a different dtype `U` on the CPU.
    pub fn cast<U: DType>(&self) -> Result<Tensor<S, U, D>> {
        // retrieve data from storage as owned Vec<T>
        let storage = self.storage.cast::<U>()?;
        Ok(from_storage::<S, U, D>(Arc::new(storage)))
    }
}

impl<T: DType, const A: usize, const B: usize, D: Dev> Tensor<R2<A, B>, T, D> {
    /// Return a view of this matrix with dimensions transposed (A x B -> B x A).
    pub fn t(&self) -> Tensor<R2<B, A>, T, D> {
        // swap strides for first two dimensions
        let mut new_strides = self.strides.clone();
        new_strides.swap(0, 1);
        from_storage_strided::<R2<B, A>, T, D>(Arc::clone(&self.storage), new_strides)
    }
}

impl<T: DType, const A: usize, const B: usize, const C: usize, D: Dev> Tensor<R3<A, B, C>, T, D> {
    /// Return a view of this tensor with last two reversed axes (A x B x C -> A x C x B).
    pub fn t(&self) -> Tensor<R3<C, B, A>, T, D> {
        // swap strides for last two dimensions
        let mut new_strides = self.strides.clone();
        new_strides.swap(1, 2);
        from_storage_strided::<R3<C, B, A>, T, D>(Arc::clone(&self.storage), new_strides)
    }
}
