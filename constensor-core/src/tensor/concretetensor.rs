use crate::{
    device::{Cpu, Dev},
    storage::Storage,
    DType, Result, Shape, R1, R2, R3,
};

#[cfg(feature = "cuda")]
use crate::device::Cuda;

use std::{borrow::Cow, marker::PhantomData, ops::Deref, sync::Arc};

#[derive(Clone)]
pub struct Tensor_<S: Shape, T: DType, D: Dev> {
    storage: Arc<Storage<T>>,
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

pub(crate) fn from_storage<S: Shape, T: DType, D: Dev>(
    storage: Arc<Storage<T>>,
) -> Tensor<S, T, D> {
    Tensor(Arc::new(Tensor_ {
        storage,
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
            /// Get data for a matrix.
            pub fn data(&self) -> Result<Cow<Vec<Vec<T>>>> {
                let data = self.storage.to_cpu_storage()?;
                let mut rows = Vec::new();
                for i in 0..A {
                    let row = (0..B).map(|j| data.as_ref().0[i * A + j]).collect();
                    rows.push(row)
                }
                Ok(Cow::Owned(rows))
            }
        }

        impl<T: DType, const A: usize, const B: usize, const C: usize>
            Tensor<R3<A, B, C>, T, $device>
        {
            /// Get data for a 3 dimensional tensor.
            pub fn data(&self) -> Result<Cow<Vec<Vec<Vec<T>>>>> {
                let data = self.storage.to_cpu_storage()?;
                let mut top_rows = Vec::new();
                for i in 0..A {
                    let mut rows = Vec::new();
                    for j in 0..B {
                        let row = (0..C).map(|k| data.as_ref().0[i * A + j * B + k]).collect();
                        rows.push(row)
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

/*macro_rules! binary_op {
    ($trait:ident, $fn:ident) => {
        impl<S: Shape, D: DType> $trait for Tensor<S, D> {
            type Output = Result<Tensor<S, D>>;
            fn $fn(self, rhs: Self) -> Self::Output {
                Ok(Self::fromTensor_(self.inner.$fn(&rhs.inner)?))
            }
        }
    };
}

binary_op!(Add, add);
binary_op!(Mul, mul);
binary_op!(Sub, sub);
binary_op!(Div, div);
*/
