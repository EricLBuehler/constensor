use crate::{
    device::{Cpu, Dev},
    storage::Storage,
    DType, Offsetable, Result, Shape, R1, R2, R3,
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

fn from_storage<S: Shape, T: DType, D: Dev>(storage: Arc<Storage<T>>) -> Tensor<S, T, D> {
    Tensor(Arc::new(Tensor_ {
        storage,
        _ghost: PhantomData,
    }))
}

macro_rules! tensor_api {
    ($device:ty) => {
        impl<S: Shape, T: DType> Tensor<S, T, $device> {
            /// Materialize a tensor filled with some value.
            pub fn full(v: T) -> Result<Self> {
                let device = <$device>::resolve()?;
                Ok(from_storage(Arc::new(device.const_impl::<T, S>(v)?)))
            }

            /// Materialize a tensor filled with zeros.
            pub fn zeros() -> Result<Self> {
                Self::full(T::ZERO)
            }

            /// Materialize a tensor filled with ones.
            pub fn ones() -> Result<Self> {
                Self::full(T::ONE)
            }

            /// Create a tensor filled with zeros with the same shape, dtype, and device as `self`.
            pub fn zeros_like(&self) -> Result<Self> {
                Tensor::<S, T, $device>::zeros()
            }

            /// Create a tensor filled with ones with the same shape, dtype, and device as `self`.
            pub fn ones_like(&self) -> Result<Self> {
                Tensor::<S, T, $device>::ones()
            }

            /// Create a tensor filled with some value with the same shape, dtype, and device as `self`.
            pub fn full_like(&self, v: T) -> Result<Self> {
                Tensor::<S, T, $device>::full(v)
            }
        }

        impl<const A: usize, O: Offsetable> Tensor<R1<A>, O, $device> {
            /// Materialize a vector ranging from `start` to `A` with step `step`.
            pub fn arange(start: O, step: O) -> Result<Self> {
                let device = <$device>::resolve()?;
                Ok(from_storage(Arc::new(
                    device.arange_impl::<O, R1<A>>(start, step)?,
                )))
            }
        }

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
