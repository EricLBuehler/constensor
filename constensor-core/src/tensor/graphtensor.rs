use std::{
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
    sync::{Arc, RwLock, RwLockReadGuard},
};

use crate::{
    device::Dev,
    graph::{BinaryOpType, Graph, GraphTensorId, Op, UnaryOpType},
    DType, Shape, R1, R2, R3,
};

use super::contiguous_strides;

/// A tensor representing an intermediary result of a graph. Performing operations
/// on this tensor will not cause any computations.
#[derive(Clone)]
pub struct GraphTensor<S: Shape, T: DType, D: Dev> {
    id: GraphTensorId,
    graph: Arc<RwLock<Graph<T>>>,
    strides: Vec<usize>,
    _ghost: PhantomData<(S, T, D)>,
}

impl<const B: usize, const M: usize, const K: usize, T: DType, D: Dev>
    GraphTensor<R3<B, M, K>, T, D>
{
    #[must_use]
    // Matrix multiplication: (B x M x K) * (B x K x N) = (B x M x N)
    pub fn matmul<const N: usize>(
        self,
        rhs: GraphTensor<R3<B, K, N>, T, D>,
    ) -> GraphTensor<R3<B, M, N>, T, D> {
        let id = self.graph.write().unwrap().next_id();
        self.graph.write().unwrap().add_op::<R3<B, M, N>>(
            Op::MatMul {
                l_id: self.id(),
                r_id: rhs.id(),
                o_id: None,
                k: K,
                alpha: T::ZERO,
                beta: T::ONE,
            },
            &self.strides,
            &id,
        );
        GraphTensor {
            id,
            graph: self.graph.clone(),
            strides: self.strides.clone(),
            _ghost: PhantomData,
        }
    }

    #[must_use]
    // Matrix multiplication: (B x M x K) * (B x K x N) = (B x M x N)
    /// out = out * alpha + beta * lhs * rhs
    pub fn matmul_axpby<const N: usize>(
        self,
        rhs: GraphTensor<R3<B, K, N>, T, D>,
        out: GraphTensor<R3<B, M, N>, T, D>,
        alpha: T,
        beta: T,
    ) -> GraphTensor<R3<B, M, N>, T, D> {
        let id = self.graph.write().unwrap().next_id();
        self.graph.write().unwrap().add_op::<R3<B, M, N>>(
            Op::MatMul {
                l_id: self.id(),
                r_id: rhs.id(),
                o_id: Some(out.id()),
                k: K,
                alpha,
                beta,
            },
            &self.strides,
            &id,
        );
        GraphTensor {
            id,
            graph: self.graph.clone(),
            strides: self.strides.clone(),
            _ghost: PhantomData,
        }
    }
}

impl<S: Shape, T: DType, D: Dev> GraphTensor<S, T, D> {
    #[must_use]
    /// Create a tensor filled with some value.
    pub fn fill(graph: &mut Graph<T>, v: T) -> Self {
        let id = graph.next_id();
        let strides = contiguous_strides(&S::shape());
        graph.add_op::<S>(Op::Fill { v }, &strides, &id);
        Self {
            id,
            graph: Arc::new(RwLock::new(graph.clone())),
            strides,
            _ghost: PhantomData,
        }
    }

    #[must_use]
    /// Create a tensor filled with zeros.
    pub fn zeros(graph: &mut Graph<T>) -> Self {
        Self::fill(graph, T::ZERO)
    }

    #[must_use]
    /// Create a tensor filled with ones.
    pub fn ones(graph: &mut Graph<T>) -> Self {
        Self::fill(graph, T::ONE)
    }

    #[must_use]
    /// Elementwise unary square root.
    pub fn sqrt(self) -> GraphTensor<S, T, D> {
        let id = self.graph.write().unwrap().next_id();
        self.graph.write().unwrap().add_op::<S>(
            Op::UnaryOp {
                v_id: self.id(),
                operator: UnaryOpType::Sqrt,
            },
            &self.strides,
            &id,
        );
        Self {
            id,
            graph: self.graph.clone(),
            strides: self.strides.clone(),
            _ghost: PhantomData,
        }
    }

    #[must_use]
    /// Create a tensor filled with uniform random values in [0,1).
    pub fn rand(graph: &mut Graph<T>) -> Self {
        let id = graph.next_id();
        let strides = contiguous_strides(&S::shape());
        graph.add_op::<S>(Op::Rand, &strides, &id);
        GraphTensor {
            id,
            graph: Arc::new(RwLock::new(graph.clone())),
            strides,
            _ghost: PhantomData,
        }
    }

    #[must_use]
    /// Create a tensor filled with normally distributed random values (mean, std).
    pub fn randn(graph: &mut Graph<T>, mean: T, std: T) -> Self {
        let id = graph.next_id();
        let strides = contiguous_strides(&S::shape());
        graph.add_op::<S>(Op::Randn { mean, std }, &strides, &id);
        GraphTensor {
            id,
            graph: Arc::new(RwLock::new(graph.clone())),
            strides,
            _ghost: PhantomData,
        }
    }
}

impl<S: Shape, T: DType, D: Dev> GraphTensor<S, T, D> {
    /// Retrieve the graph for this `GraphTensor`.
    pub fn graph(&self) -> RwLockReadGuard<Graph<T>> {
        self.graph.read().unwrap()
    }

    /// Get the graph tensor ID.
    pub fn id(&self) -> GraphTensorId {
        self.id.clone()
    }
}

impl<const A: usize, T: DType, D: Dev> GraphTensor<R1<A>, T, D> {
    #[must_use]
    /// A GraphTensor representing a vector ranging from `start` to `stop` with `step` computed using A.
    pub fn arange(graph: &mut Graph<T>, start: T, stop: T) -> Self {
        let id = graph.next_id();
        let step = (stop.to_f64() - start.to_f64()) / (A as f64);
        let strides = contiguous_strides(&[A]);
        graph.add_op::<R1<A>>(
            Op::Arange {
                start,
                step: T::from_f64(step),
                stop,
            },
            &strides,
            &id,
        );
        Self {
            id,
            graph: Arc::new(RwLock::new(graph.clone())),
            strides,
            _ghost: PhantomData,
        }
    }
}

impl<T: DType, const A: usize, const B: usize, D: Dev> GraphTensor<R2<A, B>, T, D> {
    /// Return a view of this matrix with dimensions transposed (A x B -> B x A).
    pub fn t(&self) -> GraphTensor<R2<B, A>, T, D> {
        // swap strides for first two dimensions
        let mut new_strides = self.strides.clone();
        new_strides.swap(0, 1);

        let id = self.graph.write().unwrap().next_id();

        self.graph.write().unwrap().add_op::<R1<A>>(
            Op::Permute {
                v_id: self.id.clone(),
            },
            &new_strides,
            &id,
        );
        GraphTensor {
            id,
            graph: self.graph.clone(),
            strides: new_strides,
            _ghost: PhantomData,
        }
    }
}

impl<T: DType, const A: usize, const B: usize, const C: usize, D: Dev>
    GraphTensor<R3<A, B, C>, T, D>
{
    /// Return a view of this tensor with last two reversed axes (A x B x C -> A x C x B).
    pub fn t(&self) -> GraphTensor<R3<A, C, B>, T, D> {
        // swap strides for last two dimensions
        let mut new_strides = self.strides.clone();
        new_strides.swap(1, 2);

        let id = self.graph.write().unwrap().next_id();

        self.graph.write().unwrap().add_op::<R1<A>>(
            Op::Permute {
                v_id: self.id.clone(),
            },
            &new_strides,
            &id,
        );
        GraphTensor {
            id,
            graph: self.graph.clone(),
            strides: new_strides,
            _ghost: PhantomData,
        }
    }
}

macro_rules! graphtensor_binop {
    ($trait:ident, $fn_name:ident) => {
        impl<S: Shape, T: DType, D: Dev> $trait for GraphTensor<S, T, D> {
            type Output = GraphTensor<S, T, D>;
            /// Add an elementwise operation to the graph.
            fn $fn_name(self, rhs: Self) -> Self::Output {
                let id = self.graph.write().unwrap().next_id();
                self.graph.write().unwrap().add_op::<S>(
                    Op::BinaryOp {
                        l_id: self.id(),
                        r_id: rhs.id(),
                        operator: BinaryOpType::$trait,
                    },
                    &self.strides,
                    &id,
                );
                Self {
                    id,
                    graph: self.graph.clone(),
                    strides: self.strides.clone(),
                    _ghost: PhantomData,
                }
            }
        }
    };
}

graphtensor_binop!(Add, add);
graphtensor_binop!(Div, div);
graphtensor_binop!(Mul, mul);
graphtensor_binop!(Sub, sub);

impl<S: Shape, T: DType + Neg<Output = T>, D: Dev> Neg for GraphTensor<S, T, D> {
    type Output = GraphTensor<S, T, D>;
    /// Add an elementwise addition operation to the graph.
    fn neg(self) -> Self::Output {
        let id = self.graph.write().unwrap().next_id();
        self.graph.write().unwrap().add_op::<S>(
            Op::UnaryOp {
                v_id: self.id(),
                operator: UnaryOpType::Neg,
            },
            &self.strides,
            &id,
        );
        Self {
            id,
            graph: self.graph.clone(),
            strides: self.strides.clone(),
            _ghost: PhantomData,
        }
    }
}
