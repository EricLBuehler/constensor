use std::{
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
    sync::{Arc, RwLock, RwLockReadGuard},
};

use crate::{
    device::Dev,
    graph::{BinaryOpType, Graph, GraphTensorId, Op, UnaryOpType},
    DType, Shape, R1, R3,
};

/// A tensor representing an intermediary result of a graph. Performing operations
/// on this tensor will not cause any computations.
#[derive(Clone)]
pub struct GraphTensor<S: Shape, T: DType, D: Dev> {
    id: GraphTensorId,
    graph: Arc<RwLock<Graph<T>>>,
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
        self.graph
            .write()
            .unwrap()
            .add_op::<R3<B, M, N>>(Op::MatMul {
                l_id: self.id(),
                r_id: rhs.id(),
                o_id: None,
                k: K,
                alpha: T::ZERO,
                beta: T::ONE,
            });
        GraphTensor {
            id: self.graph.write().unwrap().next_id(),
            graph: self.graph.clone(),
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
        self.graph
            .write()
            .unwrap()
            .add_op::<R3<B, M, N>>(Op::MatMul {
                l_id: self.id(),
                r_id: rhs.id(),
                o_id: Some(out.id()),
                k: K,
                alpha,
                beta,
            });
        GraphTensor {
            id: self.graph.write().unwrap().next_id(),
            graph: self.graph.clone(),
            _ghost: PhantomData,
        }
    }
}

impl<S: Shape, T: DType, D: Dev> GraphTensor<S, T, D> {
    #[must_use]
    /// Create a tensor filled with some value.
    pub fn fill(graph: &mut Graph<T>, v: T) -> Self {
        let id = graph.next_id();
        graph.add_op::<S>(Op::Fill { v });
        Self {
            id,
            graph: Arc::new(RwLock::new(graph.clone())),
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
        self.graph.write().unwrap().add_op::<S>(Op::UnaryOp {
            v_id: self.id(),
            operator: UnaryOpType::Sqrt,
        });
        Self {
            id: self.graph.write().unwrap().next_id(),
            graph: self.graph.clone(),
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
        graph.add_op::<R1<A>>(Op::Arange {
            start,
            step: T::from_f64(step),
            stop,
        });
        Self {
            id,
            graph: Arc::new(RwLock::new(graph.clone())),
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
                self.graph.write().unwrap().add_op::<S>(Op::BinaryOp {
                    l_id: self.id(),
                    r_id: rhs.id(),
                    operator: BinaryOpType::$trait,
                });
                Self {
                    id: self.graph.write().unwrap().next_id(),
                    graph: self.graph.clone(),
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
        self.graph.write().unwrap().add_op::<S>(Op::UnaryOp {
            v_id: self.id(),
            operator: UnaryOpType::Neg,
        });
        Self {
            id: self.graph.write().unwrap().next_id(),
            graph: self.graph.clone(),
            _ghost: PhantomData,
        }
    }
}
