use std::{
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
    sync::{Arc, RwLock, RwLockReadGuard},
};

use crate::{
    device::Dev,
    graph::{BinaryOpType, Graph, GraphTensorId, Op, UnaryOpType},
    tensor::from_storage,
    DType, Result, Shape, Tensor, R1,
};

/// A tensor representing an intermediary result of a graph. Performing operations
/// on this tensor will not cause any computations.
#[derive(Clone)]
pub struct GraphTensor<S: Shape, T: DType, D: Dev> {
    id: GraphTensorId,
    graph: Arc<RwLock<Graph<T>>>,
    _ghost: PhantomData<(S, T, D)>,
}

impl<S: Shape, T: DType, D: Dev> GraphTensor<S, T, D> {
    #[must_use]
    /// Create a tensor filled with some value.
    pub fn fill(mut graph: Graph<T>, v: T) -> Self {
        let id = graph.next_id();
        graph.add_op(Op::Fill { v });
        Self {
            id,
            graph: Arc::new(RwLock::new(graph)),
            _ghost: PhantomData,
        }
    }

    #[must_use]
    /// Create a tensor filled with zeros.
    pub fn zeros(graph: Graph<T>) -> Self {
        Self::fill(graph, T::ZERO)
    }

    #[must_use]
    /// Create a tensor filled with ones.
    pub fn ones(graph: Graph<T>) -> Self {
        Self::fill(graph, T::ONE)
    }

    /// Retrieve the graph for this `GraphTensor`.
    pub fn graph(&self) -> RwLockReadGuard<Graph<T>> {
        self.graph.read().unwrap()
    }

    /// Get the graph tensor ID.
    pub fn id(&self) -> GraphTensorId {
        self.id
    }

    /// Convert this `GraphTensor` into a concrete `Tensor`.
    /// Only unsigned operations.
    pub fn to_tensor_unsigned(self) -> Result<Tensor<S, T, D>> {
        let graph = self.graph.read().unwrap();
        let nodes = &*graph.get_ops();

        let device = D::resolve()?;
        let storage = device.compile_and_run_graph_unsigned::<T, S>(nodes)?;
        Ok(from_storage(Arc::new(storage)))
    }
}

impl<S: Shape, T: DType + Neg<Output = T>, D: Dev> GraphTensor<S, T, D> {
    /// Convert this `GraphTensor` into a concrete `Tensor`.
    /// Only unsigned operations.
    pub fn to_tensor(self) -> Result<Tensor<S, T, D>> {
        let graph = self.graph.read().unwrap();
        let nodes = &*graph.get_ops();

        let device = D::resolve()?;
        let storage = device.compile_and_run_graph::<T, S>(nodes)?;
        Ok(from_storage(Arc::new(storage)))
    }
}

impl<const A: usize, T: DType, D: Dev> GraphTensor<R1<A>, T, D> {
    /// A GraphTensor representing a vector ranging from `start` to `A` with step `step`.
    pub fn arange(mut graph: Graph<T>, start: T, step: T) -> Self {
        let id = graph.next_id();
        graph.add_op(Op::Arange { start, step });
        Self {
            id,
            graph: Arc::new(RwLock::new(graph)),
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
                self.graph.write().unwrap().add_op(Op::BinaryOp {
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
        self.graph.write().unwrap().add_op(Op::UnaryOp {
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
