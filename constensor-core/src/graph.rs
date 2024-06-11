use std::{
    ops::{Deref, Neg},
    sync::{Arc, RwLock, RwLockReadGuard},
};

use crate::DType;

#[derive(Clone)]
pub struct Graph<T: DType> {
    data: Arc<RwLock<Vec<Op<T>>>>,
    id: Arc<RwLock<usize>>,
}

impl<T: DType> Graph<T> {
    pub fn empty() -> Self {
        Self {
            data: Arc::new(RwLock::new(Vec::new())),
            id: Arc::new(RwLock::new(0)),
        }
    }

    pub fn get_ops(&self) -> RwLockReadGuard<Vec<Op<T>>> {
        self.data.read().unwrap()
    }
    pub(crate) fn add_op(&self, op: Op<T>) {
        self.data.write().unwrap().push(op);
    }

    #[must_use]
    pub(crate) fn next_id(&mut self) -> GraphTensorId {
        let next = GraphTensorId(*self.id.read().unwrap());
        *self.id.write().unwrap() += 1;
        next
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum BinaryOpType {
    Add,
    Div,
    Sub,
    Mul,
}

impl BinaryOpType {
    pub fn to_c_op(&self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Div => "/",
            Self::Sub => "-",
            Self::Mul => "*",
        }
    }

    pub fn to_closure<T: DType>(&self) -> impl Fn(T, T) -> T {
        match self {
            Self::Add => |x, y| x + y,
            Self::Div => |x, y| x / y,
            Self::Sub => |x, y| x - y,
            Self::Mul => |x, y| x * y,
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum UnaryOpType {
    Neg,
}

impl UnaryOpType {
    pub fn to_c_op(&self) -> &'static str {
        match self {
            Self::Neg => "-",
        }
    }

    pub fn to_closure<T: DType + Neg<Output = T>>(&self) -> impl Fn(T) -> T {
        match self {
            Self::Neg => |x: T| -x,
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum Op<T: DType> {
    Fill {
        v: T,
    },
    Arange {
        start: T,
        step: T,
    },
    BinaryOp {
        l_id: GraphTensorId,
        r_id: GraphTensorId,
        operator: BinaryOpType,
    },
    UnaryOp {
        v_id: GraphTensorId,
        operator: UnaryOpType,
    },
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GraphTensorId(usize);

impl Deref for GraphTensorId {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<usize> for GraphTensorId {
    fn from(value: usize) -> Self {
        Self(value)
    }
}
