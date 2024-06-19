use std::{
    cell::Cell,
    fmt::Display,
    ops::Neg,
    rc::Rc,
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
        let next = GraphTensorId::from(*self.id.read().unwrap());
        *self.id.write().unwrap() += 1;
        next
    }

    /// Optimize by looking for mul-add pairs, convert to FMA
    fn optimize_fma(&mut self) {
        let ops = self.data.write().unwrap().clone();
        let mut new_ops = ops.clone();

        // This contains the indices of the first of the pair.
        for (x_id, x) in ops.iter().enumerate() {
            if let Op::BinaryOp {
                l_id: a_id,
                r_id: b_id,
                operator: BinaryOpType::Mul,
            } = x
            {
                // Check if next op uses this
                if let Op::BinaryOp {
                    l_id: l_y,
                    r_id: r_y,
                    operator: BinaryOpType::Add,
                } = &ops[x_id + 1]
                {
                    let y_id = x_id + 1;
                    if <&GraphTensorId as Into<usize>>::into(l_y) == x_id
                        || <&GraphTensorId as Into<usize>>::into(r_y) == x_id
                    {
                        // Want to see what is being added to the result of the mul
                        let rhs_add = if <&GraphTensorId as Into<usize>>::into(l_y) == x_id {
                            r_y
                        } else {
                            l_y
                        };
                        new_ops[y_id] = Op::FusedMulAdd {
                            a_id: GraphTensorId::from(a_id.0.get()),
                            b_id: GraphTensorId::from(b_id.0.get()),
                            c_id: GraphTensorId::from(rhs_add.0.get()),
                        };
                        new_ops[x_id] = Op::NoOp;

                        // Look for ops which actually use this one
                        for user in ops.iter() {
                            let ids = match user {
                                Op::Arange { start: _, step: _ } => vec![],
                                Op::BinaryOp {
                                    l_id,
                                    r_id,
                                    operator: _,
                                } => vec![l_id, r_id],
                                Op::Fill { v: _ } => vec![],
                                Op::UnaryOp { v_id, operator: _ } => vec![v_id],
                                Op::FusedMulAdd { a_id, b_id, c_id } => {
                                    vec![a_id, b_id, c_id]
                                }
                                Op::NoOp => vec![],
                            };
                            let used_ids = ids
                                .into_iter()
                                .filter(|id| <&GraphTensorId as Into<usize>>::into(id) != y_id)
                                .collect::<Vec<_>>();
                            if !used_ids.is_empty() {
                                for id in used_ids {
                                    // Tell the ops which use the result of the fma to source from there
                                    id.0.set(y_id);
                                }
                            }
                        }
                    }
                }
            }
        }

        *self.data.write().unwrap() = new_ops;
    }

    /// Apply the following optimizations
    /// - Fuse mul,add
    pub(crate) fn optimize(&mut self) {
        self.optimize_fma();
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
    Sqrt,
}

impl UnaryOpType {
    /// Can assume that the type T is available.
    pub fn fill_in_c_op(&self, val: impl Display) -> String {
        match self {
            Self::Neg => format!("-{val}"),
            Self::Sqrt => format!("static_cast<T>( sqrt( static_cast<double>({val}) ) )"),
        }
    }

    pub fn to_closure<T: DType + Neg<Output = T>>(&self) -> impl Fn(T) -> T {
        match self {
            Self::Neg => |x: T| -x,
            Self::Sqrt => |x: T| x.sqrt(),
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
    /// a * b + c
    FusedMulAdd {
        a_id: GraphTensorId,
        b_id: GraphTensorId,
        c_id: GraphTensorId,
    },
    NoOp,
}

#[derive(Clone, PartialEq, Debug)]
/// Graph tensor IDs can be cloned.
pub struct GraphTensorId(Rc<Cell<usize>>);

impl From<GraphTensorId> for usize {
    fn from(value: GraphTensorId) -> Self {
        value.0.get()
    }
}

impl From<&GraphTensorId> for usize {
    fn from(value: &GraphTensorId) -> Self {
        value.0.get()
    }
}

impl From<usize> for GraphTensorId {
    fn from(value: usize) -> Self {
        Self(Rc::new(Cell::new(value)))
    }
}
