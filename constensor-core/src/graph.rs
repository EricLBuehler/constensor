use std::{
    cell::Cell,
    collections::HashMap,
    env,
    fmt::Display,
    fs,
    path::Path,
    process::Command,
    rc::Rc,
    sync::{Arc, RwLock, RwLockReadGuard},
};

use crate::{DType, Result, Shape};

use petgraph::Graph as PetGraph;
use petgraph::{
    dot::{Config, Dot},
    graph::NodeIndex,
};

#[derive(Clone)]
pub struct GraphNode<T: DType> {
    pub op: Op<T>,
    pub shape: Vec<usize>,
}

#[derive(Clone)]
pub struct Graph<T: DType> {
    data: Arc<RwLock<Vec<GraphNode<T>>>>,
    id: Arc<RwLock<usize>>,
}

impl<T: DType> Graph<T> {
    /// Create an empty Graph
    pub fn empty() -> Self {
        Self {
            data: Arc::new(RwLock::new(Vec::new())),
            id: Arc::new(RwLock::new(0)),
        }
    }

    /// Read-only access to the list of operations
    pub fn get_ops(&self) -> RwLockReadGuard<Vec<GraphNode<T>>> {
        self.data.read().unwrap()
    }

    /// Append an operation to the graph
    pub(crate) fn add_op<S: Shape>(&self, op: Op<T>) {
        self.data.write().unwrap().push(GraphNode {
            op,
            shape: S::shape(),
        });
    }

    /// Generate the next unique tensor ID
    #[must_use]
    pub(crate) fn next_id(&mut self) -> GraphTensorId {
        let next = GraphTensorId::from(*self.id.read().unwrap());
        *self.id.write().unwrap() += 1;
        next
    }

    pub fn to_petgraph(&self) -> PetGraph<String, ()> {
        let ops = self.data.read().unwrap();
        let mut g = PetGraph::<String, ()>::new();
        // map from op‐index → Some(node) if we created a node, or None if it was a NoOp
        let mut idx_map: Vec<Option<NodeIndex>> = Vec::with_capacity(ops.len());

        // 1) Add only non‐NoOp nodes
        for op in ops.iter() {
            match op.op {
                Op::NoOp => {
                    idx_map.push(None);
                }
                _ => {
                    let label = match &op.op {
                        Op::Fill { v, .. } => format!("Fill({:?})", v),
                        Op::Arange {
                            start, step, stop, ..
                        } => {
                            format!(
                                "Arange(start={:?}, step={:?}, stop={:?})",
                                start, step, stop
                            )
                        }
                        Op::BinaryOp { operator, .. } => format!("BinOp({})", operator.as_c_op()),
                        Op::InplaceBinaryOp { operator, .. } => {
                            format!("InplaceBinOp({})", operator.as_c_op())
                        }
                        Op::UnaryOp { operator, .. } => format!("UnOp({:?})", operator),
                        Op::FusedMulAdd { .. } => "FMA".to_string(),
                        Op::InplaceFusedMulAdd { .. } => "InplaceFMA".to_string(),
                        // Matrix multiplication
                        Op::MatMul { .. } => "MatMul".to_string(),
                        // we already matched NoOp above
                        Op::NoOp => unreachable!(),
                    };
                    let node = g.add_node(label);
                    idx_map.push(Some(node));
                }
            }
        }

        // 2) Walk ops again and only connect edges for those dst nodes that exist
        for (i, op) in ops.iter().enumerate() {
            // if this op was NoOp, skip entirely
            let dst = match idx_map[i] {
                Some(dst) => dst,
                None => continue,
            };
            match &op.op {
                Op::BinaryOp { l_id, r_id, .. } | Op::InplaceBinaryOp { l_id, r_id, .. } => {
                    if let Some(src) = idx_map[usize::from(l_id)] {
                        g.add_edge(src, dst, ());
                    }
                    if let Some(src) = idx_map[usize::from(r_id)] {
                        g.add_edge(src, dst, ());
                    }
                }
                Op::UnaryOp { v_id, .. } => {
                    if let Some(src) = idx_map[usize::from(v_id)] {
                        g.add_edge(src, dst, ());
                    }
                }
                Op::FusedMulAdd {
                    a_id, b_id, c_id, ..
                }
                | Op::InplaceFusedMulAdd {
                    a_id, b_id, c_id, ..
                } => {
                    for src_id in [a_id, b_id, c_id] {
                        if let Some(src) = idx_map[usize::from(src_id)] {
                            g.add_edge(src, dst, ());
                        }
                    }
                }
                Op::MatMul { l_id, r_id, .. } => {
                    if let Some(src) = idx_map[usize::from(l_id)] {
                        g.add_edge(src, dst, ());
                    }
                    if let Some(src) = idx_map[usize::from(r_id)] {
                        g.add_edge(src, dst, ());
                    }
                }
                // NoOp and Fill/Arange don’t create incoming edges
                Op::NoOp | Op::Fill { .. } | Op::Arange { .. } => {}
            }
        }

        g
    }

    /// Produce a DOT format string of this graph.
    pub fn to_dot(&self) -> String {
        let g = self.to_petgraph();
        format!("{:?}", Dot::with_config(&g, &[Config::EdgeNoLabel]))
    }

    /// Visualize the graph by saving it to this file.
    ///
    /// Install graphvis:
    /// - brew install graphviz
    /// - apt install graphviz
    pub fn visualize<P: AsRef<Path>>(&self, filename: P) -> Result<()> {
        let path = filename.as_ref();
        let tmp_dir = env::temp_dir();
        let dot_path = tmp_dir.join("graph.dot");
        let png_path = path.to_path_buf();

        fs::write(&dot_path, self.to_dot())?;
        let status = Command::new("dot")
            .args([
                "-Tpng",
                &dot_path.display().to_string(),
                "-o",
                &png_path.display().to_string(),
            ])
            .status()?;
        if !status.success() {
            panic!("Graphviz failed");
        }

        Ok(())
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
            } = &x.op
            {
                // Check if next op uses this
                if let Op::BinaryOp {
                    l_id: l_y,
                    r_id: r_y,
                    operator: BinaryOpType::Add,
                } = &ops[x_id + 1].op
                {
                    let y_id = x_id + 1;
                    if <&GraphTensorId as Into<usize>>::into(l_y) == x_id
                        || <&GraphTensorId as Into<usize>>::into(r_y) == x_id
                            && x.shape == ops[x_id + 1].shape
                    {
                        // Want to see what is being added to the result of the mul
                        let rhs_add = if <&GraphTensorId as Into<usize>>::into(l_y) == x_id {
                            r_y
                        } else {
                            l_y
                        };
                        new_ops[y_id] = GraphNode {
                            op: Op::FusedMulAdd {
                                a_id: GraphTensorId::from(a_id.0.get()),
                                b_id: GraphTensorId::from(b_id.0.get()),
                                c_id: GraphTensorId::from(rhs_add.0.get()),
                            },
                            shape: x.shape.clone(),
                        };
                        new_ops[x_id] = GraphNode {
                            op: Op::NoOp,
                            shape: x.shape.clone(),
                        };

                        // Look for ops which actually use this one
                        for user in ops.iter() {
                            let ids = match &user.op {
                                Op::Arange {
                                    start: _,
                                    step: _,
                                    stop: _,
                                    ..
                                } => vec![],
                                Op::BinaryOp { l_id, r_id, .. }
                                | Op::InplaceBinaryOp { l_id, r_id, .. } => vec![l_id, r_id],
                                Op::Fill { v: _, .. } => vec![],
                                Op::UnaryOp {
                                    v_id, operator: _, ..
                                } => vec![v_id],
                                Op::FusedMulAdd {
                                    a_id, b_id, c_id, ..
                                }
                                | Op::InplaceFusedMulAdd {
                                    a_id, b_id, c_id, ..
                                } => {
                                    vec![a_id, b_id, c_id]
                                }
                                Op::MatMul { l_id, r_id, .. } => vec![l_id, r_id],
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

        // Remove any NoOp entries before storing back to the graph
        let filtered_ops = new_ops
            .into_iter()
            .filter(|op| !matches!(op.op, Op::NoOp))
            .collect::<Vec<_>>();
        *self.data.write().unwrap() = filtered_ops;
    }

    /// Count how often each tensor id is used as an input.
    fn count_input_usage(ops: &[GraphNode<T>]) -> HashMap<usize, usize> {
        let mut usage: HashMap<usize, usize> = HashMap::new();
        for op in ops {
            match &op.op {
                Op::BinaryOp { l_id, r_id, .. } | Op::InplaceBinaryOp { l_id, r_id, .. } => {
                    *usage.entry(usize::from(l_id)).or_default() += 1;
                    *usage.entry(usize::from(r_id)).or_default() += 1;
                }
                Op::UnaryOp { v_id, .. } => {
                    *usage.entry(usize::from(v_id)).or_default() += 1;
                }
                Op::FusedMulAdd {
                    a_id, b_id, c_id, ..
                }
                | Op::InplaceFusedMulAdd {
                    a_id, b_id, c_id, ..
                } => {
                    *usage.entry(usize::from(a_id)).or_default() += 1;
                    *usage.entry(usize::from(b_id)).or_default() += 1;
                    *usage.entry(usize::from(c_id)).or_default() += 1;
                }
                Op::MatMul { l_id, r_id, .. } => {
                    *usage.entry(usize::from(l_id)).or_default() += 1;
                    *usage.entry(usize::from(r_id)).or_default() += 1;
                }
                Op::NoOp | Op::Fill { .. } | Op::Arange { .. } => {}
            }
        }
        usage
    }

    /// Optimize by inplacing binary operations when inputs are not reused.
    fn optimize_inplace_bin(&mut self) {
        let ops = self.data.write().unwrap().clone();
        let mut new_ops = ops.clone();
        let usage = Self::count_input_usage(&ops);
        // Transform eligible BinaryOps into InplaceBinaryOps.
        for (i, op) in ops.iter().enumerate() {
            if let Op::BinaryOp {
                l_id,
                r_id,
                operator,
            } = &op.op
            {
                let l_idx = usize::from(l_id);
                let r_idx = usize::from(r_id);
                let l_use = usage.get(&l_idx).copied().unwrap_or(0);
                let r_use = usage.get(&r_idx).copied().unwrap_or(0);
                if l_use == 1 || r_use == 1 {
                    // Choose target for in-place: if both, default to lhs.
                    let target = if r_use > l_use {
                        r_id.clone()
                    } else {
                        l_id.clone()
                    };
                    // Replace with InplaceBinaryOp
                    new_ops[i] = GraphNode {
                        op: Op::InplaceBinaryOp {
                            out: target.clone(),
                            l_id: l_id.clone(),
                            r_id: r_id.clone(),
                            operator: *operator,
                        },
                        shape: op.shape.clone(),
                    };
                    // Update all future uses of this op's result (index i) to use 'target'.
                    for fut in new_ops.iter_mut().skip(i + 1) {
                        match &fut.op {
                            Op::BinaryOp { l_id, r_id, .. }
                            | Op::InplaceBinaryOp { l_id, r_id, .. } => {
                                if usize::from(l_id) == i {
                                    l_id.0.set(usize::from(&target));
                                }
                                if usize::from(r_id) == i {
                                    r_id.0.set(usize::from(&target));
                                }
                            }
                            Op::UnaryOp { v_id, .. } => {
                                if usize::from(v_id) == i {
                                    v_id.0.set(usize::from(&target));
                                }
                            }
                            Op::FusedMulAdd {
                                a_id, b_id, c_id, ..
                            }
                            | Op::InplaceFusedMulAdd {
                                a_id, b_id, c_id, ..
                            } => {
                                if usize::from(a_id) == i {
                                    a_id.0.set(usize::from(&target));
                                }
                                if usize::from(b_id) == i {
                                    b_id.0.set(usize::from(&target));
                                }
                                if usize::from(c_id) == i {
                                    c_id.0.set(usize::from(&target));
                                }
                            }
                            Op::MatMul { l_id, r_id, .. } => {
                                if usize::from(l_id) == i {
                                    l_id.0.set(usize::from(&target));
                                }
                                if usize::from(r_id) == i {
                                    r_id.0.set(usize::from(&target));
                                }
                            }
                            Op::NoOp | Op::Fill { .. } | Op::Arange { .. } => {}
                        }
                    }
                }
            }
        }
        // Commit the transformed op list.
        *self.data.write().unwrap() = new_ops;
    }

    /// Optimize by inplacing fused multiply-add (FMA) operations when inputs are not reused.
    fn optimize_inplace_fma(&mut self) {
        let ops = self.data.write().unwrap().clone();
        let mut new_ops = ops.clone();
        let usage = Self::count_input_usage(&ops);
        for (i, op) in ops.iter().enumerate() {
            if let Op::FusedMulAdd { a_id, b_id, c_id } = &op.op {
                let mut target = None;
                // If an input is used only once, we can reuse its buffer; default order: a_id, then b_id, then c_id
                if *usage.get(&usize::from(a_id)).unwrap_or(&0) == 1 {
                    target = Some(a_id.clone());
                } else if *usage.get(&usize::from(b_id)).unwrap_or(&0) == 1 {
                    target = Some(b_id.clone());
                } else if *usage.get(&usize::from(c_id)).unwrap_or(&0) == 1 {
                    target = Some(c_id.clone());
                }
                if let Some(out) = target {
                    new_ops[i] = GraphNode {
                        op: Op::InplaceFusedMulAdd {
                            out: out.clone(),
                            a_id: a_id.clone(),
                            b_id: b_id.clone(),
                            c_id: c_id.clone(),
                        },
                        shape: op.shape.clone(),
                    };
                    // Update all future ops that reference the original index i
                    for fut in new_ops.iter_mut().skip(i + 1) {
                        match &fut.op {
                            Op::BinaryOp { l_id, r_id, .. }
                            | Op::InplaceBinaryOp { l_id, r_id, .. } => {
                                if usize::from(l_id) == i {
                                    l_id.0.set(usize::from(&out));
                                }
                                if usize::from(r_id) == i {
                                    r_id.0.set(usize::from(&out));
                                }
                            }
                            Op::UnaryOp { v_id, .. } => {
                                if usize::from(v_id) == i {
                                    v_id.0.set(usize::from(&out));
                                }
                            }
                            Op::FusedMulAdd {
                                a_id, b_id, c_id, ..
                            }
                            | Op::InplaceFusedMulAdd {
                                a_id, b_id, c_id, ..
                            } => {
                                if usize::from(a_id) == i {
                                    a_id.0.set(usize::from(&out));
                                }
                                if usize::from(b_id) == i {
                                    b_id.0.set(usize::from(&out));
                                }
                                if usize::from(c_id) == i {
                                    c_id.0.set(usize::from(&out));
                                }
                            }
                            Op::MatMul { l_id, r_id, .. } => {
                                if usize::from(l_id) == i {
                                    l_id.0.set(usize::from(&out));
                                }
                                if usize::from(r_id) == i {
                                    r_id.0.set(usize::from(&out));
                                }
                            }
                            Op::NoOp | Op::Fill { .. } | Op::Arange { .. } => {}
                        }
                    }
                }
            }
        }
        *self.data.write().unwrap() = new_ops;
    }

    /// Optimize this graph.
    ///
    /// Apply the following optimizations
    /// - Fuse mul,add
    pub fn optimize(&mut self) {
        self.optimize_fma();
        self.optimize_inplace_bin();
        self.optimize_inplace_fma();
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum BinaryOpType {
    Add,
    Div,
    Sub,
    Mul,
}

impl BinaryOpType {
    pub fn as_c_op(&self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Div => "/",
            Self::Sub => "-",
            Self::Mul => "*",
        }
    }

    pub fn as_closure<T: DType>(&self) -> impl Fn(T, T) -> T {
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
    pub fn fill_in_c_op(&self, val: impl Display) -> String {
        match self {
            Self::Neg => format!("-{val}"),
            Self::Sqrt => format!("static_cast<T>( sqrt( static_cast<double>({val}) ) )"),
        }
    }

    pub fn to_closure<T: DType>(&self) -> impl Fn(T) -> T {
        match self {
            Self::Neg => T::maybe_neg,
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
        stop: T,
    },
    BinaryOp {
        l_id: GraphTensorId,
        r_id: GraphTensorId,
        operator: BinaryOpType,
    },
    InplaceBinaryOp {
        out: GraphTensorId,
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
    /// a * b + c
    InplaceFusedMulAdd {
        out: GraphTensorId,
        a_id: GraphTensorId,
        b_id: GraphTensorId,
        c_id: GraphTensorId,
    },
    /// Matrix multiplication: l_id (A x B) * r_id (B x C) -> (A x C)
    MatMul {
        l_id: GraphTensorId,
        r_id: GraphTensorId,
        /// Inner dimension B
        k: usize,
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
