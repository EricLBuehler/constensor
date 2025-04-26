use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    collections::HashMap,
    env,
    fmt::Display,
    fs,
    hash::Hash,
    marker::PhantomData,
    path::Path,
    process::Command,
    sync::{Arc, RwLock, RwLockReadGuard},
};

use crate::{device::Dev, tensor::concretetensor::from_storage, DType, Result, Shape, Tensor};

use cubecl::CubeType;
use petgraph::Graph as PetGraph;
use petgraph::{dot::Dot, graph::NodeIndex};

#[derive(Clone, Debug)]
pub struct GraphNode<T: DType> {
    pub op: Op<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub id: GraphTensorId,
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
    pub(crate) fn add_op<S: Shape>(&self, op: Op<T>, strides: &[usize], id: &GraphTensorId) {
        self.data.write().unwrap().push(GraphNode {
            op,
            shape: S::shape(),
            strides: strides.to_vec(),
            id: id.clone(),
        });
    }

    /// Generate the next unique tensor ID
    #[must_use]
    pub(crate) fn next_id(&mut self) -> GraphTensorId {
        let next = GraphTensorId::out_of_place(*self.id.read().unwrap());
        *self.id.write().unwrap() += 1;
        next
    }

    pub fn to_petgraph(&self) -> PetGraph<String, String> {
        let ops = self.data.read().unwrap();
        let mut g = PetGraph::<String, String>::new();
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
                        Op::Fill { v, .. } => format!("Fill({v:?})"),
                        Op::Arange {
                            start, step, stop, ..
                        } => {
                            format!("Arange(start={start:?}, step={step:?}, stop={stop:?})")
                        }
                        Op::Rand => "Rand".to_string(),
                        Op::Randn { mean, std } => {
                            format!("Randn(mean={mean:?}, std={std:?})")
                        }
                        Op::BinaryOp { operator, .. } => format!("BinOp({})", operator.as_c_op()),
                        Op::UnaryOp { operator, .. } => format!("UnOp({operator:?})"),
                        Op::FusedMulAdd { .. } => "FMA".to_string(),
                        // Matrix multiplication
                        Op::MatMul { .. } => "MatMul".to_string(),
                        Op::Permute { v_id: _ } => "Permute".to_string(),
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
                Op::BinaryOp { l_id, r_id, .. } => {
                    if let Some(src) = idx_map[l_id.get()] {
                        let mut label = "l".to_string();
                        if l_id.is_inplace() {
                            label.push('*');
                        }
                        g.add_edge(src, dst, label.clone());
                    }
                    if let Some(src) = idx_map[r_id.get()] {
                        let mut label = "r".to_string();
                        if r_id.is_inplace() {
                            label.push('*');
                        }
                        g.add_edge(src, dst, label.clone());
                    }
                }
                Op::UnaryOp { v_id, .. } => {
                    if let Some(src) = idx_map[v_id.get()] {
                        let mut label = "v".to_string();
                        if v_id.is_inplace() {
                            label.push('*');
                        }
                        g.add_edge(src, dst, label.clone());
                    }
                }
                Op::FusedMulAdd {
                    a_id, b_id, c_id, ..
                } => {
                    for (prefix, src_id) in [("a", a_id), ("b", b_id), ("c", c_id)].iter() {
                        if let Some(src) = idx_map[src_id.get()] {
                            let mut label = prefix.to_string();
                            if src_id.is_inplace() {
                                label.push('*');
                            }
                            g.add_edge(src, dst, label.clone());
                        }
                    }
                }
                Op::MatMul {
                    l_id, r_id, o_id, ..
                } => {
                    if let Some(src) = idx_map[l_id.get()] {
                        let mut label = "l".to_string();
                        if l_id.is_inplace() {
                            label.push('*');
                        }
                        g.add_edge(src, dst, label.clone());
                    }
                    if let Some(src) = idx_map[r_id.get()] {
                        let mut label = "r".to_string();
                        if r_id.is_inplace() {
                            label.push('*');
                        }
                        g.add_edge(src, dst, label.clone());
                    }
                    if let Some(o_id) = o_id {
                        if let Some(src) = idx_map[o_id.get()] {
                            let mut label = "o".to_string();
                            if o_id.is_inplace() {
                                label.push('*');
                            }
                            g.add_edge(src, dst, label.clone());
                        }
                    }
                }
                Op::Permute { v_id, .. } => {
                    if let Some(src) = idx_map[v_id.get()] {
                        let mut label = "v".to_string();
                        if v_id.is_inplace() {
                            label.push('*');
                        }
                        g.add_edge(src, dst, label.clone());
                    }
                }
                // NoOp, Fill/Arange, Rand/Randn don’t create incoming edges
                Op::NoOp | Op::Fill { .. } | Op::Arange { .. } | Op::Rand | Op::Randn { .. } => {}
            }
        }

        g
    }

    /// Produce a DOT format string of this graph.
    pub fn to_dot(&self) -> String {
        let g = self.to_petgraph();
        format!("{:?}", Dot::with_config(&g, &[]))
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

    /// Optimize by performing constant folding:
    ///   - Fold BinaryOp and UnaryOp when all operands are constant Fill ops.
    fn optimize_const(&mut self) {
        // Clone current ops for inspection
        let ops = self.data.read().unwrap().clone();
        let mut new_ops = ops.clone();
        for (i, node) in ops.iter().enumerate() {
            match &node.op {
                Op::BinaryOp {
                    l_id,
                    r_id,
                    operator,
                } => {
                    let l_idx = l_id.get();
                    let r_idx = r_id.get();
                    // both operands are constant fills
                    if let Op::Fill { v: v1 } = &new_ops[l_idx].op {
                        if let Op::Fill { v: v2 } = &new_ops[r_idx].op {
                            let v = operator.as_closure()(*v1, *v2);
                            new_ops[i] = GraphNode {
                                op: Op::Fill { v },
                                ..node.clone()
                            };
                        }
                    }
                }
                Op::UnaryOp { v_id, operator } => {
                    let idx = v_id.get();
                    // operand is a constant fill
                    if let Op::Fill { v: v0 } = &new_ops[idx].op {
                        let v = operator.to_closure()(*v0);
                        new_ops[i] = GraphNode {
                            op: Op::Fill { v },
                            ..node.clone()
                        };
                    }
                }
                _ => {}
            }
        }
        // Commit folded constants
        *self.data.write().unwrap() = new_ops;
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
                    if l_y.get() == x_id || r_y.get() == x_id && x.shape == ops[x_id + 1].shape {
                        // Want to see what is being added to the result of the mul
                        let rhs_add = if l_y.get() == x_id { r_y } else { l_y };
                        new_ops[y_id] = GraphNode {
                            op: Op::FusedMulAdd {
                                a_id: a_id.clone(),
                                b_id: b_id.clone(),
                                c_id: rhs_add.clone(),
                            },
                            ..x.clone()
                        };
                        new_ops[x_id] = GraphNode {
                            op: Op::NoOp,
                            ..x.clone()
                        };

                        // Look for ops which actually use this one
                        for user in new_ops.iter() {
                            let ids = match &user.op {
                                Op::Arange {
                                    start: _,
                                    step: _,
                                    stop: _,
                                    ..
                                } => vec![],
                                Op::Rand => vec![],
                                Op::Randn { mean: _, std: _ } => vec![],
                                Op::BinaryOp { l_id, r_id, .. } => vec![l_id, r_id],
                                Op::Fill { v: _, .. } => vec![],
                                Op::UnaryOp {
                                    v_id, operator: _, ..
                                } => vec![v_id],
                                Op::FusedMulAdd {
                                    a_id, b_id, c_id, ..
                                } => {
                                    vec![a_id, b_id, c_id]
                                }
                                Op::MatMul {
                                    l_id, r_id, o_id, ..
                                } => o_id
                                    .as_ref()
                                    .map(|o| vec![l_id, r_id, o])
                                    .unwrap_or(vec![l_id, r_id]),
                                Op::Permute { v_id } => vec![v_id],
                                Op::NoOp => vec![],
                            };

                            // We are going to remove the noop so this is necessary to fix the indices.
                            let used_ids = ids
                                .into_iter()
                                .filter(|id| id.get() == y_id)
                                .collect::<Vec<_>>();
                            if !used_ids.is_empty() {
                                for id in used_ids {
                                    // Tell the ops which use the result of the fma to source from there
                                    id.set(x_id);
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
    #[allow(clippy::mutable_key_type)]
    fn count_input_usage(ops: &[GraphNode<T>]) -> HashMap<GraphTensorId, usize> {
        #[allow(clippy::mutable_key_type)]
        let mut usage: HashMap<GraphTensorId, usize> = HashMap::new();
        for op in ops {
            match &op.op {
                Op::BinaryOp { l_id, r_id, .. } => {
                    *usage.entry(l_id.clone()).or_default() += 1;
                    *usage.entry(r_id.clone()).or_default() += 1;
                }
                Op::UnaryOp { v_id, .. } => {
                    *usage.entry(v_id.clone()).or_default() += 1;
                }
                Op::FusedMulAdd {
                    a_id, b_id, c_id, ..
                } => {
                    *usage.entry(a_id.clone()).or_default() += 1;
                    *usage.entry(b_id.clone()).or_default() += 1;
                    *usage.entry(c_id.clone()).or_default() += 1;
                }
                Op::MatMul {
                    l_id, r_id, o_id, ..
                } => {
                    *usage.entry(l_id.clone()).or_default() += 1;
                    *usage.entry(r_id.clone()).or_default() += 1;
                    if let Some(o_id) = o_id {
                        *usage.entry(o_id.clone()).or_default() += 1;
                    }
                }
                Op::Permute { v_id } => {
                    *usage.entry(v_id.clone()).or_default() += 1;
                }
                // No input usage for these ops
                Op::NoOp | Op::Fill { .. } | Op::Arange { .. } | Op::Rand | Op::Randn { .. } => {}
            }
        }
        usage
    }

    /// Optimize by inplacing binary operations when inputs are not reused.
    fn optimize_inplace_bin(&mut self) {
        let ops = self.data.write().unwrap().clone();
        let mut new_ops = ops.clone();
        #[allow(clippy::mutable_key_type)]
        let usage = Self::count_input_usage(&ops);
        // Transform eligible BinaryOps into InplaceBinaryOps.
        for (i, op) in ops.iter().enumerate() {
            if let Op::BinaryOp {
                l_id,
                r_id,
                operator,
            } = &op.op
            {
                let l_use = usage.get(l_id).copied().unwrap_or(0);
                let r_use = usage.get(r_id).copied().unwrap_or(0);
                if l_use <= 1 || r_use <= 1 {
                    // Choose target for in-place: if both, default to lhs.
                    let target = if r_use > l_use {
                        r_id.clone()
                    } else {
                        l_id.clone()
                    };
                    // Replace with InplaceBinaryOp
                    new_ops[i] = GraphNode {
                        op: Op::BinaryOp {
                            l_id: l_id.clone().to_inplace_if(&target == l_id),
                            r_id: r_id.clone().to_inplace_if(&target == r_id),
                            operator: *operator,
                        },
                        ..op.clone()
                    };
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
        #[allow(clippy::mutable_key_type)]
        let usage = Self::count_input_usage(&ops);
        for (i, op) in ops.iter().enumerate() {
            if let Op::FusedMulAdd { a_id, b_id, c_id } = &op.op {
                let mut target = None;
                // If an input is used only once, we can reuse its buffer; default order: a_id, then b_id, then c_id
                if *usage.get(a_id).unwrap_or(&0) <= 1 {
                    target = Some(a_id.clone());
                } else if *usage.get(b_id).unwrap_or(&0) <= 1 {
                    target = Some(b_id.clone());
                } else if *usage.get(c_id).unwrap_or(&0) <= 1 {
                    target = Some(c_id.clone());
                }
                if let Some(out) = target {
                    new_ops[i] = GraphNode {
                        op: Op::FusedMulAdd {
                            a_id: a_id.clone().to_inplace_if(&out == a_id),
                            b_id: b_id.clone().to_inplace_if(&out == b_id),
                            c_id: c_id.clone().to_inplace_if(&out == c_id),
                        },
                        ..op.clone()
                    };
                }
            }
        }
        *self.data.write().unwrap() = new_ops;
    }

    /// Optimize by inplacing the output of a matmul when inputs are not reused.
    fn optimize_inplace_matmul(&mut self) {
        let ops = self.data.write().unwrap().clone();
        let mut new_ops = ops.clone();
        #[allow(clippy::mutable_key_type)]
        let usage = Self::count_input_usage(&ops);
        // Transform eligible BinaryOps into InplaceBinaryOps.
        for (i, op) in ops.iter().enumerate() {
            if let Op::MatMul {
                o_id: Some(o_id),
                l_id,
                r_id,
                k,
                alpha,
                beta,
            } = &op.op
            {
                let o_use = usage.get(o_id).copied().unwrap_or(0);
                if o_use <= 1 {
                    // Replace with InplaceBinaryOp
                    new_ops[i] = GraphNode {
                        op: Op::MatMul {
                            o_id: Some(o_id.to_inplace()),
                            l_id: l_id.clone(),
                            r_id: r_id.clone(),
                            k: *k,
                            alpha: *alpha,
                            beta: *beta,
                        },
                        ..op.clone()
                    };
                }
            }
        }
        // Commit the transformed op list.
        *self.data.write().unwrap() = new_ops;
    }

    /// Remove nodes whose outputs are never used, except the final output node.
    fn optimize_dead_code(&mut self) {
        // Clone current ops
        let old_ops = self.data.read().unwrap().clone();
        let n = old_ops.len();
        // Mark reachable nodes: start from final output
        let mut keep = vec![false; n];
        if n > 0 {
            keep[n - 1] = true;
        }
        // Propagate reachability backwards
        for i in (0..n).rev() {
            if keep[i] {
                match &old_ops[i].op {
                    Op::BinaryOp { l_id, r_id, .. } => {
                        keep[l_id.get()] = true;
                        keep[r_id.get()] = true;
                    }
                    Op::UnaryOp { v_id, .. } => {
                        keep[v_id.get()] = true;
                    }
                    Op::FusedMulAdd {
                        a_id, b_id, c_id, ..
                    } => {
                        keep[a_id.get()] = true;
                        keep[b_id.get()] = true;
                        keep[c_id.get()] = true;
                    }
                    Op::MatMul {
                        l_id, r_id, o_id, ..
                    } => {
                        keep[l_id.get()] = true;
                        keep[r_id.get()] = true;
                        if let Some(o_id) = o_id {
                            keep[o_id.get()] = true;
                        }
                    }
                    Op::Permute { v_id, .. } => {
                        keep[v_id.get()] = true;
                    }
                    Op::NoOp
                    | Op::Fill { .. }
                    | Op::Arange { .. }
                    | Op::Rand
                    | Op::Randn { .. } => (),
                }
            }
        }
        // Build new ops and map old indices to new indices
        let mut index_map = std::collections::HashMap::new();
        let mut new_ops = Vec::new();
        for (old_idx, node) in old_ops.into_iter().enumerate() {
            if keep[old_idx] {
                let new_idx = new_ops.len();
                index_map.insert(old_idx, new_idx);
                new_ops.push(node);
            }
        }
        // Update tensor IDs in remaining ops
        for node in new_ops.iter_mut() {
            match &mut node.op {
                Op::BinaryOp { l_id, r_id, .. } => {
                    let old_l = l_id.get();
                    let old_r = r_id.get();
                    l_id.set(*index_map.get(&old_l).unwrap());
                    r_id.set(*index_map.get(&old_r).unwrap());
                }
                Op::UnaryOp { v_id, .. } => {
                    let old_v = v_id.get();
                    v_id.set(*index_map.get(&old_v).unwrap());
                }
                Op::FusedMulAdd {
                    a_id, b_id, c_id, ..
                } => {
                    let old_a = a_id.get();
                    let old_b = b_id.get();
                    let old_c = c_id.get();
                    a_id.set(*index_map.get(&old_a).unwrap());
                    b_id.set(*index_map.get(&old_b).unwrap());
                    c_id.set(*index_map.get(&old_c).unwrap());
                }
                Op::MatMul {
                    l_id, r_id, o_id, ..
                } => {
                    let old_l = l_id.get();
                    let old_r = r_id.get();
                    l_id.set(*index_map.get(&old_l).unwrap());
                    r_id.set(*index_map.get(&old_r).unwrap());
                    if let Some(o_id) = o_id {
                        let old_o = o_id.get();
                        o_id.set(*index_map.get(&old_o).unwrap());
                    }
                }
                _ => {}
            }
        }
        // Commit pruned graph
        *self.data.write().unwrap() = new_ops;
    }

    /// Optimize this graph.
    ///
    /// Apply the following optimizations:
    /// - Constant folding of elementwise fills
    /// - Fuse mul-add into FMA
    /// - Inplace binary operations when safe
    /// - Inplace fused multiply-add when safe
    /// - Inplace matrix-multiplication when safe
    /// - Dead code removal
    pub fn optimize(&mut self) {
        // Constant folding first
        self.optimize_const();
        // Fuse mul-add into FMA
        self.optimize_fma();
        self.optimize_inplace_bin();
        self.optimize_inplace_fma();
        self.optimize_inplace_matmul();
        // Remove dead code
        self.optimize_dead_code();
    }

    /// Compile this graph and insert device-specific optimizations such as CUDA streams.
    pub fn compile<S: Shape, D: Dev>(self) -> Result<CompiledGraph<S, T, D>> {
        if self
            .data
            .read()
            .unwrap()
            .last()
            .is_some_and(|last| last.shape != S::shape())
        {
            let read = self.data.read();
            let last = read.as_ref().unwrap().last().unwrap();

            crate::bail!(
                "Graph compiled shape is {:?} does not match the last node shape {:?}!",
                &last.shape,
                S::shape()
            );
        }

        let device = D::resolve()?;

        device.compile(self.data.read().unwrap().clone())
    }
}

/// A representation of the compiled graph. The shape is the output shape.
pub enum CompiledGraph<S: Shape, T: DType, D: Dev> {
    Cpu {
        order: Vec<usize>,
        graph: Vec<GraphNode<T>>,
        ghost: PhantomData<(S, T, D)>,
    },
    /// Compiled graph for WGPU backend.
    Wgpu {
        order: Vec<usize>,
        graph: Vec<GraphNode<T>>,
        ghost: PhantomData<(S, T, D)>,
    },
    #[cfg(feature = "cuda")]
    Cuda {
        kernels: Vec<crate::cuda_backend::CudaCompiledKernel<T>>,
        ghost: PhantomData<(S, T, D)>,
    },
}

impl<S: Shape, T: DType, D: Dev> CompiledGraph<S, T, D> {
    /// Run the precompiled graph. This executes all nodes on the specified backend device and returns a concrete tensor.
    pub fn run(&self) -> Result<Tensor<S, T, D>> {
        let device = D::resolve()?;
        let storage = device.run_graph(self)?;
        Ok(from_storage(Arc::new(storage)))
    }
}

#[derive(CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq)]
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
    /// (B x M x K) * (B x K x N) = (B x M x N)
    /// out = out * alpha + beta * lhs * rhs
    MatMul {
        l_id: GraphTensorId,
        r_id: GraphTensorId,
        o_id: Option<GraphTensorId>,
        k: usize,
        alpha: T,
        beta: T,
    },
    /// Fill with uniform random values in [0, 1).
    Rand,
    /// Fill with normally distributed random values (mean, std).
    Randn {
        mean: T,
        std: T,
    },
    // Permutation operator.
    Permute {
        v_id: GraphTensorId,
    },
    NoOp,
}

#[derive(Clone, Debug)]
/// Graph tensor IDs can be cloned.
pub enum GraphTensorId {
    OutOfPlace(Arc<AtomicUsize>),
    InPlace(Arc<AtomicUsize>),
}

impl Hash for GraphTensorId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(self.get());
    }
}

impl GraphTensorId {
    pub fn out_of_place(value: usize) -> Self {
        Self::OutOfPlace(Arc::new(AtomicUsize::new(value)))
    }

    pub fn inplace(value: usize) -> Self {
        Self::InPlace(Arc::new(AtomicUsize::new(value)))
    }

    pub fn to_inplace(&self) -> Self {
        match self {
            Self::OutOfPlace(x) | Self::InPlace(x) => Self::inplace(x.load(Ordering::SeqCst)),
        }
    }

    pub fn to_inplace_if(&self, predicate: bool) -> Self {
        match self {
            Self::OutOfPlace(x) | Self::InPlace(x) if predicate => {
                Self::inplace(x.load(Ordering::SeqCst))
            }
            _ => self.clone(),
        }
    }

    pub fn get(&self) -> usize {
        match self {
            GraphTensorId::InPlace(x) | GraphTensorId::OutOfPlace(x) => x.load(Ordering::SeqCst),
        }
    }

    pub fn set(&self, value: usize) {
        match self {
            GraphTensorId::InPlace(x) | GraphTensorId::OutOfPlace(x) => {
                x.store(value, Ordering::SeqCst);
            }
        }
    }

    pub fn is_inplace(&self) -> bool {
        matches!(self, Self::InPlace(_))
    }
}

// Manually implement equality by comparing the numeric IDs and in‐place flag:
impl PartialEq for GraphTensorId {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get() && self.is_inplace() == other.is_inplace()
    }
}

impl Eq for GraphTensorId {}
