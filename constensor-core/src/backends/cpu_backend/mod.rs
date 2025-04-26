use petgraph::algo::toposort;
use petgraph::graphmap::DiGraphMap;
use std::{borrow::Cow, marker::PhantomData};

use pool::{BufferPool, PooledBuffer};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::device::Dev;
use crate::storage::Storage;
use crate::tensor::contiguous_strides;
use crate::Shape;
use crate::{
    storage::{BackendDevice, BackendStorage},
    CompiledGraph, DType, GraphNode, Op, Result,
};
use rand::Rng;
use rand_distr::{Distribution, Normal};

mod pool;
// Concurrency primitives for dynamic DAG scheduler
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex, RwLock};

pub struct CpuDevice;

#[derive(Clone)]
pub struct CpuStorage<T: DType>(pub(crate) Vec<T>);

impl<T: DType> BackendStorage<T> for CpuStorage<T> {
    fn to_cpu_storage(&self) -> Result<Cow<CpuStorage<T>>> {
        Ok(Cow::Borrowed(self))
    }
    fn cast<U: DType>(&self) -> Result<Storage<U>> {
        let new = self.0.iter().map(|x| U::from_f64(x.to_f64()));
        Ok(Storage::Cpu(CpuStorage(new.collect())))
    }
}

impl BackendDevice for CpuDevice {
    type Storage<X: DType> = CpuStorage<X>;

    fn compile<S: Shape, T: DType, D: Dev>(
        &self,
        graph: Vec<GraphNode<T>>,
    ) -> Result<CompiledGraph<S, T, D>> {
        // Build a dependency graph of tensor indices
        let mut dep_graph = DiGraphMap::<usize, ()>::new();
        for id in 0..graph.len() {
            dep_graph.add_node(id);
        }

        for node in graph.iter() {
            let idx = node.id.get();
            match &node.op {
                Op::BinaryOp { l_id, r_id, .. } => {
                    dep_graph.add_edge(l_id.get(), idx, ());
                    dep_graph.add_edge(r_id.get(), idx, ());
                }
                Op::UnaryOp { v_id, .. } => {
                    dep_graph.add_edge(v_id.get(), idx, ());
                }
                Op::FusedMulAdd { a_id, b_id, c_id } => {
                    dep_graph.add_edge(a_id.get(), idx, ());
                    dep_graph.add_edge(b_id.get(), idx, ());
                    dep_graph.add_edge(c_id.get(), idx, ());
                }
                Op::MatMul {
                    l_id, r_id, o_id, ..
                } => {
                    dep_graph.add_edge(l_id.get(), idx, ());
                    dep_graph.add_edge(r_id.get(), idx, ());
                    if let Some(o_id) = o_id {
                        dep_graph.add_edge(o_id.get(), idx, ());
                    }
                }
                Op::Permute { v_id } => {
                    dep_graph.add_edge(v_id.get(), idx, ());
                }
                // NoOp, Fill/Arange, Rand/Randn don’t create incoming edges
                Op::NoOp | Op::Fill { .. } | Op::Arange { .. } | Op::Rand | Op::Randn { .. } => {}
            }
        }

        // Compute topological order
        let order = toposort(&dep_graph, None).expect("Cycle detected in graph!");

        Ok(CompiledGraph::Cpu {
            order,
            graph,
            ghost: PhantomData,
        })
    }

    fn run_graph<S: Shape, T: DType + Send + Sync + 'static, D: Dev>(
        &self,
        graph: &CompiledGraph<S, T, D>,
    ) -> Result<Self::Storage<T>> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{mpsc, Arc, Mutex, RwLock};

        // Thread-safe buffer pool
        let pool: Arc<Mutex<BufferPool<T>>> = Arc::new(Mutex::new(BufferPool::new()));

        // Extract the compiled node list
        #[allow(irrefutable_let_patterns)]
        let CompiledGraph::Cpu {
            graph: node_graph, ..
        } = graph
        else {
            unreachable!("Expected CPU compiled graph");
        };
        // Clone into an Arc for sharing
        let node_graph = Arc::new(node_graph.clone());
        let n = node_graph.len();

        // Prepare slots for results and strides
        let results: Arc<Vec<RwLock<Option<PooledBuffer<T>>>>> =
            Arc::new((0..n).map(|_| RwLock::new(None)).collect());
        let results_strides: Arc<Vec<RwLock<Option<Vec<usize>>>>> =
            Arc::new((0..n).map(|_| RwLock::new(None)).collect());

        // Build adjacency: children lists and indegree counts
        let mut children = vec![Vec::new(); n];
        let indegree_vec = (0..n).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>();
        for node in node_graph.iter() {
            let dst = node.id.get();
            match &node.op {
                Op::BinaryOp { l_id, r_id, .. } => {
                    let p1 = l_id.get();
                    let p2 = r_id.get();
                    children[p1].push(dst);
                    children[p2].push(dst);
                    indegree_vec[dst].fetch_add(2, Ordering::SeqCst);
                }
                Op::UnaryOp { v_id, .. } => {
                    let p = v_id.get();
                    children[p].push(dst);
                    indegree_vec[dst].fetch_add(1, Ordering::SeqCst);
                }
                Op::FusedMulAdd { a_id, b_id, c_id } => {
                    for &p in &[a_id.get(), b_id.get(), c_id.get()] {
                        children[p].push(dst);
                        indegree_vec[dst].fetch_add(1, Ordering::SeqCst);
                    }
                }
                Op::MatMul {
                    l_id, r_id, o_id, ..
                } => {
                    let p1 = l_id.get();
                    let p2 = r_id.get();
                    children[p1].push(dst);
                    children[p2].push(dst);
                    indegree_vec[dst].fetch_add(2, Ordering::SeqCst);
                    if let Some(o) = o_id {
                        let p3 = o.get();
                        children[p3].push(dst);
                        indegree_vec[dst].fetch_add(1, Ordering::SeqCst);
                    }
                }
                Op::Permute { v_id } => {
                    let p = v_id.get();
                    children[p].push(dst);
                    indegree_vec[dst].fetch_add(1, Ordering::SeqCst);
                }
                _ => {}
            }
        }
        let indegree = Arc::new(indegree_vec);
        let children = Arc::new(children);

        // Channel to signal when the final node completes
        let final_idx = n - 1;
        let (tx, rx) = mpsc::channel();

        // Spawn initial tasks for nodes with zero indegree
        for idx in 0..n {
            if indegree[idx].load(Ordering::SeqCst) == 0 {
                let pool = pool.clone();
                let node_graph = node_graph.clone();
                let results = results.clone();
                let results_strides = results_strides.clone();
                let indegree = indegree.clone();
                let children = children.clone();
                let tx = tx.clone();
                rayon::spawn(move || {
                    eval_node(
                        idx,
                        &node_graph,
                        &pool,
                        &results,
                        &results_strides,
                        &indegree,
                        &children,
                        final_idx,
                        tx,
                    );
                });
            }
        }
        // Drop the extra sender in main thread
        drop(tx);

        // Wait for the final node to complete
        rx.recv()
            .expect("Failed to receive completion of final node");

        // Extract and return the final result
        let mut final_lock = results[final_idx].write().unwrap();
        let pooled = final_lock.take().expect("Final result missing");
        let output = pooled.into_inner();
        Ok(CpuStorage(output))
    }
}

/// Recursively evaluate a node, scheduling its children when their dependencies are ready.
#[allow(clippy::too_many_arguments)]
fn eval_node<T: DType + Send + Sync + 'static>(
    idx: usize,
    node_graph: &Arc<Vec<GraphNode<T>>>,
    pool: &Arc<Mutex<BufferPool<T>>>,
    results: &Arc<Vec<RwLock<Option<PooledBuffer<T>>>>>,
    results_strides: &Arc<Vec<RwLock<Option<Vec<usize>>>>>,
    indegree: &Arc<Vec<AtomicUsize>>,
    children: &Arc<Vec<Vec<usize>>>,
    final_idx: usize,
    tx: mpsc::Sender<()>,
) {
    // Prepare RNG for random ops
    let mut rng = rand::rng();
    let node = &node_graph[idx];
    let out_shape = &node.shape;
    let out_elem_count: usize = out_shape.iter().product();

    // Compute this node's buffer
    let computed: PooledBuffer<T> = match &node.op {
        Op::Fill { v } => {
            let mut buf = pool.lock().unwrap().get_empty_buffer(out_elem_count);
            buf.extend(std::iter::repeat_n(*v, out_elem_count));
            PooledBuffer::new(buf, pool.clone())
        }
        Op::Arange { start, step, stop } => {
            let mut buf = pool.lock().unwrap().get_empty_buffer(out_elem_count);
            let mut x = start.to_f64();
            while x < stop.to_f64() {
                buf.push(T::from_f64(x));
                x += step.to_f64();
            }
            PooledBuffer::new(buf, pool.clone())
        }
        Op::Rand => {
            let mut buf = pool.lock().unwrap().get_buffer(out_elem_count);
            for elt in &mut buf {
                *elt = T::from_f64(rng.random());
            }
            PooledBuffer::new(buf, pool.clone())
        }
        Op::Randn { mean, std } => {
            let mean_f = mean.to_f64();
            let std_f = std.to_f64();
            let normal = Normal::new(mean_f, std_f).unwrap();
            let mut buf = pool.lock().unwrap().get_buffer(out_elem_count);
            for elt in &mut buf {
                *elt = T::from_f64(normal.sample(&mut rng));
            }
            PooledBuffer::new(buf, pool.clone())
        }
        Op::UnaryOp { v_id, operator } => {
            let src_guard = results[v_id.get()].read().unwrap();
            let src = src_guard.as_ref().unwrap();
            let op_fn = operator.to_closure();
            let mut out = pool.lock().unwrap().get_buffer(out_elem_count);
            out.par_iter_mut()
                .zip(&**src)
                .for_each(|(o, x)| *o = op_fn(*x));
            PooledBuffer::new(out, pool.clone())
        }
        Op::BinaryOp {
            l_id,
            r_id,
            operator,
        } => {
            if l_id.is_inplace() {
                let mut left = results[l_id.get()].write().unwrap().take().unwrap();
                let right_guard = results[r_id.get()].read().unwrap();
                let right = right_guard.as_ref().unwrap();
                T::binary_simd_op_inplace_lhs(&mut left, right, *operator);
                left
            } else if r_id.is_inplace() {
                let mut right = results[r_id.get()].write().unwrap().take().unwrap();
                let left_guard = results[l_id.get()].read().unwrap();
                let left = left_guard.as_ref().unwrap();
                T::binary_simd_op_inplace_rhs(left, &mut right, *operator);
                right
            } else {
                let left_guard = results[l_id.get()].read().unwrap();
                let left = left_guard.as_ref().unwrap();
                let right_guard = results[r_id.get()].read().unwrap();
                let right = right_guard.as_ref().unwrap();
                let mut out = pool.lock().unwrap().get_buffer(out_elem_count);
                T::binary_simd_op(left, right, &mut out, *operator);
                PooledBuffer::new(out, pool.clone())
            }
        }
        Op::FusedMulAdd { a_id, b_id, c_id } => {
            if a_id.is_inplace() {
                let mut a_buf = results[a_id.get()].write().unwrap().take().unwrap();
                let b_guard = results[b_id.get()].read().unwrap();
                let b_buf = b_guard.as_ref().unwrap();
                let c_guard = results[c_id.get()].read().unwrap();
                let c_buf = c_guard.as_ref().unwrap();
                T::fma_op_inplace_a(&mut a_buf, b_buf, c_buf);
                a_buf
            } else if b_id.is_inplace() {
                let mut b_buf = results[b_id.get()].write().unwrap().take().unwrap();
                let a_guard = results[a_id.get()].read().unwrap();
                let a_buf = a_guard.as_ref().unwrap();
                let c_guard = results[c_id.get()].read().unwrap();
                let c_buf = c_guard.as_ref().unwrap();
                T::fma_op_inplace_b(a_buf, &mut b_buf, c_buf);
                b_buf
            } else if c_id.is_inplace() {
                let mut c_buf = results[c_id.get()].write().unwrap().take().unwrap();
                let a_guard = results[a_id.get()].read().unwrap();
                let a_buf = a_guard.as_ref().unwrap();
                let b_guard = results[b_id.get()].read().unwrap();
                let b_buf = b_guard.as_ref().unwrap();
                T::fma_op_inplace_c(a_buf, b_buf, &mut c_buf);
                c_buf
            } else {
                let a_guard = results[a_id.get()].read().unwrap();
                let a_buf = a_guard.as_ref().unwrap();
                let b_guard = results[b_id.get()].read().unwrap();
                let b_buf = b_guard.as_ref().unwrap();
                let c_guard = results[c_id.get()].read().unwrap();
                let c_buf = c_guard.as_ref().unwrap();
                let mut out = pool.lock().unwrap().get_buffer(out_elem_count);
                T::fma_op(a_buf, b_buf, c_buf, &mut out);
                PooledBuffer::new(out, pool.clone())
            }
        }
        Op::MatMul {
            l_id,
            r_id,
            o_id,
            k,
            alpha,
            beta,
        } => {
            let shape = &node.shape;
            let b = shape[0];
            let m = shape[1];
            let n = shape[2];
            let (mut out_buf, out_stride) = if let Some(o) = o_id {
                if o.is_inplace() {
                    let buf = results[o.get()].write().unwrap().take().unwrap();
                    let st = results_strides[o.get()]
                        .read()
                        .unwrap()
                        .as_ref()
                        .unwrap()
                        .clone();
                    (buf, st)
                } else {
                    let buf_guard = results[o.get()].read().unwrap();
                    let buf_clone = buf_guard.as_ref().unwrap();
                    let st_guard = results_strides[o.get()].read().unwrap();
                    let st = st_guard.as_ref().unwrap().clone();
                    (PooledBuffer::new((*buf_clone).clone(), pool.clone()), st)
                }
            } else {
                let st = contiguous_strides(&[b, m, n]);
                let buf = pool.lock().unwrap().get_buffer(b * m * n);
                (PooledBuffer::new(buf, pool.clone()), st)
            };
            let a_guard = results[l_id.get()].read().unwrap();
            let a_buf = a_guard.as_ref().unwrap();
            let b_guard = results[r_id.get()].read().unwrap();
            let b_buf = b_guard.as_ref().unwrap();
            let a_str_guard = results_strides[l_id.get()].read().unwrap();
            let a_str = a_str_guard.as_ref().unwrap();
            let b_str_guard = results_strides[r_id.get()].read().unwrap();
            let b_str = b_str_guard.as_ref().unwrap();
            T::launch_gemm(
                a_buf,
                a_str,
                b_buf,
                b_str,
                b,
                m,
                n,
                *k,
                &mut out_buf,
                &out_stride,
                *alpha,
                *beta,
            );
            out_buf
        }
        Op::Permute { v_id } => {
            if v_id.is_inplace() {
                results[v_id.get()].write().unwrap().take().unwrap()
            } else {
                let buf_guard = results[v_id.get()].read().unwrap();
                let buf = buf_guard.as_ref().unwrap();
                PooledBuffer::new((*buf).clone(), pool.clone())
            }
        }
        Op::NoOp => panic!("NoOp should not be evaluated"),
    };
    // store result and strides
    *results[idx].write().unwrap() = Some(computed);
    *results_strides[idx].write().unwrap() = Some(node.strides.clone());
    // signal final
    if idx == final_idx {
        let _ = tx.send(());
    }
    // schedule children
    for &child in &children[idx] {
        if indegree[child].fetch_sub(1, Ordering::SeqCst) == 1 {
            let pool2 = pool.clone();
            let ng2 = node_graph.clone();
            let res2 = results.clone();
            let rs2 = results_strides.clone();
            let indeg2 = indegree.clone();
            let ch2 = children.clone();
            let tx2 = tx.clone();
            rayon::spawn(move || {
                eval_node(
                    child, &ng2, &pool2, &res2, &rs2, &indeg2, &ch2, final_idx, tx2,
                );
            });
        }
    }
}
