use petgraph::algo::toposort;
use petgraph::graphmap::DiGraphMap;
use std::borrow::Cow;
use std::cell::RefCell;
use std::rc::Rc;

use pool::{BufferPool, PooledBuffer};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{
    storage::{BackendDevice, BackendStorage},
    DType, GraphNode, Op, Result,
};

mod pool;

pub struct CpuDevice;

#[derive(Clone)]
pub struct CpuStorage<T: DType>(pub(crate) Vec<T>);

impl<T: DType> BackendStorage<T> for CpuStorage<T> {
    fn to_cpu_storage(&self) -> Result<Cow<CpuStorage<T>>> {
        // Note: copying all data here.
        Ok(Cow::Owned(self.clone()))
    }
}

impl BackendDevice for CpuDevice {
    type Storage<X: DType> = CpuStorage<X>;

    fn compile_and_run_graph<T: DType>(&self, graph: &[GraphNode<T>]) -> Result<Self::Storage<T>> {
        {
            // Create a shared buffer pool
            let pool = Rc::new(RefCell::new(BufferPool::<T>::new()));

            // Build a dependency graph of tensor indices
            let mut dep_graph = DiGraphMap::<usize, ()>::new();
            for idx in 0..graph.len() {
                dep_graph.add_node(idx);
            }

            for (idx, node) in graph.iter().enumerate() {
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
                    Op::MatMul { l_id, r_id, .. } => {
                        dep_graph.add_edge(l_id.get(), idx, ());
                        dep_graph.add_edge(r_id.get(), idx, ());
                    }
                    // NoOp and Fill/Arange don’t create incoming edges
                    Op::NoOp | Op::Fill { .. } | Op::Arange { .. } => {}
                }
            }

            // Compute topological order
            let order = toposort(&dep_graph, None).expect("Cycle detected in graph!");

            // Prepare storage for intermediate results
            let mut results: Vec<Option<PooledBuffer<T>>> = Vec::with_capacity(graph.len());
            results.resize_with(graph.len(), || None);

            // Evaluate nodes in topological order
            for idx in order {
                let op = &graph[idx];

                let out_shape = &op.shape;
                let out_elem_count: usize = out_shape.iter().product();

                let computed = match &op.op {
                    Op::BinaryOp {
                        l_id,
                        r_id,
                        operator,
                    } => {
                        if l_id.is_inplace() {
                            let mut l_buf = results[l_id.get()].take().unwrap();
                            let r_buf = results[r_id.get()].as_ref().unwrap();
                            T::binary_simd_op_inplace_lhs(&mut l_buf, r_buf, *operator);
                            l_buf
                        } else if r_id.is_inplace() {
                            let mut r_buf = results[r_id.get()].take().unwrap();
                            let l_buf = results[l_id.get()].as_ref().unwrap();
                            T::binary_simd_op_inplace_rhs(l_buf, &mut r_buf, *operator);
                            r_buf
                        } else {
                            let l_buf = results[l_id.get()].as_ref().unwrap();
                            let r_buf = results[r_id.get()].as_ref().unwrap();
                            let mut out = pool.borrow_mut().get_buffer(out_elem_count);
                            T::binary_simd_op(l_buf, r_buf, &mut out, *operator);
                            PooledBuffer::new(out, pool.clone())
                        }
                    }
                    Op::Fill { v } => {
                        let mut buf = pool.borrow_mut().get_empty_buffer(out_elem_count);
                        buf.extend(std::iter::repeat_n(*v, out_elem_count));
                        PooledBuffer::new(buf, pool.clone())
                    }
                    Op::Arange { start, step, stop } => {
                        let mut buf = pool.borrow_mut().get_empty_buffer(out_elem_count);
                        let mut x = start.to_f64();
                        while x < stop.to_f64() {
                            buf.push(T::from_f64(x));
                            x += step.to_f64();
                        }
                        PooledBuffer::new(buf, pool.clone())
                    }
                    Op::UnaryOp { v_id, operator } => {
                        let buf = results[v_id.get()].as_ref().unwrap();
                        let op_fn = operator.to_closure();
                        let mut out = pool.borrow_mut().get_buffer(out_elem_count);
                        out.par_iter_mut()
                            .zip(&**buf)
                            .for_each(|(out, x): (&mut T, &T)| *out = op_fn(*x));
                        PooledBuffer::new(out, pool.clone())
                    }
                    Op::FusedMulAdd { a_id, b_id, c_id } => {
                        if a_id.is_inplace() {
                            let mut a_buf = results[a_id.get()].take().unwrap();
                            let b_buf = results[b_id.get()].as_ref().unwrap();
                            let c_buf = results[c_id.get()].as_ref().unwrap();

                            T::fma_op_inplace_a(&mut a_buf, b_buf, c_buf);
                            a_buf
                        } else if b_id.is_inplace() {
                            let mut b_buf = results[b_id.get()].take().unwrap();
                            let a_buf = results[a_id.get()].as_ref().unwrap();
                            let c_buf = results[c_id.get()].as_ref().unwrap();

                            T::fma_op_inplace_b(a_buf, &mut b_buf, c_buf);
                            b_buf
                        } else if c_id.is_inplace() {
                            let mut c_buf = results[c_id.get()].take().unwrap();
                            let a_buf = results[a_id.get()].as_ref().unwrap();
                            let b_buf = results[b_id.get()].as_ref().unwrap();

                            T::fma_op_inplace_c(a_buf, b_buf, &mut c_buf);
                            c_buf
                        } else {
                            let a_buf = results[a_id.get()].as_ref().unwrap();
                            let b_buf = results[b_id.get()].as_ref().unwrap();
                            let c_buf = results[c_id.get()].as_ref().unwrap();

                            let mut out = pool.borrow_mut().get_buffer(out_elem_count);
                            T::fma_op(a_buf, b_buf, c_buf, &mut out);
                            PooledBuffer::new(out, pool.clone())
                        }
                    }
                    // Matrix multiplication: multiply two 2D tensors A (m x k) and B (k x n)
                    Op::MatMul {
                        l_id,
                        r_id,
                        o_id,
                        k,
                        alpha,
                        beta,
                    } => {
                        let a_buf = results[l_id.get()].as_ref().unwrap();
                        let b_buf = results[r_id.get()].as_ref().unwrap();
                        // Determine output dimensions from shape S (must be 2D)
                        let shape = out_shape;
                        assert!(shape.len() == 3);
                        let b = shape[0];
                        let m = shape[1];
                        let n = shape[2];

                        let mut out = if let Some(o_id) = o_id {
                            let o_buf = results[o_id.get()].as_ref().unwrap();
                            (*o_buf).clone()
                        } else {
                            pool.borrow_mut().get_buffer(m * n)
                        };

                        T::launch_gemm(a_buf, b_buf, b, m, n, *k, &mut out, *alpha, *beta);

                        PooledBuffer::new(out, pool.clone())
                    }
                    Op::NoOp => unreachable!("NoOp should not be evaluated."),
                };
                results[idx] = Some(computed);
            }

            // Extract final result
            let final_idx = graph.len() - 1;
            let output = results[final_idx].take().unwrap().into_inner();
            Ok(CpuStorage(output))
        }
    }
}
