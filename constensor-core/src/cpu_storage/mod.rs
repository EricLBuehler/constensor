use petgraph::algo::toposort;
use petgraph::graphmap::DiGraphMap;
use std::borrow::Cow;
use std::cell::RefCell;
use std::rc::Rc;

use pool::{BufferPool, PooledBuffer};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{
    graph::GraphTensorId,
    storage::{BackendDevice, BackendStorage},
    DType, GraphNode, Op, Result,
};
// (matmul implemented directly; gemm crate no longer used)
// use gemm::{gemm, Parallelism};

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
                    Op::BinaryOp { l_id, r_id, .. } | Op::InplaceBinaryOp { l_id, r_id, .. } => {
                        let l_idx = <&GraphTensorId as Into<usize>>::into(l_id);
                        let r_idx = <&GraphTensorId as Into<usize>>::into(r_id);
                        dep_graph.add_edge(l_idx, idx, ());
                        dep_graph.add_edge(r_idx, idx, ());
                    }
                    Op::UnaryOp { v_id, .. } => {
                        let v_idx = <&GraphTensorId as Into<usize>>::into(v_id);
                        dep_graph.add_edge(v_idx, idx, ());
                    }
                    Op::FusedMulAdd { a_id, b_id, c_id }
                    | Op::InplaceFusedMulAdd {
                        a_id, b_id, c_id, ..
                    } => {
                        let a_idx = <&GraphTensorId as Into<usize>>::into(a_id);
                        let b_idx = <&GraphTensorId as Into<usize>>::into(b_id);
                        let c_idx = <&GraphTensorId as Into<usize>>::into(c_id);
                        dep_graph.add_edge(a_idx, idx, ());
                        dep_graph.add_edge(b_idx, idx, ());
                        dep_graph.add_edge(c_idx, idx, ());
                    }
                    Op::MatMul { l_id, r_id, .. } => {
                        let l_idx = <&GraphTensorId as Into<usize>>::into(l_id);
                        let r_idx = <&GraphTensorId as Into<usize>>::into(r_id);
                        dep_graph.add_edge(l_idx, idx, ());
                        dep_graph.add_edge(r_idx, idx, ());
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
                        let l_idx = <&GraphTensorId as Into<usize>>::into(l_id);
                        let r_idx = <&GraphTensorId as Into<usize>>::into(r_id);
                        let l_buf = results[l_idx].as_ref().unwrap();
                        let r_buf = results[r_idx].as_ref().unwrap();
                        let mut out = pool.borrow_mut().get_buffer(out_elem_count);
                        T::binary_simd_op(l_buf, r_buf, &mut out, *operator);
                        PooledBuffer::new(out, pool.clone())
                    }
                    Op::InplaceBinaryOp {
                        out,
                        l_id,
                        r_id,
                        operator,
                    } => {
                        let l_idx = <&GraphTensorId as Into<usize>>::into(l_id);
                        let r_idx = <&GraphTensorId as Into<usize>>::into(r_id);
                        let o_idx = <&GraphTensorId as Into<usize>>::into(out);
                        if o_idx == l_idx {
                            let mut l_buf = results[l_idx].take().unwrap();
                            let r_buf = results[r_idx].as_ref().unwrap();
                            T::binary_simd_op_inplace_lhs(&mut l_buf, r_buf, *operator);
                            l_buf
                        } else if o_idx == r_idx {
                            let mut r_buf = results[r_idx].take().unwrap();
                            let l_buf = results[l_idx].as_ref().unwrap();
                            T::binary_simd_op_inplace_rhs(l_buf, &mut r_buf, *operator);
                            r_buf
                        } else {
                            unreachable!()
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
                        let v_idx = <&GraphTensorId as Into<usize>>::into(v_id);
                        let buf = results[v_idx].as_ref().unwrap();
                        let op_fn = operator.to_closure();
                        let mut out = pool.borrow_mut().get_buffer(out_elem_count);
                        out.par_iter_mut()
                            .zip(&**buf)
                            .for_each(|(out, x): (&mut T, &T)| *out = op_fn(*x));
                        PooledBuffer::new(out, pool.clone())
                    }
                    Op::FusedMulAdd { a_id, b_id, c_id } => {
                        let a_idx = <&GraphTensorId as Into<usize>>::into(a_id);
                        let b_idx = <&GraphTensorId as Into<usize>>::into(b_id);
                        let c_idx = <&GraphTensorId as Into<usize>>::into(c_id);
                        let a_buf = results[a_idx].as_ref().unwrap();
                        let b_buf = results[b_idx].as_ref().unwrap();
                        let c_buf = results[c_idx].as_ref().unwrap();

                        let mut out = pool.borrow_mut().get_buffer(out_elem_count);
                        T::fma_op(a_buf, b_buf, c_buf, &mut out);
                        PooledBuffer::new(out, pool.clone())
                    }
                    Op::InplaceFusedMulAdd {
                        a_id,
                        b_id,
                        c_id,
                        out,
                    } => {
                        let a_idx = <&GraphTensorId as Into<usize>>::into(a_id);
                        let b_idx = <&GraphTensorId as Into<usize>>::into(b_id);
                        let c_idx = <&GraphTensorId as Into<usize>>::into(c_id);
                        let o_idx = <&GraphTensorId as Into<usize>>::into(out);

                        if o_idx == a_idx {
                            let mut a_buf = results[a_idx].take().unwrap();
                            let b_buf = results[b_idx].as_ref().unwrap();
                            let c_buf = results[c_idx].as_ref().unwrap();

                            T::fma_op_inplace_a(&mut a_buf, b_buf, c_buf);
                            a_buf
                        } else if o_idx == b_idx {
                            let mut b_buf = results[b_idx].take().unwrap();
                            let a_buf = results[a_idx].as_ref().unwrap();
                            let c_buf = results[c_idx].as_ref().unwrap();

                            T::fma_op_inplace_b(a_buf, &mut b_buf, c_buf);
                            b_buf
                        } else if o_idx == c_idx {
                            let mut c_buf = results[c_idx].take().unwrap();
                            let a_buf = results[a_idx].as_ref().unwrap();
                            let b_buf = results[b_idx].as_ref().unwrap();

                            T::fma_op_inplace_c(a_buf, b_buf, &mut c_buf);
                            c_buf
                        } else {
                            unreachable!()
                        }
                    }
                    // Matrix multiplication: multiply two 2D tensors A (m x k) and B (k x n)
                    Op::MatMul { l_id, r_id, k } => {
                        let l_idx = <&GraphTensorId as Into<usize>>::into(l_id);
                        let r_idx = <&GraphTensorId as Into<usize>>::into(r_id);
                        let a_buf = results[l_idx].as_ref().unwrap();
                        let b_buf = results[r_idx].as_ref().unwrap();
                        // Determine output dimensions from shape S (must be 2D)
                        let shape = out_shape;
                        assert!(shape.len() == 2, "MatMul requires 2D output shape");
                        let m = shape[0];
                        let n = shape[1];
                        let k_dim = k;
                        // Allocate output buffer and compute C = A * B via direct multiplication
                        let k_dim = *k_dim;
                        let mut out = pool.borrow_mut().get_buffer(m * n);
                        // Debug: print dimensions and input buffers
                        println!(
                            "[CPU MatMul] m={}, n={}, k_dim={}, a_buf={:?}, b_buf={:?}",
                            m,
                            n,
                            k_dim,
                            &a_buf[..],
                            &b_buf[..],
                        );
                        // Prepare full input buffers for A (m x k_dim) and B (k_dim x n)
                        let a_data = &a_buf[..];
                        let b_data = &b_buf[..];
                        // Compute matrix multiplication: out[i,j] = sum_p a_data[i,k] * b_data[k,j]
                        for i in 0..m {
                            for j in 0..n {
                                let mut sum = T::ZERO;
                                for p in 0..k_dim {
                                    sum = sum + a_data[i * k_dim + p] * b_data[p * n + j];
                                }
                                out[i * n + j] = sum;
                            }
                        }
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
