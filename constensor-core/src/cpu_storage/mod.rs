use std::borrow::Cow;

use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{
    graph::{BinaryOpType, GraphTensorId},
    storage::{BackendDevice, BackendStorage},
    DType, Op, Result, Shape, SignedDType,
};

pub struct CpuDevice;

#[derive(Clone)]
pub struct CpuStorage<T: DType>(pub(crate) Vec<T>);

impl<T: DType> BackendStorage<T> for CpuStorage<T> {
    fn to_cpu_storage(&self) -> Result<std::borrow::Cow<CpuStorage<T>>> {
        // Note: copying all data here.
        Ok(Cow::Owned(self.clone()))
    }
}

fn evaluate_node<T: DType, S: Shape>(op: &Op<T>, graph: &[Op<T>]) -> Vec<T> {
    match op {
        Op::BinaryOp {
            l_id,
            r_id,
            operator,
        } => {
            let mut l =
                evaluate_node::<T, S>(&graph[<&GraphTensorId as Into<usize>>::into(l_id)], graph);
            let r =
                evaluate_node::<T, S>(&graph[<&GraphTensorId as Into<usize>>::into(r_id)], graph);
            T::binary_simd_op(&mut l, r, *operator);
            l
        }
        Op::Fill { v } => {
            vec![*v; S::element_count()]
        }
        Op::Arange { start, step } => {
            let mut accum = Vec::with_capacity(S::element_count());
            for i in 0..S::element_count() {
                accum.push(T::offset(i, *start, *step));
            }
            accum
        }
        Op::UnaryOp {
            v_id: _,
            operator: _,
        } => {
            unreachable!()
        }
        Op::FusedMulAdd { a_id, b_id, c_id } => {
            let mut a =
                evaluate_node::<T, S>(&graph[<&GraphTensorId as Into<usize>>::into(a_id)], graph);
            let b =
                evaluate_node::<T, S>(&graph[<&GraphTensorId as Into<usize>>::into(b_id)], graph);
            let c =
                evaluate_node::<T, S>(&graph[<&GraphTensorId as Into<usize>>::into(c_id)], graph);
            let mul_op = BinaryOpType::Mul.as_closure::<T>();
            let add_op = BinaryOpType::Add.as_closure::<T>();
            a.par_iter_mut()
                .zip(b)
                .zip(c)
                .for_each(|((a, b), c)| *a = add_op(mul_op(*a, b), c));
            a
        }
        Op::NoOp => unreachable!("no-op ops should never be reached."),
    }
}

fn evaluate_node_signed<T: DType + SignedDType, S: Shape>(
    op: &Op<T>,
    graph: &[Op<T>],
) -> Result<Vec<T>> {
    match op {
        Op::UnaryOp { v_id, operator } => {
            let mut v = evaluate_node_signed::<T, S>(
                &graph[<&GraphTensorId as Into<usize>>::into(v_id)],
                graph,
            )?;
            let op = operator.to_closure();
            v.par_iter_mut().for_each(|x| *x = op(*x));
            Ok(v)
        }
        Op::BinaryOp {
            l_id,
            r_id,
            operator,
        } => {
            let mut l = evaluate_node_signed::<T, S>(
                &graph[<&GraphTensorId as Into<usize>>::into(l_id)],
                graph,
            )?;
            let r = evaluate_node_signed::<T, S>(
                &graph[<&GraphTensorId as Into<usize>>::into(r_id)],
                graph,
            )?;
            T::binary_simd_op(&mut l, r, *operator);
            Ok(l)
        }
        Op::FusedMulAdd { a_id, b_id, c_id } => {
            let mut a = evaluate_node_signed::<T, S>(
                &graph[<&GraphTensorId as Into<usize>>::into(a_id)],
                graph,
            )?;
            let b = evaluate_node_signed::<T, S>(
                &graph[<&GraphTensorId as Into<usize>>::into(b_id)],
                graph,
            )?;
            let c = evaluate_node_signed::<T, S>(
                &graph[<&GraphTensorId as Into<usize>>::into(c_id)],
                graph,
            )?;
            let mul_op = BinaryOpType::Mul.as_closure::<T>();
            let add_op = BinaryOpType::Add.as_closure::<T>();
            a.par_iter_mut()
                .zip(b)
                .zip(c)
                .for_each(|((a, b), c)| *a = add_op(mul_op(*a, b), c));
            Ok(a)
        }
        other => Ok(evaluate_node::<T, S>(other, graph)),
    }
}

impl BackendDevice for CpuDevice {
    type Storage<X: DType> = CpuStorage<X>;

    fn compile_and_run_graph<S: Shape, T: DType>(
        &self,
        graph: &[Op<T>],
    ) -> Result<Self::Storage<T>> {
        Ok(CpuStorage(evaluate_node::<T, S>(
            graph.last().unwrap(),
            graph,
        )))
    }

    fn compile_and_run_graph_signed<S: Shape, T: DType + SignedDType>(
        &self,
        graph: &[Op<T>],
    ) -> Result<Self::Storage<T>> {
        Ok(CpuStorage(evaluate_node_signed::<T, S>(
            graph.last().unwrap(),
            graph,
        )?))
    }
}
