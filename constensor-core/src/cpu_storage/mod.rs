use std::borrow::Cow;

use crate::{
    storage::{BackendDevice, BackendStorage},
    DType, Error, Op, Result, Shape, SignedDType,
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

fn evaluate_node_unsigned<T: DType, S: Shape>(op: &Op<T>, graph: &[Op<T>]) -> Vec<T> {
    match op {
        Op::BinaryOp {
            l_id,
            r_id,
            operator,
        } => {
            let l_name = evaluate_node_unsigned::<T, S>(&graph[**l_id], graph);
            let r_name = evaluate_node_unsigned::<T, S>(&graph[**r_id], graph);
            let mut out = vec![T::ZERO; l_name.len()];
            let op = operator.to_closure();
            for (i, (x, y)) in l_name.iter().zip(r_name).enumerate() {
                out[i] = op(*x, y);
            }
            out
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
    }
}

fn evaluate_node<T: DType + SignedDType, S: Shape>(op: &Op<T>, graph: &[Op<T>]) -> Result<Vec<T>> {
    match op {
        Op::UnaryOp { v_id, operator } => {
            let v_name = evaluate_node::<T, S>(&graph[**v_id], graph)?;
            let mut out = vec![T::ZERO; v_name.len()];
            let op = operator.to_closure();
            for (i, x) in v_name.iter().enumerate() {
                out[i] = op(*x).ok_or(Error::Msg("`sqrt` value was negative".to_string()))?;
            }
            Ok(out)
        }
        Op::BinaryOp {
            l_id,
            r_id,
            operator,
        } => {
            let l_name = evaluate_node::<T, S>(&graph[**l_id], graph)?;
            let r_name = evaluate_node::<T, S>(&graph[**r_id], graph)?;
            let mut out = vec![T::ZERO; l_name.len()];
            let op = operator.to_closure();
            for (i, (x, y)) in l_name.iter().zip(r_name).enumerate() {
                out[i] = op(*x, y);
            }
            Ok(out)
        }
        other => Ok(evaluate_node_unsigned::<T, S>(other, graph)),
    }
}

impl BackendDevice for CpuDevice {
    type Storage<X: DType> = CpuStorage<X>;

    fn compile_and_run_graph_unsigned<S: Shape, T: DType>(
        &self,
        graph: &[Op<T>],
    ) -> Result<Self::Storage<T>> {
        Ok(CpuStorage(evaluate_node_unsigned::<T, S>(
            graph.last().unwrap(),
            graph,
        )))
    }

    fn compile_and_run_graph<S: Shape, T: DType + SignedDType>(
        &self,
        graph: &[Op<T>],
    ) -> Result<Self::Storage<T>> {
        Ok(CpuStorage(evaluate_node::<T, S>(
            graph.last().unwrap(),
            graph,
        )?))
    }
}
