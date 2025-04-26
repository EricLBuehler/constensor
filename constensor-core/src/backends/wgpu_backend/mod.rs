use std::{borrow::Cow, marker::PhantomData};

use super::scheduler::topo_order;
use crate::Op;
use cubecl::{cube, prelude::*, wgpu::WgpuRuntime};

use crate::{
    device::Dev,
    graph::BinaryOpType,
    storage::{BackendDevice, BackendStorage, Storage},
    CompiledGraph, DType, GraphNode, Result, Shape,
};

use super::cpu_backend::CpuStorage;

type RT = WgpuRuntime;

pub struct WgpuDevice;

pub struct WgpuStorage<X: DType> {
    ghost: PhantomData<X>,
}

impl<X: DType> BackendStorage<X> for WgpuStorage<X> {
    fn to_cpu_storage(&self) -> Result<Cow<CpuStorage<X>>> {
        todo!()
    }

    fn cast<U: DType>(&self) -> Result<Storage<U>> {
        todo!()
    }
}

#[cube(launch_unchecked)]
fn binary<F: Float>(
    a: &Sequence<Array<F>>,
    b: &Sequence<Array<F>>,
    out: &mut Sequence<Array<F>>,
    #[comptime] numel: u32,
    #[comptime] ops: Sequence<BinaryOpType>,
) {
    if ABSOLUTE_POS < numel {
        #[unroll]
        for index in 0..ops.len() {
            let op = comptime! { ops.index(index.clone()) };
            let a = a.index(index);
            let b = b.index(index);
            let o = out.index_mut(index);

            match op {
                BinaryOpType::Add => o[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS],
                BinaryOpType::Sub => o[ABSOLUTE_POS] = a[ABSOLUTE_POS] - b[ABSOLUTE_POS],
                BinaryOpType::Mul => o[ABSOLUTE_POS] = a[ABSOLUTE_POS] * b[ABSOLUTE_POS],
                BinaryOpType::Div => o[ABSOLUTE_POS] = a[ABSOLUTE_POS] / b[ABSOLUTE_POS],
            }
        }
    }
}

impl BackendDevice for WgpuDevice {
    type Storage<X: DType> = WgpuStorage<X>;

    fn compile<S: Shape, T: DType, D: Dev>(
        &self,
        graph: Vec<GraphNode<T>>,
    ) -> Result<CompiledGraph<S, T, D>> {
        // Compute topological order using shared scheduler
        let order = topo_order(&graph);
        Ok(CompiledGraph::Wgpu {
            order,
            graph,
            ghost: PhantomData,
        })
    }

    fn run_graph<S: Shape, T: DType, D: Dev>(
        &self,
        comp: &CompiledGraph<S, T, D>,
    ) -> Result<Self::Storage<T>> {
        // Sequential execution of nodes in topological order.
        #[allow(irrefutable_let_patterns)]
        let CompiledGraph::Wgpu {
            order,
            graph,
            ghost,
        } = comp
        else {
            unreachable!("Expected Wgpu compiled graph");
        };
        for &idx in order.iter() {
            let node = &graph[idx];
            if let Op::BinaryOp {
                l_id,
                r_id,
                operator,
            } = &node.op
            {
                // TODO: dispatch binary operation via cubecl kernel
                // e.g., binary::<T, RT>(...)
            }
        }
        Ok(WgpuStorage { ghost: PhantomData })
    }
}
