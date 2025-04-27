use std::{borrow::Cow, collections::HashMap, marker::PhantomData};

use super::scheduler::topo_order;
use crate::Op;
use cubecl::{
    channel::MutexComputeChannel,
    cube,
    prelude::*,
    server::Handle,
    wgpu::{WgpuRuntime, WgpuServer},
};

use crate::{
    device::Dev,
    graph::BinaryOpType,
    storage::{BackendDevice, BackendStorage, Storage},
    CompiledGraph, DType, GraphNode, Result, Shape,
};

mod kernels;

use super::cpu_backend::CpuStorage;

type RT = WgpuRuntime;

#[cfg(any(feature = "cuda", feature = "hip"))]
const DEVICE: cubecl::wgpu::WgpuDevice = cubecl::wgpu::WgpuDevice::DiscreteGpu(0);
#[cfg(feature = "metal")]
const DEVICE: cubecl::wgpu::WgpuDevice = cubecl::wgpu::WgpuDevice::IntegratedGpu(0);
#[cfg(not(any(feature = "cuda", feature = "hip", feature = "metal")))]
const DEVICE: cubecl::wgpu::WgpuDevice = cubecl::wgpu::WgpuDevice::DefaultDevice;

fn client() -> ComputeClient<WgpuServer, MutexComputeChannel<WgpuServer>> {
    RT::client(&DEVICE)
}

const VECTORIZATION: u32 = 4;

pub struct WgpuDevice;

pub struct WgpuStorage<X: DType> {
    handle: Handle,
    ghost: PhantomData<X>,
}

impl<X: DType> BackendStorage<X> for WgpuStorage<X> {
    fn to_cpu_storage(&self) -> Result<Cow<CpuStorage<X>>> {
        let client = client();

        let bytes = client.read_one(self.handle.clone().binding());
        let output = X::from_bytes(&bytes);

        Ok(Cow::Owned(CpuStorage(output.to_vec())))
    }

    fn cast<U: DType>(&self) -> Result<Storage<U>> {
        todo!()
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

        let client = client();

        let mut handles = HashMap::new();
        for &idx in order.iter() {
            let node = &graph[idx];
            let out_elem_count: usize = node.shape.iter().product();

            match &node.op {
                Op::Fill { v } => {
                    let output_handle = client.empty(out_elem_count * core::mem::size_of::<T>());

                    let out: ArrayArg<'_, RT> = unsafe {
                        ArrayArg::from_raw_parts::<T>(
                            &output_handle,
                            out_elem_count,
                            VECTORIZATION as u8,
                        )
                    };
                    unsafe {
                        kernels::fill::launch_unchecked::<T, WgpuRuntime>(
                            &client,
                            CubeCount::Static(VECTORIZATION, 1, 1),
                            CubeDim::new((out_elem_count as u32).div_ceil(VECTORIZATION), 1, 1),
                            out,
                            ScalarArg::new(*v),
                            out_elem_count as u32,
                        );
                    };

                    handles.insert(idx, output_handle.clone());
                }
                Op::BinaryOp {
                    l_id,
                    r_id,
                    operator,
                } => {
                    let a_handle = &handles[&l_id.get()];
                    let b_handle = &handles[&r_id.get()];
                    let output_handle = client.empty(out_elem_count * core::mem::size_of::<T>());

                    unsafe {
                        let mut a_seq: SequenceArg<'_, RT, Array<T>> = SequenceArg::new();
                        a_seq.push(ArrayArg::from_raw_parts::<T>(
                            &a_handle,
                            out_elem_count,
                            VECTORIZATION as u8,
                        ));

                        let mut b_seq: SequenceArg<'_, RT, Array<T>> = SequenceArg::new();
                        b_seq.push(ArrayArg::from_raw_parts::<T>(
                            &b_handle,
                            out_elem_count,
                            VECTORIZATION as u8,
                        ));

                        let mut out_seq: SequenceArg<'_, RT, Array<T>> = SequenceArg::new();
                        out_seq.push(ArrayArg::from_raw_parts::<T>(
                            &output_handle,
                            out_elem_count,
                            VECTORIZATION as u8,
                        ));

                        let mut ops = Sequence::new();
                        ops.push(*operator);
                        kernels::binary::launch_unchecked(
                            &client,
                            CubeCount::Static(VECTORIZATION, 1, 1),
                            CubeDim::new((out_elem_count as u32).div_ceil(VECTORIZATION), 1, 1),
                            a_seq,
                            b_seq,
                            out_seq,
                            out_elem_count as u32,
                            ops,
                        );
                    };

                    handles.insert(idx, output_handle.clone());
                }

                _ => todo!(),
            }
        }

        let key = *handles.keys().max().unwrap();
        let final_handle = handles.remove(&key).expect("No output");
        Ok(WgpuStorage {
            handle: final_handle,
            ghost: PhantomData,
        })
    }
}
