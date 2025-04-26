use std::{borrow::Cow, marker::PhantomData};

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
        #[cfg(any(feature = "cuda", feature = "hip"))]
        let device = cubecl::wgpu::WgpuDevice::DiscreteGpu(0);
        #[cfg(feature = "metal")]
        let device = cubecl::wgpu::WgpuDevice::IntegratedGpu(0);
        #[cfg(not(any(feature = "cuda", feature = "hip", feature = "metal")))]
        let device = cubecl::wgpu::WgpuDevice::DefaultDevice;

        let client = RT::client(&device);

        let a = &[1., 2., 3., 4., 5., 6., 7., 8.];
        let b = &[1., 2., 3., 4., 5., 6., 7., 8.];
        let vectorization = 4;
        let output_handle = client.empty(a.len() * core::mem::size_of::<f32>());
        let a_handle = client.create(f32::as_bytes(a));
        let b_handle = client.create(f32::as_bytes(b));

        unsafe {
            let mut a_seq = SequenceArg::new();
            a_seq.push(ArrayArg::from_raw_parts::<f32>(
                &a_handle,
                a.len(),
                vectorization as u8,
            ));

            let mut b_seq = SequenceArg::new();
            b_seq.push(ArrayArg::from_raw_parts::<f32>(
                &b_handle,
                b.len(),
                vectorization as u8,
            ));

            let mut out_seq = SequenceArg::new();
            out_seq.push(ArrayArg::from_raw_parts::<f32>(
                &output_handle,
                a.len(),
                vectorization as u8,
            ));

            let mut ops = Sequence::new();
            ops.push(BinaryOpType::Add);
            binary::launch_unchecked::<f32, RT>(
                &client,
                CubeCount::Static(vectorization, 1, 1),
                CubeDim::new((a.len() as u32).div_ceil(vectorization), 1, 1),
                a_seq,
                b_seq,
                out_seq,
                a.len() as u32,
                ops,
            )
        };

        let bytes = client.read_one(output_handle.binding());
        let output = f32::from_bytes(&bytes);

        println!("Executed runtime {:?} => {output:?}", RT::name(&client));
        todo!()
    }

    fn run_graph<S: Shape, T: DType, D: Dev>(
        &self,
        graph: &CompiledGraph<S, T, D>,
    ) -> Result<Self::Storage<T>> {
        todo!()
    }
}
