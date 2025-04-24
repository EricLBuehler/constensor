#[cfg(feature = "cuda")]
use constensor_core::Cuda;
use constensor_core::{CompiledGraph, Cpu, Graph, GraphTensor, R2};

macro_rules! test_for_device_fma {
    ($dev:ty, $name:ident) => {
        mod $name {
            use super::*;
            #[test]
            fn float_fma() {
                let mut graph = Graph::empty();
                let a = GraphTensor::<R2<3, 4>, f32, $dev>::fill(&mut graph, 2.0);
                let b = GraphTensor::<R2<3, 4>, f32, $dev>::fill(&mut graph, 3.0);
                let c = GraphTensor::<R2<3, 4>, f32, $dev>::fill(&mut graph, 4.0);
                let _res = a * b + c;
                let compiled: CompiledGraph<R2<3, 4>, f32, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                assert_eq!(tensor.data().unwrap().to_vec(), vec![vec![10.0; 4]; 3],);
            }

            #[test]
            fn integral_fma() {
                let mut graph = Graph::empty();
                let a = GraphTensor::<R2<3, 4>, i32, $dev>::fill(&mut graph, 2);
                let b = GraphTensor::<R2<3, 4>, i32, $dev>::fill(&mut graph, 3);
                let c = GraphTensor::<R2<3, 4>, i32, $dev>::fill(&mut graph, 4);
                let _res = a * b + c;
                let compiled: CompiledGraph<R2<3, 4>, i32, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                assert_eq!(tensor.data().unwrap().to_vec(), vec![vec![10; 4]; 3],);
            }
        }
    };
}

#[cfg(feature = "half")]
macro_rules! test_for_device_half_fma {
    ($dev:ty, $name:ident) => {
        mod $name {
            use super::*;
            #[test]
            fn float_fma() {
                use half::f16;

                let mut graph = Graph::empty();
                let a =
                    GraphTensor::<R2<3, 4>, f16, $dev>::fill(&mut graph, f16::from_f64_const(2.0));
                let b =
                    GraphTensor::<R2<3, 4>, f16, $dev>::fill(&mut graph, f16::from_f64_const(3.0));
                let c =
                    GraphTensor::<R2<3, 4>, f16, $dev>::fill(&mut graph, f16::from_f64_const(4.0));
                let _res = a * b + c;
                let compiled: CompiledGraph<R2<3, 4>, f16, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                assert_eq!(
                    tensor.data().unwrap().to_vec(),
                    vec![vec![f16::from_f64_const(10.0); 4]; 3],
                );
            }
        }
    };
}

#[cfg(feature = "bfloat")]
macro_rules! test_for_device_bfloat_fma {
    ($dev:ty, $name:ident) => {
        mod $name {
            use super::*;
            #[test]
            fn float_fma() {
                use half::bf16;

                let mut graph = Graph::empty();
                let a = GraphTensor::<R2<3, 4>, bf16, $dev>::fill(
                    &mut graph,
                    bf16::from_f64_const(2.0),
                );
                let b = GraphTensor::<R2<3, 4>, bf16, $dev>::fill(
                    &mut graph,
                    bf16::from_f64_const(3.0),
                );
                let c = GraphTensor::<R2<3, 4>, bf16, $dev>::fill(
                    &mut graph,
                    bf16::from_f64_const(4.0),
                );
                let _res = a * b + c;
                let compiled: CompiledGraph<R2<3, 4>, bf16, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                assert_eq!(
                    tensor.data().unwrap().to_vec(),
                    vec![vec![bf16::from_f64_const(10.0); 4]; 3],
                );
            }
        }
    };
}

test_for_device_fma!(Cpu, cpu_tests_fma);
#[cfg(feature = "cuda")]
test_for_device_fma!(Cuda<0>, cuda_tests_fma);

#[cfg(feature = "half")]
test_for_device_half_fma!(Cpu, cpu_tests_fma_half);
#[cfg(all(feature = "cuda", feature = "half"))]
test_for_device_half_fma!(Cuda<0>, cuda_tests_fma_half);

#[cfg(feature = "bfloat")]
test_for_device_bfloat_fma!(Cpu, cpu_tests_fma_bfloat);
#[cfg(all(feature = "cuda", feature = "half"))]
test_for_device_bfloat_fma!(Cuda<0>, cuda_tests_fma_float);
