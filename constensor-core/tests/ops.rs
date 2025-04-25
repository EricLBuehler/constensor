use std::f32::consts::PI;

#[cfg(feature = "cuda")]
use constensor_core::Cuda;
use constensor_core::{CompiledGraph, Cpu, Graph, GraphTensor, R1, R2, R3};
#[cfg(feature = "bfloat")]
use half::bf16;
#[cfg(feature = "half")]
use half::f16;

macro_rules! test_for_device_float {
    ($dev:ty, $name:ident) => {
        mod $name {
            use super::*;
            #[test]
            fn fill() {
                let mut graph = Graph::empty();
                let _gt = GraphTensor::<R2<3, 4>, f32, $dev>::fill(&mut graph, 0.0);
                let compiled: CompiledGraph<R2<3, 4>, f32, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                assert_eq!(
                    tensor.data().unwrap().to_vec(),
                    vec![
                        [0.0, 0.0, 0.0, 0.0,],
                        [0.0, 0.0, 0.0, 0.0,],
                        [0.0, 0.0, 0.0, 0.0,],
                    ],
                );
            }

            #[test]
            fn add_div() {
                let mut graph = Graph::empty();
                let x = GraphTensor::<R2<3, 4>, f32, $dev>::fill(&mut graph, 1.0);
                let y = GraphTensor::<R2<3, 4>, f32, $dev>::fill(&mut graph, 2.0);
                let z = GraphTensor::<R2<3, 4>, f32, $dev>::fill(&mut graph, 4.0);
                let c = x + y;
                let _res = z / c;
                let compiled: CompiledGraph<R2<3, 4>, f32, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                assert_eq!(
                    tensor.data().unwrap().to_vec(),
                    vec![
                        [1.3333334, 1.3333334, 1.3333334, 1.3333334,],
                        [1.3333334, 1.3333334, 1.3333334, 1.3333334,],
                        [1.3333334, 1.3333334, 1.3333334, 1.3333334,],
                    ],
                );
            }

            #[test]
            fn arange() {
                let mut graph = Graph::empty();
                let x = GraphTensor::<R1<4>, f32, $dev>::fill(&mut graph, 1.0);
                let y = GraphTensor::<R1<4>, f32, $dev>::arange(&mut graph, 0.0, 1.0);
                let _res = x + y;
                let compiled: CompiledGraph<R1<4>, f32, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                assert_eq!(tensor.data().unwrap().to_vec(), vec![1.0, 1.25, 1.5, 1.75]);
            }

            #[test]
            fn matmul() {
                let mut graph = Graph::empty();
                let a = GraphTensor::<R3<1, 2, 3>, f32, $dev>::ones(&mut graph);
                let b = GraphTensor::<R3<1, 3, 2>, f32, $dev>::ones(&mut graph);
                let _c = a.matmul(b);
                let compiled: CompiledGraph<R3<1, 2, 2>, f32, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                let expected: [Vec<[f32; 2]>; 1] = [vec![[3.0, 3.0], [3.0, 3.0]]];
                assert_eq!(tensor.data().unwrap().to_vec(), expected);
            }

            #[test]
            fn matmul_axpby() {
                let mut graph = Graph::empty();
                let a = GraphTensor::<R3<1, 2, 3>, f32, $dev>::ones(&mut graph);
                let b = GraphTensor::<R3<1, 3, 2>, f32, $dev>::ones(&mut graph);
                let o = GraphTensor::<R3<1, 2, 2>, f32, $dev>::ones(&mut graph);
                let _c = a.matmul_axpby(b, o, 1., 1.);
                let compiled: CompiledGraph<R3<1, 2, 2>, f32, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                let expected: [Vec<[f32; 2]>; 1] = [vec![[4.0, 4.0], [4.0, 4.0]]];
                assert_eq!(tensor.data().unwrap().to_vec(), expected);
            }
        }
    };
}

test_for_device_float!(Cpu, cpu_tests_float);
#[cfg(feature = "cuda")]
test_for_device_float!(Cuda<0>, cuda_tests_float);

macro_rules! test_for_device_int {
    ($dev:ty, $name:ident) => {
        mod $name {
            use super::*;
            #[test]
            fn fill() {
                let mut graph = Graph::empty();
                let _gt = GraphTensor::<R2<3, 4>, i32, $dev>::fill(&mut graph, 0);
                let compiled: CompiledGraph<R2<3, 4>, i32, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                assert_eq!(
                    tensor.data().unwrap().to_vec(),
                    vec![[0, 0, 0, 0,], [0, 0, 0, 0,], [0, 0, 0, 0,],],
                );
            }

            #[test]
            fn add_div() {
                let mut graph = Graph::empty();
                let x = GraphTensor::<R2<3, 4>, i32, $dev>::fill(&mut graph, 1);
                let y = GraphTensor::<R2<3, 4>, i32, $dev>::fill(&mut graph, 2);
                let z = GraphTensor::<R2<3, 4>, i32, $dev>::fill(&mut graph, 4);
                let c = x + y;
                let _res = z / c;
                let compiled: CompiledGraph<R2<3, 4>, i32, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                assert_eq!(
                    tensor.data().unwrap().to_vec(),
                    vec![[1, 1, 1, 1,], [1, 1, 1, 1,], [1, 1, 1, 1,],],
                );
            }

            #[test]
            fn arange() {
                let mut graph = Graph::empty();
                let x = GraphTensor::<R1<4>, i32, $dev>::fill(&mut graph, 1);
                let y = GraphTensor::<R1<4>, i32, $dev>::arange(&mut graph, 0, 4);
                let _res = x + y;
                let compiled: CompiledGraph<R1<4>, i32, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                assert_eq!(tensor.data().unwrap().to_vec(), vec![1, 2, 3, 4]);
            }

            #[cfg(not(feature = "cuda"))]
            #[test]
            fn matmul() {
                let mut graph = Graph::empty();
                let a = GraphTensor::<R3<1, 2, 3>, i32, $dev>::ones(&mut graph);
                let b = GraphTensor::<R3<1, 3, 2>, i32, $dev>::ones(&mut graph);
                let _c = a.matmul(b);
                let compiled: CompiledGraph<R3<1, 2, 2>, i32, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                let expected: [Vec<[i32; 2]>; 1] = [vec![[3, 3], [3, 3]]];
                assert_eq!(tensor.data().unwrap().to_vec(), expected);
            }

            #[cfg(not(feature = "cuda"))]
            #[test]
            fn matmul_axpby() {
                let mut graph = Graph::empty();
                let a = GraphTensor::<R3<1, 2, 3>, i32, $dev>::ones(&mut graph);
                let b = GraphTensor::<R3<1, 3, 2>, i32, $dev>::ones(&mut graph);
                let o = GraphTensor::<R3<1, 2, 2>, i32, $dev>::ones(&mut graph);
                let _c = a.matmul_axpby(b, o, 1, 1);
                let compiled: CompiledGraph<R3<1, 2, 2>, i32, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                let expected: [Vec<[i32; 2]>; 1] = [vec![[4, 4], [4, 4]]];
                assert_eq!(tensor.data().unwrap().to_vec(), expected);
            }
        }
    };
}

test_for_device_int!(Cpu, cpu_tests_int);
#[cfg(feature = "cuda")]
test_for_device_int!(Cuda<0>, cuda_tests_int);

#[cfg(feature = "half")]
macro_rules! test_for_device_half {
    ($dev:ty, $name:ident) => {
        mod $name {
            use super::*;
            #[test]
            fn fill() {
                let mut graph = Graph::empty();
                let _gt =
                    GraphTensor::<R2<3, 4>, f16, $dev>::fill(&mut graph, f16::from_f64_const(0.0));
                let compiled: CompiledGraph<R2<3, 4>, f16, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                assert_eq!(
                    tensor.data().unwrap().to_vec(),
                    vec![vec![f16::from_f64_const(0.0); 4]; 3],
                );
            }

            #[test]
            fn add_div() {
                let mut graph = Graph::empty();
                let x =
                    GraphTensor::<R2<3, 4>, f16, $dev>::fill(&mut graph, f16::from_f64_const(1.0));
                let y =
                    GraphTensor::<R2<3, 4>, f16, $dev>::fill(&mut graph, f16::from_f64_const(2.0));
                let z =
                    GraphTensor::<R2<3, 4>, f16, $dev>::fill(&mut graph, f16::from_f64_const(4.0));
                let c = x + y;
                let _res = z / c;
                let compiled: CompiledGraph<R2<3, 4>, f16, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                assert_eq!(
                    tensor.data().unwrap().to_vec(),
                    vec![vec![f16::from_f64_const(1.3330078); 4]; 3],
                );
            }

            #[test]
            fn arange() {
                let mut graph = Graph::empty();
                let x = GraphTensor::<R1<4>, f16, $dev>::fill(&mut graph, f16::from_f64_const(1.0));
                let y = GraphTensor::<R1<4>, f16, $dev>::arange(
                    &mut graph,
                    f16::from_f64_const(0.0),
                    f16::from_f64_const(1.0),
                );
                let _res = x + y;
                let compiled: CompiledGraph<R1<4>, f16, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                assert_eq!(
                    tensor.data().unwrap().to_vec(),
                    vec![
                        f16::from_f64_const(1.0),
                        f16::from_f64_const(1.25),
                        f16::from_f64_const(1.5),
                        f16::from_f64_const(1.75)
                    ]
                );
            }
        }
    };
}

#[cfg(feature = "half")]
test_for_device_half!(Cpu, cpu_tests_half);
#[cfg(all(feature = "cuda", feature = "half"))]
test_for_device_half!(Cuda<0>, cuda_tests_half);

#[cfg(feature = "bfloat")]
macro_rules! test_for_device_bfloat {
    ($dev:ty, $name:ident) => {
        mod $name {
            use super::*;
            #[test]
            fn fill() {
                let mut graph = Graph::empty();
                let _gt = GraphTensor::<R2<3, 4>, bf16, $dev>::fill(
                    &mut graph,
                    bf16::from_f64_const(0.0),
                );
                let compiled: CompiledGraph<R2<3, 4>, bf16, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                assert_eq!(
                    tensor.data().unwrap().to_vec(),
                    vec![vec![bf16::from_f64_const(0.0); 4]; 3],
                );
            }

            #[test]
            fn add_div() {
                let mut graph = Graph::empty();
                let x = GraphTensor::<R2<3, 4>, bf16, $dev>::fill(
                    &mut graph,
                    bf16::from_f64_const(1.0),
                );
                let y = GraphTensor::<R2<3, 4>, bf16, $dev>::fill(
                    &mut graph,
                    bf16::from_f64_const(2.0),
                );
                let z = GraphTensor::<R2<3, 4>, bf16, $dev>::fill(
                    &mut graph,
                    bf16::from_f64_const(4.0),
                );
                let c = x + y;
                let _res = z / c;
                let compiled: CompiledGraph<R2<3, 4>, bf16, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                assert_eq!(
                    tensor.data().unwrap().to_vec(),
                    vec![vec![bf16::from_f64_const(1.3330078); 4]; 3],
                );
            }

            #[test]
            fn arange() {
                let mut graph = Graph::empty();
                let x =
                    GraphTensor::<R1<4>, bf16, $dev>::fill(&mut graph, bf16::from_f64_const(1.0));
                let y = GraphTensor::<R1<4>, bf16, $dev>::arange(
                    &mut graph,
                    bf16::from_f64_const(0.0),
                    bf16::from_f64_const(1.0),
                );
                let _res = x + y;
                let compiled: CompiledGraph<R1<4>, bf16, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                assert_eq!(
                    tensor.data().unwrap().to_vec(),
                    vec![
                        bf16::from_f64_const(1.0),
                        bf16::from_f64_const(1.25),
                        bf16::from_f64_const(1.5),
                        bf16::from_f64_const(1.75)
                    ]
                );
            }
        }
    };
}

#[cfg(feature = "bfloat")]
test_for_device_bfloat!(Cpu, cpu_tests_bfloat);
#[cfg(all(feature = "cuda", feature = "bfloat"))]
test_for_device_bfloat!(Cuda<0>, cuda_tests_bfloat);

macro_rules! test_for_device_float_unary {
    ($dev:ty, $name:ident) => {
        mod $name {
            use super::*;

            #[test]
            fn add_div_neg() {
                let mut graph = Graph::empty();
                let x = GraphTensor::<R2<3, 4>, f32, $dev>::fill(&mut graph, 1.0);
                let y = GraphTensor::<R2<3, 4>, f32, $dev>::fill(&mut graph, 2.0);
                let z = GraphTensor::<R2<3, 4>, f32, $dev>::fill(&mut graph, 4.0);
                let c = x + -y;
                let _res = z / c;
                let compiled: CompiledGraph<R2<3, 4>, f32, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                assert_eq!(tensor.data().unwrap().to_vec(), vec![vec![-4.0; 4]; 3],);
            }
        }
    };
}

test_for_device_float_unary!(Cpu, cpu_tests_float_unary);
#[cfg(feature = "cuda")]
test_for_device_float_unary!(Cuda<0>, cuda_tests_float_unary);

macro_rules! test_for_device_sqrt {
    ($dev:ty, $name:ident) => {
        mod $name {
            use super::*;

            #[test]
            fn sqrt_float() {
                let mut graph = Graph::empty();
                let x = GraphTensor::<R2<3, 4>, f32, $dev>::fill(&mut graph, 4.0);
                let _res = x.sqrt();
                let compiled: CompiledGraph<R2<3, 4>, f32, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                assert_eq!(tensor.data().unwrap().to_vec(), vec![vec![2.0; 4]; 3],);
            }

            #[test]
            fn sqrt_int() {
                let mut graph = Graph::empty();
                let x = GraphTensor::<R2<3, 4>, i32, $dev>::fill(&mut graph, 5);
                let _res = x.sqrt();
                let compiled: CompiledGraph<R2<3, 4>, i32, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                assert_eq!(tensor.data().unwrap().to_vec(), vec![vec![2; 4]; 3],);
            }
        }
    };
}

test_for_device_sqrt!(Cpu, cpu_tests_sqrt);
#[cfg(feature = "cuda")]
test_for_device_sqrt!(Cuda<0>, cuda_tests_sqrt);

macro_rules! test_for_device_rand {
    ($dev:ty, $name:ident) => {
        mod $name {
            use super::*;
            #[test]
            fn rand_uniform() {
                let mut graph = Graph::empty();
                let _x: GraphTensor<R1<8>, f32, $dev> =
                    GraphTensor::<R1<8>, f32, $dev>::rand(&mut graph);
                let compiled: CompiledGraph<R1<8>, f32, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                let data = tensor.data().unwrap().to_vec();
                for &v in &data {
                    assert!((0.0..1.0).contains(&v), "value {v} out of [0,1)");
                }
            }

            #[test]
            fn randn_zero_std() {
                let mut graph = Graph::empty();
                let _x: GraphTensor<R1<8>, f32, $dev> =
                    GraphTensor::<R1<8>, f32, $dev>::randn(&mut graph, PI, 0.0);
                let compiled: CompiledGraph<R1<8>, f32, $dev> = graph.compile().unwrap();
                let tensor = compiled.run().unwrap();
                assert_eq!(tensor.data().unwrap().to_vec(), vec![PI; 8]);
            }
        }
    };
}

test_for_device_rand!(Cpu, cpu_tests_rand);
#[cfg(feature = "cuda")]
test_for_device_rand!(Cuda<0>, cuda_tests_rand);
