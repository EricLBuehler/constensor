#[cfg(feature = "cuda")]
use constensor_core::Cuda;
use constensor_core::{Cpu, Graph, GraphTensor, R1, R2};
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
                let graph = Graph::empty();
                let gt = GraphTensor::<R2<3, 4>, f32, $dev>::fill(graph.clone(), 0.0);
                let tensor = gt.to_tensor().unwrap();
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
                let graph = Graph::empty();
                let x = GraphTensor::<R2<3, 4>, f32, $dev>::fill(graph.clone(), 1.0);
                let y = GraphTensor::<R2<3, 4>, f32, $dev>::fill(graph.clone(), 2.0);
                let z = GraphTensor::<R2<3, 4>, f32, $dev>::fill(graph.clone(), 4.0);
                let c = x + y;
                let res = z / c;
                let tensor = res.to_tensor().unwrap();
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
                let graph = Graph::empty();
                let x = GraphTensor::<R1<3>, f32, $dev>::fill(graph.clone(), 1.0);
                let y = GraphTensor::<R1<3>, f32, $dev>::arange(graph.clone(), 0.0, 1.0);
                let res = x + y;
                let tensor = res.to_tensor().unwrap();
                assert_eq!(tensor.data().unwrap().to_vec(), vec![1.0, 2.0, 3.0]);
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
                let graph = Graph::empty();
                let gt = GraphTensor::<R2<3, 4>, i32, $dev>::fill(graph.clone(), 0);
                let tensor = gt.to_tensor().unwrap();
                assert_eq!(
                    tensor.data().unwrap().to_vec(),
                    vec![[0, 0, 0, 0,], [0, 0, 0, 0,], [0, 0, 0, 0,],],
                );
            }

            #[test]
            fn add_div() {
                let graph = Graph::empty();
                let x = GraphTensor::<R2<3, 4>, i32, $dev>::fill(graph.clone(), 1);
                let y = GraphTensor::<R2<3, 4>, i32, $dev>::fill(graph.clone(), 2);
                let z = GraphTensor::<R2<3, 4>, i32, $dev>::fill(graph.clone(), 4);
                let c = x + y;
                let res = z / c;
                let tensor = res.to_tensor().unwrap();
                assert_eq!(
                    tensor.data().unwrap().to_vec(),
                    vec![[1, 1, 1, 1,], [1, 1, 1, 1,], [1, 1, 1, 1,],],
                );
            }

            #[test]
            fn arange() {
                let graph = Graph::empty();
                let x = GraphTensor::<R1<3>, i32, $dev>::fill(graph.clone(), 1);
                let y = GraphTensor::<R1<3>, i32, $dev>::arange(graph.clone(), 0, 1);
                let res = x + y;
                let tensor = res.to_tensor().unwrap();
                assert_eq!(tensor.data().unwrap().to_vec(), vec![1, 2, 3]);
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
                let graph = Graph::empty();
                let gt = GraphTensor::<R2<3, 4>, f16, $dev>::fill(
                    graph.clone(),
                    f16::from_f32_const(0.0),
                );
                let tensor = gt.to_tensor().unwrap();
                assert_eq!(
                    tensor.data().unwrap().to_vec(),
                    vec![vec![f16::from_f32_const(0.0); 4]; 3],
                );
            }

            #[test]
            fn add_div() {
                let graph = Graph::empty();
                let x = GraphTensor::<R2<3, 4>, f16, $dev>::fill(
                    graph.clone(),
                    f16::from_f32_const(1.0),
                );
                let y = GraphTensor::<R2<3, 4>, f16, $dev>::fill(
                    graph.clone(),
                    f16::from_f32_const(2.0),
                );
                let z = GraphTensor::<R2<3, 4>, f16, $dev>::fill(
                    graph.clone(),
                    f16::from_f32_const(4.0),
                );
                let c = x + y;
                let res = z / c;
                let tensor = res.to_tensor().unwrap();
                assert_eq!(
                    tensor.data().unwrap().to_vec(),
                    vec![vec![f16::from_f32_const(1.3330078); 4]; 3],
                );
            }

            #[test]
            fn arange() {
                let graph = Graph::empty();
                let x =
                    GraphTensor::<R1<3>, f16, $dev>::fill(graph.clone(), f16::from_f32_const(1.0));
                let y = GraphTensor::<R1<3>, f16, $dev>::arange(
                    graph.clone(),
                    f16::from_f32_const(0.0),
                    f16::from_f32_const(1.0),
                );
                let res = x + y;
                let tensor = res.to_tensor().unwrap();
                assert_eq!(
                    tensor.data().unwrap().to_vec(),
                    vec![
                        f16::from_f32_const(1.0),
                        f16::from_f32_const(2.0),
                        f16::from_f32_const(3.0)
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
                let graph = Graph::empty();
                let gt = GraphTensor::<R2<3, 4>, bf16, $dev>::fill(
                    graph.clone(),
                    bf16::from_f32_const(0.0),
                );
                let tensor = gt.to_tensor().unwrap();
                assert_eq!(
                    tensor.data().unwrap().to_vec(),
                    vec![vec![bf16::from_f32_const(0.0); 4]; 3],
                );
            }

            #[test]
            fn add_div() {
                let graph = Graph::empty();
                let x = GraphTensor::<R2<3, 4>, bf16, $dev>::fill(
                    graph.clone(),
                    bf16::from_f32_const(1.0),
                );
                let y = GraphTensor::<R2<3, 4>, bf16, $dev>::fill(
                    graph.clone(),
                    bf16::from_f32_const(2.0),
                );
                let z = GraphTensor::<R2<3, 4>, bf16, $dev>::fill(
                    graph.clone(),
                    bf16::from_f32_const(4.0),
                );
                let c = x + y;
                let res = z / c;
                let tensor = res.to_tensor().unwrap();
                assert_eq!(
                    tensor.data().unwrap().to_vec(),
                    vec![vec![bf16::from_f32_const(1.3330078); 4]; 3],
                );
            }

            #[test]
            fn arange() {
                let graph = Graph::empty();
                let x = GraphTensor::<R1<3>, bf16, $dev>::fill(
                    graph.clone(),
                    bf16::from_f32_const(1.0),
                );
                let y = GraphTensor::<R1<3>, bf16, $dev>::arange(
                    graph.clone(),
                    bf16::from_f32_const(0.0),
                    bf16::from_f32_const(1.0),
                );
                let res = x + y;
                let tensor = res.to_tensor().unwrap();
                assert_eq!(
                    tensor.data().unwrap().to_vec(),
                    vec![
                        bf16::from_f32_const(1.0),
                        bf16::from_f32_const(2.0),
                        bf16::from_f32_const(3.0)
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
