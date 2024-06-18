#[cfg(feature = "cuda")]
use constensor_core::Cuda;
use constensor_core::{Cpu, Graph, GraphTensor, R2};

macro_rules! test_for_device_fma {
    ($dev:ty, $name:ident) => {
        mod $name {
            use super::*;
            #[test]
            fn simple_fma() {
                let graph = Graph::empty();
                let a = GraphTensor::<R2<3, 4>, f32, $dev>::fill(graph.clone(), 2.0);
                let b = GraphTensor::<R2<3, 4>, f32, $dev>::fill(graph.clone(), 3.0);
                let c = GraphTensor::<R2<3, 4>, f32, $dev>::fill(graph.clone(), 4.0);
                let res = a * b + c;
                let tensor = res.to_tensor_signed().unwrap();
                assert_eq!(tensor.data().unwrap().to_vec(), vec![vec![10.0; 4]; 3],);
            }
        }
    };
}

test_for_device_fma!(Cpu, cpu_tests_fma);
#[cfg(feature = "cuda")]
test_for_device_fma!(Cuda<0>, cuda_tests_fma);
