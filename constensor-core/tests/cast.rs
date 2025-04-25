use constensor_core::{Graph, GraphTensor, CompiledGraph, Cpu, R1, R2, R3};

// Test casting a 1D tensor from f32 to f64
#[test]
fn cast_f32_to_f64_1d() {
    let mut graph = Graph::empty();
    let _x = GraphTensor::<R1<4>, f32, Cpu>::fill(&mut graph, 1.5);
    let compiled: CompiledGraph<R1<4>, f32, Cpu> = graph.compile().unwrap();
    let tensor = compiled.run().unwrap();
    let casted = tensor.cast::<f64>().unwrap();
    let data = casted.data().unwrap().into_owned();
    assert_eq!(data, vec![1.5_f64; 4]);
}

// Test casting a 2D tensor from f64 to f32
#[test]
fn cast_f64_to_f32_2d() {
    let mut graph = Graph::empty();
    let _x = GraphTensor::<R2<2, 3>, f64, Cpu>::fill(&mut graph, 2.75);
    let compiled: CompiledGraph<R2<2, 3>, f64, Cpu> = graph.compile().unwrap();
    let tensor = compiled.run().unwrap();
    let casted = tensor.cast::<f32>().unwrap();
    let data = casted.data().unwrap().into_owned();
    assert_eq!(data, vec![vec![2.75_f32; 3]; 2]);
}

// Test casting a 3D tensor from i32 to f32
#[test]
fn cast_i32_to_f32_3d() {
    let mut graph = Graph::empty();
    let _x = GraphTensor::<R3<1, 2, 3>, i32, Cpu>::fill(&mut graph, 7);
    let compiled: CompiledGraph<R3<1, 2, 3>, i32, Cpu> = graph.compile().unwrap();
    let tensor = compiled.run().unwrap();
    let casted = tensor.cast::<f32>().unwrap();
    let data = casted.data().unwrap().into_owned();
    let expected = vec![vec![vec![7.0_f32; 3]; 2]; 1];
    assert_eq!(data, expected);
}

// Test casting from f32 to i32 truncates toward zero
#[test]
fn cast_f32_to_i32_truncate() {
    let mut graph = Graph::empty();
    let _x = GraphTensor::<R1<3>, f32, Cpu>::fill(&mut graph, 1.9);
    let compiled: CompiledGraph<R1<3>, f32, Cpu> = graph.compile().unwrap();
    let tensor = compiled.run().unwrap();
    let casted = tensor.cast::<i32>().unwrap();
    let data = casted.data().unwrap().into_owned();
    assert_eq!(data, vec![1_i32; 3]);
}