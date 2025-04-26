use constensor_core::{Graph, GraphTensor, Tensor, Wgpu, R2};

fn main() {
    let mut graph: Graph<f32> = Graph::empty();
    let a = GraphTensor::<R2<3, 4>, f32, Wgpu>::fill(&mut graph, 1.0);
    let b = GraphTensor::<R2<3, 4>, f32, Wgpu>::fill(&mut graph, 2.0);
    let res = a + b;

    graph.visualize("graph.png").unwrap();

    let compiled: constensor_core::CompiledGraph<R2<3, 4>, f32, Wgpu> = graph.compile().unwrap();
    let res = compiled.run().unwrap();

    let tensor: Tensor<R2<3, 4>, f32, Wgpu> = res;

    assert_eq!(tensor.data().unwrap().to_vec(), vec![vec![9.0; 4]; 3],);
}
