use constensor_core::{Cpu, Graph, GraphTensor, Tensor, R2};

fn main() {
    let graph: Graph<f32> = Graph::empty();
    let x: GraphTensor<R2<3, 4>, f32, Cpu> = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(graph.clone(), 1.0);
    let y: GraphTensor<R2<3, 4>, f32, Cpu> = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(graph.clone(), 2.0);
    let z: GraphTensor<R2<3, 4>, f32, Cpu> = x + y;
    let tensor: Tensor<R2<3, 4>, f32, Cpu> = z.to_tensor().unwrap();
    assert_eq!(tensor.data().unwrap().to_vec(), vec![vec![3.0; 4]; 3],);
}
