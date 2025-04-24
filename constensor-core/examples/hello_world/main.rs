use constensor_core::{Cpu, Cuda, Graph, GraphTensor, Tensor, R1, R2};

fn main() {
    let mut graph: Graph<f32> = Graph::empty();
    // let arange = GraphTensor::<R1<10>, f32, Cuda<0>>::arange(&mut graph, 0., 1.);
    let a = GraphTensor::<R2<3, 4>, f32, Cuda<0>>::fill(&mut graph, 1.0);
    let b = GraphTensor::<R2<3, 4>, f32, Cuda<0>>::fill(&mut graph, 2.0);
    let c = GraphTensor::<R2<3, 4>, f32, Cuda<0>>::fill(&mut graph, 3.0);
    let d = GraphTensor::<R2<3, 4>, f32, Cuda<0>>::fill(&mut graph, 4.0);
    let res = a * b + c;
    let _out = res + d;

    graph.optimize();
    let compiled = graph.compile().unwrap();
    let res: Tensor<R2<3, 4>, f32, _> = compiled.run().unwrap();

    // graph.visualize("graph.png").unwrap();

    let tensor: Tensor<R2<3, 4>, f32, Cuda<0>> = res;

    assert_eq!(tensor.data().unwrap().to_vec(), vec![vec![9.0; 4]; 3],);
}
