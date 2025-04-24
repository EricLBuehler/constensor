use constensor_core::{Cpu, Graph, GraphTensor, Tensor, R1, R2};

fn main() {
    let mut graph: Graph<f32> = Graph::empty();
    let _arange = GraphTensor::<R1<10>, f32, Cpu>::arange(&mut graph, 0., 1.);
    let a = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(&mut graph, 1.0);
    let b = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(&mut graph, 2.0);
    let c = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(&mut graph, 3.0);
    let d = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(&mut graph, 4.0);
    let res = a * b + c;
    let _out = res + d;

    graph.optimize();

    graph.visualize("graph.png").unwrap();

    let compiled: constensor_core::CompiledGraph<R2<3, 4>, f32, Cpu> = graph.compile().unwrap();
    let res = compiled.run().unwrap();

    let tensor: Tensor<R2<3, 4>, f32, Cpu> = res;

    assert_eq!(tensor.data().unwrap().to_vec(), vec![vec![9.0; 4]; 3],);
}
