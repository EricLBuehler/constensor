use constensor_core::{Cpu, Graph, GraphTensor, R3};

fn main() {
    let mut graph = Graph::empty();
    let a = GraphTensor::<R3<1, 2, 3>, f32, Cpu>::ones(&mut graph);
    let b = GraphTensor::<R3<1, 3, 2>, f32, Cpu>::ones(&mut graph);
    let o = GraphTensor::<R3<1, 2, 2>, f32, Cpu>::ones(&mut graph);
    let c = a.matmul_axpby(b, o, 1., 1.);

    graph.optimize();

    graph.visualize("graph.png").unwrap();

    let tensor = c.to_tensor().unwrap();
    let expected: [Vec<[f32; 2]>; 1] = [vec![[4.0, 4.0], [4.0, 4.0]]];
    assert_eq!(tensor.data().unwrap().to_vec(), expected);
}
