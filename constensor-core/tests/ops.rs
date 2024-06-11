use constensor_core::{Cpu, Graph, GraphTensor, R1, R2};

/*#[test]
fn fill() {
    let graph = Graph::empty();
    let _ = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(graph.clone(), 0.0);
}

#[test]
fn add_div() {
    let graph = Graph::empty();
    let x = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(graph.clone(), 1.0);
    let y = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(graph.clone(), 2.0);
    let z = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(graph.clone(), 4.0);
    let c = x + y;
    let res = z / c;
    let _ = res.to_tensor();
}*/

#[test]
fn arange() {
    let graph = Graph::empty();
    let x = GraphTensor::<R1<3>, f32, Cpu>::fill(graph.clone(), 1.0);
    let y = GraphTensor::<R1<3>, f32, Cpu>::arange(graph.clone(), 0.0, 1.0);
    let res = x + y;
    let _ = res.to_tensor();
}
