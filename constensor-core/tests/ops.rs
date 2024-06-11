use constensor_core::{Cpu, Graph, GraphTensor, R1, R2};

#[test]
fn fill() {
    let graph = Graph::empty();
    let gt = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(graph.clone(), 0.0);
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
    let x = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(graph.clone(), 1.0);
    let y = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(graph.clone(), 2.0);
    let z = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(graph.clone(), 4.0);
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
    let x = GraphTensor::<R1<3>, f32, Cpu>::fill(graph.clone(), 1.0);
    let y = GraphTensor::<R1<3>, f32, Cpu>::arange(graph.clone(), 0.0, 1.0);
    let res = x + y;
    let tensor = res.to_tensor().unwrap();
    assert_eq!(tensor.data().unwrap().to_vec(), vec![1.0, 2.0, 3.0]);
}
