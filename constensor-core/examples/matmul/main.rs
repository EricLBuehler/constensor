use constensor_core::{CompiledGraph, Cpu, DType, Graph, GraphTensor, R3};
use std::time::Instant;

fn bench<T: DType, const B: usize, const M: usize, const K: usize, const N: usize>(
    type_name: &str,
    alpha: T,
    beta: T,
) {
    // Number of times to run the matmul for averaging
    let iterations = 1;
    let mut total = std::time::Duration::new(0, 0);

    let mut graph = Graph::empty();
    let a = GraphTensor::<R3<B, M, K>, T, Cpu>::fill(&mut graph, T::from_f64(1.));
    let b = GraphTensor::<R3<B, N, K>, T, Cpu>::fill(&mut graph, T::from_f64(2.)).t();
    // let b = GraphTensor::<R3<B, K, N>, T, Cpu>::fill(&mut graph, T::from_f64(2.));
    let o = GraphTensor::<R3<B, M, N>, T, Cpu>::fill(&mut graph, T::from_f64(3.));
    let _c = a.matmul_axpby(b, o, alpha, beta);

    graph.optimize();
    let compiled: CompiledGraph<R3<B, M, N>, T, Cpu> = graph.compile().unwrap();

    for _ in 0..iterations {
        let start = Instant::now();

        let tensor = std::hint::black_box(compiled.run().unwrap());
        dbg!(tensor.data().unwrap());

        total += start.elapsed();
    }

    let avg = total / (iterations as u32);
    println!("Average execution time for {type_name} over {iterations} iterations: {avg:?}");
}

fn main() {
    const B: usize = 1;
    const M: usize = 2;
    const N: usize = 2;
    const K: usize = 2;

    bench::<f32, B, M, K, N>("f32", 1.0, 1.0);
    // bench::<i32, B, M, K, N>("i32", 1, 1);
}
