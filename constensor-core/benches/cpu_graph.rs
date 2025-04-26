use constensor_core::{Cpu, Graph, GraphTensor, R3};
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_cpu_graph_matmul_128(c: &mut Criterion) {
    const N: usize = 128;
    type Shape = R3<1, N, N>;
    let mut graph = Graph::<f32>::empty();
    let a = GraphTensor::<Shape, f32, Cpu>::rand(&mut graph);
    let b = GraphTensor::<Shape, f32, Cpu>::rand(&mut graph);
    let _c = a.matmul(b);
    graph.optimize();
    let compiled = graph.compile::<Shape, Cpu>().unwrap();
    c.bench_function("cpu_graph_matmul_128x128", |bencher| {
        bencher.iter(|| compiled.run().unwrap());
    });
}

fn bench_cpu_graph_matmul_64(c: &mut Criterion) {
    const N: usize = 64;
    type Shape = R3<1, N, N>;
    let mut graph = Graph::<f32>::empty();
    let a = GraphTensor::<Shape, f32, Cpu>::rand(&mut graph);
    let b = GraphTensor::<Shape, f32, Cpu>::rand(&mut graph);
    let _c = a.matmul(b);
    graph.optimize();
    let compiled = graph.compile::<Shape, Cpu>().unwrap();
    c.bench_function("cpu_graph_matmul_64x64", |bencher| {
        bencher.iter(|| compiled.run().unwrap());
    });
}

fn bench_cpu_graph_matmul_256(c: &mut Criterion) {
    const N: usize = 256;
    type Shape = R3<1, N, N>;
    let mut graph = Graph::<f32>::empty();
    let a = GraphTensor::<Shape, f32, Cpu>::rand(&mut graph);
    let b = GraphTensor::<Shape, f32, Cpu>::rand(&mut graph);
    let _c = a.matmul(b);
    graph.optimize();
    let compiled = graph.compile::<Shape, Cpu>().unwrap();
    c.bench_function("cpu_graph_matmul_256x256", |bencher| {
        bencher.iter(|| compiled.run().unwrap());
    });
}

criterion_group!(benches,
    bench_cpu_graph_matmul_64,
    bench_cpu_graph_matmul_128,
    bench_cpu_graph_matmul_256
);
criterion_main!(benches);
