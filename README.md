<h1 align="center">
  constensor
</h1>

<h3 align="center">
ML framework featuring compile time checks and accelerated by a JIT compiler.
</h3>

<p align="center"><a href="https://ericlbuehler.github.io/constensor/constensor_core/"><b>Rust Documentation</b></a>

</p>

Constensor is a fast alternative to Candle which provides the following key features:
- **Compile time shape, dtype, and device checking**: Develop quickly
- **Opt-in half precision support**: Run on any GPU
- **Elementwise JIT kernel fusion**: Accelerate CUDA kernels automatically by fusing binary and unary operations


```rust
use constensor_core::{Cpu, Graph, GraphTensor, Tensor, R2};

fn main() {
    let graph: Graph<f32> = Graph::empty();
    let x: GraphTensor<R2<3, 4>, f32, Cpu> = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(graph.clone(), 1.0);
    let y: GraphTensor<R2<3, 4>, f32, Cpu> = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(graph.clone(), 2.0);
    let z: GraphTensor<R2<3, 4>, f32, Cpu> = x + y;
    let tensor: Tensor<R2<3, 4>, f32, Cpu> = z.to_tensor().unwrap();
    assert_eq!(tensor.data().unwrap().to_vec(), vec![vec![3.0; 4]; 3],);
}
```

## Opt-in half precision support
Via the following feature flags:
- `half` for f16
- `bfloat` for bf16