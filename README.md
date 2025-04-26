# constensor

[![Crates.io](https://img.shields.io/crates/v/constensor-core.svg)](https://crates.io/crates/constensor-core) [![docs.rs](https://docs.rs/constensor-core/badge.svg)](https://docs.rs/constensor-core) [![CI](https://github.com/EricLBuehler/constensor/actions/workflows/ci.yml/badge.svg)](https://github.com/EricLBuehler/constensor/actions) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Experimental machine learning framework featuring a graph-based JIT compiler.

## Features

- Compile-time shape, dtype, and device checking
- Opt-in half-precision (`f16`) and bfloat16 (`bf16`) support
- Advanced AI compiler optimizations:
  - Elementwise JIT kernel fusion
  - Automatic inlining and in-placing
  - Constant folding
  - Dead code elimination
- Multi-device support (CPU, optional CUDA)
- Graph visualization (requires Graphviz)
- Zero-cost abstractions with idiomatic Rust API
- Broad GPU support:
  - CUDA (NVIDIA): `--features cuda`
  - HIP (AMD): `--features hip`
  - Metal (Apple): `--features metal`
  - WGPU (recommended for [all others](https://github.com/gfx-rs/wgpu?tab=readme-ov-file#supported-platforms)): `--features wgpu`

## Installation

Add `constensor-core` to your `Cargo.toml`:

```toml
[dependencies]
constensor-core = "0.1.1"
```

To enable optional features (CUDA, half-precision, bfloat16):

```toml
[dependencies.constensor-core]
version = "0.1.1"
features = ["cuda", "half", "bfloat"]
```

Or using `cargo add`:

```bash
cargo add constensor-core
```

## Quick Start

```rust
use constensor_core::{Cpu, Graph, GraphTensor, Tensor, R2};

fn main() {
    // Create an empty computation graph
    let mut graph: Graph<f32> = Graph::empty();

    // Define graph tensors
    let x = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(&mut graph, 1.0);
    let y = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(&mut graph, 2.0);

    // Build computation
    let z = y.clone() + y * x;

    // Optimize the graph
    graph.optimize();

    // Visualize the optimized graph (requires Graphviz)
    graph.visualize("graph.png").unwrap();

    // Execute and retrieve the result
    let tensor: Tensor<R2<3, 4>, f32, Cpu> = z.to_tensor().unwrap();
    assert_eq!(tensor.data().unwrap().to_vec(), vec![vec![4.0; 4]; 3]);
}
```

## Examples

Run the provided examples:

```bash
cargo run --example hello_world --features half
cargo run --example matmul --features cuda,bfloat
```

See more examples in [`constensor-core/examples`](constensor-core/examples).

## Documentation

API documentation is available on [docs.rs](https://docs.rs/constensor-core).

## More Info

- DeepWiki: https://deepwiki.com/EricLBuehler/constensor

## Contributing

Contributions are welcome! Please open issues and submit pull requests.

- Run tests with all features:

  ```bash
  cargo test --workspace --features "cuda half bfloat"
  ```

- Format code: `cargo fmt --all`
- Lint code: `cargo clippy --all-targets -- -D warnings`

## License

Licensed under MIT. See [LICENSE](LICENSE) for details.
