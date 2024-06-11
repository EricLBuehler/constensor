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

## Opt-in half precision support
Via the following feature flags:
- `half` for f16
- `bfloat` for bf16