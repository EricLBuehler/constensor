pub mod concretetensor;
pub mod graphtensor;

pub use concretetensor::Tensor;
pub use graphtensor::GraphTensor;

/// Compute default (contiguous) strides for a tensor of given shape.
pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut acc = 1;
    // Iterate dims in reverse to accumulate products
    for dim in shape.iter().rev() {
        strides.push(acc);
        acc *= *dim;
    }
    strides.reverse();
    strides
}
