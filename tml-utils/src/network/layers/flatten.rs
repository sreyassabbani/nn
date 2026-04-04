//! Flatten/no-op runtime layer.

use crate::Float;

use super::{Layer, LayerDims};

/// Identity layer used when the runtime reinterprets higher-rank activations as
/// flat vectors.
#[derive(Debug)]
pub struct Flatten<const N: usize>;

impl<const N: usize> Flatten<N> {
    pub fn init() -> Self {
        Flatten
    }

    pub fn forward(&self, input: &[Float; N], output: &mut [Float; N]) {
        output.copy_from_slice(input);
    }

    pub fn backward(
        &self,
        _input: &[Float; N],
        _output: &[Float; N],
        output_grad: &[Float; N],
        input_grad: &mut [Float; N],
    ) {
        input_grad.copy_from_slice(output_grad);
    }
}

impl<const N: usize> LayerDims for Flatten<N> {
    const INPUT: usize = N;
    const OUTPUT: usize = N;
}

impl<const N: usize> Layer<N, N> for Flatten<N> {
    fn forward(&self, input: &[Float; N], output: &mut [Float; N]) {
        Flatten::forward(self, input, output);
    }

    fn backward(
        &mut self,
        input: &[Float; N],
        output: &[Float; N],
        output_grad: &[Float; N],
        input_grad: &mut [Float; N],
    ) {
        Flatten::backward(self, input, output, output_grad, input_grad);
    }
}
