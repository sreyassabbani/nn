//! Elementwise activation runtime layers.

use crate::Float;

use super::{Layer, LayerDims};

/// Elementwise rectified linear activation.
#[derive(Debug)]
pub struct ReLU<const N: usize>;

/// Elementwise sigmoid activation.
#[derive(Debug)]
pub struct Sigmoid<const N: usize>;

impl<const N: usize> ReLU<N> {
    pub fn init() -> Self {
        ReLU
    }

    pub fn forward(&self, input: &[Float; N], output: &mut [Float; N]) {
        for i in 0..N {
            output[i] = input[i].max(0.0);
        }
    }

    pub fn backward(
        &self,
        input: &[Float; N],
        _output: &[Float; N],
        output_grad: &[Float; N],
        input_grad: &mut [Float; N],
    ) {
        for i in 0..N {
            input_grad[i] = if input[i] > 0.0 { output_grad[i] } else { 0.0 };
        }
    }
}

impl<const N: usize> LayerDims for ReLU<N> {
    const INPUT: usize = N;
    const OUTPUT: usize = N;
}

impl<const N: usize> Layer<N, N> for ReLU<N> {
    fn forward(&self, input: &[Float; N], output: &mut [Float; N]) {
        ReLU::forward(self, input, output);
    }

    fn backward(
        &mut self,
        input: &[Float; N],
        output: &[Float; N],
        output_grad: &[Float; N],
        input_grad: &mut [Float; N],
    ) {
        ReLU::backward(self, input, output, output_grad, input_grad);
    }
}

impl<const N: usize> Sigmoid<N> {
    pub fn init() -> Self {
        Sigmoid
    }

    pub fn forward(&self, input: &[Float; N], output: &mut [Float; N]) {
        for i in 0..N {
            output[i] = 1.0 / (1.0 + (-input[i]).exp());
        }
    }

    pub fn backward(
        &self,
        _input: &[Float; N],
        output: &[Float; N],
        output_grad: &[Float; N],
        input_grad: &mut [Float; N],
    ) {
        for i in 0..N {
            let y = output[i];
            input_grad[i] = output_grad[i] * y * (1.0 - y);
        }
    }
}

impl<const N: usize> LayerDims for Sigmoid<N> {
    const INPUT: usize = N;
    const OUTPUT: usize = N;
}

impl<const N: usize> Layer<N, N> for Sigmoid<N> {
    fn forward(&self, input: &[Float; N], output: &mut [Float; N]) {
        Sigmoid::forward(self, input, output);
    }

    fn backward(
        &mut self,
        input: &[Float; N],
        output: &[Float; N],
        output_grad: &[Float; N],
        input_grad: &mut [Float; N],
    ) {
        Sigmoid::backward(self, input, output, output_grad, input_grad);
    }
}
