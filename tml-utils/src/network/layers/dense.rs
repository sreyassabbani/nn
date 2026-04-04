//! Dense runtime layer implementation.

use crate::Float;
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::super::{Initializer, Optimizer, XavierUniform};
use super::{Layer, LayerDims};

/// A fully connected affine layer.
///
/// `IN` is the input feature width and `OUT` is the output feature width.
/// Weights are stored row-major by output unit.
#[derive(Debug)]
pub struct DenseLayer<const IN: usize, const OUT: usize> {
    pub(crate) weights: Box<[Float]>,
    pub(crate) biases: Box<[Float; OUT]>,
    pub(crate) weight_grads: Box<[Float]>,
    pub(crate) bias_grads: Box<[Float; OUT]>,
    pub(crate) use_bias: bool,
}

impl<const IN: usize, const OUT: usize> DenseLayer<IN, OUT> {
    /// Initializes with Xavier-uniform weights and zero bias.
    pub fn init() -> Self {
        Self::with_initializer(XavierUniform)
    }

    /// Initializes deterministically from a seed.
    pub fn seeded(seed: u64) -> Self {
        Self::with_initializer_and_seed(XavierUniform, seed)
    }

    /// Initializes with an explicit initializer.
    pub fn with_initializer<I: Initializer>(initializer: I) -> Self {
        let mut rng = rand::rng();
        Self::with_initializer_and_rng(initializer, &mut rng)
    }

    /// Initializes with an explicit initializer and deterministic seed.
    pub fn with_initializer_and_seed<I: Initializer>(initializer: I, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::with_initializer_and_rng(initializer, &mut rng)
    }

    /// Initializes with an explicit RNG source.
    pub fn with_initializer_and_rng<I: Initializer, R: Rng + ?Sized>(
        initializer: I,
        rng: &mut R,
    ) -> Self {
        let mut weights = vec![0.0; IN * OUT].into_boxed_slice();
        initializer.fill(&mut weights, IN, OUT, rng);
        Self {
            weights,
            biases: Box::new([0.0; OUT]),
            weight_grads: vec![0.0; IN * OUT].into_boxed_slice(),
            bias_grads: Box::new([0.0; OUT]),
            use_bias: true,
        }
    }

    /// Returns a copy of the layer that omits the bias term.
    pub fn without_bias(mut self) -> Self {
        self.use_bias = false;
        self.biases.fill(0.0);
        self
    }

    pub fn forward(&self, input: &[Float; IN], output: &mut [Float; OUT]) {
        for (o, out) in output.iter_mut().enumerate() {
            let row = &self.weights[o * IN..(o + 1) * IN];
            let mut sum = if self.use_bias { self.biases[o] } else { 0.0 };
            for (weight, inp) in row.iter().zip(input.iter()) {
                sum += *weight * *inp;
            }
            *out = sum;
        }
    }

    pub fn backward(
        &mut self,
        input: &[Float; IN],
        _output: &[Float; OUT],
        output_grad: &[Float; OUT],
        input_grad: &mut [Float; IN],
    ) {
        input_grad.fill(0.0);

        for (o, &grad) in output_grad.iter().enumerate() {
            let row = &self.weights[o * IN..(o + 1) * IN];
            for (in_grad, weight) in input_grad.iter_mut().zip(row.iter()) {
                *in_grad += *weight * grad;
            }
        }

        for (o, &grad) in output_grad.iter().enumerate() {
            if self.use_bias {
                self.bias_grads[o] += grad;
            }
            let row_grads = &mut self.weight_grads[o * IN..(o + 1) * IN];
            for (weight_grad, inp) in row_grads.iter_mut().zip(input.iter()) {
                *weight_grad += grad * *inp;
            }
        }
    }
}

impl<const IN: usize, const OUT: usize> LayerDims for DenseLayer<IN, OUT> {
    const INPUT: usize = IN;
    const OUTPUT: usize = OUT;
}

impl<const IN: usize, const OUT: usize> Layer<IN, OUT> for DenseLayer<IN, OUT> {
    fn forward(&self, input: &[Float; IN], output: &mut [Float; OUT]) {
        DenseLayer::forward(self, input, output);
    }

    fn backward(
        &mut self,
        input: &[Float; IN],
        output: &[Float; OUT],
        output_grad: &[Float; OUT],
        input_grad: &mut [Float; IN],
    ) {
        DenseLayer::backward(self, input, output, output_grad, input_grad);
    }

    fn zero_grad(&mut self) {
        self.weight_grads.fill(0.0);
        self.bias_grads.fill(0.0);
    }

    fn apply_gradients(&mut self, optimizer: &mut dyn Optimizer, slot: &mut usize, scale: Float) {
        optimizer.update_parameter(*slot, &mut self.weights, &self.weight_grads, scale);
        *slot += 1;
        if self.use_bias {
            optimizer.update_parameter(
                *slot,
                self.biases.as_mut_slice(),
                self.bias_grads.as_slice(),
                scale,
            );
            *slot += 1;
        }
        self.zero_grad();
    }
}
