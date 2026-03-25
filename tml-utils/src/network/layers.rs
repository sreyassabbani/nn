use crate::Float;
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::{Initializer, Optimizer, XavierUniform};

pub trait Layer<const IN: usize, const OUT: usize> {
    fn forward(&self, input: &[Float; IN], output: &mut [Float; OUT]);
    fn backward(
        &mut self,
        input: &[Float; IN],
        output: &[Float; OUT],
        output_grad: &[Float; OUT],
        input_grad: &mut [Float; IN],
    );

    fn zero_grad(&mut self) {}

    fn apply_gradients(
        &mut self,
        _optimizer: &mut dyn Optimizer,
        _slot: &mut usize,
        _scale: Float,
    ) {
    }
}

pub trait LayerDims {
    const INPUT: usize;
    const OUTPUT: usize;
}

#[derive(Debug)]
pub struct DenseLayer<const IN: usize, const OUT: usize> {
    pub(crate) weights: Box<[Float]>,
    pub(crate) biases: Box<[Float; OUT]>,
    pub(crate) weight_grads: Box<[Float]>,
    pub(crate) bias_grads: Box<[Float; OUT]>,
    pub(crate) use_bias: bool,
}

#[derive(Debug)]
pub struct ReLU<const N: usize>;

#[derive(Debug)]
pub struct Sigmoid<const N: usize>;

#[derive(Debug)]
pub struct Flatten<const N: usize>;

impl<const IN: usize, const OUT: usize> DenseLayer<IN, OUT> {
    pub fn init() -> Self {
        Self::with_initializer(XavierUniform)
    }

    pub fn seeded(seed: u64) -> Self {
        Self::with_initializer_and_seed(XavierUniform, seed)
    }

    pub fn with_initializer<I: Initializer>(initializer: I) -> Self {
        let mut rng = rand::rng();
        Self::with_initializer_and_rng(initializer, &mut rng)
    }

    pub fn with_initializer_and_seed<I: Initializer>(initializer: I, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::with_initializer_and_rng(initializer, &mut rng)
    }

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
