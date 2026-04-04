//! Leaf runtime wrapping one classic layer implementation.

use std::fmt;

use crate::Float;
use crate::network::Layer;

use super::GraphRuntime;

#[derive(Debug)]
#[doc(hidden)]
pub struct LeafRuntime<L, const INPUT: usize, const OUTPUT: usize> {
    layer: L,
}

impl<L, const INPUT: usize, const OUTPUT: usize> LeafRuntime<L, INPUT, OUTPUT> {
    pub(crate) fn new(layer: L) -> Self {
        Self { layer }
    }
}

impl<L, const INPUT: usize, const OUTPUT: usize> GraphRuntime for LeafRuntime<L, INPUT, OUTPUT>
where
    L: Layer<INPUT, OUTPUT> + fmt::Debug + 'static,
{
    fn forward(&self, input: &[Float]) -> Vec<Float> {
        let input: &[Float; INPUT] = input
            .try_into()
            .expect("leaf runtime input length must match the layer input");
        let mut output = [0.0; OUTPUT];
        self.layer.forward(input, &mut output);
        output.to_vec()
    }

    fn backward(&mut self, input: &[Float], output: &[Float], output_grad: &[Float]) -> Vec<Float> {
        let input: &[Float; INPUT] = input
            .try_into()
            .expect("leaf runtime input length must match the layer input");
        let output: &[Float; OUTPUT] = output
            .try_into()
            .expect("leaf runtime output length must match the layer output");
        let output_grad: &[Float; OUTPUT] = output_grad
            .try_into()
            .expect("leaf runtime gradient length must match the layer output");
        let mut input_grad = [0.0; INPUT];
        self.layer
            .backward(input, output, output_grad, &mut input_grad);
        input_grad.to_vec()
    }

    fn zero_grad(&mut self) {
        self.layer.zero_grad();
    }

    fn apply_gradients(
        &mut self,
        optimizer: &mut dyn crate::network::Optimizer,
        slot: &mut usize,
        scale: Float,
    ) {
        self.layer.apply_gradients(optimizer, slot, scale);
    }
}
