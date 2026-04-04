//! Leaf runtime wrapping one classic layer implementation.

use std::fmt;

use crate::Float;
use crate::network::DenseLayer;
use crate::network::{Layer, Optimizer};

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

/// Runtime wrapper that applies one dense layer independently across every
/// chunk of the final axis.
#[derive(Debug)]
#[doc(hidden)]
pub struct LinearRuntime<const CHUNKS: usize, const INPUT: usize, const OUTPUT: usize> {
    layer: DenseLayer<INPUT, OUTPUT>,
}

impl<const CHUNKS: usize, const INPUT: usize, const OUTPUT: usize>
    LinearRuntime<CHUNKS, INPUT, OUTPUT>
{
    pub(crate) fn new(layer: DenseLayer<INPUT, OUTPUT>) -> Self {
        Self { layer }
    }
}

impl<const CHUNKS: usize, const INPUT: usize, const OUTPUT: usize> GraphRuntime
    for LinearRuntime<CHUNKS, INPUT, OUTPUT>
{
    fn forward(&self, input: &[Float]) -> Vec<Float> {
        assert_eq!(
            input.len(),
            CHUNKS * INPUT,
            "linear runtime input length must match the chunked layer input",
        );

        let mut output = vec![0.0; CHUNKS * OUTPUT];
        for (input_chunk, output_chunk) in input
            .chunks_exact(INPUT)
            .zip(output.chunks_exact_mut(OUTPUT))
        {
            let input_chunk: &[Float; INPUT] = input_chunk
                .try_into()
                .expect("linear runtime input chunk must match the layer input");
            let output_chunk: &mut [Float; OUTPUT] = output_chunk
                .try_into()
                .expect("linear runtime output chunk must match the layer output");
            self.layer.forward(input_chunk, output_chunk);
        }
        output
    }

    fn backward(&mut self, input: &[Float], output: &[Float], output_grad: &[Float]) -> Vec<Float> {
        assert_eq!(
            input.len(),
            CHUNKS * INPUT,
            "linear runtime input length must match the chunked layer input",
        );
        assert_eq!(
            output.len(),
            CHUNKS * OUTPUT,
            "linear runtime output length must match the chunked layer output",
        );
        assert_eq!(
            output_grad.len(),
            CHUNKS * OUTPUT,
            "linear runtime gradient length must match the chunked layer output",
        );

        let mut input_grad = vec![0.0; CHUNKS * INPUT];
        for (((input_chunk, output_chunk), grad_chunk), input_grad_chunk) in input
            .chunks_exact(INPUT)
            .zip(output.chunks_exact(OUTPUT))
            .zip(output_grad.chunks_exact(OUTPUT))
            .zip(input_grad.chunks_exact_mut(INPUT))
        {
            let input_chunk: &[Float; INPUT] = input_chunk
                .try_into()
                .expect("linear runtime input chunk must match the layer input");
            let output_chunk: &[Float; OUTPUT] = output_chunk
                .try_into()
                .expect("linear runtime output chunk must match the layer output");
            let grad_chunk: &[Float; OUTPUT] = grad_chunk
                .try_into()
                .expect("linear runtime gradient chunk must match the layer output");
            let mut chunk_input_grad = [0.0; INPUT];
            self.layer
                .backward(input_chunk, output_chunk, grad_chunk, &mut chunk_input_grad);
            input_grad_chunk.copy_from_slice(&chunk_input_grad);
        }
        input_grad
    }

    fn zero_grad(&mut self) {
        self.layer.zero_grad();
    }

    fn apply_gradients(&mut self, optimizer: &mut dyn Optimizer, slot: &mut usize, scale: Float) {
        self.layer.apply_gradients(optimizer, slot, scale);
    }
}
