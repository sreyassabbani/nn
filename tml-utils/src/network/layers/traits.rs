//! Shared traits for the statically shaped runtime layers.

use crate::Float;

use super::super::Optimizer;

/// A statically shaped trainable or non-trainable layer.
///
/// [`Layer`] is the execution-facing trait used by materialized blueprint
/// runtimes. Both input and output widths are const-generic parameters, so the
/// layer can read/write fixed-size arrays without dynamic shape checks.
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

/// Compile-time width metadata for a runtime layer.
pub trait LayerDims {
    const INPUT: usize;
    const OUTPUT: usize;
}
