//! Loss traits and built-in loss helpers for materialized models.

use crate::Float;
use std::fmt;

/// A differentiable loss over fixed-size outputs.
///
/// Implementors compute both the scalar loss value and the gradient of that
/// loss with respect to the model output.
pub trait LossFunction<const N: usize>: fmt::Debug {
    fn loss_and_grad(
        &self,
        output: &[Float; N],
        target: &[Float; N],
        grad: &mut [Float; N],
    ) -> Float;
}

/// Mean-squared error over two fixed-size vectors.
#[derive(Debug, Clone, Copy, Default)]
pub struct MeanSquaredError;

/// Computes mean-squared error and writes the output gradient in one pass.
pub fn mse_loss<const N: usize>(
    output: &[Float; N],
    target: &[Float; N],
    grad: &mut [Float; N],
) -> Float {
    let scale = 2.0 / N as Float;
    let loss = output
        .iter()
        .zip(target.iter())
        .zip(grad.iter_mut())
        .map(|((&o, &t), g)| {
            let diff = o - t;
            *g = diff * scale;
            diff * diff
        })
        .sum::<Float>();
    loss / N as Float
}

impl<const N: usize> LossFunction<N> for MeanSquaredError {
    fn loss_and_grad(
        &self,
        output: &[Float; N],
        target: &[Float; N],
        grad: &mut [Float; N],
    ) -> Float {
        mse_loss(output, target, grad)
    }
}
