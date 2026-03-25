use crate::Float;
use std::fmt;

pub trait LossFunction<const N: usize>: fmt::Debug {
    fn loss_and_grad(
        &self,
        output: &[Float; N],
        target: &[Float; N],
        grad: &mut [Float; N],
    ) -> Float;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MeanSquaredError;

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
