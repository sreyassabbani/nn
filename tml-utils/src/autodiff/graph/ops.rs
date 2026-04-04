//! Scalar operations supported by [`super::ExprGraph`].

use crate::Float;

/// Scalar operations supported by [`super::ExprGraph`].
#[derive(Debug, Clone, Copy)]
pub enum Op {
    Scale(Float),
    Sin,
    Cos,
    Pow(i32),
    Add,
    Mul,
}

impl Op {
    pub(super) fn validate_arity(self, inputs_len: usize) {
        let ok = match self {
            Op::Scale(_) | Op::Sin | Op::Cos | Op::Pow(_) => inputs_len == 1,
            Op::Add | Op::Mul => inputs_len >= 2,
        };

        assert!(
            ok,
            "invalid arity for {:?}: expected {}, got {}",
            self,
            match self {
                Op::Scale(_) | Op::Sin | Op::Cos | Op::Pow(_) => "1",
                Op::Add | Op::Mul => ">= 2",
            },
            inputs_len
        );
    }

    pub(super) fn apply(self, inputs: &[Float]) -> Float {
        match self {
            Op::Scale(factor) => inputs[0] * factor,
            Op::Sin => inputs[0].sin(),
            Op::Cos => inputs[0].cos(),
            Op::Pow(exp) => inputs[0].powi(exp),
            Op::Add => inputs.iter().sum(),
            Op::Mul => inputs.iter().product(),
        }
    }

    pub(super) fn compute_derivative(self, inputs: &[Float], input_idx: usize) -> Float {
        match self {
            Op::Scale(factor) => factor,
            Op::Sin => inputs[0].cos(),
            Op::Cos => -inputs[0].sin(),
            Op::Pow(exp) => {
                if exp == 0 {
                    0.0
                } else {
                    exp as Float * inputs[0].powi(exp - 1)
                }
            }
            Op::Add => 1.0,
            Op::Mul => inputs
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != input_idx)
                .map(|(_, &x)| x)
                .product(),
        }
    }
}
