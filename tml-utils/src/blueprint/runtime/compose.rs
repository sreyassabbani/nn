//! Runtime combinators for composed blueprints.

use crate::Float;
use crate::network::Optimizer;

use super::GraphRuntime;

#[derive(Debug)]
#[doc(hidden)]
pub struct SeqRuntime<Left, Right> {
    pub(crate) left: Left,
    pub(crate) right: Right,
}

impl<Left, Right> GraphRuntime for SeqRuntime<Left, Right>
where
    Left: GraphRuntime,
    Right: GraphRuntime,
{
    fn forward(&self, input: &[Float]) -> Vec<Float> {
        let mid = self.left.forward(input);
        self.right.forward(&mid)
    }

    fn backward(
        &mut self,
        input: &[Float],
        _output: &[Float],
        output_grad: &[Float],
    ) -> Vec<Float> {
        let mid = self.left.forward(input);
        let right_out = self.right.forward(&mid);
        let mid_grad = self.right.backward(&mid, &right_out, output_grad);
        self.left.backward(input, &mid, &mid_grad)
    }

    fn zero_grad(&mut self) {
        self.left.zero_grad();
        self.right.zero_grad();
    }

    fn apply_gradients(&mut self, optimizer: &mut dyn Optimizer, slot: &mut usize, scale: Float) {
        self.left.apply_gradients(optimizer, slot, scale);
        self.right.apply_gradients(optimizer, slot, scale);
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub struct ResidualRuntime<Inner> {
    pub(crate) inner: Inner,
}

impl<Inner> GraphRuntime for ResidualRuntime<Inner>
where
    Inner: GraphRuntime,
{
    fn forward(&self, input: &[Float]) -> Vec<Float> {
        let inner = self.inner.forward(input);
        assert_eq!(
            inner.len(),
            input.len(),
            "residual requires the body to preserve shape",
        );
        input.iter().zip(inner.iter()).map(|(x, y)| x + y).collect()
    }

    fn backward(
        &mut self,
        input: &[Float],
        _output: &[Float],
        output_grad: &[Float],
    ) -> Vec<Float> {
        let inner_out = self.inner.forward(input);
        let inner_input_grad = self.inner.backward(input, &inner_out, output_grad);
        output_grad
            .iter()
            .zip(inner_input_grad.iter())
            .map(|(skip, body)| skip + body)
            .collect()
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn apply_gradients(&mut self, optimizer: &mut dyn Optimizer, slot: &mut usize, scale: Float) {
        self.inner.apply_gradients(optimizer, slot, scale);
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub struct SumRuntime<Left, Right> {
    pub(crate) left: Left,
    pub(crate) right: Right,
}

impl<Left, Right> GraphRuntime for SumRuntime<Left, Right>
where
    Left: GraphRuntime,
    Right: GraphRuntime,
{
    fn forward(&self, input: &[Float]) -> Vec<Float> {
        let left = self.left.forward(input);
        let right = self.right.forward(input);
        assert_eq!(
            left.len(),
            right.len(),
            "sum requires matching branch shapes"
        );
        left.iter().zip(right.iter()).map(|(l, r)| l + r).collect()
    }

    fn backward(
        &mut self,
        input: &[Float],
        _output: &[Float],
        output_grad: &[Float],
    ) -> Vec<Float> {
        let left_out = self.left.forward(input);
        let right_out = self.right.forward(input);
        let left_input_grad = self.left.backward(input, &left_out, output_grad);
        let right_input_grad = self.right.backward(input, &right_out, output_grad);
        left_input_grad
            .iter()
            .zip(right_input_grad.iter())
            .map(|(l, r)| l + r)
            .collect()
    }

    fn zero_grad(&mut self) {
        self.left.zero_grad();
        self.right.zero_grad();
    }

    fn apply_gradients(&mut self, optimizer: &mut dyn Optimizer, slot: &mut usize, scale: Float) {
        self.left.apply_gradients(optimizer, slot, scale);
        self.right.apply_gradients(optimizer, slot, scale);
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub struct ConcatRuntime<Left, Right> {
    pub(crate) left: Left,
    pub(crate) right: Right,
}

impl<Left, Right> GraphRuntime for ConcatRuntime<Left, Right>
where
    Left: GraphRuntime,
    Right: GraphRuntime,
{
    fn forward(&self, input: &[Float]) -> Vec<Float> {
        let mut output = self.left.forward(input);
        output.extend(self.right.forward(input));
        output
    }

    fn backward(
        &mut self,
        input: &[Float],
        _output: &[Float],
        output_grad: &[Float],
    ) -> Vec<Float> {
        let left_out = self.left.forward(input);
        let right_out = self.right.forward(input);
        let split = left_out.len();
        assert_eq!(
            output_grad.len(),
            split + right_out.len(),
            "concat gradient length must match the merged branch outputs",
        );
        let left_input_grad = self.left.backward(input, &left_out, &output_grad[..split]);
        let right_input_grad = self
            .right
            .backward(input, &right_out, &output_grad[split..]);
        left_input_grad
            .iter()
            .zip(right_input_grad.iter())
            .map(|(l, r)| l + r)
            .collect()
    }

    fn zero_grad(&mut self) {
        self.left.zero_grad();
        self.right.zero_grad();
    }

    fn apply_gradients(&mut self, optimizer: &mut dyn Optimizer, slot: &mut usize, scale: Float) {
        self.left.apply_gradients(optimizer, slot, scale);
        self.right.apply_gradients(optimizer, slot, scale);
    }
}
