//! Shared runtime wrapper for explicitly shared fragments.

use std::{cell::RefCell, rc::Rc};

use crate::Float;
use crate::network::Optimizer;

use super::GraphRuntime;

#[derive(Debug)]
#[doc(hidden)]
pub struct SharedRuntime<Inner> {
    pub(crate) inner: Rc<RefCell<Inner>>,
}

impl<Inner> GraphRuntime for SharedRuntime<Inner>
where
    Inner: GraphRuntime + 'static,
{
    fn forward(&self, input: &[Float]) -> Vec<Float> {
        self.inner.borrow().forward(input)
    }

    fn backward(&mut self, input: &[Float], output: &[Float], output_grad: &[Float]) -> Vec<Float> {
        self.inner.borrow_mut().backward(input, output, output_grad)
    }

    fn zero_grad(&mut self) {
        self.inner.borrow_mut().zero_grad();
    }

    fn apply_gradients(&mut self, optimizer: &mut dyn Optimizer, slot: &mut usize, scale: Float) {
        self.inner
            .borrow_mut()
            .apply_gradients(optimizer, slot, scale);
    }
}
