//! Materialization context and the shared runtime trait.

use std::{any::Any, collections::HashMap, fmt};

use rand::{Rng as _, SeedableRng, rngs::StdRng};

use crate::Float;
use crate::network::Optimizer;

use super::super::InitConfig;

/// Shared context used while turning blueprint specs into executable runtimes.
pub struct MaterializeContext {
    pub(crate) rng: StdRng,
    pub(crate) shared: HashMap<usize, Box<dyn Any>>,
}

impl MaterializeContext {
    /// Creates a new context from [`InitConfig`].
    pub fn new(config: InitConfig) -> Self {
        let seed = config.seed.unwrap_or_else(|| rand::rng().random());
        Self {
            rng: StdRng::seed_from_u64(seed),
            shared: HashMap::new(),
        }
    }
}

#[doc(hidden)]
pub trait GraphRuntime: fmt::Debug {
    fn forward(&self, input: &[Float]) -> Vec<Float>;
    fn backward(&mut self, input: &[Float], output: &[Float], output_grad: &[Float]) -> Vec<Float>;
    fn zero_grad(&mut self);
    fn apply_gradients(&mut self, optimizer: &mut dyn Optimizer, slot: &mut usize, scale: Float);
}
