//! Training configuration, losses, optimizers, and runtime layer primitives.
//!
//! This module is the execution substrate used by the blueprint runtime. It
//! contains:
//!
//! - optimizer implementations such as [`crate::network::TrainConfig`],
//!   [`crate::network::Adam`], and [`crate::network::Sgd`]
//! - loss traits and built-in losses such as
//!   [`crate::network::LossFunction`] and
//!   [`crate::network::MeanSquaredError`]
//! - low-level statically shaped layer primitives used by materialized models
//!
//! It is no longer a separate model-construction API.

mod layers;
mod loss;
mod optim;
#[cfg(test)]
mod tests;

pub use layers::{DenseLayer, Flatten, Layer, LayerDims, ReLU, Sigmoid};
pub use loss::{LossFunction, MeanSquaredError, mse_loss};
pub use optim::{
    Adam, Initializer, KaimingUniform, Optimizer, Sgd, TrainConfig, Uniform, XavierUniform,
};
