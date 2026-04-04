//! Runtime layer primitives used by materialized blueprints.
//!
//! These layers are intentionally small, const-generic execution units. They
//! are not the primary architecture-construction API; [`crate::blueprint`]
//! owns that role. Instead, the blueprint runtime materializes down to these
//! pieces for forward/backward execution.

mod activations;
mod dense;
mod flatten;
mod traits;

pub use activations::{ReLU, Sigmoid};
pub use dense::DenseLayer;
pub use flatten::Flatten;
pub use traits::{Layer, LayerDims};
