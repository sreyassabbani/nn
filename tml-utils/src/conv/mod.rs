//! Low-level convolution layers and geometry helpers.

mod geometry;
mod layer;
#[cfg(test)]
mod tests;
mod traits;

pub use geometry::conv_out_dim;
pub use layer::{Conv, Filter};
pub use traits::{ConvIO, ConvOps};
