//! Legacy sequential builder APIs.
//!
//! This module keeps the original const-generic sequential builders available
//! while the newer blueprint architecture language is being refined. The
//! public surface is intentionally small:
//!
//! - [`ModelBuilder`] chooses an input kind.
//! - [`VectorBuilder`] appends flat transforms such as dense layers.
//! - [`ImageBuilder`] appends image-specific transforms such as convolution.
//! - [`Sequential`] is the materialized trainable model.

mod image;
mod model;
mod runtime;
mod sequential;
mod vector;

pub use image::ImageBuilder;
pub use model::ModelBuilder;
pub use sequential::Sequential;
pub use vector::VectorBuilder;
