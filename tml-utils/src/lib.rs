#![allow(incomplete_features)]
#![feature(generic_const_exprs, adt_const_params, unsized_const_params)]
//! Core implementation crate for `tml`.
//!
//! `tml_utils` contains the typed tensor, shape, blueprint, autodiff, and
//! training building blocks used by the top-level [`tml`](https://docs.rs/tml)
//! crate. Most downstream users should prefer `tml`, which re-exports the
//! intended public API with the `network!`, `shape!`, and `tensor!` macros.

pub type Float = f64;

#[doc(hidden)]
pub mod __private;

/// Reverse-mode and forward-mode scalar autodiff primitives.
pub mod autodiff;
/// Typed architecture blueprints and materialization.
pub mod blueprint;
/// Low-level convolution kernels and helpers.
pub mod conv;
/// Dataset sample helpers.
pub mod data;
/// Classic sequential-network components retained during the migration.
pub mod network;
/// Compile-time tensor shapes.
pub mod shape;
/// Typed tensors and tensor views.
pub mod tensor;
/// Built-in reusable vision fragments.
pub mod vision;

pub use data::Sample;
pub use shape::TensorShape;
pub use tensor::{Tensor, TensorMut, TensorRef};
