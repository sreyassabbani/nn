//! Scalar autodiff graphs, reusable tapes, and operator-overloaded variables.
//!
//! This module exposes two related APIs:
//! - [`crate::autodiff::ExprGraph`] for explicit graph construction and
//!   reusable forward/reverse tapes.
//! - [`crate::autodiff::Tape`] and [`crate::autodiff::Var`] for a more
//!   ergonomic, operator-overloaded scalar workflow.

mod expr_macro;
mod graph;
mod tape;
#[cfg(test)]
mod tests;

pub use graph::{EvalTape, ExprGraph, Node, NodeId, Op, ReverseTape};
pub use tape::{Gradients, Tape, TapeError, Var};
