//! Explicit scalar expression graphs and reusable evaluation tapes.

mod eval;
mod model;
mod ops;
mod tapes;

pub use model::{ExprGraph, Node, NodeId};
pub use ops::Op;
pub use tapes::{EvalTape, ReverseTape};
