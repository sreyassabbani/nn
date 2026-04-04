//! Runtime support for materialized blueprints.
//!
//! The blueprint API keeps architecture descriptions in typed specs. This
//! module contains the executable runtimes produced when those specs are
//! materialized.

mod compose;
mod context;
mod leaf;
mod runner;
mod shared;

pub use compose::{ConcatRuntime, ResidualRuntime, SeqRuntime, SumRuntime};
pub use context::{GraphRuntime, MaterializeContext};
pub use leaf::{LeafRuntime, LinearRuntime};
pub use runner::GraphRunner;
pub use shared::SharedRuntime;
