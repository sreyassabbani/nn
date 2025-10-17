#![feature(generic_const_exprs)]
#![feature(generic_const_items)]
#![allow(incomplete_features)]

// proc macro
pub use nn_macros::network;

#[macro_use]
mod tensor;

pub use tensor::Tensor;

// helper stuff for proc macro
pub mod network;

// exposes `graph!` decl macro
pub mod autodiff;
