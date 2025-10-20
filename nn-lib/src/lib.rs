#![allow(internal_features)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(core_intrinsics)]
#![feature(generic_const_items)]
#![feature(specialization)]

// proc macro
pub use nn_macros::network;

#[macro_use]
mod tensor;

pub mod conv;

pub use tensor::Tensor;

// helper stuff for proc macro
pub mod network;

// exposes `graph!` decl macro
pub mod autodiff;
