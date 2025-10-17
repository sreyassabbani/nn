#![feature(generic_arg_infer)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

// proc macro
pub use nn_macros::network;

// helper stuff for proc macro
pub mod network;

// exposes `graph!` decl macro
pub mod autodiff;
