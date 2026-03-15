#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub type Float = f64;

#[doc(hidden)]
pub struct Assert<const CHECK: bool>;

#[doc(hidden)]
pub trait IsTrue {}

impl IsTrue for Assert<true> {}

mod tensor;
pub mod shape;

pub mod conv;
pub mod data;

pub use autodiff::{EvalTape, ExprGraph, Gradients, NodeId, Op, ReverseTape, Tape, TapeError, Var};
pub use data::Sample;
pub use network::{DenseLayer, Flatten, Layer, ReLU, Sigmoid, TrainConfig, mse_loss};
pub use shape::TensorShape;
#[doc(hidden)]
pub use tensor::__tensor_from_literal;
pub use tensor::{Tensor, TensorView, TensorViewMut};

// helper stuff for proc macro
pub mod network;

// exposes `expr!` decl macro
pub mod autodiff;
