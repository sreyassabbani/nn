#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub type Float = f64;

#[doc(hidden)]
pub struct Assert<const CHECK: bool>;

#[doc(hidden)]
pub trait IsTrue {}

impl IsTrue for Assert<true> {}

#[doc(hidden)]
pub trait ReshapePreservesElementCount<const FROM: usize, const TO: usize> {}

impl<const N: usize> ReshapePreservesElementCount<N, N> for () {}

pub mod shape;
mod tensor;

pub mod conv;
pub mod data;

pub use autodiff::{EvalTape, ExprGraph, Gradients, NodeId, Op, ReverseTape, Tape, TapeError, Var};
pub use data::Sample;
pub use network::{
    Adam, AppendLayer, Chain, DenseLayer, End, Flatten, Initializer, IntoChain, KaimingUniform,
    Layer, LayerDims, Loss, MeanSquaredError, Optimizer, ReLU, Sequential, Sgd, Sigmoid,
    TrainConfig, Uniform, XavierUniform, mse_loss,
};
pub use shape::TensorShape;
#[doc(hidden)]
pub use tensor::__tensor_from_literal;
pub use tensor::{Tensor, TensorMut, TensorRef};

// helper stuff for proc macro
pub mod network;

// exposes `expr!` decl macro
pub mod autodiff;
