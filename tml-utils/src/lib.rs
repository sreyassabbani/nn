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

#[doc(hidden)]
pub trait ConvGeometryIsValid<
    const H: usize,
    const W: usize,
    const FH: usize,
    const FW: usize,
    const S: usize,
    const P: usize,
> {}

impl<
        const H: usize,
        const W: usize,
        const FH: usize,
        const FW: usize,
        const S: usize,
        const P: usize,
    > ConvGeometryIsValid<H, W, FH, FW, S, P> for ()
where
    Assert<{ conv::conv_out_dim(H, P, FH, S) > 0 }>: IsTrue,
    Assert<{ conv::conv_out_dim(W, P, FW, S) > 0 }>: IsTrue,
{
}

pub mod shape;
mod tensor;

pub mod conv;
pub mod data;

pub use autodiff::{EvalTape, ExprGraph, Gradients, NodeId, Op, ReverseTape, Tape, TapeError, Var};
pub use data::Sample;
pub use network::{
    Adam, DenseLayer, Flatten, Initializer, KaimingUniform, Layer, LayerDims, LossFunction,
    MeanSquaredError, ModelBuilder, Optimizer, ReLU, Sequential, Sgd, Sigmoid, TrainConfig,
    Uniform, XavierUniform, mse_loss,
};
pub use shape::TensorShape;
#[doc(hidden)]
pub use tensor::__tensor_from_literal;
pub use tensor::{Tensor, TensorMut, TensorRef};

// helper stuff for proc macro
pub mod network;

// exposes `expr!` decl macro
pub mod autodiff;
