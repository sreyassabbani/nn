#![allow(incomplete_features)]
#![feature(generic_const_exprs, adt_const_params, unsized_const_params)]

pub type Float = f64;

#[doc(hidden)]
pub trait ReshapePreservesElementCount<const FROM: usize, const TO: usize> {}

impl<const N: usize> ReshapePreservesElementCount<N, N> for () {}

#[doc(hidden)]
pub trait ConvKernelFitsInput<
    const H: usize,
    const W: usize,
    const FH: usize,
    const FW: usize,
    const S: usize,
    const P: usize,
>
{
}

#[doc(hidden)]
pub const fn assert_conv_kernel_fits_input(
    h: usize,
    w: usize,
    fh: usize,
    fw: usize,
    stride: usize,
    pad: usize,
) {
    if conv::conv_out_dim(h, pad, fh, stride) == 0 {
        panic!("conv kernel does not fit input height");
    }
    if conv::conv_out_dim(w, pad, fw, stride) == 0 {
        panic!("conv kernel does not fit input width");
    }
}

#[doc(hidden)]
pub const fn conv_kernel_fits_input_checked(
    h: usize,
    w: usize,
    fh: usize,
    fw: usize,
    stride: usize,
    pad: usize,
) -> usize {
    assert_conv_kernel_fits_input(h, w, fh, fw, stride, pad);
    0
}

impl<
    const H: usize,
    const W: usize,
    const FH: usize,
    const FW: usize,
    const S: usize,
    const P: usize,
> ConvKernelFitsInput<H, W, FH, FW, S, P> for ()
where
    [(); conv_kernel_fits_input_checked(H, W, FH, FW, S, P)]:,
{
}

#[doc(hidden)]
pub trait ConvGeometryIsValid<
    const H: usize,
    const W: usize,
    const FH: usize,
    const FW: usize,
    const S: usize,
    const P: usize,
>: ConvKernelFitsInput<H, W, FH, FW, S, P>
{
}

impl<
    const H: usize,
    const W: usize,
    const FH: usize,
    const FW: usize,
    const S: usize,
    const P: usize,
> ConvGeometryIsValid<H, W, FH, FW, S, P> for ()
where
    (): ConvKernelFitsInput<H, W, FH, FW, S, P>,
{
}

mod blueprint;
pub mod shape;
mod tensor;

pub mod conv;
pub mod data;
pub mod vision;

pub use autodiff::{EvalTape, ExprGraph, Gradients, NodeId, Op, ReverseTape, Tape, TapeError, Var};
pub use blueprint::{
    Axis, Blueprint, BlueprintSpec, Fragment, FragmentExt, GraphRuntime, HeadsSpec, InitConfig,
    MaterializeContext, Model, PredictRuntime, TrainRuntime, TransformSpec, concat, conv, dense,
    dense_no_bias, features_input, flatten, identity, image_input, into_blueprint, relu,
    repeat_stage, residual, root, share, share_fragment, share_fragment_with_id, sigmoid, sum,
    validate_blueprint, validate_headed_blueprint, volume_input,
};
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

#[doc(hidden)]
pub const fn shared_name_id(name: &str) -> usize {
    let bytes = name.as_bytes();
    let mut hash = 1469598103934665603usize;
    let mut idx = 0;
    while idx < bytes.len() {
        hash ^= bytes[idx] as usize;
        hash = hash.wrapping_mul(1099511628211usize);
        idx += 1;
    }
    hash
}

// helper stuff for proc macro
pub mod network;

// exposes `expr!` decl macro
pub mod autodiff;
