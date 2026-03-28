//! Public entry point for `tml`.
//!
//! `tml` is the ergonomic facade crate. It re-exports the `network!`, `shape!`,
//! `tensor!`, and `expr!` macros alongside the typed tensor, blueprint, and
//! training APIs that back them.

pub use tml_macro::network;
pub use tml_utils::__private;
pub use tml_utils::Float;
pub use tml_utils::autodiff::{
    EvalTape, ExprGraph, Gradients, NodeId, Op, ReverseTape, Tape, TapeError, Var,
};
pub use tml_utils::blueprint::{
    Axis, Blueprint, BlueprintSpec, Fragment, FragmentExt, GraphRuntime, HeadsSpec, InitConfig,
    MaterializeContext, Model, PredictRuntime, TrainRuntime, TransformSpec, concat, conv, dense,
    dense_no_bias, features_input, flatten, identity, image_input, into_blueprint, relu,
    repeat_stage, residual, root, share, share_fragment, share_fragment_with_id, sigmoid, sum,
    validate_blueprint, validate_headed_blueprint, volume_input,
};
pub use tml_utils::conv;
pub use tml_utils::data::Sample;
pub use tml_utils::expr;
pub use tml_utils::network::{
    Adam, DenseLayer, Flatten, Initializer, KaimingUniform, Layer, LayerDims, LossFunction,
    MeanSquaredError, ModelBuilder, Optimizer, ReLU, Sequential, Sgd, Sigmoid, TrainConfig,
    Uniform, XavierUniform, mse_loss,
};
pub use tml_utils::shape;
pub use tml_utils::shape::{Dim, Nil, TensorShape};
pub use tml_utils::tensor;
pub use tml_utils::tensor::{Tensor, TensorLiteral, TensorMut, TensorRef};
pub use tml_utils::vision;
