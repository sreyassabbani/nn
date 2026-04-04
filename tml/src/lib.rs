//! Public entry point for `tml`.
//!
//! `tml` is the ergonomic facade crate. It re-exports the `network!`, `shape!`,
//! `tensor!`, and `expr!` macros alongside the typed tensor, blueprint, and
//! training APIs that back them.

pub use tml_macro::network;
pub use tml_utils::__private;
pub use tml_utils::Float;
pub use tml_utils::autodiff;
pub use tml_utils::autodiff::{
    EvalTape, ExprGraph, Gradients, NodeId, Op, ReverseTape, Tape, TapeError, Var,
};
pub use tml_utils::blueprint;
pub use tml_utils::blueprint::{
    Axis, Blueprint, BlueprintSpec, Fragment, FragmentExt, GraphRuntime, HeadedModel, HeadsSpec,
    InitConfig, LinearOverLastAxis, LinearSpec, MaterializeContext, Model, PredictRuntime,
    TransformSpec, concat, conv, dense, dense_no_bias, features_input, flatten, identity,
    image_input, into_blueprint, linear, linear_no_bias, relu, repeat_stage, residual, root,
    share_fragment_with_id, sigmoid, sum, validate_blueprint, validate_headed_blueprint,
    volume_input,
};
pub use tml_utils::conv;
pub use tml_utils::data::Sample;
pub use tml_utils::expr;
pub use tml_utils::network;
pub use tml_utils::network::{Adam, LossFunction, MeanSquaredError, Sgd, TrainConfig};
pub use tml_utils::shape;
pub use tml_utils::shape::{Dim, Nil, TensorShape};
pub use tml_utils::tensor;
pub use tml_utils::tensor::{Tensor, TensorLiteral, TensorMut, TensorRef};
pub use tml_utils::vision;
