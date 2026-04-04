//! Typed architecture blueprints and rooted model materialization.
//!
//! The blueprint system separates three concerns:
//!
//! - [`crate::blueprint::Blueprint`] stores an architecture specification
//!   without parameter state.
//! - rooting helpers such as [`crate::blueprint::root`] attach an explicit
//!   input shape.
//! - spec traits such as [`crate::blueprint::TransformSpec`] provide
//!   compile-time shape rules.
//!
//! This keeps architecture composition, shape validation, and runtime
//! materialization distinct instead of conflating them into one builder object.

mod api;
mod fragments;
mod rooted;
mod runtime;
mod specs;
mod transforms;

pub use api::{Axis, Blueprint, HeadedModel, InitConfig, Model, PredictRuntime, TrainRuntime};
pub use fragments::{
    Fragment, FragmentExt, into_blueprint, share_fragment, share_fragment_with_id,
};
pub use rooted::{
    Rooted, RootedBlueprint, features_input, image_input, root, validate_blueprint,
    validate_headed_blueprint, volume_input,
};
pub use runtime::{GraphRuntime, MaterializeContext};
pub use specs::{
    BlueprintSpec, ConcatAlong, ConcatSpec, ConvSpec, DenseExpectsFlatInput, DenseSpec,
    FlattenSpec, HeadsSpec, IdentitySpec, ReLUSpec, RepeatStageSpec, ResidualSpec, SeqCompatible,
    SeqSpec, ShapePreserving, SharedSpec, SigmoidSpec, SumSpec, TransformSpec,
};
#[allow(unused_imports)]
pub use transforms::{
    concat, conv, dense, dense_no_bias, flatten, identity, relu, repeat_stage, residual, share,
    sigmoid, sum,
};

pub(crate) use rooted::{describe_shape, features_axis};
pub(crate) use runtime::GraphRunner;
