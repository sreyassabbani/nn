//! Built-in transform constructors and typed transform implementations.
//!
//! This module holds the type-level rewrite rules that map an input
//! [`TensorShape`](crate::shape::TensorShape) to an output shape, runtime, and
//! parameter count. The public constructor functions here are the building
//! blocks re-exported at the top of the `tml` facade crate.

mod compose;
mod leaf;

use super::{
    Blueprint, ConcatSpec, ConvSpec, DenseSpec, FlattenSpec, IdentitySpec, ReLUSpec,
    RepeatStageSpec, ResidualSpec, SharedSpec, SigmoidSpec, SumSpec,
};

/// Creates a bias-enabled dense transform.
pub fn dense<const OUT: usize>() -> Blueprint<DenseSpec<OUT, true>> {
    Blueprint::new(DenseSpec)
}

/// Creates a bias-free dense transform.
pub fn dense_no_bias<const OUT: usize>() -> Blueprint<DenseSpec<OUT, false>> {
    Blueprint::new(DenseSpec)
}

/// Creates a rectified linear activation transform.
pub fn relu() -> Blueprint<ReLUSpec> {
    Blueprint::new(ReLUSpec)
}

/// Creates a sigmoid activation transform.
pub fn sigmoid() -> Blueprint<SigmoidSpec> {
    Blueprint::new(SigmoidSpec)
}

/// Creates an identity transform.
pub fn identity() -> Blueprint<IdentitySpec> {
    Blueprint::new(IdentitySpec)
}

/// Flattens all axes into a single `features` axis.
pub fn flatten() -> Blueprint<FlattenSpec> {
    Blueprint::new(FlattenSpec)
}

/// Creates a typed convolution transform.
pub fn conv<
    const OUT: usize,
    const KH: usize,
    const KW: usize,
    const STRIDE: usize,
    const PAD: usize,
>() -> Blueprint<ConvSpec<OUT, KH, KW, STRIDE, PAD>> {
    Blueprint::new(ConvSpec)
}

/// Wraps a shape-preserving block in an additive residual connection.
pub fn residual<Spec>(body: Blueprint<Spec>) -> Blueprint<ResidualSpec<Spec>> {
    Blueprint::new(ResidualSpec {
        body: body.into_inner(),
    })
}

/// Sums the outputs of two blueprints fed the same input.
pub fn sum<Left, Right>(
    left: Blueprint<Left>,
    right: Blueprint<Right>,
) -> Blueprint<SumSpec<Left, Right>> {
    Blueprint::new(SumSpec {
        left: left.into_inner(),
        right: right.into_inner(),
    })
}

/// Concatenates the outputs of two blueprints along the selected axis.
pub fn concat<Left, Right>(
    axis: super::Axis,
    left: Blueprint<Left>,
    right: Blueprint<Right>,
) -> Blueprint<ConcatSpec<Left, Right>> {
    Blueprint::new(ConcatSpec {
        axis,
        left: left.into_inner(),
        right: right.into_inner(),
    })
}

/// Marks a blueprint as explicitly shared by identity.
pub fn share<Spec>(blueprint: &Blueprint<Spec>) -> Blueprint<SharedSpec<Spec>>
where
    Spec: Clone,
{
    Blueprint::new(SharedSpec {
        id: blueprint as *const _ as usize,
        inner: blueprint.as_inner().clone(),
    })
}

#[doc(hidden)]
pub fn repeat_stage<Spec>(blueprint: Blueprint<Spec>) -> Blueprint<RepeatStageSpec<Spec>> {
    Blueprint::new(RepeatStageSpec {
        inner: blueprint.into_inner(),
    })
}
