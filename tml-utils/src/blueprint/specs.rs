//! Blueprint spec traits and type-level contract markers.

use crate::shape::TensorShape;
use std::{collections::HashSet, fmt};

use super::{Axis, MaterializeContext};

/// An inspectable blueprint specification for a fixed input shape.
///
/// `InputShape` is the compile-time tensor shape accepted by the blueprint.
/// Implementors provide summary, shape-trace, and parameter-count behavior.
pub trait BlueprintSpec<InputShape: TensorShape + 'static>: Clone + fmt::Debug + 'static {
    fn push_summary(&self, lines: &mut Vec<String>);
    fn push_shape_trace(&self, input_axes: &[Axis], lines: &mut Vec<String>);
    fn parameter_count(&self, seen_shared: &mut HashSet<usize>) -> usize;
}

impl<InputShape, Spec> BlueprintSpec<InputShape> for Spec
where
    InputShape: TensorShape + 'static,
    Spec: TransformSpec<InputShape>,
{
    fn push_summary(&self, lines: &mut Vec<String>) {
        <Spec as TransformSpec<InputShape>>::push_summary(self, lines);
    }

    fn push_shape_trace(&self, input_axes: &[Axis], lines: &mut Vec<String>) {
        <Spec as TransformSpec<InputShape>>::push_shape_trace(self, input_axes, lines);
    }

    fn parameter_count(&self, seen_shared: &mut HashSet<usize>) -> usize {
        <Spec as TransformSpec<InputShape>>::parameter_count(self, seen_shared)
    }
}

/// A [`BlueprintSpec`] that materializes into named prediction heads.
pub trait HeadsSpec<InputShape: TensorShape + 'static>: BlueprintSpec<InputShape> {
    /// The structured prediction value returned by the materialized runtime.
    type Output: 'static;
    /// The concrete runtime used to produce [`HeadsSpec::Output`].
    type Runtime: 'static;

    fn materialize_heads(&self, ctx: &mut MaterializeContext) -> Self::Runtime;
}

/// A single transform or transform-combinator over a fixed input shape.
///
/// `InputShape` is the compile-time shape accepted by the transform.
/// [`TransformSpec::OutputShape`] is the compile-time shape produced after the
/// transform runs.
pub trait TransformSpec<InputShape: TensorShape + 'static>: Clone + fmt::Debug + 'static {
    /// The compile-time output shape of the transform.
    type OutputShape: TensorShape;
    /// The flattened element count of [`TransformSpec::OutputShape`].
    const OUTPUT_SIZE: usize;
    /// The runtime object produced during materialization.
    type Runtime: 'static;

    fn materialize(&self, ctx: &mut MaterializeContext) -> Self::Runtime;
    fn push_summary(&self, lines: &mut Vec<String>) {
        lines.push(self.description());
    }
    fn push_shape_trace(&self, input_axes: &[Axis], lines: &mut Vec<String>) -> Box<[Axis]> {
        let output_axes = self.output_axes(input_axes);
        lines.push(format!(
            "{}: {} -> {}",
            self.description(),
            super::describe_shape::<InputShape>(input_axes),
            super::describe_shape::<Self::OutputShape>(&output_axes)
        ));
        output_axes
    }
    fn parameter_count(&self, seen_shared: &mut HashSet<usize>) -> usize;
    fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]>;
    fn description(&self) -> String;
}

/// Shape-level contract for dense layers that require a flat input.
#[doc(hidden)]
pub trait DenseExpectsFlatInput<const OUT: usize, const BIAS: bool>: TensorShape + 'static {
    type Runtime: 'static;

    fn materialize_dense(ctx: &mut MaterializeContext) -> Self::Runtime;
    fn dense_parameter_count() -> usize;
}

/// Shape-level contract for last-axis linear transforms.
///
/// Unlike [`DenseExpectsFlatInput`], this preserves every prefix axis and only
/// rewrites the extent of the final axis.
#[doc(hidden)]
pub trait LinearOverLastAxis<const OUT: usize, const BIAS: bool>: TensorShape + 'static {
    type OutputShape: TensorShape;
    type Runtime: 'static;
    const OUTPUT_SIZE: usize;

    fn materialize_linear(ctx: &mut MaterializeContext) -> Self::Runtime;
    fn linear_parameter_count() -> usize;
}

/// Dense layer spec marker.
///
/// `OUT` is the output feature width. `BIAS` controls whether the runtime dense
/// layer includes a trainable bias term.
#[derive(Debug, Clone, Copy)]
pub struct DenseSpec<const OUT: usize, const BIAS: bool = true>;

/// Last-axis linear transform spec marker.
///
/// `OUT` is the new width of the final axis. Every prefix axis is preserved.
#[derive(Debug, Clone, Copy)]
pub struct LinearSpec<const OUT: usize, const BIAS: bool = true>;

/// ReLU spec marker.
#[derive(Debug, Clone, Copy)]
pub struct ReLUSpec;

/// Sigmoid spec marker.
#[derive(Debug, Clone, Copy)]
pub struct SigmoidSpec;

/// Identity spec marker.
#[derive(Debug, Clone, Copy)]
pub struct IdentitySpec;

/// Flatten spec marker.
#[derive(Debug, Clone, Copy)]
pub struct FlattenSpec;

/// 2D convolution spec marker.
///
/// Generic parameters:
/// - `OUT`: number of output channels
/// - `KH`: kernel height
/// - `KW`: kernel width
/// - `STRIDE`: shared vertical/horizontal stride
/// - `PAD`: shared vertical/horizontal padding
#[derive(Debug, Clone, Copy)]
pub struct ConvSpec<
    const OUT: usize,
    const KH: usize,
    const KW: usize,
    const STRIDE: usize,
    const PAD: usize,
>;

/// Sequential composition of two transform specs.
///
/// [`SeqSpec`] is the type-level equivalent of “run `left`, then feed its
/// output into `right`”.
#[derive(Debug, Clone)]
pub struct SeqSpec<Left, Right> {
    pub(crate) left: Left,
    pub(crate) right: Right,
}

/// Additive residual application of one body spec.
///
/// The body must be shape-preserving so that its output can be added back to
/// the original input.
#[derive(Debug, Clone)]
pub struct ResidualSpec<Body> {
    pub(crate) body: Body,
}

/// Sum of two branch specs fed the same input.
#[derive(Debug, Clone)]
pub struct SumSpec<Left, Right> {
    pub(crate) left: Left,
    pub(crate) right: Right,
}

/// Concatenation of two branch specs along one named axis.
#[derive(Debug, Clone)]
pub struct ConcatSpec<Left, Right> {
    pub(crate) axis: Axis,
    pub(crate) left: Left,
    pub(crate) right: Right,
}

/// Marker for a stage reused via `repeat`.
#[derive(Debug, Clone)]
pub struct RepeatStageSpec<Spec> {
    pub(crate) inner: Spec,
}

/// Marker for a shared fragment with a stable sharing identity.
#[derive(Debug, Clone)]
pub struct SharedSpec<Spec> {
    pub(crate) id: usize,
    pub(crate) inner: Spec,
}

/// Compile-time compatibility rule for sequencing two transforms.
///
/// Implementations connect an `InputShape`, a left transform, and a right
/// transform into a valid composed runtime and output shape.
#[doc(hidden)]
pub trait SeqCompatible<InputShape, Left, Right>: TensorShape + 'static
where
    InputShape: TensorShape + 'static,
    Left: TransformSpec<InputShape>,
    Right: TransformSpec<Left::OutputShape>,
{
    type OutputShape: TensorShape;
    const OUTPUT_SIZE: usize;
    type Runtime: 'static;

    fn materialize_seq(left: &Left, right: &Right, ctx: &mut MaterializeContext) -> Self::Runtime;
}

/// Marker trait for transforms that leave the input shape unchanged.
pub trait ShapePreserving<InputShape: TensorShape + 'static>:
    TransformSpec<InputShape, OutputShape = InputShape>
{
}

impl<InputShape, Spec> ShapePreserving<InputShape> for Spec
where
    InputShape: TensorShape + 'static,
    Spec: TransformSpec<InputShape, OutputShape = InputShape>,
{
}

/// Compile-time concat rule for two branch output shapes.
///
/// This determines whether two branch outputs may be concatenated along a
/// selected axis and, if so, what output shape results.
#[doc(hidden)]
pub trait ConcatAlong<
    InputShape: TensorShape + 'static,
    LeftOut: TensorShape,
    RightOut: TensorShape,
>
{
    type OutputShape: TensorShape;
    fn axis_ok(axis: Axis) -> bool;
}
