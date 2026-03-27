use crate::network::{LossFunction, MeanSquaredError, TrainConfig};
use crate::shape::TensorShape;
use crate::{Float, Sample};
use std::{collections::HashSet, fmt, marker::PhantomData};

mod fragments;
mod runtime;
mod transforms;

pub use fragments::{Fragment, FragmentExt, into_blueprint, share_fragment};
pub use runtime::{GraphRuntime, MaterializeContext};
#[allow(unused_imports)]
pub use transforms::{
    concat, conv, dense, dense_no_bias, flatten, identity, relu, repeat_stage, residual, share,
    sigmoid, sum,
};

use runtime::GraphRunner;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    Features,
    Channels,
    Length,
    Depth,
    Height,
    Width,
}

impl Axis {
    fn as_str(self) -> &'static str {
        match self {
            Axis::Features => "features",
            Axis::Channels => "channels",
            Axis::Length => "length",
            Axis::Depth => "depth",
            Axis::Height => "height",
            Axis::Width => "width",
        }
    }
}

#[derive(Debug, Clone)]
pub struct InitConfig {
    seed: Option<u64>,
}

impl InitConfig {
    pub fn new() -> Self {
        Self { seed: None }
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

impl Default for InitConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct Blueprint<Spec> {
    spec: Spec,
}

impl<Spec> Blueprint<Spec> {
    pub const fn new(spec: Spec) -> Self {
        Self { spec }
    }

    #[doc(hidden)]
    pub fn into_inner(self) -> Spec {
        self.spec
    }

    #[doc(hidden)]
    pub fn as_inner(&self) -> &Spec {
        &self.spec
    }
}

impl<Spec> Blueprint<Spec>
where
    Spec: Clone,
{
    pub fn then<Next>(self, next: Blueprint<Next>) -> Blueprint<SeqSpec<Spec, Next>> {
        Blueprint::new(SeqSpec {
            left: self.spec,
            right: next.spec,
        })
    }
}

pub struct Model<const INPUT: usize, const OUTPUT: usize> {
    inner: Box<dyn TrainRuntime<INPUT, OUTPUT>>,
}

impl<const INPUT: usize, const OUTPUT: usize> Model<INPUT, OUTPUT> {
    fn new(inner: Box<dyn TrainRuntime<INPUT, OUTPUT>>) -> Self {
        Self { inner }
    }

    pub fn predict(&self, input: &[Float; INPUT]) -> [Float; OUTPUT] {
        self.inner.predict(input)
    }

    pub fn fit(&mut self, samples: &[Sample<INPUT, OUTPUT>], config: TrainConfig) -> Float {
        self.fit_with_loss(samples, &MeanSquaredError, config)
    }

    pub fn fit_with_loss(
        &mut self,
        samples: &[Sample<INPUT, OUTPUT>],
        loss_fn: &dyn LossFunction<OUTPUT>,
        config: TrainConfig,
    ) -> Float {
        self.inner.fit_with_loss(samples, loss_fn, config)
    }
}

impl<const INPUT: usize, const OUTPUT: usize> fmt::Debug for Model<INPUT, OUTPUT> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Model")
            .field("input", &INPUT)
            .field("output", &OUTPUT)
            .finish()
    }
}

#[derive(Debug)]
pub struct HeadedModel<const INPUT: usize, Output> {
    inner: Box<dyn PredictRuntime<INPUT, Output>>,
}

impl<const INPUT: usize, Output> HeadedModel<INPUT, Output> {
    fn new(inner: Box<dyn PredictRuntime<INPUT, Output>>) -> Self {
        Self { inner }
    }

    pub fn predict(&self, input: &[Float; INPUT]) -> Output {
        self.inner.predict(input)
    }
}

pub trait PredictRuntime<const INPUT: usize, Output>: fmt::Debug {
    fn predict(&self, input: &[Float; INPUT]) -> Output;
}

pub trait TrainRuntime<const INPUT: usize, const OUTPUT: usize>:
    PredictRuntime<INPUT, [Float; OUTPUT]>
{
    fn fit_with_loss(
        &mut self,
        samples: &[Sample<INPUT, OUTPUT>],
        loss_fn: &dyn LossFunction<OUTPUT>,
        config: TrainConfig,
    ) -> Float;
}

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct Rooted<InputShape, Spec>
where
    InputShape: TensorShape + 'static,
{
    spec: Spec,
    axis_names: Box<[Axis]>,
    _shape: PhantomData<InputShape>,
}

pub type RootedBlueprint<InputShape, Spec> = Blueprint<Rooted<InputShape, Spec>>;

pub fn root<InputShape, Spec>(
    spec: Blueprint<Spec>,
    axis_names: Vec<Axis>,
) -> RootedBlueprint<InputShape, Spec>
where
    InputShape: TensorShape + 'static,
{
    Blueprint::new(Rooted {
        spec: spec.into_inner(),
        axis_names: axis_names.into_boxed_slice(),
        _shape: PhantomData,
    })
}

pub fn features_input<const N: usize, Spec>(
    spec: Blueprint<Spec>,
) -> RootedBlueprint<crate::shape!(N), Spec> {
    root::<crate::shape!(N), _>(spec, vec![Axis::Features])
}

pub fn image_input<const C: usize, const H: usize, const W: usize, Spec>(
    spec: Blueprint<Spec>,
) -> RootedBlueprint<crate::shape!(C, H, W), Spec> {
    root::<crate::shape!(C, H, W), _>(spec, vec![Axis::Channels, Axis::Height, Axis::Width])
}

pub fn volume_input<const C: usize, const D: usize, const H: usize, const W: usize, Spec>(
    spec: Blueprint<Spec>,
) -> RootedBlueprint<crate::shape!(C, D, H, W), Spec> {
    root::<crate::shape!(C, D, H, W), _>(
        spec,
        vec![Axis::Channels, Axis::Depth, Axis::Height, Axis::Width],
    )
}

#[doc(hidden)]
pub fn validate_blueprint<InputShape, Spec>(
    blueprint: RootedBlueprint<InputShape, Spec>,
) -> RootedBlueprint<InputShape, Spec>
where
    InputShape: TensorShape + 'static,
    Spec: TransformSpec<InputShape>,
{
    blueprint
}

#[doc(hidden)]
pub fn validate_headed_blueprint<InputShape, Spec>(
    blueprint: RootedBlueprint<InputShape, Spec>,
) -> RootedBlueprint<InputShape, Spec>
where
    InputShape: TensorShape + 'static,
    Spec: HeadsSpec<InputShape>,
{
    blueprint
}

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

impl<InputShape, Spec> Blueprint<Rooted<InputShape, Spec>>
where
    InputShape: TensorShape + 'static,
    Spec: BlueprintSpec<InputShape>,
{
    pub fn summary(&self) -> String {
        let mut lines = vec![format!(
            "input {}",
            describe_shape::<InputShape>(&self.spec.axis_names)
        )];
        self.spec.spec.push_summary(&mut lines);
        lines.join("\n")
    }

    pub fn shape_trace(&self) -> String {
        let mut lines = Vec::new();
        self.spec
            .spec
            .push_shape_trace(&self.spec.axis_names, &mut lines);
        lines.join("\n")
    }

    pub fn parameter_count(&self) -> usize {
        let mut seen = HashSet::new();
        self.spec.spec.parameter_count(&mut seen)
    }
}

impl<InputShape, Spec> Blueprint<Rooted<InputShape, Spec>>
where
    InputShape: TensorShape + 'static,
    Spec: TransformSpec<InputShape>,
    Spec::Runtime: GraphRuntime + 'static,
{
    pub fn materialize(
        &self,
        config: InitConfig,
    ) -> Model<{ InputShape::SIZE }, { Spec::OUTPUT_SIZE }> {
        let mut ctx = MaterializeContext::new(config);
        let runtime = self.spec.spec.materialize(&mut ctx);
        Model::new(Box::new(GraphRunner::new(runtime)))
    }
}

pub trait HeadsSpec<InputShape: TensorShape + 'static>: BlueprintSpec<InputShape> {
    type Output: 'static;
    type Runtime: 'static;

    fn materialize_heads(&self, ctx: &mut MaterializeContext) -> Self::Runtime;
}

impl<InputShape, Spec> Blueprint<Rooted<InputShape, Spec>>
where
    InputShape: TensorShape + 'static,
    Spec: HeadsSpec<InputShape>,
    Spec::Runtime: PredictRuntime<{ InputShape::SIZE }, Spec::Output> + 'static,
{
    pub fn materialize_heads(
        &self,
        config: InitConfig,
    ) -> HeadedModel<{ InputShape::SIZE }, Spec::Output> {
        let mut ctx = MaterializeContext::new(config);
        let runtime = self.spec.spec.materialize_heads(&mut ctx);
        HeadedModel::new(Box::new(runtime))
    }

    pub fn heads_summary(&self) -> String {
        self.summary()
    }

    pub fn heads_shape_trace(&self) -> String {
        self.shape_trace()
    }
}

fn describe_shape<Shape: TensorShape>(axes: &[Axis]) -> String {
    let dims = Shape::dims();
    if dims.len() == axes.len() {
        let parts = axes
            .iter()
            .zip(dims.iter())
            .map(|(axis, dim)| format!("{}: {}", axis.as_str(), dim))
            .collect::<Vec<_>>();
        format!("({})", parts.join(", "))
    } else {
        format!("{dims:?}")
    }
}

fn features_axis() -> Box<[Axis]> {
    vec![Axis::Features].into_boxed_slice()
}

pub trait TransformSpec<InputShape: TensorShape + 'static>: Clone + fmt::Debug + 'static {
    type OutputShape: TensorShape;
    const OUTPUT_SIZE: usize;
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
            describe_shape::<InputShape>(input_axes),
            describe_shape::<Self::OutputShape>(&output_axes)
        ));
        output_axes
    }
    fn parameter_count(&self, seen_shared: &mut HashSet<usize>) -> usize;
    fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]>;
    fn description(&self) -> String;
}

#[doc(hidden)]
pub trait DenseExpectsFlatInput<const OUT: usize, const BIAS: bool>: TensorShape + 'static {
    type Runtime: 'static;

    fn materialize_dense(ctx: &mut MaterializeContext) -> Self::Runtime;
    fn dense_parameter_count() -> usize;
}

#[derive(Debug, Clone, Copy)]
pub struct DenseSpec<const OUT: usize, const BIAS: bool = true>;

#[derive(Debug, Clone, Copy)]
pub struct ReLUSpec;

#[derive(Debug, Clone, Copy)]
pub struct SigmoidSpec;

#[derive(Debug, Clone, Copy)]
pub struct IdentitySpec;

#[derive(Debug, Clone, Copy)]
pub struct FlattenSpec;

#[derive(Debug, Clone, Copy)]
pub struct ConvSpec<
    const OUT: usize,
    const KH: usize,
    const KW: usize,
    const STRIDE: usize,
    const PAD: usize,
>;

#[derive(Debug, Clone)]
pub struct SeqSpec<Left, Right> {
    pub(crate) left: Left,
    pub(crate) right: Right,
}

#[derive(Debug, Clone)]
pub struct ResidualSpec<Body> {
    pub(crate) body: Body,
}

#[derive(Debug, Clone)]
pub struct SumSpec<Left, Right> {
    pub(crate) left: Left,
    pub(crate) right: Right,
}

#[derive(Debug, Clone)]
pub struct ConcatSpec<Left, Right> {
    pub(crate) axis: Axis,
    pub(crate) left: Left,
    pub(crate) right: Right,
}

#[derive(Debug, Clone)]
pub struct RepeatStageSpec<Spec> {
    pub(crate) inner: Spec,
}

#[derive(Debug, Clone)]
pub struct SharedSpec<Spec> {
    pub(crate) id: usize,
    pub(crate) inner: Spec,
}

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
