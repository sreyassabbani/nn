use std::{
    any::Any,
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt,
    marker::PhantomData,
    rc::Rc,
};

use rand::{Rng as _, SeedableRng, rngs::StdRng};

use crate::conv::{Conv, conv_out_dim};
use crate::network::{
    DenseLayer, Flatten, Layer, LossFunction, MeanSquaredError, Optimizer, ReLU, Sigmoid,
    TrainConfig, XavierUniform,
};
use crate::shape::TensorShape;
use crate::{ConvGeometryIsValid, Float, Sample, shape};

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

    pub(crate) fn into_inner(self) -> Spec {
        self.spec
    }

    pub(crate) fn as_inner(&self) -> &Spec {
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

pub fn root<InputShape, Spec>(spec: Blueprint<Spec>, axis_names: Vec<Axis>) -> Blueprint<Rooted<InputShape, Spec>>
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
) -> Blueprint<Rooted<crate::shape!(N), Spec>> {
    root::<crate::shape!(N), _>(spec, vec![Axis::Features])
}

pub fn image_input<const C: usize, const H: usize, const W: usize, Spec>(
    spec: Blueprint<Spec>,
) -> Blueprint<Rooted<crate::shape!(C, H, W), Spec>> {
    root::<crate::shape!(C, H, W), _>(
        spec,
        vec![Axis::Channels, Axis::Height, Axis::Width],
    )
}

pub fn volume_input<const C: usize, const D: usize, const H: usize, const W: usize, Spec>(
    spec: Blueprint<Spec>,
) -> Blueprint<Rooted<crate::shape!(C, D, H, W), Spec>> {
    root::<crate::shape!(C, D, H, W), _>(
        spec,
        vec![Axis::Channels, Axis::Depth, Axis::Height, Axis::Width],
    )
}

impl<InputShape, Spec> Blueprint<Rooted<InputShape, Spec>>
where
    InputShape: TensorShape + 'static,
    Spec: TransformSpec<InputShape>,
    Spec::Runtime: GraphRuntime<{ InputShape::SIZE }, { Spec::OUTPUT_SIZE }> + 'static,
    [(); InputShape::SIZE]:,
    [(); Spec::OUTPUT_SIZE]:,
{
    pub fn materialize(&self, config: InitConfig) -> Model<{ InputShape::SIZE }, { Spec::OUTPUT_SIZE }> {
        let mut ctx = MaterializeContext::new(config);
        let runtime = self.spec.spec.materialize(&mut ctx);
        Model::new(Box::new(GraphRunner::new(runtime)))
    }

    pub fn summary(&self) -> String {
        let mut lines = vec![format!(
            "input {}",
            describe_shape::<InputShape>(&self.spec.axis_names)
        )];
        <Spec as TransformSpec<InputShape>>::push_summary(&self.spec.spec, &mut lines);
        lines.join("\n")
    }

    pub fn shape_trace(&self) -> String {
        let mut lines = Vec::new();
        <Spec as TransformSpec<InputShape>>::push_shape_trace(
            &self.spec.spec,
            &self.spec.axis_names,
            &mut lines,
        );
        lines.join("\n")
    }

    pub fn parameter_count(&self) -> usize {
        let mut seen = HashSet::new();
        <Spec as TransformSpec<InputShape>>::parameter_count(&self.spec.spec, &mut seen)
    }
}

pub trait HeadsSpec<InputShape: TensorShape + 'static>: Clone + fmt::Debug + 'static {
    type Output: 'static;
    type Runtime: 'static;

    fn materialize_heads(&self, ctx: &mut MaterializeContext) -> Self::Runtime;
    fn push_summary(&self, lines: &mut Vec<String>);
    fn push_shape_trace(&self, input_axes: &[Axis], lines: &mut Vec<String>);
    fn parameter_count(&self, seen_shared: &mut HashSet<usize>) -> usize;
}

impl<InputShape, Spec> Blueprint<Rooted<InputShape, Spec>>
where
    InputShape: TensorShape + 'static,
    Spec: HeadsSpec<InputShape>,
    Spec::Runtime: PredictRuntime<{ InputShape::SIZE }, Spec::Output> + 'static,
    [(); InputShape::SIZE]:,
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
        let mut lines = vec![format!(
            "input {}",
            describe_shape::<InputShape>(&self.spec.axis_names)
        )];
        self.spec.spec.push_summary(&mut lines);
        lines.join("\n")
    }

    pub fn heads_shape_trace(&self) -> String {
        let mut lines = Vec::new();
        self.spec.spec.push_shape_trace(&self.spec.axis_names, &mut lines);
        lines.join("\n")
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

#[derive(Debug, Clone, Copy)]
pub struct DenseSpec<const OUT: usize, const BIAS: bool = true>;

#[derive(Debug, Clone, Copy)]
pub struct ReLUSpec;

#[derive(Debug, Clone, Copy)]
pub struct SigmoidSpec;

#[derive(Debug, Clone, Copy)]
pub struct FlattenSpec;

#[derive(Debug, Clone, Copy)]
pub struct ConvSpec<const OUT: usize, const KH: usize, const KW: usize, const STRIDE: usize, const PAD: usize>;

#[derive(Debug, Clone)]
pub struct SeqSpec<Left, Right> {
    left: Left,
    right: Right,
}

#[derive(Debug, Clone)]
pub struct ResidualSpec<Body> {
    body: Body,
}

#[derive(Debug, Clone)]
pub struct SumSpec<Left, Right> {
    left: Left,
    right: Right,
}

#[derive(Debug, Clone)]
pub struct ConcatSpec<Left, Right> {
    axis: Axis,
    left: Left,
    right: Right,
}

#[derive(Debug, Clone)]
pub struct SharedSpec<Spec> {
    id: usize,
    inner: Spec,
}

pub fn dense<const OUT: usize>() -> Blueprint<DenseSpec<OUT, true>> {
    Blueprint::new(DenseSpec)
}

pub fn dense_no_bias<const OUT: usize>() -> Blueprint<DenseSpec<OUT, false>> {
    Blueprint::new(DenseSpec)
}

pub fn relu() -> Blueprint<ReLUSpec> {
    Blueprint::new(ReLUSpec)
}

pub fn sigmoid() -> Blueprint<SigmoidSpec> {
    Blueprint::new(SigmoidSpec)
}

pub fn flatten() -> Blueprint<FlattenSpec> {
    Blueprint::new(FlattenSpec)
}

pub fn conv<
    const OUT: usize,
    const KH: usize,
    const KW: usize,
    const STRIDE: usize,
    const PAD: usize,
>() -> Blueprint<ConvSpec<OUT, KH, KW, STRIDE, PAD>> {
    Blueprint::new(ConvSpec)
}

pub fn residual<Spec>(body: Blueprint<Spec>) -> Blueprint<ResidualSpec<Spec>> {
    Blueprint::new(ResidualSpec { body: body.into_inner() })
}

pub fn sum<Left, Right>(left: Blueprint<Left>, right: Blueprint<Right>) -> Blueprint<SumSpec<Left, Right>> {
    Blueprint::new(SumSpec {
        left: left.into_inner(),
        right: right.into_inner(),
    })
}

pub fn concat<Left, Right>(
    axis: Axis,
    left: Blueprint<Left>,
    right: Blueprint<Right>,
) -> Blueprint<ConcatSpec<Left, Right>> {
    Blueprint::new(ConcatSpec {
        axis,
        left: left.into_inner(),
        right: right.into_inner(),
    })
}

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
pub struct MaterializeContext {
    rng: StdRng,
    shared: HashMap<usize, Box<dyn Any>>,
}

impl MaterializeContext {
    pub fn new(config: InitConfig) -> Self {
        let seed = config.seed.unwrap_or_else(|| rand::rng().random());
        Self {
            rng: StdRng::seed_from_u64(seed),
            shared: HashMap::new(),
        }
    }
}

#[doc(hidden)]
pub trait GraphRuntime<const INPUT: usize, const OUTPUT: usize>: fmt::Debug {
    type Workspace;

    fn workspace(&self) -> Self::Workspace;
    fn forward_with_workspace(&self, input: &[Float; INPUT], workspace: &mut Self::Workspace);
    fn output<'a>(workspace: &'a Self::Workspace) -> &'a [Float; OUTPUT];
    fn set_output_grad(workspace: &mut Self::Workspace, grad: &[Float; OUTPUT]);
    fn backward_with_workspace(
        &mut self,
        input: &[Float; INPUT],
        input_grad: &mut [Float; INPUT],
        workspace: &mut Self::Workspace,
    );
    fn zero_grad(&mut self);
    fn apply_gradients(
        &mut self,
        optimizer: &mut dyn Optimizer,
        slot: &mut usize,
        scale: Float,
    );
}

#[derive(Debug)]
#[doc(hidden)]
pub struct LeafWorkspace<const OUT: usize> {
    activation: Box<[Float; OUT]>,
    gradient: Box<[Float; OUT]>,
}

#[derive(Debug)]
#[doc(hidden)]
pub struct LeafRuntime<L, const INPUT: usize, const OUTPUT: usize> {
    layer: L,
}

impl<L, const INPUT: usize, const OUTPUT: usize> LeafRuntime<L, INPUT, OUTPUT> {
    fn new(layer: L) -> Self {
        Self { layer }
    }
}

impl<L, const INPUT: usize, const OUTPUT: usize> GraphRuntime<INPUT, OUTPUT>
    for LeafRuntime<L, INPUT, OUTPUT>
where
    L: Layer<INPUT, OUTPUT> + fmt::Debug + 'static,
{
    type Workspace = LeafWorkspace<OUTPUT>;

    fn workspace(&self) -> Self::Workspace {
        LeafWorkspace {
            activation: Box::new([0.0; OUTPUT]),
            gradient: Box::new([0.0; OUTPUT]),
        }
    }

    fn forward_with_workspace(&self, input: &[Float; INPUT], workspace: &mut Self::Workspace) {
        self.layer.forward(input, workspace.activation.as_mut());
    }

    fn output<'a>(workspace: &'a Self::Workspace) -> &'a [Float; OUTPUT] {
        workspace.activation.as_ref()
    }

    fn set_output_grad(workspace: &mut Self::Workspace, grad: &[Float; OUTPUT]) {
        workspace.gradient.copy_from_slice(grad);
    }

    fn backward_with_workspace(
        &mut self,
        input: &[Float; INPUT],
        input_grad: &mut [Float; INPUT],
        workspace: &mut Self::Workspace,
    ) {
        self.layer.backward(
            input,
            workspace.activation.as_ref(),
            workspace.gradient.as_ref(),
            input_grad,
        );
    }

    fn zero_grad(&mut self) {
        self.layer.zero_grad();
    }

    fn apply_gradients(
        &mut self,
        optimizer: &mut dyn Optimizer,
        slot: &mut usize,
        scale: Float,
    ) {
        self.layer.apply_gradients(optimizer, slot, scale);
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub struct SeqWorkspace<LeftWs, const MID: usize, RightWs> {
    left: LeftWs,
    mid_grad: Box<[Float; MID]>,
    right: RightWs,
}

#[derive(Debug)]
#[doc(hidden)]
pub struct SeqRuntime<Left, Right, const INPUT: usize, const MID: usize, const OUTPUT: usize> {
    left: Left,
    right: Right,
}

impl<Left, Right, const INPUT: usize, const MID: usize, const OUTPUT: usize>
    GraphRuntime<INPUT, OUTPUT> for SeqRuntime<Left, Right, INPUT, MID, OUTPUT>
where
    Left: GraphRuntime<INPUT, MID>,
    Right: GraphRuntime<MID, OUTPUT>,
{
    type Workspace = SeqWorkspace<Left::Workspace, MID, Right::Workspace>;

    fn workspace(&self) -> Self::Workspace {
        SeqWorkspace {
            left: self.left.workspace(),
            mid_grad: Box::new([0.0; MID]),
            right: self.right.workspace(),
        }
    }

    fn forward_with_workspace(&self, input: &[Float; INPUT], workspace: &mut Self::Workspace) {
        self.left.forward_with_workspace(input, &mut workspace.left);
        self.right
            .forward_with_workspace(Left::output(&workspace.left), &mut workspace.right);
    }

    fn output<'a>(workspace: &'a Self::Workspace) -> &'a [Float; OUTPUT] {
        Right::output(&workspace.right)
    }

    fn set_output_grad(workspace: &mut Self::Workspace, grad: &[Float; OUTPUT]) {
        Right::set_output_grad(&mut workspace.right, grad);
    }

    fn backward_with_workspace(
        &mut self,
        input: &[Float; INPUT],
        input_grad: &mut [Float; INPUT],
        workspace: &mut Self::Workspace,
    ) {
        self.right.backward_with_workspace(
            Left::output(&workspace.left),
            workspace.mid_grad.as_mut(),
            &mut workspace.right,
        );
        Left::set_output_grad(&mut workspace.left, workspace.mid_grad.as_ref());
        self.left.backward_with_workspace(input, input_grad, &mut workspace.left);
    }

    fn zero_grad(&mut self) {
        self.left.zero_grad();
        self.right.zero_grad();
    }

    fn apply_gradients(
        &mut self,
        optimizer: &mut dyn Optimizer,
        slot: &mut usize,
        scale: Float,
    ) {
        self.left.apply_gradients(optimizer, slot, scale);
        self.right.apply_gradients(optimizer, slot, scale);
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub struct ResidualWorkspace<InnerWs, const N: usize> {
    inner: InnerWs,
    output: Box<[Float; N]>,
    gradient: Box<[Float; N]>,
    inner_input_grad: Box<[Float; N]>,
}

#[derive(Debug)]
#[doc(hidden)]
pub struct ResidualRuntime<Inner, const N: usize> {
    inner: Inner,
}

impl<Inner, const N: usize> GraphRuntime<N, N> for ResidualRuntime<Inner, N>
where
    Inner: GraphRuntime<N, N>,
{
    type Workspace = ResidualWorkspace<Inner::Workspace, N>;

    fn workspace(&self) -> Self::Workspace {
        ResidualWorkspace {
            inner: self.inner.workspace(),
            output: Box::new([0.0; N]),
            gradient: Box::new([0.0; N]),
            inner_input_grad: Box::new([0.0; N]),
        }
    }

    fn forward_with_workspace(&self, input: &[Float; N], workspace: &mut Self::Workspace) {
        self.inner.forward_with_workspace(input, &mut workspace.inner);
        let inner = Inner::output(&workspace.inner);
        for i in 0..N {
            workspace.output[i] = input[i] + inner[i];
        }
    }

    fn output<'a>(workspace: &'a Self::Workspace) -> &'a [Float; N] {
        workspace.output.as_ref()
    }

    fn set_output_grad(workspace: &mut Self::Workspace, grad: &[Float; N]) {
        workspace.gradient.copy_from_slice(grad);
    }

    fn backward_with_workspace(
        &mut self,
        input: &[Float; N],
        input_grad: &mut [Float; N],
        workspace: &mut Self::Workspace,
    ) {
        Inner::set_output_grad(&mut workspace.inner, workspace.gradient.as_ref());
        self.inner.backward_with_workspace(
            input,
            workspace.inner_input_grad.as_mut(),
            &mut workspace.inner,
        );
        for i in 0..N {
            input_grad[i] = workspace.gradient[i] + workspace.inner_input_grad[i];
        }
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn apply_gradients(
        &mut self,
        optimizer: &mut dyn Optimizer,
        slot: &mut usize,
        scale: Float,
    ) {
        self.inner.apply_gradients(optimizer, slot, scale);
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub struct BinaryMergeWorkspace<LeftWs, RightWs, const INPUT: usize, const OUTPUT: usize> {
    left: LeftWs,
    right: RightWs,
    output: Box<[Float; OUTPUT]>,
    gradient: Box<[Float; OUTPUT]>,
    left_input_grad: Box<[Float; INPUT]>,
    right_input_grad: Box<[Float; INPUT]>,
}

#[derive(Debug)]
#[doc(hidden)]
pub struct SumRuntime<Left, Right, const INPUT: usize, const OUTPUT: usize> {
    left: Left,
    right: Right,
}

impl<Left, Right, const INPUT: usize, const OUTPUT: usize> GraphRuntime<INPUT, OUTPUT>
    for SumRuntime<Left, Right, INPUT, OUTPUT>
where
    Left: GraphRuntime<INPUT, OUTPUT>,
    Right: GraphRuntime<INPUT, OUTPUT>,
{
    type Workspace = BinaryMergeWorkspace<Left::Workspace, Right::Workspace, INPUT, OUTPUT>;

    fn workspace(&self) -> Self::Workspace {
        BinaryMergeWorkspace {
            left: self.left.workspace(),
            right: self.right.workspace(),
            output: Box::new([0.0; OUTPUT]),
            gradient: Box::new([0.0; OUTPUT]),
            left_input_grad: Box::new([0.0; INPUT]),
            right_input_grad: Box::new([0.0; INPUT]),
        }
    }

    fn forward_with_workspace(&self, input: &[Float; INPUT], workspace: &mut Self::Workspace) {
        self.left.forward_with_workspace(input, &mut workspace.left);
        self.right.forward_with_workspace(input, &mut workspace.right);
        let left = Left::output(&workspace.left);
        let right = Right::output(&workspace.right);
        for i in 0..OUTPUT {
            workspace.output[i] = left[i] + right[i];
        }
    }

    fn output<'a>(workspace: &'a Self::Workspace) -> &'a [Float; OUTPUT] {
        workspace.output.as_ref()
    }

    fn set_output_grad(workspace: &mut Self::Workspace, grad: &[Float; OUTPUT]) {
        workspace.gradient.copy_from_slice(grad);
    }

    fn backward_with_workspace(
        &mut self,
        input: &[Float; INPUT],
        input_grad: &mut [Float; INPUT],
        workspace: &mut Self::Workspace,
    ) {
        Left::set_output_grad(&mut workspace.left, workspace.gradient.as_ref());
        Right::set_output_grad(&mut workspace.right, workspace.gradient.as_ref());
        self.left.backward_with_workspace(input, workspace.left_input_grad.as_mut(), &mut workspace.left);
        self.right.backward_with_workspace(input, workspace.right_input_grad.as_mut(), &mut workspace.right);
        for i in 0..INPUT {
            input_grad[i] = workspace.left_input_grad[i] + workspace.right_input_grad[i];
        }
    }

    fn zero_grad(&mut self) {
        self.left.zero_grad();
        self.right.zero_grad();
    }

    fn apply_gradients(
        &mut self,
        optimizer: &mut dyn Optimizer,
        slot: &mut usize,
        scale: Float,
    ) {
        self.left.apply_gradients(optimizer, slot, scale);
        self.right.apply_gradients(optimizer, slot, scale);
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub struct ConcatRuntime<Left, Right, const INPUT: usize, const LEFT_OUT: usize, const RIGHT_OUT: usize> {
    left: Left,
    right: Right,
}

impl<Left, Right, const INPUT: usize, const LEFT_OUT: usize, const RIGHT_OUT: usize>
    GraphRuntime<INPUT, { LEFT_OUT + RIGHT_OUT }>
    for ConcatRuntime<Left, Right, INPUT, LEFT_OUT, RIGHT_OUT>
where
    Left: GraphRuntime<INPUT, LEFT_OUT>,
    Right: GraphRuntime<INPUT, RIGHT_OUT>,
    [(); LEFT_OUT + RIGHT_OUT]:,
{
    type Workspace =
        BinaryMergeWorkspace<Left::Workspace, Right::Workspace, INPUT, { LEFT_OUT + RIGHT_OUT }>;

    fn workspace(&self) -> Self::Workspace {
        BinaryMergeWorkspace {
            left: self.left.workspace(),
            right: self.right.workspace(),
            output: Box::new([0.0; LEFT_OUT + RIGHT_OUT]),
            gradient: Box::new([0.0; LEFT_OUT + RIGHT_OUT]),
            left_input_grad: Box::new([0.0; INPUT]),
            right_input_grad: Box::new([0.0; INPUT]),
        }
    }

    fn forward_with_workspace(&self, input: &[Float; INPUT], workspace: &mut Self::Workspace) {
        self.left.forward_with_workspace(input, &mut workspace.left);
        self.right.forward_with_workspace(input, &mut workspace.right);
        let left = Left::output(&workspace.left);
        let right = Right::output(&workspace.right);
        workspace.output[..LEFT_OUT].copy_from_slice(left);
        workspace.output[LEFT_OUT..].copy_from_slice(right);
    }

    fn output<'a>(workspace: &'a Self::Workspace) -> &'a [Float; LEFT_OUT + RIGHT_OUT] {
        workspace.output.as_ref()
    }

    fn set_output_grad(workspace: &mut Self::Workspace, grad: &[Float; LEFT_OUT + RIGHT_OUT]) {
        workspace.gradient.copy_from_slice(grad);
    }

    fn backward_with_workspace(
        &mut self,
        input: &[Float; INPUT],
        input_grad: &mut [Float; INPUT],
        workspace: &mut Self::Workspace,
    ) {
        let left_grad: &[Float; LEFT_OUT] = workspace.gradient[..LEFT_OUT].try_into().expect("left grad");
        let right_grad: &[Float; RIGHT_OUT] = workspace.gradient[LEFT_OUT..].try_into().expect("right grad");
        Left::set_output_grad(&mut workspace.left, left_grad);
        Right::set_output_grad(&mut workspace.right, right_grad);
        self.left.backward_with_workspace(input, workspace.left_input_grad.as_mut(), &mut workspace.left);
        self.right.backward_with_workspace(input, workspace.right_input_grad.as_mut(), &mut workspace.right);
        for i in 0..INPUT {
            input_grad[i] = workspace.left_input_grad[i] + workspace.right_input_grad[i];
        }
    }

    fn zero_grad(&mut self) {
        self.left.zero_grad();
        self.right.zero_grad();
    }

    fn apply_gradients(
        &mut self,
        optimizer: &mut dyn Optimizer,
        slot: &mut usize,
        scale: Float,
    ) {
        self.left.apply_gradients(optimizer, slot, scale);
        self.right.apply_gradients(optimizer, slot, scale);
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub struct SharedRuntime<Inner, const INPUT: usize, const OUTPUT: usize> {
    inner: Rc<RefCell<Inner>>,
}

impl<Inner, const INPUT: usize, const OUTPUT: usize> GraphRuntime<INPUT, OUTPUT>
    for SharedRuntime<Inner, INPUT, OUTPUT>
where
    Inner: GraphRuntime<INPUT, OUTPUT> + 'static,
{
    type Workspace = Inner::Workspace;

    fn workspace(&self) -> Self::Workspace {
        self.inner.borrow().workspace()
    }

    fn forward_with_workspace(&self, input: &[Float; INPUT], workspace: &mut Self::Workspace) {
        self.inner.borrow().forward_with_workspace(input, workspace);
    }

    fn output<'a>(workspace: &'a Self::Workspace) -> &'a [Float; OUTPUT] {
        Inner::output(workspace)
    }

    fn set_output_grad(workspace: &mut Self::Workspace, grad: &[Float; OUTPUT]) {
        Inner::set_output_grad(workspace, grad);
    }

    fn backward_with_workspace(
        &mut self,
        input: &[Float; INPUT],
        input_grad: &mut [Float; INPUT],
        workspace: &mut Self::Workspace,
    ) {
        self.inner
            .borrow_mut()
            .backward_with_workspace(input, input_grad, workspace);
    }

    fn zero_grad(&mut self) {
        self.inner.borrow_mut().zero_grad();
    }

    fn apply_gradients(
        &mut self,
        optimizer: &mut dyn Optimizer,
        slot: &mut usize,
        scale: Float,
    ) {
        self.inner
            .borrow_mut()
            .apply_gradients(optimizer, slot, scale);
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub struct GraphRunner<Runtime, const INPUT: usize, const OUTPUT: usize> {
    runtime: Runtime,
}

impl<Runtime, const INPUT: usize, const OUTPUT: usize> GraphRunner<Runtime, INPUT, OUTPUT> {
    fn new(runtime: Runtime) -> Self {
        Self { runtime }
    }
}

impl<Runtime, const INPUT: usize, const OUTPUT: usize> PredictRuntime<INPUT, [Float; OUTPUT]>
    for GraphRunner<Runtime, INPUT, OUTPUT>
where
    Runtime: GraphRuntime<INPUT, OUTPUT> + 'static,
{
    fn predict(&self, input: &[Float; INPUT]) -> [Float; OUTPUT] {
        let mut workspace = self.runtime.workspace();
        self.runtime.forward_with_workspace(input, &mut workspace);
        let mut out = [0.0; OUTPUT];
        out.copy_from_slice(Runtime::output(&workspace));
        out
    }
}

impl<Runtime, const INPUT: usize, const OUTPUT: usize> TrainRuntime<INPUT, OUTPUT>
    for GraphRunner<Runtime, INPUT, OUTPUT>
where
    Runtime: GraphRuntime<INPUT, OUTPUT> + 'static,
{
    fn fit_with_loss(
        &mut self,
        samples: &[Sample<INPUT, OUTPUT>],
        loss_fn: &dyn LossFunction<OUTPUT>,
        mut config: TrainConfig,
    ) -> Float {
        if samples.is_empty() || config.epochs == 0 {
            return 0.0;
        }

        let batch_size = config.batch_size.max(1);
        let mut workspace = self.runtime.workspace();
        let mut input_grad = [0.0; INPUT];
        let mut order = (0..samples.len()).collect::<Vec<_>>();
        let mut shuffler = config.shuffle_seed.map(StdRng::seed_from_u64);
        let mut total_loss = 0.0;
        let mut steps = 0usize;

        for _ in 0..config.epochs {
            if let Some(rng) = shuffler.as_mut() {
                use rand::seq::SliceRandom;
                order.shuffle(rng);
            }

            for batch in order.chunks(batch_size) {
                self.runtime.zero_grad();
                let mut batch_loss = 0.0;

                for &sample_idx in batch {
                    let sample = &samples[sample_idx];
                    self.runtime.forward_with_workspace(&sample.input, &mut workspace);
                    let mut grad = [0.0; OUTPUT];
                    let loss = loss_fn.loss_and_grad(
                        Runtime::output(&workspace),
                        &sample.target,
                        &mut grad,
                    );
                    Runtime::set_output_grad(&mut workspace, &grad);
                    self.runtime.backward_with_workspace(
                        &sample.input,
                        &mut input_grad,
                        &mut workspace,
                    );
                    batch_loss += loss;
                }

                config.optimizer_mut().begin_step();
                let mut slot = 0usize;
                self.runtime.apply_gradients(
                    config.optimizer_mut(),
                    &mut slot,
                    1.0 / batch.len() as Float,
                );
                total_loss += batch_loss / batch.len() as Float;
                steps += 1;
            }
        }

        total_loss / steps as Float
    }
}

impl<InputShape, const OUT: usize, const BIAS: bool> TransformSpec<InputShape> for DenseSpec<OUT, BIAS>
where
    InputShape: TensorShape + 'static,
    [(); InputShape::SIZE]:,
{
    type OutputShape = shape!(OUT);
    const OUTPUT_SIZE: usize = OUT;
    type Runtime = LeafRuntime<DenseLayer<{ InputShape::SIZE }, OUT>, { InputShape::SIZE }, OUT>;

    fn materialize(&self, ctx: &mut MaterializeContext) -> Self::Runtime {
        let layer = DenseLayer::<{ InputShape::SIZE }, OUT>::with_initializer_and_rng(
            XavierUniform,
            &mut ctx.rng,
        );
        let layer = if BIAS { layer } else { layer.without_bias() };
        LeafRuntime::new(layer)
    }

    fn parameter_count(&self, _seen_shared: &mut HashSet<usize>) -> usize {
        let bias = if BIAS { OUT } else { 0 };
        InputShape::SIZE * OUT + bias
    }

    fn output_axes(&self, _input_axes: &[Axis]) -> Box<[Axis]> {
        features_axis()
    }

    fn description(&self) -> String {
        if BIAS {
            format!("dense({OUT})")
        } else {
            format!("dense({OUT}, bias: false)")
        }
    }
}

impl<InputShape> TransformSpec<InputShape> for ReLUSpec
where
    InputShape: TensorShape + 'static,
    [(); InputShape::SIZE]:,
{
    type OutputShape = InputShape;
    const OUTPUT_SIZE: usize = InputShape::SIZE;
    type Runtime = LeafRuntime<ReLU<{ InputShape::SIZE }>, { InputShape::SIZE }, { InputShape::SIZE }>;

    fn materialize(&self, _ctx: &mut MaterializeContext) -> Self::Runtime {
        LeafRuntime::new(ReLU::<{ InputShape::SIZE }>::init())
    }

    fn parameter_count(&self, _seen_shared: &mut HashSet<usize>) -> usize {
        0
    }

    fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
        input_axes.to_vec().into_boxed_slice()
    }

    fn description(&self) -> String {
        "relu".to_string()
    }
}

impl<InputShape> TransformSpec<InputShape> for SigmoidSpec
where
    InputShape: TensorShape + 'static,
    [(); InputShape::SIZE]:,
{
    type OutputShape = InputShape;
    const OUTPUT_SIZE: usize = InputShape::SIZE;
    type Runtime =
        LeafRuntime<Sigmoid<{ InputShape::SIZE }>, { InputShape::SIZE }, { InputShape::SIZE }>;

    fn materialize(&self, _ctx: &mut MaterializeContext) -> Self::Runtime {
        LeafRuntime::new(Sigmoid::<{ InputShape::SIZE }>::init())
    }

    fn parameter_count(&self, _seen_shared: &mut HashSet<usize>) -> usize {
        0
    }

    fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
        input_axes.to_vec().into_boxed_slice()
    }

    fn description(&self) -> String {
        "sigmoid".to_string()
    }
}

impl<InputShape> TransformSpec<InputShape> for FlattenSpec
where
    InputShape: TensorShape + 'static,
    [(); InputShape::SIZE]:,
{
    type OutputShape = shape!(InputShape::SIZE);
    const OUTPUT_SIZE: usize = InputShape::SIZE;
    type Runtime =
        LeafRuntime<Flatten<{ InputShape::SIZE }>, { InputShape::SIZE }, { InputShape::SIZE }>;

    fn materialize(&self, _ctx: &mut MaterializeContext) -> Self::Runtime {
        LeafRuntime::new(Flatten::<{ InputShape::SIZE }>::init())
    }

    fn parameter_count(&self, _seen_shared: &mut HashSet<usize>) -> usize {
        0
    }

    fn output_axes(&self, _input_axes: &[Axis]) -> Box<[Axis]> {
        features_axis()
    }

    fn description(&self) -> String {
        "flatten".to_string()
    }
}

impl<const C: usize, const H: usize, const W: usize, const OUT: usize, const KH: usize, const KW: usize, const STRIDE: usize, const PAD: usize>
    TransformSpec<shape!(C, H, W)> for ConvSpec<OUT, KH, KW, STRIDE, PAD>
where
    [(); C * H * W]:,
    [(); OUT * conv_out_dim(H, PAD, KH, STRIDE) * conv_out_dim(W, PAD, KW, STRIDE)]:,
    (): ConvGeometryIsValid<H, W, KH, KW, STRIDE, PAD>,
{
    type OutputShape = shape!(OUT, conv_out_dim(H, PAD, KH, STRIDE), conv_out_dim(W, PAD, KW, STRIDE));
    const OUTPUT_SIZE: usize =
        OUT * conv_out_dim(H, PAD, KH, STRIDE) * conv_out_dim(W, PAD, KW, STRIDE);
    type Runtime = LeafRuntime<
        Conv<W, H, C, KH, KW, OUT, STRIDE, PAD>,
        { C * H * W },
        { OUT * conv_out_dim(H, PAD, KH, STRIDE) * conv_out_dim(W, PAD, KW, STRIDE) },
    >;

    fn materialize(&self, ctx: &mut MaterializeContext) -> Self::Runtime {
        LeafRuntime::new(
            Conv::<W, H, C, KH, KW, OUT, STRIDE, PAD>::with_initializer_and_rng(
                XavierUniform,
                &mut ctx.rng,
            ),
        )
    }

    fn parameter_count(&self, _seen_shared: &mut HashSet<usize>) -> usize {
        OUT * KH * KW * C + OUT
    }

    fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
        input_axes.to_vec().into_boxed_slice()
    }

    fn description(&self) -> String {
        if KH == KW {
            format!("conv({OUT}, kernel: {KH}, stride: {STRIDE}, pad: {PAD})")
        } else {
            format!("conv({OUT}, kernel: ({KH}, {KW}), stride: {STRIDE}, pad: {PAD})")
        }
    }
}

impl<InputShape, Left, Right> TransformSpec<InputShape> for SeqSpec<Left, Right>
where
    InputShape: TensorShape + 'static,
    Left: TransformSpec<InputShape>,
    Right: TransformSpec<Left::OutputShape>,
    [(); InputShape::SIZE]:,
    [(); Left::OUTPUT_SIZE]:,
    [(); Right::OUTPUT_SIZE]:,
    [(); <Left::OutputShape as TensorShape>::SIZE]:,
    [(); <Right::OutputShape as TensorShape>::SIZE]:,
{
    type OutputShape = Right::OutputShape;
    const OUTPUT_SIZE: usize = Right::OUTPUT_SIZE;
    type Runtime = SeqRuntime<
        Left::Runtime,
        Right::Runtime,
        { InputShape::SIZE },
        { Left::OUTPUT_SIZE },
        { Right::OUTPUT_SIZE },
    >;

    fn materialize(&self, ctx: &mut MaterializeContext) -> Self::Runtime {
        SeqRuntime {
            left: self.left.materialize(ctx),
            right: self.right.materialize(ctx),
        }
    }

    fn push_summary(&self, lines: &mut Vec<String>) {
        <Left as TransformSpec<InputShape>>::push_summary(&self.left, lines);
        <Right as TransformSpec<Left::OutputShape>>::push_summary(&self.right, lines);
    }

    fn push_shape_trace(&self, input_axes: &[Axis], lines: &mut Vec<String>) -> Box<[Axis]> {
        let mid_axes = <Left as TransformSpec<InputShape>>::push_shape_trace(
            &self.left,
            input_axes,
            lines,
        );
        <Right as TransformSpec<Left::OutputShape>>::push_shape_trace(&self.right, &mid_axes, lines)
    }

    fn parameter_count(&self, seen_shared: &mut HashSet<usize>) -> usize {
        <Left as TransformSpec<InputShape>>::parameter_count(&self.left, seen_shared)
            + <Right as TransformSpec<Left::OutputShape>>::parameter_count(&self.right, seen_shared)
    }

    fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
        let mid = self.left.output_axes(input_axes);
        self.right.output_axes(&mid)
    }

    fn description(&self) -> String {
        "sequence".to_string()
    }
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

impl<InputShape, Body> TransformSpec<InputShape> for ResidualSpec<Body>
where
    InputShape: TensorShape + 'static,
    Body: ShapePreserving<InputShape>,
    [(); InputShape::SIZE]:,
{
    type OutputShape = InputShape;
    const OUTPUT_SIZE: usize = InputShape::SIZE;
    type Runtime = ResidualRuntime<Body::Runtime, { InputShape::SIZE }>;

    fn materialize(&self, ctx: &mut MaterializeContext) -> Self::Runtime {
        ResidualRuntime {
            inner: self.body.materialize(ctx),
        }
    }

    fn push_summary(&self, lines: &mut Vec<String>) {
        lines.push("residual".to_string());
        <Body as TransformSpec<InputShape>>::push_summary(&self.body, lines);
    }

    fn push_shape_trace(&self, input_axes: &[Axis], lines: &mut Vec<String>) -> Box<[Axis]> {
        lines.push(format!(
            "residual: {} -> {}",
            describe_shape::<InputShape>(input_axes),
            describe_shape::<InputShape>(input_axes)
        ));
        <Body as TransformSpec<InputShape>>::push_shape_trace(&self.body, input_axes, lines);
        input_axes.to_vec().into_boxed_slice()
    }

    fn parameter_count(&self, seen_shared: &mut HashSet<usize>) -> usize {
        <Body as TransformSpec<InputShape>>::parameter_count(&self.body, seen_shared)
    }

    fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
        input_axes.to_vec().into_boxed_slice()
    }

    fn description(&self) -> String {
        "residual".to_string()
    }
}

impl<InputShape, Left, Right> TransformSpec<InputShape> for SumSpec<Left, Right>
where
    InputShape: TensorShape + 'static,
    Left: TransformSpec<InputShape>,
    Right: TransformSpec<InputShape, OutputShape = Left::OutputShape>,
    [(); InputShape::SIZE]:,
    [(); Left::OUTPUT_SIZE]:,
    [(); <Left::OutputShape as TensorShape>::SIZE]:,
{
    type OutputShape = Left::OutputShape;
    const OUTPUT_SIZE: usize = Left::OUTPUT_SIZE;
    type Runtime = SumRuntime<
        Left::Runtime,
        Right::Runtime,
        { InputShape::SIZE },
        { Left::OUTPUT_SIZE },
    >;

    fn materialize(&self, ctx: &mut MaterializeContext) -> Self::Runtime {
        SumRuntime {
            left: self.left.materialize(ctx),
            right: self.right.materialize(ctx),
        }
    }

    fn parameter_count(&self, seen_shared: &mut HashSet<usize>) -> usize {
        <Left as TransformSpec<InputShape>>::parameter_count(&self.left, seen_shared)
            + <Right as TransformSpec<InputShape>>::parameter_count(&self.right, seen_shared)
    }

    fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
        self.left.output_axes(input_axes)
    }

    fn description(&self) -> String {
        "sum".to_string()
    }
}

#[doc(hidden)]
pub trait ConcatAlong<InputShape: TensorShape + 'static, LeftOut: TensorShape, RightOut: TensorShape> {
    type OutputShape: TensorShape;
    fn axis_ok(axis: Axis) -> bool;
}

impl<const A: usize, const B: usize> ConcatAlong<shape!(A), shape!(A), shape!(B)> for ()
where
    [(); A + B]:,
{
    type OutputShape = shape!(A + B);
    fn axis_ok(axis: Axis) -> bool {
        axis == Axis::Features
    }
}

impl<const C1: usize, const C2: usize, const H: usize, const W: usize>
    ConcatAlong<shape!(C1, H, W), shape!(C1, H, W), shape!(C2, H, W)> for ()
where
    [(); C1 * H * W]:,
    [(); C2 * H * W]:,
    [(); (C1 + C2) * H * W]:,
{
    type OutputShape = shape!(C1 + C2, H, W);
    fn axis_ok(axis: Axis) -> bool {
        axis == Axis::Channels
    }
}

impl<InputShape, Left, Right> TransformSpec<InputShape> for ConcatSpec<Left, Right>
where
    InputShape: TensorShape + 'static,
    Left: TransformSpec<InputShape>,
    Right: TransformSpec<InputShape>,
    (): ConcatAlong<InputShape, Left::OutputShape, Right::OutputShape>,
    [(); InputShape::SIZE]:,
    [(); Left::OUTPUT_SIZE]:,
    [(); Right::OUTPUT_SIZE]:,
    [(); <Left::OutputShape as TensorShape>::SIZE]:,
    [(); <Right::OutputShape as TensorShape>::SIZE]:,
    [(); <<() as ConcatAlong<InputShape, Left::OutputShape, Right::OutputShape>>::OutputShape as TensorShape>::SIZE]:,
{
    type OutputShape = <() as ConcatAlong<InputShape, Left::OutputShape, Right::OutputShape>>::OutputShape;
    const OUTPUT_SIZE: usize = Left::OUTPUT_SIZE + Right::OUTPUT_SIZE;
    type Runtime = ConcatRuntime<
        Left::Runtime,
        Right::Runtime,
        { InputShape::SIZE },
        { Left::OUTPUT_SIZE },
        { Right::OUTPUT_SIZE },
    >;

    fn materialize(&self, ctx: &mut MaterializeContext) -> Self::Runtime {
        assert!(
            <() as ConcatAlong<InputShape, Left::OutputShape, Right::OutputShape>>::axis_ok(self.axis),
            "unsupported concat axis for the current shapes"
        );
        ConcatRuntime {
            left: self.left.materialize(ctx),
            right: self.right.materialize(ctx),
        }
    }

    fn parameter_count(&self, seen_shared: &mut HashSet<usize>) -> usize {
        <Left as TransformSpec<InputShape>>::parameter_count(&self.left, seen_shared)
            + <Right as TransformSpec<InputShape>>::parameter_count(&self.right, seen_shared)
    }

    fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
        input_axes.to_vec().into_boxed_slice()
    }

    fn description(&self) -> String {
        format!("concat({})", self.axis.as_str())
    }
}

impl<InputShape, Spec> TransformSpec<InputShape> for SharedSpec<Spec>
where
    InputShape: TensorShape + 'static,
    Spec: TransformSpec<InputShape>,
    Spec::Runtime: 'static,
    [(); InputShape::SIZE]:,
    [(); Spec::OUTPUT_SIZE]:,
    [(); <Spec::OutputShape as TensorShape>::SIZE]:,
{
    type OutputShape = Spec::OutputShape;
    const OUTPUT_SIZE: usize = Spec::OUTPUT_SIZE;
    type Runtime =
        SharedRuntime<Spec::Runtime, { InputShape::SIZE }, { Spec::OUTPUT_SIZE }>;

    fn materialize(&self, ctx: &mut MaterializeContext) -> Self::Runtime {
        if let Some(existing) = ctx.shared.get(&self.id) {
            let rc = existing
                .downcast_ref::<Rc<RefCell<Spec::Runtime>>>()
                .expect("shared runtime type mismatch")
                .clone();
            return SharedRuntime { inner: rc };
        }

        let runtime = self.inner.materialize(ctx);
        let rc = Rc::new(RefCell::new(runtime));
        ctx.shared.insert(self.id, Box::new(rc.clone()));
        SharedRuntime { inner: rc }
    }

    fn parameter_count(&self, seen_shared: &mut HashSet<usize>) -> usize {
        if seen_shared.insert(self.id) {
            <Spec as TransformSpec<InputShape>>::parameter_count(&self.inner, seen_shared)
        } else {
            0
        }
    }

    fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
        self.inner.output_axes(input_axes)
    }

    fn description(&self) -> String {
        format!("share({})", self.id)
    }
}
