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
use crate::{ConvKernelFitsInput, Float, Sample, shape};

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
pub struct RepeatStageSpec<Spec> {
    inner: Spec,
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

pub fn identity() -> Blueprint<IdentitySpec> {
    Blueprint::new(IdentitySpec)
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
    Blueprint::new(ResidualSpec {
        body: body.into_inner(),
    })
}

pub fn sum<Left, Right>(
    left: Blueprint<Left>,
    right: Blueprint<Right>,
) -> Blueprint<SumSpec<Left, Right>> {
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
pub fn repeat_stage<Spec>(blueprint: Blueprint<Spec>) -> Blueprint<RepeatStageSpec<Spec>> {
    Blueprint::new(RepeatStageSpec {
        inner: blueprint.into_inner(),
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
pub trait GraphRuntime: fmt::Debug {
    fn forward(&self, input: &[Float]) -> Vec<Float>;
    fn backward(&mut self, input: &[Float], output: &[Float], output_grad: &[Float]) -> Vec<Float>;
    fn zero_grad(&mut self);
    fn apply_gradients(&mut self, optimizer: &mut dyn Optimizer, slot: &mut usize, scale: Float);
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

impl<L, const INPUT: usize, const OUTPUT: usize> GraphRuntime for LeafRuntime<L, INPUT, OUTPUT>
where
    L: Layer<INPUT, OUTPUT> + fmt::Debug + 'static,
{
    fn forward(&self, input: &[Float]) -> Vec<Float> {
        let input: &[Float; INPUT] = input
            .try_into()
            .expect("leaf runtime input length must match the layer input");
        let mut output = [0.0; OUTPUT];
        self.layer.forward(input, &mut output);
        output.to_vec()
    }

    fn backward(&mut self, input: &[Float], output: &[Float], output_grad: &[Float]) -> Vec<Float> {
        let input: &[Float; INPUT] = input
            .try_into()
            .expect("leaf runtime input length must match the layer input");
        let output: &[Float; OUTPUT] = output
            .try_into()
            .expect("leaf runtime output length must match the layer output");
        let output_grad: &[Float; OUTPUT] = output_grad
            .try_into()
            .expect("leaf runtime gradient length must match the layer output");
        let mut input_grad = [0.0; INPUT];
        self.layer
            .backward(input, output, output_grad, &mut input_grad);
        input_grad.to_vec()
    }

    fn zero_grad(&mut self) {
        self.layer.zero_grad();
    }

    fn apply_gradients(&mut self, optimizer: &mut dyn Optimizer, slot: &mut usize, scale: Float) {
        self.layer.apply_gradients(optimizer, slot, scale);
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub struct SeqRuntime<Left, Right> {
    left: Left,
    right: Right,
}

impl<Left, Right> GraphRuntime for SeqRuntime<Left, Right>
where
    Left: GraphRuntime,
    Right: GraphRuntime,
{
    fn forward(&self, input: &[Float]) -> Vec<Float> {
        let mid = self.left.forward(input);
        self.right.forward(&mid)
    }

    fn backward(
        &mut self,
        input: &[Float],
        _output: &[Float],
        output_grad: &[Float],
    ) -> Vec<Float> {
        let mid = self.left.forward(input);
        let right_out = self.right.forward(&mid);
        let mid_grad = self.right.backward(&mid, &right_out, output_grad);
        self.left.backward(input, &mid, &mid_grad)
    }

    fn zero_grad(&mut self) {
        self.left.zero_grad();
        self.right.zero_grad();
    }

    fn apply_gradients(&mut self, optimizer: &mut dyn Optimizer, slot: &mut usize, scale: Float) {
        self.left.apply_gradients(optimizer, slot, scale);
        self.right.apply_gradients(optimizer, slot, scale);
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub struct ResidualRuntime<Inner> {
    inner: Inner,
}

impl<Inner> GraphRuntime for ResidualRuntime<Inner>
where
    Inner: GraphRuntime,
{
    fn forward(&self, input: &[Float]) -> Vec<Float> {
        let inner = self.inner.forward(input);
        assert_eq!(
            inner.len(),
            input.len(),
            "residual requires the body to preserve shape",
        );
        input.iter().zip(inner.iter()).map(|(x, y)| x + y).collect()
    }

    fn backward(
        &mut self,
        input: &[Float],
        _output: &[Float],
        output_grad: &[Float],
    ) -> Vec<Float> {
        let inner_out = self.inner.forward(input);
        let inner_input_grad = self.inner.backward(input, &inner_out, output_grad);
        output_grad
            .iter()
            .zip(inner_input_grad.iter())
            .map(|(skip, body)| skip + body)
            .collect()
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn apply_gradients(&mut self, optimizer: &mut dyn Optimizer, slot: &mut usize, scale: Float) {
        self.inner.apply_gradients(optimizer, slot, scale);
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub struct SumRuntime<Left, Right> {
    left: Left,
    right: Right,
}

impl<Left, Right> GraphRuntime for SumRuntime<Left, Right>
where
    Left: GraphRuntime,
    Right: GraphRuntime,
{
    fn forward(&self, input: &[Float]) -> Vec<Float> {
        let left = self.left.forward(input);
        let right = self.right.forward(input);
        assert_eq!(
            left.len(),
            right.len(),
            "sum requires matching branch shapes"
        );
        left.iter().zip(right.iter()).map(|(l, r)| l + r).collect()
    }

    fn backward(
        &mut self,
        input: &[Float],
        _output: &[Float],
        output_grad: &[Float],
    ) -> Vec<Float> {
        let left_out = self.left.forward(input);
        let right_out = self.right.forward(input);
        let left_input_grad = self.left.backward(input, &left_out, output_grad);
        let right_input_grad = self.right.backward(input, &right_out, output_grad);
        left_input_grad
            .iter()
            .zip(right_input_grad.iter())
            .map(|(l, r)| l + r)
            .collect()
    }

    fn zero_grad(&mut self) {
        self.left.zero_grad();
        self.right.zero_grad();
    }

    fn apply_gradients(&mut self, optimizer: &mut dyn Optimizer, slot: &mut usize, scale: Float) {
        self.left.apply_gradients(optimizer, slot, scale);
        self.right.apply_gradients(optimizer, slot, scale);
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub struct ConcatRuntime<Left, Right> {
    left: Left,
    right: Right,
}

impl<Left, Right> GraphRuntime for ConcatRuntime<Left, Right>
where
    Left: GraphRuntime,
    Right: GraphRuntime,
{
    fn forward(&self, input: &[Float]) -> Vec<Float> {
        let mut output = self.left.forward(input);
        output.extend(self.right.forward(input));
        output
    }

    fn backward(
        &mut self,
        input: &[Float],
        _output: &[Float],
        output_grad: &[Float],
    ) -> Vec<Float> {
        let left_out = self.left.forward(input);
        let right_out = self.right.forward(input);
        let split = left_out.len();
        assert_eq!(
            output_grad.len(),
            split + right_out.len(),
            "concat gradient length must match the merged branch outputs",
        );
        let left_input_grad = self.left.backward(input, &left_out, &output_grad[..split]);
        let right_input_grad = self
            .right
            .backward(input, &right_out, &output_grad[split..]);
        left_input_grad
            .iter()
            .zip(right_input_grad.iter())
            .map(|(l, r)| l + r)
            .collect()
    }

    fn zero_grad(&mut self) {
        self.left.zero_grad();
        self.right.zero_grad();
    }

    fn apply_gradients(&mut self, optimizer: &mut dyn Optimizer, slot: &mut usize, scale: Float) {
        self.left.apply_gradients(optimizer, slot, scale);
        self.right.apply_gradients(optimizer, slot, scale);
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub struct SharedRuntime<Inner> {
    inner: Rc<RefCell<Inner>>,
}

impl<Inner> GraphRuntime for SharedRuntime<Inner>
where
    Inner: GraphRuntime + 'static,
{
    fn forward(&self, input: &[Float]) -> Vec<Float> {
        self.inner.borrow().forward(input)
    }

    fn backward(&mut self, input: &[Float], output: &[Float], output_grad: &[Float]) -> Vec<Float> {
        self.inner.borrow_mut().backward(input, output, output_grad)
    }

    fn zero_grad(&mut self) {
        self.inner.borrow_mut().zero_grad();
    }

    fn apply_gradients(&mut self, optimizer: &mut dyn Optimizer, slot: &mut usize, scale: Float) {
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
    Runtime: GraphRuntime + 'static,
{
    fn predict(&self, input: &[Float; INPUT]) -> [Float; OUTPUT] {
        self.runtime
            .forward(input)
            .try_into()
            .expect("graph runtime output length must match the model output")
    }
}

impl<Runtime, const INPUT: usize, const OUTPUT: usize> TrainRuntime<INPUT, OUTPUT>
    for GraphRunner<Runtime, INPUT, OUTPUT>
where
    Runtime: GraphRuntime + 'static,
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
                    let output = self.runtime.forward(&sample.input);
                    let output_arr: [Float; OUTPUT] = output
                        .as_slice()
                        .try_into()
                        .expect("graph runtime output length must match the loss output");
                    let mut grad = [0.0; OUTPUT];
                    let loss = loss_fn.loss_and_grad(&output_arr, &sample.target, &mut grad);
                    let _ = self.runtime.backward(&sample.input, &output, &grad);
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

impl<const N: usize, const OUT: usize, const BIAS: bool> DenseExpectsFlatInput<OUT, BIAS>
    for shape!(N)
where
    [(); N]:,
{
    type Runtime = LeafRuntime<DenseLayer<N, OUT>, N, OUT>;

    fn materialize_dense(ctx: &mut MaterializeContext) -> Self::Runtime {
        let layer = DenseLayer::<N, OUT>::with_initializer_and_rng(XavierUniform, &mut ctx.rng);
        let layer = if BIAS { layer } else { layer.without_bias() };
        LeafRuntime::new(layer)
    }

    fn dense_parameter_count() -> usize {
        let bias = if BIAS { OUT } else { 0 };
        N * OUT + bias
    }
}

impl<InputShape, const OUT: usize, const BIAS: bool> TransformSpec<InputShape>
    for DenseSpec<OUT, BIAS>
where
    InputShape: DenseExpectsFlatInput<OUT, BIAS>,
{
    type OutputShape = shape!(OUT);
    const OUTPUT_SIZE: usize = OUT;
    type Runtime = InputShape::Runtime;

    fn materialize(&self, ctx: &mut MaterializeContext) -> Self::Runtime {
        InputShape::materialize_dense(ctx)
    }

    fn parameter_count(&self, _seen_shared: &mut HashSet<usize>) -> usize {
        InputShape::dense_parameter_count()
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

macro_rules! impl_pointwise_transform {
    ($spec:ty, $layer:ident, $desc:literal) => {
        impl<const N: usize> TransformSpec<shape!(N)> for $spec
        where
            [(); N]:,
        {
            type OutputShape = shape!(N);
            const OUTPUT_SIZE: usize = N;
            type Runtime = LeafRuntime<$layer<N>, N, N>;

            fn materialize(&self, _ctx: &mut MaterializeContext) -> Self::Runtime {
                LeafRuntime::new($layer::<N>::init())
            }

            fn parameter_count(&self, _seen_shared: &mut HashSet<usize>) -> usize {
                0
            }

            fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
                input_axes.to_vec().into_boxed_slice()
            }

            fn description(&self) -> String {
                $desc.to_string()
            }
        }

        impl<const A: usize, const B: usize> TransformSpec<shape!(A, B)> for $spec
        where
            [(); A * B]:,
        {
            type OutputShape = shape!(A, B);
            const OUTPUT_SIZE: usize = A * B;
            type Runtime = LeafRuntime<$layer<{ A * B }>, { A * B }, { A * B }>;

            fn materialize(&self, _ctx: &mut MaterializeContext) -> Self::Runtime {
                LeafRuntime::new($layer::<{ A * B }>::init())
            }

            fn parameter_count(&self, _seen_shared: &mut HashSet<usize>) -> usize {
                0
            }

            fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
                input_axes.to_vec().into_boxed_slice()
            }

            fn description(&self) -> String {
                $desc.to_string()
            }
        }

        impl<const C: usize, const H: usize, const W: usize> TransformSpec<shape!(C, H, W)>
            for $spec
        where
            [(); C * H * W]:,
        {
            type OutputShape = shape!(C, H, W);
            const OUTPUT_SIZE: usize = C * H * W;
            type Runtime = LeafRuntime<$layer<{ C * H * W }>, { C * H * W }, { C * H * W }>;

            fn materialize(&self, _ctx: &mut MaterializeContext) -> Self::Runtime {
                LeafRuntime::new($layer::<{ C * H * W }>::init())
            }

            fn parameter_count(&self, _seen_shared: &mut HashSet<usize>) -> usize {
                0
            }

            fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
                input_axes.to_vec().into_boxed_slice()
            }

            fn description(&self) -> String {
                $desc.to_string()
            }
        }

        impl<const C: usize, const D: usize, const H: usize, const W: usize>
            TransformSpec<shape!(C, D, H, W)> for $spec
        where
            [(); C * D * H * W]:,
        {
            type OutputShape = shape!(C, D, H, W);
            const OUTPUT_SIZE: usize = C * D * H * W;
            type Runtime =
                LeafRuntime<$layer<{ C * D * H * W }>, { C * D * H * W }, { C * D * H * W }>;

            fn materialize(&self, _ctx: &mut MaterializeContext) -> Self::Runtime {
                LeafRuntime::new($layer::<{ C * D * H * W }>::init())
            }

            fn parameter_count(&self, _seen_shared: &mut HashSet<usize>) -> usize {
                0
            }

            fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
                input_axes.to_vec().into_boxed_slice()
            }

            fn description(&self) -> String {
                $desc.to_string()
            }
        }
    };
}

macro_rules! impl_flatten_transform {
    ([$($gen:tt)*] $shape:ty, $size:expr) => {
        impl<$($gen)*> TransformSpec<$shape> for FlattenSpec
        where
            [(); { $size }]:,
        {
            type OutputShape = shape!({ $size });
            const OUTPUT_SIZE: usize = $size;
            type Runtime = LeafRuntime<Flatten<{ $size }>, { $size }, { $size }>;

            fn materialize(&self, _ctx: &mut MaterializeContext) -> Self::Runtime {
                LeafRuntime::new(Flatten::<{ $size }>::init())
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
    };
}

impl_pointwise_transform!(ReLUSpec, ReLU, "relu");
impl_pointwise_transform!(SigmoidSpec, Sigmoid, "sigmoid");
impl_pointwise_transform!(IdentitySpec, Flatten, "identity");

impl_flatten_transform!([const N: usize] shape!(N), N);
impl_flatten_transform!([const A: usize, const B: usize] shape!(A, B), A * B);
impl_flatten_transform!(
    [const C: usize, const H: usize, const W: usize]
    shape!(C, H, W),
    C * H * W
);
impl_flatten_transform!(
    [const C: usize, const D: usize, const H: usize, const W: usize]
    shape!(C, D, H, W),
    C * D * H * W
);

impl<
    const C: usize,
    const H: usize,
    const W: usize,
    const OUT: usize,
    const KH: usize,
    const KW: usize,
    const STRIDE: usize,
    const PAD: usize,
> TransformSpec<shape!(C, H, W)> for ConvSpec<OUT, KH, KW, STRIDE, PAD>
where
    [(); C * H * W]:,
    [(); OUT * conv_out_dim(H, PAD, KH, STRIDE) * conv_out_dim(W, PAD, KW, STRIDE)]:,
    (): ConvKernelFitsInput<H, W, KH, KW, STRIDE, PAD>,
{
    type OutputShape = shape!(
        OUT,
        conv_out_dim(H, PAD, KH, STRIDE),
        conv_out_dim(W, PAD, KW, STRIDE)
    );
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

macro_rules! impl_seq_compatible {
    ([$($gen:tt)*] $shape:ty, $size:expr) => {
        impl<$($gen)*, Left, Right> SeqCompatible<$shape, Left, Right> for $shape
        where
            Left: TransformSpec<$shape>,
            Right: TransformSpec<Left::OutputShape>,
            [(); { $size }]:,
        {
            type OutputShape = Right::OutputShape;
            const OUTPUT_SIZE: usize = Right::OUTPUT_SIZE;
            type Runtime = SeqRuntime<Left::Runtime, Right::Runtime>;

            fn materialize_seq(
                left: &Left,
                right: &Right,
                ctx: &mut MaterializeContext,
            ) -> Self::Runtime {
                SeqRuntime {
                    left: left.materialize(ctx),
                    right: right.materialize(ctx),
                }
            }
        }
    };
}

impl_seq_compatible!([const N: usize] shape!(N), N);
impl_seq_compatible!([const A: usize, const B: usize] shape!(A, B), A * B);
impl_seq_compatible!(
    [const C: usize, const H: usize, const W: usize]
    shape!(C, H, W),
    C * H * W
);
impl_seq_compatible!(
    [const C: usize, const D: usize, const H: usize, const W: usize]
    shape!(C, D, H, W),
    C * D * H * W
);

impl<InputShape, Left, Right> TransformSpec<InputShape> for SeqSpec<Left, Right>
where
    InputShape: TensorShape + 'static,
    Left: TransformSpec<InputShape>,
    Right: TransformSpec<Left::OutputShape>,
    InputShape: SeqCompatible<InputShape, Left, Right>,
{
    type OutputShape = <InputShape as SeqCompatible<InputShape, Left, Right>>::OutputShape;
    const OUTPUT_SIZE: usize = <InputShape as SeqCompatible<InputShape, Left, Right>>::OUTPUT_SIZE;
    type Runtime = <InputShape as SeqCompatible<InputShape, Left, Right>>::Runtime;

    fn materialize(&self, ctx: &mut MaterializeContext) -> Self::Runtime {
        <InputShape as SeqCompatible<InputShape, Left, Right>>::materialize_seq(
            &self.left,
            &self.right,
            ctx,
        )
    }

    fn push_summary(&self, lines: &mut Vec<String>) {
        <Left as TransformSpec<InputShape>>::push_summary(&self.left, lines);
        <Right as TransformSpec<Left::OutputShape>>::push_summary(&self.right, lines);
    }

    fn push_shape_trace(&self, input_axes: &[Axis], lines: &mut Vec<String>) -> Box<[Axis]> {
        let mid_axes =
            <Left as TransformSpec<InputShape>>::push_shape_trace(&self.left, input_axes, lines);
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
    type Runtime = ResidualRuntime<Body::Runtime>;

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
{
    type OutputShape = Left::OutputShape;
    const OUTPUT_SIZE: usize = Left::OUTPUT_SIZE;
    type Runtime = SumRuntime<Left::Runtime, Right::Runtime>;

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
pub trait ConcatAlong<
    InputShape: TensorShape + 'static,
    LeftOut: TensorShape,
    RightOut: TensorShape,
>
{
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
{
    type OutputShape =
        <() as ConcatAlong<InputShape, Left::OutputShape, Right::OutputShape>>::OutputShape;
    const OUTPUT_SIZE: usize = Left::OUTPUT_SIZE + Right::OUTPUT_SIZE;
    type Runtime = ConcatRuntime<Left::Runtime, Right::Runtime>;

    fn materialize(&self, ctx: &mut MaterializeContext) -> Self::Runtime {
        assert!(
            <() as ConcatAlong<InputShape, Left::OutputShape, Right::OutputShape>>::axis_ok(
                self.axis
            ),
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

impl<InputShape, Spec> TransformSpec<InputShape> for RepeatStageSpec<Spec>
where
    InputShape: TensorShape + 'static,
    Spec: ShapePreserving<InputShape>,
    [(); InputShape::SIZE]:,
{
    type OutputShape = InputShape;
    const OUTPUT_SIZE: usize = InputShape::SIZE;
    type Runtime = Spec::Runtime;

    fn materialize(&self, ctx: &mut MaterializeContext) -> Self::Runtime {
        self.inner.materialize(ctx)
    }

    fn push_summary(&self, lines: &mut Vec<String>) {
        <Spec as TransformSpec<InputShape>>::push_summary(&self.inner, lines);
    }

    fn push_shape_trace(&self, input_axes: &[Axis], lines: &mut Vec<String>) -> Box<[Axis]> {
        <Spec as TransformSpec<InputShape>>::push_shape_trace(&self.inner, input_axes, lines)
    }

    fn parameter_count(&self, seen_shared: &mut HashSet<usize>) -> usize {
        <Spec as TransformSpec<InputShape>>::parameter_count(&self.inner, seen_shared)
    }

    fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
        self.inner.output_axes(input_axes)
    }

    fn description(&self) -> String {
        self.inner.description()
    }
}

impl<InputShape, Spec> TransformSpec<InputShape> for SharedSpec<Spec>
where
    InputShape: TensorShape + 'static,
    Spec: TransformSpec<InputShape>,
    Spec::Runtime: 'static,
    [(); InputShape::SIZE]:,
    [(); Spec::OUTPUT_SIZE]:,
{
    type OutputShape = Spec::OutputShape;
    const OUTPUT_SIZE: usize = Spec::OUTPUT_SIZE;
    type Runtime = SharedRuntime<Spec::Runtime>;

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
