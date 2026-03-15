use crate::{Float, Sample};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};

pub trait Initializer {
    fn fill<R: Rng + ?Sized>(
        &self,
        values: &mut [Float],
        fan_in: usize,
        fan_out: usize,
        rng: &mut R,
    );
}

#[derive(Debug, Clone, Copy)]
pub struct Uniform {
    pub low: Float,
    pub high: Float,
}

impl Uniform {
    pub const fn new(low: Float, high: Float) -> Self {
        Self { low, high }
    }
}

impl Initializer for Uniform {
    fn fill<R: Rng + ?Sized>(
        &self,
        values: &mut [Float],
        _fan_in: usize,
        _fan_out: usize,
        rng: &mut R,
    ) {
        for value in values {
            *value = rng.random_range(self.low..self.high);
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct XavierUniform;

impl Initializer for XavierUniform {
    fn fill<R: Rng + ?Sized>(
        &self,
        values: &mut [Float],
        fan_in: usize,
        fan_out: usize,
        rng: &mut R,
    ) {
        let denom = (fan_in + fan_out).max(1) as Float;
        let bound = (6.0 / denom).sqrt();
        Uniform::new(-bound, bound).fill(values, fan_in, fan_out, rng);
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct KaimingUniform;

impl Initializer for KaimingUniform {
    fn fill<R: Rng + ?Sized>(
        &self,
        values: &mut [Float],
        fan_in: usize,
        fan_out: usize,
        rng: &mut R,
    ) {
        let denom = fan_in.max(1) as Float;
        let bound = (6.0 / denom).sqrt();
        Uniform::new(-bound, bound).fill(values, fan_in, fan_out, rng);
    }
}

pub trait Optimizer {
    fn begin_step(&mut self) {}
    fn update_parameter(
        &mut self,
        slot: usize,
        params: &mut [Float],
        grads: &[Float],
        scale: Float,
    );
}

#[derive(Debug, Clone, Copy)]
pub struct Sgd {
    pub lr: Float,
    pub weight_decay: Float,
}

impl Sgd {
    pub const fn new(lr: Float) -> Self {
        Self {
            lr,
            weight_decay: 0.0,
        }
    }

    pub const fn with_weight_decay(mut self, weight_decay: Float) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for Sgd {
    fn update_parameter(
        &mut self,
        _slot: usize,
        params: &mut [Float],
        grads: &[Float],
        scale: Float,
    ) {
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            let update = *grad * scale + self.weight_decay * *param;
            *param -= self.lr * update;
        }
    }
}

#[derive(Debug, Clone)]
pub struct Adam {
    pub lr: Float,
    pub beta1: Float,
    pub beta2: Float,
    pub epsilon: Float,
    pub weight_decay: Float,
    step: usize,
    first_moment: Vec<Box<[Float]>>,
    second_moment: Vec<Box<[Float]>>,
}

impl Adam {
    pub fn new(lr: Float) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            step: 0,
            first_moment: Vec::new(),
            second_moment: Vec::new(),
        }
    }

    pub const fn with_weight_decay(mut self, weight_decay: Float) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    fn ensure_slot(&mut self, slot: usize, len: usize) {
        while self.first_moment.len() <= slot {
            self.first_moment.push(Vec::new().into_boxed_slice());
            self.second_moment.push(Vec::new().into_boxed_slice());
        }
        if self.first_moment[slot].len() != len {
            self.first_moment[slot] = vec![0.0; len].into_boxed_slice();
            self.second_moment[slot] = vec![0.0; len].into_boxed_slice();
        }
    }
}

impl Optimizer for Adam {
    fn begin_step(&mut self) {
        self.step += 1;
    }

    fn update_parameter(
        &mut self,
        slot: usize,
        params: &mut [Float],
        grads: &[Float],
        scale: Float,
    ) {
        self.ensure_slot(slot, params.len());
        let bias_correction1 = 1.0 - self.beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step as i32);

        let first = &mut self.first_moment[slot];
        let second = &mut self.second_moment[slot];
        for i in 0..params.len() {
            let grad = grads[i] * scale + self.weight_decay * params[i];
            first[i] = self.beta1 * first[i] + (1.0 - self.beta1) * grad;
            second[i] = self.beta2 * second[i] + (1.0 - self.beta2) * grad * grad;
            let m_hat = first[i] / bias_correction1.max(f64::EPSILON);
            let v_hat = second[i] / bias_correction2.max(f64::EPSILON);
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub shuffle_seed: Option<u64>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 1,
            batch_size: 1,
            shuffle_seed: None,
        }
    }
}

pub trait Loss<const N: usize> {
    fn loss_and_grad(&self, output: &[Float; N], target: &[Float; N]) -> ([Float; N], Float);
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MeanSquaredError;

pub fn mse_loss<const N: usize>(
    output: &[Float; N],
    target: &[Float; N],
    grad: &mut [Float; N],
) -> Float {
    let scale = 2.0 / N as Float;
    let loss = output
        .iter()
        .zip(target.iter())
        .zip(grad.iter_mut())
        .map(|((&o, &t), g)| {
            let diff = o - t;
            *g = diff * scale;
            diff * diff
        })
        .sum::<Float>();
    loss / N as Float
}

impl<const N: usize> Loss<N> for MeanSquaredError {
    fn loss_and_grad(&self, output: &[Float; N], target: &[Float; N]) -> ([Float; N], Float) {
        let mut grad = [0.0; N];
        let loss = mse_loss(output, target, &mut grad);
        (grad, loss)
    }
}

pub trait Layer<const IN: usize, const OUT: usize> {
    fn forward(&self, input: &[Float; IN], output: &mut [Float; OUT]);
    fn backward(
        &mut self,
        input: &[Float; IN],
        output: &[Float; OUT],
        output_grad: &[Float; OUT],
        input_grad: &mut [Float; IN],
    );

    fn zero_grad(&mut self) {}

    fn apply_gradients<O: Optimizer>(
        &mut self,
        _optimizer: &mut O,
        _slot: &mut usize,
        _scale: Float,
    ) {
    }
}

pub trait LayerDims {
    const INPUT: usize;
    const OUTPUT: usize;
}

#[derive(Debug)]
pub struct DenseLayer<const IN: usize, const OUT: usize> {
    weights: Box<[Float]>,
    biases: Box<[Float; OUT]>,
    weight_grads: Box<[Float]>,
    bias_grads: Box<[Float; OUT]>,
}

#[derive(Debug)]
pub struct ReLU<const N: usize>;

#[derive(Debug)]
pub struct Sigmoid<const N: usize>;

#[derive(Debug)]
pub struct Flatten<const N: usize>;

impl<const IN: usize, const OUT: usize> DenseLayer<IN, OUT> {
    pub fn init() -> Self {
        Self::with_initializer(XavierUniform)
    }

    pub fn seeded(seed: u64) -> Self {
        Self::with_initializer_and_seed(XavierUniform, seed)
    }

    pub fn with_initializer<I: Initializer>(initializer: I) -> Self {
        let mut rng = rand::rng();
        Self::with_initializer_and_rng(initializer, &mut rng)
    }

    pub fn with_initializer_and_seed<I: Initializer>(initializer: I, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::with_initializer_and_rng(initializer, &mut rng)
    }

    pub fn with_initializer_and_rng<I: Initializer, R: Rng + ?Sized>(
        initializer: I,
        rng: &mut R,
    ) -> Self {
        let mut weights = vec![0.0; IN * OUT].into_boxed_slice();
        initializer.fill(&mut weights, IN, OUT, rng);
        Self {
            weights,
            biases: Box::new([0.0; OUT]),
            weight_grads: vec![0.0; IN * OUT].into_boxed_slice(),
            bias_grads: Box::new([0.0; OUT]),
        }
    }

    pub fn forward(&self, input: &[Float; IN], output: &mut [Float; OUT]) {
        for (o, out) in output.iter_mut().enumerate() {
            let row = &self.weights[o * IN..(o + 1) * IN];
            let mut sum = self.biases[o];
            for (weight, inp) in row.iter().zip(input.iter()) {
                sum += *weight * *inp;
            }
            *out = sum;
        }
    }

    pub fn backward(
        &mut self,
        input: &[Float; IN],
        _output: &[Float; OUT],
        output_grad: &[Float; OUT],
        input_grad: &mut [Float; IN],
    ) {
        input_grad.fill(0.0);

        for (o, &grad) in output_grad.iter().enumerate() {
            let row = &self.weights[o * IN..(o + 1) * IN];
            for (in_grad, weight) in input_grad.iter_mut().zip(row.iter()) {
                *in_grad += *weight * grad;
            }
        }

        for (o, &grad) in output_grad.iter().enumerate() {
            self.bias_grads[o] += grad;
            let row_grads = &mut self.weight_grads[o * IN..(o + 1) * IN];
            for (weight_grad, inp) in row_grads.iter_mut().zip(input.iter()) {
                *weight_grad += grad * *inp;
            }
        }
    }
}

impl<const IN: usize, const OUT: usize> LayerDims for DenseLayer<IN, OUT> {
    const INPUT: usize = IN;
    const OUTPUT: usize = OUT;
}

impl<const IN: usize, const OUT: usize> Layer<IN, OUT> for DenseLayer<IN, OUT> {
    fn forward(&self, input: &[Float; IN], output: &mut [Float; OUT]) {
        DenseLayer::forward(self, input, output);
    }

    fn backward(
        &mut self,
        input: &[Float; IN],
        output: &[Float; OUT],
        output_grad: &[Float; OUT],
        input_grad: &mut [Float; IN],
    ) {
        DenseLayer::backward(self, input, output, output_grad, input_grad);
    }

    fn zero_grad(&mut self) {
        self.weight_grads.fill(0.0);
        self.bias_grads.fill(0.0);
    }

    fn apply_gradients<O: Optimizer>(&mut self, optimizer: &mut O, slot: &mut usize, scale: Float) {
        optimizer.update_parameter(*slot, &mut self.weights, &self.weight_grads, scale);
        *slot += 1;
        optimizer.update_parameter(*slot, self.biases.as_mut_slice(), self.bias_grads.as_slice(), scale);
        *slot += 1;
        self.zero_grad();
    }
}

impl<const N: usize> ReLU<N> {
    pub fn init() -> Self {
        ReLU
    }

    pub fn forward(&self, input: &[Float; N], output: &mut [Float; N]) {
        for i in 0..N {
            output[i] = input[i].max(0.0);
        }
    }

    pub fn backward(
        &self,
        input: &[Float; N],
        _output: &[Float; N],
        output_grad: &[Float; N],
        input_grad: &mut [Float; N],
    ) {
        for i in 0..N {
            input_grad[i] = if input[i] > 0.0 { output_grad[i] } else { 0.0 };
        }
    }
}

impl<const N: usize> LayerDims for ReLU<N> {
    const INPUT: usize = N;
    const OUTPUT: usize = N;
}

impl<const N: usize> Layer<N, N> for ReLU<N> {
    fn forward(&self, input: &[Float; N], output: &mut [Float; N]) {
        ReLU::forward(self, input, output);
    }

    fn backward(
        &mut self,
        input: &[Float; N],
        output: &[Float; N],
        output_grad: &[Float; N],
        input_grad: &mut [Float; N],
    ) {
        ReLU::backward(self, input, output, output_grad, input_grad);
    }
}

impl<const N: usize> Sigmoid<N> {
    pub fn init() -> Self {
        Sigmoid
    }

    pub fn forward(&self, input: &[Float; N], output: &mut [Float; N]) {
        for i in 0..N {
            output[i] = 1.0 / (1.0 + (-input[i]).exp());
        }
    }

    pub fn backward(
        &self,
        _input: &[Float; N],
        output: &[Float; N],
        output_grad: &[Float; N],
        input_grad: &mut [Float; N],
    ) {
        for i in 0..N {
            let y = output[i];
            input_grad[i] = output_grad[i] * y * (1.0 - y);
        }
    }
}

impl<const N: usize> LayerDims for Sigmoid<N> {
    const INPUT: usize = N;
    const OUTPUT: usize = N;
}

impl<const N: usize> Layer<N, N> for Sigmoid<N> {
    fn forward(&self, input: &[Float; N], output: &mut [Float; N]) {
        Sigmoid::forward(self, input, output);
    }

    fn backward(
        &mut self,
        input: &[Float; N],
        output: &[Float; N],
        output_grad: &[Float; N],
        input_grad: &mut [Float; N],
    ) {
        Sigmoid::backward(self, input, output, output_grad, input_grad);
    }
}

impl<const N: usize> Flatten<N> {
    pub fn init() -> Self {
        Flatten
    }

    pub fn forward(&self, input: &[Float; N], output: &mut [Float; N]) {
        output.copy_from_slice(input);
    }

    pub fn backward(
        &self,
        _input: &[Float; N],
        _output: &[Float; N],
        output_grad: &[Float; N],
        input_grad: &mut [Float; N],
    ) {
        input_grad.copy_from_slice(output_grad);
    }
}

impl<const N: usize> LayerDims for Flatten<N> {
    const INPUT: usize = N;
    const OUTPUT: usize = N;
}

impl<const N: usize> Layer<N, N> for Flatten<N> {
    fn forward(&self, input: &[Float; N], output: &mut [Float; N]) {
        Flatten::forward(self, input, output);
    }

    fn backward(
        &mut self,
        input: &[Float; N],
        output: &[Float; N],
        output_grad: &[Float; N],
        input_grad: &mut [Float; N],
    ) {
        Flatten::backward(self, input, output, output_grad, input_grad);
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct End;

#[derive(Debug)]
pub struct Chain<Head, Tail, const MID: usize> {
    head: Head,
    tail: Tail,
}

impl<Head, Tail, const MID: usize> Chain<Head, Tail, MID> {
    pub const fn new(head: Head, tail: Tail) -> Self {
        Self { head, tail }
    }

    pub fn head(&self) -> &Head {
        &self.head
    }

    pub fn tail(&self) -> &Tail {
        &self.tail
    }
}

pub trait IntoChain: LayerDims + Sized {
    fn into_chain(self) -> Chain<Self, End, { Self::OUTPUT }> {
        Chain::new(self, End)
    }
}

impl<T> IntoChain for T
where
    T: LayerDims,
    [(); T::OUTPUT]:,
{
}

pub trait AppendLayer<Next>: Sized {
    type Output;
    fn then(self, next: Next) -> Self::Output;
}

impl<Next> AppendLayer<Next> for End
where
    Next: LayerDims,
    [(); Next::OUTPUT]:,
{
    type Output = Chain<Next, End, { Next::OUTPUT }>;

    fn then(self, next: Next) -> Self::Output {
        Chain::new(next, End)
    }
}

impl<Head, Tail, const MID: usize, Next> AppendLayer<Next> for Chain<Head, Tail, MID>
where
    Tail: AppendLayer<Next>,
{
    type Output = Chain<Head, Tail::Output, MID>;

    fn then(self, next: Next) -> Self::Output {
        Chain::new(self.head, self.tail.then(next))
    }
}

#[derive(Debug)]
pub struct TerminalWorkspace<const OUT: usize> {
    activation: Box<[Float; OUT]>,
    gradient: Box<[Float; OUT]>,
}

#[derive(Debug)]
pub struct ChainWorkspace<const MID: usize, TailWorkspace> {
    activation: Box<[Float; MID]>,
    gradient: Box<[Float; MID]>,
    tail: TailWorkspace,
}

#[derive(Debug)]
pub struct SequentialWorkspace<BodyWorkspace, const INPUT: usize> {
    body: BodyWorkspace,
    input_grad: Box<[Float; INPUT]>,
}

pub trait ModuleChain<const INPUT: usize, const OUTPUT: usize> {
    type Workspace;

    fn workspace(&self) -> Self::Workspace;
    fn forward_with_workspace(&self, input: &[Float; INPUT], workspace: &mut Self::Workspace);
    fn output(workspace: &Self::Workspace) -> &[Float; OUTPUT];
    fn set_output_grad(workspace: &mut Self::Workspace, grad: &[Float; OUTPUT]);
    fn backward_with_workspace(
        &mut self,
        input: &[Float; INPUT],
        input_grad: &mut [Float; INPUT],
        workspace: &mut Self::Workspace,
    );
    fn zero_grad(&mut self);
    fn apply_gradients<O: Optimizer>(
        &mut self,
        optimizer: &mut O,
        slot: &mut usize,
        scale: Float,
    );
}

impl<Head, const INPUT: usize, const OUTPUT: usize> ModuleChain<INPUT, OUTPUT>
    for Chain<Head, End, OUTPUT>
where
    Head: Layer<INPUT, OUTPUT>,
{
    type Workspace = TerminalWorkspace<OUTPUT>;

    fn workspace(&self) -> Self::Workspace {
        TerminalWorkspace {
            activation: Box::new([0.0; OUTPUT]),
            gradient: Box::new([0.0; OUTPUT]),
        }
    }

    fn forward_with_workspace(&self, input: &[Float; INPUT], workspace: &mut Self::Workspace) {
        self.head.forward(input, workspace.activation.as_mut());
    }

    fn output(workspace: &Self::Workspace) -> &[Float; OUTPUT] {
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
        self.head.backward(
            input,
            workspace.activation.as_ref(),
            workspace.gradient.as_ref(),
            input_grad,
        );
    }

    fn zero_grad(&mut self) {
        self.head.zero_grad();
    }

    fn apply_gradients<O: Optimizer>(
        &mut self,
        optimizer: &mut O,
        slot: &mut usize,
        scale: Float,
    ) {
        self.head.apply_gradients(optimizer, slot, scale);
    }
}

impl<Head, Tail, const INPUT: usize, const MID: usize, const OUTPUT: usize>
    ModuleChain<INPUT, OUTPUT> for Chain<Head, Tail, MID>
where
    Head: Layer<INPUT, MID>,
    Tail: ModuleChain<MID, OUTPUT>,
{
    type Workspace = ChainWorkspace<MID, Tail::Workspace>;

    fn workspace(&self) -> Self::Workspace {
        ChainWorkspace {
            activation: Box::new([0.0; MID]),
            gradient: Box::new([0.0; MID]),
            tail: self.tail.workspace(),
        }
    }

    fn forward_with_workspace(&self, input: &[Float; INPUT], workspace: &mut Self::Workspace) {
        self.head.forward(input, workspace.activation.as_mut());
        self.tail
            .forward_with_workspace(workspace.activation.as_ref(), &mut workspace.tail);
    }

    fn output(workspace: &Self::Workspace) -> &[Float; OUTPUT] {
        Tail::output(&workspace.tail)
    }

    fn set_output_grad(workspace: &mut Self::Workspace, grad: &[Float; OUTPUT]) {
        Tail::set_output_grad(&mut workspace.tail, grad);
    }

    fn backward_with_workspace(
        &mut self,
        input: &[Float; INPUT],
        input_grad: &mut [Float; INPUT],
        workspace: &mut Self::Workspace,
    ) {
        self.tail.backward_with_workspace(
            workspace.activation.as_ref(),
            workspace.gradient.as_mut(),
            &mut workspace.tail,
        );
        self.head.backward(
            input,
            workspace.activation.as_ref(),
            workspace.gradient.as_ref(),
            input_grad,
        );
    }

    fn zero_grad(&mut self) {
        self.head.zero_grad();
        self.tail.zero_grad();
    }

    fn apply_gradients<O: Optimizer>(
        &mut self,
        optimizer: &mut O,
        slot: &mut usize,
        scale: Float,
    ) {
        self.head.apply_gradients(optimizer, slot, scale);
        self.tail.apply_gradients(optimizer, slot, scale);
    }
}

#[derive(Debug)]
pub struct Sequential<Layers, const INPUT: usize, const OUTPUT: usize>
where
    Layers: ModuleChain<INPUT, OUTPUT>,
{
    layers: Layers,
}

impl<Layers, const INPUT: usize, const OUTPUT: usize> Sequential<Layers, INPUT, OUTPUT>
where
    Layers: ModuleChain<INPUT, OUTPUT>,
{
    pub const fn new(layers: Layers) -> Self {
        Self { layers }
    }

    pub fn layers(&self) -> &Layers {
        &self.layers
    }

    pub fn layers_mut(&mut self) -> &mut Layers {
        &mut self.layers
    }

    pub fn workspace(&self) -> SequentialWorkspace<Layers::Workspace, INPUT> {
        SequentialWorkspace {
            body: self.layers.workspace(),
            input_grad: Box::new([0.0; INPUT]),
        }
    }

    pub fn predict(&self, input: &[Float; INPUT]) -> [Float; OUTPUT] {
        let mut workspace = self.workspace();
        self.predict_with_workspace(input, &mut workspace)
    }

    pub fn predict_with_workspace(
        &self,
        input: &[Float; INPUT],
        workspace: &mut SequentialWorkspace<Layers::Workspace, INPUT>,
    ) -> [Float; OUTPUT] {
        self.layers.forward_with_workspace(input, &mut workspace.body);
        let mut result = [0.0; OUTPUT];
        result.copy_from_slice(Layers::output(&workspace.body));
        result
    }

    pub fn predict_in_place(&self, input: &[Float; INPUT]) -> [Float; OUTPUT] {
        self.predict(input)
    }

    pub fn fit<O: Optimizer>(
        &mut self,
        samples: &[Sample<INPUT, OUTPUT>],
        optimizer: &mut O,
        config: TrainConfig,
    ) -> Float {
        self.fit_with_loss(samples, &MeanSquaredError, optimizer, config)
    }

    pub fn fit_with_loss<L: Loss<OUTPUT>, O: Optimizer>(
        &mut self,
        samples: &[Sample<INPUT, OUTPUT>],
        loss_fn: &L,
        optimizer: &mut O,
        config: TrainConfig,
    ) -> Float {
        if samples.is_empty() || config.epochs == 0 {
            return 0.0;
        }

        let batch_size = config.batch_size.max(1);
        let mut workspace = self.workspace();
        let mut order = (0..samples.len()).collect::<Vec<_>>();
        let mut shuffler = config.shuffle_seed.map(StdRng::seed_from_u64);
        let mut total_loss = 0.0;
        let mut steps = 0usize;

        for _ in 0..config.epochs {
            if let Some(rng) = shuffler.as_mut() {
                order.shuffle(rng);
            }

            for batch in order.chunks(batch_size) {
                self.layers.zero_grad();
                let mut batch_loss = 0.0;

                for &sample_idx in batch {
                    let sample = &samples[sample_idx];
                    self.layers
                        .forward_with_workspace(&sample.input, &mut workspace.body);
                    let (grad, loss) =
                        loss_fn.loss_and_grad(Layers::output(&workspace.body), &sample.target);
                    Layers::set_output_grad(&mut workspace.body, &grad);
                    self.layers.backward_with_workspace(
                        &sample.input,
                        workspace.input_grad.as_mut(),
                        &mut workspace.body,
                    );
                    batch_loss += loss;
                }

                optimizer.begin_step();
                let mut slot = 0usize;
                self.layers
                    .apply_gradients(optimizer, &mut slot, 1.0 / batch.len() as Float);
                total_loss += batch_loss / batch.len() as Float;
                steps += 1;
            }
        }

        total_loss / steps as Float
    }

    pub fn fit_default<O: Optimizer>(
        &mut self,
        samples: &[Sample<INPUT, OUTPUT>],
        optimizer: &mut O,
    ) -> Float {
        self.fit(samples, optimizer, TrainConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Float, b: Float, eps: Float) {
        let diff = (a - b).abs();
        assert!(diff <= eps, "expected {a} ~= {b} (diff={diff}, eps={eps})");
    }

    #[test]
    fn mse_loss_matches_manual_computation() {
        let output = [2.0, -1.0];
        let target = [1.0, 1.0];
        let mut grad = [0.0; 2];
        let loss = mse_loss(&output, &target, &mut grad);
        approx_eq(loss, 2.5, 1e-12);
        assert_eq!(grad, [1.0, -2.0]);
    }

    #[test]
    fn dense_input_gradient_matches_finite_difference() {
        let mut layer = DenseLayer::<2, 2>::with_initializer_and_seed(Uniform::new(-0.3, 0.3), 7);
        layer.weights.copy_from_slice(&[0.4, -0.2, 0.1, 0.3]);
        *layer.biases = [0.05, -0.1];

        let input = [0.7, -1.2];
        let output_grad = [0.8, -0.4];
        let mut output = [0.0; 2];
        let mut input_grad = [0.0; 2];

        layer.zero_grad();
        layer.forward(&input, &mut output);
        layer.backward(&input, &output, &output_grad, &mut input_grad);

        let eps = 1e-7;
        for i in 0..2 {
            let mut plus = input;
            let mut minus = input;
            plus[i] += eps;
            minus[i] -= eps;

            let mut plus_out = [0.0; 2];
            let mut minus_out = [0.0; 2];
            layer.forward(&plus, &mut plus_out);
            layer.forward(&minus, &mut minus_out);
            let objective_plus = plus_out
                .iter()
                .zip(output_grad.iter())
                .map(|(o, g)| o * g)
                .sum::<Float>();
            let objective_minus = minus_out
                .iter()
                .zip(output_grad.iter())
                .map(|(o, g)| o * g)
                .sum::<Float>();
            let numeric = (objective_plus - objective_minus) / (2.0 * eps);
            approx_eq(input_grad[i], numeric, 1e-6);
        }
    }

    #[test]
    fn dense_weight_gradient_matches_finite_difference() {
        let mut layer = DenseLayer::<2, 2>::with_initializer_and_seed(Uniform::new(-0.3, 0.3), 11);
        layer.weights.copy_from_slice(&[0.4, -0.2, 0.1, 0.3]);
        *layer.biases = [0.05, -0.1];

        let input = [0.7, -1.2];
        let output_grad = [0.8, -0.4];
        let mut output = [0.0; 2];
        let mut input_grad = [0.0; 2];

        layer.zero_grad();
        layer.forward(&input, &mut output);
        layer.backward(&input, &output, &output_grad, &mut input_grad);

        let weight_idx = 1;
        let eps = 1e-7;
        let mut plus = DenseLayer::<2, 2>::with_initializer_and_seed(Uniform::new(-0.3, 0.3), 0);
        plus.weights.copy_from_slice(&layer.weights);
        plus.biases.copy_from_slice(layer.biases.as_ref());
        plus.weights[weight_idx] += eps;
        let mut minus = DenseLayer::<2, 2>::with_initializer_and_seed(Uniform::new(-0.3, 0.3), 0);
        minus.weights.copy_from_slice(&layer.weights);
        minus.biases.copy_from_slice(layer.biases.as_ref());
        minus.weights[weight_idx] -= eps;

        let mut plus_out = [0.0; 2];
        let mut minus_out = [0.0; 2];
        plus.forward(&input, &mut plus_out);
        minus.forward(&input, &mut minus_out);
        let objective_plus = plus_out
            .iter()
            .zip(output_grad.iter())
            .map(|(o, g)| o * g)
            .sum::<Float>();
        let objective_minus = minus_out
            .iter()
            .zip(output_grad.iter())
            .map(|(o, g)| o * g)
            .sum::<Float>();
        let numeric = (objective_plus - objective_minus) / (2.0 * eps);

        approx_eq(layer.weight_grads[weight_idx], numeric, 1e-6);
    }

    #[test]
    fn seeded_initialization_is_reproducible() {
        let a = DenseLayer::<3, 2>::seeded(42);
        let b = DenseLayer::<3, 2>::seeded(42);
        assert_eq!(&*a.weights, &*b.weights);
        assert_eq!(&*a.biases, &*b.biases);
    }

    #[test]
    fn sequential_training_decreases_loss_with_seeded_shuffle() {
        let layers = Chain::<_, _, 8>::new(
            DenseLayer::<1, 8>::seeded(3),
            Chain::<_, _, 8>::new(
                ReLU::<8>::init(),
                Chain::<_, _, 1>::new(DenseLayer::<8, 1>::seeded(4), End),
            ),
        );
        let mut model = Sequential::new(layers);
        let samples = (-20..=20)
            .map(|i| {
                let x = i as Float / 10.0;
                Sample::new([x], [2.0 * x - 0.5])
            })
            .collect::<Vec<_>>();
        let mut optimizer = Adam::new(0.03);
        let config = TrainConfig {
            epochs: 250,
            batch_size: 8,
            shuffle_seed: Some(9),
        };

        let before = samples
            .iter()
            .map(|sample| {
                let output = model.predict(&sample.input);
                MeanSquaredError.loss_and_grad(&output, &sample.target).1
            })
            .sum::<Float>()
            / samples.len() as Float;
        let during = model.fit(&samples, &mut optimizer, config);
        let after = samples
            .iter()
            .map(|sample| {
                let output = model.predict(&sample.input);
                MeanSquaredError.loss_and_grad(&output, &sample.target).1
            })
            .sum::<Float>()
            / samples.len() as Float;

        assert!(during < before, "training step average should improve");
        assert!(after < before * 0.2, "expected loss to fall sharply");
    }
}
