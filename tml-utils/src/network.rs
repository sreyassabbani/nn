use crate::conv::{Conv, conv_out_dim};
use crate::{ConvGeometryIsValid, Float, Sample};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::fmt;

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

pub trait Optimizer: fmt::Debug {
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

pub struct TrainConfig {
    optimizer: Box<dyn Optimizer>,
    pub epochs: usize,
    pub batch_size: usize,
    pub shuffle_seed: Option<u64>,
}

impl fmt::Debug for TrainConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrainConfig")
            .field("optimizer", &self.optimizer)
            .field("epochs", &self.epochs)
            .field("batch_size", &self.batch_size)
            .field("shuffle_seed", &self.shuffle_seed)
            .finish()
    }
}

impl TrainConfig {
    pub fn new<O: Optimizer + 'static>(optimizer: O) -> Self {
        Self {
            optimizer: Box::new(optimizer),
            epochs: 1,
            batch_size: 1,
            shuffle_seed: None,
        }
    }

    pub fn sgd(lr: Float) -> Self {
        Self::new(Sgd::new(lr))
    }

    pub fn adam(lr: Float) -> Self {
        Self::new(Adam::new(lr))
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    pub fn shuffle_seed(mut self, shuffle_seed: u64) -> Self {
        self.shuffle_seed = Some(shuffle_seed);
        self
    }

    fn optimizer_mut(&mut self) -> &mut dyn Optimizer {
        self.optimizer.as_mut()
    }
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self::adam(1e-3)
    }
}

pub trait LossFunction<const N: usize>: fmt::Debug {
    fn loss_and_grad(&self, output: &[Float; N], target: &[Float; N], grad: &mut [Float; N])
    -> Float;
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

impl<const N: usize> LossFunction<N> for MeanSquaredError {
    fn loss_and_grad(
        &self,
        output: &[Float; N],
        target: &[Float; N],
        grad: &mut [Float; N],
    ) -> Float {
        mse_loss(output, target, grad)
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

    fn apply_gradients(&mut self, _optimizer: &mut dyn Optimizer, _slot: &mut usize, _scale: Float) {}
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

    fn apply_gradients(
        &mut self,
        optimizer: &mut dyn Optimizer,
        slot: &mut usize,
        scale: Float,
    ) {
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

mod private {
    use super::*;

    #[derive(Debug, Clone, Copy, Default)]
    pub struct End;

    #[derive(Debug)]
    pub struct Chain<Head, Tail, const MID: usize> {
        pub(super) head: Head,
        pub(super) tail: Tail,
    }

    impl<Head, Tail, const MID: usize> Chain<Head, Tail, MID> {
        pub const fn new(head: Head, tail: Tail) -> Self {
            Self { head, tail }
        }
    }

    pub trait AppendLayer<Next, const NEXT_OUTPUT: usize>: Sized {
        type Output;
        fn then(self, next: Next) -> Self::Output;
    }

    impl<Next, const NEXT_OUTPUT: usize> AppendLayer<Next, NEXT_OUTPUT> for End {
        type Output = Chain<Next, End, NEXT_OUTPUT>;

        fn then(self, next: Next) -> Self::Output {
            Chain::new(next, End)
        }
    }

    impl<Head, Tail, const MID: usize, Next, const NEXT_OUTPUT: usize>
        AppendLayer<Next, NEXT_OUTPUT> for Chain<Head, Tail, MID>
    where
        Tail: AppendLayer<Next, NEXT_OUTPUT>,
    {
        type Output = Chain<Head, <Tail as AppendLayer<Next, NEXT_OUTPUT>>::Output, MID>;

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
    pub struct StackWorkspace<BodyWorkspace, const INPUT: usize> {
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
        fn apply_gradients(&mut self, optimizer: &mut dyn Optimizer, slot: &mut usize, scale: Float);
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

        fn apply_gradients(
            &mut self,
            optimizer: &mut dyn Optimizer,
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

        fn apply_gradients(
            &mut self,
            optimizer: &mut dyn Optimizer,
            slot: &mut usize,
            scale: Float,
        ) {
            self.head.apply_gradients(optimizer, slot, scale);
            self.tail.apply_gradients(optimizer, slot, scale);
        }
    }

    #[derive(Debug)]
    pub struct Stack<Layers, const INPUT: usize, const OUTPUT: usize>
    where
        Layers: ModuleChain<INPUT, OUTPUT>,
    {
        layers: Layers,
    }

    impl<Layers, const INPUT: usize, const OUTPUT: usize> Stack<Layers, INPUT, OUTPUT>
    where
        Layers: ModuleChain<INPUT, OUTPUT>,
    {
        pub const fn new(layers: Layers) -> Self {
            Self { layers }
        }

        pub fn predict(&self, input: &[Float; INPUT]) -> [Float; OUTPUT] {
            let mut workspace = StackWorkspace {
                body: self.layers.workspace(),
                input_grad: Box::new([0.0; INPUT]),
            };
            self.layers.forward_with_workspace(input, &mut workspace.body);
            let mut result = [0.0; OUTPUT];
            result.copy_from_slice(Layers::output(&workspace.body));
            result
        }

        pub fn fit_with_loss(
            &mut self,
            samples: &[Sample<INPUT, OUTPUT>],
            loss_fn: &dyn LossFunction<OUTPUT>,
            mut config: TrainConfig,
        ) -> Float {
            if samples.is_empty() || config.epochs == 0 {
                return 0.0;
            }

            let batch_size = config.batch_size.max(1);
            let mut workspace = StackWorkspace {
                body: self.layers.workspace(),
                input_grad: Box::new([0.0; INPUT]),
            };
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
                        let mut grad = [0.0; OUTPUT];
                        let loss =
                            loss_fn.loss_and_grad(Layers::output(&workspace.body), &sample.target, &mut grad);
                        Layers::set_output_grad(&mut workspace.body, &grad);
                        self.layers.backward_with_workspace(
                            &sample.input,
                            workspace.input_grad.as_mut(),
                            &mut workspace.body,
                        );
                        batch_loss += loss;
                    }

                    config.optimizer_mut().begin_step();
                    let mut slot = 0usize;
                    self.layers.apply_gradients(
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

    pub trait ModelRuntime<const INPUT: usize, const OUTPUT: usize>: fmt::Debug {
        fn predict(&self, input: &[Float; INPUT]) -> [Float; OUTPUT];
        fn fit_with_loss(
            &mut self,
            samples: &[Sample<INPUT, OUTPUT>],
            loss_fn: &dyn LossFunction<OUTPUT>,
            config: TrainConfig,
        ) -> Float;
    }

    impl<Layers, const INPUT: usize, const OUTPUT: usize> ModelRuntime<INPUT, OUTPUT>
        for Stack<Layers, INPUT, OUTPUT>
    where
        Layers: ModuleChain<INPUT, OUTPUT> + fmt::Debug + 'static,
    {
        fn predict(&self, input: &[Float; INPUT]) -> [Float; OUTPUT] {
            Stack::predict(self, input)
        }

        fn fit_with_loss(
            &mut self,
            samples: &[Sample<INPUT, OUTPUT>],
            loss_fn: &dyn LossFunction<OUTPUT>,
            config: TrainConfig,
        ) -> Float {
            Stack::fit_with_loss(self, samples, loss_fn, config)
        }
    }
}

pub struct Sequential<const INPUT: usize, const OUTPUT: usize> {
    inner: Box<dyn private::ModelRuntime<INPUT, OUTPUT>>,
}

impl<const INPUT: usize, const OUTPUT: usize> fmt::Debug for Sequential<INPUT, OUTPUT> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sequential")
            .field("input", &INPUT)
            .field("output", &OUTPUT)
            .finish()
    }
}

impl<const INPUT: usize, const OUTPUT: usize> Sequential<INPUT, OUTPUT> {
    fn from_runtime<R>(runtime: R) -> Self
    where
        R: private::ModelRuntime<INPUT, OUTPUT> + 'static,
    {
        Self {
            inner: Box::new(runtime),
        }
    }

    pub fn predict(&self, input: &[Float; INPUT]) -> [Float; OUTPUT] {
        self.inner.predict(input)
    }

    pub fn predict_in_place(&self, input: &[Float; INPUT]) -> [Float; OUTPUT] {
        self.predict(input)
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

#[derive(Debug, Clone, Copy, Default)]
pub struct ModelBuilder;

impl ModelBuilder {
    pub const fn new() -> Self {
        Self
    }

    pub fn input<const N: usize>(self) -> VectorBuilder<private::End, N, N> {
        VectorBuilder {
            layers: private::End,
        }
    }

    pub fn image_input<const C: usize, const H: usize, const W: usize>(
        self,
    ) -> ImageBuilder<private::End, { C * H * W }, C, H, W>
    where
        [(); C * H * W]:,
    {
        ImageBuilder {
            layers: private::End,
        }
    }
}

pub struct VectorBuilder<Layers, const INPUT: usize, const CURRENT: usize> {
    layers: Layers,
}

impl<Layers, const INPUT: usize, const CURRENT: usize> fmt::Debug
    for VectorBuilder<Layers, INPUT, CURRENT>
where
    Layers: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VectorBuilder")
            .field("input", &INPUT)
            .field("current", &CURRENT)
            .finish()
    }
}

impl<Layers, const INPUT: usize, const CURRENT: usize> VectorBuilder<Layers, INPUT, CURRENT> {
    pub const fn flatten(self) -> Self {
        self
    }

    pub fn dense<const NEXT: usize>(
        self,
    ) -> VectorBuilder<
        <Layers as private::AppendLayer<DenseLayer<CURRENT, NEXT>, NEXT>>::Output,
        INPUT,
        NEXT,
    >
    where
        Layers: private::AppendLayer<DenseLayer<CURRENT, NEXT>, NEXT>,
    {
        VectorBuilder {
            layers: self.layers.then(DenseLayer::<CURRENT, NEXT>::init()),
        }
    }

    pub fn relu(
        self,
    ) -> VectorBuilder<
        <Layers as private::AppendLayer<ReLU<CURRENT>, CURRENT>>::Output,
        INPUT,
        CURRENT,
    >
    where
        Layers: private::AppendLayer<ReLU<CURRENT>, CURRENT>,
    {
        VectorBuilder {
            layers: self.layers.then(ReLU::<CURRENT>::init()),
        }
    }

    pub fn sigmoid(
        self,
    ) -> VectorBuilder<
        <Layers as private::AppendLayer<Sigmoid<CURRENT>, CURRENT>>::Output,
        INPUT,
        CURRENT,
    >
    where
        Layers: private::AppendLayer<Sigmoid<CURRENT>, CURRENT>,
    {
        VectorBuilder {
            layers: self.layers.then(Sigmoid::<CURRENT>::init()),
        }
    }

    pub fn build(self) -> Sequential<INPUT, CURRENT>
    where
        Layers: private::ModuleChain<INPUT, CURRENT> + fmt::Debug + 'static,
    {
        Sequential::from_runtime(private::Stack::new(self.layers))
    }
}

pub struct ImageBuilder<
    Layers,
    const INPUT: usize,
    const C: usize,
    const H: usize,
    const W: usize,
> {
    layers: Layers,
}

impl<Layers, const INPUT: usize, const C: usize, const H: usize, const W: usize> fmt::Debug
    for ImageBuilder<Layers, INPUT, C, H, W>
where
    Layers: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageBuilder")
            .field("input", &INPUT)
            .field("channels", &C)
            .field("height", &H)
            .field("width", &W)
            .finish()
    }
}

impl<Layers, const INPUT: usize, const C: usize, const H: usize, const W: usize>
    ImageBuilder<Layers, INPUT, C, H, W>
where
    [(); INPUT]:,
{
    pub fn relu(self) -> ImageBuilder<
        <Layers as private::AppendLayer<ReLU<{ C * H * W }>, { C * H * W }>>::Output,
        INPUT,
        C,
        H,
        W,
    >
    where
        [(); C * H * W]:,
        Layers: private::AppendLayer<ReLU<{ C * H * W }>, { C * H * W }>,
    {
        ImageBuilder {
            layers: self.layers.then(ReLU::<{ C * H * W }>::init()),
        }
    }

    pub fn sigmoid(self) -> ImageBuilder<
        <Layers as private::AppendLayer<Sigmoid<{ C * H * W }>, { C * H * W }>>::Output,
        INPUT,
        C,
        H,
        W,
    >
    where
        [(); C * H * W]:,
        Layers: private::AppendLayer<Sigmoid<{ C * H * W }>, { C * H * W }>,
    {
        ImageBuilder {
            layers: self.layers.then(Sigmoid::<{ C * H * W }>::init()),
        }
    }

    pub fn conv<
        const OC: usize,
        const FH: usize,
        const FW: usize,
        const S: usize,
        const P: usize,
    >(
        self,
    ) -> ImageBuilder<
        <Layers as private::AppendLayer<
            Conv<W, H, C, FH, FW, OC, S, P>,
            { OC * conv_out_dim(H, P, FH, S) * conv_out_dim(W, P, FW, S) },
        >>::Output,
        INPUT,
        OC,
        { conv_out_dim(H, P, FH, S) },
        { conv_out_dim(W, P, FW, S) },
    >
    where
        [(); C * H * W]:,
        [(); OC * conv_out_dim(H, P, FH, S) * conv_out_dim(W, P, FW, S)]:,
        (): ConvGeometryIsValid<H, W, FH, FW, S, P>,
        Layers: private::AppendLayer<
            Conv<W, H, C, FH, FW, OC, S, P>,
            { OC * conv_out_dim(H, P, FH, S) * conv_out_dim(W, P, FW, S) },
        >,
    {
        ImageBuilder {
            layers: self.layers.then(Conv::<W, H, C, FH, FW, OC, S, P>::init()),
        }
    }

    pub fn flatten(self) -> VectorBuilder<Layers, INPUT, { C * H * W }>
    where
        [(); C * H * W]:,
    {
        VectorBuilder { layers: self.layers }
    }

    pub fn build(self) -> Sequential<INPUT, { C * H * W }>
    where
        [(); C * H * W]:,
        Layers: private::ModuleChain<INPUT, { C * H * W }> + fmt::Debug + 'static,
    {
        Sequential::from_runtime(private::Stack::new(self.layers))
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
    fn builder_training_decreases_loss_with_seeded_shuffle() {
        let mut model = ModelBuilder::new()
            .input::<1>()
            .dense::<8>()
            .relu()
            .dense::<1>()
            .build();
        let samples = (-20..=20)
            .map(|i| {
                let x = i as Float / 10.0;
                Sample::new([x], [2.0 * x - 0.5])
            })
            .collect::<Vec<_>>();
        let config = TrainConfig::adam(0.03)
            .epochs(250)
            .batch_size(8)
            .shuffle_seed(9);

        let before = samples
            .iter()
            .map(|sample| {
                let output = model.predict(&sample.input);
                let mut grad = [0.0; 1];
                MeanSquaredError.loss_and_grad(&output, &sample.target, &mut grad)
            })
            .sum::<Float>()
            / samples.len() as Float;
        let during = model.fit(&samples, config);
        let after = samples
            .iter()
            .map(|sample| {
                let output = model.predict(&sample.input);
                let mut grad = [0.0; 1];
                MeanSquaredError.loss_and_grad(&output, &sample.target, &mut grad)
            })
            .sum::<Float>()
            / samples.len() as Float;

        assert!(during < before, "training step average should improve");
        assert!(after < before * 0.2, "expected loss to fall sharply");
    }
}
