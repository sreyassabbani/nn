use crate::conv::{Conv, conv_out_dim};
use crate::{ConvGeometryIsValid, Float, Sample};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::fmt;

use super::{
    DenseLayer, Layer, LossFunction, MeanSquaredError, Optimizer, ReLU, Sigmoid, TrainConfig,
};

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
        fn apply_gradients(
            &mut self,
            optimizer: &mut dyn Optimizer,
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
            self.layers
                .forward_with_workspace(input, &mut workspace.body);
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
                        let loss = loss_fn.loss_and_grad(
                            Layers::output(&workspace.body),
                            &sample.target,
                            &mut grad,
                        );
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

pub struct ImageBuilder<Layers, const INPUT: usize, const C: usize, const H: usize, const W: usize>
{
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
    pub fn relu(
        self,
    ) -> ImageBuilder<
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

    pub fn sigmoid(
        self,
    ) -> ImageBuilder<
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

    pub fn conv<const OC: usize, const FH: usize, const FW: usize, const S: usize, const P: usize>(
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
        VectorBuilder {
            layers: self.layers,
        }
    }

    pub fn build(self) -> Sequential<INPUT, { C * H * W }>
    where
        [(); C * H * W]:,
        Layers: private::ModuleChain<INPUT, { C * H * W }> + fmt::Debug + 'static,
    {
        Sequential::from_runtime(private::Stack::new(self.layers))
    }
}
