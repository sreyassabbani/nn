//! Internal runtime and workspace plumbing for the legacy sequential builders.
//!
//! None of the types in this module are part of the intended public story.
//! They exist to keep the older builder-based training path working while the
//! typed blueprint API evolves.

use crate::{Float, Sample};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::fmt;

use super::super::{Layer, LossFunction, Optimizer, TrainConfig};

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

impl<Head, Tail, const MID: usize, Next, const NEXT_OUTPUT: usize> AppendLayer<Next, NEXT_OUTPUT>
    for Chain<Head, Tail, MID>
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

    fn apply_gradients(&mut self, optimizer: &mut dyn Optimizer, slot: &mut usize, scale: Float) {
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

    fn apply_gradients(&mut self, optimizer: &mut dyn Optimizer, slot: &mut usize, scale: Float) {
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
