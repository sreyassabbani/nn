use std::{any::Any, cell::RefCell, collections::HashMap, fmt, rc::Rc};

use rand::{Rng as _, SeedableRng, rngs::StdRng};

use crate::network::{Layer, LossFunction, Optimizer, TrainConfig};
use crate::{Float, Sample};

use super::{InitConfig, PredictRuntime, TrainRuntime};

pub struct MaterializeContext {
    pub(super) rng: StdRng,
    pub(super) shared: HashMap<usize, Box<dyn Any>>,
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
    pub(super) fn new(layer: L) -> Self {
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
    pub(super) left: Left,
    pub(super) right: Right,
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
    pub(super) inner: Inner,
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
    pub(super) left: Left,
    pub(super) right: Right,
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
    pub(super) left: Left,
    pub(super) right: Right,
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
    pub(super) inner: Rc<RefCell<Inner>>,
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
    pub(super) fn new(runtime: Runtime) -> Self {
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
