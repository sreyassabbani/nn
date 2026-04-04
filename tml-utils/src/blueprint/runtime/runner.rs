//! Trainable model wrapper around a graph runtime.

use rand::{SeedableRng, rngs::StdRng};

use crate::network::{LossFunction, TrainConfig};
use crate::{Float, Sample};

use super::GraphRuntime;
use crate::blueprint::{PredictRuntime, TrainRuntime};

#[derive(Debug)]
#[doc(hidden)]
pub struct GraphRunner<Runtime, const INPUT: usize, const OUTPUT: usize> {
    runtime: Runtime,
}

impl<Runtime, const INPUT: usize, const OUTPUT: usize> GraphRunner<Runtime, INPUT, OUTPUT> {
    pub(crate) fn new(runtime: Runtime) -> Self {
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
