//! Classic sequential training APIs retained during the architecture-language redesign.

mod builder;
mod layers;
mod loss;
mod optim;

pub use builder::{ImageBuilder, ModelBuilder, Sequential, VectorBuilder};
pub use layers::{DenseLayer, Flatten, Layer, LayerDims, ReLU, Sigmoid};
pub use loss::{LossFunction, MeanSquaredError, mse_loss};
pub use optim::{
    Adam, Initializer, KaimingUniform, Optimizer, Sgd, TrainConfig, Uniform, XavierUniform,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Float, Sample};

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
