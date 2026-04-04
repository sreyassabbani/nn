use super::Conv;
use crate::Float;
use crate::network::Layer;

type ConvCase = Conv<3, 3, 1, 2, 2, 1, 1, 0>;
const IN_SIZE: usize = 3 * 3;
const OUT_SIZE: usize = 4;

fn approx_eq(a: Float, b: Float, eps: Float) {
    let diff = (a - b).abs();
    assert!(diff <= eps, "expected {a} ~= {b} (diff={diff}, eps={eps})");
}

fn configured_conv() -> ConvCase {
    let mut conv = ConvCase::init();
    for (i, w) in conv.filters[0]
        .weights
        .raw_mut_slice()
        .iter_mut()
        .enumerate()
    {
        *w = 0.1 * (i as Float + 1.0);
    }
    conv.biases[0] = 0.05;
    conv
}

fn objective(conv: &ConvCase, input: &[Float; IN_SIZE], output_grad: &[Float; OUT_SIZE]) -> Float {
    let mut output = [0.0; OUT_SIZE];
    conv.forward_flat(input, &mut output);
    output
        .iter()
        .zip(output_grad.iter())
        .map(|(o, g)| o * g)
        .sum()
}

#[test]
fn input_gradient_matches_finite_difference() {
    let mut conv = configured_conv();
    let input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let output_grad = [0.3, -0.2, 0.1, 0.4];
    let mut input_grad = [0.0; IN_SIZE];

    conv.zero_grad();
    conv.backward_flat(&input, &output_grad, &mut input_grad);

    let eps = 1e-7;
    for i in 0..IN_SIZE {
        let mut plus = input;
        let mut minus = input;
        plus[i] += eps;
        minus[i] -= eps;
        let f_plus = objective(&conv, &plus, &output_grad);
        let f_minus = objective(&conv, &minus, &output_grad);
        let numeric = (f_plus - f_minus) / (2.0 * eps);
        approx_eq(input_grad[i], numeric, 1e-6);
    }
}

#[test]
fn weight_update_matches_finite_difference_gradient() {
    let mut conv = configured_conv();
    let input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let output_grad = [0.3, -0.2, 0.1, 0.4];
    let mut input_grad = [0.0; IN_SIZE];
    let weight_idx = 2;

    let eps = 1e-7;
    let mut conv_plus = configured_conv();
    conv_plus.filters[0].weights.raw_mut_slice()[weight_idx] += eps;
    let mut conv_minus = configured_conv();
    conv_minus.filters[0].weights.raw_mut_slice()[weight_idx] -= eps;
    let numeric = (objective(&conv_plus, &input, &output_grad)
        - objective(&conv_minus, &input, &output_grad))
        / (2.0 * eps);

    conv.zero_grad();
    conv.backward_flat(&input, &output_grad, &mut input_grad);
    let analytic = conv.filters[0].grads[weight_idx];

    approx_eq(analytic, numeric, 1e-6);
}
