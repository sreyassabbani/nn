use rand::{Rng, SeedableRng, rngs::StdRng};
use std::array;

use crate::network::{Initializer, Layer, LayerDims, Optimizer, XavierUniform};
use crate::{__private::ConvGeometryIsValid, Float, tensor::Tensor};

use super::conv_out_dim;

#[derive(Debug, Clone)]
/// One convolution filter with weights and accumulated gradients.
pub struct Filter<const H: usize, const W: usize, const D: usize> {
    pub(super) weights: Tensor<crate::shape!(H, W, D)>,
    pub(super) grads: Box<[Float]>,
}

impl<const H: usize, const W: usize, const D: usize> Filter<H, W, D> {
    pub(super) fn zeroed() -> Self {
        Self {
            weights: Tensor::<crate::shape!(H, W, D)>::from_boxed(
                vec![0.0 as Float; H * W * D].into_boxed_slice(),
            ),
            grads: vec![0.0 as Float; H * W * D].into_boxed_slice(),
        }
    }

    pub(super) fn weights(&self) -> &[Float] {
        self.weights.raw_slice()
    }

    pub(super) fn grads_mut(&mut self) -> &mut [Float] {
        &mut self.grads[..]
    }
}

/// A statically-shaped convolutional layer.
///
/// Generic parameters:
/// - `IW`, `IH`: input width and height
/// - `IC`: input channels
/// - `FH`, `FW`: filter height and width
/// - `OC`: output channels
/// - `S`: stride
/// - `P`: zero padding
#[derive(Debug)]
pub struct Conv<
    const IW: usize,
    const IH: usize,
    const IC: usize,
    const FH: usize,
    const FW: usize,
    const OC: usize,
    const S: usize,
    const P: usize,
> {
    pub(super) filters: [Filter<FH, FW, IC>; OC],
    pub(super) biases: Box<[Float; OC]>,
    pub(super) bias_grads: Box<[Float; OC]>,
}

impl<
    const IW: usize,
    const IH: usize,
    const IC: usize,
    const FH: usize,
    const FW: usize,
    const OC: usize,
    const S: usize,
    const P: usize,
> Conv<IW, IH, IC, FH, FW, OC, S, P>
where
    [(); IC * IH * IW]:,
    [(); OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)]:,
    (): ConvGeometryIsValid<IH, IW, FH, FW, S, P>,
{
    /// Creates a convolution layer with Xavier-uniform initialization.
    pub fn init() -> Self {
        Self::with_initializer(XavierUniform)
    }

    /// Creates a reproducibly initialized convolution layer.
    pub fn seeded(seed: u64) -> Self {
        Self::with_initializer_and_seed(XavierUniform, seed)
    }

    /// Creates a convolution layer with a caller-provided initializer.
    pub fn with_initializer<I: Initializer>(initializer: I) -> Self {
        let mut rng = rand::rng();
        Self::with_initializer_and_rng(initializer, &mut rng)
    }

    /// Creates a convolution layer with a caller-provided initializer and seed.
    pub fn with_initializer_and_seed<I: Initializer>(initializer: I, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::with_initializer_and_rng(initializer, &mut rng)
    }

    /// Creates a convolution layer with a caller-provided initializer and RNG.
    pub fn with_initializer_and_rng<I: Initializer, R: Rng + ?Sized>(
        initializer: I,
        rng: &mut R,
    ) -> Self {
        let mut conv = Conv {
            filters: array::from_fn(|_| Filter::zeroed()),
            biases: Box::new([0.0 as Float; OC]),
            bias_grads: Box::new([0.0 as Float; OC]),
        };
        let fan_in = FH * FW * IC;
        let fan_out = FH * FW * OC;
        for filter in &mut conv.filters {
            initializer.fill(filter.weights.raw_mut_slice(), fan_in, fan_out, rng);
        }
        conv
    }

    /// Allocates an output tensor matching this layer's output geometry.
    pub fn create_output_space(&self) -> <Self as super::ConvIO>::Output {
        Tensor::<crate::shape!(
            OC,
            conv_out_dim(IH, P, FH, S),
            conv_out_dim(IW, P, FW, S)
        )>::from_boxed(
            vec![
                0.0 as Float;
                OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)
            ]
            .into_boxed_slice(),
        )
    }

    /// Wraps flat input data in the layer's typed input tensor.
    pub fn input_from_data(&self, data: [Float; IC * IH * IW]) -> <Self as super::ConvIO>::Input {
        Tensor::<crate::shape!(IC, IH, IW)>::from_boxed(Vec::from(data).into_boxed_slice())
    }

    /// Runs a typed forward pass.
    pub fn forward(
        &self,
        input: &Tensor<crate::shape!(IC, IH, IW)>,
        output: &mut Tensor<
            crate::shape!(OC, conv_out_dim(IH, P, FH, S), conv_out_dim(IW, P, FW, S)),
        >,
    ) {
        let input_arr: &[Float; IC * IH * IW] = input.raw_slice().try_into().expect("bad input");
        let output_arr: &mut [Float; OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)] =
            output.raw_mut_slice().try_into().expect("bad output");
        self.forward_flat(input_arr, output_arr);
    }

    /// Runs a forward pass over flat arrays.
    pub fn forward_flat(
        &self,
        input: &[Float; IC * IH * IW],
        output: &mut [Float; OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)],
    ) {
        let out_h = conv_out_dim(IH, P, FH, S);
        let out_w = conv_out_dim(IW, P, FW, S);

        for oc in 0..OC {
            let filter_data = self.filters[oc].weights();

            for y in 0..out_h {
                for x in 0..out_w {
                    let mut sum = self.biases[oc];

                    for ky in 0..FH {
                        for kx in 0..FW {
                            for ic in 0..IC {
                                let in_y = y * S + ky;
                                let in_x = x * S + kx;
                                let in_y = in_y as isize - P as isize;
                                let in_x = in_x as isize - P as isize;

                                if in_y >= 0
                                    && in_y < IH as isize
                                    && in_x >= 0
                                    && in_x < IW as isize
                                {
                                    let in_y = in_y as usize;
                                    let in_x = in_x as usize;
                                    let input_idx = ic * IH * IW + in_y * IW + in_x;
                                    let filter_idx = (ky * FW + kx) * IC + ic;
                                    sum += filter_data[filter_idx] * input[input_idx];
                                }
                            }
                        }
                    }

                    let output_idx = oc * out_h * out_w + y * out_w + x;
                    output[output_idx] = sum;
                }
            }
        }
    }

    /// Backpropagates a flat output gradient into input and parameter gradients.
    pub fn backward_flat(
        &mut self,
        input: &[Float; IC * IH * IW],
        output_grad: &[Float; OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)],
        input_grad: &mut [Float; IC * IH * IW],
    ) {
        let out_h = conv_out_dim(IH, P, FH, S);
        let out_w = conv_out_dim(IW, P, FW, S);

        input_grad.fill(0.0);

        for oc in 0..OC {
            let Filter { weights, grads } = &mut self.filters[oc];
            let filter_weights = weights.raw_slice();
            let filter_grads = &mut grads[..];

            for y in 0..out_h {
                for x in 0..out_w {
                    let output_idx = oc * out_h * out_w + y * out_w + x;
                    let grad = output_grad[output_idx];
                    self.bias_grads[oc] += grad;

                    for ky in 0..FH {
                        for kx in 0..FW {
                            for ic in 0..IC {
                                let in_y = y * S + ky;
                                let in_x = x * S + kx;
                                let in_y = in_y as isize - P as isize;
                                let in_x = in_x as isize - P as isize;

                                if in_y >= 0
                                    && in_y < IH as isize
                                    && in_x >= 0
                                    && in_x < IW as isize
                                {
                                    let in_y = in_y as usize;
                                    let in_x = in_x as usize;
                                    let input_idx = ic * IH * IW + in_y * IW + in_x;
                                    let filter_idx = (ky * FW + kx) * IC + ic;

                                    filter_grads[filter_idx] += grad * input[input_idx];
                                    input_grad[input_idx] += grad * filter_weights[filter_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

impl<
    const IW: usize,
    const IH: usize,
    const IC: usize,
    const FH: usize,
    const FW: usize,
    const OC: usize,
    const S: usize,
    const P: usize,
> LayerDims for Conv<IW, IH, IC, FH, FW, OC, S, P>
where
    [(); IC * IH * IW]:,
    [(); OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)]:,
    (): ConvGeometryIsValid<IH, IW, FH, FW, S, P>,
{
    const INPUT: usize = IC * IH * IW;
    const OUTPUT: usize = OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S);
}

impl<
    const IW: usize,
    const IH: usize,
    const IC: usize,
    const FH: usize,
    const FW: usize,
    const OC: usize,
    const S: usize,
    const P: usize,
> Layer<{ IC * IH * IW }, { OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S) }>
    for Conv<IW, IH, IC, FH, FW, OC, S, P>
where
    [(); IC * IH * IW]:,
    [(); OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)]:,
    (): ConvGeometryIsValid<IH, IW, FH, FW, S, P>,
{
    fn forward(
        &self,
        input: &[Float; IC * IH * IW],
        output: &mut [Float; OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)],
    ) {
        self.forward_flat(input, output);
    }

    fn backward(
        &mut self,
        input: &[Float; IC * IH * IW],
        _output: &[Float; OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)],
        output_grad: &[Float; OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)],
        input_grad: &mut [Float; IC * IH * IW],
    ) {
        self.backward_flat(input, output_grad, input_grad);
    }

    fn zero_grad(&mut self) {
        self.bias_grads.fill(0.0);
        for filter in &mut self.filters {
            filter.grads_mut().fill(0.0);
        }
    }

    fn apply_gradients(&mut self, optimizer: &mut dyn Optimizer, slot: &mut usize, scale: Float) {
        for filter in &mut self.filters {
            optimizer.update_parameter(
                *slot,
                filter.weights.raw_mut_slice(),
                filter.grads.as_ref(),
                scale,
            );
            *slot += 1;
            filter.grads_mut().fill(0.0);
        }
        optimizer.update_parameter(
            *slot,
            self.biases.as_mut_slice(),
            self.bias_grads.as_slice(),
            scale,
        );
        *slot += 1;
        self.bias_grads.fill(0.0);
    }
}
