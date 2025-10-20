use std::{array, marker::PhantomData};

use crate::tensor::Tensor;

// Define the DenseLayer struct with weights and biases
#[derive(Debug)]
pub struct DenseLayer<const IN: usize, const OUT: usize> {
    weights: Box<[[f32; IN]; OUT]>,
    biases: Box<[f32; OUT]>,
}

// Rectified Linear Unit
#[derive(Debug)]
pub struct ReLU<const N: usize>;

// Sigmoid
#[derive(Debug)]
pub struct Sigmoid<const N: usize>;

// height, width, and depth (input channel size)
// pub struct Filter<const H: usize, const W: usize, const D: usize>([[[f32; H]; W]; D]);
#[derive(Debug, Clone)]
pub struct Filter<const H: usize, const W: usize, const D: usize>(
    Tensor<{ H * W * D }, 3, shape_ty!(H, W, D)>,
)
where
    Tensor<{ H * W * D }, 3, shape_ty!(H, W, D)>: Sized; // current limitation of compiler's const generic features

impl<const H: usize, const W: usize, const D: usize> Default for Filter<H, W, D>
where
    Tensor<{ H * W * D }, 3, shape_ty!(H, W, D)>: Sized,
{
    fn default() -> Self {
        let mut arr = [0.; H * W * D];
        rand::fill(&mut arr);

        Self(Tensor {
            data: Box::new(arr),
            _shape_marker: PhantomData,
        })
    }
}

/// A convolutional layer
///
/// `KH` - kernel/filter height
/// `KW` - kernel/filter width
/// `IC` - number of input channels
/// `OC` - number of output channels (equivalently, number of kernels/filters)
/// `S` - stride
/// `P` - padding
#[derive(Debug)]
pub struct Conv<
    const IW: usize,
    const IH: usize,
    const IC: usize,
    const KH: usize,
    const KW: usize,
    const OC: usize,
    const S: usize,
    const P: usize,
> where
    Tensor<{ KH * KW * IC }, 3, shape_ty!(KH, KW, IC)>: Sized,
{
    data: [Filter<KH, KW, IC>; OC],
}
impl<
    const IW: usize,
    const IH: usize,
    const IC: usize,
    const KH: usize,
    const KW: usize,
    const OC: usize,
    const S: usize,
    const P: usize,
> Conv<IW, IH, IC, KH, KW, OC, S, P>
where
    Tensor<{ KH * KW * IC }, 3, shape_ty!(KH, KW, IC)>: Sized,
{
    pub fn init() -> Self {
        Conv {
            data: array::from_fn(|_| Filter::default()),
        }
    }

    pub fn create_output_space(
        &self,
    ) -> Tensor<
        { OC * ((IH + 2 * P - KH) / S + 1) * ((IW + 2 * P - KW) / S + 1) },
        3,
        shape_ty!(OC, (IH + 2 * P - KH) / S + 1, (IW + 2 * P - KW) / S + 1),
    > {
        Tensor::new()
    }

    pub fn forward(
        &self,
        input: &Tensor<{ IC * IH * IW }, 3, shape_ty!(IC, IH, IW)>,
        output: &mut Tensor<
            { OC * ((IH + 2 * P - KH) / S + 1) * ((IW + 2 * P - KW) / S + 1) },
            3,
            shape_ty!(OC, (IH + 2 * P - KH) / S + 1, (IW + 2 * P - KW) / S + 1),
        >,
    ) {
        let out_h = (IH + 2 * P - KH) / S + 1;
        let out_w = (IW + 2 * P - KW) / S + 1;

        for oc in 0..OC {
            let filter = &self.data[oc].0; // Filter is Tensor<..., shape_ty!(KH, KW, IC)>

            for y in 0..out_h {
                for x in 0..out_w {
                    let mut sum = 0.0;

                    // apply filter
                    for ky in 0..KH {
                        for kx in 0..KW {
                            for ic in 0..IC {
                                // calculate input position (accounting for stride)
                                let in_y = (y * S + ky) as isize - P as isize;
                                let in_x = (x * S + kx) as isize - P as isize;

                                // check if within valid input bounds (zero padding outside)
                                if in_y >= 0
                                    && in_y < IH as isize
                                    && in_x >= 0
                                    && in_x < IW as isize
                                {
                                    // Input shape: (IC, IH, IW) -> index as [ic, y, x]
                                    let input_val = input.at([ic, in_y as usize, in_x as usize]);
                                    // Filter shape: (KH, KW, IC) -> index as [ky, kx, ic]
                                    let filter_val = filter.at([ky, kx, ic]);

                                    sum += filter_val * input_val;
                                }
                            }
                        }
                    }

                    // Output shape: (OC, out_h, out_w) -> index as [oc, y, x]
                    output.set([oc, y, x], sum);
                }
            }
        }
    }
}

pub trait ConvIO {
    type Output;
    type Input;
    type OutputShape;
    type InputShape;
    const N: usize;
}

impl<
    const IW: usize,
    const IH: usize,
    const IC: usize,
    const KH: usize,
    const KW: usize,
    const OC: usize,
    const S: usize,
    const P: usize,
> ConvIO for Conv<IW, IH, IC, KH, KW, OC, S, P>
where
    Tensor<{ IC * IH * IW }, 3, shape_ty!(IC, IH, IW)>: Sized,
    Tensor<{ KH * KW * IC }, 3, shape_ty!(KH, KW, IC)>: Sized,
    Tensor<
        { OC * ((IH + 2 * P - KH) / S + 1) * ((IW + 2 * P - KW) / S + 1) },
        3,
        shape_ty!(OC, (IH + 2 * P - KH) / S + 1, (IW + 2 * P - KW) / S + 1),
    >: Sized,
{
    const N: usize = IC * IH * IW; // Fixed: was IW * IH * IC
    type Input = Tensor<{ IC * IH * IW }, 3, shape_ty!(IC, IH, IW)>; // Fixed order
    type Output = Tensor<
        { OC * ((IH + 2 * P - KH) / S + 1) * ((IW + 2 * P - KW) / S + 1) },
        3,
        Self::OutputShape,
    >;
    type InputShape = shape_ty!(IC, IH, IW); // Fixed order
    type OutputShape = shape_ty!(OC, (IH + 2 * P - KH) / S + 1, (IW + 2 * P - KW) / S + 1);
}

// Forward pass implementation for ReLU
impl<const N: usize> ReLU<N> {
    pub fn init() -> Self {
        ReLU
    }

    pub fn forward(&self, input: &[f32], output: &mut [f32])
    // where
    //     I: AsRef<[f32; N]>,
    {
        for i in 0..N {
            output[i] = input.as_ref()[i].max(0.0);
        }
    }
}

// Forward pass implementation for Sigmoid
impl<const N: usize> Sigmoid<N> {
    pub fn init() -> Self {
        Sigmoid
    }

    /// You can pass a reference to owned values in &Box<>
    pub fn forward(&self, input: &[f32], output: &mut [f32])
    // where
    //     I: AsRef<[f32; N]>,
    {
        for i in 0..N {
            output[i] = 1.0 / (1.0 + (-input.as_ref()[i]).exp());
        }
    }
}

// Initialize DenseLayer (simplified; real init would use proper randomization)
impl<const IN: usize, const OUT: usize> DenseLayer<IN, OUT> {
    pub fn init() -> Self {
        Self {
            weights: Box::new([[0.0; IN]; OUT]),
            biases: Box::new([0.0; OUT]),
        }
    }

    // Forward pass for DenseLayer (basic implementation)
    //
    // used to be forward<I: AsRef<[f32; IN]>>(... input: I, ...)
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        for o in 0..OUT {
            let mut sum = self.biases[o];
            for i in 0..IN {
                sum += self.weights[o][i] * input.as_ref()[i];
            }
            output[o] = sum;
        }
    }
}
