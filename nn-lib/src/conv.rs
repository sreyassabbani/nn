use crate::tensor::Tensor;
use std::{array, marker::PhantomData};

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
/// `FH` - filter/kernel height
/// `FW` - filter/kernel width
/// `IC` - number of input channels
/// `OC` - number of output channels (equivalently, number of kernels/filters)
/// `S` - stride
/// `P` - padding
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
> where
    Tensor<{ FH * FW * IC }, 3, shape_ty!(FH, FW, IC)>: Sized,
{
    data: [Filter<FH, FW, IC>; OC],
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
    Tensor<{ FH * FW * IC }, 3, shape_ty!(FH, FW, IC)>: Sized,
{
    pub fn init() -> Self {
        Conv {
            data: array::from_fn(|_| Filter::default()),
        }
    }

    pub fn create_output_space(
        &self,
    ) -> Tensor<
        { OC * ((IH + 2 * P - FH) / S + 1) * ((IW + 2 * P - FW) / S + 1) },
        3,
        shape_ty!(OC, (IH + 2 * P - FH) / S + 1, (IW + 2 * P - FW) / S + 1),
    > {
        Tensor::new()
    }

    pub fn forward(
        &self,
        input: &Tensor<{ IC * IH * IW }, 3, shape_ty!(IC, IH, IW)>,
        output: &mut Tensor<
            { OC * ((IH + 2 * P - FH) / S + 1) * ((IW + 2 * P - FW) / S + 1) },
            3,
            shape_ty!(OC, (IH + 2 * P - FH) / S + 1, (IW + 2 * P - FW) / S + 1),
        >,
    ) {
        let out_h = (IH + 2 * P - FH) / S + 1;
        let out_w = (IW + 2 * P - FW) / S + 1;

        for oc in 0..OC {
            let filter = &self.data[oc].0; // Filter is Tensor<..., shape_ty!(FH, FW, IC)>

            for y in 0..out_h {
                for x in 0..out_w {
                    let mut sum = 0.0;

                    // apply filter
                    for ky in 0..FH {
                        for kx in 0..FW {
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
                                    // Filter shape: (FH, FW, IC) -> index as [ky, kx, ic]
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
    type FilterShape;
    const N: usize;
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
> ConvIO for Conv<IW, IH, IC, FH, FW, OC, S, P>
where
    Tensor<{ IC * IH * IW }, 3, shape_ty!(IC, IH, IW)>: Sized,
    Tensor<{ FH * FW * IC }, 3, shape_ty!(FH, FW, IC)>: Sized,
    Tensor<
        { OC * ((IH + 2 * P - FH) / S + 1) * ((IW + 2 * P - FW) / S + 1) },
        3,
        shape_ty!(OC, (IH + 2 * P - FH) / S + 1, (IW + 2 * P - FW) / S + 1),
    >: Sized,
{
    const N: usize = IC * IH * IW;
    type Input = Tensor<{ IC * IH * IW }, 3, shape_ty!(IC, IH, IW)>;
    type Output = Tensor<
        { OC * ((IH + 2 * P - FH) / S + 1) * ((IW + 2 * P - FW) / S + 1) },
        3,
        Self::OutputShape,
    >;
    type InputShape = shape_ty!(IC, IH, IW);
    type OutputShape = shape_ty!(OC, (IH + 2 * P - FH) / S + 1, (IW + 2 * P - FW) / S + 1);
    type FilterShape = shape_ty!(IC, FH, FW);
}
