use std::array;

use crate::{__private::ConvGeometryIsValid, Float, tensor::Tensor};

use super::{Conv, conv_out_dim};

/// Type-level input/output tensor metadata for convolution layers.
#[allow(dead_code)]
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
    [(); IC * IH * IW]:,
    [(); OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)]:,
{
    const N: usize = IC * IH * IW;
    type Input = Tensor<crate::shape!(IC, IH, IW)>;
    type Output = Tensor<Self::OutputShape>;
    type InputShape = crate::shape!(IC, IH, IW);
    type OutputShape = crate::shape!(OC, conv_out_dim(IH, P, FH, S), conv_out_dim(IW, P, FW, S));
    type FilterShape = crate::shape!(FH, FW, IC);
}

/// Flat-array convenience trait for generic conv code.
#[allow(dead_code)]
pub trait ConvOps: ConvIO {
    type InputArray;
    type OutputArray;
    type FilterArray;

    const INPUT_SIZE: usize;
    const OUTPUT_SIZE: usize;
    const FILTER_SIZE: usize;

    fn init() -> Self;
    fn forward_flat(&self, input: &Self::InputArray, output: &mut Self::OutputArray);
    fn input_from_fn<F: FnMut(usize) -> Float>(f: F) -> Self::InputArray;
    fn output_zeroed() -> Self::OutputArray;
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
> ConvOps for Conv<IW, IH, IC, FH, FW, OC, S, P>
where
    [(); FH * FW * IC]:,
    [(); IC * IH * IW]:,
    [(); OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)]:,
    (): ConvGeometryIsValid<IH, IW, FH, FW, S, P>,
{
    type InputArray = [Float; IC * IH * IW];
    type OutputArray = [Float; OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S)];
    type FilterArray = [Float; FH * FW * IC];

    const INPUT_SIZE: usize = IC * IH * IW;
    const OUTPUT_SIZE: usize = OC * conv_out_dim(IH, P, FH, S) * conv_out_dim(IW, P, FW, S);
    const FILTER_SIZE: usize = FH * FW * IC;

    fn init() -> Self {
        Conv::<IW, IH, IC, FH, FW, OC, S, P>::init()
    }

    fn forward_flat(&self, input: &Self::InputArray, output: &mut Self::OutputArray) {
        Conv::<IW, IH, IC, FH, FW, OC, S, P>::forward_flat(self, input, output);
    }

    fn input_from_fn<F: FnMut(usize) -> Float>(f: F) -> Self::InputArray {
        array::from_fn(f)
    }

    fn output_zeroed() -> Self::OutputArray {
        array::from_fn(|_| 0.0 as Float)
    }
}
