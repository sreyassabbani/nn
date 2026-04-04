use std::collections::HashSet;

use crate::__private::ConvKernelFitsInput;
use crate::conv::{Conv, conv_out_dim};
use crate::network::{DenseLayer, Flatten, ReLU, Sigmoid, XavierUniform};
use crate::shape::{Dim, Nil};

use super::super::runtime::{LeafRuntime, MaterializeContext};
use super::super::{
    Axis, ConvSpec, DenseExpectsFlatInput, DenseSpec, FlattenSpec, IdentitySpec, ReLUSpec,
    SigmoidSpec, TransformSpec, features_axis,
};

#[doc(hidden)]
impl<const N: usize, const NAME: &'static str, const OUT: usize, const BIAS: bool>
    DenseExpectsFlatInput<OUT, BIAS> for Dim<N, Nil, NAME>
where
    [(); N]:,
{
    type Runtime = LeafRuntime<DenseLayer<N, OUT>, N, OUT>;

    fn materialize_dense(ctx: &mut MaterializeContext) -> Self::Runtime {
        let layer = DenseLayer::<N, OUT>::with_initializer_and_rng(XavierUniform, &mut ctx.rng);
        let layer = if BIAS { layer } else { layer.without_bias() };
        LeafRuntime::new(layer)
    }

    fn dense_parameter_count() -> usize {
        let bias = if BIAS { OUT } else { 0 };
        N * OUT + bias
    }
}

impl<InputShape, const OUT: usize, const BIAS: bool> TransformSpec<InputShape>
    for DenseSpec<OUT, BIAS>
where
    InputShape: DenseExpectsFlatInput<OUT, BIAS>,
{
    type OutputShape = Dim<OUT, Nil, "features">;
    const OUTPUT_SIZE: usize = OUT;
    type Runtime = InputShape::Runtime;

    fn materialize(&self, ctx: &mut MaterializeContext) -> Self::Runtime {
        InputShape::materialize_dense(ctx)
    }

    fn parameter_count(&self, _seen_shared: &mut HashSet<usize>) -> usize {
        InputShape::dense_parameter_count()
    }

    fn output_axes(&self, _input_axes: &[Axis]) -> Box<[Axis]> {
        features_axis()
    }

    fn description(&self) -> String {
        if BIAS {
            format!("dense({OUT})")
        } else {
            format!("dense({OUT}, bias: false)")
        }
    }
}

macro_rules! impl_pointwise_transform {
    ($spec:ty, $layer:ident, $desc:literal) => {
        impl<const N: usize, const N_NAME: &'static str> TransformSpec<Dim<N, Nil, N_NAME>>
            for $spec
        where
            [(); N]:,
        {
            type OutputShape = Dim<N, Nil, N_NAME>;
            const OUTPUT_SIZE: usize = N;
            type Runtime = LeafRuntime<$layer<N>, N, N>;

            fn materialize(&self, _ctx: &mut MaterializeContext) -> Self::Runtime {
                LeafRuntime::new($layer::<N>::init())
            }

            fn parameter_count(&self, _seen_shared: &mut HashSet<usize>) -> usize {
                0
            }

            fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
                input_axes.to_vec().into_boxed_slice()
            }

            fn description(&self) -> String {
                $desc.to_string()
            }
        }

        impl<const A: usize, const B: usize, const A_NAME: &'static str, const B_NAME: &'static str>
            TransformSpec<Dim<A, Dim<B, Nil, B_NAME>, A_NAME>> for $spec
        where
            [(); A * B]:,
        {
            type OutputShape = Dim<A, Dim<B, Nil, B_NAME>, A_NAME>;
            const OUTPUT_SIZE: usize = A * B;
            type Runtime = LeafRuntime<$layer<{ A * B }>, { A * B }, { A * B }>;

            fn materialize(&self, _ctx: &mut MaterializeContext) -> Self::Runtime {
                LeafRuntime::new($layer::<{ A * B }>::init())
            }

            fn parameter_count(&self, _seen_shared: &mut HashSet<usize>) -> usize {
                0
            }

            fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
                input_axes.to_vec().into_boxed_slice()
            }

            fn description(&self) -> String {
                $desc.to_string()
            }
        }

        impl<
            const C: usize,
            const H: usize,
            const W: usize,
            const C_NAME: &'static str,
            const H_NAME: &'static str,
            const W_NAME: &'static str,
        > TransformSpec<Dim<C, Dim<H, Dim<W, Nil, W_NAME>, H_NAME>, C_NAME>> for $spec
        where
            [(); C * H * W]:,
        {
            type OutputShape = Dim<C, Dim<H, Dim<W, Nil, W_NAME>, H_NAME>, C_NAME>;
            const OUTPUT_SIZE: usize = C * H * W;
            type Runtime = LeafRuntime<$layer<{ C * H * W }>, { C * H * W }, { C * H * W }>;

            fn materialize(&self, _ctx: &mut MaterializeContext) -> Self::Runtime {
                LeafRuntime::new($layer::<{ C * H * W }>::init())
            }

            fn parameter_count(&self, _seen_shared: &mut HashSet<usize>) -> usize {
                0
            }

            fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
                input_axes.to_vec().into_boxed_slice()
            }

            fn description(&self) -> String {
                $desc.to_string()
            }
        }

        impl<
            const C: usize,
            const D: usize,
            const H: usize,
            const W: usize,
            const C_NAME: &'static str,
            const D_NAME: &'static str,
            const H_NAME: &'static str,
            const W_NAME: &'static str,
        > TransformSpec<Dim<C, Dim<D, Dim<H, Dim<W, Nil, W_NAME>, H_NAME>, D_NAME>, C_NAME>>
            for $spec
        where
            [(); C * D * H * W]:,
        {
            type OutputShape = Dim<C, Dim<D, Dim<H, Dim<W, Nil, W_NAME>, H_NAME>, D_NAME>, C_NAME>;
            const OUTPUT_SIZE: usize = C * D * H * W;
            type Runtime =
                LeafRuntime<$layer<{ C * D * H * W }>, { C * D * H * W }, { C * D * H * W }>;

            fn materialize(&self, _ctx: &mut MaterializeContext) -> Self::Runtime {
                LeafRuntime::new($layer::<{ C * D * H * W }>::init())
            }

            fn parameter_count(&self, _seen_shared: &mut HashSet<usize>) -> usize {
                0
            }

            fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
                input_axes.to_vec().into_boxed_slice()
            }

            fn description(&self) -> String {
                $desc.to_string()
            }
        }
    };
}

macro_rules! impl_flatten_transform {
    ([$($gen:tt)*] $shape:ty, $size:expr) => {
        impl<$($gen)*> TransformSpec<$shape> for FlattenSpec
        where
            [(); { $size }]:,
        {
            type OutputShape = Dim<{ $size }, Nil, "features">;
            const OUTPUT_SIZE: usize = $size;
            type Runtime = LeafRuntime<Flatten<{ $size }>, { $size }, { $size }>;

            fn materialize(&self, _ctx: &mut MaterializeContext) -> Self::Runtime {
                LeafRuntime::new(Flatten::<{ $size }>::init())
            }

            fn parameter_count(&self, _seen_shared: &mut HashSet<usize>) -> usize {
                0
            }

            fn output_axes(&self, _input_axes: &[Axis]) -> Box<[Axis]> {
                features_axis()
            }

            fn description(&self) -> String {
                "flatten".to_string()
            }
        }
    };
}

impl_pointwise_transform!(ReLUSpec, ReLU, "relu");
impl_pointwise_transform!(SigmoidSpec, Sigmoid, "sigmoid");
impl_pointwise_transform!(IdentitySpec, Flatten, "identity");

impl_flatten_transform!([const N: usize, const N_NAME: &'static str] Dim<N, Nil, N_NAME>, N);
impl_flatten_transform!(
    [
        const A: usize,
        const B: usize,
        const A_NAME: &'static str,
        const B_NAME: &'static str
    ] Dim<A, Dim<B, Nil, B_NAME>, A_NAME>,
    A * B
);
impl_flatten_transform!(
    [
        const C: usize,
        const H: usize,
        const W: usize,
        const C_NAME: &'static str,
        const H_NAME: &'static str,
        const W_NAME: &'static str
    ]
    Dim<C, Dim<H, Dim<W, Nil, W_NAME>, H_NAME>, C_NAME>,
    C * H * W
);
impl_flatten_transform!(
    [
        const C: usize,
        const D: usize,
        const H: usize,
        const W: usize,
        const C_NAME: &'static str,
        const D_NAME: &'static str,
        const H_NAME: &'static str,
        const W_NAME: &'static str
    ]
    Dim<C, Dim<D, Dim<H, Dim<W, Nil, W_NAME>, H_NAME>, D_NAME>, C_NAME>,
    C * D * H * W
);

impl<
    const C: usize,
    const H: usize,
    const W: usize,
    const C_NAME: &'static str,
    const H_NAME: &'static str,
    const W_NAME: &'static str,
    const OUT: usize,
    const KH: usize,
    const KW: usize,
    const STRIDE: usize,
    const PAD: usize,
> TransformSpec<Dim<C, Dim<H, Dim<W, Nil, W_NAME>, H_NAME>, C_NAME>>
    for ConvSpec<OUT, KH, KW, STRIDE, PAD>
where
    [(); C * H * W]:,
    [(); OUT * conv_out_dim(H, PAD, KH, STRIDE) * conv_out_dim(W, PAD, KW, STRIDE)]:,
    (): ConvKernelFitsInput<H, W, KH, KW, STRIDE, PAD>,
{
    type OutputShape = Dim<
        OUT,
        Dim<
            { conv_out_dim(H, PAD, KH, STRIDE) },
            Dim<{ conv_out_dim(W, PAD, KW, STRIDE) }, Nil, W_NAME>,
            H_NAME,
        >,
        C_NAME,
    >;
    const OUTPUT_SIZE: usize =
        OUT * conv_out_dim(H, PAD, KH, STRIDE) * conv_out_dim(W, PAD, KW, STRIDE);
    type Runtime = LeafRuntime<
        Conv<W, H, C, KH, KW, OUT, STRIDE, PAD>,
        { C * H * W },
        { OUT * conv_out_dim(H, PAD, KH, STRIDE) * conv_out_dim(W, PAD, KW, STRIDE) },
    >;

    fn materialize(&self, ctx: &mut MaterializeContext) -> Self::Runtime {
        LeafRuntime::new(
            Conv::<W, H, C, KH, KW, OUT, STRIDE, PAD>::with_initializer_and_rng(
                XavierUniform,
                &mut ctx.rng,
            ),
        )
    }

    fn parameter_count(&self, _seen_shared: &mut HashSet<usize>) -> usize {
        OUT * KH * KW * C + OUT
    }

    fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
        input_axes.to_vec().into_boxed_slice()
    }

    fn description(&self) -> String {
        if KH == KW {
            format!("conv({OUT}, kernel: {KH}, stride: {STRIDE}, pad: {PAD})")
        } else {
            format!("conv({OUT}, kernel: ({KH}, {KW}), stride: {STRIDE}, pad: {PAD})")
        }
    }
}
