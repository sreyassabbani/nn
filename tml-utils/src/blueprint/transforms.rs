use std::{cell::RefCell, collections::HashSet, rc::Rc};

use crate::__private::ConvKernelFitsInput;
use crate::conv::{Conv, conv_out_dim};
use crate::network::{DenseLayer, Flatten, ReLU, Sigmoid, XavierUniform};
use crate::shape::{Dim, Nil, TensorShape};

use super::runtime::{
    ConcatRuntime, LeafRuntime, MaterializeContext, ResidualRuntime, SeqRuntime, SharedRuntime,
    SumRuntime,
};
use super::{
    Axis, Blueprint, ConcatAlong, ConcatSpec, ConvSpec, DenseExpectsFlatInput, DenseSpec,
    FlattenSpec, IdentitySpec, ReLUSpec, RepeatStageSpec, ResidualSpec, SeqCompatible, SeqSpec,
    ShapePreserving, SharedSpec, SigmoidSpec, SumSpec, TransformSpec, describe_shape,
    features_axis,
};

pub fn dense<const OUT: usize>() -> Blueprint<DenseSpec<OUT, true>> {
    Blueprint::new(DenseSpec)
}

pub fn dense_no_bias<const OUT: usize>() -> Blueprint<DenseSpec<OUT, false>> {
    Blueprint::new(DenseSpec)
}

pub fn relu() -> Blueprint<ReLUSpec> {
    Blueprint::new(ReLUSpec)
}

pub fn sigmoid() -> Blueprint<SigmoidSpec> {
    Blueprint::new(SigmoidSpec)
}

pub fn identity() -> Blueprint<IdentitySpec> {
    Blueprint::new(IdentitySpec)
}

pub fn flatten() -> Blueprint<FlattenSpec> {
    Blueprint::new(FlattenSpec)
}

pub fn conv<
    const OUT: usize,
    const KH: usize,
    const KW: usize,
    const STRIDE: usize,
    const PAD: usize,
>() -> Blueprint<ConvSpec<OUT, KH, KW, STRIDE, PAD>> {
    Blueprint::new(ConvSpec)
}

pub fn residual<Spec>(body: Blueprint<Spec>) -> Blueprint<ResidualSpec<Spec>> {
    Blueprint::new(ResidualSpec {
        body: body.into_inner(),
    })
}

pub fn sum<Left, Right>(
    left: Blueprint<Left>,
    right: Blueprint<Right>,
) -> Blueprint<SumSpec<Left, Right>> {
    Blueprint::new(SumSpec {
        left: left.into_inner(),
        right: right.into_inner(),
    })
}

pub fn concat<Left, Right>(
    axis: Axis,
    left: Blueprint<Left>,
    right: Blueprint<Right>,
) -> Blueprint<ConcatSpec<Left, Right>> {
    Blueprint::new(ConcatSpec {
        axis,
        left: left.into_inner(),
        right: right.into_inner(),
    })
}

pub fn share<Spec>(blueprint: &Blueprint<Spec>) -> Blueprint<SharedSpec<Spec>>
where
    Spec: Clone,
{
    Blueprint::new(SharedSpec {
        id: blueprint as *const _ as usize,
        inner: blueprint.as_inner().clone(),
    })
}

#[doc(hidden)]
pub fn repeat_stage<Spec>(blueprint: Blueprint<Spec>) -> Blueprint<RepeatStageSpec<Spec>> {
    Blueprint::new(RepeatStageSpec {
        inner: blueprint.into_inner(),
    })
}

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

macro_rules! impl_seq_compatible {
    ([$($gen:tt)*] $shape:ty, $size:expr) => {
        impl<$($gen)*, Left, Right> SeqCompatible<$shape, Left, Right> for $shape
        where
            Left: TransformSpec<$shape>,
            Right: TransformSpec<Left::OutputShape>,
            [(); { $size }]:,
        {
            type OutputShape = Right::OutputShape;
            const OUTPUT_SIZE: usize = Right::OUTPUT_SIZE;
            type Runtime = SeqRuntime<Left::Runtime, Right::Runtime>;

            fn materialize_seq(
                left: &Left,
                right: &Right,
                ctx: &mut MaterializeContext,
            ) -> Self::Runtime {
                SeqRuntime {
                    left: left.materialize(ctx),
                    right: right.materialize(ctx),
                }
            }
        }
    };
}

impl_seq_compatible!([const N: usize, const N_NAME: &'static str] Dim<N, Nil, N_NAME>, N);
impl_seq_compatible!(
    [
        const A: usize,
        const B: usize,
        const A_NAME: &'static str,
        const B_NAME: &'static str
    ] Dim<A, Dim<B, Nil, B_NAME>, A_NAME>,
    A * B
);
impl_seq_compatible!(
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
impl_seq_compatible!(
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

impl<InputShape, Left, Right> TransformSpec<InputShape> for SeqSpec<Left, Right>
where
    InputShape: TensorShape + 'static,
    Left: TransformSpec<InputShape>,
    Right: TransformSpec<Left::OutputShape>,
    InputShape: SeqCompatible<InputShape, Left, Right>,
{
    type OutputShape = <InputShape as SeqCompatible<InputShape, Left, Right>>::OutputShape;
    const OUTPUT_SIZE: usize = <InputShape as SeqCompatible<InputShape, Left, Right>>::OUTPUT_SIZE;
    type Runtime = <InputShape as SeqCompatible<InputShape, Left, Right>>::Runtime;

    fn materialize(&self, ctx: &mut MaterializeContext) -> Self::Runtime {
        <InputShape as SeqCompatible<InputShape, Left, Right>>::materialize_seq(
            &self.left,
            &self.right,
            ctx,
        )
    }

    fn push_summary(&self, lines: &mut Vec<String>) {
        <Left as TransformSpec<InputShape>>::push_summary(&self.left, lines);
        <Right as TransformSpec<Left::OutputShape>>::push_summary(&self.right, lines);
    }

    fn push_shape_trace(&self, input_axes: &[Axis], lines: &mut Vec<String>) -> Box<[Axis]> {
        let mid_axes =
            <Left as TransformSpec<InputShape>>::push_shape_trace(&self.left, input_axes, lines);
        <Right as TransformSpec<Left::OutputShape>>::push_shape_trace(&self.right, &mid_axes, lines)
    }

    fn parameter_count(&self, seen_shared: &mut HashSet<usize>) -> usize {
        <Left as TransformSpec<InputShape>>::parameter_count(&self.left, seen_shared)
            + <Right as TransformSpec<Left::OutputShape>>::parameter_count(&self.right, seen_shared)
    }

    fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
        let mid = self.left.output_axes(input_axes);
        self.right.output_axes(&mid)
    }

    fn description(&self) -> String {
        "sequence".to_string()
    }
}

impl<InputShape, Body> TransformSpec<InputShape> for ResidualSpec<Body>
where
    InputShape: TensorShape + 'static,
    Body: ShapePreserving<InputShape>,
    [(); InputShape::SIZE]:,
{
    type OutputShape = InputShape;
    const OUTPUT_SIZE: usize = InputShape::SIZE;
    type Runtime = ResidualRuntime<Body::Runtime>;

    fn materialize(&self, ctx: &mut MaterializeContext) -> Self::Runtime {
        ResidualRuntime {
            inner: self.body.materialize(ctx),
        }
    }

    fn push_summary(&self, lines: &mut Vec<String>) {
        lines.push("residual".to_string());
        <Body as TransformSpec<InputShape>>::push_summary(&self.body, lines);
    }

    fn push_shape_trace(&self, input_axes: &[Axis], lines: &mut Vec<String>) -> Box<[Axis]> {
        lines.push(format!(
            "residual: {} -> {}",
            describe_shape::<InputShape>(input_axes),
            describe_shape::<InputShape>(input_axes)
        ));
        <Body as TransformSpec<InputShape>>::push_shape_trace(&self.body, input_axes, lines);
        input_axes.to_vec().into_boxed_slice()
    }

    fn parameter_count(&self, seen_shared: &mut HashSet<usize>) -> usize {
        <Body as TransformSpec<InputShape>>::parameter_count(&self.body, seen_shared)
    }

    fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
        input_axes.to_vec().into_boxed_slice()
    }

    fn description(&self) -> String {
        "residual".to_string()
    }
}

impl<InputShape, Left, Right> TransformSpec<InputShape> for SumSpec<Left, Right>
where
    InputShape: TensorShape + 'static,
    Left: TransformSpec<InputShape>,
    Right: TransformSpec<InputShape, OutputShape = Left::OutputShape>,
    [(); InputShape::SIZE]:,
    [(); Left::OUTPUT_SIZE]:,
{
    type OutputShape = Left::OutputShape;
    const OUTPUT_SIZE: usize = Left::OUTPUT_SIZE;
    type Runtime = SumRuntime<Left::Runtime, Right::Runtime>;

    fn materialize(&self, ctx: &mut MaterializeContext) -> Self::Runtime {
        SumRuntime {
            left: self.left.materialize(ctx),
            right: self.right.materialize(ctx),
        }
    }

    fn parameter_count(&self, seen_shared: &mut HashSet<usize>) -> usize {
        <Left as TransformSpec<InputShape>>::parameter_count(&self.left, seen_shared)
            + <Right as TransformSpec<InputShape>>::parameter_count(&self.right, seen_shared)
    }

    fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
        self.left.output_axes(input_axes)
    }

    fn description(&self) -> String {
        "sum".to_string()
    }
}

impl<InputShape, const A: usize, const B: usize, const NAME: &'static str>
    ConcatAlong<InputShape, Dim<A, Nil, NAME>, Dim<B, Nil, NAME>> for ()
where
    InputShape: TensorShape + 'static,
    [(); A + B]:,
{
    type OutputShape = Dim<{ A + B }, Nil, NAME>;
    fn axis_ok(axis: Axis) -> bool {
        axis == Axis::new(NAME)
    }
}

impl<
    InputShape,
    const C1: usize,
    const C2: usize,
    const H: usize,
    const W: usize,
    const C_NAME: &'static str,
    const H_NAME: &'static str,
    const W_NAME: &'static str,
>
    ConcatAlong<
        InputShape,
        Dim<C1, Dim<H, Dim<W, Nil, W_NAME>, H_NAME>, C_NAME>,
        Dim<C2, Dim<H, Dim<W, Nil, W_NAME>, H_NAME>, C_NAME>,
    > for ()
where
    InputShape: TensorShape + 'static,
    [(); C1 * H * W]:,
    [(); C2 * H * W]:,
    [(); (C1 + C2) * H * W]:,
{
    type OutputShape = Dim<{ C1 + C2 }, Dim<H, Dim<W, Nil, W_NAME>, H_NAME>, C_NAME>;
    fn axis_ok(axis: Axis) -> bool {
        axis == Axis::new(C_NAME)
    }
}

impl<InputShape, Left, Right> TransformSpec<InputShape> for ConcatSpec<Left, Right>
where
    InputShape: TensorShape + 'static,
    Left: TransformSpec<InputShape>,
    Right: TransformSpec<InputShape>,
    (): ConcatAlong<InputShape, Left::OutputShape, Right::OutputShape>,
    [(); InputShape::SIZE]:,
    [(); Left::OUTPUT_SIZE]:,
    [(); Right::OUTPUT_SIZE]:,
{
    type OutputShape =
        <() as ConcatAlong<InputShape, Left::OutputShape, Right::OutputShape>>::OutputShape;
    const OUTPUT_SIZE: usize = Left::OUTPUT_SIZE + Right::OUTPUT_SIZE;
    type Runtime = ConcatRuntime<Left::Runtime, Right::Runtime>;

    fn materialize(&self, ctx: &mut MaterializeContext) -> Self::Runtime {
        assert!(
            <() as ConcatAlong<InputShape, Left::OutputShape, Right::OutputShape>>::axis_ok(
                self.axis
            ),
            "unsupported concat axis for the current shapes"
        );
        ConcatRuntime {
            left: self.left.materialize(ctx),
            right: self.right.materialize(ctx),
        }
    }

    fn parameter_count(&self, seen_shared: &mut HashSet<usize>) -> usize {
        <Left as TransformSpec<InputShape>>::parameter_count(&self.left, seen_shared)
            + <Right as TransformSpec<InputShape>>::parameter_count(&self.right, seen_shared)
    }

    fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
        input_axes.to_vec().into_boxed_slice()
    }

    fn description(&self) -> String {
        format!("concat({})", self.axis.as_str())
    }
}

impl<InputShape, Spec> TransformSpec<InputShape> for RepeatStageSpec<Spec>
where
    InputShape: TensorShape + 'static,
    Spec: ShapePreserving<InputShape>,
    [(); InputShape::SIZE]:,
{
    type OutputShape = InputShape;
    const OUTPUT_SIZE: usize = InputShape::SIZE;
    type Runtime = Spec::Runtime;

    fn materialize(&self, ctx: &mut MaterializeContext) -> Self::Runtime {
        self.inner.materialize(ctx)
    }

    fn push_summary(&self, lines: &mut Vec<String>) {
        <Spec as TransformSpec<InputShape>>::push_summary(&self.inner, lines);
    }

    fn push_shape_trace(&self, input_axes: &[Axis], lines: &mut Vec<String>) -> Box<[Axis]> {
        <Spec as TransformSpec<InputShape>>::push_shape_trace(&self.inner, input_axes, lines)
    }

    fn parameter_count(&self, seen_shared: &mut HashSet<usize>) -> usize {
        <Spec as TransformSpec<InputShape>>::parameter_count(&self.inner, seen_shared)
    }

    fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
        self.inner.output_axes(input_axes)
    }

    fn description(&self) -> String {
        self.inner.description()
    }
}

impl<InputShape, Spec> TransformSpec<InputShape> for SharedSpec<Spec>
where
    InputShape: TensorShape + 'static,
    Spec: TransformSpec<InputShape>,
    Spec::Runtime: 'static,
    [(); InputShape::SIZE]:,
    [(); Spec::OUTPUT_SIZE]:,
{
    type OutputShape = Spec::OutputShape;
    const OUTPUT_SIZE: usize = Spec::OUTPUT_SIZE;
    type Runtime = SharedRuntime<Spec::Runtime>;

    fn materialize(&self, ctx: &mut MaterializeContext) -> Self::Runtime {
        if let Some(existing) = ctx.shared.get(&self.id) {
            let rc = existing
                .downcast_ref::<Rc<RefCell<Spec::Runtime>>>()
                .expect("shared runtime type mismatch")
                .clone();
            return SharedRuntime { inner: rc };
        }

        let runtime = self.inner.materialize(ctx);
        let rc = Rc::new(RefCell::new(runtime));
        ctx.shared.insert(self.id, Box::new(rc.clone()));
        SharedRuntime { inner: rc }
    }

    fn parameter_count(&self, seen_shared: &mut HashSet<usize>) -> usize {
        if seen_shared.insert(self.id) {
            <Spec as TransformSpec<InputShape>>::parameter_count(&self.inner, seen_shared)
        } else {
            0
        }
    }

    fn output_axes(&self, input_axes: &[Axis]) -> Box<[Axis]> {
        self.inner.output_axes(input_axes)
    }

    fn description(&self) -> String {
        format!("share({})", self.id)
    }
}
