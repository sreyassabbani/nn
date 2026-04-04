use std::{cell::RefCell, collections::HashSet, rc::Rc};

use crate::shape::{Dim, Nil, TensorShape};

use super::super::runtime::{
    ConcatRuntime, MaterializeContext, ResidualRuntime, SeqRuntime, SharedRuntime, SumRuntime,
};
use super::super::{
    Axis, ConcatAlong, ConcatSpec, RepeatStageSpec, ResidualSpec, SeqCompatible, SeqSpec,
    ShapePreserving, SharedSpec, SumSpec, TransformSpec, describe_shape,
};

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
