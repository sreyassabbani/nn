use syn::{Expr, Ident, LitInt};

#[derive(Clone)]
pub struct InputField {
    pub name: Ident,
    pub extent: Expr,
}

#[derive(Clone)]
pub struct InputSpec {
    pub fields: Vec<InputField>,
}

#[derive(Clone)]
pub enum KernelSpec {
    Scalar(Expr),
    Pair(Expr, Expr),
    Triple(Expr, Expr, Expr),
}

#[derive(Clone)]
pub struct DenseSpec {
    pub output: Expr,
    pub bias: bool,
}

#[derive(Clone)]
pub struct LinearSpec {
    pub output: Expr,
    pub bias: bool,
}

#[derive(Clone)]
pub struct ConvSpec {
    pub out_channels: Expr,
    pub kernel: KernelSpec,
    pub stride: Option<Expr>,
    pub pad: Option<Expr>,
    pub dilation: Option<Expr>,
}

#[derive(Clone)]
pub struct HeadAst {
    pub name: Ident,
    pub pipeline: PipelineAst,
}

#[derive(Clone)]
pub enum StepAst {
    Dense(DenseSpec),
    Linear(LinearSpec),
    Conv(Box<ConvSpec>),
    ReLU,
    Sigmoid,
    Flatten,
    Save {
        name: Ident,
    },
    SumFrom {
        name: Ident,
    },
    ConcatFrom {
        name: Ident,
        axis: Ident,
    },
    Ref(Expr),
    Share(Expr),
    Residual(Expr),
    Repeat {
        times: LitInt,
        body: Expr,
    },
    Concat {
        axis: Ident,
        branches: Vec<PipelineAst>,
    },
    Sum {
        branches: Vec<PipelineAst>,
    },
    Heads {
        heads: Vec<HeadAst>,
    },
}

#[derive(Clone)]
pub struct PipelineAst {
    pub steps: Vec<StepAst>,
}

#[derive(Clone)]
pub struct NetworkAst {
    pub input: Option<InputSpec>,
    pub pipeline: PipelineAst,
}
