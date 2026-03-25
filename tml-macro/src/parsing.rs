use crate::ast::{
    ConvSpec, DenseSpec, HeadAst, InputSpec, KernelSpec, NetworkAst, PipelineAst, StepAst,
};
use syn::parse::{Parse, ParseBuffer, ParseStream};
use syn::{Expr, ExprPath, Ident, LitBool, LitInt, Result, Token, parenthesized};

impl Parse for NetworkAst {
    fn parse(input: ParseStream) -> Result<Self> {
        let input_spec = if peek_ident(input, "input") {
            Some(parse_input(input)?)
        } else {
            None
        };

        if input_spec.is_some() && !input.is_empty() {
            input.parse::<Token![->]>()?;
        }

        let pipeline = parse_pipeline(input, |_| false)?;
        Ok(Self {
            input: input_spec,
            pipeline,
        })
    }
}

fn parse_pipeline<F>(input: ParseStream, should_stop: F) -> Result<PipelineAst>
where
    F: Fn(&ParseBuffer<'_>) -> bool,
{
    let mut steps = Vec::new();

    while !input.is_empty() && !should_stop(input) {
        let step = parse_step(input)?;
        let is_terminal_heads = matches!(step, StepAst::Heads { .. });
        steps.push(step);

        if input.is_empty() || should_stop(input) {
            break;
        }

        if is_terminal_heads {
            return Err(input.error("heads { ... } must be the final stage"));
        }

        input.parse::<Token![->]>()?;
        if input.is_empty() || should_stop(input) {
            return Err(input.error("expected a stage after ->"));
        }
    }

    Ok(PipelineAst { steps })
}

fn parse_step(input: ParseStream) -> Result<StepAst> {
    if let Some(keyword) = peek_ident_string(input) {
        return match keyword.as_str() {
            "dense" => parse_dense(input),
            "conv" => parse_conv(input),
            "relu" | "ReLU" => {
                let _: Ident = input.parse()?;
                Ok(StepAst::ReLU)
            }
            "sigmoid" | "Sigmoid" => {
                let _: Ident = input.parse()?;
                Ok(StepAst::Sigmoid)
            }
            "flatten" | "Flatten" => {
                let _: Ident = input.parse()?;
                Ok(StepAst::Flatten)
            }
            "share" => parse_share(input),
            "residual" => parse_residual(input),
            "repeat" => parse_repeat(input),
            "concat" => parse_concat(input),
            "sum" => parse_sum(input),
            "heads" => parse_heads(input),
            _ => Ok(StepAst::Ref(parse_reusable_expr(input)?)),
        };
    }

    Ok(StepAst::Ref(parse_reusable_expr(input)?))
}

fn parse_dense(input: ParseStream) -> Result<StepAst> {
    let _: Ident = input.parse()?;
    let content;
    parenthesized!(content in input);

    let output: Expr = content.parse()?;
    let mut bias = true;

    while content.peek(Token![,]) {
        content.parse::<Token![,]>()?;
        if content.is_empty() {
            break;
        }

        let key: Ident = content.parse()?;
        content.parse::<Token![:]>()?;
        match key.to_string().as_str() {
            "bias" => {
                let value: LitBool = content.parse()?;
                bias = value.value;
            }
            _ => return Err(syn::Error::new(key.span(), "unsupported dense option")),
        }
    }

    Ok(StepAst::Dense(DenseSpec { output, bias }))
}

fn parse_conv(input: ParseStream) -> Result<StepAst> {
    let _: Ident = input.parse()?;
    let content;
    parenthesized!(content in input);

    let out_channels: Expr = content.parse()?;
    let mut kernel: Option<KernelSpec> = None;
    let mut stride = None;
    let mut pad = None;
    let mut dilation = None;

    while content.peek(Token![,]) {
        content.parse::<Token![,]>()?;
        if content.is_empty() {
            break;
        }

        let key: Ident = content.parse()?;
        content.parse::<Token![:]>()?;
        match key.to_string().as_str() {
            "kernel" => {
                kernel = Some(parse_kernel_value(&content)?);
            }
            "stride" => stride = Some(content.parse()?),
            "pad" => pad = Some(content.parse()?),
            "dilation" => dilation = Some(content.parse()?),
            _ => return Err(syn::Error::new(key.span(), "unsupported conv option")),
        }
    }

    let Some(kernel) = kernel else {
        return Err(content.error("conv(...) requires kernel: ..."));
    };

    Ok(StepAst::Conv(Box::new(ConvSpec {
        out_channels,
        kernel,
        stride,
        pad,
        dilation,
    })))
}

fn parse_share(input: ParseStream) -> Result<StepAst> {
    let _: Ident = input.parse()?;
    let content;
    parenthesized!(content in input);
    let expr: Expr = content.parse()?;
    Ok(StepAst::Share(expr))
}

fn parse_residual(input: ParseStream) -> Result<StepAst> {
    let _: Ident = input.parse()?;
    let content;
    parenthesized!(content in input);
    let expr: Expr = content.parse()?;
    Ok(StepAst::Residual(expr))
}

fn parse_repeat(input: ParseStream) -> Result<StepAst> {
    let _: Ident = input.parse()?;
    let content;
    parenthesized!(content in input);
    let times: LitInt = content.parse()?;
    content.parse::<Token![,]>()?;
    let body: Expr = content.parse()?;
    Ok(StepAst::Repeat { times, body })
}

fn parse_concat(input: ParseStream) -> Result<StepAst> {
    let _: Ident = input.parse()?;
    let axis_content;
    parenthesized!(axis_content in input);
    let axis: Ident = axis_content.parse()?;
    if !axis_content.is_empty() {
        return Err(axis_content.error("concat(axis) expects a single axis name"));
    }

    let branches_content;
    syn::bracketed!(branches_content in input);
    let branches = parse_branch_list(&branches_content)?;

    Ok(StepAst::Concat { axis, branches })
}

fn parse_sum(input: ParseStream) -> Result<StepAst> {
    let _: Ident = input.parse()?;
    let branches_content;
    syn::bracketed!(branches_content in input);
    let branches = parse_branch_list(&branches_content)?;
    Ok(StepAst::Sum { branches })
}

fn parse_heads(input: ParseStream) -> Result<StepAst> {
    let _: Ident = input.parse()?;
    let content;
    syn::braced!(content in input);

    let mut heads = Vec::new();
    while !content.is_empty() {
        let name: Ident = content.parse()?;
        content.parse::<Token![:]>()?;
        let pipeline = parse_pipeline(&content, |stream| stream.peek(Token![,]))?;
        heads.push(HeadAst { name, pipeline });
        if content.peek(Token![,]) {
            content.parse::<Token![,]>()?;
        }
    }

    if heads.is_empty() {
        return Err(content.error("heads { ... } must declare at least one head"));
    }

    Ok(StepAst::Heads { heads })
}

fn parse_branch_list(input: &ParseBuffer<'_>) -> Result<Vec<PipelineAst>> {
    let mut branches = Vec::new();
    while !input.is_empty() {
        branches.push(parse_pipeline(input, |stream| stream.peek(Token![,]))?);
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
        }
    }

    if branches.is_empty() {
        return Err(input.error("expected at least one branch"));
    }

    Ok(branches)
}

fn parse_input(input: ParseStream) -> Result<InputSpec> {
    let _: Ident = input.parse()?;
    let content;
    parenthesized!(content in input);

    let fields = parse_named_fields(&content)?;
    match fields.as_slice() {
        [(name, features)] if name == "features" => Ok(InputSpec::Features {
            features: features.clone(),
        }),
        [(a, channels), (b, height), (c, width)]
            if a == "channels" && b == "height" && c == "width" =>
        {
            Ok(InputSpec::Image {
                channels: channels.clone(),
                height: height.clone(),
                width: width.clone(),
            })
        }
        [(a, channels), (b, depth), (c, height), (d, width)]
            if a == "channels" && b == "depth" && c == "height" && d == "width" =>
        {
            Ok(InputSpec::Volume {
                channels: channels.clone(),
                depth: depth.clone(),
                height: height.clone(),
                width: width.clone(),
            })
        }
        _ => Err(content.error(
            "input(...) expects exactly one of: features: N | channels: C, height: H, width: W | channels: C, depth: D, height: H, width: W",
        )),
    }
}

fn parse_named_fields(input: &ParseBuffer<'_>) -> Result<Vec<(Ident, Expr)>> {
    let mut fields = Vec::new();
    while !input.is_empty() {
        let name: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        let value: Expr = input.parse()?;
        fields.push((name, value));
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
        }
    }
    Ok(fields)
}

fn parse_kernel_value(input: &ParseBuffer<'_>) -> Result<KernelSpec> {
    if input.peek(syn::token::Paren) {
        let content;
        parenthesized!(content in input);
        let values = parse_expr_list(&content)?;
        return match values.as_slice() {
            [a, b] => Ok(KernelSpec::Pair(a.clone(), b.clone())),
            [a, b, c] => Ok(KernelSpec::Triple(a.clone(), b.clone(), c.clone())),
            _ => Err(content.error("kernel expects either a scalar or a 2/3-tuple")),
        };
    }

    Ok(KernelSpec::Scalar(input.parse()?))
}

fn parse_expr_list(content: &ParseBuffer<'_>) -> Result<Vec<Expr>> {
    let mut values = Vec::new();
    if content.is_empty() {
        return Ok(values);
    }

    values.push(content.parse::<Expr>()?);
    while content.peek(Token![,]) {
        content.parse::<Token![,]>()?;
        if content.is_empty() {
            break;
        }
        values.push(content.parse::<Expr>()?);
    }
    Ok(values)
}

fn parse_reusable_expr(input: ParseStream) -> Result<Expr> {
    if input.peek(Ident) && !input.peek2(Token![!]) {
        let path = input.parse::<syn::Path>()?;
        return Ok(Expr::Path(ExprPath {
            attrs: Vec::new(),
            qself: None,
            path,
        }));
    }

    input.parse()
}

fn peek_ident(input: ParseStream, expected: &str) -> bool {
    matches!(peek_ident_string(input).as_deref(), Some(name) if name == expected)
}

fn peek_ident_string(input: ParseStream) -> Option<String> {
    let fork = input.fork();
    fork.parse::<Ident>().ok().map(|ident| ident.to_string())
}
