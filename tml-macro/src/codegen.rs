use crate::ast::{
    ConvSpec, DenseSpec, HeadAst, InputSpec, KernelSpec, NetworkAst, PipelineAst, StepAst,
};
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use std::collections::BTreeMap;
use syn::{Error, Expr, ExprLit, ExprPath, ExprTuple, Lit, Result, spanned::Spanned};

pub fn generate_network(ast: &NetworkAst) -> Result<TokenStream2> {
    match ast.input.as_ref() {
        Some(input) => generate_rooted_network(input, &ast.pipeline),
        None => lower_pipeline_expr(&ast.pipeline),
    }
}

fn generate_rooted_network(input: &InputSpec, pipeline: &PipelineAst) -> Result<TokenStream2> {
    if let Some((prefix_steps, heads)) = split_terminal_heads(&pipeline.steps)? {
        return generate_headed_root(input, prefix_steps, heads);
    }

    let spec = lower_pipeline_expr(pipeline)?;
    Ok(wrap_root(input, spec))
}

fn split_terminal_heads(steps: &[StepAst]) -> Result<Option<(&[StepAst], &[HeadAst])>> {
    for (idx, step) in steps.iter().enumerate() {
        if let StepAst::Heads { heads } = step {
            if idx + 1 != steps.len() {
                return Err(Error::new(
                    Span::call_site(),
                    "heads { ... } must be the final stage",
                ));
            }
            return Ok(Some((&steps[..idx], heads.as_slice())));
        }
    }

    Ok(None)
}

fn lower_pipeline_expr(pipeline: &PipelineAst) -> Result<TokenStream2> {
    let mut current: Option<TokenStream2> = None;
    let mut saved = BTreeMap::<String, TokenStream2>::new();

    for step in &pipeline.steps {
        if let StepAst::Heads { .. } = step {
            return Err(Error::new(
                Span::call_site(),
                "heads { ... } is only valid in a rooted network with input(...)",
            ));
        }

        match step {
            StepAst::Save { name } => {
                let snapshot = current
                    .clone()
                    .unwrap_or_else(|| quote! { ::tml::identity() });
                let key = name.to_string();
                if saved.insert(key.clone(), snapshot).is_some() {
                    return Err(Error::new(
                        name.span(),
                        format!("saved source `{key}` is already defined in this pipeline"),
                    ));
                }
            }
            StepAst::SumFrom { name } => {
                let saved_expr = load_saved_source(&saved, name)?;
                let acc = current
                    .take()
                    .unwrap_or_else(|| quote! { ::tml::identity() });
                current = Some(quote! { ::tml::sum(#acc, #saved_expr) });
            }
            StepAst::ConcatFrom { name, axis } => {
                let saved_expr = load_saved_source(&saved, name)?;
                let acc = current
                    .take()
                    .unwrap_or_else(|| quote! { ::tml::identity() });
                let axis = axis_tokens(axis)?;
                current = Some(quote! { ::tml::concat(#axis, #acc, #saved_expr) });
            }
            _ => {
                let next = lower_step(step)?;
                current = Some(match current {
                    Some(acc) => quote! { (#acc).then(#next) },
                    None => next,
                });
            }
        }
    }

    Ok(current.unwrap_or_else(|| quote! { ::tml::identity() }))
}

fn lower_step(step: &StepAst) -> Result<TokenStream2> {
    match step {
        StepAst::Dense(spec) => lower_dense(spec),
        StepAst::Conv(spec) => lower_conv(spec),
        StepAst::ReLU => Ok(quote! { ::tml::relu() }),
        StepAst::Sigmoid => Ok(quote! { ::tml::sigmoid() }),
        StepAst::Flatten => Ok(quote! { ::tml::flatten() }),
        StepAst::Save { .. } | StepAst::SumFrom { .. } | StepAst::ConcatFrom { .. } => {
            Err(Error::new(
                Span::call_site(),
                "internal lowering error: stateful source stages must be handled by the pipeline lowerer",
            ))
        }
        StepAst::Ref(expr) => lower_reusable_expr(expr),
        StepAst::Share(expr) => lower_share(expr),
        StepAst::Residual(expr) => {
            let inner = lower_reusable_expr(expr)?;
            Ok(quote! { ::tml::residual(#inner) })
        }
        StepAst::Repeat { times, body } => lower_repeat(times, body),
        StepAst::Concat { axis, branches } => lower_concat(axis, branches),
        StepAst::Sum { branches } => lower_sum(branches),
        StepAst::Heads { .. } => Err(Error::new(
            Span::call_site(),
            "heads { ... } can only appear as the terminal stage of a rooted network",
        )),
    }
}

fn lower_dense(spec: &DenseSpec) -> Result<TokenStream2> {
    let output = &spec.output;
    Ok(if spec.bias {
        quote! { ::tml::dense::<{ #output }>() }
    } else {
        quote! { ::tml::dense_no_bias::<{ #output }>() }
    })
}

fn lower_conv(spec: &ConvSpec) -> Result<TokenStream2> {
    let out = &spec.out_channels;
    let (kh, kw) = lower_kernel(&spec.kernel)?;
    let stride = scalar_expr(spec.stride.as_ref(), "stride", quote! { 1 })?;
    let pad = scalar_expr(spec.pad.as_ref(), "pad", quote! { 0 })?;

    if let Some(dilation) = &spec.dilation {
        match dilation {
            Expr::Lit(ExprLit {
                lit: Lit::Int(value),
                ..
            }) if value.base10_parse::<usize>()? == 1 => {}
            _ => {
                return Err(Error::new(
                    dilation.span(),
                    "conv dilation is not implemented yet; only dilation: 1 is currently accepted",
                ));
            }
        }
    }

    Ok(quote! {
        ::tml::conv::<{ #out }, { #kh }, { #kw }, { #stride }, { #pad }>()
    })
}

fn lower_kernel(kernel: &KernelSpec) -> Result<(TokenStream2, TokenStream2)> {
    match kernel {
        KernelSpec::Scalar(expr) => Ok((quote! { #expr }, quote! { #expr })),
        KernelSpec::Pair(h, w) => Ok((quote! { #h }, quote! { #w })),
        KernelSpec::Triple(a, b, c) => Err(Error::new(
            c.span(),
            format!(
                "3D conv kernels are not implemented yet; got kernel: ({}, {}, {})",
                quote! { #a },
                quote! { #b },
                quote! { #c }
            ),
        )),
    }
}

fn scalar_expr(expr: Option<&Expr>, label: &str, default: TokenStream2) -> Result<TokenStream2> {
    let Some(expr) = expr else {
        return Ok(default);
    };

    if matches!(expr, Expr::Tuple(ExprTuple { .. })) {
        return Err(Error::new(
            expr.span(),
            format!("{label} tuples are not implemented yet"),
        ));
    }

    Ok(quote! { #expr })
}

fn lower_reusable_expr(expr: &Expr) -> Result<TokenStream2> {
    if let Some(name) = path_name(expr) {
        return match name.as_str() {
            "relu" => Ok(quote! { ::tml::relu() }),
            "sigmoid" => Ok(quote! { ::tml::sigmoid() }),
            "flatten" => Ok(quote! { ::tml::flatten() }),
            "output" => Err(Error::new(
                expr.span(),
                "-> output has been removed; the final stage is already the output",
            )),
            _ => Ok(quote! { ::tml::into_blueprint((#expr).clone()) }),
        };
    }

    if let Some(name) = call_name(expr) {
        return match name.as_str() {
            "share" => lower_share_expr(&call_arg(expr)?),
            "dense" | "conv" | "residual" | "repeat" | "relu" | "sigmoid" | "flatten" => {
                Err(Error::new(
                    expr.span(),
                    "inline transform calls belong directly in network!; assign chunks with let or use the transform syntax directly",
                ))
            }
            "output" => Err(Error::new(
                expr.span(),
                "-> output has been removed; the final stage is already the output",
            )),
            _ => Ok(quote! { ::tml::into_blueprint((#expr).clone()) }),
        };
    }

    Ok(quote! { ::tml::into_blueprint((#expr).clone()) })
}

fn lower_share(expr: &Expr) -> Result<TokenStream2> {
    lower_share_expr(expr)
}

fn lower_share_expr(expr: &Expr) -> Result<TokenStream2> {
    let Expr::Path(ExprPath { path, .. }) = expr else {
        return Err(Error::new(
            expr.span(),
            "share(...) only accepts a previously bound fragment identifier in this redesign",
        ));
    };

    if path.segments.len() != 1 {
        return Err(Error::new(
            expr.span(),
            "share(...) only accepts a local fragment binding, not a path expression",
        ));
    }

    let share_id = quote! {
        ::tml::shared_name_id(concat!(module_path!(), "::", stringify!(#expr)))
    };

    Ok(quote! { ::tml::share_fragment_with_id(&#expr, #share_id) })
}

fn lower_repeat(times: &syn::LitInt, body: &Expr) -> Result<TokenStream2> {
    let n = times.base10_parse::<usize>()?;
    let body_expr = lower_reusable_expr(body)?;
    if n == 0 {
        return Ok(quote! { ::tml::identity() });
    }

    let mut stages = Vec::with_capacity(n);
    for _ in 0..n {
        stages.push(quote! { __tml_repeat.clone() });
    }

    let mut chain = stages[0].clone();
    for stage in stages.iter().skip(1) {
        chain = quote! { (#chain).then(#stage) };
    }

    Ok(quote! {{
        let __tml_repeat = ::tml::repeat_stage(#body_expr);
        #chain
    }})
}

fn lower_concat(axis: &syn::Ident, branches: &[PipelineAst]) -> Result<TokenStream2> {
    let axis = axis_tokens(axis)?;
    fold_branches(branches, |left, right| {
        quote! { ::tml::concat(#axis, #left, #right) }
    })
}

fn lower_sum(branches: &[PipelineAst]) -> Result<TokenStream2> {
    fold_branches(branches, |left, right| {
        quote! { ::tml::sum(#left, #right) }
    })
}

fn load_saved_source(
    saved: &BTreeMap<String, TokenStream2>,
    name: &syn::Ident,
) -> Result<TokenStream2> {
    saved
        .get(&name.to_string())
        .cloned()
        .ok_or_else(|| Error::new(name.span(), format!("unknown saved source `{}`", name)))
}

fn fold_branches<F>(branches: &[PipelineAst], combine: F) -> Result<TokenStream2>
where
    F: Fn(TokenStream2, TokenStream2) -> TokenStream2,
{
    let mut it = branches.iter();
    let first = it
        .next()
        .ok_or_else(|| Error::new(Span::call_site(), "expected at least one branch"))?;
    let mut acc = lower_pipeline_expr(first)?;
    for branch in it {
        let next = lower_pipeline_expr(branch)?;
        acc = combine(acc, next);
    }
    Ok(acc)
}

fn generate_headed_root(
    input: &InputSpec,
    prefix_steps: &[StepAst],
    heads: &[HeadAst],
) -> Result<TokenStream2> {
    let prefix_expr = if prefix_steps.is_empty() {
        quote! { ::tml::identity() }
    } else {
        lower_pipeline_expr(&PipelineAst {
            steps: prefix_steps.to_vec(),
        })?
    };

    let input_shape = input_shape_ty(input);
    let input_size = input_size_expr(input);

    let spec_ident = format_ident!("__TmlHeadsSpec");
    let runtime_ident = format_ident!("__TmlHeadsRuntime");
    let output_ident = format_ident!("__TmlHeadsOutput");

    let head_type_idents = (0..heads.len())
        .map(|idx| format_ident!("HeadSpec{idx}"))
        .collect::<Vec<_>>();
    let head_runtime_idents = (0..heads.len())
        .map(|idx| format_ident!("HeadRt{idx}"))
        .collect::<Vec<_>>();
    let head_field_idents = (0..heads.len())
        .map(|idx| format_ident!("head_{idx}"))
        .collect::<Vec<_>>();
    let head_output_idents = (0..heads.len())
        .map(|idx| format_ident!("HEAD_{idx}_OUT"))
        .collect::<Vec<_>>();

    let head_names = heads.iter().map(|head| &head.name).collect::<Vec<_>>();
    let head_exprs = heads
        .iter()
        .map(|head| lower_pipeline_expr(&head.pipeline))
        .collect::<Result<Vec<_>>>()?;

    let rooted_heads = {
        let spec = quote! {
            ::tml::Blueprint::new(#spec_ident {
                prefix: #prefix_expr,
                #(#head_field_idents: #head_exprs,)*
            })
        };
        match input {
            InputSpec::Features { features } => {
                quote! { ::tml::features_input::<{ #features }, _>(#spec) }
            }
            InputSpec::Image {
                channels,
                height,
                width,
            } => {
                quote! {
                    ::tml::image_input::<{ #channels }, { #height }, { #width }, _>(#spec)
                }
            }
            InputSpec::Volume {
                channels,
                depth,
                height,
                width,
            } => {
                quote! {
                    ::tml::volume_input::<{ #channels }, { #depth }, { #height }, { #width }, _>(#spec)
                }
            }
        }
    };

    let head_summary_lines = head_names
        .iter()
        .zip(head_type_idents.iter())
        .zip(head_field_idents.iter())
        .map(|((name, ty), field)| {
            quote! {
                lines.push(format!("head {}", stringify!(#name)));
                <#ty as ::tml::TransformSpec<PrefixSpec::OutputShape>>::push_summary(
                    self.#field.as_inner(),
                    lines,
                );
            }
        });
    let head_trace_lines = head_names
        .iter()
        .zip(head_type_idents.iter())
        .zip(head_field_idents.iter())
        .map(|((name, ty), field)| {
            quote! {
                lines.push(format!("head {}", stringify!(#name)));
                <#ty as ::tml::TransformSpec<PrefixSpec::OutputShape>>::push_shape_trace(
                    self.#field.as_inner(),
                    &mid_axes,
                    lines,
                );
            }
        });
    let head_param_counts =
        head_type_idents
            .iter()
            .zip(head_field_idents.iter())
            .map(|(ty, field)| {
                quote! {
                    + <#ty as ::tml::TransformSpec<PrefixSpec::OutputShape>>::parameter_count(
                        self.#field.as_inner(),
                        seen_shared,
                    )
                }
            });

    let head_workspace_inits = head_names
        .iter()
        .zip(head_runtime_idents.iter())
        .zip(head_field_idents.iter())
        .zip(head_output_idents.iter())
        .map(|(((name, _rt_ty), field), out_const)| {
            quote! {
                let #name: [::tml::Float; #out_const] = self
                    .#field
                    .forward(prefix_out.as_slice())
                    .try_into()
                    .expect("head runtime output length must match the declared head shape");
            }
        });

    let output_const_params_a = head_output_idents.clone();
    let output_const_params_b = head_output_idents.clone();
    let output_const_params_c = head_output_idents.clone();
    let output_const_params_d = head_output_idents.clone();
    let output_const_params_e = head_output_idents.clone();
    let output_const_params_f = head_output_idents.clone();
    let output_field_defs = head_names
        .iter()
        .zip(head_output_idents.iter())
        .map(|(name, out_const)| quote! { pub #name: [::tml::Float; #out_const], });
    let output_init_fields = head_names.iter().map(|name| quote! { #name });

    let head_bounds = head_type_idents
        .iter()
        .zip(head_output_idents.iter())
        .map(|(ty, _out_const)| {
            quote! {
                #ty: ::tml::TransformSpec<PrefixSpec::OutputShape>,
                #ty::Runtime: ::tml::GraphRuntime + 'static,
                [(); #ty::OUTPUT_SIZE]:,
            }
        })
        .collect::<Vec<_>>();
    let runtime_bounds = head_runtime_idents
        .iter()
        .map(|rt_ty| {
            quote! {
                #rt_ty: ::tml::GraphRuntime,
            }
        })
        .collect::<Vec<_>>();

    Ok(quote! {{
        #[derive(Clone, Debug)]
        struct #spec_ident<PrefixSpec, #(#head_type_idents),*> {
            prefix: ::tml::Blueprint<PrefixSpec>,
            #(#head_field_idents: ::tml::Blueprint<#head_type_idents>,)*
        }

        #[derive(Debug)]
        struct #output_ident<#(const #output_const_params_a: usize),*> {
            #(#output_field_defs)*
        }

        #[derive(Debug)]
        struct #runtime_ident<PrefixRt, #(#head_runtime_idents),*, #(const #output_const_params_b: usize),*> {
            prefix: PrefixRt,
            #(#head_field_idents: #head_runtime_idents,)*
        }

        impl<PrefixRt, #(#head_runtime_idents),*, #(const #output_const_params_c: usize),*>
            ::tml::PredictRuntime<{ #input_size }, #output_ident<#(#output_const_params_d),*>>
            for #runtime_ident<PrefixRt, #(#head_runtime_idents),*, #(#output_const_params_e),*>
        where
            PrefixRt: ::tml::GraphRuntime,
            #(#runtime_bounds)*
        {
            fn predict(
                &self,
                input: &[::tml::Float; { #input_size }],
            ) -> #output_ident<#(#output_const_params_f),*> {
                let prefix_out = self.prefix.forward(input);

                #(#head_workspace_inits)*

                #output_ident { #(#output_init_fields),* }
            }
        }

        impl<PrefixSpec, #(#head_type_idents),*> ::tml::BlueprintSpec<#input_shape>
            for #spec_ident<PrefixSpec, #(#head_type_idents),*>
        where
            PrefixSpec: ::tml::TransformSpec<#input_shape>,
            PrefixSpec::Runtime: ::tml::GraphRuntime + 'static,
            [(); PrefixSpec::OUTPUT_SIZE]:,
            #(#head_bounds)*
        {
            fn push_summary(&self, lines: &mut Vec<String>) {
                <PrefixSpec as ::tml::TransformSpec<#input_shape>>::push_summary(
                    self.prefix.as_inner(),
                    lines,
                );
                lines.push("heads".to_string());
                #(#head_summary_lines)*
            }

            fn push_shape_trace(&self, input_axes: &[::tml::Axis], lines: &mut Vec<String>) {
                let mid_axes = <PrefixSpec as ::tml::TransformSpec<#input_shape>>::push_shape_trace(
                    self.prefix.as_inner(),
                    input_axes,
                    lines,
                );
                lines.push("heads".to_string());
                #(#head_trace_lines)*
            }

            fn parameter_count(
                &self,
                seen_shared: &mut ::std::collections::HashSet<usize>,
            ) -> usize {
                <PrefixSpec as ::tml::TransformSpec<#input_shape>>::parameter_count(
                    self.prefix.as_inner(),
                    seen_shared,
                )
                #(#head_param_counts)*
            }
        }

        impl<PrefixSpec, #(#head_type_idents),*> ::tml::HeadsSpec<#input_shape>
            for #spec_ident<PrefixSpec, #(#head_type_idents),*>
        where
            PrefixSpec: ::tml::TransformSpec<#input_shape>,
            PrefixSpec::Runtime: ::tml::GraphRuntime + 'static,
            [(); PrefixSpec::OUTPUT_SIZE]:,
            #(#head_bounds)*
        {
            type Output = #output_ident<#({ #head_type_idents::OUTPUT_SIZE }),*>;
            type Runtime = #runtime_ident<
                PrefixSpec::Runtime,
                #(#head_type_idents::Runtime),*,
                #({ #head_type_idents::OUTPUT_SIZE }),*
            >;

            fn materialize_heads(
                &self,
                ctx: &mut ::tml::MaterializeContext,
            ) -> Self::Runtime {
                #runtime_ident {
                    prefix: self.prefix.as_inner().materialize(ctx),
                    #(#head_field_idents: self.#head_field_idents.as_inner().materialize(ctx),)*
                }
            }
        }

        ::tml::validate_headed_blueprint(#rooted_heads)
    }})
}

fn wrap_root(input: &InputSpec, body: TokenStream2) -> TokenStream2 {
    match input {
        InputSpec::Features { features } => {
            quote! { ::tml::validate_blueprint(::tml::features_input::<{ #features }, _>(#body)) }
        }
        InputSpec::Image {
            channels,
            height,
            width,
        } => {
            quote! {
                ::tml::validate_blueprint(
                    ::tml::image_input::<{ #channels }, { #height }, { #width }, _>(#body)
                )
            }
        }
        InputSpec::Volume {
            channels,
            depth,
            height,
            width,
        } => {
            quote! {
                ::tml::validate_blueprint(
                    ::tml::volume_input::<{ #channels }, { #depth }, { #height }, { #width }, _>(#body)
                )
            }
        }
    }
}

fn input_shape_ty(input: &InputSpec) -> TokenStream2 {
    match input {
        InputSpec::Features { features } => quote! { ::tml::shape!(features: #features) },
        InputSpec::Image {
            channels,
            height,
            width,
        } => quote! { ::tml::shape!(channels: #channels, height: #height, width: #width) },
        InputSpec::Volume {
            channels,
            depth,
            height,
            width,
        } => {
            quote! { ::tml::shape!(channels: #channels, depth: #depth, height: #height, width: #width) }
        }
    }
}

fn input_size_expr(input: &InputSpec) -> TokenStream2 {
    match input {
        InputSpec::Features { features } => quote! { #features },
        InputSpec::Image {
            channels,
            height,
            width,
        } => quote! { #channels * #height * #width },
        InputSpec::Volume {
            channels,
            depth,
            height,
            width,
        } => quote! { #channels * #depth * #height * #width },
    }
}

fn axis_tokens(axis: &syn::Ident) -> Result<TokenStream2> {
    Ok(match axis.to_string().as_str() {
        "features" => quote! { ::tml::Axis::Features },
        "channels" => quote! { ::tml::Axis::Channels },
        "length" => quote! { ::tml::Axis::Length },
        "depth" => quote! { ::tml::Axis::Depth },
        "height" => quote! { ::tml::Axis::Height },
        "width" => quote! { ::tml::Axis::Width },
        _ => {
            return Err(Error::new(
                axis.span(),
                "unknown axis; expected one of features, channels, length, depth, height, width",
            ));
        }
    })
}

fn path_name(expr: &Expr) -> Option<String> {
    let Expr::Path(path) = expr else {
        return None;
    };
    if path.path.segments.len() == 1 {
        Some(path.path.segments[0].ident.to_string())
    } else {
        None
    }
}

fn call_name(expr: &Expr) -> Option<String> {
    let Expr::Call(call) = expr else {
        return None;
    };
    path_name(call.func.as_ref())
}

fn call_arg(expr: &Expr) -> Result<Expr> {
    let Expr::Call(call) = expr else {
        return Err(Error::new(expr.span(), "expected a call expression"));
    };
    if call.args.len() != 1 {
        return Err(Error::new(expr.span(), "expected exactly one argument"));
    }
    Ok(call.args[0].clone())
}
