//! Pipeline-stage lowering.

use crate::ast::{PipelineAst, StepAst};
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use std::collections::BTreeMap;
use syn::{Error, Result};

use super::reusable::lower_reusable_expr;
use super::stages::{
    axis_tokens, lower_concat, lower_conv, lower_dense, lower_linear, lower_repeat, lower_share,
    lower_sum,
};

pub(super) fn lower_pipeline_expr(pipeline: &PipelineAst) -> Result<TokenStream2> {
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
        StepAst::Linear(spec) => lower_linear(spec),
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

fn load_saved_source(
    saved: &BTreeMap<String, TokenStream2>,
    name: &syn::Ident,
) -> Result<TokenStream2> {
    saved
        .get(&name.to_string())
        .cloned()
        .ok_or_else(|| Error::new(name.span(), format!("unknown saved source `{}`", name)))
}

pub(super) fn fold_branches<F>(branches: &[PipelineAst], combine: F) -> Result<TokenStream2>
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
