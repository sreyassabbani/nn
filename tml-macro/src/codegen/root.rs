//! Rooted-network lowering helpers.

use crate::ast::{HeadAst, InputSpec, PipelineAst, StepAst};
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use syn::{Error, Result};

use super::heads::generate_headed_root;
use super::pipeline::lower_pipeline_expr;

pub(super) fn generate_rooted_network(
    input: &InputSpec,
    pipeline: &PipelineAst,
) -> Result<TokenStream2> {
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

pub(super) fn wrap_root(input: &InputSpec, body: TokenStream2) -> TokenStream2 {
    let shape = input_shape_ty(input);
    let axes = input_axis_vec(input);
    quote! {
        ::tml::validate_blueprint(::tml::root::<#shape, _>(#body, #axes))
    }
}

pub(super) fn input_shape_ty(input: &InputSpec) -> TokenStream2 {
    let names = input.fields.iter().map(|field| &field.name);
    let extents = input.fields.iter().map(|field| &field.extent);
    quote! { ::tml::shape!(#(#names: #extents),*) }
}

pub(super) fn input_size_expr(input: &InputSpec) -> TokenStream2 {
    let mut extents = input.fields.iter().map(|field| &field.extent);
    let first = extents
        .next()
        .map(|expr| quote! { #expr })
        .unwrap_or_else(|| quote! { 1 });
    extents.fold(first, |acc, expr| quote! { (#acc) * (#expr) })
}

pub(super) fn input_axis_vec(input: &InputSpec) -> TokenStream2 {
    let names = input.fields.iter().map(|field| &field.name);
    quote! { vec![#(::tml::Axis::new(stringify!(#names))),*] }
}
