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

pub(super) fn input_shape_ty(input: &InputSpec) -> TokenStream2 {
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

pub(super) fn input_size_expr(input: &InputSpec) -> TokenStream2 {
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
