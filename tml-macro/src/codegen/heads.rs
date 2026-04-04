//! Lowering for terminal `heads { ... }` networks.

use crate::ast::{HeadAst, InputSpec, PipelineAst, StepAst};
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::Result;

use super::pipeline::lower_pipeline_expr;
use super::root::{input_axis_vec, input_shape_ty, input_size_expr};

pub(super) fn generate_headed_root(
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
        let axes = input_axis_vec(input);
        quote! { ::tml::root::<#input_shape, _>(#spec, #axes) }
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
