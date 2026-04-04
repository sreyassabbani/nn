//! Lowering helpers for individual transform stages.

use crate::ast::{ConvSpec, DenseSpec, KernelSpec, LinearSpec, PipelineAst};
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Error, Expr, ExprLit, ExprPath, ExprTuple, Lit, Result, spanned::Spanned};

use super::pipeline::fold_branches;
use super::reusable::lower_reusable_expr;

pub(super) fn lower_dense(spec: &DenseSpec) -> Result<TokenStream2> {
    let output = &spec.output;
    Ok(if spec.bias {
        quote! { ::tml::dense::<{ #output }>() }
    } else {
        quote! { ::tml::dense_no_bias::<{ #output }>() }
    })
}

pub(super) fn lower_linear(spec: &LinearSpec) -> Result<TokenStream2> {
    let output = &spec.output;
    Ok(if spec.bias {
        quote! { ::tml::linear::<{ #output }>() }
    } else {
        quote! { ::tml::linear_no_bias::<{ #output }>() }
    })
}

pub(super) fn lower_conv(spec: &ConvSpec) -> Result<TokenStream2> {
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

pub(super) fn lower_share(expr: &Expr) -> Result<TokenStream2> {
    lower_share_expr(expr)
}

pub(super) fn lower_share_expr(expr: &Expr) -> Result<TokenStream2> {
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
        ::tml::__private::shared_name_id(concat!(module_path!(), "::", stringify!(#expr)))
    };

    Ok(quote! { ::tml::share_fragment_with_id(&#expr, #share_id) })
}

pub(super) fn lower_repeat(times: &syn::LitInt, body: &Expr) -> Result<TokenStream2> {
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

pub(super) fn lower_concat(axis: &syn::Ident, branches: &[PipelineAst]) -> Result<TokenStream2> {
    let axis = axis_tokens(axis)?;
    fold_branches(branches, |left, right| {
        quote! { ::tml::concat(#axis, #left, #right) }
    })
}

pub(super) fn lower_sum(branches: &[PipelineAst]) -> Result<TokenStream2> {
    fold_branches(branches, |left, right| quote! { ::tml::sum(#left, #right) })
}

pub(super) fn axis_tokens(axis: &syn::Ident) -> Result<TokenStream2> {
    let axis = axis.to_string();
    Ok(quote! { ::tml::Axis::new(#axis) })
}
