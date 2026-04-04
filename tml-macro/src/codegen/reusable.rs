//! Lowering of reusable Rust-defined fragments referenced from `network!`.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Error, Expr, ExprCall, Result, spanned::Spanned};

use super::stages::lower_share_expr;

pub(super) fn lower_reusable_expr(expr: &Expr) -> Result<TokenStream2> {
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
    let Expr::Call(ExprCall { args, .. }) = expr else {
        return Err(Error::new(expr.span(), "expected a call expression"));
    };
    if args.len() != 1 {
        return Err(Error::new(expr.span(), "expected exactly one argument"));
    }
    Ok(args[0].clone())
}
