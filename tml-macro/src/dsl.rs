use proc_macro2::TokenStream as TokenStream2;
use quote::quote;

#[derive(Clone, Debug)]
pub enum ShapeSpec {
    Vec {
        n: TokenStream2,
    },
    Image {
        c: TokenStream2,
        h: TokenStream2,
        w: TokenStream2,
    },
}

impl ShapeSpec {
    pub fn size_expr(&self) -> TokenStream2 {
        match self {
            ShapeSpec::Vec { n } => quote! { #n },
            ShapeSpec::Image { c, h, w } => quote! { #c * #h * #w },
        }
    }
}

pub fn max_expr(exprs: Vec<TokenStream2>) -> TokenStream2 {
    let mut iter = exprs.into_iter();
    let first = iter.next().unwrap_or_else(|| quote! { 0 });
    iter.fold(first, |acc, expr| quote! { __nn_max(#acc, #expr) })
}
