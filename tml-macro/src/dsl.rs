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
