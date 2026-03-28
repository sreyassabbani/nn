//! Lowering from parsed [`crate::ast`] nodes into Rust tokens.

mod heads;
mod pipeline;
mod reusable;
mod root;
mod stages;

use crate::ast::NetworkAst;
use proc_macro2::TokenStream as TokenStream2;
use syn::Result;

pub fn generate_network(ast: &NetworkAst) -> Result<TokenStream2> {
    match ast.input.as_ref() {
        Some(input) => root::generate_rooted_network(input, &ast.pipeline),
        None => pipeline::lower_pipeline_expr(&ast.pipeline),
    }
}
