//! Procedural macros for the `tml` facade crate.
//!
//! This crate currently exposes [`network!`](macro@network), the typed
//! architecture DSL used by the public `tml` API.

use proc_macro::TokenStream;
use syn::parse_macro_input;

mod ast;
mod codegen;
mod parsing;

#[proc_macro]
pub fn network(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as ast::NetworkAst);
    match codegen::generate_network(&ast) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}
