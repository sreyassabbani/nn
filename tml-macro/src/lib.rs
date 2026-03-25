use proc_macro::TokenStream;
use syn::parse_macro_input;

mod codegen;
mod parsing;

#[proc_macro]
pub fn network(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as parsing::NetworkAst);
    match codegen::generate_network(&ast) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}
