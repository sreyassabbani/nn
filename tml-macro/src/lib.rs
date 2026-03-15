use proc_macro::TokenStream;
use syn::parse_macro_input;

mod codegen;
mod dsl;
mod ir;
mod parsing;

#[proc_macro]
pub fn network(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as parsing::NetworkAst);
    match ir::NetworkIr::lower(ast) {
        Ok(ir) => codegen::generate_network(&ir).into(),
        Err(err) => err.to_compile_error().into(),
    }
}
