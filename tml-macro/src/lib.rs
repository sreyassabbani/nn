use proc_macro::TokenStream;
use syn::parse_macro_input;

mod codegen;
mod dsl;
mod parsing;

#[proc_macro]
pub fn network(input: TokenStream) -> TokenStream {
    let network_def = parse_macro_input!(input as parsing::NetworkDef);
    codegen::generate_network(network_def).into()
}
