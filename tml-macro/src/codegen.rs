use crate::ir::NetworkIr;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;

pub fn generate_network(ir: &NetworkIr) -> TokenStream2 {
    let conv_checks = &ir.conv_checks;
    let builder = ir
        .layers
        .iter()
        .fold(ir.input_builder.clone(), |builder, layer| {
            let step = layer.builder_step.clone();
            quote! { #builder #step }
        });

    quote! {{
        #(#conv_checks)*
        #builder.build()
    }}
}
