use crate::ir::NetworkIr;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;

pub fn generate_network(ir: &NetworkIr) -> TokenStream2 {
    let conv_checks = &ir.conv_checks;
    let layers = ir
        .layers
        .iter()
        .map(|layer| {
            let layer_type = layer.layer_type.clone();
            let out_size = layer.out_size.clone();
            (quote! { <#layer_type>::init() }, out_size)
        })
        .collect::<Vec<_>>();

    let chain = layers.into_iter().rev().fold(
        quote! { ::tml::network::End },
        |tail, (layer, out_size)| {
            quote! { ::tml::network::Chain::<_, _, { #out_size }>::new(#layer, #tail) }
        },
    );

    quote! {{
        #(#conv_checks)*
        ::tml::network::Sequential::new(#chain)
    }}
}
