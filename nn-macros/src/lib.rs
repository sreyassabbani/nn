use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Ident, LitInt, Token, parse_macro_input};

// Custom parsing for our network DSL
mod parsing {
    use super::*;
    use syn::parse::{Parse, ParseStream};

    pub struct NetworkDef {
        pub input_size: usize,
        pub layers: Vec<Layer>,
    }

    pub enum Layer {
        Dense(usize),
        ReLU,
        Sigmoid,
    }

    impl Parse for NetworkDef {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            // Parse: input(10) -> dense(20) -> relu -> dense(1) -> sigmoid -> output
            input.parse::<Ident>()?; // "input"

            let content;
            syn::parenthesized!(content in input);
            let input_size: LitInt = content.parse()?;

            input.parse::<Token![->]>()?;

            let mut layers = Vec::new();

            while !input.is_empty() {
                let layer_name: Ident = input.parse()?;

                match layer_name.to_string().as_str() {
                    "dense" => {
                        let content;
                        syn::parenthesized!(content in input);
                        let size: LitInt = content.parse()?;
                        layers.push(Layer::Dense(size.base10_parse()?));
                    }
                    "relu" => {
                        layers.push(Layer::ReLU);
                    }
                    "sigmoid" => {
                        layers.push(Layer::Sigmoid);
                    }
                    "output" => break,
                    _ => return Err(syn::Error::new(layer_name.span(), "Unknown layer type")),
                }

                if !input.is_empty() && !input.peek(Token![->]) {
                    break;
                }

                if input.peek(Token![->]) {
                    input.parse::<Token![->]>()?;
                }
            }

            Ok(NetworkDef {
                input_size: input_size.base10_parse()?,
                layers,
            })
        }
    }
}

use parsing::*;

#[proc_macro]
pub fn network(input: TokenStream) -> TokenStream {
    let network_def = parse_macro_input!(input as NetworkDef);

    // Generate the network code
    let generated = generate_network(network_def);

    generated.into()
}

fn generate_network(def: NetworkDef) -> TokenStream2 {
    let input_size = def.input_size;
    let layer_count = def.layers.len();
    let buffer_count = layer_count + 1; // One more buffer than layers

    // Calculate output size by following the layers
    let mut current_size = input_size;
    let mut layer_types = Vec::new();
    let mut buffer_types = vec![quote! { [f32; #input_size] }];
    let mut buffer_inits = vec![quote! { [0.0; #input_size] }];

    for layer in &def.layers {
        match layer {
            Layer::Dense(out_size) => {
                layer_types.push(quote! { DenseLayer<#current_size, #out_size> });
                buffer_types.push(quote! { [f32; #out_size] });
                buffer_inits.push(quote! { [0.0; #out_size] });
                current_size = *out_size;
            }
            Layer::ReLU => {
                layer_types.push(quote! { ReLU });
                buffer_types.push(quote! { [f32; #current_size] });
                buffer_inits.push(quote! { [0.0; #current_size] });
            }
            Layer::Sigmoid => {
                layer_types.push(quote! { Sigmoid });
                buffer_types.push(quote! { [f32; #current_size] });
                buffer_inits.push(quote! { [0.0; #current_size] });
            }
        }
    }

    let output_size = current_size;

    // Generate forward pass calls
    let mut forward_calls = Vec::new();
    for i in 0..layer_count {
        let layer_idx = syn::Index::from(i);
        let buf_in_idx = syn::Index::from(i);
        let buf_out_idx = syn::Index::from(i + 1);

        forward_calls.push(quote! {
            self.layers.#layer_idx.forward(&self.buffers.#buf_in_idx, &mut self.buffers.#buf_out_idx);
        });
    }

    // Generate layer initializations
    let layer_inits = layer_types.iter().map(|layer_type| {
        quote! { <#layer_type as LayerInit>::init() }
    });

    let final_buffer_idx = syn::Index::from(layer_count);

    quote! {
        {
            #[derive(Debug)]
            struct Network {
                layers: (#(#layer_types,)*),
                buffers: (#(#buffer_types,)*),
            }

            impl Network {
                pub fn new() -> Self {
                    Network {
                        layers: (#(#layer_inits,)*),
                        buffers: (#(#buffer_inits,)*),
                    }
                }
            }

            impl NetworkTrait<#input_size, #output_size> for Network {
                fn forward(&mut self, input: &[f32; #input_size]) -> [f32; #output_size] {
                    // Copy input to first buffer
                    self.buffers.0 = *input;

                    // Run forward pass
                    #(#forward_calls)*

                    // Return final buffer
                    self.buffers.#final_buffer_idx
                }

                fn train(&mut self, _data: &[[f32; #input_size]], _targets: &[[f32; #output_size]]) {
                    // Placeholder for training logic
                }
            }

            Network::new()
        }
    }
}
