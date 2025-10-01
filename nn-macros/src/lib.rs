use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
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
        Conv {
            /// Number of output channels/features in the output. Alternatively, this may be interpreted as the number of filters in the convolutional layer.
            out_channels: usize,
            kernel: usize,
            stride: usize,
            padding: usize,
        },
        Dense(usize),
        ReLU,
        Sigmoid,
    }

    impl Parse for NetworkDef {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            input.parse::<Ident>()?;

            let content;
            ::syn::parenthesized!(content in input);
            let input_size: LitInt = content.parse()?;

            input.parse::<Token![->]>()?;

            let mut layers = Vec::new();

            while !input.is_empty() {
                let layer_name: Ident = input.parse()?;

                match layer_name.to_string().as_str() {
                    "dense" => {
                        let content;
                        ::syn::parenthesized!(content in input);
                        let size: LitInt = content.parse()?;
                        layers.push(Layer::Dense(size.base10_parse()?));
                    }
                    "relu" | "ReLU" => {
                        layers.push(Layer::ReLU);
                    }
                    "sigmoid" | "Sigmoid" => {
                        layers.push(Layer::Sigmoid);
                    }
                    "conv" | "Conv" => {
                        // parse parens with comma-separated ints; allow optional named args later
                        let content;
                        ::syn::parenthesized!(content in input);

                        // minimal syntax: conv(out, kernel) or conv(out, kernel, stride, pad)
                        let out_c: LitInt = content.parse()?;
                        let _comma = content.parse::<Token![,]>()?;
                        let k: LitInt = content.parse()?;

                        // defaults:
                        let mut stride = 1usize;
                        let mut pad = 0usize;

                        // optional additional positional args
                        if content.peek(Token![,]) {
                            content.parse::<Token![,]>()?;
                            let s: LitInt = content.parse()?;
                            stride = s.base10_parse()?;
                            if content.peek(Token![,]) {
                                content.parse::<Token![,]>()?;
                                let p: LitInt = content.parse()?;
                                pad = p.base10_parse()?;
                            }
                        }

                        layers.push(Layer::Conv {
                            out_channels: out_c.base10_parse()?,
                            kernel: k.base10_parse()?,
                            stride,
                            padding: pad,
                        });
                    }
                    "output" => break,
                    _ => return Err(::syn::Error::new(layer_name.span(), "Unknown layer type")),
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

#[proc_macro]
pub fn network(input: TokenStream) -> TokenStream {
    let network_def = parse_macro_input!(input as parsing::NetworkDef);

    // Generate the network code
    let generated = generate_network(network_def);

    generated.into()
}

fn generate_network(def: parsing::NetworkDef) -> TokenStream2 {
    let input_size = def.input_size;
    let layer_count = def.layers.len();

    // Calculate maximum buffer size needed
    let mut current_size = input_size;
    let mut max_size = input_size;
    let layer_types: Vec<_> = def
        .layers
        .into_iter()
        .map(|layer| {
            use parsing::Layer;

            return match layer {
                Layer::Dense(out_size) => {
                    let l = quote! { ::nn::network::DenseLayer<#current_size, #out_size> };
                    current_size = out_size;
                    max_size = max_size.max(out_size);
                    return l;
                }
                // Working data buffer's size stays the same for activation functions
                Layer::ReLU => quote! { ::nn::network::ReLU<#current_size> },
                Layer::Sigmoid => quote! { ::nn::network::Sigmoid<#current_size> },
                Layer::Conv {
                    out_channels,
                    kernel,
                    stride,
                    padding,
                } => quote! { ::nn::network::Conv<#current_size> },
            };
        })
        .collect();

    let output_size = current_size;

    // Generate forward pass with buffer reuse
    let mut forward_calls = Vec::new();
    let mut use_buf_a = true;

    for i in 0..layer_count {
        let layer_idx = ::syn::Index::from(i);
        let (input_buf, output_buf) = if use_buf_a {
            (quote! { &self._buf_a }, quote! { &mut self._buf_b })
        } else {
            (quote! { &self._buf_b }, quote! { &mut self._buf_a })
        };

        // forward_calls.push(quote! {
        //     self.layers.#layer_idx.forward(
        //         <&[f32; #current_size]>::try_into(#input_buf[..#current_size]).unwrap(),
        //         <&mut [f32; #current_size]>::try_into(&mut #output_buf[..#current_size]).unwrap(),
        //     );
        // });

        forward_calls.push(quote! {
            self.layers.#layer_idx.forward(
                #input_buf[..#current_size],
                #output_buf[..#current_size],
            );
        });

        use_buf_a = !use_buf_a;
    }

    // Generate layer initializations
    let layer_inits = layer_types.iter().map(|layer_type| {
        quote! { <#layer_type>::init() }
    });

    let final_buffer = if (layer_count % 2) == 1 {
        quote! { self._buf_b }
    } else {
        quote! { self._buf_a }
    };

    quote! {
        {
            #[derive(Debug)]
            struct Network<Layers> {
                layers: Layers,
                // Double buffering approach with fixed-size boxes
                _buf_a: Box<[f32; #max_size]>,
                _buf_b: Box<[f32; #max_size]>,
            }

            struct NetworkWorkspace {

            }

            impl Network<(#(#layer_types,)*)> {
                pub fn new() -> Self {
                    Network {
                        layers: (#(#layer_inits,)*),
                        _buf_a: Box::new([Default::default(); #max_size]),
                        _buf_b: Box::new([Default::default(); #max_size]),
                    }
                }

                pub fn forward_with_workspace(&self, input: &[f32; #input_size], workspace: &mut NetworkWorkspace) -> [f32; #output_size] {
                    // used to be forward<I: AsRef<[f32; #input_size]>>(... input: I)

                    // Copy input to first buffer
                    // self._buf_a[..#input_size].copy_from_slice(input);

                    // Run forward pass with ping-pong buffers
                    // #(#forward_calls)*;

                    // Extract result from final buffer
                    let mut result = [0.0; #output_size];
                    result.copy_from_slice(&(#final_buffer)[..#output_size]);
                    result
                }

                pub fn forward(&self, input: &[f32; #input_size]) -> [f32; #output_size] {
                    // Copy input to first buffer
                    // self.buffers.0 = *input;

                    // Run forward pass
                    // #(#forward_calls)*

                    // Return final buffer
                    // #final_buffer
                    [0.0; #output_size]
                }

                pub fn train<D: AsRef<[[f32; #input_size]]>, T: AsRef<[[f32; #output_size]]>>(&mut self, data: D, targets: T) {
                    // Loop over each case
                    let targets = targets.as_ref().iter();
                    let data = data.as_ref().iter();

                    for (input, target) in data.zip(targets) {
                        let out = self.forward(input);
                        let loss: f32 = out.iter().zip(target.iter()).map(|(o, t)| (o - t).powi(2)).sum();
                        // sum (y hat - y)^2

                        // for layer in self.layers.iter() {

                        // }
                    }

                    // Training implementation
                }
            }

            Network::<(#(#layer_types,)*)>::new()
        }
    }
}
