use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{Expr, Ident, Token};

#[derive(Debug, Clone)]
pub enum InputShape {
    Vec {
        n: TokenStream2,
    },
    Image {
        c: TokenStream2,
        h: TokenStream2,
        w: TokenStream2,
    },
}

fn parse_expr_list(content: &syn::parse::ParseBuffer<'_>) -> syn::Result<Vec<Expr>> {
    let mut values = Vec::new();
    if content.is_empty() {
        return Ok(values);
    }

    values.push(content.parse::<Expr>()?);
    while content.peek(Token![,]) {
        content.parse::<Token![,]>()?;
        if content.is_empty() {
            break;
        }
        values.push(content.parse::<Expr>()?);
    }

    Ok(values)
}

#[derive(Debug, Clone)]
pub enum LayerSpecKind {
    Dense {
        output: TokenStream2,
    },
    ReLU,
    Sigmoid,
    Flatten,
    Conv {
        out_channels: TokenStream2,
        kernel_h: TokenStream2,
        kernel_w: TokenStream2,
        stride: TokenStream2,
        padding: TokenStream2,
    },
}

#[derive(Debug, Clone)]
pub struct LayerSpec {
    pub kind: LayerSpecKind,
}

pub struct NetworkDef {
    pub input: InputShape,
    pub layers: Vec<LayerSpec>,
}

impl Parse for NetworkDef {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let _name: Ident = input.parse()?;

        let content;
        ::syn::parenthesized!(content in input);
        let input_dims = parse_expr_list(&content)?;
        let input_shape = match input_dims.as_slice() {
            [n] => InputShape::Vec { n: quote! { #n } },
            [c, h, w] => InputShape::Image {
                c: quote! { #c },
                h: quote! { #h },
                w: quote! { #w },
            },
            _ => {
                return Err(::syn::Error::new(
                    content.span(),
                    "input must be (N) or (C, H, W)",
                ));
            }
        };

        input.parse::<Token![->]>()?;

        let mut layers = Vec::new();

        while !input.is_empty() {
            let layer_name: Ident = input.parse()?;

            match layer_name.to_string().as_str() {
                "dense" => {
                    let content;
                    ::syn::parenthesized!(content in input);
                    let next_size: Expr = content.parse()?;
                    layers.push(LayerSpec {
                        kind: LayerSpecKind::Dense {
                            output: quote! { #next_size },
                        },
                    });
                }
                "relu" | "ReLU" => {
                    layers.push(LayerSpec {
                        kind: LayerSpecKind::ReLU,
                    });
                }
                "sigmoid" | "Sigmoid" => {
                    layers.push(LayerSpec {
                        kind: LayerSpecKind::Sigmoid,
                    });
                }
                "flatten" | "Flatten" => {
                    layers.push(LayerSpec {
                        kind: LayerSpecKind::Flatten,
                    });
                }
                "conv" | "Conv" => {
                    let content;
                    ::syn::parenthesized!(content in input);
                    let args = parse_expr_list(&content)?;

                    let kind = match args.as_slice() {
                        [out_channels, kernel] => LayerSpecKind::Conv {
                            out_channels: quote! { #out_channels },
                            kernel_h: quote! { #kernel },
                            kernel_w: quote! { #kernel },
                            stride: quote! { 1 },
                            padding: quote! { 0 },
                        },
                        [out_channels, kernel, stride] => LayerSpecKind::Conv {
                            out_channels: quote! { #out_channels },
                            kernel_h: quote! { #kernel },
                            kernel_w: quote! { #kernel },
                            stride: quote! { #stride },
                            padding: quote! { 0 },
                        },
                        [out_channels, kernel, stride, padding] => LayerSpecKind::Conv {
                            out_channels: quote! { #out_channels },
                            kernel_h: quote! { #kernel },
                            kernel_w: quote! { #kernel },
                            stride: quote! { #stride },
                            padding: quote! { #padding },
                        },
                        [out_channels, kernel_h, kernel_w, stride, padding] => {
                            LayerSpecKind::Conv {
                                out_channels: quote! { #out_channels },
                                kernel_h: quote! { #kernel_h },
                                kernel_w: quote! { #kernel_w },
                                stride: quote! { #stride },
                                padding: quote! { #padding },
                            }
                        }
                        _ => {
                            return Err(::syn::Error::new(
                                content.span(),
                                "conv expects (out, k) | (out, k, stride) | (out, k, stride, pad) | (out, k_h, k_w, stride, pad)",
                            ));
                        }
                    };

                    layers.push(LayerSpec { kind });
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
            input: input_shape,
            layers,
        })
    }
}
