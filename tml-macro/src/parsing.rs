use proc_macro2::Span;
use syn::parse::{Parse, ParseStream};
use syn::{Expr, Ident, Token};

#[derive(Clone)]
pub enum InputShape {
    Vec { n: Expr },
    Image(Box<ImageShape>),
}

#[derive(Clone)]
pub struct ImageShape {
    pub c: Expr,
    pub h: Expr,
    pub w: Expr,
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

#[derive(Clone)]
pub enum LayerSpecKind {
    Dense { output: Expr },
    ReLU,
    Sigmoid,
    Flatten,
    Conv(Box<ConvSpec>),
}

#[derive(Clone)]
pub struct ConvSpec {
    pub out_channels: Expr,
    pub kernel_h: Expr,
    pub kernel_w: Expr,
    pub stride: Expr,
    pub padding: Expr,
}

#[derive(Clone)]
pub struct LayerSpec {
    pub span: Span,
    pub kind: LayerSpecKind,
}

#[derive(Clone)]
pub struct NetworkAst {
    pub input: InputShape,
    pub layers: Vec<LayerSpec>,
}

impl Parse for NetworkAst {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let _name: Ident = input.parse()?;

        let content;
        ::syn::parenthesized!(content in input);
        let input_dims = parse_expr_list(&content)?;
        let input_shape = match input_dims.as_slice() {
            [n] => InputShape::Vec { n: n.clone() },
            [c, h, w] => InputShape::Image(Box::new(ImageShape {
                c: c.clone(),
                h: h.clone(),
                w: w.clone(),
            })),
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
            let layer_span = layer_name.span();

            match layer_name.to_string().as_str() {
                "dense" => {
                    let content;
                    ::syn::parenthesized!(content in input);
                    let next_size: Expr = content.parse()?;
                    layers.push(LayerSpec {
                        span: layer_span,
                        kind: LayerSpecKind::Dense { output: next_size },
                    });
                }
                "relu" | "ReLU" => {
                    layers.push(LayerSpec {
                        span: layer_span,
                        kind: LayerSpecKind::ReLU,
                    });
                }
                "sigmoid" | "Sigmoid" => {
                    layers.push(LayerSpec {
                        span: layer_span,
                        kind: LayerSpecKind::Sigmoid,
                    });
                }
                "flatten" | "Flatten" => {
                    layers.push(LayerSpec {
                        span: layer_span,
                        kind: LayerSpecKind::Flatten,
                    });
                }
                "conv" | "Conv" => {
                    let content;
                    ::syn::parenthesized!(content in input);
                    let args = parse_expr_list(&content)?;

                    let kind = match args.as_slice() {
                        [out_channels, kernel] => LayerSpecKind::Conv(Box::new(ConvSpec {
                            out_channels: out_channels.clone(),
                            kernel_h: kernel.clone(),
                            kernel_w: kernel.clone(),
                            stride: syn::parse_quote!(1),
                            padding: syn::parse_quote!(0),
                        })),
                        [out_channels, kernel, stride] => LayerSpecKind::Conv(Box::new(ConvSpec {
                            out_channels: out_channels.clone(),
                            kernel_h: kernel.clone(),
                            kernel_w: kernel.clone(),
                            stride: stride.clone(),
                            padding: syn::parse_quote!(0),
                        })),
                        [out_channels, kernel, stride, padding] => {
                            LayerSpecKind::Conv(Box::new(ConvSpec {
                                out_channels: out_channels.clone(),
                                kernel_h: kernel.clone(),
                                kernel_w: kernel.clone(),
                                stride: stride.clone(),
                                padding: padding.clone(),
                            }))
                        }
                        [out_channels, kernel_h, kernel_w, stride, padding] => {
                            LayerSpecKind::Conv(Box::new(ConvSpec {
                                out_channels: out_channels.clone(),
                                kernel_h: kernel_h.clone(),
                                kernel_w: kernel_w.clone(),
                                stride: stride.clone(),
                                padding: padding.clone(),
                            }))
                        }
                        _ => {
                            return Err(::syn::Error::new(
                                content.span(),
                                "conv expects (out, k) | (out, k, stride) | (out, k, stride, pad) | (out, k_h, k_w, stride, pad)",
                            ));
                        }
                    };

                    layers.push(LayerSpec {
                        span: layer_span,
                        kind,
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

        Ok(NetworkAst {
            input: input_shape,
            layers,
        })
    }
}
