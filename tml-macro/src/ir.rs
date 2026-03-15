use crate::dsl::{ShapeSpec, max_expr};
use crate::parsing::{InputShape, LayerSpecKind, NetworkAst};
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;

#[derive(Debug, Clone)]
pub struct LayerIr {
    pub layer_type: TokenStream2,
    pub in_size: TokenStream2,
    pub out_size: TokenStream2,
}

#[derive(Debug, Clone)]
pub struct NetworkIr {
    pub input_size: TokenStream2,
    pub output_size: TokenStream2,
    pub max_buf_size: TokenStream2,
    pub layers: Vec<LayerIr>,
    pub conv_checks: Vec<TokenStream2>,
}

impl NetworkIr {
    pub fn lower(ast: NetworkAst) -> syn::Result<Self> {
        let input_shape = match ast.input {
            InputShape::Vec { n } => ShapeSpec::Vec { n: quote! { #n } },
            InputShape::Image(image) => ShapeSpec::Image {
                c: {
                    let c = &image.c;
                    quote! { #c }
                },
                h: {
                    let h = &image.h;
                    quote! { #h }
                },
                w: {
                    let w = &image.w;
                    quote! { #w }
                },
            },
        };

        let input_size = input_shape.size_expr();
        let mut current_shape = input_shape;
        let mut layers = Vec::with_capacity(ast.layers.len());
        let mut layer_out_sizes = Vec::with_capacity(ast.layers.len());
        let mut conv_checks = Vec::with_capacity(ast.layers.len());

        for layer in &ast.layers {
            let in_size = current_shape.size_expr();
            let (next_shape, layer_type) = match &layer.kind {
                LayerSpecKind::Dense { output } => match current_shape {
                    ShapeSpec::Vec { n } => {
                        let output = quote! { #output };
                        (
                            ShapeSpec::Vec { n: output.clone() },
                            quote! { ::tml::network::DenseLayer<{ #n }, { #output }> },
                        )
                    }
                    ShapeSpec::Image { .. } => {
                        return Err(syn::Error::new(
                            layer.span,
                            "dense expects a vector input; add flatten before dense",
                        ));
                    }
                },
                LayerSpecKind::ReLU => {
                    let size = current_shape.size_expr();
                    (current_shape, quote! { ::tml::network::ReLU<{ #size }> })
                }
                LayerSpecKind::Sigmoid => {
                    let size = current_shape.size_expr();
                    (current_shape, quote! { ::tml::network::Sigmoid<{ #size }> })
                }
                LayerSpecKind::Flatten => {
                    let size = current_shape.size_expr();
                    (
                        ShapeSpec::Vec { n: size.clone() },
                        quote! { ::tml::network::Flatten<{ #size }> },
                    )
                }
                LayerSpecKind::Conv(conv) => {
                    match &current_shape {
                        ShapeSpec::Image { c, h, w } => {
                            let out_channels = {
                                let out_channels = &conv.out_channels;
                                quote! { #out_channels }
                            };
                            let kernel_h = {
                                let kernel_h = &conv.kernel_h;
                                quote! { #kernel_h }
                            };
                            let kernel_w = {
                                let kernel_w = &conv.kernel_w;
                                quote! { #kernel_w }
                            };
                            let stride = {
                                let stride = &conv.stride;
                                quote! { #stride }
                            };
                            let padding = {
                                let padding = &conv.padding;
                                quote! { #padding }
                            };

                            conv_checks.push(quote! {
                            const _: () = {
                                if !(::tml::conv::conv_out_dim(#h, #padding, #kernel_h, #stride) > 0) {
                                    panic!("conv: invalid height (check input H, kernel, stride, padding)");
                                }
                                if !(::tml::conv::conv_out_dim(#w, #padding, #kernel_w, #stride) > 0) {
                                    panic!("conv: invalid width (check input W, kernel, stride, padding)");
                                }
                            };
                        });

                            let out_h = quote! { ::tml::conv::conv_out_dim(#h, #padding, #kernel_h, #stride) };
                            let out_w = quote! { ::tml::conv::conv_out_dim(#w, #padding, #kernel_w, #stride) };

                            (
                                ShapeSpec::Image {
                                    c: out_channels.clone(),
                                    h: out_h,
                                    w: out_w,
                                },
                                quote! {
                                    ::tml::conv::Conv<{ #w }, { #h }, { #c }, { #kernel_h }, { #kernel_w }, { #out_channels }, { #stride }, { #padding }>
                                },
                            )
                        }
                        ShapeSpec::Vec { .. } => {
                            return Err(syn::Error::new(
                                layer.span,
                                "conv expects a (C, H, W) input shape",
                            ));
                        }
                    }
                }
            };

            let out_size = next_shape.size_expr();
            layers.push(LayerIr {
                layer_type,
                in_size,
                out_size: out_size.clone(),
            });
            layer_out_sizes.push(out_size);
            current_shape = next_shape;
        }

        let output_size = current_shape.size_expr();
        let max_buf_size = max_expr(
            std::iter::once(input_size.clone())
                .chain(layer_out_sizes)
                .collect(),
        );

        Ok(Self {
            input_size,
            output_size,
            max_buf_size,
            layers,
            conv_checks,
        })
    }
}
