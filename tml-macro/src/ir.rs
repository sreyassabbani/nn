use crate::dsl::ShapeSpec;
use crate::parsing::{InputShape, LayerSpecKind, NetworkAst};
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;

#[derive(Debug, Clone)]
pub struct LayerIr {
    pub builder_step: TokenStream2,
}

#[derive(Debug, Clone)]
pub struct NetworkIr {
    pub input_builder: TokenStream2,
    pub layers: Vec<LayerIr>,
    pub conv_checks: Vec<TokenStream2>,
}

impl NetworkIr {
    pub fn lower(ast: NetworkAst) -> syn::Result<Self> {
        let (input_shape, input_builder) = match ast.input {
            InputShape::Vec { n } => (
                ShapeSpec::Vec { n: quote! { #n } },
                quote! { ::tml::network::ModelBuilder::new().input::<{ #n }>() },
            ),
            InputShape::Image(image) => {
                let c = &image.c;
                let h = &image.h;
                let w = &image.w;
                (
                    ShapeSpec::Image {
                        c: quote! { #c },
                        h: quote! { #h },
                        w: quote! { #w },
                    },
                    quote! { ::tml::network::ModelBuilder::new().image_input::<{ #c }, { #h }, { #w }>() },
                )
            }
        };

        let mut current_shape = input_shape;
        let mut layers = Vec::with_capacity(ast.layers.len());
        let mut conv_checks = Vec::with_capacity(ast.layers.len());

        for layer in &ast.layers {
            let (next_shape, builder_step) = match &layer.kind {
                LayerSpecKind::Dense { output } => match current_shape {
                    ShapeSpec::Vec { .. } => {
                        let output = quote! { #output };
                        (
                            ShapeSpec::Vec { n: output.clone() },
                            quote! { .dense::<{ #output }>() },
                        )
                    }
                    ShapeSpec::Image { .. } => {
                        return Err(syn::Error::new(
                            layer.span,
                            "dense expects a vector input; add flatten before dense",
                        ));
                    }
                },
                LayerSpecKind::ReLU => (current_shape, quote! { .relu() }),
                LayerSpecKind::Sigmoid => (current_shape, quote! { .sigmoid() }),
                LayerSpecKind::Flatten => {
                    let size = current_shape.size_expr();
                    (ShapeSpec::Vec { n: size.clone() }, quote! { .flatten() })
                }
                LayerSpecKind::Conv(conv) => {
                    match &current_shape {
                        ShapeSpec::Image { c: _, h, w } => {
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
                                    .conv::<{ #out_channels }, { #kernel_h }, { #kernel_w }, { #stride }, { #padding }>()
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

            layers.push(LayerIr { builder_step });
            current_shape = next_shape;
        }

        Ok(Self {
            input_builder,
            layers,
            conv_checks,
        })
    }
}
