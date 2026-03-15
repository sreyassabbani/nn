use crate::ir::NetworkIr;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};

pub fn generate_network(ir: &NetworkIr) -> TokenStream2 {
    let layer_count = ir.layers.len();
    let input_size_expr = &ir.input_size;
    let output_size_expr = &ir.output_size;
    let max_size_expr = &ir.max_buf_size;
    let conv_checks = &ir.conv_checks;

    let layer_types = ir
        .layers
        .iter()
        .map(|layer| layer.layer_type.clone())
        .collect::<Vec<_>>();
    let layer_io = ir
        .layers
        .iter()
        .map(|layer| (layer.in_size.clone(), layer.out_size.clone()))
        .collect::<Vec<_>>();

    let layer_inits = layer_types.iter().map(|layer_type| {
        quote! { <#layer_type>::init() }
    });

    let act_idents = (0..layer_count)
        .map(|i| format_ident!("act_{}", i))
        .collect::<Vec<_>>();
    let grad_idents = (0..layer_count)
        .map(|i| format_ident!("grad_{}", i))
        .collect::<Vec<_>>();

    let activation_fields = act_idents
        .iter()
        .zip(layer_io.iter())
        .map(|(ident, (_, out_size))| {
            quote! { #ident: Box<[::tml::Float; #out_size]> }
        });
    let gradient_fields = grad_idents
        .iter()
        .zip(layer_io.iter())
        .map(|(ident, (_, out_size))| {
            quote! { #ident: Box<[::tml::Float; #out_size]> }
        });

    let activation_inits = act_idents
        .iter()
        .zip(layer_io.iter())
        .map(|(ident, (_, out_size))| {
            quote! { #ident: Box::new([Default::default(); #out_size]) }
        });
    let gradient_inits = grad_idents
        .iter()
        .zip(layer_io.iter())
        .map(|(ident, (_, out_size))| {
            quote! { #ident: Box::new([Default::default(); #out_size]) }
        });

    let mut forward_calls_ws = Vec::new();
    for (i, (in_size, out_size)) in layer_io.iter().enumerate() {
        let layer_idx = ::syn::Index::from(i);
        let input_ref = if i == 0 {
            quote! { workspace.act_input.as_ref() }
        } else {
            let prev = &act_idents[i - 1];
            quote! { workspace.#prev.as_ref() }
        };
        let out_ident = &act_idents[i];
        forward_calls_ws.push(quote! {
            ::tml::network::Layer::<{ #in_size }, { #out_size }>::forward(
                &layers.#layer_idx,
                #input_ref,
                workspace.#out_ident.as_mut(),
            );
        });
    }

    let mut forward_calls_buf = Vec::new();
    let mut use_buf_a = true;
    for (i, (in_size, out_size)) in layer_io.iter().enumerate() {
        let layer_idx = ::syn::Index::from(i);
        let (input_buf, output_buf) = if use_buf_a {
            (quote! { buf_a }, quote! { buf_b })
        } else {
            (quote! { buf_b }, quote! { buf_a })
        };

        forward_calls_buf.push(quote! {
            let input_arr: &[::tml::Float; #in_size] =
                <&[::tml::Float; #in_size]>::try_from(&#input_buf[..#in_size])
                    .expect("invalid input buffer size");
            let output_arr: &mut [::tml::Float; #out_size] =
                <&mut [::tml::Float; #out_size]>::try_from(&mut #output_buf[..#out_size])
                    .expect("invalid output buffer size");
            ::tml::network::Layer::<{ #in_size }, { #out_size }>::forward(
                &self.layers.#layer_idx,
                input_arr,
                output_arr,
            );
        });

        use_buf_a = !use_buf_a;
    }

    let final_buffer = if (layer_count % 2) == 1 {
        quote! { buf_b }
    } else {
        quote! { buf_a }
    };

    let mut backward_calls = Vec::new();
    for (i, (in_size, out_size)) in layer_io.iter().enumerate().rev() {
        let layer_idx = ::syn::Index::from(i);
        let input_act = if i == 0 {
            quote! { workspace.act_input.as_ref() }
        } else {
            let prev = &act_idents[i - 1];
            quote! { workspace.#prev.as_ref() }
        };
        let output_act = &act_idents[i];
        let output_grad = &grad_idents[i];
        let input_grad = if i == 0 {
            quote! { workspace.grad_input.as_mut() }
        } else {
            let prev = &grad_idents[i - 1];
            quote! { workspace.#prev.as_mut() }
        };

        backward_calls.push(quote! {
            ::tml::network::Layer::<{ #in_size }, { #out_size }>::backward(
                &mut layers.#layer_idx,
                #input_act,
                workspace.#output_act.as_ref(),
                workspace.#output_grad.as_ref(),
                #input_grad,
                lr,
            );
        });
    }

    let last_act_ident = act_idents
        .last()
        .cloned()
        .unwrap_or_else(|| format_ident!("act_input"));
    let last_grad_ident = grad_idents
        .last()
        .cloned()
        .unwrap_or_else(|| format_ident!("grad_input"));

    quote! {
        {
            const fn __nn_max(a: usize, b: usize) -> usize {
                if a > b { a } else { b }
            }

            const INPUT_SIZE: usize = #input_size_expr;
            const OUTPUT_SIZE: usize = #output_size_expr;
            const MAX_BUF: usize = #max_size_expr;

            #(#conv_checks)*

            #[derive(Debug)]
            struct Network<Layers> {
                layers: Layers,
                _buf_a: Box<[::tml::Float; MAX_BUF]>,
                _buf_b: Box<[::tml::Float; MAX_BUF]>,
            }

            #[derive(Debug)]
            struct NetworkWorkspace {
                act_input: Box<[::tml::Float; INPUT_SIZE]>,
                #(#activation_fields,)*
                grad_input: Box<[::tml::Float; INPUT_SIZE]>,
                #(#gradient_fields,)*
            }

            impl NetworkWorkspace {
                fn new() -> Self {
                    Self {
                        act_input: Box::new([Default::default(); INPUT_SIZE]),
                        #(#activation_inits,)*
                        grad_input: Box::new([Default::default(); INPUT_SIZE]),
                        #(#gradient_inits,)*
                    }
                }
            }

            impl Default for NetworkWorkspace {
                fn default() -> Self {
                    Self::new()
                }
            }

            impl Network<(#(#layer_types,)*)> {
                pub fn new() -> Self {
                    Network {
                        layers: (#(#layer_inits,)*),
                        _buf_a: Box::new([Default::default(); MAX_BUF]),
                        _buf_b: Box::new([Default::default(); MAX_BUF]),
                    }
                }

                pub fn workspace(&self) -> NetworkWorkspace {
                    NetworkWorkspace::new()
                }

                pub fn predict(
                    &self,
                    input: &[::tml::Float; INPUT_SIZE],
                ) -> [::tml::Float; OUTPUT_SIZE] {
                    let mut workspace = NetworkWorkspace::new();
                    let output = self.predict_with_workspace(input, &mut workspace);
                    let mut result = [0.0 as ::tml::Float; OUTPUT_SIZE];
                    result.copy_from_slice(output);
                    result
                }

                fn predict_with_workspace_layers<'a>(
                    layers: &(#(#layer_types,)*),
                    input: &[::tml::Float; INPUT_SIZE],
                    workspace: &'a mut NetworkWorkspace,
                ) -> &'a [::tml::Float; OUTPUT_SIZE] {
                    workspace.act_input.copy_from_slice(input);
                    #(#forward_calls_ws)*
                    &workspace.#last_act_ident
                }

                pub fn predict_with_workspace<'a>(
                    &self,
                    input: &[::tml::Float; INPUT_SIZE],
                    workspace: &'a mut NetworkWorkspace,
                ) -> &'a [::tml::Float; OUTPUT_SIZE] {
                    Self::predict_with_workspace_layers(&self.layers, input, workspace)
                }

                pub fn predict_in_place(
                    &mut self,
                    input: &[::tml::Float; INPUT_SIZE],
                ) -> [::tml::Float; OUTPUT_SIZE] {
                    let (buf_a, buf_b) = (&mut self._buf_a[..], &mut self._buf_b[..]);

                    buf_a[..INPUT_SIZE].copy_from_slice(input);

                    #(#forward_calls_buf)*

                    let mut result = [0.0 as ::tml::Float; OUTPUT_SIZE];
                    result.copy_from_slice(&#final_buffer[..OUTPUT_SIZE]);
                    result
                }

                fn backward_with_workspace_layers(
                    layers: &mut (#(#layer_types,)*),
                    workspace: &mut NetworkWorkspace,
                    lr: ::tml::Float,
                ) {
                    #(#backward_calls)*
                }

                fn train_step_layers(
                    layers: &mut (#(#layer_types,)*),
                    input: &[::tml::Float; INPUT_SIZE],
                    target: &[::tml::Float; OUTPUT_SIZE],
                    mut workspace: NetworkWorkspace,
                    lr: ::tml::Float,
                ) -> (NetworkWorkspace, ::tml::Float) {
                    Self::predict_with_workspace_layers(layers, input, &mut workspace);
                    let loss = ::tml::network::mse_loss(
                        workspace.#last_act_ident.as_ref(),
                        target,
                        workspace.#last_grad_ident.as_mut(),
                    );
                    Self::backward_with_workspace_layers(layers, &mut workspace, lr);
                    (workspace, loss)
                }

                pub fn fit(
                    &mut self,
                    samples: &[::tml::Sample<INPUT_SIZE, OUTPUT_SIZE>],
                    config: ::tml::network::TrainConfig,
                ) -> ::tml::Float {
                    if samples.is_empty() || config.epochs == 0 {
                        return 0.0;
                    }

                    let mut workspace = NetworkWorkspace::new();
                    let mut total_loss = 0.0;
                    let mut steps = 0usize;
                    let layers = &mut self.layers;

                    for _ in 0..config.epochs {
                        for sample in samples {
                            let (next_workspace, loss) = Self::train_step_layers(
                                layers,
                                &sample.input,
                                &sample.target,
                                workspace,
                                config.lr,
                            );
                            workspace = next_workspace;
                            total_loss += loss;
                            steps += 1;
                        }
                    }

                    total_loss / steps as ::tml::Float
                }

                pub fn fit_default(
                    &mut self,
                    samples: &[::tml::Sample<INPUT_SIZE, OUTPUT_SIZE>],
                ) -> ::tml::Float {
                    self.fit(samples, ::tml::network::TrainConfig::default())
                }
            }

            Network::<(#(#layer_types,)*)>::new()
        }
    }
}
