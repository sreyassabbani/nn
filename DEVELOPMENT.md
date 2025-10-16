# Development

1. Clone the repo.

Notice that in order to run this, I have `rust-toolchain.toml` set to `toolchain.channel = "nightly"`. You may also opt to have control of every commands by selecting `cargo +nightly ...`.


> [!NOTE]
> For the development philosophy, goals, and design decisions for this library, see [DESIGN](DESIGN.md).

# Explanations

## The `network!` proc macro

This macro is the spotlight of this library. The goal of it is to generate a struct `Network<Layers>`. The generic parameter here is a tuple of every layer that is supplied.

For example, executing `cargo run --example linear-regression --features="unstable"`, you will see the following type debugged (here it is formatted prettier):

```rs
linear_regression::main::Network<(
  nn::network::DenseLayer<784, 128>,
  nn::network::ReLU<128>,
  nn::network::DenseLayer<128, 64>,
  nn::network::Sigmoid<64>,
  nn::network::DenseLayer<64, 10>
)>
```

While parsing macro contents, `proc_macro2::TokenStream`s from `quote!` are declaratively collected into special variables in *three* main stages:

In the first stage, an instance of `parsing::NetworkDef` is formed. This is the first step done through `syn::parse_macro_input!`. These are
- `input_size: usize`
- `layers: Vec<Layer>`

Then, in the second stage, the following constants are generated, all extracted from `layers` and `input_size`
- `layer_types`
- `forward_calls`
- `layer_inits`
- other buffer setup (`max_size` from `layers` `use_buf_a`, `final_buffer`)

<details>
  <summary>The <code>parsing::Layer</code> type</summary>
  
  An enum defined as
  
  ```rs
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
  ```
</details>

??? There is a lot of bypassing that is done especially around the Rust orphan rule by defining structs temporarily during the expansion of the macro.

# Development of the `Tensor`
Goals, like always, are to utilize as many zero-cost abstractions as possible, parse and not validate, etc.

- I would like to keep the dimensions of the tensor part of the type information while also being extremely general. That is, I don't want to separately define (or even macro) `Tensor2x3` or `Tensor2x4x2`, etc. 
- Moreover, I would like a convenient `reshape()` functionality. As we are going to store the tensor data contiguously, I would like to be able to create separate "indices/views" onto the tensor.

Looking at both requirements at the same time, we could do either:

1. `Tensor<(1, 2, 3)>`
1a.`Tensor<[1, 2, 3]>` 
2. `Tensor<20>` with `TensorView<(1, 2, 3)>`
  - Combining two tensors must be done via a `TensorView` (this is the only way while preventing ambiguity).
