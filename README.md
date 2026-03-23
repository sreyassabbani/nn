# `tml` _/ˈtɪ̆ml̩/_

An **experimental** type-safe machine learning library. Implementing concepts from scratch is cool, so that's what I'm doing here.

### Philosophy

A _lot_ of thought went into developer experience/API design, internal data flow, and performance. I'm starting to develop a set of principles in my everyday work nowadays, with the "parent principle" resting on the library user (everything rests on the end user): <ins>make the default the correct choice</ins>. I don't want to sell a religion, but the following is a tentative list of the best basis (orthogonal axes) that spans good software.

- make invalid states unrepresentable
  - special case: [parse, don't validate](https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/)
- design for local reasoning
- fast[^1]

To be the best learning experience for me, I'm adding another principle:

- no external math libraries

This will limit the performance of the library, but this is a learning experience anyway. 

For more on the development philosophy, goals, and design decisions for this library, see [DESIGN](DESIGN.md).

### Development

1. Clone the repo.
2. Requirements: `rustc` nightly.

Notice that in order to run this, I have `rust-toolchain.toml` set to `toolchain.channel = "nightly"`. You may also opt to have control of every commands by selecting `cargo +nightly ...`.


[^1]: yes. and also we don't talk about Rust compile times

## Quickstart (current, work-in-progress API)

Nightly-only (until `generic_const_exprs` stabilizes). In your crate root:
```rs
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
```

### Network DSL

`network!` is syntax sugar for building a typed `Sequential` model. The public builder API is the core runtime surface, and `TrainConfig` owns the optimizer:
```rs
use tml::{TrainConfig, network};

let mut net = network! {
    input(1) -> dense(8) -> relu -> dense(1) -> output
};

let samples = [
    ([0.0], [1.0]),
    ([1.0], [3.0]),
].map(tml::Sample::from);

let loss = net.fit(
    &samples,
    TrainConfig::sgd(0.05)
        .epochs(200)
        .batch_size(2)
        .shuffle_seed(7),
);

let prediction = net.predict(&[0.5]);
```

For image inputs, keep `flatten` explicit before dense layers:
```rs
use tml::network;

let net = network! {
    input(3, 32, 32) -> conv(8, 3, 1, 1) -> relu -> flatten -> dense(10) -> output
};

let logits = net.predict(&[0.0; 3 * 32 * 32]);
```

You can also build the same runtime manually, without the DSL:
```rs
use tml::ModelBuilder;

let net = ModelBuilder::new()
    .input::<1>()
    .dense::<8>()
    .relu()
    .dense::<1>()
    .build();
```

### Tensor Algebra

Typed tensors use `shape!(...)` for shape syntax, constructors on `Tensor<Shape>`, `tensor![...]` for literals, and `TensorRef` / `TensorMut` for borrowed sub-tensors:
```rs
use tml::{Tensor, shape, tensor};

type Image = shape!(3, 32, 32);

let zeros: Tensor<Image> = Tensor::zeros();

let grid = tensor![
    [1., 2.],
    [3., 4.]
];

// convert 2x2 grid to 4 flat
let flat = grid.as_ref().reshape::<shape!(4)>();
assert_eq!(flat.as_slice(), &[1., 2., 3., 4.]);

// grab sub-tensors
let row = grid.get_ref(1);
assert_eq!(row.as_slice(), &[3., 4.]);
```

This style is what the public tensor API is optimized around:
- name shapes with `type` aliases when they matter semantically
- use `Tensor::<Shape>::zeros()` / `random()` for construction
- use `tensor![...]` for actual tensor literals
- use `get_ref()` / `get_mut()` when you want borrowed sub-tensors
- use `reshape::<shape!(...)>()` when you want the same data with a different shape type

The examples in [`tml/examples`](tml/examples) are a good starting point:
- [`tensor_basics.rs`](tml/examples/tensor_basics.rs) shows `shape!`, `Tensor<Shape>`, `TensorRef`, `TensorMut`, literals, indexing, and reshape
- [`conv.rs`](tml/examples/conv.rs) shows the typed network DSL on image inputs
- [`sequential_manual.rs`](tml/examples/sequential_manual.rs) shows the same training runtime without `network!`
- [`linear_regression.rs`](tml/examples/linear_regression.rs) and [`quadratic_regression.rs`](tml/examples/quadratic_regression.rs) show simple end-to-end training flows

### Automatic differentiation

The scalar autodiff modules are still available, but they are currently an experimental side path. The main network/tensor training path uses explicit layer backprop plus optimizers.
```rs
use tml::Tape;

let mut tape = Tape::new();
let x = tape.input("x", 2.0);
let y = tape.input("y", 1.0);
let z = (x * x + y.sin()).cos();
let grads = tape.gradients(&z);

println!("value = {}", grads.value);
println!("dx = {:?}", grads.get("x"));
println!("dy = {:?}", grads.get("y"));
```

### Expression DSL
```rs
use tml::expr;

let expr = expr! {
  inputs: [x, y]
  x -> Pow(2) -> @x2
  y -> Cos -> @ycos
  (@x2, @ycos) -> Add -> @out
  output @out
};

let (value, grad) = expr.eval_one(&[2.0, 1.0]);
```
