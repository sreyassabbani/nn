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

## Quickstart (current API)

Nightly-only (until `generic_const_exprs` stabilizes). In your crate root:
```rs
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
```

Network DSL (vector or image inputs, with explicit `flatten` before dense layers):
```rs
use tml::network;

let mut net = network! {
  input(3, 32, 32) -> conv(8, 3, 1, 1) -> relu -> flatten -> dense(10) -> output
};

let logits = net.predict(&[0.0; 3 * 32 * 32]);
```

Typed tensors use `shape!(...)` for shape syntax, constructors on `Tensor<Shape>`, and `tensor![...]` for literals:
```rs
use tml::{Tensor, tensor};

type Image = tml::shape!(3, 32, 32);

let zeros: Tensor<Image> = Tensor::zeros();
let grid = tensor![[1.0, 2.0], [3.0, 4.0]];
let flat = grid.reshape::<tml::shape!(4)>();
```

This style is what the public tensor API is optimized around:
- name shapes with `type` aliases when they matter semantically
- use `Tensor::<Shape>::zeros()` / `random()` for construction
- use `tensor![...]` for actual tensor literals
- use `reshape::<shape!(...)>()` when you want a different view of the same storage

The examples in [`tml/examples`](/Users/sreysus/workflow/tml/tml/examples) are a good starting point:
- [`tensor_basics.rs`](/Users/sreysus/workflow/tml/tml/examples/tensor_basics.rs) shows `shape!`, `Tensor<Shape>`, literals, indexing, and reshape
- [`conv.rs`](/Users/sreysus/workflow/tml/tml/examples/conv.rs) shows the typed network DSL on image inputs
- [`linear_regression.rs`](/Users/sreysus/workflow/tml/tml/examples/linear_regression.rs) and [`quadratic_regression.rs`](/Users/sreysus/workflow/tml/tml/examples/quadratic_regression.rs) show simple end-to-end training flows

Rust-like autodiff with a tape:
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

Expression DSL:
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
