While iterating and refactoring this project, I had the following in mind.

> [!NOTE]
> This document contains historical design thinking. The current public
> direction is blueprint-first:
>
> - Rust defines reusable fragments and contracts
> - `network!` owns composition
> - rooted blueprints materialize explicit model state
>
> For the current branch reality and API pressure-testing, see
> [`ARCHITECTURE_LANGUAGE.md`](ARCHITECTURE_LANGUAGE.md).

## Overview of Goals and Philosophy
- Strong Type Safety: Through Rust, leverage type-driven design features and patterns (const generics, zero-cost abstractions, typestate, etc) to catch mismatches and human errors <ins>at compile time</ins>; higly reflective of ["parse, don't validate"](https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/).
- Ergonomic API: Provide a <ins>clear and concise API</ins> for model construction, training, and testing.
- Constrained Modularity: Allow legal mixing and matching of layers without committing to a rigid static pipeline.
- Performance: Use <ins>zero-cost abstractions</ins>, heap-allocated (`Box`-ed) arrays, and row-major order data layout.
- Minimal Dependencies: Keep <ins>external dependencies to a minimum</ins>, relying on crates and the standard library at very select times; e.g., the `rand` crate.

These were the main design philosophies I kept in mind through the project. For more detail, consult the remaining of this document.

# Linear Algebra
Since I wanted to make this project absolutely from the ground up, I would end up making some quick linear algebra utilities.

1. Store vectors and matrices in row-major order for <ins>cache-friendly access</ins> during matrix-vector multiplication.
2. Represent fixed-size arrays behind a `Box<[T; N]>` or `Box<[[T; N]; M]>` to <ins>minimize stack usage while retaining compile-time size checks</ins>.
```rs
pub struct Matrix<T, const N: usize, const M: usize> {
  entries: Box<[[T; N]; M]>,
}
```
3. Generate random vector and matrix using the `rand` crate.

# API design
The project originally explored builder-style APIs, but the current direction is
cleaner:

- reusable fragments and contracts live in ordinary Rust
- `network!` owns architecture composition
- the core enforces shape and extent rules
- rooted blueprints materialize explicit trainable model state

### Current shape of the API

```rs
use tml::{InitConfig, TrainConfig, network};

let arch = network! {
  input(features: 128) -> dense(64) -> relu -> dense(1)
};

let mut model = arch.materialize(InitConfig::new().seed(7));
```

For image-like models:

```rs
use tml::{InitConfig, network, vision};

let tower = vision::common::stem::<8, 16>();

let arch = network! {
  input(channels: 3, height: 32, width: 32)
    -> tower
    -> save(low)
    -> vision::common::residual_block::<16>()
    -> concat_from(low, channels)
    -> conv(12, kernel: 1)
    -> flatten
    -> dense(10)
};

let model = arch.materialize(InitConfig::new());
```

### Training/Testing

```rs
use tml::{Float, Sample, TrainConfig};

let samples = [(1.0, 2.0), (2.0, 4.0)].map(|(x, y)| Sample::new([x], [y]));

let loss = model.fit(
  &samples,
  TrainConfig::sgd(0.01)
    .epochs(200)
    .batch_size(2)
    .shuffle_seed(7),
);

let y = model.predict(&[3.0]);
dbg!(loss, y);
```
