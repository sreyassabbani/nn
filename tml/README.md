# `tml`

Experimental, nightly-only, type-safe machine learning library for Rust.

`tml` focuses on typed tensors, a typed network DSL, and learning-oriented implementations built from scratch.

## Nightly requirement

This crate currently depends on `generic_const_exprs`, so downstream crates generally need:

```rs
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
```

## Quickstart

```rs
use tml::{InitConfig, TrainConfig, network};

let arch = network! {
    input(features: 1) -> dense(8) -> relu -> dense(1)
};

let mut net = arch.materialize(InitConfig::new().seed(7));

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

## Tensors

```rs
use tml::{Tensor, shape, tensor};

type Image = shape!(3, 32, 32);

let zeros: Tensor<Image> = Tensor::zeros();

let grid = tensor![
    [1., 2.],
    [3., 4.]
];

let flat = grid.as_ref().reshape::<shape!(4)>();
assert_eq!(flat.as_slice(), &[1., 2., 3., 4.]);
```

## Examples

- [`tensor_basics.rs`](https://github.com/sreyassabbani/tml/blob/main/tml/examples/tensor_basics.rs)
- [`conv.rs`](https://github.com/sreyassabbani/tml/blob/main/tml/examples/conv.rs)
- [`named_sources.rs`](https://github.com/sreyassabbani/tml/blob/main/tml/examples/named_sources.rs)
- [`rust_fragments.rs`](https://github.com/sreyassabbani/tml/blob/main/tml/examples/rust_fragments.rs)
- [`api-previews/current/satellite_fpn.rs`](https://github.com/sreyassabbani/tml/blob/main/api-previews/current/satellite_fpn.rs)
- [`api-previews/current/spectrogram_events.rs`](https://github.com/sreyassabbani/tml/blob/main/api-previews/current/spectrogram_events.rs)
- [`api-previews/frontier/`](https://github.com/sreyassabbani/tml/tree/main/api-previews/frontier)
- [`linear_regression.rs`](https://github.com/sreyassabbani/tml/blob/main/tml/examples/linear_regression.rs)

## Version note

`tml` `0.3.0` and later refer to this experimental machine learning library. Older crates.io releases under the same name were for an unrelated project.
