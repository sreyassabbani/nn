While iterating and refactoring this project, I had the following in mind.

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
There was a lot of thought put into API design. For example, consider where weights associated between two layers would be stored. Although it might not have seemed logical to associate weights with one layer, this ended up being the case to avoid keeping track of more indices, slightly improving performance and code readability.

### Initialization

```rs
let network = ModelBuilder::new()
  .input(dim!(128))
  .hidden(dense!(64).activation(ReLU))
  .output(dim!(1));
```
This would be the dream to walk towards. Firstly, the macro might be yelling at you. Let me explain. Since I wanted to put type-safety as the highest priority, I would need to have called an initializer function like `LayerBuilder::dense::<128, 128>()` no matter what because of const generics, which just looks ugly. So, a macro needed to be invoked in this context, which I call `dense!(n)`.

However, the API ended up looking like this:
```rs
let network = ModelBuilder::new()
  .input(dim!(128))
  .hidden(dense!(64, 64).activation(ReLU))
  .output(dim!(1));
```
The stable Rust compiler can unfortunately not do inferences with const generics yet, so you have to be very explicit with the input and output dimensions of each layer. There [have](https://users.rust-lang.org/t/rust-type-inference-failing-with-const-generic/122682) [been](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://stackoverflow.com/questions/71233548/rust-cannot-infer-the-value-of-const-parameter-when-a-default-is-provided&ved=2ahUKEwi0te-q7rSNAxWMSzABHVFpA7UQFnoECCEQAQ&usg=AOvVaw3mVoJL87CW0fTQjEX-4A8n) [numerous](https://github.com/rust-lang/rust/issues/98931) discussions on const generic inferences before, and it seems to be [pretty close](https://blog.rust-lang.org/inside-rust/2025/03/05/inferred-const-generic-arguments/) to becoming stabilized; this is something I would go back and fix once the Rust team stabilizes it.

For now, to use the cleaner syntax, you must enable the `unstable` library feature and write `#![feature(generic_arg_infer)]` on Nightly builds.

Even in Nightly, I'm thinking of making the API look like this:
```rs
let network = ModelBuilder::new()
  .input(dim!(128))
  .hidden(LayerBuilder::dense(dim!(64)).activation(ReLU))
  .output(dim!(1));
```
Although it's significantly more terse, I feel it makes the structure of what you're building so much more obvious.

### Training/Testing

```rs
let training_data = [(1.0, 2.0), /* more */].map(DataSample::from);
network.train(&training_data);

let testing_data = [(3.0, 15.0), /* more */].map(DataSample::from);
let cost = network.run(&testing_data);
dbg!(cost);
```

> [!NOTE]
> This part is still being refactored.
