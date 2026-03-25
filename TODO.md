# TODO

This document is the current high-level backlog and ideation scratchpad for `tml`.

It is intentionally a mix of:
- product direction
- public API design goals
- subsystem architecture goals
- diagnostics goals
- concrete implementation phases
- open questions that should be answered before writing more code

The main principle is simple:

> Do not just make the library more capable. Make it more coherent, more lovable, and more obviously the right way to build typed ML systems in Rust.

## North Star

`tml` should aim to become a real framework, but not by imitating the least inspiring parts of existing frameworks.

The strongest version of `tml` would be:
- shape-first
- compile-time constrained where it matters
- minimal in surface area
- expressive in architecture composition
- explicit about invariants
- excellent in diagnostics
- internally powered by real tensor autograd
- externally pleasant enough that users want to keep using it

This means:
- no public API that forces users to restate inferred facts
- no public API that makes users manually wire obvious dimension metadata
- no macro-owned architecture
- no accidental split between the "nice syntax" path and the "real power" path

## Active Direction

The strongest current direction is:
- `tml` should feel like architecture as a typed language
- there should be one blessed architecture syntax family
- that syntax family should be `network!`
- plain Rust `let` bindings should handle naming and reuse
- `network!` should compile into typed architecture blueprints
- blueprints and materialized model state should be distinct concepts
- the public architecture DSL should own its own shape/input syntax
- axis names should be visible early in the architecture language

Some ideas that could easily be dismissed as "later" should instead be active design constraints now:
- axis-tagged architecture input syntax
- shape-trace diagnostics
- explicit graph combinator contracts
- blueprint vs model-state separation
- thinking in transforms instead of just layers

## What The Public API Should Feel Like

There should be one blessed user path:

1. Define reusable architecture blueprints with `network!`.
2. Define a rooted architecture with `network!` and one input declaration.
3. Materialize it into model state.
4. Train with `fit(samples, loss, TrainConfig)`.

Example direction:

```rust
let stem = network! {
    conv(32, kernel: 3, pad: 1) -> relu
};

let arch = network! {
    input(channels: 3, height: 32, width: 32)
    -> stem
    -> concat(channels)[
        conv(32, kernel: 1),
        conv(32, kernel: 3, pad: 1),
        conv(32, kernel: 5, pad: 2),
    ]
    -> flatten
    -> dense(10)
};

let mut model = arch.init(seed: 7);

let report = model.fit(
    &samples,
    losses::cross_entropy,
    TrainConfig::adam(1e-3).epochs(10).batch(32).shuffle(7),
);
```

The key properties of that API:
- the input shape is declared once
- layers only introduce local new facts
- reusable pieces are first-class values
- blueprints and model state are not conflated
- training stays compact
- the syntax feels like architecture, not like compiler appeasement

## One Core, One Skin

There should be one real architecture core, and one primary user-facing syntax skin over it.

That means:
- a typed internal `Arch`/blueprint algebra should be the real engine
- `network!` should lower into that engine for both reusable chunks and rooted architectures
- the macro should not own the training logic, inference logic, or shape logic

This is the right split:
- macros are allowed to improve syntax
- macros are not allowed to be the architecture

The current lean is:
- keep exactly one public architecture macro: `network!`
- do not introduce a second equally-official `block!`
- use plain Rust `let` for reusable blueprint naming

## API Principles

### 1. Users specify input shape once

Bad:
- public `Linear<In, Out>`
- public `Conv2d<InChannels, OutChannels, ...>`
- any path where users repeat known intermediate dimensions

Good:
- `input(channels: 3, height: 32, width: 32) -> conv(32, kernel: 3, pad: 1) -> relu -> dense(10)`

### 2. Layers should describe only their local intent

Examples:
- `dense(64)` means "produce 64 features"
- `conv(32, kernel: 3, pad: 1)` means "produce 32 channels with a 3-sized kernel and padding 1"
- `flatten` means "collapse spatial structure"

The library should infer the rest.

### 3. No dimension-specific public conv API

Avoid:
- `conv1d`
- `conv2d`
- `conv3d`

Prefer:
- one `conv`
- infer spatial rank from the input shape

Examples:
- `input(channels: 4, length: 1024) -> conv(32, kernel: 7)`
- `input(channels: 3, height: 224, width: 224) -> conv(32, kernel: 3)`
- `input(channels: 1, depth: 64, height: 64, width: 64) -> conv(8, kernel: (3, 3, 5))`

Meaning of `kernel: 3`:
- repeat kernel size `3` across all spatial axes

Meaning of `kernel: (3, 5)`:
- use an anisotropic kernel

The same rule should hold for:
- `pad(...)`
- `stride(...)`
- `dilation(...)`

## Architecture DSL Owns Its Own Shape Syntax

The architecture language should not awkwardly nest tensor-shape macros as a primary public style.

Bad as the main architecture-facing story:

```rust
network! {
    input(shape!(channels: 3, height: 32, width: 32))
}
```

Preferred architecture-level syntax:

```rust
network! {
    input(channels: 3, height: 32, width: 32)
}
```

and for flat inputs:

```rust
network! {
    input(features: 784)
}
```

Tensor-level syntax can still use `shape!(...)`.
Architecture-level syntax should use architecture-owned input syntax.

## Reusable Architecture Pieces

The framework should treat reusable subgraphs as first-class values.

That means a user should not need to write:
- a custom struct
- an `impl Module`
- manual plumbing for obvious architecture reuse

Instead:

```rust
let stem = network! {
    conv(32, kernel: 3, pad: 1) -> relu
};

let arch = network! {
    input(channels: 3, height: 32, width: 32)
    -> stem
    -> stem
    -> flatten
    -> dense(10)
};

let mut model = arch.init(seed: 42);
```

This also opens the door to better composition features:
- `residual(block)`
- `concat(axis)[...]`
- `sum[...]`
- `repeat(block, n)`
- later: `fork`, `zip`, `heads`

Examples worth chasing:

```rust
let res = network! {
    conv(32, kernel: 3, pad: 1) -> relu -> conv(32, kernel: 3, pad: 1)
};

let arch = network! {
    input(channels: 32, height: 64, width: 64)
    -> residual(res)
    -> flatten
    -> dense(10)
};
```

```rust
let arch = network! {
    input(channels: 3, height: 224, width: 224)
    -> concat(channels)[
        conv(32, kernel: 1),
        conv(32, kernel: 3, pad: 1),
        conv(32, kernel: 5, pad: 2),
    ]
    -> dense(1000)
};
```

### Reuse semantics

This should be an active design rule now, not a later detail.

If a user writes:

```rust
let stem = network! {
    conv(32, kernel: 3, pad: 1) -> relu
};
```

and then uses `stem` multiple times, the semantics should be:
- `stem` is a blueprint
- each use duplicates structure with fresh parameters
- reuse is structural reuse, not implicit weight sharing

If shared parameters are later introduced, they must be explicit.

### Blueprint vs model state

This should also be treated as a first-class design concept now.

`network!` should likely produce a blueprint.
Materialized parameters should likely come from something like:

```rust
let mut model = arch.init(seed: 42);
```

Benefits:
- architecture and parameter state are separate concepts
- summaries and inspection become cleaner
- reuse semantics become cleaner
- device placement later becomes cleaner
- serialization/introspection later becomes cleaner

## Tensor System Goals

The tensor subsystem still needs major work.

Current strengths:
- typed shapes
- contiguous storage
- reshape
- views/borrows
- elementwise math
- some matrix operations

Still needed:
- broadcasting
- better reductions
- more matrix/tensor ops
- more principled layout handling
- better diagnostics
- a real story for batched computation
- a real tensor autograd engine

### Tensor priorities

#### Broadcasting

Need a real broadcasting model for:
- scalar with tensor
- vector with matrix
- channel bias additions
- loss computations
- parameterized operations

This will become foundational once tensor autograd exists.

#### Better reductions

Need reductions that are not just global summaries:
- reduce along axis
- keepdims-like behavior or an equivalent shape-safe design
- sum/mean/max over selected axes

#### Layout / strides

Right now tensors are basically contiguous row-major views.

Eventually we need:
- explicit contiguity knowledge
- a real layout concept
- rules for when reshape is legal
- rules for when borrowed reshape is legal
- non-contiguous or strided views, if introduced, with strong semantics

Important note:
- borrowed reshape is correct in the current contiguous model
- once non-contiguous views exist, reshape must become layout-aware

#### More matrix/tensor ops

Need a much stronger base:
- batched matmul
- generalized transpose/permutation
- concat / split
- softmax
- normalization primitives
- pooling
- masking/select-style operations

## Tensor Autograd

This is one of the biggest remaining architectural gaps.

Right now:
- scalar autodiff exists separately
- network training uses handwritten backprop
- conv uses handwritten gradients

That split is not a durable architecture.

The long-term goal should be:
- tensor operations become the differentiable primitives
- layers become compositions of tensor ops
- optimizers step parameter tensors

This should pair with the architecture-level idea that models are typed blueprints of transforms, not just lists of layers with handwritten backprop paths.

This does not need to leak into the common high-level API.

Users should ideally keep writing:

```rust
model.fit(&samples, losses::mse, TrainConfig::adam(1e-3))
```

But internally, that should eventually lower into tensor-autograd-backed training.

### Autograd migration plan

1. Build a minimal tensor autograd core.
2. Support:
   - add
   - mul
   - matmul
   - broadcasted bias add
   - relu
   - mse
   - sum / mean
3. Rebuild dense + relu + loss on top of it.
4. Keep conv manual briefly if needed.
5. Later rebuild conv on the same tensor-autograd substrate.

### Important product principle

The existence of autograd should be obvious in the architecture and capabilities of the library, but not necessarily in the everyday syntax of users.

The common API should stay clean.

## Batching and Dynamic Shapes

General dynamic axes may be too expensive as an immediate redesign target.

The more practical near-term idea is:
- keep sample shapes statically known
- introduce a first-class `Batch<Shape>` concept before general `Dyn`

Why:
- batch size is the most urgent runtime-sized dimension
- it avoids infecting the entire shape system too early
- it preserves stronger static guarantees on per-sample structure

This should be revisited after tensor autograd and broader tensor ops are stronger.

## Diagnostics

Diagnostics are part of the product.

The current style of errors is too low-level in places:
- `Assert<false>: IsTrue`
- internal helper types leaking
- ugly internal composition types in messages

That is not acceptable as a long-term experience in a type-heavy framework.

### Better constraint names

Prefer named concepts over boolean-const hacks in user-facing failures.

Examples:
- `ReshapePreservesElementCount<From, To>`
- `ConvKernelFitsInput<Input, Kernel, Stride, Padding>`
- `DenseRequiresFlatInput<Input>`

Rust will still not emit perfect prose, but this is far better than raw `Assert<false>`.

### Pretty shape syntax in errors

Shape readability matters.

Important ideas:
- axis-tagged shapes:

```rust
type Image = shape!(channels: 3, height: 32, width: 32);
```

- better macro expansion strategies so error messages preserve user-written shape syntax longer
- avoiding public leakage of low-level storage/core helper types

### Shape trace diagnostics

This should be treated as an active product goal now.

Errors should try to explain the architecture's shape flow, not just a final mismatch.

Ideal direction:

```text
cannot apply residual(body)

current shape:
  (channels: 32, height: 64, width: 64)

body output shape:
  (channels: 64, height: 32, width: 32)

residual requires the body to preserve shape
```

That is a far better product than a raw type-level assertion failure.

## Architecture Algebra

The internal architecture representation should eventually support more than a plain chain.

Core operations worth designing around:
- `then`
- `residual`
- `concat`
- `sum`
- `repeat`
- later: `fork`, `zip`, `heads`

The key idea:
- model structure should be a small algebra of composable graph pieces
- not a zoo of unrelated builder paths

This is what allows `tml` to feel like a language for architectures, not just a set of structs.

### Semantics and contracts

These should be active design constraints now.

#### `residual(block)`

Semantics:
- take current tensor `x`
- compute `y = block(x)`
- return `x + y`

Contract:
- `block` must accept the current shape
- `block` must return the exact same shape

Future extension worth designing for now:
- projection residuals with an explicit skip transform

#### `concat(axis)[a, b, c]`

Semantics:
- feed the same input to every branch
- concatenate outputs along the explicit axis

Contract:
- all branches accept the same input
- all outputs have the same rank
- all non-concatenated axes match exactly

Explicit axis should be required from day one.

#### `sum[a, b, c]`

Semantics:
- feed the same input to every branch
- elementwise add the results

Contract:
- all branches accept the same input
- all branch outputs have the exact same shape

`sum[...]` is worth treating as an early important primitive, not just a late embellishment.

#### `repeat(n, block)`

Semantics need to be deliberate:
- most natural default is fresh parameters at each repetition
- likely best first version requires shape-preserving blocks

This is important enough to influence the algebra now, even if not implemented immediately.

## Transforms, Not Just Layers

The public mental model should shift away from "just a list of layers."

Useful categories:
- parametric transforms: `dense`, `conv`, `norm`
- structural transforms: `flatten`, `reshape`, `permute`
- merge transforms: `residual`, `concat`, `sum`
- reductions: `pool`, `mean`, `max`
- heads: classification/regression/embedding heads

This is a better organizing idea for a graph-capable architecture language.

## Multi-Head Architectures

This is important enough to keep in active view now, not as a distant afterthought.

Many real architectures do not have one output tensor.

Direction worth designing toward:

```rust
let arch = network! {
    input(channels: 3, height: 224, width: 224)
    -> stem
    -> heads {
        logits: flatten -> dense(1000),
        embedding: flatten -> dense(128),
    }
};
```

This reinforces the need for:
- a graph-capable blueprint core
- a distinction between architecture blueprints and materialized model state
- diagnostics that can talk about named outputs

## What To Avoid

Things that may be technically workable but should not become the core public story:
- public `Module` boilerplate as the main path
- public `Linear<In, Out>`-style repeated dimensions
- separate user-facing paths for vector models and image models
- separate user-facing paths for 1D/2D/3D conv
- training APIs that explode into too many builder objects
- a split where the macro path is "nice" but the non-macro path is where the real power lives
- awkward nested macro syntax as the main architecture input story

## Open Questions

These are real design questions, not implementation details.

### 1. What is the one blessed syntax?

Options:
- `network!` as the public face for both reusable chunks and rooted architectures
- some alternative single syntax skin, if and only if it is clearly better

What should not happen:
- multiple equal-status public styles competing in the docs

### 2. How much should be macro syntax?

Macro syntax is justified when it buys:
- type-safe literal-like dimensions such as `dense(64)`
- clearer architecture notation

Macro syntax is not justified when it owns actual framework behavior.

### 3. When should batching become first-class?

Likely before general dynamic axes, but after the tensor/autograd direction is clearer.

### 4. How far should the architecture algebra go?

At minimum:
- sequential
- residual
- concat
- sum

Question:
- should branching and merging be part of the first real redesign, or a later extension?

### 5. How much should the high-level API hide autograd?

Current instinct:
- mostly hide it
- keep it explicit only for advanced/lower-level APIs

## Concrete Next Phases

### Phase 1: Freeze public direction

- Decide the one blessed public syntax
- Decide whether `network!` alone is the public architecture language
- Decide the first version of the architecture algebra
- Decide blueprint vs materialized model-state semantics
- Decide architecture-owned input/shape syntax

### Phase 2: Build the real architecture core

- typed internal architecture representation
- reusable block values
- sequential composition
- residual composition
- concat composition
- sum composition
- blueprint materialization into model state

### Phase 3: Make syntax lower into the core

- `network!`
- possibly compatibility layers for current builder APIs

### Phase 4: Strengthen tensor foundations

- broadcasting
- axis-aware reductions
- stronger matrix/tensor ops
- layout/contiguity semantics

### Phase 5: Introduce tensor autograd

- minimal op coverage
- dense/relu/loss migration first
- conv later

### Phase 6: Diagnostics pass

- named constraints
- prettier type errors
- less internal type leakage

## Litmus Test

At every design step, ask:

1. Is this the only thing the user should have to say here?
2. Is the library forcing them to repeat a fact it already knows?
3. Is this exposing implementation detail instead of modeling intent?
4. Does this make the core more unified, or create another path?
5. Will this feel obvious and satisfying to use six months from now?

If the answer is no, stop and redesign before adding more code.
