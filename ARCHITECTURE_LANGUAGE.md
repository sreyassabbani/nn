# Architecture Language Prototyping

This document is the current source of truth for the `network!` DSL redesign.

It replaces the older axis-tag direction that assumed a small fixed semantic axis
ontology in the public API.

The working thesis is:

> `tml` should feel like architecture as a typed language.

That means the public API should optimize for:
- architectural clarity
- compile-time structural checks
- a single blessed path
- explicit invariants
- good diagnostics

It should not optimize for:
- making every internal runtime detail type-level
- forcing users into a global axis ontology
- adding second or third parallel API styles

## Development Philosophy

The development loop for the DSL should be strict and repetitive:

1. Prototype many real user-facing examples first.
2. Treat those examples as the actual product surface.
3. Check whether each example is genuinely expressible in Rust and in a proc-macro DSL.
4. Reject beautiful-but-impossible syntax quickly.
5. Refine the examples into explicit idioms and naming rules.
6. Only then implement or refactor the core.

This project should explicitly prefer:
- prototype-first design
- feasibility proofs over wishful thinking
- strict idioms over vague flexibility

## Hard Feasibility Boundaries

These are the current important boundaries imposed by Rust and by the chosen DSL model.

### 1. Axis labels inside `network!` are DSL symbols, not ordinary Rust values

This means code like this is **not** valid ordinary Rust:

```rust
let stem = vision::common::stem(on: c, over: [y, x], widths: [32, 64]);
```

unless `c`, `y`, and `x` are actual Rust values, which they are not in the current design.

Therefore, axis-parameterized helpers are only feasible in one of these forms:

- inside `network!`, where the macro parses the helper call as DSL syntax
- as `network!` blueprint chunks that contain free axis symbols to be resolved later
- as ordinary Rust helpers that do **not** take axis labels and instead rely on surrounding DSL defaults

The first two are the most plausible.

### 2. Dependent typing is not available

The DSL can enforce many structural rules at compile time, but it cannot become a full dependent type system.

That means the right question is never:
- "Can Rust express the ideal mathematical formulation directly?"

It is:
- "Can Rust express a useful checked approximation with good ergonomics and diagnostics?"

### 3. Built-in transforms can be open-ended without a global ontology

The earlier public `AxisRole` idea was too rigid.

A stronger model is:
- axis labels are user-chosen DSL symbols
- transforms explicitly state which labels they act on
- the macro validates label existence, reuse, propagation, and compatibility

This is still "typed" because the symbols are checked and carried structurally by the DSL.

It is not the bad kind of stringly typed programming.

## Core Direction

The public DSL should move toward:

- open-ended axis labels
- explicit per-transform axis operands
- scoped defaults to remove repetition
- reusable blueprints with free axis symbols
- namespaced built-in blueprint helpers that are only meaningful **inside** `network!`

The public API should not require:
- suffix tags like `height[spatial]`
- a fixed global set of semantic axis roles
- multiple public construction styles

## Canonical Axis Model

The current best model is:

- an axis is identified by a label in the DSL
- transforms may preserve, collapse, split, rename, or merge labels
- shape checks are phrased in terms of label existence and compatibility

Examples:

- `conv` updates one label and rewrites extents over other labels
- `pool` reduces extents over chosen labels
- `flatten` collapses one or more labels into a new label
- `concat(axis)` concatenates along an existing label
- `sum[...]` requires exact label/shape agreement across branches

This means the library does not need to decide once and for all what "spatial" means.
It only needs to check that a transform is applied coherently to the labels the user selected.

## Candidate DSL: Explicit Form

This is the most honest and most general candidate surface:

```rust
let arch = network! {
    input(c: 3, y: 32, x: 32)
    -> conv(32, on: c, over: [y, x], kernel: 3, pad: 1)
    -> relu
    -> conv(64, on: c, over: [y, x], kernel: 3, stride: 2, pad: 1)
    -> pool(max, over: [y, x], kernel: 2, stride: 2)
    -> flatten(into: f)
    -> dense(10)
};
```

Semantics:
- `on: c` selects the label whose extent becomes the output width/channel count
- `over: [y, x]` selects the labels the kernel/reduction runs over
- labels persist unless explicitly collapsed or renamed

This candidate is likely feasible in Rust because:
- `network!` already parses custom syntax
- labels are just macro tokens
- named arguments and bracketed axis lists are straightforward to parse

## Candidate DSL: Scoped Defaults

This is currently the strongest ergonomic candidate:

```rust
let arch = network! {
    input(c: 3, y: 32, x: 32)
    defaults {
        conv(on: c, over: [y, x]);
        pool(over: [y, x]);
        flatten(into: f);
    }
    -> conv(32, kernel: 3, pad: 1)
    -> relu
    -> conv(64, kernel: 3, stride: 2, pad: 1)
    -> pool(max, kernel: 2, stride: 2)
    -> flatten
    -> dense(10)
};
```

Why this is strong:
- still one public syntax family
- removes repetitive axis operands
- avoids a public semantic ontology
- lets users stay explicit when needed

This should become a first-class design target.

## Candidate DSL: Reusable Blueprint Chunks

Reusable chunks should remain plain Rust bindings around `network!`.

The crucial idea is that a chunk may contain free axis symbols:

```rust
let stem = network! {
    defaults {
        conv(on: c, over: [y, x]);
        pool(over: [y, x]);
    }
    -> conv(32, kernel: 3, pad: 1)
    -> relu
    -> pool(max, kernel: 2, stride: 2)
};

let arch = network! {
    input(c: 3, y: 32, x: 32)
    -> stem
    -> stem
    -> flatten(into: f)
    -> dense(10)
};
```

This is plausible because the chunk is not ordinary Rust code using `c`, `y`, and `x`;
it is still parsed entirely by `network!`.

The insertion site provides the labels that make those symbols meaningful.

This is plausible, but it has a real ergonomics risk:
- chunks implicitly depend on ambient label names
- users may accidentally treat `c`, `y`, and `x` as globally blessed names
- reuse across differently-labeled architectures is awkward

Because of that, free-symbol chunks should no longer be treated as the leading candidate.

## Candidate DSL: Parameterized Blueprint Chunks

This is now the strongest reusable-chunk candidate.

Example:

```rust
let stem = network! {
    params(c, y, x)
    defaults {
        conv(on: c, over: [y, x]);
        pool(over: [y, x]);
    }
    -> conv(32, kernel: 3, pad: 1)
    -> relu
    -> pool(max, kernel: 2, stride: 2)
};

let arch = network! {
    input(rgb: 3, row: 32, col: 32)
    -> stem(c: rgb, y: row, x: col)
    -> flatten(into: f)
    -> dense(10)
};
```

Why this is stronger than free-symbol chunks:
- still one syntax family
- still uses plain Rust `let` for naming
- no invalid ordinary Rust outside `network!`
- reusable chunks become label-agnostic
- users can bind architecture-local names to chunk-local parameters explicitly

Why this is likely feasible:
- `network!` can parse a header such as `params(c, y, x)`
- `network!` can parse `stem(c: rgb, y: row, x: col)` as blueprint application syntax
- the macro can enforce arity and symbol binding at expansion time

This should become the leading reusable-chunk design target.

## Candidate DSL: Blueprint Application With Defaults

Parameterized chunks become even stronger when paired with defaults:

```rust
let residual_block = network! {
    params(c, y, x)
    defaults {
        conv(on: c, over: [y, x]);
    }
    -> residual(
        network! {
            conv(64, kernel: 3, pad: 1)
            -> relu
            -> conv(64, kernel: 3, pad: 1)
        }
    )
};

let arch = network! {
    input(ch: 64, h: 56, w: 56)
    -> residual_block(c: ch, y: h, x: w)
    -> residual_block(c: ch, y: h, x: w)
    -> flatten(into: f)
    -> dense(1000)
};
```

This preserves:
- explicit architectural intent
- local axis binding
- reuse without global axis names
- a single public language

## Candidate DSL: Built-In Helper Namespaces

The user is right to be strict about namespacing.

If built-in reusable helpers exist, they should not likely live in a flat `vision::*` namespace.

Current naming direction:
- `vision::common::*` for reusable generic blocks
- `vision::resnet::*` for ResNet-flavored helpers
- `vision::inception::*` for Inception-style helpers
- similarly later for `audio::*`, `sequence::*`, etc.

The key feasibility rule is:

- helper calls that mention DSL labels must appear **inside** `network!`

Example candidate:

```rust
let arch = network! {
    input(c: 3, y: 224, x: 224)
    -> vision::common::stem(on: c, over: [y, x], widths: [32, 64])
    -> vision::common::residual_block(on: c, over: [y, x], width: 64)
    -> pool(avg, over: [y, x])
    -> flatten(into: f)
    -> dense(1000)
};
```

This is feasible if and only if the call is treated as DSL syntax by `network!`, not as a normal Rust call expression.

The stronger version of that rule is:

- user-defined reusable chunks should prefer parameterized `network!` values
- built-in helpers may use the same application model internally
- built-in helpers should not introduce a different conceptual path from user-defined chunks

That means built-ins should ideally feel like predeclared parameterized blueprint templates, not like a different subsystem.

Possible direction:

```rust
let arch = network! {
    input(c: 3, y: 224, x: 224)
    -> vision::common::stem(c, y, x; widths: [32, 64])
    -> vision::common::residual_block(c, y, x; width: 64)
    -> pool(avg, over: [y, x])
    -> flatten(into: f)
    -> dense(1000)
};
```

This syntax is intentionally macro-only. The semicolon separates blueprint-axis arguments
from ordinary helper configuration.

It is not yet chosen, but it is the most promising built-in-helper shape so far.

## Parameter Declaration and Application

This now needs stricter choices.

### Declaration syntax candidates

#### Candidate A: closure-like pipes

```rust
network! { |c, y, x| ... }
```

Pros:
- short
- visually suggests abstraction

Cons:
- looks too much like ordinary Rust closure syntax
- suggests capture/value semantics that do not exist here
- makes the DSL feel less like one coherent language and more like macro trickery

Current judgment:
- reject as the leading idiom

#### Candidate B: explicit `params(...)` header

```rust
network! {
    params(c, y, x)
    ...
}
```

Pros:
- self-describing
- clearly DSL-specific
- aligns with other header-like constructs such as `input(...)` and `defaults { ... }`
- easier to document as formal blueprint parameters, not runtime values

Cons:
- slightly more verbose

Current judgment:
- strongest declaration form

### Application syntax candidates

#### Candidate A: positional application

```rust
-> stem(rgb, row, col)
```

Pros:
- short

Cons:
- order-sensitive
- weak readability once there are more than two or three labels
- mismatches the goal of strict, explicit idioms

Current judgment:
- maybe acceptable as shorthand
- not the best primary documented form

#### Candidate B: named application

```rust
-> stem(c: rgb, y: row, x: col)
```

Pros:
- explicit
- robust to reordering
- much clearer for helpers with more than two parameters

Cons:
- more verbose

Current judgment:
- strongest primary documented form

#### Candidate C: shorthand application when names align

```rust
-> stem(c, y, x)
```

This should be treated as possible sugar for:

```rust
-> stem(c: c, y: y, x: x)
```

Current judgment:
- good optional shorthand
- only when formal and actual labels intentionally match

### Current recommendation

The strongest combination right now is:

- declare reusable chunks with `params(...)`
- document named application as the main form
- optionally support shorthand positional-or-elided application only as sugar

Examples:

```rust
let stem = network! {
    params(c, y, x)
    defaults {
        conv(on: c, over: [y, x]);
    }
    -> conv(32, kernel: 3, pad: 1)
    -> relu
};

let cifar = network! {
    input(c: 3, y: 32, x: 32)
    -> stem(c, y, x)
    -> flatten(into: f)
    -> dense(10)
};

let microscope = network! {
    input(channels: 1, row: 128, col: 128)
    -> stem(c: channels, y: row, x: col)
    -> flatten(into: feat)
    -> dense(2)
};
```

## Transform Semantics

These should be treated as strict rules.

### `conv`

Candidate syntax:

```rust
conv(OUT, on: axis, over: [axes...], kernel: K, stride: S, pad: P)
```

Rules:
- `on:` selects exactly one existing label
- `over:` selects one or more existing labels
- `on:` must not also appear in `over:`
- `kernel`, `stride`, `pad`, and `dilation` arity must match the length of `over:` after scalar expansion
- output preserves the same label set unless explicit renaming is introduced later
- extent of `on:` becomes `OUT`
- extents of labels in `over:` are updated by convolution geometry

### `pool`

Candidate syntax:

```rust
pool(max, over: [axes...], kernel: K, stride: S)
```

Rules:
- `over:` selects one or more existing labels
- pooled labels remain present unless a later `reduce_*` form removes them
- extents of `over:` labels are updated by pooling geometry

### `flatten`

Candidate syntax:

```rust
flatten(into: f)
flatten(over: [c, y, x], into: f)
```

Rules:
- `flatten(into: f)` means collapse all current labels into `f`
- the explicit `over:` form is useful when partial flattening is introduced later
- collapsed labels disappear and are replaced by `into:`

### `concat`

Candidate syntax:

```rust
concat(c)[left, right, branch3]
```

Rules:
- `c` must be present in every branch output
- all non-`c` labels must match exactly
- resulting extent of `c` is the sum of branch extents along `c`

### `sum`

Candidate syntax:

```rust
sum[left, right, branch3]
```

Rules:
- every branch must expose exactly the same labels in the same order
- every label extent must match exactly

### `residual`

Candidate syntax:

```rust
residual(block)
```

Rules:
- `block` must preserve the full labeled shape
- result is elementwise add with the original input

### `repeat`

Candidate syntax:

```rust
repeat(3, block)
```

Rules for v1:
- `block` must preserve the labeled shape
- repetition uses fresh parameters unless wrapped in explicit sharing

## Pressure From Other Fields

The current DSL is much stronger than the earlier `conv2d`-style direction, but it is still
too CNN-biased if it stops here.

If `tml` is going to matter across fields, the language must survive pressure from:
- transformers and sequence models
- U-Nets and long-skip encoder-decoder models
- multimodal fusion
- scientific/operator models
- graph/set architectures
- architectures with multiple named inputs and outputs

The main lesson is:

> `network!` should not be designed as "a nicer CNN builder".

It should be designed as a language of labeled transforms that happens to express CNNs cleanly.

### 1. `dense` is too narrow as a general primitive

Current `dense(10)` works well after `flatten`, but it is too tied to the "single flat feature axis"
mental model.

Sequence and transformer-style models need a more general projection primitive:

```rust
linear(on: f, out: 3072)
```

meaning:
- preserve every label except `f`
- replace the extent of `f` with `3072`

Example:

```rust
let mlp = network! {
    params(tok, f)
    -> linear(on: f, out: 3072)
    -> gelu
    -> linear(on: f, out: 768)
};
```

This suggests a stronger public story:

- `linear(on: axis, out: N)` is the general primitive
- `dense(N)` becomes sugar for the common flat/single-feature-axis case

This is likely the right move if the library wants to feel general rather than conv-centric.

### 2. `on:` / `over:` should generalize beyond convolution

The strongest part of the current DSL is the `on:` / `over:` split.

That vocabulary should likely become the general language of transforms:

- `conv(out, on: c, over: [y, x], ...)`
- `pool(max, over: [y, x], ...)`
- `attend(on: f, over: [tok], heads: 12)`
- `scan(on: f, over: [tok], state: 16)`
- `reduce(mean, over: [tok])`

This is much better than teaching separate mental models for every domain.

### 3. Attention pressure

If the DSL cannot express transformer-like blocks naturally, it is not yet general enough.

Candidate direction:

```rust
let block = network! {
    params(tok, f)
    -> residual(
        network! {
            norm(rms, over: [f])
            -> attend(on: f, over: [tok], heads: 12)
        }
    )
    -> residual(
        network! {
            norm(rms, over: [f])
            -> linear(on: f, out: 3072)
            -> gelu
            -> linear(on: f, out: 768)
        }
    )
};
```

This is important because it pressure-tests:
- per-axis projection
- per-axis normalization
- residual blocks beyond CNNs

### 4. Structural transforms need to become first-class

CNNs are not the only architectures that reshape local structure.

The DSL likely needs structural transforms such as:

- `patchify(over: [y, x], into: tok, patch: 16)`
- `unpatchify(from: tok, over: [y, x], patch: 16)`
- `upsample(over: [y, x], scale: 2)`
- `downsample(over: [y, x], factor: 2)`
- `permute([tok, f])`

Example ViT-like sketch:

```rust
let arch = network! {
    input(c: 3, y: 224, x: 224)
    -> patchify(over: [y, x], into: tok, patch: 16)
    -> linear(on: c, out: 768)
    -> repeat(12, network! {
        params(tok, f)
        -> residual(network! {
            norm(rms, over: [f])
            -> attend(on: f, over: [tok], heads: 12)
        })
        -> residual(network! {
            norm(rms, over: [f])
            -> linear(on: f, out: 3072)
            -> gelu
            -> linear(on: f, out: 768)
        })
    }(tok: tok, f: c))
    -> reduce(mean, over: [tok])
    -> dense(1000)
};
```

Even if this exact syntax changes, the capability pressure is real.

### 5. Long-skip architectures need activation references

The current language can express local branches and residuals, but that is not enough for U-Net,
FPN, or other architectures with long-lived skip activations.

That means the DSL likely needs an explicit activation-reference mechanism.

Candidate direction:

```rust
network! {
    input(c: 3, y: 256, x: 256)
    -> save(enc1)
    -> downsample(over: [y, x], factor: 2)
    -> save(enc2)
    -> downsample(over: [y, x], factor: 2)
    -> upsample(over: [y, x], scale: 2)
    -> concat(c)[current, from(enc2)]
    -> upsample(over: [y, x], scale: 2)
    -> concat(c)[current, from(enc1)]
    -> dense(2)
}
```

This is not yet the chosen syntax.

But the design pressure is unavoidable:
- plain sequential chunk composition is not enough for long-range DAGs
- the language probably needs named activation references eventually

### 6. Multi-input architectures are a real requirement

Many important architectures need more than one input:
- encoder-decoder with conditioning
- multimodal fusion
- cross attention
- graph models with node and edge inputs

So the language should probably reserve a future symmetric counterpart to `heads`:

```rust
network! {
    inputs {
        image(c: 3, y: 224, x: 224),
        text(tok: 77, f: 768),
    }
    -> ...
}
```

Even if this is not implemented yet, the single-input assumption should not harden into the core.

### 7. Graph and irregular domains should not be forced into fake convolution language

Some domains are not well-modeled by regular windowed axes.

Graph models, set models, and operator-learning models may need domain-specific transforms such as:

- `message_pass(nodes: n, features: f, edges: e, out: 128)`
- `aggregate(sum, over: [nodes])`
- `cross_attend(on: f, query: [target], context: [source], heads: 8)`

The lesson is not "make everything graph-specific".

It is:
- keep the core language generic and label-driven
- allow field-specific namespaces to define stronger reusable abstractions

### 8. Field namespaces should reflect real architectural families

The namespace idea should broaden accordingly:

- `vision::common::*`
- `vision::resnet::*`
- `vision::unet::*`
- `sequence::common::*`
- `sequence::transformer::*`
- `audio::common::*`
- `graph::common::*`
- `multimodal::fusion::*`

The goal is:
- one language
- many domain libraries built on top of it

not:
- one monolithic generic core that forces every field into the same primitive vocabulary

## Example Users and Prototype Matrix

The DSL should be tested against concrete user archetypes before implementation.

### 1. Small image-classification user

```rust
let arch = network! {
    input(c: 3, y: 32, x: 32)
    defaults {
        conv(on: c, over: [y, x]);
        pool(over: [y, x]);
        flatten(into: f);
    }
    -> conv(32, kernel: 3, pad: 1)
    -> relu
    -> pool(max, kernel: 2, stride: 2)
    -> flatten
    -> dense(10)
};
```

### 2. Audio spectrogram user

```rust
let arch = network! {
    input(c: 1, t: 256, hz: 80)
    -> conv(32, on: c, over: [t, hz], kernel: [5, 3], pad: [2, 1])
    -> relu
    -> pool(max, over: [t], kernel: 2, stride: 2)
    -> flatten(into: f)
    -> dense(4)
};
```

### 3. Sequence convolution user

```rust
let arch = network! {
    input(c: 64, tok: 512)
    -> conv(128, on: c, over: [tok], kernel: 7, pad: 3)
    -> relu
    -> pool(avg, over: [tok], kernel: 2, stride: 2)
    -> flatten(into: f)
    -> dense(2)
};
```

### 4. 3D volume user

```rust
let arch = network! {
    input(c: 1, z: 64, y: 64, x: 64)
    -> conv(8, on: c, over: [z, y, x], kernel: [3, 3, 5], pad: [1, 1, 2])
    -> relu
    -> flatten(into: f)
    -> dense(3)
};
```

### 5. Custom scientific user with domain-specific labels

```rust
let arch = network! {
    input(sensor: 16, time: 1024, freq: 80)
    -> conv(64, on: sensor, over: [time, freq], kernel: [5, 3], pad: [2, 1])
    -> pool(max, over: [time], kernel: 2, stride: 2)
    -> flatten(into: feat)
    -> dense(32)
};
```

### 6. Multi-branch vision user

```rust
let arch = network! {
    input(c: 3, y: 224, x: 224)
    -> concat(c)[
        conv(32, on: c, over: [y, x], kernel: 1),
        conv(32, on: c, over: [y, x], kernel: 3, pad: 1),
        conv(32, on: c, over: [y, x], kernel: 5, pad: 2),
    ]
    -> flatten(into: f)
    -> dense(1000)
};
```

### 7. Multi-head user

```rust
let arch = network! {
    input(c: 3, y: 224, x: 224)
    -> conv(32, on: c, over: [y, x], kernel: 3, pad: 1)
    -> heads {
        logits: flatten(into: f) -> dense(1000),
        embedding: flatten(into: e) -> dense(128),
    }
};
```

### 8. Reusable user-defined conv chunk across different local labels

```rust
let stem = network! {
    params(c, y, x)
    defaults {
        conv(on: c, over: [y, x]);
    }
    -> conv(32, kernel: 3, pad: 1)
    -> relu
};

let cifar = network! {
    input(c: 3, y: 32, x: 32)
    -> stem(c, y, x)
    -> flatten(into: f)
    -> dense(10)
};

let microscope = network! {
    input(channels: 1, row: 128, col: 128)
    -> stem(c: channels, y: row, x: col)
    -> flatten(into: feat)
    -> dense(2)
};
```

### 9. Built-in helper user

```rust
let arch = network! {
    input(c: 3, y: 224, x: 224)
    -> vision::common::stem(c, y, x; widths: [32, 64])
    -> vision::common::residual_block(c, y, x; width: 64)
    -> vision::common::residual_block(c, y, x; width: 64)
    -> pool(avg, over: [y, x])
    -> flatten(into: f)
    -> dense(1000)
};
```

### 10. Transformer-style user

```rust
let block = network! {
    params(tok, f)
    -> residual(
        network! {
            norm(rms, over: [f])
            -> attend(on: f, over: [tok], heads: 12)
        }
    )
    -> residual(
        network! {
            norm(rms, over: [f])
            -> linear(on: f, out: 3072)
            -> gelu
            -> linear(on: f, out: 768)
        }
    )
};

let arch = network! {
    input(tok: 512, f: 768)
    -> block(tok: tok, f: f)
    -> block(tok: tok, f: f)
    -> reduce(mean, over: [tok])
    -> dense(2)
};
```

### 11. U-Net-like user

```rust
let arch = network! {
    input(c: 3, y: 256, x: 256)
    -> save(enc1)
    -> downsample(over: [y, x], factor: 2)
    -> save(enc2)
    -> downsample(over: [y, x], factor: 2)
    -> upsample(over: [y, x], scale: 2)
    -> concat(c)[current, from(enc2)]
    -> upsample(over: [y, x], scale: 2)
    -> concat(c)[current, from(enc1)]
    -> dense(2)
};
```

### 12. Multimodal user

```rust
let arch = network! {
    inputs {
        image(c: 3, y: 224, x: 224),
        text(tok: 77, f: 768),
    }
    -> ...
};
```

### 13. Graph-model user

```rust
let arch = network! {
    input(node: 1024, feat: 128, edge: 4096)
    -> graph::common::message_pass(node, feat, edge; out: 256)
    -> graph::common::message_pass(node, feat, edge; out: 256)
    -> reduce(mean, over: [node])
    -> dense(10)
};
```

These examples are the design test suite. If the DSL cannot support them cleanly,
the DSL is not done.

## Prototype Comparison

### Option A: Free-symbol chunks

Example:

```rust
let stem = network! {
    defaults {
        conv(on: c, over: [y, x]);
    }
    -> conv(32, kernel: 3, pad: 1)
    -> relu
};
```

Pros:
- very terse
- no new application syntax

Cons:
- ambient-name coupling
- weak reuse across differently-labeled architectures
- risks creating accidental house names that feel mandatory

Current judgment:
- acceptable as an implementation stepping stone
- not the best long-term public idiom

### Option B: Parameterized chunks

Example:

```rust
let stem = network! {
    params(c, y, x)
    defaults {
        conv(on: c, over: [y, x]);
    }
    -> conv(32, kernel: 3, pad: 1)
    -> relu
};
```

Pros:
- explicit binding
- reusable across arbitrary local names
- still one syntax family
- avoids invalid non-Rust helper syntax outside `network!`

Cons:
- adds application syntax and parser complexity

Current judgment:
- strongest overall candidate

### Option C: Built-in helper intrinsics only

Example:

```rust
network! {
    input(c: 3, y: 224, x: 224)
    -> vision::common::stem(c, y, x; widths: [32, 64])
}
```

Pros:
- very ergonomic for common patterns

Cons:
- not enough on its own
- risks creating "special library magic" that users cannot emulate

Current judgment:
- good only if it is built on the same parameterized blueprint model as user-defined chunks

## Strict Idioms and Best Practices

These should be treated as style rules, not vague suggestions.

### 1. Keep `network!` as the single public architecture surface

No second equally-official builder story should compete with it.

### 2. Prefer explicit labels over semantic suffixes

Prefer:

```rust
input(c: 3, y: 32, x: 32)
```

over:

```rust
input(rgb: 3[channel], y: 32[spatial], x: 32[spatial])
```

### 3. Use explicit `on:` and `over:` when writing transforms without defaults

This keeps transform meaning local and avoids hidden ontology.

### 4. Use `defaults { ... }` when three or more transforms share the same axis mapping

This should become the idiomatic way to remove repetition.

### 5. Prefer `params(...)` for reusable chunks

Prefer:

```rust
let stem = network! {
    params(c, y, x)
    ...
};
```

Avoid using closure-like pipe syntax as the leading documented pattern.

### 6. Prefer named chunk application when labels differ

Prefer:

```rust
-> stem(c: channels, y: row, x: col)
```

Allow:

```rust
-> stem(c, y, x)
```

only as shorthand when names intentionally align.

### 7. Keep frequently-transformed axes short

Preferred conventions in examples and docs:
- `c` for channel-like or feature-carrier axis in convolutional pipelines
- `y`, `x` for 2D grid axes
- `z`, `y`, `x` for 3D grid axes
- `t` for time-like axis
- `f` or `feat` for flattened feature output

These are not mandatory globally, but they should likely be the house style in docs and built-ins.

### 8. Put reusable built-in helpers under domain namespaces

Prefer:
- `vision::common::stem(...)`
- `vision::common::residual_block(...)`
- `vision::resnet::basic_block(...)`
- `vision::resnet::bottleneck(...)`

Avoid:
- flat `vision::stem(...)` if the namespace is likely to grow crowded
- ad hoc root-level helper names that are hard to organize later

### 9. Built-in helpers should mirror user-defined chunk application

Prefer forms like:

```rust
-> vision::common::stem(c, y, x; widths: [32, 64])
```

or:

```rust
-> vision::common::stem(c: channels, y: row, x: col; widths: [32, 64])
```

The helper story should not feel like a separate subsystem.

### 10. Do not expose ontology-like axis names as the main story

The DSL should not teach users that they must think in terms of an approved set of semantic axis roles.

### 11. Prefer general primitives over domain-locked names when they scale

Prefer:
- `linear(on: f, out: 768)`
- `attend(on: f, over: [tok], heads: 12)`
- `reduce(mean, over: [tok])`

over forcing every field into:
- `dense(...)`
- `conv2d(...)`
- `multihead_attention(...)`

when a cleaner labeled-axis primitive would scale better.

### 12. Domain namespaces should package patterns, not replace the language

The language should stay primary.

Domain helpers should be:
- reusable templates
- stronger idioms
- family-specific convenience

not a second architecture subsystem.

## What Still Needs Proof

The following still need prototyping before implementation decisions are locked:

1. Whether free-symbol chunks should survive at all, or be kept only as an internal stepping stone.
2. Whether named application should be the only documented form, with shorthand intentionally undocumented.
3. Whether `defaults { ... }` should be block-scoped or whole-pipeline scoped.
4. Whether built-in helper invocations inside `network!` are worth the parser complexity.
5. Whether `linear(on: axis, out: N)` should become the general primitive, with `dense(N)` demoted to sugar.
6. Whether labels should preserve identity across `conv` and `linear`, or whether explicit renaming becomes necessary sooner.
7. What the first activation-reference syntax should be for long-skip DAGs.
8. What the first multi-input syntax should be.
9. Whether `flatten(into: f)` should be mandatory or whether plain `flatten` should default to `f`/`features`.

## Current Lean

The strongest current design stance is:

- axis labels are open-ended DSL symbols
- transforms explicitly say which labels they act on
- scoped defaults remove repetition
- parameterized reusable chunks are the strongest reuse model
- free-symbol chunks are acceptable only as a possible shorthand, not as the leading idiom
- `params(...)` is the strongest chunk declaration form
- named chunk application is the strongest primary application form
- the DSL likely needs a general `linear(on: ..., out: ...)` primitive
- long-skip activation references are likely unavoidable for serious DAG architectures
- multi-input syntax should be treated as a real requirement, not a niche extension
- built-in helper namespaces are acceptable, but only inside `network!`
- built-in helpers should conceptually be predeclared parameterized blueprint templates
- no public semantic axis taxonomy unless a real implementation pressure makes it unavoidable

This is the version of the DSL that should be pressure-tested next, not the older fixed-role axis model.
