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

## What Still Needs Proof

The following still need prototyping before implementation decisions are locked:

1. Whether free-symbol chunks should survive at all, or be kept only as an internal stepping stone.
2. Whether named application should be the only documented form, with shorthand intentionally undocumented.
3. Whether `defaults { ... }` should be block-scoped or whole-pipeline scoped.
4. Whether built-in helper invocations inside `network!` are worth the parser complexity.
5. Whether labels should preserve identity across `conv`, or whether explicit renaming becomes necessary sooner.
6. Whether `flatten(into: f)` should be mandatory or whether plain `flatten` should default to `f`/`features`.

## Current Lean

The strongest current design stance is:

- axis labels are open-ended DSL symbols
- transforms explicitly say which labels they act on
- scoped defaults remove repetition
- parameterized reusable chunks are the strongest reuse model
- free-symbol chunks are acceptable only as a possible shorthand, not as the leading idiom
- `params(...)` is the strongest chunk declaration form
- named chunk application is the strongest primary application form
- built-in helper namespaces are acceptable, but only inside `network!`
- built-in helpers should conceptually be predeclared parameterized blueprint templates
- no public semantic axis taxonomy unless a real implementation pressure makes it unavoidable

This is the version of the DSL that should be pressure-tested next, not the older fixed-role axis model.
