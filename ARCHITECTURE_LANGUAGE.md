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

## Macro Justification Tests

`network!` is only a good idea if it continues to pass all of these tests:

1. It must express architecture more clearly than plain Rust would.
2. It must buy real compile-time checking, not just syntax sugar.
3. It must stay a small architecture language, not a second general-purpose language.
4. It must lower into a real core, not become the runtime itself.
5. It must not force users into multiple equally-official API paths.

If the DSL ever stops passing those tests, it should be narrowed or replaced.

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

## Stronger Candidate: Interface-Oriented Chunk Parameters

Raw parameter lists like `params(c, y, x)` are workable, but they still leave too much
implicit:
- which params are single-axis slots vs axis-list slots
- what the chunk is really asking for at the call site
- how built-in helpers should mirror the same interface model

A stronger candidate is:

```rust
let stem = network! {
    params(ch: c, spatial: [y, x])
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
    -> stem(ch: rgb, spatial: [row, col])
    -> flatten(into: f)
    -> dense(10)
};
```

This separates:
- **interface slot names**: `ch`, `spatial`
- **local body labels**: `c`, `y`, `x`

And it encodes parameter kind directly in the syntax:
- `ch: c` is a single-axis parameter
- `spatial: [y, x]` is a fixed-arity axis-list parameter

Why this is stronger:
- chunk interfaces become self-documenting
- call sites become more readable
- arity checking becomes natural
- user-defined chunks and built-in helpers can share the same application model
- it avoids baking a global public axis ontology into the API

Current judgment:
- this is stronger than raw `params(c, y, x)`
- it should become the leading reusable-chunk interface model

But it is still not obviously strong enough.

The name `spatial` in examples above is **not** special.
It is just a user-chosen interface-slot name.

That means this:

```rust
params(spatial: [y, x])
```

currently guarantees only:
- that the slot is named `spatial`
- that it binds exactly two axes

It does **not** guarantee any deeper meaning.

That is a real weakness and needs adversarial pressure.

## Chunk Signatures and Parameter Safety

The parameter list in a reusable chunk is not just documentation.

It should be treated as a compile-time interface signature.

Example:

```rust
let block = network! {
    params(seq: [tok], feat: f)
    -> linear(on: f, out: 768)
};
```

The type-safety story should be:

- `params(seq: [tok], feat: f)` declares formal interface slots, not arbitrary free tokens
- each slot has a kind inferred from syntax:
  - `feat: f` is a single-axis slot
  - `seq: [tok]` is a one-axis-list slot
  - `spatial: [y, x]` would be a two-axis-list slot
- the chunk body may only reference:
  - local symbols introduced by declared parameters
  - labels introduced structurally inside the chunk
  - labels introduced by nested block scopes
- applying the chunk requires every formal interface slot to be bound exactly once
- extra or duplicate bindings are rejected
- named application is the primary form because order should not matter
- axis-list parameters preserve their declared arity
- the chunk’s output signature is derived symbolically in terms of local formal labels
- applying the chunk remaps that derived output signature onto actual labels

That means this should be rejected:

```rust
let block = network! {
    params(seq: [tok], feat: f)
    -> linear(on: g, out: 768)
};
```

because `g` is neither a formal parameter nor an introduced local label.

This should also be rejected:

```rust
let arch = network! {
    input(tokens: 512, feat: 768)
    -> block(seq: [tokens])
};
```

because `feat` is unbound.

This should also be rejected:

```rust
let arch = network! {
    input(c: 64, y: 56, x: 56)
    -> stem(ch: c, spatial: [y])
};
```

because `spatial` was declared as a two-axis slot and only one axis was supplied.

This is the point where `params(...)` becomes real type-safety rather than macro decoration.

## Adversarial Stress Tests For Chunk Interfaces

The right way to judge chunk-parameter syntax is to try to break it.

### 1. Name theater

This is legal-looking:

```rust
let block = network! {
    params(spatial: [tok, feat])
    -> linear(on: feat, out: 768)
};
```

The slot name `spatial` means nothing here.

This is not a bug by itself, but it proves an important point:

- slot names must not be mistaken for built-in semantics
- the DSL should not imply that names like `spatial` or `channel` carry magical meaning

### 2. Arity spoofing

This also looks fine:

```rust
let stem = network! {
    params(ch: c, plane: [y, x])
    defaults {
        conv(on: c, over: [y, x]);
    }
    -> conv(32, kernel: 3, pad: 1)
};

let spectrogram = network! {
    input(sensor: 16, time: 1024, freq: 80)
    -> stem(ch: sensor, plane: [time, freq])
};
```

This might actually be perfectly valid.
The point is:

- `[y, x]` only constrains arity and order
- it does not encode any stronger structural contract

That means interface-slot lists are more general than the earlier ontology-based model,
but also weaker than they may first appear.

### 3. Overlap and aliasing

This should almost certainly be rejected:

```rust
let arch = network! {
    input(c: 64, y: 56, x: 56)
    -> stem(ch: c, plane: [c, x])
};
```

because the same actual axis `c` is being bound into two distinct interface roles.

More pathological:

```rust
let block = network! {
    params(seq: [tok], feat: f)
    -> attend(on: f, over: [tok], heads: 12)
};

let arch = network! {
    input(tok: 512)
    -> block(seq: [tok], feat: tok)
};
```

This collapses two intended roles onto the same axis.

Current judgment:
- distinct interface slots should bind to distinct actual axes by default
- if aliasing is ever allowed, it should be explicit, not accidental

### 4. Order fragility

This is subtle:

```rust
-> stem(ch: c, plane: [y, x])
-> stem(ch: c, plane: [x, y])
```

Both satisfy the same arity.
But many transforms interpret axis-list order as semantically significant for:
- kernel shape
- stride shape
- dilation shape
- later partial selection

So list-based slots are still more fragile than they look.

### 5. Namespace collision pressure

The language is already accumulating multiple naming spaces:
- axis labels
- chunk interface slot names
- saved activation names
- head names
- helper config keys

If those are not kept distinct, the DSL will rot quickly.

Current judgment:
- chunk slot names, axis labels, and saved-source names should live in separate namespaces

## Stronger Candidate: Structured Slot Parameters

The adversarial cases above suggest that raw axis-list slots may still be too weak.

A stronger direction is to let chunk interfaces declare **structured slots**.

Example:

```rust
let stem = network! {
    params(ch, plane { row, col })
    defaults {
        conv(on: ch, over: plane);
        pool(over: plane);
    }
    -> conv(32, kernel: 3, pad: 1)
    -> relu
};

let arch = network! {
    input(rgb: 3, row: 32, col: 32)
    -> stem(ch: rgb, plane: { row: row, col: col })
    -> flatten(into: f)
    -> dense(10)
};
```

This is stronger because:
- `plane` is a first-class interface slot, not just a nickname for `[y, x]`
- member names like `row` and `col` make order explicit
- application can check exact member coverage, not just list length
- transforms can consume the whole structured slot:
  - `over: plane`
- or eventually refer to members:
  - `plane.row`
  - `plane.col`

This also keeps semantics open-ended:
- `plane` is not a magical ontology word
- users could name the slot `domain`, `grid`, `window`, `spectrogram`, etc.

Current judgment:
- structured slots are now a stronger candidate than plain axis-list slots
- the current `params(ch: c, spatial: [y, x])` form should be treated as a stepping stone, not a settled answer

## Dimension Safety Is A Separate Problem

Axis safety and dimension safety are not the same thing.

The current DSL work is getting better at:
- label existence
- label binding
- slot structure
- branch/source naming

But that is still weaker than true shape safety.

Real type-safety also needs to account for:
- rank
- per-axis extents
- symbolic extent equalities
- geometry validity
- branch compatibility
- source compatibility over time

This is the actual hard part.

### What the current design still does not prove strongly enough

Even a well-structured chunk like:

```rust
let stem = network! {
    params(ch, plane { row, col })
    defaults {
        conv(on: ch, over: plane);
    }
    -> conv(32, kernel: 3, pad: 1)
    -> relu
};
```

still only says:
- there is one `ch` axis
- there is one two-member `plane` slot
- the body uses those symbols consistently

It does **not yet** fully say:
- what symbolic extents those members carry
- how output extents are derived symbolically
- what equalities later branches must satisfy
- how saved activations constrain later references

So yes: the current design is still too weak if judged as a full typed-shape story.

That weakness should be treated as a design bug, not just future polish.

## Symbolic Extent Model

The stronger direction is:

- every input axis introduces a symbolic extent variable
- every chunk interface member introduces symbolic extent variables
- every transform rewrites a symbolic labeled-shape expression
- compatibility checks are stated over those symbolic expressions

Example mental model:

```rust
input(c: 3, y: 32, x: 32)
```

introduces something like:
- `c : 3`
- `y : 32`
- `x : 32`

More generally, a chunk interface like:

```rust
params(ch, plane { row, col })
```

should implicitly mean:
- `ch` has some symbolic extent `C`
- `plane.row` has some symbolic extent `H`
- `plane.col` has some symbolic extent `W`

Then:

```rust
conv(64, on: ch, over: plane, kernel: 3, pad: 1)
```

rewrites the symbolic shape from:
- `(ch: C, plane.row: H, plane.col: W)`

to:
- `(ch: 64, plane.row: conv_out(H, 3, 1, 1), plane.col: conv_out(W, 3, 1, 1))`

That is the level where residual/concat/save/from can become truly type-safe.

## Adversarial Stress Tests For Dimension Safety

Trying to break the DSL on dimensions exposes the next set of required contracts.

### 1. Residual mismatch after hidden geometry change

This should be rejected:

```rust
let block = network! {
    params(ch, plane { row, col })
    -> residual {
        conv(64, on: ch, over: plane, kernel: 3, stride: 2, pad: 1)
    }
};
```

because the body rewrites:
- `ch`
- `plane.row`
- `plane.col`

and therefore cannot satisfy residual shape preservation.

This requires symbolic extent rewriting, not just label tracking.

### 2. Concat with branch-local extent drift

This should be rejected:

```rust
network! {
    input(c: 32, y: 56, x: 56)
    -> concat(c) {
        conv(32, on: c, over: [y, x], kernel: 3, pad: 1),
        conv(32, on: c, over: [y, x], kernel: 3, stride: 2, pad: 1),
    }
}
```

because the non-concatenated extents of `y` and `x` no longer match.

Again: label agreement is not enough.

### 3. Saved-source misuse after later shape drift

This should be rejected:

```rust
network! {
    input(c: 32, y: 56, x: 56)
    -> save(enc)
    -> downsample(over: [y, x], factor: 2)
    -> concat(c) {
        current,
        from(enc),
    }
}
```

unless the current branch has been restored to the saved shape.

This means `save(name)` must record full symbolic labeled shape, not just a label set.

### 4. Multi-input fusion with false projection agreement

This should be rejected:

```rust
network! {
    inputs {
        left(tok: 128, feat: 768),
        right(tok: 64, feat: 768),
    }
    -> concat(tok) {
        from(left),
        from(right),
    }
}
```

if the intended rule is that all non-concatenated axes agree and concat only changes `tok`.

This is another place where named-source safety depends on symbolic extent checks.

### 5. Structured-slot partial equivalence

This should probably be rejected:

```rust
let block = network! {
    params(ch, plane { row, col })
    -> permute([ch, plane.col, plane.row])
    -> conv(64, on: ch, over: plane, kernel: [3, 5], pad: [1, 2])
};
```

unless `over: plane` is defined to preserve the declared member order of `plane` even after permutation.

So structured slots need a rule:
- are they ordered interfaces?
- or merely named member sets?

Current judgment:
- structured slots should preserve a canonical declared member order
- named application removes binding-order ambiguity
- transform semantics over structured slots should use that canonical order unless they explicitly project members

## Type-Safety Layers

The DSL now looks like it needs multiple distinct layers of safety:

1. **Symbol safety**
   - unknown labels/slots/sources are rejected

2. **Interface safety**
   - chunk parameters are fully bound
   - member coverage is exact
   - slot kinds and arities match

3. **Alias safety**
   - overlapping actual-axis bindings are rejected by default

4. **Shape-expression safety**
   - transforms rewrite symbolic labeled shapes correctly
   - residual/sum/concat compare those symbolic shapes

5. **Geometry safety**
   - kernels/strides/padding/dilation are valid for the derived extents

6. **Source safety**
   - `save(name)` / `from(name)` and named inputs preserve exact symbolic shapes

7. **Output safety**
   - `heads { ... }` yields a typed named-output signature

If `tml` wants to earn “typed machine learning,” it needs a credible story for all seven layers.

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
    -> vision::common::stem(ch: c, spatial: [y, x], widths: [32, 64])
    -> vision::common::residual_block(ch: c, spatial: [y, x], width: 64)
    -> pool(avg, over: [y, x])
    -> flatten(into: f)
    -> dense(1000)
};
```

This syntax is intentionally macro-only.

Current judgment:
- built-in helpers should probably expose the same interface-slot style as user chunks
- that keeps helper calls and user-defined blueprint application in the same conceptual model

## Inline Combinator Blocks

Nested `network!` invocations inside `network!` are a smell.

This:

```rust
-> residual(
    network! {
        norm(rms, over: [f])
        -> attend(on: f, over: [tok], heads: 12)
    }
)
```

is workable, but it is not clean enough as the main story.

A stronger direction is inline combinator blocks:

```rust
-> residual {
    norm(rms, over: [f])
    -> attend(on: f, over: [tok], heads: 12)
}
```

and:

```rust
-> concat(c) {
    conv(32, on: c, over: [y, x], kernel: 1),
    conv(32, on: c, over: [y, x], kernel: 3, pad: 1),
    conv(32, on: c, over: [y, x], kernel: 5, pad: 2),
}
```

This keeps one language and avoids nested macro noise.

Current judgment:
- inline combinator blocks are likely better than nested `network!` values for the common path
- nested reusable chunks still matter for named reuse and parameterization

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

#### Candidate B: explicit raw `params(...)` header

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

#### Candidate C: interface-oriented `params(...)` header

```rust
network! {
    params(ch: c, spatial: [y, x])
    ...
}
```

Pros:
- makes slot kinds explicit
- separates public chunk interface names from local internal labels
- makes call sites much clearer
- gives built-in helpers and user chunks the same application shape
- scales better beyond simple CNN examples

Cons:
- more syntax to parse
- slightly more verbose

Current judgment:
- strongest overall declaration form
- should likely replace raw `params(c, y, x)` as the primary idiom

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

#### Candidate B: named application for raw params

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

#### Candidate C: named application for interface-oriented params

```rust
-> stem(ch: rgb, spatial: [row, col])
```

Pros:
- exposes the chunk interface instead of its implementation-local symbols
- naturally supports fixed-arity axis-list binding
- aligns cleanly with built-in helper syntax
- is the clearest story for type-safe reuse

Cons:
- more verbose than positional application

Current judgment:
- strongest primary application form overall

#### Candidate D: shorthand application when names align

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

- declare reusable chunks with interface-oriented `params(...)`
- document interface-slot application as the main form
- optionally support shorthand positional-or-elided application only as sugar

Examples:

```rust
let stem = network! {
    params(ch: c, spatial: [y, x])
    defaults {
        conv(on: c, over: [y, x]);
    }
    -> conv(32, kernel: 3, pad: 1)
    -> relu
};

let cifar = network! {
    input(c: 3, y: 32, x: 32)
    -> stem(ch: c, spatial: [y, x])
    -> flatten(into: f)
    -> dense(10)
};

let microscope = network! {
    input(channels: 1, row: 128, col: 128)
    -> stem(ch: channels, spatial: [row, col])
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
    params(seq: [tok], feat: f)
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
    params(seq: [tok], feat: f)
    -> residual {
        norm(rms, over: [f])
        -> attend(on: f, over: [tok], heads: 12)
    }
    -> residual {
        norm(rms, over: [f])
        -> linear(on: f, out: 3072)
        -> gelu
        -> linear(on: f, out: 768)
    }
};
```

This is important because it pressure-tests:
- per-axis projection
- per-axis normalization
- residual blocks beyond CNNs
- type-safe reusable chunk interfaces beyond image-style naming

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
        params(seq: [tok], feat: f)
        -> residual {
            norm(rms, over: [f])
            -> attend(on: f, over: [tok], heads: 12)
        }
        -> residual {
            norm(rms, over: [f])
            -> linear(on: f, out: 3072)
            -> gelu
            -> linear(on: f, out: 768)
        }
    }(seq: [tok], feat: c))
    -> reduce(mean, over: [tok])
    -> dense(1000)
};
```

Even if this exact syntax changes, the capability pressure is real.

### 5. Long-skip architectures need activation references

The current language can express local branches and residuals, but that is not enough for U-Net,
FPN, or other architectures with long-lived skip activations.

That means the DSL likely needs an explicit activation-reference mechanism.

The strongest current idea is:
- `save(name)` stores the current activation under a label
- `from(name)` reintroduces a previously saved activation as a branch source

This is also the likely foundation for future multi-input handling.

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

The most interesting design connection here is:

- named inputs are just initial saved activations
- long skips are saved intermediate activations
- both may be handled by one graph-reference mechanism

That suggests the language should avoid solving these as two unrelated subsystems.

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
    params(ch: c, spatial: [y, x])
    defaults {
        conv(on: c, over: [y, x]);
    }
    -> conv(32, kernel: 3, pad: 1)
    -> relu
};

let cifar = network! {
    input(c: 3, y: 32, x: 32)
    -> stem(ch: c, spatial: [y, x])
    -> flatten(into: f)
    -> dense(10)
};

let microscope = network! {
    input(channels: 1, row: 128, col: 128)
    -> stem(ch: channels, spatial: [row, col])
    -> flatten(into: feat)
    -> dense(2)
};
```

### 9. Built-in helper user

```rust
let arch = network! {
    input(c: 3, y: 224, x: 224)
    -> vision::common::stem(ch: c, spatial: [y, x], widths: [32, 64])
    -> vision::common::residual_block(ch: c, spatial: [y, x], width: 64)
    -> vision::common::residual_block(ch: c, spatial: [y, x], width: 64)
    -> pool(avg, over: [y, x])
    -> flatten(into: f)
    -> dense(1000)
};
```

### 10. Transformer-style user

```rust
let block = network! {
    params(seq: [tok], feat: f)
    -> residual {
        norm(rms, over: [f])
        -> attend(on: f, over: [tok], heads: 12)
    }
    -> residual {
        norm(rms, over: [f])
        -> linear(on: f, out: 3072)
        -> gelu
        -> linear(on: f, out: 768)
    }
};

let arch = network! {
    input(tok: 512, f: 768)
    -> block(seq: [tok], feat: f)
    -> block(seq: [tok], feat: f)
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
    -> concat(f) {
        from(image)
        -> vision::common::stem(ch: c, spatial: [y, x], widths: [32, 64])
        -> reduce(mean, over: [y, x]),

        from(text)
        -> sequence::transformer::encoder(seq: [tok], feat: f, depth: 6, heads: 12)
        -> reduce(mean, over: [tok]),
    }
    -> linear(on: f, out: 2)
};
```

### 13. Graph-model user

```rust
let arch = network! {
    input(node: 1024, feat: 128, edge: 4096)
    -> graph::common::message_pass(nodes: node, feat: feat, edges: edge, out: 256)
    -> graph::common::message_pass(nodes: node, feat: feat, edges: edge, out: 256)
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

### Option B: Raw parameterized chunks

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
- workable, but no longer the strongest candidate

### Option C: Interface-oriented parameterized chunks

Example:

```rust
let stem = network! {
    params(ch: c, spatial: [y, x])
    defaults {
        conv(on: c, over: [y, x]);
    }
    -> conv(32, kernel: 3, pad: 1)
    -> relu
};
```

Pros:
- interface-slot kinds are explicit
- call sites are clearer
- fixed-arity axis-list binding is natural
- built-in helpers can mirror the same model directly

Cons:
- a bit more verbose
- needs slightly richer parser support

Current judgment:
- strongest overall candidate

### Option D: Built-in helper intrinsics only

Example:

```rust
network! {
    input(c: 3, y: 224, x: 224)
    -> vision::common::stem(ch: c, spatial: [y, x], widths: [32, 64])
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

### 5. Prefer explicit structured interfaces for reusable chunks

Prefer:

```rust
let stem = network! {
    params(ch, plane { row, col })
    ...
};
```

Avoid using closure-like pipe syntax, raw unnamed parameter lists, or ontology-heavy slot names as the leading documented pattern.

### 6. Prefer interface-slot application

Prefer:

```rust
-> stem(ch: channels, plane: { row: row, col: col })
```

Allow:

```rust
-> stem(ch: c, plane: { row: y, col: x })
```

as the normal explicit form. Raw positional shorthand should only exist, if at all, as undocumented sugar.

### 7. Keep frequently-transformed axes short

Preferred conventions in examples and docs:
- `c` for channel-like or feature-carrier axis in convolutional pipelines
- `y`, `x` for 2D grid axes
- `z`, `y`, `x` for 3D grid axes
- `t` for time-like axis
- `f` or `feat` for flattened feature output

These are not mandatory globally, but they should likely be the house style in docs and built-ins.

Preferred structured-slot names in docs should stay neutral and structural:
- `plane`
- `grid`
- `domain`
- `seq`

Avoid treating ontology-heavy names like `spatial` as though they were built-in concepts.

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
-> vision::common::stem(ch: c, plane: { row: y, col: x }, widths: [32, 64])
```

or:

```rust
-> vision::common::stem(ch: channels, plane: { row: row, col: col }, widths: [32, 64])
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

### 13. Reusable chunks must have explicit interfaces

`params(...)` is not optional decoration.

It is the chunk interface and should be documented that way.

### 14. Prefer inline combinator blocks for local graph structure

Prefer:

```rust
-> residual { ... }
-> concat(c) { ... }
```

over nested `network!` blocks for common inline cases.

## What Still Needs Proof

The following still need prototyping before implementation decisions are locked:

1. Whether free-symbol chunks should survive at all, or be kept only as an internal stepping stone.
2. Whether named application should be the only documented form, with shorthand intentionally undocumented.
3. Whether `defaults { ... }` should be block-scoped or whole-pipeline scoped.
4. Whether built-in helper invocations inside `network!` are worth the parser complexity.
5. Whether `linear(on: axis, out: N)` should become the general primitive, with `dense(N)` demoted to sugar.
6. Whether structured slot parameters should replace plain axis-list slots as the primary chunk interface model.
7. Whether chunk application should reject overlapping actual-axis bindings by default.
8. How symbolic extent variables should be represented in the DSL/core boundary.
9. Whether structured slots should preserve canonical member order or only named membership.
10. Whether labels should preserve identity across `conv` and `linear`, or whether explicit renaming becomes necessary sooner.
11. What the first activation-reference syntax should be for long-skip DAGs.
12. Whether inputs and saved activations should be unified as named sources under one reference model.
13. What the first multi-input syntax should be.
14. Whether `flatten(into: f)` should be mandatory or whether plain `flatten` should default to `f`/`features`.

## Current Lean

The strongest current design stance is:

- axis labels are open-ended DSL symbols
- transforms explicitly say which labels they act on
- scoped defaults remove repetition
- parameterized reusable chunks are still the strongest reuse model
- free-symbol chunks are acceptable only as a possible shorthand, not as the leading idiom
- structured chunk interfaces may be stronger than plain axis-list slot binding
- `params(...)` should be treated as a real interface/signature, not decoration
- named chunk application is the strongest primary application form
- overlapping actual-axis bindings should probably be rejected by default
- axis/interface safety alone is not enough; symbolic extent rewriting is required for real shape safety
- saved sources and inputs should probably share one exact symbolic-shape reference model
- the DSL likely needs a general `linear(on: ..., out: ...)` primitive
- inline combinator blocks are likely cleaner than nested `network!` for local structure
- long-skip activation references are likely unavoidable for serious DAG architectures
- inputs and long-skip references likely want one underlying named-source model
- multi-input syntax should be treated as a real requirement, not a niche extension
- built-in helper namespaces are acceptable, but only inside `network!`
- built-in helpers should conceptually be predeclared parameterized blueprint templates
- no public semantic axis taxonomy unless a real implementation pressure makes it unavoidable

This is the version of the DSL that should be pressure-tested next, not the older fixed-role axis model.
