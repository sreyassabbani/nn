# API Previews

These files are design pressure-tests for `tml`.

- `current/` contains sketches that match the branch's current architecture
  surface, but are kept out of Cargo's compiled `examples/` set because they
  are heavier and slow down routine checks.
- `frontier/` contains intentionally non-implemented sketches that probe where
  the DSL still strains: multi-input roots, token-native operators, richer
  graph regions, and domain-specific transforms.

Use them as a design lab, not as a stability promise.

Current branch reality:
- Rust-defined reusable fragments are real.
- `network!` composition is real.
- rooted blueprints with `materialize(InitConfig)` are real.
- `save(name)`, `sum_from(name)`, and `concat_from(name, axis)` are real.
- arbitrary named input axes are real.
- last-axis `linear(...)` is real.

Still frontier:
- multi-input roots
- explicit axis operands such as `linear(on: ..., out: ...)`
- attention/scan/reduce operators over named axes
- true graph-runtime activation capture
- rich volumetric and multimodal built-ins
