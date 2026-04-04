#![allow(unused)]

// Target sketch: this is not implemented on the current branch.
//
// Why it matters:
// - pressure-tests whether `dense(...)` should give way to `linear(on: ..., out: ...)`
// - pressure-tests token/feature operators without image assumptions
// - pressure-tests nested graph structure that is not just "conv with skips"

use tml::{network, sequence};

fn main() {
    let block = sequence::transformer::block::<768, 3072, 12>();

    let _arch = network! {
        input(tokens: 512, features: 768)
            -> block
            -> save(low_context)
            -> block
            -> block
            -> sum_from(low_context)
            -> norm(rms)
            -> linear(1, on: features)
    };
}
