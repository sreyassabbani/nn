#![allow(unused)]

// Target sketch: this is not implemented on the current branch.
//
// Why it matters:
// - pressure-tests token-native operators without image assumptions
// - pressure-tests local graph regions inside a mostly topological narration
// - pressure-tests whether `dense(...)` should give way to `linear(on: ..., out: ...)`

use tml::{network, sequence};

fn main() {
    let encoder = sequence::transformer::block::<512, 2048, 8>();

    let _arch = network! {
        input(tokens: 4096, features: 512)
            -> encoder
            -> save(long_context)
            -> encoder
            -> branch {
                score: select(last) -> linear(1, on: features),
                trace: sum_from(long_context) -> norm(rms) -> linear(64, on: features),
            }
    };
}
