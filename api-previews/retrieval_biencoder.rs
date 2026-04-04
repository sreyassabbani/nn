#![allow(unused)]

// Target sketch: this is not implemented on the current branch.
//
// Why it matters:
// - pressure-tests multi-input roots
// - pressure-tests shared Rust fragments
// - pressure-tests non-image architectures without collapsing into builder APIs

use tml::{network, sequence};

fn main() {
    let tower = sequence::retrieval::encoder::<768, 3072, 12>();

    let _arch = network! {
        inputs {
            query(tokens: 256, features: 768) -> share(tower) -> save(query_repr)
            candidate(tokens: 256, features: 768) -> share(tower) -> save(candidate_repr)
        }
        -> cosine_from(query_repr, candidate_repr)
    };
}
