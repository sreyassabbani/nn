#![allow(unused)]

// Current-surface sketch for sequence-like data without a dedicated sequence
// namespace yet.

use tml::network;

fn main() {
    let local_context = network! {
        linear(96) -> relu -> linear(64)
    };

    let _arch = network! {
        input(events: 256, features: 64)
            -> save(raw)
            -> local_context
            -> sum_from(raw)
            -> flatten
            -> dense(32)
            -> relu
            -> dense(4)
    };
}
