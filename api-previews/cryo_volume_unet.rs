#![allow(unused)]

// Target sketch: this is not implemented on the current branch.
//
// Why it matters:
// - pressure-tests real 3D conv
// - pressure-tests long skips on large graphs
// - pressure-tests whether the topological surface is still readable

use tml::{network, volume};

fn main() {
    let down = volume::common::stem::<8, 16>();
    let block = volume::common::residual_block::<16>();

    let _arch = network! {
        input(channels: 1, depth: 96, height: 192, width: 192)
            -> down
            -> save(enc1)
            -> block
            -> save(enc2)
            -> block
            -> upsample(scale: 2)
            -> concat_from(enc2, channels)
            -> conv(16, kernel: (3, 3, 3), pad: 1)
            -> upsample(scale: 2)
            -> concat_from(enc1, channels)
            -> conv(4, kernel: (3, 3, 3), pad: 1)
    };
}
