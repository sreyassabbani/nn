#![allow(unused)]

// Target sketch: this is not implemented on the current branch.
//
// Why it matters:
// - pressure-tests multi-input roots
// - pressure-tests mixing image, tabular, and note-text signals
// - pressure-tests whether the topological surface can stay readable in
//   production-ish multimodal pipelines

use tml::{network, sequence, vision};

fn main() {
    let image_tower = vision::common::stem::<16, 32>();
    let note_tower = sequence::retrieval::encoder::<768, 3072, 12>();

    let _arch = network! {
        inputs {
            photo(channels: 1, height: 512, width: 512)
                -> image_tower
                -> flatten
                -> save(photo_repr)

            note(tokens: 1024, features: 768)
                -> share(note_tower)
                -> save(note_repr)

            labs(features: 48)
                -> dense(128)
                -> relu
                -> save(labs_repr)
        }
        -> fuse[photo_repr, note_repr, labs_repr]
        -> linear(5, on: features)
    };
}
