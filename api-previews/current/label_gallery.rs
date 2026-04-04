#![allow(incomplete_features)]
#![feature(generic_const_exprs, adt_const_params, unsized_const_params)]

use tml::{Axis, conv, dense, flatten, identity, relu, root, validate_blueprint};

type Spectrogram = tml::shape!(sensor: 1, time: 64, freq: 80);
type PatchGrid = tml::shape!(patch: 16, row: 14, col: 14);
type Tokens = tml::shape!(tokens: 128);

fn main() {
    let spectrogram = validate_blueprint(root::<Spectrogram, _>(
        conv::<8, 3, 3, 1, 1>()
            .then(relu())
            .then(flatten())
            .then(dense::<16>()),
        vec![Axis::CHANNELS, Axis::HEIGHT, Axis::WIDTH],
    ));

    let patch_mixer = validate_blueprint(root::<PatchGrid, _>(
        conv::<32, 1, 1, 1, 0>()
            .then(relu())
            .then(flatten())
            .then(dense::<64>()),
        vec![Axis::CHANNELS, Axis::HEIGHT, Axis::WIDTH],
    ));

    let token_fanout = validate_blueprint(root::<Tokens, _>(
        tml::concat(Axis::new("tokens"), identity(), identity()).then(dense::<32>()),
        Vec::new(),
    ));

    println!("{}", spectrogram.summary());
    println!("{}", patch_mixer.summary());
    println!("{}", token_fanout.summary());
}
