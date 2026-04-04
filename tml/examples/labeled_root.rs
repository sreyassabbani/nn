#![allow(incomplete_features)]
#![feature(generic_const_exprs, adt_const_params, unsized_const_params)]

use tml::{Axis, InitConfig, conv, dense, flatten, relu, root, validate_blueprint};

type Spectrogram = tml::shape!(sensor: 1, time: 8, freq: 8);

fn main() {
    let spec = conv::<2, 3, 3, 1, 1>()
        .then(relu())
        .then(flatten())
        .then(dense::<3>());

    let arch = validate_blueprint(root::<Spectrogram, _>(
        spec,
        vec![Axis::CHANNELS, Axis::HEIGHT, Axis::WIDTH],
    ));

    println!("summary:\n{}", arch.summary());
    println!("shape trace:\n{}", arch.shape_trace());

    let model = arch.materialize(InitConfig::new().seed(21));
    let output = model.predict(&[0.25; 64]);
    println!("prediction: {output:?}");
}
