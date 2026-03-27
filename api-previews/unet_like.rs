#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use tml::{InitConfig, network, vision};

fn main() {
    let stem = vision::common::stem::<2, 2>();
    let block = vision::common::residual_block::<2>();

    let arch = network! {
        input(channels: 1, height: 16, width: 16)
            -> stem
            -> save(enc1)
            -> block
            -> save(enc2)
            -> block
            -> concat_from(enc2, channels)
            -> conv(2, kernel: 1)
            -> concat_from(enc1, channels)
            -> conv(2, kernel: 1)
            -> flatten
            -> dense(2)
    };

    println!("summary:\n{}", arch.summary());
    println!("shape trace:\n{}", arch.shape_trace());

    let model = arch.materialize(InitConfig::new().seed(23));
    let prediction = model.predict(&[0.25; 256]);
    println!("prediction = {prediction:?}");
}
