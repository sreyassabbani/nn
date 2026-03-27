#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use tml::{InitConfig, network};

fn main() {
    let arch = network! {
        input(features: 2)
            -> dense(8)
            -> relu
            -> save(skip)
            -> dense(8)
            -> relu
            -> sum_from(skip)
            -> dense(1)
    };

    println!("summary:\n{}", arch.summary());
    println!("shape trace:\n{}", arch.shape_trace());

    let model = arch.materialize(InitConfig::new().seed(11));
    let prediction = model.predict(&[0.25, -0.5]);
    println!("prediction = {prediction:?}");
}
