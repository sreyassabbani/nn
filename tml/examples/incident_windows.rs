#![allow(incomplete_features)]
#![feature(generic_const_exprs, adt_const_params, unsized_const_params)]

use tml::{InitConfig, network};

fn main() {
    let arch = network! {
        input(windows: 6, metrics: 12)
            -> linear(8)
            -> relu
            -> flatten
            -> dense(3)
    };

    println!("{}", arch.summary());
    println!("{}", arch.shape_trace());

    let model = arch.materialize(InitConfig::new().seed(41));
    let out = model.predict(&[0.25; 72]);
    println!("incident-window logits: {out:?}");
}
