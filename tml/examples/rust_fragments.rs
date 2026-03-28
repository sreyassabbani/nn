#![allow(incomplete_features)]
#![feature(generic_const_exprs, adt_const_params, unsized_const_params)]

use tml::{FragmentExt, InitConfig, network, vision};

fn main() {
    let tower = vision::common::stem::<2, 2>().then_fragment(vision::common::residual_block::<2>());

    let arch = network! {
        input(channels: 2, height: 8, width: 8) -> tower -> flatten -> dense(4)
    };

    println!("summary:\n{}", arch.summary());

    let model = arch.materialize(InitConfig::new().seed(7));
    let prediction = model.predict(&[0.5; 128]);
    println!("prediction = {prediction:?}");
}
