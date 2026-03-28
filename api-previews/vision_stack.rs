#![allow(incomplete_features)]
#![feature(generic_const_exprs, adt_const_params, unsized_const_params)]

use tml::{FragmentExt, InitConfig, network, vision};

fn main() {
    let tower = vision::common::stem::<4, 4>().then_fragment(vision::common::residual_block::<4>());

    let arch = network! {
        input(channels: 1, height: 16, width: 16)
            -> share(tower)
            -> share(tower)
            -> flatten
            -> dense(3)
    };

    println!("summary:\n{}", arch.summary());

    let model = arch.materialize(InitConfig::new().seed(19));
    let prediction = model.predict(&[0.5; 256]);
    println!("prediction = {prediction:?}");
}
