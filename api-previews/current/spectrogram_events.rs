#![allow(incomplete_features)]
#![feature(generic_const_exprs, adt_const_params, unsized_const_params)]

use tml::{InitConfig, network, vision};

fn main() {
    let front_end = vision::common::stem::<4, 8>();
    let refine = vision::common::residual_block::<8>();

    let arch = network! {
        input(channels: 1, height: 96, width: 256)
            -> front_end
            -> save(low_band)
            -> refine
            -> save(mid_band)
            -> refine
            -> concat_from(mid_band, channels)
            -> conv(8, kernel: 1)
            -> concat_from(low_band, channels)
            -> conv(6, kernel: 1)
            -> flatten
            -> dense(4)
    };

    println!("summary:\n{}", arch.summary());
    println!("shape trace:\n{}", arch.shape_trace());

    let model = arch.materialize(InitConfig::new().seed(17));
    let prediction = model.predict(&[0.0; 96 * 256]);
    println!("prediction = {prediction:?}");
}
