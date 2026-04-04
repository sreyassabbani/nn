#![allow(incomplete_features)]
#![feature(generic_const_exprs, adt_const_params, unsized_const_params)]

use tml::{FragmentExt, InitConfig, network, vision};

fn main() {
    let encoder =
        vision::common::stem::<8, 12>().then_fragment(vision::common::residual_block::<12>());
    let block = vision::common::residual_block::<12>();

    let arch = network! {
        input(channels: 6, height: 64, width: 64)
            -> encoder
            -> save(p2)
            -> block
            -> save(p3)
            -> block
            -> concat_from(p3, channels)
            -> conv(12, kernel: 1)
            -> concat_from(p2, channels)
            -> conv(10, kernel: 1)
            -> flatten
            -> dense(3)
    };

    println!("summary:\n{}", arch.summary());
    println!("shape trace:\n{}", arch.shape_trace());

    let model = arch.materialize(InitConfig::new().seed(29));
    let prediction = model.predict(&[0.1; 6 * 64 * 64]);
    println!("prediction = {prediction:?}");
}
