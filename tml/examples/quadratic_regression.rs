#![allow(incomplete_features)]
#![feature(generic_const_exprs, adt_const_params, unsized_const_params)]

use tml::{Float, InitConfig, Sample, TrainConfig, network};

// y = 0.5x^2 - x + 0.2
fn magic(x: Float) -> Float {
    0.5 * x * x - x + 0.2
}

fn main() {
    let arch = network! {
        input(features: 1) -> dense(16) -> relu -> dense(1)
    };
    let mut model = arch.materialize(InitConfig::new().seed(11));

    let samples = (-40..=40)
        .map(|i| {
            let x = i as Float / 10.0;
            let y = magic(x);
            Sample::new([x], [y])
        })
        .collect::<Vec<_>>();

    let config = TrainConfig::adam(0.01)
        .epochs(2500)
        .batch_size(16)
        .shuffle_seed(11);
    let loss = model.fit(&samples, config);
    println!("final loss: {loss}");

    for x in [-2.5, -1.0, 0.0, 1.2, 2.5] {
        let target = magic(x);
        let pred = model.predict(&[x]);
        println!("x = {x:>5.2} -> y = {:.4} (target {:.4})", pred[0], target);
    }
}
