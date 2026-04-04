#![allow(incomplete_features)]
#![feature(generic_const_exprs, adt_const_params, unsized_const_params)]

use tml::{Float, InitConfig, Sample, TrainConfig, network};

#[test]
fn network_macro_returns_blueprint() {
    let arch = network! {
        input(features: 4) -> dense(3) -> relu -> dense(2)
    };
    let ty = std::any::type_name_of_val(&arch);
    assert!(ty.contains("Blueprint"), "type = {ty}");
}

#[test]
fn zero_epochs_returns_zero_loss() {
    let arch = network! {
        input(features: 1) -> dense(1)
    };
    let mut model = arch.materialize(InitConfig::new().seed(7));
    let samples = vec![Sample::new([1.0], [5.0]), Sample::new([2.0], [8.0])];
    let loss = model.fit(&samples, TrainConfig::sgd(0.1).epochs(0));
    assert_eq!(loss, 0.0);
}

#[test]
fn conv_pipeline_infers_consistent_shape() {
    let arch = network! {
        input(channels: 1, height: 4, width: 4)
            -> conv(2, kernel: 3)
            -> relu
            -> flatten
            -> dense(2)
    };
    let model = arch.materialize(InitConfig::new().seed(11));
    let out = model.predict(&[0.0 as Float; 16]);
    assert_eq!(out.len(), 2);
}
