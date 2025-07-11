#![feature(generic_arg_infer)]

use nn::{
    activation::ReLU,
    layer::{dense, dim},
    network::ModelBuilder,
};

fn main() {
    let network = ModelBuilder::new()
        .input(dim!(1))
        .hidden(dense!(1).activation(ReLU))
        .output(dim!(1));

    let output = network.run(vec![1.0]);
    dbg!(output);

    // TODO: expected API

    // train
    // let train_data = [(1.0, 2.0), (3.0, 2.0)].map(DataSample::from);
    // let eta = 0.3; // learning rate
    // let epochs = 2000;
    // network.train(&train_data, 0.3, epochs);

    // test
    // let test_data = [(5.0, 2.0), (52.0, 2.0)];
    // let cost = network.cost(&test_data);

    // dbg!(cost);
}
