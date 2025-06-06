#![feature(generic_arg_infer)]

use nn::activation::{ReLU, Sigmoid};
use nn::{
    layer::{dense, dim},
    network::ModelBuilder,
};

fn main() {
    let network = ModelBuilder::new()
        .input(dim!(128))
        .hidden(dense!(64).activation(ReLU))
        .hidden(dense!(64).activation(Sigmoid))
        .output(dim!(1));

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
