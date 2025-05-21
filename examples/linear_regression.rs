use nn::activation::{ReLU, Sigmoid};
use nn::layer::{LayerBuilder, ModelBuilder};

fn main() {
    let layer = LayerBuilder::dense::<128>().activation(ReLU);
    let network = ModelBuilder::new()
        .add_layer::<ReLU, 64>(LayerBuilder::dense::<128>().activation(ReLU))
        .add_layer::<Sigmoid, 1>(LayerBuilder::dense::<64>().activation(Sigmoid))
        .build();

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
