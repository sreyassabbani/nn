#![allow(unused)]
#![feature(generic_arg_infer)]

use nn::graph;
use nn::network as nt;

fn main() {
    let mut nt = nt! {
        input(784) -> dense(128) -> relu -> dense(64) -> sigmoid -> dense(10) -> output
    };

    nt.forward(&[0.0; 784]);

    println!("{}", type_of(&nt));

    // Example with a computation graph
    let mut graph = graph! {
        input -> pow(2) -> cos -> scale((1.0 / 3.0)) -> scale((1.0 / 3.0)) -> scale(3.0) -> output
    };

    let (f_of_2, f_p_of_2) = graph.compute(2.0);
    println!("{f_of_2}");
    println!("{f_p_of_2}");

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

fn type_of<T>(_: &T) -> &'static str {
    std::any::type_name::<T>()
}
