#![allow(unused)]
#![feature(generic_arg_infer)]

use std::f64::consts::PI;

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
        input -> Pow(2) -> Cos -> Scale((1.0 / 3.0)) -> Scale((1.0 / 3.0)) -> Scale(3.0) -> output
    };

    let (f_of_2, f_p_of_2) = graph.compute(2.0);
    println!("{f_of_2}");
    println!("{f_p_of_2}");

    // TODO: expected API

    // Multi-input autodiff example
    let mut multi = graph! {
        inputs: [x, y]
        x -> Pow(2) -> @x_sq
        y -> Sin -> @y_sin
        (@x_sq, @y_sin) -> Add -> @result
        output @result
    };

    let (value, grad) = multi
        .compute(&[2.0, PI / 2.0])
        .into_iter()
        .next()
        .expect("compute should return vec of 1 element here - sad");

    println!("multi value: {}", value);
    println!("multi grad: {:?}", grad);

    // Mixed chaining example
    let mut mixed = graph! {
        inputs: [x, y]
        x -> Pow(2) -> @temp1
        y -> Cos -> @temp2
        (@temp1, @temp2) -> Mul -> @res
        output @res
    };

    let (mval, mgrad) = mixed
        .compute(&[1.0, 0.0])
        .into_iter()
        .next()
        .expect("compute shoudl return a vec of 1 element here - sad");

    println!("mixed value: {}", mval);
    println!("mixed grad: {:?}", mgrad);

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
