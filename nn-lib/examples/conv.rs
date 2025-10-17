#![allow(unused)]
#![feature(generic_arg_infer)]

use std::f64::consts::PI;

use nn::graph;
use nn::network as nt;
use nn::tensor;

fn main() {
    // let mut nt = nt! {
    //     input(784) -> dense(128) -> relu -> dense(64) -> sigmoid -> dense(10) -> output
    // };

    // nt.forward(&[0.0; 784]);
    let tn = tensor!(2, 3);

    println!("{}", type_of(&nt));
}

fn type_of<T>(_: &T) -> &'static str {
    std::any::type_name::<T>()
}
