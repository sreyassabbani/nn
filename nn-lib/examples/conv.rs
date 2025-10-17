#![allow(unused)]

use std::f64::consts::PI;

use nn::graph;
use nn::network as nt;
use nn::tensor;

fn main() {
    let tn = tensor!(2, 3);

    println!("{}", type_of(&tn));

    let c = nn::network::Conv::<2, 2, 2, 2, 2, 2, 0, 0>::init();

    dbg!(&c);
}

fn type_of<T>(_: &T) -> &'static str {
    std::any::type_name::<T>()
}
