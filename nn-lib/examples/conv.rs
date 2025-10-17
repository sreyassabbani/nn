#![allow(unused)]

use std::f64::consts::PI;

use nn::network::Conv;
use nn::tensor;

fn main() {
    let tn = tensor!(2, 3, 89, 1, 1, 2, 3, 4);

    println!("{}", type_of(&tn));

    #[rustfmt::skip]
    let c = Conv::<
        2,
        2,
        2,
        2,
        2,
        2,
        0,
        0
    >::init();

    dbg!(&c);
}

fn type_of<T>(_: &T) -> &'static str {
    std::any::type_name::<T>()
}
