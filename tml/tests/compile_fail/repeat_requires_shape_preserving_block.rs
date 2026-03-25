#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use tml::network;

fn main() {
    let _ = network! {
        input(features: 2) -> repeat(2, dense(3))
    };
}
