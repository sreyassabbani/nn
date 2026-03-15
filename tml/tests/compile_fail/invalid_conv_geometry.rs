#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use tml::network;

fn main() {
    let _ = network! {
        input(1, 4, 4) -> conv(2, 5, 1, 0) -> output
    };
}
