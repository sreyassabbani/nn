#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use tml::network;

fn main() {
    let _ = network! {
        input(1, 4, 4) -> dense(2) -> output
    };
}
