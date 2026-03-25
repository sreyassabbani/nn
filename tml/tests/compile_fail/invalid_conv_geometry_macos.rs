#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use tml::network;

fn main() {
    let _ = network! {
        input(channels: 1, height: 4, width: 4) -> conv(2, kernel: 5)
    };
}
