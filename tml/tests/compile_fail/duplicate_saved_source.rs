#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use tml::network;

fn main() {
    let _ = network! {
        input(features: 2) -> dense(2) -> save(skip) -> relu -> save(skip) -> dense(1)
    };
}
