#![allow(incomplete_features)]
#![feature(generic_const_exprs, adt_const_params, unsized_const_params)]

use tml::network;

fn main() {
    let _ = network! {
        input(features: 2) -> repeat(2, dense(3))
    };
}
