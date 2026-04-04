#![allow(unused)]

// Current-surface sketch:
// - arbitrary named input axes are real
// - last-axis `linear(...)` is real
// - long skips still use `save(...)` / `sum_from(...)`

use tml::network;

fn main() {
    let mixer = network! {
        linear(24) -> relu -> linear(16) -> relu
    };

    let _arch = network! {
        input(stations: 18, wavelet: 16)
            -> save(raw_band)
            -> mixer
            -> linear(16)
            -> sum_from(raw_band)
            -> flatten
            -> dense(6)
    };
}
