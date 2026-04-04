#![allow(incomplete_features)]
#![allow(unused)]
#![feature(generic_const_exprs, adt_const_params, unsized_const_params)]

use tml::conv::Conv;
use tml::{Tensor, shape, tensor};

fn main() {
    let mut tn = Tensor::<shape!(2, 3)>::zeros();
    let literal = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let flat = literal.clone().reshape::<shape!(6)>();
    let top_row = literal.get_ref(0);

    println!("{}", tn.at([1, 1]));
    tn.set([1, 1], 2.);
    println!("{}", tn.at([1, 2]));
    println!("{:?}", flat.as_slice());
    println!("{:?}", top_row.as_slice());

    #[rustfmt::skip]
    let mut c = Conv::<
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        0
    >::init();

    let mut avg_out_space = c.create_output_space();
    let mut cur_out_space = c.create_output_space();

    let n = 10000;

    for _ in 0..n {
        // re-randomize conv
        c = Conv::init();

        let input = c.input_from_data([1.; 8]);

        c.forward(&input, &mut cur_out_space);
        avg_out_space += &cur_out_space;
    }

    let avg = avg_out_space / n as f64;
    println!("{:?}", avg.as_slice());
}

fn type_of<T>(_: &T) -> &'static str {
    std::any::type_name::<T>()
}
