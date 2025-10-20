#![allow(unused)]

use nn::network::Conv;
use nn::{Tensor, tensor};
use std::any::type_name_of_val;

fn main() {
    let mut tn = tensor!(2, 3);

    // println!("{}", type_of(&tn));
    // println!("{:?}", tn.get(0));

    println!("{}", tn.at([1, 1]));

    tn.set([1, 1], 2.);

    println!("{}", tn.at([1, 2]));

    #[rustfmt::skip]
    let c = Conv::<
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        0
    >::init();

    let mut out_space = c.create_output_space();
    let input = Tensor::from([1.; 8])
        .reshape::<<Conv<2, 2, 2, 2, 2, 2, 1, 0> as nn::network::ConvIO>::InputShape>();

    dbg!(&c);
    dbg!(&input);

    c.forward(&input, &mut out_space);

    dbg!(&out_space);
}

// OUTPUT:

// nn::tensor::Tensor<12816, 15, [[[[[[[[f64; 4]; 3]; 2]; 1]; 1]; 89]; 3]; 2]>
// [nn-lib/examples/conv.rs:25:5] &c = Conv {
//     data: [
//         Filter(
//             Tensor {
//                 data: [
//                     0.0,
//                     0.0,
//                     0.0,
//                     0.0,
//                     0.0,
//                     0.0,
//                     0.0,
//                     0.0,
//                 ],
//                 _shape_marker: PhantomData<[[[f64; 2]; 2]; 2]>,
//             },
//         ),
//         Filter(
//             Tensor {
//                 data: [
//                     0.0,
//                     0.0,
//                     0.0,
//                     0.0,
//                     0.0,
//                     0.0,
//                     0.0,
//                     0.0,
//                 ],
//                 _shape_marker: PhantomData<[[[f64; 2]; 2]; 2]>,
//             },
//         ),
//     ],
// }

fn type_of<T>(_: &T) -> &'static str {
    std::any::type_name::<T>()
}
