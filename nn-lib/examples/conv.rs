#![allow(unused)]

use std::f64::consts::PI;

use nn::network::Conv;
use nn::tensor;

fn main() {
    let tn = tensor!(2, 3);

    println!("{}", type_of(&tn));
    println!("{:?}", tn.get(0));
    println!("{}", tn.at([1, 1]));

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
