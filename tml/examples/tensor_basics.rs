#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use tml::{Tensor, TensorMut, TensorRef, shape, tensor};

type Image = shape!(1, 2, 3);

fn main() {
    let mut zeros: Tensor<Image> = Tensor::zeros();
    zeros.set([0, 1, 2], 9.0);
    println!("typed zeros = {:?}", zeros.as_slice());

    let literal = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    println!("literal[1, 2] = {}", literal.at([1, 2]));

    let flat = literal.clone().reshape::<shape!(6)>();
    println!("flat = {:?}", flat.as_slice());

    let row: TensorRef<'_, shape!(3)> = literal.get_ref(1);
    println!("row = {:?}", row.as_slice());

    let mut literal_mut = literal.clone();
    let mut row_mut: TensorMut<'_, shape!(3)> = literal_mut.get_mut(0);
    row_mut.set([1], 9.0);
    println!("mutated literal = {:?}", literal_mut.as_slice());
}
