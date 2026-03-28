#![allow(incomplete_features)]
#![feature(generic_const_exprs, adt_const_params, unsized_const_params)]

use tml::{Tensor, TensorMut, TensorRef, TensorShape, shape, tensor};

type Image = shape!(channel: 1, row: 2, col: 3);

fn main() {
    dbg!(Image::dims());
    dbg!(Image::axis_names());

    let mut zeros: Tensor<Image> = Tensor::zeros();
    zeros.set([0, 1, 2], 9.0);
    dbg!(&zeros.as_slice());

    let literal = tensor![as shape!(row: 2, col: 3); [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]];
    dbg!(&literal.at([1, 2]));
    dbg!(literal.rank());
    dbg!(<shape!(row: 2, col: 3) as TensorShape>::axis_names());

    let flat = literal.clone().reshape::<shape!(row: 3, col: 2)>();
    dbg!(&flat.map(|x| x * 2.));

    let row: TensorRef<'_, shape!(col: 3)> = literal.get_ref(1);
    dbg!(&row);

    // mutated literal
    let mut literal_mut = literal.clone();
    let mut row_mut: TensorMut<'_, shape!(col: 3)> = literal_mut.get_mut(0);
    row_mut.set([1], 9.0);
    dbg!(&literal_mut.as_slice());
}
