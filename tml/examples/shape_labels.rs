#![allow(incomplete_features)]
#![feature(generic_const_exprs, adt_const_params, unsized_const_params)]

use tml::{Tensor, TensorShape, shape};

type Image = shape!(rgb: 1, row: 2, col: 3);
type Spectrogram = shape!(sensor: 2, time: 4, freq: 8);
type Tokens = shape!(tokens: 16, embed: 64);

fn main() {
    print_shape::<Image>("Image");
    print_shape::<Spectrogram>("Spectrogram");
    print_shape::<Tokens>("Tokens");

    let mut image: Tensor<Image> = Tensor::zeros();
    image.set([0, 1, 2], 9.0);

    println!("image dims: {:?}", Image::dims());
    println!("image axis names: {:?}", Image::axis_names());
    println!("image value at [0, 1, 2]: {}", image.at([0, 1, 2]));
}

fn print_shape<Shape: TensorShape>(name: &str) {
    println!(
        "{name}: rank={}, size={}, dims={:?}, axis_names={:?}",
        Shape::RANK,
        Shape::SIZE,
        Shape::dims(),
        Shape::axis_names()
    );
}
