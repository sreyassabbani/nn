#![allow(incomplete_features)]
#![feature(generic_const_exprs, adt_const_params, unsized_const_params)]

use tml::{Tensor, TensorShape, shape, tensor};

type Image = shape!(rgb: 1, row: 2, col: 3);
type Spectrogram = shape!(sensor: 2, time: 4, freq: 8);
type Tokens = shape!(tokens: 16, embed: 64);

fn main() {
    print_shape::<Image>("Image");
    print_shape::<Spectrogram>("Spectrogram");
    print_shape::<Tokens>("Tokens");

    let mut image: Tensor<Image> = Tensor::zeros();
    image.set([0, 1, 2], 9.0);

    let literal = tensor! { shape: shape!(sensor: 2, time: 4, freq: 8); [
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
            [24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0],
        ],
        [
            [32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0],
            [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0],
            [48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0],
            [56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0],
        ],
    ] };

    println!("image dims: {:?}", Image::dims());
    println!("image axis names: {:?}", Image::axis_names());
    println!("image value at [0, 1, 2]: {}", image.at([0, 1, 2]));
    println!("literal axis names: {:?}", Spectrogram::axis_names());
    println!("literal mean: {}", literal.mean());
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
