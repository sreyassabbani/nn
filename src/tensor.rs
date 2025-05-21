use rand::random;
use std::array;

#[derive(Debug)]
pub struct Vector<T, const N: usize> {
    entries: Box<[T; N]>,
}

#[derive(Debug)]
pub struct Matrix<T, const N: usize, const M: usize> {
    entries: Box<[[T; N]; M]>,
}

impl<const N: usize> Vector<f64, N> {
    pub fn random() -> Self {
        Self {
            entries: Box::new(array::from_fn(|_| random::<f64>())),
        }
    }
}

impl<const N: usize, const M: usize> Matrix<f64, N, M> {
    pub fn random() -> Self {
        Self {
            entries: Box::new([0; M].map(|_| array::from_fn(|_| random::<f64>()))),
        }
    }
}
