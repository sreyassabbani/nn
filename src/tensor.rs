use rand::random;
use std::{array, ops};

#[derive(Debug)]
pub struct Vector<T, const N: usize> {
    entries: Box<[T; N]>,
}

#[derive(Debug)]
pub struct Matrix<T, const M: usize, const N: usize> {
    entries: Box<[[T; M]; M]>,
}

impl<const N: usize> Vector<f64, N> {
    pub fn random() -> Self {
        Self {
            entries: Box::new(array::from_fn(|_| random::<f64>())),
        }
    }
}

impl<const M: usize, const N: usize> Matrix<f64, M, N> {
    pub fn random() -> Self {
        Self {
            entries: Box::new([0; M].map(|_| array::from_fn(|_| random::<f64>()))),
        }
    }
}

impl<const M: usize, const N: usize> ops::Mul<Vec<f64>> for &Matrix<f64, M, N> {
    type Output = Vec<f64>;
    fn mul(self, rhs: Vec<f64>) -> Self::Output {
        self.entries
            .iter()
            .map(|row| row.iter().zip(rhs.iter()).map(|(l, r)| l * r).sum())
            .into_iter()
            .collect()
    }
}

impl<const N: usize> ops::Add<Vec<f64>> for &Vector<f64, N> {
    type Output = Vec<f64>;
    fn add(self, rhs: Vec<f64>) -> Self::Output {
        self.entries
            .iter()
            .zip(rhs.iter())
            .map(|(l, r)| l + r)
            .into_iter()
            .collect()
    }
}
