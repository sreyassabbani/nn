use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct Tensor<const N: usize, Shape> {
    data: [f64; N],
    _shape_marker: PhantomData<Shape>,
}

impl<const N: usize, Shape> Tensor<N, Shape> {
    pub fn new() -> Self {
        Self {
            data: [0.; N],
            _shape_marker: PhantomData,
        }
    }
}

impl<const N: usize, Shape> Default for Tensor<N, Shape> {
    fn default() -> Self {
        Self::new()
    }
}

#[macro_export]
macro_rules! shape_ty {
    ($d:expr) => {
        [f64; $d]
    };
    ($first:expr, $($rest:expr),+ $(,)?) => {
        [$crate::shape_ty!($($rest),+); $first]
    };
}

#[macro_export]
macro_rules! tensor {
    ($first:expr $(, $rest:expr)* $(,)?) => {
        {
            const N: usize = $first $( * $rest )*;
            type Shape = $crate::shape_ty!($first $(, $rest)*);

            <$crate::Tensor::<N, Shape>>::new()
        }
    };
}
