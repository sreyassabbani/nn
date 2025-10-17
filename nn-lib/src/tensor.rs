use std::{marker::PhantomData, ops};

#[derive(Debug, Clone)]
pub struct Tensor<const N: usize, Shape, Index> {
    data: [f64; N],
    _shape_marker: PhantomData<Shape>,
    _index_marker: PhantomData<Index>,
}

impl<const N: usize, Shape, Index> Tensor<N, Shape, Index> {
    pub fn new() -> Self {
        Self {
            data: [0.; N],
            _shape_marker: PhantomData,
            _index_marker: PhantomData,
        }
    }

    pub fn reshape<AltShp>(self) -> Tensor<N, AltShp, Index>
    where
        Tensor<N, AltShp, Index>: Sized,
    {
        assert_eq!(size_of::<AltShp>(), N);
        let Tensor { data, .. } = self;

        Tensor {
            data,
            _shape_marker: PhantomData::<AltShp>,
            _index_marker: PhantomData,
        }
    }
}

impl<const N: usize, Shape, Index> ops::Index<Index> for Tensor<N, Shape, Index> {
    type Output = f64;

    fn index(&self, index: Index) -> &Self::Output {
        &0.
    }
}

impl<const N: usize, Shape, Index> Default for Tensor<N, Shape, Index> {
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
macro_rules! index_ty {
    ($discard:expr) => {
        (usize)
    };

    ($discard:expr, $($rest:expr),+ $(,)?) => {
        $crate::index_ty!(@acc [usize,] $($rest),+)
    };

    (@acc [$($rec:tt)*] $discard:expr, $($rest:expr),+) => {
        $crate::index_ty!(@acc [$($rec)* usize,] $($rest),+)
    };

    (@acc [$($rec:tt)*] $discard:expr) => {
        ($($rec)* usize)
    };
}

#[macro_export]
macro_rules! tensor {
    ($first:expr $(, $rest:expr)* $(,)?) => {
        {
            const N: usize = $first $( * $rest )*;
            type Shape = $crate::shape_ty!($first $(, $rest)*);
            type Index = $crate::index_ty!($first $(, $rest)*);

            <$crate::Tensor::<N, Shape, Index>>::new()
        }
    };
}
