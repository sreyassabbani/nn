use std::{marker::PhantomData, ops};

#[derive(Debug, Clone)]
pub struct Tensor<const N: usize, const D: usize, Shape> {
    data: [f64; N],
    _shape_marker: PhantomData<Shape>,
}

impl<const N: usize, const D: usize, Index> Tensor<N, D, Index> {
    pub fn new() -> Self {
        Self {
            data: [0.; N],
            _shape_marker: PhantomData,
        }
    }

    pub fn reshape<AltShp>(self) -> Tensor<N, D, AltShp>
    where
        Tensor<N, D, AltShp>: Sized,
    {
        assert_eq!(size_of::<AltShp>(), N);
        let Tensor { data, .. } = self;

        Tensor {
            data,
            _shape_marker: PhantomData::<AltShp>,
        }
    }
}

impl<const N: usize, const D: usize, Shape> ops::Index<[usize; D]> for Tensor<N, D, Shape> {
    type Output = f64;

    fn index(&self, index: [usize; D]) -> &Self::Output {
        &0.
    }
}

impl<const N: usize, const D: usize, Shape> Default for Tensor<N, D, Shape> {
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

// don't use this to calculate dims outside of anything. it can often lead to a "cycle detected when computing revealed normalized predicates" error
#[macro_export]
macro_rules! __dim_ty {
    () => { 0 };
    ($head:tt $($tail:tt)*) => { 1 + $crate::__dim_ty!($($tail)*) };
}

#[macro_export]
macro_rules! tensor {
    ($first:expr $(, $rest:expr)* $(,)?) => {
        {
            // number of elements
            const N: usize = $first $( * $rest )*;
            // dimension
            const D: usize = $crate::__dim_ty!($first $( * $rest )*);

            type Shape = $crate::shape_ty!($first $(, $rest)*);

            <$crate::Tensor::<N, D, Shape>>::new()
        }
    };
}
