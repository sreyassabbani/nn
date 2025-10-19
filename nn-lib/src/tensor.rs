use std::ptr;
#[rustfmt::skip]
use std::{
    intrinsics::transmute_unchecked,
    marker::PhantomData,
    ops,
    rc::Rc
};

#[derive(Debug, Clone)]
pub struct Tensor<const N: usize, const D: usize, Shape> {
    data: Rc<[f64; N]>,
    _shape_marker: PhantomData<Shape>,
}

impl<const N: usize, const D: usize, Shape> Tensor<N, D, Shape>
where
    Shape: ops::Index<usize>,
    <Shape as ops::Index<usize>>::Output: Sized + ArraySize,
{
    pub fn new() -> Self {
        Self {
            data: Rc::new([0.; N]),
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

    pub fn get(
        &self,
        index: usize,
    ) -> Tensor<
        { <<Shape as ops::Index<usize>>::Output as ArraySize>::SIZE },
        { D - 1 },
        <Shape as ops::Index<usize>>::Output,
    >
    where
        <Shape as ops::Index<usize>>::Output: Sized,
    {
        unsafe {
            let t_data = &transmute_unchecked::<&[f64; N], &Shape>(&*self.data)[index];

            let ptr = t_data as *const _ as *const f64;
            let data_arr: [f64; <<Shape as ops::Index<usize>>::Output as ArraySize>::SIZE] =
                ptr::read(ptr as *const _);

            Tensor {
                data: Rc::new(data_arr),
                _shape_marker: PhantomData,
            }
        }
    }

    pub fn at(&self, index: [usize; D]) -> f64
    where
        Tensor<
            { <<Shape as ops::Index<usize>>::Output as ArraySize>::SIZE },
            { D - 1 },
            <Shape as ops::Index<usize>>::Output,
        >: Sized,
        <Shape as ops::Index<usize>>::Output: ops::Index<usize> + ArraySize,
        <<Shape as ops::Index<usize>>::Output as ops::Index<usize>>::Output:
            Sized + ArraySize + ops::Index<usize>,
        <Shape as ops::Index<usize>>::Output: Sized,
    {
        if D == 1 {
            self.data[index[0]]
        } else {
            unsafe {
                let first = index[0];
                // Transmute &[usize] to &[usize; D-1]
                let rest: [usize; D - 1] =
                    std::ptr::read(index[1..].as_ptr() as *const [usize; D - 1]);

                self.get(first).at(rest)
            }
        }
    }

    pub fn slice<T: Iterator>(range: T) {}
}

pub trait ArraySize {
    const SIZE: usize;
}

// Base case: f64 has "size" 1
impl ArraySize for f64 {
    const SIZE: usize = 1;
}

// Recursive case: [T; N] has size N * T::SIZE
impl<T: ArraySize, const N: usize> ArraySize for [T; N] {
    const SIZE: usize = N * T::SIZE;
}

impl<const N: usize, const D: usize, Shape> Default for Tensor<N, D, Shape>
where
    Shape: ops::Index<usize>,
    <Shape as ops::Index<usize>>::Output: Sized + ArraySize,
{
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
            const D: usize = $crate::__dim_ty!($first $($rest )*);

            type Shape = $crate::shape_ty!($first $(, $rest)*);

            <$crate::Tensor::<N, D, Shape>>::new()
        }
    };
}
