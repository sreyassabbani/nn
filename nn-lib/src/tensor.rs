#[rustfmt::skip]
use std::{
    intrinsics::transmute_unchecked,
    marker::PhantomData,
    ops,
    ptr,
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

    pub fn at(&self, index: [usize; D]) -> &f64
    where
        Shape: GetFromIndex<D>,
    {
        unsafe { transmute_unchecked::<&[f64; N], &Shape>(&*self.data) }.at(index)
    }

    pub fn slice<T: Iterator>(range: T) {
        todo!()
    }
}

pub trait GetFromIndex<const N: usize> {
    fn at(&self, index: [usize; N]) -> &f64;
}

// Recursive case: nested arrays
impl<T, const M: usize, const N: usize> GetFromIndex<N> for [T; M]
where
    T: GetFromIndex<{ N - 1 }>,
{
    default fn at(&self, index: [usize; N]) -> &f64 {
        self[index[0]].at(core::array::from_fn(|i| index[i + 1]))
    }
}

// Base case: 1D array
impl<const M: usize> GetFromIndex<1> for [f64; M] {
    fn at(&self, index: [usize; 1]) -> &f64 {
        &self[index[0]]
    }
}

impl GetFromIndex<0> for f64 {
    fn at(&self, _index: [usize; 0]) -> &f64 {
        self
    }
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
