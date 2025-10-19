use std::{any::type_name, intrinsics::transmute_unchecked, marker::PhantomData, ops, rc::Rc};

#[derive(Debug, Clone)]
pub struct Tensor<const N: usize, const D: usize, Shape> {
    data: Rc<[f64; N]>,
    _shape_marker: PhantomData<Shape>,
}

impl<const N: usize, const D: usize, Index> Tensor<N, D, Index> {
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
}

trait ArraySize {
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

impl<const N: usize, const D: usize, Shape> ops::Index<usize> for Tensor<N, D, Shape>
where
    Shape: ops::Index<usize>,
    <Shape as ops::Index<usize>>::Output: Sized + ArraySize,
    Tensor<
        { <<Shape as ops::Index<usize>>::Output as ArraySize>::SIZE },
        { D - 1 },
        <Shape as ops::Index<usize>>::Output,
    >: Sized,
{
    type Output = Tensor<
        { <<Shape as ops::Index<usize>>::Output as ArraySize>::SIZE },
        { D - 1 },
        <Shape as ops::Index<usize>>::Output,
    >;

    fn index(&self, index: usize) -> &Self::Output {
        println!("{:?}", type_name::<Shape>());
        dbg!(&self.data);
        // let data = &(unsafe { transmute_unchecked::<&[f64; N], &Shape>(&*self.data) })[index];
        let data = unsafe {
            transmute_unchecked(transmute_unchecked::<&[f64; N], &Shape>(&*self.data)[index])
        };

        &Tensor {
            data: Rc::new(data),
            _shape_marker: PhantomData,
        }
    }
}

impl<const N: usize, const D: usize, Shape> ops::Index<[usize; D]> for Tensor<N, D, Shape>
where
    Shape: ops::Index<usize>,
    <Shape as ops::Index<usize>>::Output: Sized,
{
    type Output = f64;

    fn index(&self, index: [usize; D]) -> &Self::Output {
        let rdata = unsafe { transmute_unchecked::<&[f64], Shape>(&self.data) };
        let c = rdata;
        for &iidx in index[..index.len() - 1].iter() {
            let c = &c[iidx];
        }
        let c = &c[index[index.len() - 1]];
        unsafe { transmute_unchecked::<Shape, &f64>(c) }
    }
}

// #[macro_export]
// macro_rules! eidx {
//     ($tensor:expr, [$($rest:expr),*]) => {
//         $tensor$([$rest])*
//     };
// }

// impl<const N: usize, const D: usize, Shape> ops::Index<[usize; D]> for Tensor<N, D, Shape> {
//     type Output = f64;

//     fn index(&self, index: [usize; D]) -> &Self::Output {
//         // unsafe {
//         //     index
//         //         .iter()
//         //         .fold(transmute::<[f64; N], Shape>(self.data), |acc, x| acc[x])
//         // }

//         // eidx!(self.data, index)

//         Tensor {
//             data: self.data[// modify something here],
//             _shape_marker: PhantomData
//         }[index[1..]]
//     }
// }

// index recursion base case
// impl<const N: usize, Shape> ops::Index<[usize; 1]> for Tensor<N, 1, Shape> {
//     type Output = f64;

//     fn index(&self, index: [usize; 1]) -> &Self::Output {
//         self.data[index[0]]
//     }
// }

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
            const D: usize = $crate::__dim_ty!($first $($rest )*);

            type Shape = $crate::shape_ty!($first $(, $rest)*);

            <$crate::Tensor::<N, D, Shape>>::new()
        }
    };
}
