use std::{intrinsics::transmute_unchecked, marker::PhantomData, mem::transmute, ops};

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

impl<const N: usize, const D: usize, Shape> ops::Index<usize> for Tensor<N, D, Shape>
where
    Shape: ops::Index<usize>,
{
    type Output = <Shape as ops::Index<usize>>::Output;

    fn index(&self, index: usize) -> &Self::Output {
        (unsafe { transmute::<&[f64], Shape>(&self.data) })[index]
    }
}

impl<const N: usize, const D: usize, Shape> ops::Index<[usize; D]> for Tensor<N, D, Shape>
where
    Shape: ops::Index<usize>,
    <Shape as ops::Index<usize>>::Output: Sized,
{
    type Output = f64;

    fn index(&self, index: [usize; D]) -> &Self::Output {
        unsafe {
            let base_ptr = self.data.as_ptr();

            eprintln!("=== Index Operation Debug ===");
            eprintln!("N (total elements): {}", N);
            eprintln!("D (dimensions): {}", D);
            eprintln!("Index array: {:?}", index);
            eprintln!("Shape type name: {}", std::any::type_name::<Shape>());
            eprintln!("Shape size: {} bytes", std::mem::size_of::<Shape>());
            eprintln!(
                "Shape as Index Output type: {}",
                std::any::type_name::<<Shape as ops::Index<usize>>::Output>()
            );
            eprintln!("Base ptr: {:p}", base_ptr);
            eprintln!("Data slice: {:?}", &self.data[..std::cmp::min(10, N)]);

            // Transmute to the nested array structure
            let shape_ref = &*transmute_unchecked::<*const f64, *const Shape>(base_ptr);
            eprintln!("Shape ref ptr: {:p}", shape_ref as *const Shape);

            // Index through, converting to raw pointers each time to avoid type issues
            let mut current_ptr: *const u8 = shape_ref as *const Shape as *const u8;
            eprintln!("Initial current_ptr (as u8): {:p}", current_ptr);

            for (depth, &idx) in index.iter().enumerate() {
                eprintln!("\n--- Depth {} ---", depth);
                eprintln!("Indexing with: {}", idx);
                eprintln!("Current ptr (before index): {:p}", current_ptr);

                if depth < index.len() - 1 {
                    // Not the last dimension - need to descend further
                    let shape_at_level = &*(current_ptr as *const Shape);
                    eprintln!("Shape at level ptr: {:p}", shape_at_level as *const Shape);

                    // Check what we're about to index
                    eprintln!("About to index Shape with idx={}", idx);

                    let indexed = &shape_at_level[idx];
                    let indexed_ptr = indexed as *const _ as *const u8;

                    eprintln!("Indexed result ptr: {:p}", indexed_ptr);
                    eprintln!(
                        "Offset from current: {} bytes",
                        indexed_ptr.offset_from(current_ptr)
                    );

                    // Try to see if we can safely read some data
                    let as_f64_ptr = indexed_ptr as *const f64;
                    eprintln!(
                        "As f64 ptr: {:p}, value if read: {}",
                        as_f64_ptr, *as_f64_ptr
                    );

                    current_ptr = indexed_ptr;
                } else {
                    // Last dimension - get the final f64
                    eprintln!("FINAL INDEXING");
                    let shape_at_level = &*(current_ptr as *const Shape);
                    eprintln!(
                        "Final shape at level ptr: {:p}",
                        shape_at_level as *const Shape
                    );
                    eprintln!("Final index: {}", idx);

                    let indexed = &shape_at_level[idx];
                    let result_ptr = indexed as *const _ as *const f64;

                    eprintln!("Final indexed ptr: {:p}", result_ptr);
                    eprintln!(
                        "Final offset from base: {} bytes",
                        (result_ptr as *const u8).offset_from(base_ptr as *const u8)
                    );
                    eprintln!(
                        "Final offset in f64 units: {} elements",
                        result_ptr.offset_from(base_ptr)
                    );
                    eprintln!(
                        "Expected linear index for all 1s would be around: {}",
                        (0..D).fold(0, |acc, i| acc * index[i] + index[i])
                    );

                    // Check if we're within bounds
                    let offset = result_ptr.offset_from(base_ptr);
                    eprintln!(
                        "Is offset within bounds? {} < {}: {}",
                        offset,
                        N,
                        offset >= 0 && (offset as usize) < N
                    );

                    if offset >= 0 && (offset as usize) < N {
                        eprintln!("Reading value: {}", *result_ptr);
                    } else {
                        eprintln!("WARNING: OUT OF BOUNDS ACCESS!");
                    }

                    eprintln!("=== End Debug ===\n");
                    return &*result_ptr;
                }
            }

            unreachable!()
        }
    }
    // fn index(&self, index: [usize; D]) -> &Self::Output {
    //     let rdata = unsafe { transmute_unchecked::<&[f64], Shape>(&self.data) };
    //     let c = rdata;
    //     for &iidx in index[..index.len() - 1].iter() {
    //         let c = &c[iidx];
    //     }
    //     let c = &c[index[index.len() - 1]];
    //     unsafe { transmute_unchecked::<Shape, &f64>(c) }
    // }
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
