use std::marker::PhantomData;

mod sealed {
    pub trait Sealed {}
}

#[derive(Debug, Clone, Copy)]
pub struct Nil;

#[derive(Debug, Clone, Copy)]
pub struct Dim<const N: usize, Rest>(pub(crate) PhantomData<Rest>);

impl sealed::Sealed for Nil {}
impl<const N: usize, Rest> sealed::Sealed for Dim<N, Rest> {}

pub trait TensorShape: sealed::Sealed {
    const SIZE: usize;
    const RANK: usize;

    fn offset(index: &[usize]) -> usize;
}

pub trait NonScalarShape: TensorShape {
    type Subshape: TensorShape;
    const AXIS_LEN: usize;
}

impl TensorShape for Nil {
    const SIZE: usize = 1;
    const RANK: usize = 0;

    fn offset(index: &[usize]) -> usize {
        assert!(index.is_empty(), "expected scalar index");
        0
    }
}

impl<const N: usize, Rest> TensorShape for Dim<N, Rest>
where
    Rest: TensorShape,
{
    const SIZE: usize = N * Rest::SIZE;
    const RANK: usize = 1 + Rest::RANK;

    fn offset(index: &[usize]) -> usize {
        assert_eq!(index.len(), Self::RANK, "index rank mismatch");
        let head = index[0];
        assert!(head < N, "index out of bounds");
        head * Rest::SIZE + Rest::offset(&index[1..])
    }
}

impl<const N: usize, Rest> NonScalarShape for Dim<N, Rest>
where
    Rest: TensorShape,
{
    type Subshape = Rest;
    const AXIS_LEN: usize = N;
}

#[macro_export]
macro_rules! shape {
    () => {
        $crate::shape::Nil
    };

    ($dim:expr $(,)?) => {
        $crate::shape::Dim<{ $dim }, $crate::shape::Nil>
    };

    ($first:expr, $($rest:expr),+ $(,)?) => {
        $crate::shape::Dim<{ $first }, $crate::shape!($($rest),+)>
    };
}
