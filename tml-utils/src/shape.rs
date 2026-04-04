//! Type-level tensor shapes.
//!
//! `tml` currently represents shapes as compile-time linked lists built from
//! [`crate::shape::Dim`] and [`crate::shape::Nil`]. That design keeps extents
//! in the type system while also allowing small amounts of runtime reflection
//! such as [`TensorShape::dims`] and [`TensorShape::axis_names`].

use std::marker::PhantomData;

mod sealed {
    pub trait Sealed {}
}

/// The scalar/base-case shape.
///
/// [`Nil`] is the end of the type-level shape list. A scalar tensor has shape
/// [`Nil`], rank `0`, and size `1`.
#[derive(Debug, Clone, Copy)]
pub struct Nil;

/// One axis in a type-level tensor shape.
///
/// [`Dim`] is a single cons-cell in the shape list:
/// - `N` is the length of the current axis
/// - `Rest` is the remainder of the shape
/// - `NAME` is the optional axis label preserved from `shape!`
///
/// The core idea is: in current `tml`, a shape is not a runtime object. It is
/// a type-level linked list of dimensions.
///
/// For example:
/// - `shape!(3)` becomes `Dim<3, Nil>`
/// - `shape!(2, 3, 4)` becomes `Dim<2, Dim<3, Dim<4, Nil>>>`
/// - `shape!(channels: 3, height: 32, width: 32)` becomes
///   `Dim<3, Dim<32, Dim<32, Nil, "width">, "height">, "channels">`
///
/// The struct stores no runtime data beyond [`PhantomData`]. The axis length
/// and optional name live in the type itself.
#[derive(Debug, Clone, Copy)]
pub struct Dim<const N: usize, Rest, const NAME: &'static str = "">(pub(crate) PhantomData<Rest>);

impl sealed::Sealed for Nil {}
impl<const N: usize, Rest, const NAME: &'static str> sealed::Sealed for Dim<N, Rest, NAME> {}

/// A type that can act as a tensor shape.
///
/// [`TensorShape`] is implemented recursively by [`Nil`] and [`Dim`]. It
/// exposes both the compile-time facts about a shape and a small amount of
/// runtime reflection derived from those facts.
///
/// In particular:
/// - [`TensorShape::SIZE`] is the total element count
/// - [`TensorShape::RANK`] is the number of axes
/// - [`TensorShape::offset`] computes a row-major flat index
/// - [`TensorShape::dims`] returns the extents at runtime
/// - [`TensorShape::axis_names`] returns any labels preserved by `shape!`
pub trait TensorShape: sealed::Sealed {
    /// The total number of elements in the shape.
    const SIZE: usize;
    /// The number of axes in the shape.
    const RANK: usize;

    /// Computes the row-major flat offset for one full-rank index.
    fn offset(index: &[usize]) -> usize;
    /// Returns the runtime axis extents in order.
    fn dims() -> Vec<usize>;
    /// Returns any axis labels preserved by `shape!`.
    fn axis_names() -> Vec<Option<&'static str>>;
}

/// A [`TensorShape`] with at least one axis.
///
/// This is mostly used for first-axis indexing APIs like
/// [`crate::Tensor::get_ref`], where the library needs
/// to peel off the head axis and return the remaining subshape.
pub trait NonScalarShape: TensorShape {
    /// The remainder of the shape after removing the first axis.
    type Subshape: TensorShape;

    /// The length of the first axis.
    const AXIS_LEN: usize;
}

impl TensorShape for Nil {
    const SIZE: usize = 1;
    const RANK: usize = 0;

    fn offset(index: &[usize]) -> usize {
        assert!(index.is_empty(), "expected scalar index");
        0
    }

    fn dims() -> Vec<usize> {
        Vec::new()
    }

    fn axis_names() -> Vec<Option<&'static str>> {
        Vec::new()
    }
}

impl<const N: usize, Rest, const NAME: &'static str> TensorShape for Dim<N, Rest, NAME>
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

    fn dims() -> Vec<usize> {
        let mut dims = vec![N];
        dims.extend(Rest::dims());
        dims
    }

    fn axis_names() -> Vec<Option<&'static str>> {
        let mut names = vec![(!NAME.is_empty()).then_some(NAME)];
        names.extend(Rest::axis_names());
        names
    }
}

impl<const N: usize, Rest, const NAME: &'static str> NonScalarShape for Dim<N, Rest, NAME>
where
    Rest: TensorShape,
{
    type Subshape = Rest;
    const AXIS_LEN: usize = N;
}

/// Builds a type-level tensor shape from a list of extents.
///
/// Unnamed axes:
///
/// ```rust
/// #![feature(generic_const_exprs, adt_const_params, unsized_const_params)]
/// #![allow(incomplete_features)]
/// use tml_utils::{shape, TensorShape};
///
/// type Matrix = shape!(2, 3);
/// assert_eq!(Matrix::RANK, 2);
/// assert_eq!(Matrix::SIZE, 6);
/// assert_eq!(Matrix::dims(), vec![2, 3]);
/// assert_eq!(Matrix::axis_names(), vec![None, None]);
/// ```
///
/// Named axes:
///
/// ```rust
/// #![feature(generic_const_exprs, adt_const_params, unsized_const_params)]
/// #![allow(incomplete_features)]
/// use tml_utils::{shape, TensorShape};
///
/// type Image = shape!(channels: 3, height: 32, width: 32);
/// assert_eq!(Image::dims(), vec![3, 32, 32]);
/// assert_eq!(
///     Image::axis_names(),
///     vec![Some("channels"), Some("height"), Some("width")]
/// );
/// ```
#[macro_export]
macro_rules! shape {
    () => {
        $crate::shape::Nil
    };

    ($name:ident : $dim:expr $(,)?) => {
        $crate::shape::Dim<{ $dim }, $crate::shape::Nil, { stringify!($name) }>
    };

    ($dim:expr $(,)?) => {
        $crate::shape::Dim<{ $dim }, $crate::shape::Nil>
    };

    ($name:ident : $first:expr, $($rest:tt)+) => {
        $crate::shape::Dim<{ $first }, $crate::shape!($($rest)+), { stringify!($name) }>
    };

    ($first:expr, $($rest:expr),+ $(,)?) => {
        $crate::shape::Dim<{ $first }, $crate::shape!($($rest),+)>
    };
}

#[cfg(test)]
mod tests {
    use super::TensorShape;

    #[test]
    fn unnamed_shapes_have_no_axis_names() {
        type Shape = crate::shape!(2, 3, 4);
        assert_eq!(Shape::axis_names(), vec![None, None, None]);
    }

    #[test]
    fn named_shapes_preserve_axis_names() {
        type Shape = crate::shape!(channels: 3, height: 32, width: 32);
        assert_eq!(
            Shape::axis_names(),
            vec![Some("channels"), Some("height"), Some("width")]
        );
    }
}
