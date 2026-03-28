//! Typed tensor storage and view APIs.
//!
//! [`Tensor`] owns contiguous tensor storage. [`TensorRef`] and [`TensorMut`]
//! are first-class shaped views over existing tensor data. Unlike plain Rust
//! references such as `&Tensor<_>`, the view types can be reshaped, relabeled,
//! and sliced while preserving the compile-time shape witness.

use std::fmt;

use crate::shape::TensorShape;

mod linalg;
mod literal;
mod owned;
mod storage;
#[cfg(test)]
mod tests;
mod views;

pub use literal::TensorLiteral;
use storage::StorageTensor;

/// An owning tensor with compile-time shape information.
///
/// [`Tensor`] stores its elements contiguously in row-major order. The shape is
/// part of the type parameter, so operations such as indexing, reshaping, and
/// matrix multiplication can use compile-time extent information.
///
/// For read-only or mutable shaped views into existing storage, use
/// [`TensorRef`] or [`TensorMut`].
pub struct Tensor<Shape: TensorShape>(StorageTensor<Box<[crate::Float]>, Shape>);

/// A read-only shaped tensor view.
///
/// [`TensorRef`] is not just `&Tensor<_>`. It is a first-class view value that
/// can come from a whole tensor, a sliced subview, or a reshaped view, while
/// still carrying a compile-time shape witness.
///
/// Borrowing a [`TensorRef`] again, for example as `&TensorRef<'_, Shape>`,
/// simply borrows the view wrapper itself. It does not create an additional
/// tensor access mode.
pub struct TensorRef<'a, Shape: TensorShape>(StorageTensor<&'a [crate::Float], Shape>);

/// A mutable shaped tensor view.
///
/// [`TensorMut`] behaves like a shaped `&'a mut [Float]`: it can read, write,
/// slice, and reshape the underlying elements while preserving the compile-time
/// shape witness.
///
/// Borrowing a [`TensorMut`] as `&TensorMut<'_, Shape>` or
/// `&mut TensorMut<'_, Shape>` only borrows the view wrapper. The underlying
/// mutability rules are still enforced by Rust's borrow checker.
pub struct TensorMut<'a, Shape: TensorShape>(StorageTensor<&'a mut [crate::Float], Shape>);

impl<Shape> fmt::Debug for Tensor<Shape>
where
    Shape: TensorShape,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("rank", &Shape::RANK)
            .field("elements", &self.as_slice())
            .finish()
    }
}

impl<'a, Shape> fmt::Debug for TensorRef<'a, Shape>
where
    Shape: TensorShape,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorRef")
            .field("rank", &Shape::RANK)
            .field("elements", &self.as_slice())
            .finish()
    }
}

impl<'a, Shape> fmt::Debug for TensorMut<'a, Shape>
where
    Shape: TensorShape,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorMut")
            .field("rank", &Shape::RANK)
            .field("elements", &self.as_slice())
            .finish()
    }
}

impl<Shape> Clone for Tensor<Shape>
where
    Shape: TensorShape,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<'a, Shape> Clone for TensorRef<'a, Shape>
where
    Shape: TensorShape,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, Shape> Copy for TensorRef<'a, Shape> where Shape: TensorShape {}

impl<Shape> Default for Tensor<Shape>
where
    Shape: TensorShape,
{
    fn default() -> Self {
        Self::zeros()
    }
}
