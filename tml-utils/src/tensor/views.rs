use crate::__private::{ReshapePreservesElementCount, ShapeRelabelPreservesExtents};
use crate::Float;
use crate::shape::{NonScalarShape, TensorShape};

use super::storage::StorageTensor;
use super::{TensorMut, TensorRef};

impl<'a, Shape> TensorRef<'a, Shape>
where
    Shape: TensorShape,
{
    /// Returns the total number of elements visible through the view.
    pub fn len(&self) -> usize {
        Shape::SIZE
    }

    /// Returns whether the view has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the view rank.
    pub fn rank(&self) -> usize {
        Shape::RANK
    }

    /// Returns the contiguous row-major storage referenced by the view.
    pub fn as_slice(&self) -> &[Float] {
        self.0.as_slice()
    }

    /// Reads a single element by its full multi-index.
    pub fn at(&self, index: [usize; Shape::RANK]) -> &Float {
        self.0.at(index)
    }

    /// Sums all elements visible through the view.
    pub fn sum(&self) -> Float {
        self.0.sum()
    }

    /// Computes the arithmetic mean of all view elements.
    pub fn mean(&self) -> Float {
        self.0.mean()
    }

    /// Reinterprets the view with a new shape that has the same element count.
    pub fn reshape<NewShape>(self) -> TensorRef<'a, NewShape>
    where
        NewShape: TensorShape,
        (): ReshapePreservesElementCount<{ Shape::SIZE }, { NewShape::SIZE }>,
    {
        TensorRef(StorageTensor::from_storage(self.0.storage))
    }

    /// Reinterprets the view with the same extents but different axis labels.
    pub fn relabel<NewShape>(self) -> TensorRef<'a, NewShape>
    where
        NewShape: TensorShape,
        (): ShapeRelabelPreservesExtents<Shape, NewShape>,
    {
        TensorRef(StorageTensor::from_storage(self.0.storage))
    }
}

impl<'a, Shape> TensorMut<'a, Shape>
where
    Shape: TensorShape,
{
    /// Returns the total number of elements visible through the view.
    pub fn len(&self) -> usize {
        Shape::SIZE
    }

    /// Returns whether the view has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the view rank.
    pub fn rank(&self) -> usize {
        Shape::RANK
    }

    /// Returns the contiguous row-major storage referenced by the view.
    pub fn as_slice(&self) -> &[Float] {
        self.0.as_slice()
    }

    /// Returns mutable contiguous row-major storage referenced by the view.
    pub fn as_mut_slice(&mut self) -> &mut [Float] {
        self.0.as_mut_slice()
    }

    /// Reads a single element by its full multi-index.
    pub fn at(&self, index: [usize; Shape::RANK]) -> &Float {
        self.0.at(index)
    }

    /// Writes a single element by its full multi-index.
    pub fn set(&mut self, index: [usize; Shape::RANK], value: Float) {
        self.0.set(index, value);
    }

    /// Fills the viewed storage with one value.
    pub fn fill(&mut self, value: Float) {
        self.0.fill(value);
    }

    /// Sums all elements visible through the view.
    pub fn sum(&self) -> Float {
        self.0.sum()
    }

    /// Computes the arithmetic mean of all view elements.
    pub fn mean(&self) -> Float {
        self.0.mean()
    }

    /// Reinterprets the view with a new shape that has the same element count.
    pub fn reshape<NewShape>(self) -> TensorMut<'a, NewShape>
    where
        NewShape: TensorShape,
        (): ReshapePreservesElementCount<{ Shape::SIZE }, { NewShape::SIZE }>,
    {
        TensorMut(StorageTensor::from_storage(self.0.storage))
    }

    /// Reinterprets the view with the same extents but different axis labels.
    pub fn relabel<NewShape>(self) -> TensorMut<'a, NewShape>
    where
        NewShape: TensorShape,
        (): ShapeRelabelPreservesExtents<Shape, NewShape>,
    {
        TensorMut(StorageTensor::from_storage(self.0.storage))
    }
}

impl<'a, Shape> TensorRef<'a, Shape>
where
    Shape: NonScalarShape,
{
    /// Borrows the `index`th slice along the leading axis.
    pub fn get_ref(&self, index: usize) -> TensorRef<'_, Shape::Subshape> {
        assert!(index < Shape::AXIS_LEN, "index out of bounds");
        let stride = <Shape::Subshape as TensorShape>::SIZE;
        let start = index * stride;
        let end = start + stride;
        TensorRef(StorageTensor::from_storage(&self.as_slice()[start..end]))
    }
}

impl<'a, Shape> TensorMut<'a, Shape>
where
    Shape: NonScalarShape,
{
    /// Borrows the `index`th slice along the leading axis.
    pub fn get_ref(&self, index: usize) -> TensorRef<'_, Shape::Subshape> {
        assert!(index < Shape::AXIS_LEN, "index out of bounds");
        let stride = <Shape::Subshape as TensorShape>::SIZE;
        let start = index * stride;
        let end = start + stride;
        TensorRef(StorageTensor::from_storage(&self.as_slice()[start..end]))
    }

    /// Mutably borrows the `index`th slice along the leading axis.
    pub fn get_mut(&mut self, index: usize) -> TensorMut<'_, Shape::Subshape> {
        assert!(index < Shape::AXIS_LEN, "index out of bounds");
        let stride = <Shape::Subshape as TensorShape>::SIZE;
        let start = index * stride;
        let end = start + stride;
        TensorMut(StorageTensor::from_storage(
            &mut self.as_mut_slice()[start..end],
        ))
    }
}
