use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::__private::{ReshapePreservesElementCount, ShapeRelabelPreservesExtents};
use crate::Float;
use crate::shape::{NonScalarShape, TensorShape};

use super::storage::StorageTensor;
use super::{Tensor, TensorMut, TensorRef};

impl<Shape> Tensor<Shape>
where
    Shape: TensorShape,
{
    pub(crate) fn from_boxed(storage: Box<[Float]>) -> Self {
        assert_eq!(storage.len(), Shape::SIZE, "tensor storage size mismatch");
        Self(StorageTensor::from_storage(storage))
    }

    /// Builds a tensor from flat row-major storage.
    pub fn from_flat(data: [Float; Shape::SIZE]) -> Self {
        Self::from_boxed(Vec::from(data).into_boxed_slice())
    }

    /// Fills a tensor with a repeated element.
    pub fn from_elem(value: Float) -> Self {
        Self::from_boxed(vec![value; Shape::SIZE].into_boxed_slice())
    }

    pub(crate) fn raw_slice(&self) -> &[Float] {
        self.as_slice()
    }

    pub(crate) fn raw_mut_slice(&mut self) -> &mut [Float] {
        self.as_mut_slice()
    }

    /// Returns the total number of stored elements.
    pub fn len(&self) -> usize {
        Shape::SIZE
    }

    /// Returns whether the tensor has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the tensor rank.
    pub fn rank(&self) -> usize {
        Shape::RANK
    }

    /// Returns the contiguous row-major storage.
    pub fn as_slice(&self) -> &[Float] {
        self.0.as_slice()
    }

    /// Returns mutable contiguous row-major storage.
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

    /// Fills the entire tensor with one value.
    pub fn fill(&mut self, value: Float) {
        self.0.fill(value);
    }

    /// Returns the zero tensor for the given shape.
    pub fn zeros() -> Self {
        Self::from_elem(0.0)
    }

    /// Returns a tensor of pseudorandom values sampled from `0..1`.
    pub fn random() -> Self {
        let mut rng = rand::rng();
        Self::random_with(&mut rng)
    }

    /// Returns a pseudorandom tensor built from a fixed seed.
    pub fn random_with_seed(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::random_with(&mut rng)
    }

    /// Returns a pseudorandom tensor using a caller-provided RNG.
    pub fn random_with<R>(rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
    {
        let mut out = Self::zeros();
        for value in out.as_mut_slice() {
            *value = rng.random::<Float>();
        }
        out
    }

    /// Reinterprets the tensor with a new shape that has the same element
    /// count.
    pub fn reshape<NewShape>(self) -> Tensor<NewShape>
    where
        NewShape: TensorShape,
        (): ReshapePreservesElementCount<{ Shape::SIZE }, { NewShape::SIZE }>,
    {
        Tensor::<NewShape>::from_boxed(self.0.storage)
    }

    /// Reinterprets the tensor with the same extents but different axis labels.
    pub fn relabel<NewShape>(self) -> Tensor<NewShape>
    where
        NewShape: TensorShape,
        (): ShapeRelabelPreservesExtents<Shape, NewShape>,
    {
        Tensor::<NewShape>::from_boxed(self.0.storage)
    }

    /// Returns a first-class read-only tensor view over the same storage.
    pub fn as_ref(&self) -> TensorRef<'_, Shape> {
        TensorRef(StorageTensor::from_storage(self.as_slice()))
    }

    /// Returns a first-class mutable tensor view over the same storage.
    pub fn as_mut(&mut self) -> TensorMut<'_, Shape> {
        TensorMut(StorageTensor::from_storage(self.as_mut_slice()))
    }

    /// Applies a scalar transform in place.
    pub fn map_inplace<F>(&mut self, mut f: F)
    where
        F: FnMut(Float) -> Float,
    {
        for value in self.as_mut_slice() {
            *value = f(*value);
        }
    }

    /// Returns a new tensor by applying a scalar transform elementwise.
    pub fn map<F>(&self, f: F) -> Self
    where
        F: FnMut(Float) -> Float,
    {
        let mut out = self.clone();
        out.map_inplace(f);
        out
    }

    /// Zips two tensors of the same shape and maps them elementwise.
    pub fn zip_map<F>(&self, rhs: &Self, mut f: F) -> Self
    where
        F: FnMut(Float, Float) -> Float,
    {
        let mut out = Self::zeros();
        for ((dst, lhs), rhs) in out
            .as_mut_slice()
            .iter_mut()
            .zip(self.as_slice().iter().copied())
            .zip(rhs.as_slice().iter().copied())
        {
            *dst = f(lhs, rhs);
        }
        out
    }

    /// Sums all elements in the tensor.
    pub fn sum(&self) -> Float {
        self.0.sum()
    }

    /// Computes the arithmetic mean of all tensor elements.
    pub fn mean(&self) -> Float {
        self.0.mean()
    }

    #[deprecated(note = "Tensor::slice is not implemented yet")]
    pub fn slice<T: Iterator>(_range: T) {}
}

impl<Shape> Tensor<Shape>
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

    /// Clones the `index`th slice along the leading axis into a new owning
    /// tensor.
    pub fn get(&self, index: usize) -> Tensor<Shape::Subshape> {
        let row = self.get_ref(index);
        Tensor::<Shape::Subshape>::from_boxed(row.as_slice().to_vec().into_boxed_slice())
    }
}
