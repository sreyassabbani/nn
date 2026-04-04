use std::marker::PhantomData;

use crate::{Float, shape::TensorShape};

pub(super) struct StorageTensor<Storage, Shape: TensorShape> {
    pub(super) storage: Storage,
    pub(super) _shape_marker: PhantomData<Shape>,
}

pub(super) trait StorageRef {
    fn as_slice(&self) -> &[Float];
}

pub(super) trait StorageMut: StorageRef {
    fn as_mut_slice(&mut self) -> &mut [Float];
}

impl StorageRef for Box<[Float]> {
    fn as_slice(&self) -> &[Float] {
        self
    }
}

impl StorageMut for Box<[Float]> {
    fn as_mut_slice(&mut self) -> &mut [Float] {
        self
    }
}

impl StorageRef for &[Float] {
    fn as_slice(&self) -> &[Float] {
        self
    }
}

impl StorageRef for &mut [Float] {
    fn as_slice(&self) -> &[Float] {
        self
    }
}

impl StorageMut for &mut [Float] {
    fn as_mut_slice(&mut self) -> &mut [Float] {
        self
    }
}

impl<Storage, Shape> StorageTensor<Storage, Shape>
where
    Shape: TensorShape,
{
    pub(super) fn from_storage(storage: Storage) -> Self {
        Self {
            storage,
            _shape_marker: PhantomData,
        }
    }
}

impl<Storage, Shape> StorageTensor<Storage, Shape>
where
    Storage: StorageRef,
    Shape: TensorShape,
{
    pub(super) fn as_slice(&self) -> &[Float] {
        StorageRef::as_slice(&self.storage)
    }

    pub(super) fn at(&self, index: [usize; Shape::RANK]) -> &Float {
        let offset = Shape::offset(&index);
        &self.as_slice()[offset]
    }

    pub(super) fn sum(&self) -> Float {
        self.as_slice().iter().copied().sum()
    }

    pub(super) fn mean(&self) -> Float {
        self.sum() / Shape::SIZE as Float
    }
}

impl<Storage, Shape> StorageTensor<Storage, Shape>
where
    Storage: StorageMut,
    Shape: TensorShape,
{
    pub(super) fn as_mut_slice(&mut self) -> &mut [Float] {
        StorageMut::as_mut_slice(&mut self.storage)
    }

    pub(super) fn set(&mut self, index: [usize; Shape::RANK], value: Float) {
        let offset = Shape::offset(&index);
        self.as_mut_slice()[offset] = value;
    }

    pub(super) fn fill(&mut self, value: Float) {
        self.as_mut_slice().fill(value);
    }
}

impl<Storage, Shape> Clone for StorageTensor<Storage, Shape>
where
    Storage: Clone,
    Shape: TensorShape,
{
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            _shape_marker: PhantomData,
        }
    }
}

impl<Storage, Shape> Copy for StorageTensor<Storage, Shape>
where
    Storage: Copy,
    Shape: TensorShape,
{
}
