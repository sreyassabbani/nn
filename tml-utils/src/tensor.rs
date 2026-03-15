use std::{marker::PhantomData, ops};

use crate::shape::{Dim, Nil, NonScalarShape, TensorShape};
use crate::{Assert, Float, IsTrue};

pub struct TensorBase<Storage, Shape: TensorShape> {
    storage: Storage,
    _shape_marker: PhantomData<Shape>,
}

pub type Tensor<Shape> = TensorBase<Box<[Float]>, Shape>;
pub type TensorRef<'a, Shape> = TensorBase<&'a [Float], Shape>;
pub type TensorMut<'a, Shape> = TensorBase<&'a mut [Float], Shape>;

trait StorageRef {
    fn as_slice(&self) -> &[Float];
}

trait StorageMut: StorageRef {
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

pub trait TensorLiteral {
    type Shape: TensorShape;

    fn write_flat(self, out: &mut Vec<Float>);
}

impl<Storage, Shape> std::fmt::Debug for TensorBase<Storage, Shape>
where
    Storage: std::fmt::Debug,
    Shape: TensorShape,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorBase")
            .field("storage", &self.storage)
            .finish_non_exhaustive()
    }
}

impl<Storage, Shape> Clone for TensorBase<Storage, Shape>
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

impl<Storage, Shape> Copy for TensorBase<Storage, Shape>
where
    Storage: Copy,
    Shape: TensorShape,
{
}

impl<Storage, Shape> TensorBase<Storage, Shape>
where
    Shape: TensorShape,
{
    fn from_storage(storage: Storage) -> Self {
        Self {
            storage,
            _shape_marker: PhantomData,
        }
    }
}

impl<Shape> Tensor<Shape>
where
    Shape: TensorShape,
{
    pub fn as_slice(&self) -> &[Float] {
        StorageRef::as_slice(&self.storage)
    }

    pub fn as_mut_slice(&mut self) -> &mut [Float] {
        StorageMut::as_mut_slice(&mut self.storage)
    }

    pub fn at(&self, index: [usize; Shape::RANK]) -> &Float {
        let offset = Shape::offset(&index);
        &self.as_slice()[offset]
    }

    pub fn set(&mut self, index: [usize; Shape::RANK], value: Float) {
        let offset = Shape::offset(&index);
        self.as_mut_slice()[offset] = value;
    }
}

impl<'a, Shape> TensorRef<'a, Shape>
where
    Shape: TensorShape,
{
    pub fn as_slice(&self) -> &[Float] {
        StorageRef::as_slice(&self.storage)
    }

    pub fn at(&self, index: [usize; Shape::RANK]) -> &Float {
        let offset = Shape::offset(&index);
        &self.as_slice()[offset]
    }
}

impl<'a, Shape> TensorMut<'a, Shape>
where
    Shape: TensorShape,
{
    pub fn as_slice(&self) -> &[Float] {
        StorageRef::as_slice(&self.storage)
    }

    pub fn as_mut_slice(&mut self) -> &mut [Float] {
        StorageMut::as_mut_slice(&mut self.storage)
    }

    pub fn at(&self, index: [usize; Shape::RANK]) -> &Float {
        let offset = Shape::offset(&index);
        &self.as_slice()[offset]
    }

    pub fn set(&mut self, index: [usize; Shape::RANK], value: Float) {
        let offset = Shape::offset(&index);
        self.as_mut_slice()[offset] = value;
    }
}

impl<Shape> Tensor<Shape>
where
    Shape: TensorShape,
{
    pub(crate) fn from_boxed(storage: Box<[Float]>) -> Self {
        assert_eq!(storage.len(), Shape::SIZE, "tensor storage size mismatch");
        Self::from_storage(storage)
    }

    pub fn from_flat<const N: usize>(data: [Float; N]) -> Self
    where
        Assert<{ N == Shape::SIZE }>: IsTrue,
    {
        Self::from_boxed(Vec::from(data).into_boxed_slice())
    }

    pub(crate) fn raw_slice(&self) -> &[Float] {
        self.as_slice()
    }

    pub(crate) fn raw_mut_slice(&mut self) -> &mut [Float] {
        self.as_mut_slice()
    }

    pub fn zeros() -> Self {
        Self::from_boxed(vec![0.0; Shape::SIZE].into_boxed_slice())
    }

    pub fn random() -> Self {
        let mut data = vec![0.0; Shape::SIZE];
        for value in &mut data {
            *value = rand::random::<Float>();
        }
        Self::from_boxed(data.into_boxed_slice())
    }

    pub fn reshape<NewShape>(self) -> Tensor<NewShape>
    where
        NewShape: TensorShape,
        Assert<{ Shape::SIZE == NewShape::SIZE }>: IsTrue,
    {
        Tensor::<NewShape>::from_boxed(self.storage)
    }

    pub fn as_ref(&self) -> TensorRef<'_, Shape> {
        TensorRef::from_storage(self.as_slice())
    }

    pub fn as_mut(&mut self) -> TensorMut<'_, Shape> {
        TensorMut::from_storage(self.as_mut_slice())
    }

    pub fn map_inplace<F>(&mut self, mut f: F)
    where
        F: FnMut(Float) -> Float,
    {
        for value in self.as_mut_slice() {
            *value = f(*value);
        }
    }

    pub fn map<F>(&self, f: F) -> Self
    where
        F: FnMut(Float) -> Float,
    {
        let mut out = self.clone();
        out.map_inplace(f);
        out
    }

    #[deprecated(note = "Tensor::slice is not implemented yet")]
    pub fn slice<T: Iterator>(_range: T) {}
}

impl<Shape> Tensor<Shape>
where
    Shape: NonScalarShape,
{
    pub fn get_ref(&self, index: usize) -> TensorRef<'_, Shape::Subshape> {
        assert!(index < Shape::AXIS_LEN, "index out of bounds");
        let stride = <Shape::Subshape as TensorShape>::SIZE;
        let start = index * stride;
        let end = start + stride;
        TensorRef::from_storage(&self.as_slice()[start..end])
    }
}

impl<'a, Shape> TensorRef<'a, Shape>
where
    Shape: NonScalarShape,
{
    pub fn get_ref(&self, index: usize) -> TensorRef<'_, Shape::Subshape> {
        assert!(index < Shape::AXIS_LEN, "index out of bounds");
        let stride = <Shape::Subshape as TensorShape>::SIZE;
        let start = index * stride;
        let end = start + stride;
        TensorRef::from_storage(&self.as_slice()[start..end])
    }
}

impl<Shape> Tensor<Shape>
where
    Shape: NonScalarShape,
{
    pub fn get_mut(&mut self, index: usize) -> TensorMut<'_, Shape::Subshape> {
        assert!(index < Shape::AXIS_LEN, "index out of bounds");
        let stride = <Shape::Subshape as TensorShape>::SIZE;
        let start = index * stride;
        let end = start + stride;
        TensorMut::from_storage(&mut self.as_mut_slice()[start..end])
    }
}

impl<'a, Shape> TensorMut<'a, Shape>
where
    Shape: NonScalarShape,
{
    pub fn get_ref(&self, index: usize) -> TensorRef<'_, Shape::Subshape> {
        assert!(index < Shape::AXIS_LEN, "index out of bounds");
        let stride = <Shape::Subshape as TensorShape>::SIZE;
        let start = index * stride;
        let end = start + stride;
        TensorRef::from_storage(&self.as_slice()[start..end])
    }

    pub fn get_mut(&mut self, index: usize) -> TensorMut<'_, Shape::Subshape> {
        assert!(index < Shape::AXIS_LEN, "index out of bounds");
        let stride = <Shape::Subshape as TensorShape>::SIZE;
        let start = index * stride;
        let end = start + stride;
        TensorMut::from_storage(&mut self.as_mut_slice()[start..end])
    }
}

impl<Shape> Tensor<Shape>
where
    Shape: NonScalarShape,
{
    pub fn get(&self, index: usize) -> Tensor<Shape::Subshape> {
        let row = self.get_ref(index);
        Tensor::<Shape::Subshape>::from_boxed(row.as_slice().to_vec().into_boxed_slice())
    }
}

impl<Shape> Default for Tensor<Shape>
where
    Shape: TensorShape,
{
    fn default() -> Self {
        Self::zeros()
    }
}

impl<Shape> ops::Add<&Tensor<Shape>> for Tensor<Shape>
where
    Shape: TensorShape,
{
    type Output = Tensor<Shape>;

    fn add(mut self, rhs: &Tensor<Shape>) -> Self::Output {
        for (lhs, rhs) in self.as_mut_slice().iter_mut().zip(rhs.as_slice().iter()) {
            *lhs += rhs;
        }
        self
    }
}

impl<Shape> ops::Add<&Tensor<Shape>> for &Tensor<Shape>
where
    Shape: TensorShape,
{
    type Output = Tensor<Shape>;

    fn add(self, rhs: &Tensor<Shape>) -> Self::Output {
        let mut out = self.clone();
        out += rhs;
        out
    }
}

impl<Shape> ops::AddAssign<&Tensor<Shape>> for Tensor<Shape>
where
    Shape: TensorShape,
{
    fn add_assign(&mut self, rhs: &Tensor<Shape>) {
        for (lhs, rhs) in self.as_mut_slice().iter_mut().zip(rhs.as_slice().iter()) {
            *lhs += rhs;
        }
    }
}

impl<Shape> ops::Mul<&Tensor<Shape>> for Tensor<Shape>
where
    Shape: TensorShape,
{
    type Output = Tensor<Shape>;

    fn mul(mut self, rhs: &Tensor<Shape>) -> Self::Output {
        for (lhs, rhs) in self.as_mut_slice().iter_mut().zip(rhs.as_slice().iter()) {
            *lhs *= rhs;
        }
        self
    }
}

impl<Shape> ops::Mul<&Tensor<Shape>> for &Tensor<Shape>
where
    Shape: TensorShape,
{
    type Output = Tensor<Shape>;

    fn mul(self, rhs: &Tensor<Shape>) -> Self::Output {
        let mut out = self.clone();
        out *= rhs;
        out
    }
}

impl<Shape> ops::MulAssign<&Tensor<Shape>> for Tensor<Shape>
where
    Shape: TensorShape,
{
    fn mul_assign(&mut self, rhs: &Tensor<Shape>) {
        for (lhs, rhs) in self.as_mut_slice().iter_mut().zip(rhs.as_slice().iter()) {
            *lhs *= rhs;
        }
    }
}

impl<Shape> ops::Mul<Float> for Tensor<Shape>
where
    Shape: TensorShape,
{
    type Output = Tensor<Shape>;

    fn mul(mut self, rhs: Float) -> Self::Output {
        for value in self.as_mut_slice() {
            *value *= rhs;
        }
        self
    }
}

impl<Shape> ops::MulAssign<Float> for Tensor<Shape>
where
    Shape: TensorShape,
{
    fn mul_assign(&mut self, rhs: Float) {
        for value in self.as_mut_slice() {
            *value *= rhs;
        }
    }
}

impl<Shape> ops::Div<Float> for Tensor<Shape>
where
    Shape: TensorShape,
{
    type Output = Tensor<Shape>;

    fn div(mut self, rhs: Float) -> Self::Output {
        for value in self.as_mut_slice() {
            *value /= rhs;
        }
        self
    }
}

impl<Shape> ops::DivAssign<Float> for Tensor<Shape>
where
    Shape: TensorShape,
{
    fn div_assign(&mut self, rhs: Float) {
        for value in self.as_mut_slice() {
            *value /= rhs;
        }
    }
}

impl<const N: usize> From<[Float; N]> for Tensor<Dim<N, Nil>> {
    fn from(value: [Float; N]) -> Self {
        Self::from_boxed(Vec::from(value).into_boxed_slice())
    }
}

impl TensorLiteral for Float {
    type Shape = Nil;

    fn write_flat(self, out: &mut Vec<Float>) {
        out.push(self);
    }
}

impl<T, const N: usize> TensorLiteral for [T; N]
where
    T: TensorLiteral,
{
    type Shape = Dim<N, T::Shape>;

    fn write_flat(self, out: &mut Vec<Float>) {
        for item in self {
            item.write_flat(out);
        }
    }
}

#[doc(hidden)]
pub fn __tensor_from_literal<T>(value: T) -> Tensor<T::Shape>
where
    T: TensorLiteral,
{
    let mut flat = Vec::with_capacity(<T::Shape as TensorShape>::SIZE);
    value.write_flat(&mut flat);
    Tensor::<T::Shape>::from_boxed(flat.into_boxed_slice())
}

#[macro_export]
macro_rules! tensor {
    [$($items:tt)*] => {
        $crate::__tensor_from_literal([$($items)*])
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    type T3 = crate::shape!(2, 3, 4);

    #[test]
    fn indexing_borrows_and_owned_get_match_layout() {
        let mut t = Tensor::<T3>::zeros();
        let mut value = 0.0;
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    t.set([i, j, k], value);
                    value += 1.0;
                }
            }
        }

        assert_eq!(*t.at([1, 2, 3]), 23.0);

        let row = t.get_ref(1);
        assert_eq!(*row.at([2, 3]), 23.0);

        let owned = t.get(1);
        assert_eq!(*owned.at([2, 3]), 23.0);

        let mut tmut = t.as_mut();
        let mut row_mut = tmut.get_mut(0);
        row_mut.set([0, 0], 99.0);
        assert_eq!(*t.at([0, 0, 0]), 99.0);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn get_ref_panics_on_oob_index() {
        let t = Tensor::<T3>::zeros();
        let _ = t.get_ref(2);
    }

    #[test]
    fn reshape_changes_shape_type_without_reordering_data() {
        let flat = Tensor::<crate::shape!(6)>::from_flat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let reshaped = flat.reshape::<crate::shape!(2, 3)>();
        assert_eq!(reshaped.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(*reshaped.at([1, 2]), 6.0);
    }

    #[test]
    fn tensor_literal_infers_shape_and_layout() {
        let t = crate::tensor![[1.0, 2.0], [3.0, 4.0]];
        assert_eq!(t.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(*t.at([1, 0]), 3.0);
    }
}
