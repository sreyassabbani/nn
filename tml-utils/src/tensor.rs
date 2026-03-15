use std::{marker::PhantomData, ops};

use crate::shape::{Dim, Nil, NonScalarShape, TensorShape};
use crate::{Assert, Float, IsTrue};

#[derive(Debug)]
struct TensorCore<Shape> {
    data: Box<[Float]>,
    _shape_marker: PhantomData<Shape>,
}

#[derive(Debug)]
pub struct Tensor<Shape: TensorShape> {
    core: TensorCore<Shape>,
}

#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a, Shape: TensorShape> {
    data: &'a [Float],
    _shape_marker: PhantomData<Shape>,
}

#[derive(Debug)]
pub struct TensorViewMut<'a, Shape: TensorShape> {
    data: &'a mut [Float],
    _shape_marker: PhantomData<Shape>,
}

pub trait TensorLiteral {
    type Shape: TensorShape;

    fn write_flat(self, out: &mut Vec<Float>);
}

impl<Shape> Clone for TensorCore<Shape> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            _shape_marker: PhantomData,
        }
    }
}

impl<Shape> Clone for Tensor<Shape>
where
    Shape: TensorShape,
{
    fn clone(&self) -> Self {
        Self {
            core: self.core.clone(),
        }
    }
}

impl<Shape> Tensor<Shape>
where
    Shape: TensorShape,
{
    pub(crate) fn from_boxed(data: Box<[Float]>) -> Self {
        assert_eq!(data.len(), Shape::SIZE, "tensor storage size mismatch");
        Self {
            core: TensorCore {
                data,
                _shape_marker: PhantomData,
            },
        }
    }

    pub fn from_flat<const N: usize>(data: [Float; N]) -> Self
    where
        Assert<{ N == Shape::SIZE }>: IsTrue,
    {
        Self::from_boxed(Vec::from(data).into_boxed_slice())
    }

    pub(crate) fn raw_slice(&self) -> &[Float] {
        &self.core.data
    }

    pub(crate) fn raw_mut_slice(&mut self) -> &mut [Float] {
        &mut self.core.data
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

    pub fn as_slice(&self) -> &[Float] {
        self.raw_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [Float] {
        self.raw_mut_slice()
    }

    pub fn reshape<NewShape>(self) -> Tensor<NewShape>
    where
        NewShape: TensorShape,
        Assert<{ Shape::SIZE == NewShape::SIZE }>: IsTrue,
    {
        Tensor::<NewShape>::from_boxed(self.core.data)
    }

    pub fn view(&self) -> TensorView<'_, Shape> {
        TensorView {
            data: self.as_slice(),
            _shape_marker: PhantomData,
        }
    }

    pub fn view_mut(&mut self) -> TensorViewMut<'_, Shape> {
        TensorViewMut {
            data: self.as_mut_slice(),
            _shape_marker: PhantomData,
        }
    }

    pub fn at(&self, index: [usize; Shape::RANK]) -> &Float {
        let offset = Shape::offset(&index);
        &self.core.data[offset]
    }

    pub fn set(&mut self, index: [usize; Shape::RANK], value: Float) {
        let offset = Shape::offset(&index);
        self.core.data[offset] = value;
    }

    pub fn map_inplace<F>(&mut self, mut f: F)
    where
        F: FnMut(Float) -> Float,
    {
        for value in self.core.data.iter_mut() {
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
    pub fn get(&self, index: usize) -> Tensor<Shape::Subshape> {
        let view = self.get_view(index);
        Tensor::<Shape::Subshape>::from_boxed(view.data.to_vec().into_boxed_slice())
    }

    pub fn get_view(&self, index: usize) -> TensorView<'_, Shape::Subshape> {
        assert!(index < Shape::AXIS_LEN, "index out of bounds");
        let stride = <Shape::Subshape as TensorShape>::SIZE;
        let start = index * stride;
        let end = start + stride;
        TensorView {
            data: &self.core.data[start..end],
            _shape_marker: PhantomData,
        }
    }

    pub fn get_view_mut(&mut self, index: usize) -> TensorViewMut<'_, Shape::Subshape> {
        assert!(index < Shape::AXIS_LEN, "index out of bounds");
        let stride = <Shape::Subshape as TensorShape>::SIZE;
        let start = index * stride;
        let end = start + stride;
        TensorViewMut {
            data: &mut self.core.data[start..end],
            _shape_marker: PhantomData,
        }
    }
}

impl<'a, Shape> TensorView<'a, Shape>
where
    Shape: TensorShape,
{
    pub fn as_slice(&self) -> &[Float] {
        self.data
    }

    pub fn at(&self, index: [usize; Shape::RANK]) -> &Float {
        let offset = Shape::offset(&index);
        &self.data[offset]
    }
}

impl<'a, Shape> TensorViewMut<'a, Shape>
where
    Shape: TensorShape,
{
    pub fn as_mut_slice(&mut self) -> &mut [Float] {
        self.data
    }

    pub fn at(&self, index: [usize; Shape::RANK]) -> &Float {
        let offset = Shape::offset(&index);
        &self.data[offset]
    }

    pub fn set(&mut self, index: [usize; Shape::RANK], value: Float) {
        let offset = Shape::offset(&index);
        self.data[offset] = value;
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
        for (lhs, rhs) in self.core.data.iter_mut().zip(rhs.core.data.iter()) {
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
        for (lhs, rhs) in self.core.data.iter_mut().zip(rhs.core.data.iter()) {
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
        for (lhs, rhs) in self.core.data.iter_mut().zip(rhs.core.data.iter()) {
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
        for (lhs, rhs) in self.core.data.iter_mut().zip(rhs.core.data.iter()) {
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
        for value in self.core.data.iter_mut() {
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
        for value in self.core.data.iter_mut() {
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
        for value in self.core.data.iter_mut() {
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
        for value in self.core.data.iter_mut() {
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
    fn indexing_views_and_owned_get_match_layout() {
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

        let row = t.get_view(1);
        assert_eq!(*row.at([2, 3]), 23.0);

        let owned = t.get(1);
        assert_eq!(*owned.at([2, 3]), 23.0);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn get_view_panics_on_oob_index() {
        let t = Tensor::<T3>::zeros();
        let _ = t.get_view(2);
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
