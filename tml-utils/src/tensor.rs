use std::{fmt, marker::PhantomData, ops};

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::shape::{Dim, Nil, NonScalarShape, TensorShape};
use crate::{Float, ReshapePreservesElementCount};

struct StorageTensor<Storage, Shape: TensorShape> {
    storage: Storage,
    _shape_marker: PhantomData<Shape>,
}

pub struct Tensor<Shape: TensorShape>(StorageTensor<Box<[Float]>, Shape>);
pub struct TensorRef<'a, Shape: TensorShape>(StorageTensor<&'a [Float], Shape>);
pub struct TensorMut<'a, Shape: TensorShape>(StorageTensor<&'a mut [Float], Shape>);

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

impl<Storage, Shape> StorageTensor<Storage, Shape>
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

impl<Storage, Shape> StorageTensor<Storage, Shape>
where
    Storage: StorageRef,
    Shape: TensorShape,
{
    fn as_slice(&self) -> &[Float] {
        StorageRef::as_slice(&self.storage)
    }

    fn at(&self, index: [usize; Shape::RANK]) -> &Float {
        let offset = Shape::offset(&index);
        &self.as_slice()[offset]
    }

    fn sum(&self) -> Float {
        self.as_slice().iter().copied().sum()
    }

    fn mean(&self) -> Float {
        self.sum() / Shape::SIZE as Float
    }
}

impl<Storage, Shape> StorageTensor<Storage, Shape>
where
    Storage: StorageMut,
    Shape: TensorShape,
{
    fn as_mut_slice(&mut self) -> &mut [Float] {
        StorageMut::as_mut_slice(&mut self.storage)
    }

    fn set(&mut self, index: [usize; Shape::RANK], value: Float) {
        let offset = Shape::offset(&index);
        self.as_mut_slice()[offset] = value;
    }

    fn fill(&mut self, value: Float) {
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

impl<Shape> Default for Tensor<Shape>
where
    Shape: TensorShape,
{
    fn default() -> Self {
        Self::zeros()
    }
}

impl<Shape> Tensor<Shape>
where
    Shape: TensorShape,
{
    pub(crate) fn from_boxed(storage: Box<[Float]>) -> Self {
        assert_eq!(storage.len(), Shape::SIZE, "tensor storage size mismatch");
        Self(StorageTensor::from_storage(storage))
    }

    pub fn from_flat(data: [Float; Shape::SIZE]) -> Self {
        Self::from_boxed(Vec::from(data).into_boxed_slice())
    }

    pub fn from_elem(value: Float) -> Self {
        Self::from_boxed(vec![value; Shape::SIZE].into_boxed_slice())
    }

    pub(crate) fn raw_slice(&self) -> &[Float] {
        self.as_slice()
    }

    pub(crate) fn raw_mut_slice(&mut self) -> &mut [Float] {
        self.as_mut_slice()
    }

    pub fn len(&self) -> usize {
        Shape::SIZE
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn rank(&self) -> usize {
        Shape::RANK
    }

    pub fn as_slice(&self) -> &[Float] {
        self.0.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [Float] {
        self.0.as_mut_slice()
    }

    pub fn at(&self, index: [usize; Shape::RANK]) -> &Float {
        self.0.at(index)
    }

    pub fn set(&mut self, index: [usize; Shape::RANK], value: Float) {
        self.0.set(index, value);
    }

    pub fn fill(&mut self, value: Float) {
        self.0.fill(value);
    }

    pub fn zeros() -> Self {
        Self::from_elem(0.0)
    }

    pub fn random() -> Self {
        let mut rng = rand::rng();
        Self::random_with(&mut rng)
    }

    pub fn random_with_seed(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::random_with(&mut rng)
    }

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

    pub fn reshape<NewShape>(self) -> Tensor<NewShape>
    where
        NewShape: TensorShape,
        (): ReshapePreservesElementCount<{ Shape::SIZE }, { NewShape::SIZE }>,
    {
        Tensor::<NewShape>::from_boxed(self.0.storage)
    }

    pub fn as_ref(&self) -> TensorRef<'_, Shape> {
        TensorRef(StorageTensor::from_storage(self.as_slice()))
    }

    pub fn as_mut(&mut self) -> TensorMut<'_, Shape> {
        TensorMut(StorageTensor::from_storage(self.as_mut_slice()))
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

    pub fn sum(&self) -> Float {
        self.0.sum()
    }

    pub fn mean(&self) -> Float {
        self.0.mean()
    }

    #[deprecated(note = "Tensor::slice is not implemented yet")]
    pub fn slice<T: Iterator>(_range: T) {}
}

impl<'a, Shape> TensorRef<'a, Shape>
where
    Shape: TensorShape,
{
    pub fn len(&self) -> usize {
        Shape::SIZE
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn rank(&self) -> usize {
        Shape::RANK
    }

    pub fn as_slice(&self) -> &[Float] {
        self.0.as_slice()
    }

    pub fn at(&self, index: [usize; Shape::RANK]) -> &Float {
        self.0.at(index)
    }

    pub fn sum(&self) -> Float {
        self.0.sum()
    }

    pub fn mean(&self) -> Float {
        self.0.mean()
    }
}

impl<'a, Shape> TensorMut<'a, Shape>
where
    Shape: TensorShape,
{
    pub fn len(&self) -> usize {
        Shape::SIZE
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn rank(&self) -> usize {
        Shape::RANK
    }

    pub fn as_slice(&self) -> &[Float] {
        self.0.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [Float] {
        self.0.as_mut_slice()
    }

    pub fn at(&self, index: [usize; Shape::RANK]) -> &Float {
        self.0.at(index)
    }

    pub fn set(&mut self, index: [usize; Shape::RANK], value: Float) {
        self.0.set(index, value);
    }

    pub fn fill(&mut self, value: Float) {
        self.0.fill(value);
    }

    pub fn sum(&self) -> Float {
        self.0.sum()
    }

    pub fn mean(&self) -> Float {
        self.0.mean()
    }
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
        TensorRef(StorageTensor::from_storage(&self.as_slice()[start..end]))
    }

    pub fn get_mut(&mut self, index: usize) -> TensorMut<'_, Shape::Subshape> {
        assert!(index < Shape::AXIS_LEN, "index out of bounds");
        let stride = <Shape::Subshape as TensorShape>::SIZE;
        let start = index * stride;
        let end = start + stride;
        TensorMut(StorageTensor::from_storage(&mut self.as_mut_slice()[start..end]))
    }

    pub fn get(&self, index: usize) -> Tensor<Shape::Subshape> {
        let row = self.get_ref(index);
        Tensor::<Shape::Subshape>::from_boxed(row.as_slice().to_vec().into_boxed_slice())
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
        TensorRef(StorageTensor::from_storage(&self.as_slice()[start..end]))
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
        TensorRef(StorageTensor::from_storage(&self.as_slice()[start..end]))
    }

    pub fn get_mut(&mut self, index: usize) -> TensorMut<'_, Shape::Subshape> {
        assert!(index < Shape::AXIS_LEN, "index out of bounds");
        let stride = <Shape::Subshape as TensorShape>::SIZE;
        let start = index * stride;
        let end = start + stride;
        TensorMut(StorageTensor::from_storage(&mut self.as_mut_slice()[start..end]))
    }
}

impl<const N: usize> Tensor<Dim<N, Nil>> {
    pub fn dot(&self, rhs: &Self) -> Float {
        self.as_slice()
            .iter()
            .zip(rhs.as_slice())
            .map(|(lhs, rhs)| lhs * rhs)
            .sum()
    }
}

impl<const ROWS: usize, const COLS: usize> Tensor<Dim<ROWS, Dim<COLS, Nil>>> {
    pub fn transpose(&self) -> Tensor<Dim<COLS, Dim<ROWS, Nil>>> {
        let mut out = Tensor::<Dim<COLS, Dim<ROWS, Nil>>>::zeros();
        let input = self.as_slice();
        let output = out.as_mut_slice();
        for row in 0..ROWS {
            for col in 0..COLS {
                output[col * ROWS + row] = input[row * COLS + col];
            }
        }
        out
    }

    pub fn matvec(&self, rhs: &Tensor<Dim<COLS, Nil>>) -> Tensor<Dim<ROWS, Nil>> {
        let mut out = Tensor::<Dim<ROWS, Nil>>::zeros();
        let lhs = self.as_slice();
        let rhs = rhs.as_slice();
        for row in 0..ROWS {
            let mut acc = 0.0;
            for col in 0..COLS {
                acc += lhs[row * COLS + col] * rhs[col];
            }
            out.as_mut_slice()[row] = acc;
        }
        out
    }

    pub fn matmul<const OUT_COLS: usize>(
        &self,
        rhs: &Tensor<Dim<COLS, Dim<OUT_COLS, Nil>>>,
    ) -> Tensor<Dim<ROWS, Dim<OUT_COLS, Nil>>> {
        let mut out = Tensor::<Dim<ROWS, Dim<OUT_COLS, Nil>>>::zeros();
        let lhs = self.as_slice();
        let rhs = rhs.as_slice();
        let output = out.as_mut_slice();
        for row in 0..ROWS {
            for out_col in 0..OUT_COLS {
                let mut acc = 0.0;
                for inner in 0..COLS {
                    acc += lhs[row * COLS + inner] * rhs[inner * OUT_COLS + out_col];
                }
                output[row * OUT_COLS + out_col] = acc;
            }
        }
        out
    }
}

macro_rules! impl_tensor_binop {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $op:tt) => {
        impl<Shape> ops::$trait<&Tensor<Shape>> for Tensor<Shape>
        where
            Shape: TensorShape,
        {
            type Output = Tensor<Shape>;

            fn $method(mut self, rhs: &Tensor<Shape>) -> Self::Output {
                ops::$assign_trait::$assign_method(&mut self, rhs);
                self
            }
        }

        impl<Shape> ops::$trait<&Tensor<Shape>> for &Tensor<Shape>
        where
            Shape: TensorShape,
        {
            type Output = Tensor<Shape>;

            fn $method(self, rhs: &Tensor<Shape>) -> Self::Output {
                self.clone().$method(rhs)
            }
        }

        impl<Shape> ops::$assign_trait<&Tensor<Shape>> for Tensor<Shape>
        where
            Shape: TensorShape,
        {
            fn $assign_method(&mut self, rhs: &Tensor<Shape>) {
                for (lhs, rhs) in self.as_mut_slice().iter_mut().zip(rhs.as_slice().iter().copied()) {
                    *lhs = *lhs $op rhs;
                }
            }
        }
    };
}

macro_rules! impl_tensor_scalar_binop {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $op:tt) => {
        impl<Shape> ops::$trait<Float> for Tensor<Shape>
        where
            Shape: TensorShape,
        {
            type Output = Tensor<Shape>;

            fn $method(mut self, rhs: Float) -> Self::Output {
                ops::$assign_trait::$assign_method(&mut self, rhs);
                self
            }
        }

        impl<Shape> ops::$assign_trait<Float> for Tensor<Shape>
        where
            Shape: TensorShape,
        {
            fn $assign_method(&mut self, rhs: Float) {
                for value in self.as_mut_slice() {
                    *value = *value $op rhs;
                }
            }
        }
    };
}

impl_tensor_binop!(Add, add, AddAssign, add_assign, +);
impl_tensor_binop!(Sub, sub, SubAssign, sub_assign, -);
impl_tensor_binop!(Mul, mul, MulAssign, mul_assign, *);

impl_tensor_scalar_binop!(Add, add, AddAssign, add_assign, +);
impl_tensor_scalar_binop!(Sub, sub, SubAssign, sub_assign, -);
impl_tensor_scalar_binop!(Mul, mul, MulAssign, mul_assign, *);
impl_tensor_scalar_binop!(Div, div, DivAssign, div_assign, /);

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

    #[test]
    fn tensor_debug_uses_public_type_names() {
        let tensor = crate::tensor![[1.0, 2.0], [3.0, 4.0]];
        assert!(format!("{tensor:?}").starts_with("Tensor {"));
        let row = tensor.get_ref(1);
        assert!(format!("{row:?}").starts_with("TensorRef {"));
    }

    #[test]
    fn elementwise_ops_and_reductions_work() {
        let lhs = crate::tensor![1.0, 2.0, 3.0];
        let rhs = crate::tensor![4.0, 5.0, 6.0];

        assert_eq!((&lhs + &rhs).as_slice(), &[5.0, 7.0, 9.0]);
        assert_eq!((&rhs - &lhs).as_slice(), &[3.0, 3.0, 3.0]);
        assert_eq!((&lhs * &rhs).as_slice(), &[4.0, 10.0, 18.0]);
        assert_eq!((lhs.clone() + 1.0).as_slice(), &[2.0, 3.0, 4.0]);
        assert_eq!(lhs.sum(), 6.0);
        assert_eq!(lhs.mean(), 2.0);
    }

    #[test]
    fn dot_transpose_and_matmul_work() {
        let vec_a = crate::tensor![1.0, 2.0, 3.0];
        let vec_b = crate::tensor![4.0, 5.0, 6.0];
        assert_eq!(vec_a.dot(&vec_b), 32.0);

        let lhs = crate::tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let rhs = crate::tensor![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];
        let product = lhs.matmul(&rhs);
        assert_eq!(product.as_slice(), &[58.0, 64.0, 139.0, 154.0]);

        let transposed = lhs.transpose();
        assert_eq!(transposed.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn seeded_random_is_reproducible() {
        let a = Tensor::<crate::shape!(2, 3)>::random_with_seed(7);
        let b = Tensor::<crate::shape!(2, 3)>::random_with_seed(7);
        let c = Tensor::<crate::shape!(2, 3)>::random_with_seed(9);

        assert_eq!(a.as_slice(), b.as_slice());
        assert_ne!(a.as_slice(), c.as_slice());
    }

    #[test]
    fn randomized_shape_stress_preserves_row_major_layout() {
        let mut tensor = Tensor::<crate::shape!(2, 3, 4)>::zeros();
        let mut rng = StdRng::seed_from_u64(42);

        for index in 0..tensor.len() {
            tensor.as_mut_slice()[index] = rng.random::<Float>();
        }

        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let flat_index = i * 12 + j * 4 + k;
                    assert_eq!(*tensor.at([i, j, k]), tensor.as_slice()[flat_index]);
                }
            }
        }

        let reshaped = tensor.clone().reshape::<crate::shape!(4, 3, 2)>();
        assert_eq!(tensor.as_slice(), reshaped.as_slice());
    }
}
