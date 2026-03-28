use std::ops;

use crate::Float;
use crate::shape::{Dim, Nil, TensorShape};

use super::Tensor;

impl<const N: usize, const NAME: &'static str> Tensor<Dim<N, Nil, NAME>> {
    /// Computes the dot product of two vectors.
    pub fn dot(&self, rhs: &Self) -> Float {
        self.as_slice()
            .iter()
            .zip(rhs.as_slice())
            .map(|(lhs, rhs)| lhs * rhs)
            .sum()
    }
}

impl<
    const ROWS: usize,
    const COLS: usize,
    const ROW_NAME: &'static str,
    const COL_NAME: &'static str,
> Tensor<Dim<ROWS, Dim<COLS, Nil, COL_NAME>, ROW_NAME>>
{
    /// Returns the row-major transpose of a matrix.
    pub fn transpose(&self) -> Tensor<Dim<COLS, Dim<ROWS, Nil, ROW_NAME>, COL_NAME>> {
        let mut out = Tensor::<Dim<COLS, Dim<ROWS, Nil, ROW_NAME>, COL_NAME>>::zeros();
        let input = self.as_slice();
        let output = out.as_mut_slice();
        for row in 0..ROWS {
            for col in 0..COLS {
                output[col * ROWS + row] = input[row * COLS + col];
            }
        }
        out
    }

    /// Multiplies a matrix by a vector.
    pub fn matvec(
        &self,
        rhs: &Tensor<Dim<COLS, Nil, COL_NAME>>,
    ) -> Tensor<Dim<ROWS, Nil, ROW_NAME>> {
        let mut out = Tensor::<Dim<ROWS, Nil, ROW_NAME>>::zeros();
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

    /// Multiplies two matrices in row-major layout.
    pub fn matmul<const OUT_COLS: usize, const OUT_COL_NAME: &'static str>(
        &self,
        rhs: &Tensor<Dim<COLS, Dim<OUT_COLS, Nil, OUT_COL_NAME>, COL_NAME>>,
    ) -> Tensor<Dim<ROWS, Dim<OUT_COLS, Nil, OUT_COL_NAME>, ROW_NAME>> {
        let mut out = Tensor::<Dim<ROWS, Dim<OUT_COLS, Nil, OUT_COL_NAME>, ROW_NAME>>::zeros();
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
                for (lhs, rhs) in self
                    .as_mut_slice()
                    .iter_mut()
                    .zip(rhs.as_slice().iter().copied())
                {
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

impl<const N: usize, const NAME: &'static str> From<[Float; N]> for Tensor<Dim<N, Nil, NAME>> {
    fn from(value: [Float; N]) -> Self {
        Self::from_boxed(Vec::from(value).into_boxed_slice())
    }
}
