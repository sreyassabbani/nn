use crate::{__private, Float, shape::Dim, shape::Nil, shape::TensorShape};

use super::Tensor;

/// A recursively flattenable tensor literal.
///
/// This trait powers the [`tensor!`](crate::tensor!) macro. Nested arrays carry
/// their inferred shape in the associated [`TensorLiteral::Shape`] type and
/// write their row-major elements into a flat output buffer.
pub trait TensorLiteral {
    /// The inferred compile-time shape of the literal.
    type Shape: TensorShape;

    /// Appends this literal's row-major values into `out`.
    fn write_flat(self, out: &mut Vec<Float>);
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
    __private::tensor_from_literal(value)
}

/// Builds a [`Tensor`] from a nested literal.
///
/// # Examples
///
/// Infer an unlabeled shape directly from the literal:
/// ```
/// use tml_utils as tml;
///
/// let matrix = tml::tensor![[1.0, 2.0], [3.0, 4.0]];
/// assert_eq!(matrix.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
/// ```
///
/// Attach explicit axis labels while keeping the same extents:
/// ```
/// use tml_utils as tml;
///
/// let matrix = tml::tensor! { shape: tml::shape!(row: 2, col: 2); [[1.0, 2.0], [3.0, 4.0]] };
/// assert_eq!(
///     <tml::shape!(row: 2, col: 2) as tml::TensorShape>::axis_names(),
///     vec![Some("row"), Some("col")]
/// );
/// assert_eq!(matrix.get_ref(1).as_slice(), &[3.0, 4.0]);
/// ```
#[macro_export]
macro_rules! tensor {
    { shape: $shape:ty; $value:expr } => {
        $crate::__private::tensor_from_literal($value).relabel::<$shape>()
    };
    { as $shape:ty; $value:expr } => {
        $crate::__private::tensor_from_literal($value).relabel::<$shape>()
    };
    [$($items:tt)*] => {
        $crate::__private::tensor_from_literal([$($items)*])
    };
}
