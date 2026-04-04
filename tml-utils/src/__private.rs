//! Internal support items for generated code and compile-time constraints.
//!
//! This module is intentionally hidden from normal docs and is not part of the
//! supported public API surface.

use crate::{conv, shape, tensor};

#[doc(hidden)]
pub trait ReshapePreservesElementCount<const FROM: usize, const TO: usize> {}

impl<const N: usize> ReshapePreservesElementCount<N, N> for () {}

#[doc(hidden)]
pub trait ConvKernelFitsInput<
    const H: usize,
    const W: usize,
    const FH: usize,
    const FW: usize,
    const S: usize,
    const P: usize,
>
{
}

#[doc(hidden)]
pub const fn assert_conv_kernel_fits_input(
    h: usize,
    w: usize,
    fh: usize,
    fw: usize,
    stride: usize,
    pad: usize,
) {
    if conv::conv_out_dim(h, pad, fh, stride) == 0 {
        panic!("conv kernel does not fit input height");
    }
    if conv::conv_out_dim(w, pad, fw, stride) == 0 {
        panic!("conv kernel does not fit input width");
    }
}

#[doc(hidden)]
pub const fn conv_kernel_fits_input_checked(
    h: usize,
    w: usize,
    fh: usize,
    fw: usize,
    stride: usize,
    pad: usize,
) -> usize {
    assert_conv_kernel_fits_input(h, w, fh, fw, stride, pad);
    0
}

impl<
    const H: usize,
    const W: usize,
    const FH: usize,
    const FW: usize,
    const S: usize,
    const P: usize,
> ConvKernelFitsInput<H, W, FH, FW, S, P> for ()
where
    [(); conv_kernel_fits_input_checked(H, W, FH, FW, S, P)]:,
{
}

#[doc(hidden)]
pub trait ConvGeometryIsValid<
    const H: usize,
    const W: usize,
    const FH: usize,
    const FW: usize,
    const S: usize,
    const P: usize,
>: ConvKernelFitsInput<H, W, FH, FW, S, P>
{
}

impl<
    const H: usize,
    const W: usize,
    const FH: usize,
    const FW: usize,
    const S: usize,
    const P: usize,
> ConvGeometryIsValid<H, W, FH, FW, S, P> for ()
where
    (): ConvKernelFitsInput<H, W, FH, FW, S, P>,
{
}

#[doc(hidden)]
pub trait ShapeRelabelPreservesExtents<From: shape::TensorShape, To: shape::TensorShape> {}

impl ShapeRelabelPreservesExtents<shape::Nil, shape::Nil> for () {}

impl<const N: usize, RestFrom, RestTo, const FROM_NAME: &'static str, const TO_NAME: &'static str>
    ShapeRelabelPreservesExtents<shape::Dim<N, RestFrom, FROM_NAME>, shape::Dim<N, RestTo, TO_NAME>>
    for ()
where
    RestFrom: shape::TensorShape,
    RestTo: shape::TensorShape,
    (): ShapeRelabelPreservesExtents<RestFrom, RestTo>,
{
}

#[doc(hidden)]
pub fn tensor_from_literal<T>(value: T) -> tensor::Tensor<T::Shape>
where
    T: tensor::TensorLiteral,
{
    let mut flat = Vec::with_capacity(<T::Shape as shape::TensorShape>::SIZE);
    value.write_flat(&mut flat);
    tensor::Tensor::<T::Shape>::from_boxed(flat.into_boxed_slice())
}

#[doc(hidden)]
pub const fn shared_name_id(name: &str) -> usize {
    let bytes = name.as_bytes();
    let mut hash = 1469598103934665603usize;
    let mut idx = 0;
    while idx < bytes.len() {
        hash ^= bytes[idx] as usize;
        hash = hash.wrapping_mul(1099511628211usize);
        idx += 1;
    }
    hash
}
