use std::marker::PhantomData;

use crate::activation::Activation;
use crate::tensor::{Matrix, Vector};

pub struct Dimension<const D: usize>;

#[derive(Debug)]
pub struct Layer<T, const I: usize, const O: usize> {
    pub(crate) neurons: Vector<T, I>,
    pub(crate) weights: Matrix<T, O, I>,
    pub(crate) biases: Vector<T, O>,
    pub(crate) activation: Box<dyn Activation<T>>,
}

#[rustfmt::skip]
macro_rules! all_layers {
    ($name:ident) => {
        $name!(T1);
        $name!(T1, T2);
        $name!(T1, T2, T3);
        $name!(T1, T2, T3, T4);
        $name!(T1, T2, T3, T4, T5);
        $name!(T1, T2, T3, T4, T5, T6);
        $name!(T1, T2, T3, T4, T5, T6, T7);
        $name!(T1, T2, T3, T4, T5, T6, T7, T8);
        $name!(T1, T2, T3, T4, T5, T6, T7, T8, T9);
        $name!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
        $name!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11);
        $name!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12);
        $name!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13);
        $name!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14);
        $name!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15);
        $name!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16);
    }
}

macro_rules! impl_layer {
    () => {
        impl<const I: usize, const O: usize> Layer<f64, I, O> {
            fn forward(&self, input: [f64; I]) -> Vec<f64> {
                assert_eq!(I, input.len());
                (&self.biases + &self.weights * input)
                    .iter()
                    .map(|&n| self.activation.forward(n))
                    .collect()
            }
            fn backward(&self, input: Vec<f64>) -> Vec<f64> {
                input
            }
        }
    };
}

pub struct LayerBuilder<T, AS, const I: usize, const O: usize> {
    activation: Option<Box<dyn Activation<T>>>,
    _state: PhantomData<AS>,
}

pub struct ActivationUnset;
pub struct ActivationSet<A>(PhantomData<A>);

impl<T, const I: usize, const O: usize> LayerBuilder<T, ActivationUnset, I, O> {
    pub fn dense() -> Self {
        LayerBuilder {
            activation: None,
            _state: PhantomData,
        }
    }
}

impl<const I: usize, const O: usize> LayerBuilder<f64, ActivationUnset, I, O> {
    /// Set an activation function, finalizing [`Layer<I, O>`]
    pub fn activation<A: Activation<f64> + 'static>(self, act: A) -> Layer<f64, I, O> {
        Layer {
            neurons: Vector::random(),
            weights: Matrix::random(),
            biases: Vector::random(),
            activation: Box::new(act),
        }
    }
}

impl<const I: usize, const O: usize> Layer<f64, I, O> {
    pub fn weights(&self) -> &Matrix<f64, O, I> {
        &self.weights
    }
}

/// Macro for instantiating a new `dense` layer, with an input size and an output size.
#[macro_export]
macro_rules! __dense_inner {
    ($i:expr, $o:expr) => {
        $crate::layer::LayerBuilder::<f64, $crate::layer::ActivationUnset, $i, $o>::dense()
    };
}

#[cfg(not(feature = "unstable"))]
pub use __dense_inner as dense;

/// (Unstable) macro for instantiating a new `dense` layer, made by specifying an input size
///
/// This macro is behind the "unstable" feature because `generic_arg_infer` is a nightly-only feature. Be sure to turn `generic_arg_infer` on.
#[macro_export]
macro_rules! __dense_inner_unstable {
    ($o:expr) => {
        $crate::layer::LayerBuilder::<f64, $crate::layer::ActivationUnset, _, $o>::dense()
    };
}

#[cfg(feature = "unstable")]
pub use __dense_inner_unstable as dense;

/// Macro for instantiating a new `dense` layer, with an input size and an output size.
#[macro_export]
macro_rules! __dim_inner {
    ($d:expr) => {
        $crate::layer::Dimension::<$d>
    };
}

pub use __dim_inner as dim;
