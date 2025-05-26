use std::marker::PhantomData;

use crate::activation::Activation;
use crate::tensor::{Matrix, Vector};

pub trait Transform<T> {
    fn forward(&self, input: T) -> T;
    fn backward(&self, input: T) -> T;
}

pub trait Layerable: Transform<Vec<f64>> {
    fn output_size(&self) -> usize;
}

pub struct Dimension<const D: usize>;

#[derive(Debug)]
pub struct Layer<T, const I: usize, const O: usize> {
    pub(crate) neurons: Vector<T, I>,
    pub(crate) weights: Matrix<T, O, I>,
    pub(crate) biases: Vector<T, O>,
    pub(crate) activation: Box<dyn Activation<T>>,
}

impl<const I: usize, const O: usize> Layerable for Layer<f64, I, O> {
    fn output_size(&self) -> usize {
        O
    }
}

impl<const I: usize, const O: usize> Transform<Vec<f64>> for Layer<f64, I, O> {
    fn forward(&self, input: Vec<f64>) -> Vec<f64> {
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
