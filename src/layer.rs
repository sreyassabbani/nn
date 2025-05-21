use std::marker::PhantomData;

use crate::activation::Activation;
use crate::tensor::{Matrix, Vector};

pub trait Transform<T> {
    fn forward(&self, input: T) -> T;
    fn backward(&self, input: T) -> T;
}

pub trait Layerable {
    fn output_size(&self) -> usize;
}

#[derive(Debug)]
pub struct Layer<T, const I: usize, const O: usize> {
    nodes: Vector<T, I>,
    weights: Matrix<T, O, I>,
    biases: Vector<T, O>,
    activation: Box<dyn Activation<T>>,
}

impl<T, const I: usize, const O: usize> Layerable for Layer<T, I, O> {
    fn output_size(&self) -> usize {
        O
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
            nodes: Vector::random(),
            weights: Matrix::random(),
            biases: Vector::random(),
            activation: Box::new(act),
        }
    }
}

/// Entry point for model-building
pub struct ModelBuilder;

pub struct Model<const PREV: usize> {
    layers: Vec<Box<dyn Layerable>>,
}

pub struct Perceptron<const O: usize> {
    layers: Vec<Box<dyn Layerable>>,
}

impl ModelBuilder {
    /// Instantiate a new [`ModelBuilder`].
    pub fn new() -> Self {
        ModelBuilder
    }

    /// Start the model-building process by passing in an input layer.
    pub fn input<const I: usize, const O: usize>(self, layer: Layer<f64, I, O>) -> Model<O> {
        Model {
            layers: vec![Box::new(layer)],
        }
    }
}

impl<const PREV: usize> Model<PREV> {
    /// Add hidden layers to your [`Model`].
    ///
    /// Note: your [`Model`] is generic over `PREV`, [`Model<PREV>`], until you call `output`, which finalizes the model.
    pub fn hidden<const O: usize>(mut self, layer: Layer<f64, PREV, O>) -> Model<O> {
        self.layers.push(Box::new(layer));
        Model {
            layers: self.layers,
        }
    }

    /// Add the final output layer, returning a [`Perceptron`].
    pub fn output<const O: usize>(mut self, layer: Layer<f64, PREV, O>) -> Perceptron<O> {
        self.layers.push(Box::new(layer));
        Perceptron {
            layers: self.layers,
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

pub use __dense_inner as dense;
