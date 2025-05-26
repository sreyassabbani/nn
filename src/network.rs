use crate::layer::{Layer, Layerable, Dimension};

use std::marker::PhantomData;

/// Entry point for model-building
pub struct ModelBuilder<IS, S> {
    layers: Vec<Box<dyn Layerable>>,
    _state: PhantomData<(IS, S)>,
}

pub struct InputSet<const I: usize>;
pub struct InputUnset;

pub struct Uninitialized;
pub struct Initialized;
pub struct Accepting<const PREV: usize>;

pub struct Perceptron<const I: usize, const O: usize> {
    layers: Vec<Box<dyn Layerable>>,
}

impl ModelBuilder<InputUnset, Uninitialized> {
    /// Instantiate a new [`ModelBuilder`].
    pub fn new() -> ModelBuilder<InputUnset, Initialized>{
        ModelBuilder { layers: Vec::new(), _state: PhantomData }
    }
}

impl ModelBuilder<InputUnset, Initialized> {
    /// Start the model-building process by passing in an input layer.
    pub fn input<const I: usize>(self, dim: Dimension<I>) -> ModelBuilder<InputSet<I>, Accepting<I>> {
        ModelBuilder { layers: self.layers, _state: PhantomData }
    }
}

impl<const I: usize, const PREV: usize> ModelBuilder<InputSet<I>, Accepting<PREV>> {

    /// Add hidden layers to your [`Model`].
    ///
    /// Note: your [`Model`] is generic over `PREV`, [`Model<PREV>`], until you call `output`, which finalizes the model.
    pub fn hidden<const O: usize>(mut self, layer: Layer<f64, PREV, O>) -> ModelBuilder<InputSet<I>, Accepting<O>> {
        self.layers.push(Box::new(layer));
        ModelBuilder {
            layers: self.layers,
            _state: PhantomData
        }
    }

    /// Add the final output layer, returning a [`Perceptron`].
    pub fn output<const O: usize>(mut self, dim: Dimension<O>) -> Perceptron<I, O> {
        Perceptron {
            layers: self.layers,
        }
    }
}

impl<const I: usize, const O: usize> Perceptron<I, O> {
    pub fn train(&mut self) {}

    // Run the perceptron model on a given `input` whose ownership is taken.
    // pub fn run(&self, input: Layer<f64, I, O>) {
    //     let output_layer = self
    //         .layers
    //         .iter()
    //         .fold(input, |acc, &layer| layer.forward(acc));
    // }
}
