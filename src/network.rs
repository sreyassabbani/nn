use crate::layer::{Dimension, Layer as LayerVariant};

use std::{cmp::max, marker::PhantomData};

pub enum Layer {
    L1(LayerVariant<1>),
    L2(LayerVariant<2>),
    L4(LayerVariant<4>),
    L8(LayerVariant<8>),
    L16(LayerVariant<16>),
    L32(LayerVariant<32>),
    L64(LayerVariant<64>),
    L128(LayerVariant<128>),
    L256(LayerVariant<256>),
    L512(LayerVariant<512>),
}

/// Entry point for model-building
pub struct ModelBuilder<IS, S> {
    layers: Vec<Layer>,
    _state: PhantomData<(IS, S)>,
}

pub struct InputSet<const I: usize>;
pub struct InputUnset;

pub struct Uninitialized;
pub struct Initialized;
pub struct Accepting<const PREV: usize, const BREADTH: usize>;

pub struct Perceptron<const I: usize, const O: usize, const BREADTH: usize> {
    input: [f64; BREADTH],
    layers: Vec,
}

impl ModelBuilder<InputUnset, Uninitialized> {
    /// Instantiate a new [`ModelBuilder`].
    pub fn new() -> ModelBuilder<InputUnset, Initialized> {
        ModelBuilder {
            layers: Vec::new(),
            _state: PhantomData,
        }
    }
}

impl ModelBuilder<InputUnset, Initialized> {
    /// Start the model-building process by passing in an input layer.
    pub fn input<const I: usize>(
        self,
        _dim: Dimension<I>,
    ) -> ModelBuilder<InputSet<I>, Accepting<I>> {
        ModelBuilder {
            layers: self.layers,
            _state: PhantomData,
        }
    }
}

impl<const I: usize, const PREV: usize, const BREADTH: usize>
    ModelBuilder<InputSet<I>, Accepting<PREV, BREADTH>>
{
    /// Add hidden layers to your [`Model`].
    ///
    /// Note: your [`Model`] is generic over `PREV`, [`Model<PREV>`], until you call `output`, which finalizes the model.
    pub fn hidden<const O: usize>(
        mut self,
        layer: Layer<f64, PREV, O>,
    ) -> ModelBuilder<InputSet<I>, Accepting<O, { max(BREADTH, PREV) }>> {
        self.layers.push(Box::new(layer));
        ModelBuilder {
            layers: self.layers,
            _state: PhantomData,
        }
    }

    /// Add the final output layer, returning a [`Perceptron`].
    pub fn output<const O: usize>(self, _dim: Dimension<O>) -> Perceptron<I, O> {
        Perceptron {
            layers: self.layers,
        }
    }
}

impl<const I: usize, const O: usize> Perceptron<I, O> {
    pub fn train(&mut self) {}

    pub fn layers(&self) -> &Vec<Box<dyn Layerable + 'static>> {
        &self.layers
    }

    // Run the perceptron model on a given `input` whose ownership is taken.
    pub fn run(&self, input: Vec<f64>) -> Vec<f64> {
        self.layers
            .iter()
            .fold(input, |acc, layer| layer.forward(acc))
    }
}
