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

impl<T, A: Activation<T>, const I: usize, const O: usize> Layerable for Layer<T, A, I, O> {
    fn output_size(&self) -> usize {
        O
    }
}

#[derive(Debug)]
pub struct Layer<T, A: Activation<T>, const I: usize, const O: usize> {
    nodes: Vector<T, I>,
    weights: Matrix<T, O, I>,
    biases: Vector<T, O>,
    _activation: PhantomData<A>,
}

// marker types for activation state
pub struct ActivationUnset;
pub struct ActivationSet<A>(PhantomData<A>);

// marker types for dimensions state
pub struct DimensionsUnset;
pub struct DimensionsSet<const D: usize>;

pub struct LayerBuilder<T, AS, IDS, ODS> {
    activation: Option<Box<dyn Activation<T>>>,
    _activation_state: PhantomData<AS>,
    _input_dimensions_state: PhantomData<IDS>,
    _output_dimensions_state: PhantomData<ODS>,
}

impl<T> LayerBuilder<T, ActivationUnset, DimensionsUnset, DimensionsUnset> {
    pub fn new() -> Self {
        Self {
            activation: None,
            _activation_state: PhantomData,
            _input_dimensions_state: PhantomData,
            _output_dimensions_state: PhantomData,
        }
    }
}

impl<T, IDS, ODS> LayerBuilder<T, ActivationUnset, IDS, ODS> {
    pub fn activation<A: Activation<T> + 'static>(
        self,
        activation: A,
    ) -> LayerBuilder<T, ActivationSet<A>, IDS, ODS> {
        LayerBuilder {
            activation: Some(Box::new(activation)),
            _activation_state: PhantomData,
            _input_dimensions_state: PhantomData,
            _output_dimensions_state: PhantomData,
        }
    }
}

impl<T> LayerBuilder<T, ActivationUnset, DimensionsUnset, DimensionsUnset> {
    /// Make a new [`LayerBuilder`] by specifying input dimensions. Starts the layer-building chain.
    pub fn dense<const I: usize>()
    -> LayerBuilder<T, ActivationUnset, DimensionsSet<I>, DimensionsUnset> {
        LayerBuilder {
            activation: None,
            _activation_state: PhantomData,
            _input_dimensions_state: PhantomData,
            _output_dimensions_state: PhantomData,
        }
    }
}

impl<T, A, const I: usize> LayerBuilder<T, A, DimensionsSet<I>, DimensionsUnset> {
    /// Internal function to set output dimensions. Used while zipping together layers in the model-building phase.
    fn set_output_dimensions<const O: usize>(
        self,
    ) -> LayerBuilder<T, A, DimensionsSet<I>, DimensionsSet<O>> {
        LayerBuilder {
            activation: None,
            _activation_state: PhantomData,
            _input_dimensions_state: PhantomData,
            _output_dimensions_state: PhantomData,
        }
    }
}

impl<T, AS> LayerBuilder<T, AS, DimensionsUnset, DimensionsUnset> {
    /// Add input and output dimensions for a [`LayerBuilder`].
    pub fn with_dimensions<const I: usize, const O: usize>(
        self,
    ) -> LayerBuilder<T, AS, DimensionsSet<I>, DimensionsSet<O>> {
        LayerBuilder {
            activation: self.activation,
            _activation_state: PhantomData,
            _input_dimensions_state: PhantomData,
            _output_dimensions_state: PhantomData,
        }
    }
}

impl<A: Activation<f64>, const I: usize, const O: usize>
    LayerBuilder<f64, ActivationSet<A>, DimensionsSet<I>, DimensionsSet<O>>
{
    /// Finish the layer-building chain by constructing a ready-to-go [`Layer`], consuming [`LayerBuilder`].
    pub fn build(self) -> Layer<f64, A, I, O> {
        Layer {
            nodes: Vector::random(),
            weights: Matrix::random(),
            biases: Vector::random(),
            _activation: PhantomData,
        }
    }
}

pub struct ModelBuilder<const PREV: usize> {
    layers: Vec<Box<dyn Layerable>>,
}

pub struct Model {
    layers: Vec<Box<dyn Layerable>>,
}

impl<const PREV: usize> ModelBuilder<PREV> {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_layer<A: Activation<f64> + 'static, const O: usize>(
        mut self,
        layer: LayerBuilder<f64, ActivationSet<A>, DimensionsSet<PREV>, DimensionsUnset>,
    ) -> ModelBuilder<O> {
        self.layers
            .push(Box::new(layer.set_output_dimensions::<O>().build()));

        ModelBuilder {
            layers: self.layers,
        }
    }

    pub fn build(self) -> Model {
        Model {
            layers: self.layers,
        }
    }
}
