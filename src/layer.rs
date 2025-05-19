use std::marker::PhantomData;

use crate::activation::Activation;
use crate::tensor::{Matrix, Vector};

pub trait Transform<T> {
    fn forward(&self, input: T) -> T;
    fn backward(&self, input: T) -> T;
}

pub struct Layer<T, A: Activation<T>, const I: usize, const O: usize> {
    nodes: Vector<T, I>,
    weights: Matrix<T, O, I>,
    biases: Vector<T, O>,
    _activation: PhantomData<A>,
}

// marker types for activation state
struct ActivationUnset;
struct ActivationSet<A>(PhantomData<A>);

// marker types for dimensions state
struct DimensionsUnset;
struct DimensionsSet<const I: usize, const O: usize>;

pub struct LayerBuilder<T, AS, DS> {
    activation: Option<Box<dyn Activation<T>>>,
    _activation_state: PhantomData<AS>,
    _dimensions_state: PhantomData<DS>,
}

impl<T> LayerBuilder<T, ActivationUnset, DimensionsUnset> {
    pub fn new() -> Self {
        Self {
            activation: None,
            _activation_state: PhantomData,
            _dimensions_state: PhantomData,
        }
    }
}

impl<T, DS> LayerBuilder<T, ActivationUnset, DS> {
    pub fn with_activation<A: Activation<T> + 'static>(
        self,
        activation: A,
    ) -> LayerBuilder<T, ActivationSet<A>, DS> {
        LayerBuilder {
            activation: Some(Box::new(activation)),
            _activation_state: PhantomData,
            _dimensions_state: PhantomData,
        }
    }
}

impl<T, AS> LayerBuilder<T, AS, DimensionsUnset> {
    pub fn with_dimensions<const I: usize, const O: usize>(
        self,
    ) -> LayerBuilder<T, AS, DimensionsSet<I, O>> {
        LayerBuilder {
            activation: self.activation,
            _activation_state: PhantomData,
            _dimensions_state: PhantomData,
        }
    }
}

impl<A: Activation<f64>, const I: usize, const O: usize>
    LayerBuilder<f64, ActivationSet<A>, DimensionsSet<I, O>>
{
    pub fn build(self) -> Layer<f64, A, I, O> {
        Layer {
            nodes: Vector::random(),
            weights: Matrix::random(),
            biases: Vector::random(),
            _activation: PhantomData,
        }
    }
}
