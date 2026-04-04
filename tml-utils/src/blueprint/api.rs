//! Public blueprint values and materialized model wrappers.

use crate::network::{LossFunction, MeanSquaredError, TrainConfig};
use crate::{Float, Sample};
use std::fmt;

use super::SeqSpec;

/// A blueprint-visible axis label.
///
/// [`Axis`] is intentionally lightweight: it is just a static label wrapper
/// used by blueprint summaries, shape traces, concat selection, and related
/// inspection APIs.
///
/// Unlike the tensor-shape layer, [`Axis`] does not encode extents on its own.
/// The actual compile-time dimensions still live in the [`crate::TensorShape`]
/// implementation for the input and output shapes involved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Axis(&'static str);

impl Axis {
    pub const FEATURES: Self = Self("features");
    pub const CHANNELS: Self = Self("channels");
    pub const LENGTH: Self = Self("length");
    pub const DEPTH: Self = Self("depth");
    pub const HEIGHT: Self = Self("height");
    pub const WIDTH: Self = Self("width");

    /// Creates an axis label from a static name.
    pub const fn new(name: &'static str) -> Self {
        Self(name)
    }

    /// Returns the label string.
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

/// Materialization options for a rooted [`Blueprint`].
///
/// A blueprint is an architecture description with no initialized weights.
/// [`InitConfig`] controls how that description becomes concrete model state.
///
/// At the moment the only supported option is an optional RNG seed, but the
/// type is deliberately named for future expansion.
#[derive(Debug, Clone)]
pub struct InitConfig {
    pub(crate) seed: Option<u64>,
}

impl InitConfig {
    /// Creates a default materialization config.
    pub fn new() -> Self {
        Self { seed: None }
    }

    /// Sets the initialization seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

impl Default for InitConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// A reusable architecture description with no materialized parameter state.
///
/// [`Blueprint`] is the core “architecture as data” value in `tml`. It stores
/// a specification value of type `Spec`, but it does **not** store initialized
/// weights, optimizer state, or any runtime buffers.
///
/// For example, a simple reusable chunk can be built and then sequenced:
///
/// ```rust
/// # #![feature(generic_const_exprs, adt_const_params, unsized_const_params)]
/// # #![allow(incomplete_features)]
/// use tml_utils::blueprint::{dense, relu};
///
/// let chunk = dense::<8>().then(relu());
/// let _model = chunk.then(dense::<1>());
/// ```
///
/// To turn a blueprint into a trainable model, first root it with an input
/// shape using helpers such as [`crate::blueprint::root`] or
/// [`crate::blueprint::features_input`], then call
/// [`crate::blueprint::RootedBlueprint::materialize`].
#[derive(Debug, Clone)]
pub struct Blueprint<Spec> {
    spec: Spec,
}

impl<Spec> Blueprint<Spec> {
    /// Wraps a specification value as a [`Blueprint`].
    pub const fn new(spec: Spec) -> Self {
        Self { spec }
    }

    #[doc(hidden)]
    pub fn into_inner(self) -> Spec {
        self.spec
    }

    #[doc(hidden)]
    pub fn as_inner(&self) -> &Spec {
        &self.spec
    }
}

impl<Spec> Blueprint<Spec>
where
    Spec: Clone,
{
    /// Sequences this blueprint with another blueprint.
    ///
    /// The resulting spec is [`SeqSpec`], which composes the output shape of
    /// the left stage into the input shape of the right stage.
    pub fn then<Next>(self, next: Blueprint<Next>) -> Blueprint<SeqSpec<Spec, Next>> {
        Blueprint::new(SeqSpec {
            left: self.spec,
            right: next.spec,
        })
    }
}

/// A materialized single-output model produced from a rooted blueprint.
///
/// `INPUT` and `OUTPUT` are the flattened element counts of the input and
/// output shapes, respectively.
pub struct Model<const INPUT: usize, const OUTPUT: usize> {
    inner: Box<dyn TrainRuntime<INPUT, OUTPUT>>,
}

impl<const INPUT: usize, const OUTPUT: usize> Model<INPUT, OUTPUT> {
    pub(crate) fn new(inner: Box<dyn TrainRuntime<INPUT, OUTPUT>>) -> Self {
        Self { inner }
    }

    /// Predicts the model output for one statically sized input.
    pub fn predict(&self, input: &[Float; INPUT]) -> [Float; OUTPUT] {
        self.inner.predict(input)
    }

    /// Trains with mean-squared error and returns the average training loss.
    pub fn fit(&mut self, samples: &[Sample<INPUT, OUTPUT>], config: TrainConfig) -> Float {
        self.fit_with_loss(samples, &MeanSquaredError, config)
    }

    /// Trains with an explicit loss function and returns the average loss.
    pub fn fit_with_loss(
        &mut self,
        samples: &[Sample<INPUT, OUTPUT>],
        loss_fn: &dyn LossFunction<OUTPUT>,
        config: TrainConfig,
    ) -> Float {
        self.inner.fit_with_loss(samples, loss_fn, config)
    }
}

impl<const INPUT: usize, const OUTPUT: usize> fmt::Debug for Model<INPUT, OUTPUT> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Model")
            .field("input", &INPUT)
            .field("output", &OUTPUT)
            .finish()
    }
}

/// A materialized model with named, potentially multi-head outputs.
#[derive(Debug)]
pub struct HeadedModel<const INPUT: usize, Output> {
    inner: Box<dyn PredictRuntime<INPUT, Output>>,
}

impl<const INPUT: usize, Output> HeadedModel<INPUT, Output> {
    pub(crate) fn new(inner: Box<dyn PredictRuntime<INPUT, Output>>) -> Self {
        Self { inner }
    }

    /// Predicts the named output value for one statically sized input.
    pub fn predict(&self, input: &[Float; INPUT]) -> Output {
        self.inner.predict(input)
    }
}

/// Runtime prediction interface used by materialized headed models.
pub trait PredictRuntime<const INPUT: usize, Output>: fmt::Debug {
    fn predict(&self, input: &[Float; INPUT]) -> Output;
}

/// Runtime training interface used by materialized single-output models.
pub trait TrainRuntime<const INPUT: usize, const OUTPUT: usize>:
    PredictRuntime<INPUT, [Float; OUTPUT]>
{
    fn fit_with_loss(
        &mut self,
        samples: &[Sample<INPUT, OUTPUT>],
        loss_fn: &dyn LossFunction<OUTPUT>,
        config: TrainConfig,
    ) -> Float;
}
