//! Materialized sequential models produced by the legacy builders.

use crate::{Float, Sample};
use std::fmt;

use super::super::{LossFunction, MeanSquaredError, TrainConfig};
use super::runtime;

/// A classic trainable sequential model.
///
/// [`Sequential`] is the output of the older builder-style API:
///
/// ```rust
/// # #![feature(generic_const_exprs, adt_const_params, unsized_const_params)]
/// # #![allow(incomplete_features)]
/// use tml_utils::Sample;
/// use tml_utils::network::{ModelBuilder, TrainConfig};
///
/// let mut model = ModelBuilder::new()
///     .input::<1>()
///     .dense::<8>()
///     .relu()
///     .dense::<1>()
///     .build();
///
/// let samples = [Sample::new([0.0], [0.0]), Sample::new([1.0], [1.0])];
/// let _ = model.fit(&samples, TrainConfig::adam(0.01).epochs(1));
/// ```
///
/// The input and output widths are part of the type, so prediction signatures
/// remain statically sized.
pub struct Sequential<const INPUT: usize, const OUTPUT: usize> {
    inner: Box<dyn runtime::ModelRuntime<INPUT, OUTPUT>>,
}

impl<const INPUT: usize, const OUTPUT: usize> fmt::Debug for Sequential<INPUT, OUTPUT> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sequential")
            .field("input", &INPUT)
            .field("output", &OUTPUT)
            .finish()
    }
}

impl<const INPUT: usize, const OUTPUT: usize> Sequential<INPUT, OUTPUT> {
    pub(super) fn from_runtime<R>(runtime: R) -> Self
    where
        R: runtime::ModelRuntime<INPUT, OUTPUT> + 'static,
    {
        Self {
            inner: Box::new(runtime),
        }
    }

    /// Predicts the model output for one statically sized input.
    pub fn predict(&self, input: &[Float; INPUT]) -> [Float; OUTPUT] {
        self.inner.predict(input)
    }

    /// Alias for [`Sequential::predict`], retained for compatibility.
    pub fn predict_in_place(&self, input: &[Float; INPUT]) -> [Float; OUTPUT] {
        self.predict(input)
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
