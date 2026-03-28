//! Flat/vector builder stages for the legacy sequential API.

use crate::network::{DenseLayer, ReLU, Sigmoid};
use std::fmt;

use super::Sequential;
use super::runtime::{self, AppendLayer};

/// A sequential builder whose current activation is a flat feature vector.
///
/// `INPUT` is the original input width. `CURRENT` is the width after all
/// appended stages so far.
pub struct VectorBuilder<Layers, const INPUT: usize, const CURRENT: usize> {
    pub(super) layers: Layers,
}

impl<Layers, const INPUT: usize, const CURRENT: usize> fmt::Debug
    for VectorBuilder<Layers, INPUT, CURRENT>
where
    Layers: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VectorBuilder")
            .field("input", &INPUT)
            .field("current", &CURRENT)
            .finish()
    }
}

impl<Layers, const INPUT: usize, const CURRENT: usize> VectorBuilder<Layers, INPUT, CURRENT> {
    /// Keeps the builder flat. This is a no-op retained for parity with
    /// [`super::ImageBuilder::flatten`].
    pub const fn flatten(self) -> Self {
        self
    }

    /// Appends a dense layer and updates the current feature width.
    pub fn dense<const NEXT: usize>(
        self,
    ) -> VectorBuilder<<Layers as AppendLayer<DenseLayer<CURRENT, NEXT>, NEXT>>::Output, INPUT, NEXT>
    where
        Layers: AppendLayer<DenseLayer<CURRENT, NEXT>, NEXT>,
    {
        VectorBuilder {
            layers: self.layers.then(DenseLayer::<CURRENT, NEXT>::init()),
        }
    }

    /// Appends a ReLU activation.
    pub fn relu(
        self,
    ) -> VectorBuilder<<Layers as AppendLayer<ReLU<CURRENT>, CURRENT>>::Output, INPUT, CURRENT>
    where
        Layers: AppendLayer<ReLU<CURRENT>, CURRENT>,
    {
        VectorBuilder {
            layers: self.layers.then(ReLU::<CURRENT>::init()),
        }
    }

    /// Appends a sigmoid activation.
    pub fn sigmoid(
        self,
    ) -> VectorBuilder<<Layers as AppendLayer<Sigmoid<CURRENT>, CURRENT>>::Output, INPUT, CURRENT>
    where
        Layers: AppendLayer<Sigmoid<CURRENT>, CURRENT>,
    {
        VectorBuilder {
            layers: self.layers.then(Sigmoid::<CURRENT>::init()),
        }
    }

    /// Materializes the builder into a trainable [`Sequential`] model.
    pub fn build(self) -> Sequential<INPUT, CURRENT>
    where
        Layers: runtime::ModuleChain<INPUT, CURRENT> + fmt::Debug + 'static,
    {
        Sequential::from_runtime(runtime::Stack::new(self.layers))
    }
}
