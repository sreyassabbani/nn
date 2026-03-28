//! Entry points for the legacy sequential builders.

use super::runtime::End;
use super::{ImageBuilder, VectorBuilder};

/// Starts the classic builder-based model API.
///
/// [`ModelBuilder`] chooses an input kind and returns a more specialized
/// builder. Use [`ModelBuilder::input`] for flat vectors and
/// [`ModelBuilder::image_input`] for image-like tensors.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelBuilder;

impl ModelBuilder {
    /// Creates a new builder entry point.
    pub const fn new() -> Self {
        Self
    }

    /// Starts a flat vector model with `N` input features.
    pub fn input<const N: usize>(self) -> VectorBuilder<End, N, N> {
        VectorBuilder { layers: End }
    }

    /// Starts an image model with `C x H x W` input geometry.
    pub fn image_input<const C: usize, const H: usize, const W: usize>(
        self,
    ) -> ImageBuilder<End, { C * H * W }, C, H, W>
    where
        [(); C * H * W]:,
    {
        ImageBuilder { layers: End }
    }
}
