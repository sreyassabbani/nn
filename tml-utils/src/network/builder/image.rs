//! Image-specific builder stages for the legacy sequential API.

use crate::__private::ConvGeometryIsValid;
use crate::conv::{Conv, conv_out_dim};
use crate::network::{ReLU, Sigmoid};
use std::fmt;

use super::runtime::{self, AppendLayer};
use super::{Sequential, VectorBuilder};

/// A sequential builder whose current activation is interpreted as image-like
/// `C x H x W` data.
///
/// `INPUT` is the flattened size of the original input. `C`, `H`, and `W`
/// track the current image geometry through convolution stages.
pub struct ImageBuilder<Layers, const INPUT: usize, const C: usize, const H: usize, const W: usize>
{
    pub(super) layers: Layers,
}

impl<Layers, const INPUT: usize, const C: usize, const H: usize, const W: usize> fmt::Debug
    for ImageBuilder<Layers, INPUT, C, H, W>
where
    Layers: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageBuilder")
            .field("input", &INPUT)
            .field("channels", &C)
            .field("height", &H)
            .field("width", &W)
            .finish()
    }
}

impl<Layers, const INPUT: usize, const C: usize, const H: usize, const W: usize>
    ImageBuilder<Layers, INPUT, C, H, W>
where
    [(); INPUT]:,
{
    /// Appends a ReLU activation over the flattened current image state.
    pub fn relu(
        self,
    ) -> ImageBuilder<
        <Layers as AppendLayer<ReLU<{ C * H * W }>, { C * H * W }>>::Output,
        INPUT,
        C,
        H,
        W,
    >
    where
        [(); C * H * W]:,
        Layers: AppendLayer<ReLU<{ C * H * W }>, { C * H * W }>,
    {
        ImageBuilder {
            layers: self.layers.then(ReLU::<{ C * H * W }>::init()),
        }
    }

    /// Appends a sigmoid activation over the flattened current image state.
    pub fn sigmoid(
        self,
    ) -> ImageBuilder<
        <Layers as AppendLayer<Sigmoid<{ C * H * W }>, { C * H * W }>>::Output,
        INPUT,
        C,
        H,
        W,
    >
    where
        [(); C * H * W]:,
        Layers: AppendLayer<Sigmoid<{ C * H * W }>, { C * H * W }>,
    {
        ImageBuilder {
            layers: self.layers.then(Sigmoid::<{ C * H * W }>::init()),
        }
    }

    /// Appends a 2D convolution and updates the tracked image geometry.
    pub fn conv<const OC: usize, const FH: usize, const FW: usize, const S: usize, const P: usize>(
        self,
    ) -> ImageBuilder<
        <Layers as AppendLayer<
            Conv<W, H, C, FH, FW, OC, S, P>,
            { OC * conv_out_dim(H, P, FH, S) * conv_out_dim(W, P, FW, S) },
        >>::Output,
        INPUT,
        OC,
        { conv_out_dim(H, P, FH, S) },
        { conv_out_dim(W, P, FW, S) },
    >
    where
        [(); C * H * W]:,
        [(); OC * conv_out_dim(H, P, FH, S) * conv_out_dim(W, P, FW, S)]:,
        (): ConvGeometryIsValid<H, W, FH, FW, S, P>,
        Layers: AppendLayer<
                Conv<W, H, C, FH, FW, OC, S, P>,
                { OC * conv_out_dim(H, P, FH, S) * conv_out_dim(W, P, FW, S) },
            >,
    {
        ImageBuilder {
            layers: self.layers.then(Conv::<W, H, C, FH, FW, OC, S, P>::init()),
        }
    }

    /// Reinterprets the current image activation as a flat vector builder.
    pub fn flatten(self) -> VectorBuilder<Layers, INPUT, { C * H * W }>
    where
        [(); C * H * W]:,
    {
        VectorBuilder {
            layers: self.layers,
        }
    }

    /// Materializes the current image pipeline into a trainable
    /// [`Sequential`] model.
    pub fn build(self) -> Sequential<INPUT, { C * H * W }>
    where
        [(); C * H * W]:,
        Layers: runtime::ModuleChain<INPUT, { C * H * W }> + fmt::Debug + 'static,
    {
        Sequential::from_runtime(runtime::Stack::new(self.layers))
    }
}
