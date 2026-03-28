//! Rooted blueprints and inspection helpers.

use crate::shape::TensorShape;
use std::{collections::HashSet, marker::PhantomData};

use super::{
    Axis, Blueprint, BlueprintSpec, GraphRuntime, HeadedModel, HeadsSpec, InitConfig,
    MaterializeContext, Model, TransformSpec,
};

/// A blueprint specification paired with an explicit input shape and axis names.
///
/// [`Rooted`] is the bridge between an abstract reusable [`Blueprint`] and a
/// fully validated architecture for a specific input shape.
#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct Rooted<InputShape, Spec>
where
    InputShape: TensorShape + 'static,
{
    pub(crate) spec: Spec,
    pub(crate) axis_names: Box<[Axis]>,
    pub(crate) _shape: PhantomData<InputShape>,
}

/// A [`Blueprint`] that has been rooted to a specific input shape.
pub type RootedBlueprint<InputShape, Spec> = Blueprint<Rooted<InputShape, Spec>>;

/// Attaches an explicit input shape and axis naming scheme to a [`Blueprint`].
///
/// Once rooted, the blueprint can be validated, summarized, shape-traced, and
/// materialized.
pub fn root<InputShape, Spec>(
    spec: Blueprint<Spec>,
    axis_names: Vec<Axis>,
) -> RootedBlueprint<InputShape, Spec>
where
    InputShape: TensorShape + 'static,
{
    Blueprint::new(Rooted {
        spec: spec.into_inner(),
        axis_names: resolved_axis_names::<InputShape>(&axis_names),
        _shape: PhantomData,
    })
}

/// Roots a blueprint as a flat feature-vector model.
pub fn features_input<const N: usize, Spec>(
    spec: Blueprint<Spec>,
) -> RootedBlueprint<crate::shape!(features: N), Spec> {
    root::<crate::shape!(features: N), _>(spec, vec![Axis::FEATURES])
}

/// Roots a blueprint as a labeled image input `(channels, height, width)`.
pub fn image_input<const C: usize, const H: usize, const W: usize, Spec>(
    spec: Blueprint<Spec>,
) -> RootedBlueprint<crate::shape!(channels: C, height: H, width: W), Spec> {
    root::<crate::shape!(channels: C, height: H, width: W), _>(
        spec,
        vec![Axis::CHANNELS, Axis::HEIGHT, Axis::WIDTH],
    )
}

/// Roots a blueprint as a labeled volume input
/// `(channels, depth, height, width)`.
pub fn volume_input<const C: usize, const D: usize, const H: usize, const W: usize, Spec>(
    spec: Blueprint<Spec>,
) -> RootedBlueprint<crate::shape!(channels: C, depth: D, height: H, width: W), Spec> {
    root::<crate::shape!(channels: C, depth: D, height: H, width: W), _>(
        spec,
        vec![Axis::CHANNELS, Axis::DEPTH, Axis::HEIGHT, Axis::WIDTH],
    )
}

#[doc(hidden)]
pub fn validate_blueprint<InputShape, Spec>(
    blueprint: RootedBlueprint<InputShape, Spec>,
) -> RootedBlueprint<InputShape, Spec>
where
    InputShape: TensorShape + 'static,
    Spec: TransformSpec<InputShape>,
{
    blueprint
}

#[doc(hidden)]
pub fn validate_headed_blueprint<InputShape, Spec>(
    blueprint: RootedBlueprint<InputShape, Spec>,
) -> RootedBlueprint<InputShape, Spec>
where
    InputShape: TensorShape + 'static,
    Spec: HeadsSpec<InputShape>,
{
    blueprint
}

impl<InputShape, Spec> Blueprint<Rooted<InputShape, Spec>>
where
    InputShape: TensorShape + 'static,
    Spec: BlueprintSpec<InputShape>,
{
    /// Returns a human-readable summary of the rooted blueprint.
    pub fn summary(&self) -> String {
        let rooted = self.as_inner();
        let mut lines = vec![format!(
            "input {}",
            describe_shape::<InputShape>(&rooted.axis_names)
        )];
        rooted.spec.push_summary(&mut lines);
        lines.join("\n")
    }

    /// Returns a human-readable per-stage shape trace.
    pub fn shape_trace(&self) -> String {
        let rooted = self.as_inner();
        let mut lines = Vec::new();
        rooted.spec.push_shape_trace(&rooted.axis_names, &mut lines);
        lines.join("\n")
    }

    /// Counts the distinct trainable parameters in the blueprint.
    pub fn parameter_count(&self) -> usize {
        let rooted = self.as_inner();
        let mut seen = HashSet::new();
        rooted.spec.parameter_count(&mut seen)
    }
}

impl<InputShape, Spec> Blueprint<Rooted<InputShape, Spec>>
where
    InputShape: TensorShape + 'static,
    Spec: TransformSpec<InputShape>,
    Spec::Runtime: GraphRuntime + 'static,
{
    /// Materializes the rooted blueprint into a trainable single-output model.
    pub fn materialize(
        &self,
        config: InitConfig,
    ) -> Model<{ InputShape::SIZE }, { Spec::OUTPUT_SIZE }> {
        let rooted = self.as_inner();
        let mut ctx = MaterializeContext::new(config);
        let runtime = rooted.spec.materialize(&mut ctx);
        Model::new(Box::new(super::GraphRunner::new(runtime)))
    }
}

impl<InputShape, Spec> Blueprint<Rooted<InputShape, Spec>>
where
    InputShape: TensorShape + 'static,
    Spec: HeadsSpec<InputShape>,
    Spec::Runtime: super::PredictRuntime<{ InputShape::SIZE }, Spec::Output> + 'static,
{
    /// Materializes the rooted blueprint into a multi-head prediction model.
    pub fn materialize_heads(
        &self,
        config: InitConfig,
    ) -> HeadedModel<{ InputShape::SIZE }, Spec::Output> {
        let rooted = self.as_inner();
        let mut ctx = MaterializeContext::new(config);
        let runtime = rooted.spec.materialize_heads(&mut ctx);
        HeadedModel::new(Box::new(runtime))
    }

    /// Alias for [`RootedBlueprint::summary`] retained for head-focused call
    /// sites.
    pub fn heads_summary(&self) -> String {
        self.summary()
    }

    /// Alias for [`RootedBlueprint::shape_trace`] retained for head-focused
    /// call sites.
    pub fn heads_shape_trace(&self) -> String {
        self.shape_trace()
    }
}

pub(crate) fn describe_shape<Shape: TensorShape>(axes: &[Axis]) -> String {
    let dims = Shape::dims();
    let labels = resolved_axis_names::<Shape>(axes);
    if dims.len() == labels.len() {
        let parts = labels
            .iter()
            .zip(dims.iter())
            .map(|(axis, dim)| format!("{}: {}", axis.as_str(), dim))
            .collect::<Vec<_>>();
        format!("({})", parts.join(", "))
    } else {
        format!("{dims:?}")
    }
}

pub(crate) fn features_axis() -> Box<[Axis]> {
    vec![Axis::FEATURES].into_boxed_slice()
}

pub(crate) fn resolved_axis_names<Shape: TensorShape>(fallback: &[Axis]) -> Box<[Axis]> {
    let shape_names = Shape::axis_names();
    if shape_names.is_empty() {
        return fallback.to_vec().into_boxed_slice();
    }

    shape_names
        .into_iter()
        .enumerate()
        .map(|(idx, name)| match name {
            Some(name) => Axis::new(name),
            None => fallback.get(idx).copied().unwrap_or(Axis::new("axis")),
        })
        .collect::<Vec<_>>()
        .into_boxed_slice()
}
