//! Reusable workspaces for forward- and reverse-mode graph evaluation.

use crate::Float;

/// Workspace that stores intermediate primals and gradient vectors during
/// forward-mode evaluation.
///
/// Reuse it across calls to avoid repeated allocations when performance
/// matters.
#[derive(Debug, Default)]
pub struct EvalTape {
    pub(super) primals: Vec<Float>,
    pub(super) tangents: Vec<Float>,
    pub(super) input_count: usize,
    pub(super) scratch_primals: Vec<Float>,
    pub(super) scratch_partials: Vec<Float>,
}

impl EvalTape {
    /// Creates an empty forward-mode tape.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a tape pre-sized for a known graph.
    pub fn with_capacity(nodes: usize, input_count: usize, max_arity: usize) -> Self {
        Self {
            primals: Vec::with_capacity(nodes),
            tangents: Vec::with_capacity(nodes * input_count),
            input_count,
            scratch_primals: Vec::with_capacity(max_arity),
            scratch_partials: Vec::with_capacity(max_arity),
        }
    }

    pub(super) fn reset(&mut self, nodes: usize, input_count: usize, max_arity: usize) {
        self.input_count = input_count;
        self.primals.clear();
        self.tangents.clear();
        self.primals.resize(nodes, 0.0);
        self.tangents.resize(nodes * input_count, 0.0);
        self.scratch_primals.clear();
        self.scratch_partials.clear();
        self.scratch_primals.resize(max_arity, 0.0);
        self.scratch_partials.resize(max_arity, 0.0);
    }

    pub(super) fn tangent_index(&self, node_idx: usize, input_idx: usize) -> usize {
        node_idx * self.input_count + input_idx
    }
}

/// Workspace that stores intermediate primals and adjoints during reverse-mode
/// evaluation.
#[derive(Debug, Default)]
pub struct ReverseTape {
    pub(super) primals: Vec<Float>,
    pub(super) adjoints: Vec<Float>,
    pub(super) scratch_primals: Vec<Float>,
    pub(super) scratch_partials: Vec<Float>,
}

impl ReverseTape {
    /// Creates an empty reverse-mode tape.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a tape pre-sized for a known graph.
    pub fn with_capacity(nodes: usize, max_arity: usize) -> Self {
        Self {
            primals: Vec::with_capacity(nodes),
            adjoints: Vec::with_capacity(nodes),
            scratch_primals: Vec::with_capacity(max_arity),
            scratch_partials: Vec::with_capacity(max_arity),
        }
    }

    pub(super) fn reset(&mut self, nodes: usize, max_arity: usize) {
        self.primals.clear();
        self.adjoints.clear();
        self.primals.resize(nodes, 0.0);
        self.adjoints.resize(nodes, 0.0);
        self.scratch_primals.clear();
        self.scratch_partials.clear();
        self.scratch_primals.resize(max_arity, 0.0);
        self.scratch_partials.resize(max_arity, 0.0);
    }
}
