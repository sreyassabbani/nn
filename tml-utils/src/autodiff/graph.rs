use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::Float;

/// Node identifier for expression graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId {
    pub(super) index: usize,
    graph_id: u64,
}

impl NodeId {
    fn new(index: usize, graph_id: u64) -> Self {
        Self { index, graph_id }
    }
}

static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(1);

/// Expression graph with optimized performance.
///
/// Forward evaluation is pure; reuse an [`EvalTape`] to cache intermediates
/// explicitly.
#[derive(Debug)]
pub struct ExprGraph {
    graph_id: u64,
    nodes: Vec<Node>,
    node_map: HashMap<String, NodeId>,
    inputs: Vec<NodeId>,
    pub(super) input_names: Vec<String>,
    outputs: Vec<NodeId>,
    max_arity: usize,
    next_id: usize,
}

/// Node in the computation graph.
#[derive(Debug, Clone)]
pub enum Node {
    Input(String),
    Const(Float),
    AfterOperation(Op, Box<[NodeId]>),
    Output(NodeId),
}

/// Scalar operations supported by [`ExprGraph`].
#[derive(Debug, Clone, Copy)]
pub enum Op {
    Scale(Float),
    Sin,
    Cos,
    Pow(i32),
    Add,
    Mul,
}

/// Workspace that stores intermediate primals and gradient vectors during
/// forward-mode evaluation.
///
/// Reuse it across calls to avoid repeated allocations when performance
/// matters.
#[derive(Debug, Default)]
pub struct EvalTape {
    primals: Vec<Float>,
    tangents: Vec<Float>,
    input_count: usize,
    scratch_primals: Vec<Float>,
    scratch_partials: Vec<Float>,
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

    fn reset(&mut self, nodes: usize, input_count: usize, max_arity: usize) {
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

    fn tangent_index(&self, node_idx: usize, input_idx: usize) -> usize {
        node_idx * self.input_count + input_idx
    }
}

/// Workspace that stores intermediate primals and adjoints during reverse-mode
/// evaluation.
#[derive(Debug, Default)]
pub struct ReverseTape {
    primals: Vec<Float>,
    adjoints: Vec<Float>,
    scratch_primals: Vec<Float>,
    scratch_partials: Vec<Float>,
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

    fn reset(&mut self, nodes: usize, max_arity: usize) {
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

impl Op {
    fn validate_arity(self, inputs_len: usize) {
        let ok = match self {
            Op::Scale(_) | Op::Sin | Op::Cos | Op::Pow(_) => inputs_len == 1,
            Op::Add | Op::Mul => inputs_len >= 2,
        };

        assert!(
            ok,
            "invalid arity for {:?}: expected {}, got {}",
            self,
            match self {
                Op::Scale(_) | Op::Sin | Op::Cos | Op::Pow(_) => "1",
                Op::Add | Op::Mul => ">= 2",
            },
            inputs_len
        );
    }

    pub(super) fn apply(self, inputs: &[Float]) -> Float {
        match self {
            Op::Scale(factor) => inputs[0] * factor,
            Op::Sin => inputs[0].sin(),
            Op::Cos => inputs[0].cos(),
            Op::Pow(exp) => inputs[0].powi(exp),
            Op::Add => inputs.iter().sum(),
            Op::Mul => inputs.iter().product(),
        }
    }

    pub(super) fn compute_derivative(self, inputs: &[Float], input_idx: usize) -> Float {
        match self {
            Op::Scale(factor) => factor,
            Op::Sin => inputs[0].cos(),
            Op::Cos => -inputs[0].sin(),
            Op::Pow(exp) => {
                if exp == 0 {
                    0.0
                } else {
                    exp as Float * inputs[0].powi(exp - 1)
                }
            }
            Op::Add => 1.0,
            Op::Mul => inputs
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != input_idx)
                .map(|(_, &x)| x)
                .product(),
        }
    }
}

impl ExprGraph {
    /// Creates an empty expression graph.
    pub fn new() -> Self {
        Self {
            graph_id: NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed),
            nodes: Vec::new(),
            node_map: HashMap::new(),
            inputs: Vec::new(),
            input_names: Vec::new(),
            outputs: Vec::new(),
            max_arity: 0,
            next_id: 0,
        }
    }

    fn make_node_id(&self, index: usize) -> NodeId {
        NodeId::new(index, self.graph_id)
    }

    fn is_valid_node(&self, id: NodeId) -> bool {
        id.graph_id == self.graph_id && id.index < self.next_id
    }

    fn assert_valid_node(&self, id: NodeId, context: &str) {
        assert!(
            self.is_valid_node(id),
            "{context} does not belong to this graph or is out of bounds"
        );
    }

    /// Adds a named input node.
    pub fn input(&mut self, name: String) -> NodeId {
        assert!(
            !self.node_map.contains_key(&name),
            "input name already exists: {name}"
        );

        let id = self.make_node_id(self.next_id);
        self.next_id += 1;
        self.nodes.push(Node::Input(name.clone()));
        self.node_map.insert(name.clone(), id);
        self.inputs.push(id);
        self.input_names.push(name);
        id
    }

    /// Adds a constant node.
    pub fn constant(&mut self, value: Float) -> NodeId {
        let id = self.make_node_id(self.next_id);
        self.next_id += 1;
        self.nodes.push(Node::Const(value));
        id
    }

    /// Adds an operation node whose inputs must already exist in the graph.
    pub fn operation<I>(&mut self, op: Op, inputs: I) -> NodeId
    where
        I: AsRef<[NodeId]>,
    {
        let inputs_ref = inputs.as_ref();
        op.validate_arity(inputs_ref.len());
        assert!(
            inputs_ref.iter().all(|id| self.is_valid_node(*id)),
            "operation inputs must reference earlier nodes in the same graph"
        );
        self.max_arity = self.max_arity.max(inputs_ref.len());
        let id = self.make_node_id(self.next_id);
        self.next_id += 1;
        self.nodes
            .push(Node::AfterOperation(op, Box::from(inputs_ref)));
        id
    }

    /// Marks a node as an output of the graph.
    pub fn output(&mut self, node: NodeId) -> NodeId {
        self.assert_valid_node(node, "output node");
        let id = self.make_node_id(self.next_id);
        self.next_id += 1;
        self.nodes.push(Node::Output(node));
        self.outputs.push(id);
        id
    }

    /// Allocates a forward-mode tape sized for this graph.
    pub fn fwd_tape(&self) -> EvalTape {
        EvalTape::with_capacity(self.nodes.len(), self.inputs.len(), self.max_arity)
    }

    /// Allocates a reverse-mode tape sized for this graph.
    pub fn tape(&self) -> ReverseTape {
        self.reverse_tape()
    }

    /// Allocates a reverse-mode tape sized for this graph.
    pub fn reverse_tape(&self) -> ReverseTape {
        ReverseTape::with_capacity(self.nodes.len(), self.max_arity)
    }

    /// Returns the graph's input names in declaration order.
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Pure forward evaluation that allocates its own tape.
    pub fn eval_fwd(&self, inputs: &[Float]) -> Vec<(Float, Vec<Float>)> {
        let mut tape = self.fwd_tape();
        self.eval_fwd_with_tape(inputs, &mut tape)
    }

    /// Forward evaluation that reuses the provided tape.
    pub fn eval_fwd_with_tape(
        &self,
        inputs: &[Float],
        tape: &mut EvalTape,
    ) -> Vec<(Float, Vec<Float>)> {
        assert_eq!(
            inputs.len(),
            self.inputs.len(),
            "expected {} inputs, got {}",
            self.inputs.len(),
            inputs.len()
        );

        tape.reset(self.nodes.len(), self.inputs.len(), self.max_arity);

        for (input_idx, node_id) in self.inputs.iter().enumerate() {
            let node_idx = node_id.index;
            tape.primals[node_idx] = inputs[input_idx];
            let tangent_idx = tape.tangent_index(node_idx, input_idx);
            tape.tangents[tangent_idx] = 1.0;
        }

        for (i, node) in self.nodes.iter().enumerate() {
            match node {
                Node::AfterOperation(op, inputs) => {
                    let arity = inputs.len();
                    let input_primals = &mut tape.scratch_primals[..arity];
                    for (slot, &id) in input_primals.iter_mut().zip(inputs.iter()) {
                        *slot = tape.primals[id.index];
                    }

                    tape.primals[i] = op.apply(input_primals);

                    let partials = &mut tape.scratch_partials[..arity];
                    for (j, partial) in partials.iter_mut().enumerate() {
                        *partial = op.compute_derivative(input_primals, j);
                    }

                    let input_count = tape.input_count;
                    let tangents = &mut tape.tangents;
                    for input_dim in 0..input_count {
                        let mut total = 0.0;
                        for (j, &input_id) in inputs.iter().enumerate() {
                            let idx = input_id.index * input_count + input_dim;
                            total += tangents[idx] * partials[j];
                        }
                        let out_idx = i * input_count + input_dim;
                        tangents[out_idx] = total;
                    }
                }
                Node::Const(value) => {
                    tape.primals[i] = *value;
                }
                _ => {}
            }
        }

        for (i, node) in self.nodes.iter().enumerate() {
            if let Node::Output(input_id) = node {
                tape.primals[i] = tape.primals[input_id.index];
                let src_start = tape.tangent_index(input_id.index, 0);
                let dst_start = tape.tangent_index(i, 0);
                let len = tape.input_count;
                tape.tangents
                    .copy_within(src_start..(src_start + len), dst_start);
            }
        }

        self.outputs
            .iter()
            .map(|id| {
                let idx = id.index;
                let start = tape.tangent_index(idx, 0);
                let end = start + tape.input_count;
                (tape.primals[idx], tape.tangents[start..end].to_vec())
            })
            .collect()
    }

    /// Forward evaluation specialized to a single output.
    pub fn eval_fwd_one(&self, inputs: &[Float]) -> (Float, Vec<Float>) {
        let mut tape = self.fwd_tape();
        self.eval_fwd_one_with_tape(inputs, &mut tape)
    }

    /// Single-output forward evaluation with a reusable tape.
    pub fn eval_fwd_one_with_tape(
        &self,
        inputs: &[Float],
        tape: &mut EvalTape,
    ) -> (Float, Vec<Float>) {
        let mut outputs = self.eval_fwd_with_tape(inputs, tape);
        assert!(
            outputs.len() == 1,
            "expected a single output, got {}",
            outputs.len()
        );
        outputs.remove(0)
    }

    /// Forward evaluation with gradients returned as `(name, grad)` pairs.
    pub fn eval_fwd_named(&self, inputs: &[Float]) -> Vec<(Float, Vec<(String, Float)>)> {
        let mut tape = self.fwd_tape();
        self.eval_fwd_named_with_tape(inputs, &mut tape)
    }

    /// Named forward evaluation with a reusable tape.
    pub fn eval_fwd_named_with_tape(
        &self,
        inputs: &[Float],
        tape: &mut EvalTape,
    ) -> Vec<(Float, Vec<(String, Float)>)> {
        let outputs = self.eval_fwd_with_tape(inputs, tape);
        outputs
            .into_iter()
            .map(|(value, grads)| {
                let named = self
                    .input_names
                    .iter()
                    .cloned()
                    .zip(grads)
                    .collect::<Vec<_>>();
                (value, named)
            })
            .collect()
    }

    /// Reverse-mode evaluation that allocates its own tape.
    pub fn eval(&self, inputs: &[Float]) -> Vec<(Float, Vec<Float>)> {
        let mut tape = self.reverse_tape();
        self.eval_with_tape(inputs, &mut tape)
    }

    /// Reverse-mode evaluation that reuses the provided tape.
    pub fn eval_with_tape(
        &self,
        inputs: &[Float],
        tape: &mut ReverseTape,
    ) -> Vec<(Float, Vec<Float>)> {
        self.eval_for_with_tape(inputs, &self.outputs, tape)
    }

    /// Reverse-mode evaluation for a selected set of outputs.
    pub fn eval_for(&self, inputs: &[Float], outputs: &[NodeId]) -> Vec<(Float, Vec<Float>)> {
        let mut tape = self.reverse_tape();
        self.eval_for_with_tape(inputs, outputs, &mut tape)
    }

    /// Reverse-mode evaluation for selected outputs with a reusable tape.
    pub fn eval_for_with_tape(
        &self,
        inputs: &[Float],
        outputs: &[NodeId],
        tape: &mut ReverseTape,
    ) -> Vec<(Float, Vec<Float>)> {
        assert_eq!(
            inputs.len(),
            self.inputs.len(),
            "expected {} inputs, got {}",
            self.inputs.len(),
            inputs.len()
        );
        for &output in outputs {
            self.assert_valid_node(output, "requested output");
        }

        tape.reset(self.nodes.len(), self.max_arity);

        for (input_idx, node_id) in self.inputs.iter().enumerate() {
            tape.primals[node_id.index] = inputs[input_idx];
        }

        for (i, node) in self.nodes.iter().enumerate() {
            match node {
                Node::AfterOperation(op, inputs) => {
                    let arity = inputs.len();
                    let input_primals = &mut tape.scratch_primals[..arity];
                    for (slot, &id) in input_primals.iter_mut().zip(inputs.iter()) {
                        *slot = tape.primals[id.index];
                    }
                    tape.primals[i] = op.apply(input_primals);
                }
                Node::Output(input_id) => {
                    tape.primals[i] = tape.primals[input_id.index];
                }
                Node::Const(value) => {
                    tape.primals[i] = *value;
                }
                Node::Input(_) => {}
            }
        }

        let mut results = Vec::with_capacity(outputs.len());

        for output_id in outputs {
            tape.adjoints.fill(0.0);
            tape.adjoints[output_id.index] = 1.0;

            for (i, node) in self.nodes.iter().enumerate().rev() {
                match node {
                    Node::Output(input_id) => {
                        tape.adjoints[input_id.index] += tape.adjoints[i];
                    }
                    Node::AfterOperation(op, inputs) => {
                        let arity = inputs.len();
                        let input_primals = &mut tape.scratch_primals[..arity];
                        for (slot, &id) in input_primals.iter_mut().zip(inputs.iter()) {
                            *slot = tape.primals[id.index];
                        }

                        let partials = &mut tape.scratch_partials[..arity];
                        for (j, partial) in partials.iter_mut().enumerate() {
                            *partial = op.compute_derivative(input_primals, j);
                        }

                        let adj = tape.adjoints[i];
                        if adj != 0.0 {
                            for (j, &input_id) in inputs.iter().enumerate() {
                                tape.adjoints[input_id.index] += adj * partials[j];
                            }
                        }
                    }
                    Node::Const(_) | Node::Input(_) => {}
                }
            }

            let grads = self
                .inputs
                .iter()
                .map(|id| tape.adjoints[id.index])
                .collect::<Vec<_>>();
            results.push((tape.primals[output_id.index], grads));
        }

        results
    }

    /// Single-output reverse-mode evaluation.
    pub fn eval_one(&self, inputs: &[Float]) -> (Float, Vec<Float>) {
        let mut tape = self.reverse_tape();
        self.eval_one_with_tape(inputs, &mut tape)
    }

    /// Single-output reverse-mode evaluation with a reusable tape.
    pub fn eval_one_with_tape(
        &self,
        inputs: &[Float],
        tape: &mut ReverseTape,
    ) -> (Float, Vec<Float>) {
        let mut outputs = self.eval_with_tape(inputs, tape);
        assert!(
            outputs.len() == 1,
            "expected a single output, got {}",
            outputs.len()
        );
        outputs.remove(0)
    }

    /// Reverse-mode evaluation with named gradients.
    pub fn eval_named(&self, inputs: &[Float]) -> Vec<(Float, Vec<(String, Float)>)> {
        let mut tape = self.reverse_tape();
        self.eval_named_with_tape(inputs, &mut tape)
    }

    /// Named reverse-mode evaluation with a reusable tape.
    pub fn eval_named_with_tape(
        &self,
        inputs: &[Float],
        tape: &mut ReverseTape,
    ) -> Vec<(Float, Vec<(String, Float)>)> {
        let outputs = self.eval_with_tape(inputs, tape);
        outputs
            .into_iter()
            .map(|(value, grads)| {
                let named = self
                    .input_names
                    .iter()
                    .cloned()
                    .zip(grads)
                    .collect::<Vec<_>>();
                (value, named)
            })
            .collect()
    }

    /// Reverse-mode evaluation for selected outputs with named gradients.
    pub fn eval_named_for(
        &self,
        inputs: &[Float],
        outputs: &[NodeId],
    ) -> Vec<(Float, Vec<(String, Float)>)> {
        let mut tape = self.reverse_tape();
        self.eval_named_for_with_tape(inputs, outputs, &mut tape)
    }

    /// Selected-output named reverse-mode evaluation with a reusable tape.
    pub fn eval_named_for_with_tape(
        &self,
        inputs: &[Float],
        outputs: &[NodeId],
        tape: &mut ReverseTape,
    ) -> Vec<(Float, Vec<(String, Float)>)> {
        let outputs = self.eval_for_with_tape(inputs, outputs, tape);
        outputs
            .into_iter()
            .map(|(value, grads)| {
                let named = self
                    .input_names
                    .iter()
                    .cloned()
                    .zip(grads)
                    .collect::<Vec<_>>();
                (value, named)
            })
            .collect()
    }
}

impl Default for ExprGraph {
    fn default() -> Self {
        Self::new()
    }
}
