//! Graph structure and graph-construction APIs.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::Float;

use super::{EvalTape, Op, ReverseTape};

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

/// Node in the computation graph.
#[derive(Debug, Clone)]
pub enum Node {
    Input(String),
    Const(Float),
    AfterOperation(Op, Box<[NodeId]>),
    Output(NodeId),
}

/// Expression graph with optimized performance.
///
/// Forward evaluation is pure; reuse an [`EvalTape`] to cache intermediates
/// explicitly.
#[derive(Debug)]
pub struct ExprGraph {
    pub(super) graph_id: u64,
    pub(super) nodes: Vec<Node>,
    node_map: HashMap<String, NodeId>,
    pub(super) inputs: Vec<NodeId>,
    pub(super) input_names: Vec<String>,
    pub(super) outputs: Vec<NodeId>,
    pub(super) max_arity: usize,
    next_id: usize,
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

    pub(super) fn make_node_id(&self, index: usize) -> NodeId {
        NodeId::new(index, self.graph_id)
    }

    pub(super) fn is_valid_node(&self, id: NodeId) -> bool {
        id.graph_id == self.graph_id && id.index < self.next_id
    }

    pub(super) fn assert_valid_node(&self, id: NodeId, context: &str) {
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
}

impl Default for ExprGraph {
    fn default() -> Self {
        Self::new()
    }
}
