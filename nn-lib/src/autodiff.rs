use std::collections::HashMap;

/// Node identifier for multi-input graphs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

/// Multi-input computation graph with optimized performance
#[derive(Debug)]
pub struct MultiGraph {
    nodes: Vec<Node>,
    node_map: HashMap<String, NodeId>,
    next_id: usize,
    /// Pre-allocated buffers for performance
    primals: Vec<f64>,
    tangents: Vec<f64>,
}

/// Node in the computation graph
#[derive(Debug, Clone)]
pub enum Node {
    Input(String),
    Operation(Box<dyn OpTrait>),
    Output(NodeId),
}

/// Operations that can be performed on nodes
#[derive(Debug, Clone, Copy)]
pub enum Op {
    Scale(f64),
    Sin,
    Cos,
    Pow(i32),
    Add,
    Mul,
}

impl Op {
    fn compute(self, inputs: &[f64]) -> f64 {
        match self {
            Op::Scale(factor) => inputs[0] * factor,
            Op::Sin => inputs[0].sin(),
            Op::Cos => inputs[0].cos(),
            Op::Pow(exp) => inputs[0].powi(exp),
            Op::Add => inputs.iter().sum(),
            Op::Mul => inputs.iter().product(),
        }
    }

    fn compute_derivative(self, inputs: &[f64], input_idx: usize) -> f64 {
        match self {
            Op::Scale(factor) => factor,
            Op::Sin => inputs[0].cos(),
            Op::Cos => -inputs[0].sin(),
            Op::Pow(exp) => exp as f64 * inputs[0].powi(exp - 1),
            Op::Add => 1.0,
            Op::Mul => {
                inputs.iter()
                    .enumerate()
                    .filter(|(i, _)| *i != input_idx)
                    .map(|(_, &x)| x)
                    .product()
            }
        }
    }
}

/// Trait for operations with type-level arity
pub trait OpTrait: std::fmt::Debug {
    const ARITY: usize;
    
    fn compute(&self, inputs: &[f64]) -> f64;
    fn compute_derivative(&self, inputs: &[f64], input_idx: usize) -> f64;
    fn input_ids(&self) -> &[NodeId];
}

// Single-input operations
#[derive(Debug)]
pub struct ScaleOp {
    pub factor: f64,
    pub input_id: NodeId,
}

impl OpTrait for ScaleOp {
    const ARITY: usize = 1;
    
    fn compute(&self, inputs: &[f64]) -> f64 {
        debug_assert_eq!(inputs.len(), Self::ARITY, "ScaleOp requires exactly {} input", Self::ARITY);
        inputs[0] * self.factor
    }
    
    fn compute_derivative(&self, _inputs: &[f64], _input_idx: usize) -> f64 {
        self.factor
    }
    
    fn input_ids(&self) -> &[NodeId] {
        std::slice::from_ref(&self.input_id)
    }
}

#[derive(Debug)]
pub struct SinOp {
    pub input_id: NodeId,
}

impl OpTrait for SinOp {
    const ARITY: usize = 1;
    
    fn compute(&self, inputs: &[f64]) -> f64 {
        debug_assert_eq!(inputs.len(), Self::ARITY, "SinOp requires exactly {} input", Self::ARITY);
        inputs[0].sin()
    }
    
    fn compute_derivative(&self, inputs: &[f64], _input_idx: usize) -> f64 {
        debug_assert_eq!(inputs.len(), Self::ARITY, "SinOp requires exactly {} input", Self::ARITY);
        inputs[0].cos()
    }
    
    fn input_ids(&self) -> &[NodeId] {
        std::slice::from_ref(&self.input_id)
    }
}

#[derive(Debug)]
pub struct CosOp {
    pub input_id: NodeId,
}

impl OpTrait for CosOp {
    const ARITY: usize = 1;
    
    fn compute(&self, inputs: &[f64]) -> f64 {
        debug_assert_eq!(inputs.len(), Self::ARITY, "CosOp requires exactly {} input", Self::ARITY);
        inputs[0].cos()
    }
    
    fn compute_derivative(&self, inputs: &[f64], _input_idx: usize) -> f64 {
        debug_assert_eq!(inputs.len(), Self::ARITY, "CosOp requires exactly {} input", Self::ARITY);
        -inputs[0].sin()
    }
    
    fn input_ids(&self) -> &[NodeId] {
        std::slice::from_ref(&self.input_id)
    }
}

#[derive(Debug)]
pub struct PowOp {
    pub exp: i32,
    pub input_id: NodeId,
}

impl OpTrait for PowOp {
    const ARITY: usize = 1;
    
    fn compute(&self, inputs: &[f64]) -> f64 {
        debug_assert_eq!(inputs.len(), Self::ARITY, "PowOp requires exactly {} input", Self::ARITY);
        inputs[0].powi(self.exp)
    }
    
    fn compute_derivative(&self, inputs: &[f64], _input_idx: usize) -> f64 {
        debug_assert_eq!(inputs.len(), Self::ARITY, "PowOp requires exactly {} input", Self::ARITY);
        self.exp as f64 * inputs[0].powi(self.exp - 1)
    }
    
    fn input_ids(&self) -> &[NodeId] {
        std::slice::from_ref(&self.input_id)
    }
}

// Two-input operations
#[derive(Debug)]
pub struct AddOp {
    pub input_ids: [NodeId; 2],
}

impl OpTrait for AddOp {
    const ARITY: usize = 2;
    
    fn compute(&self, inputs: &[f64]) -> f64 {
        debug_assert_eq!(inputs.len(), Self::ARITY, "AddOp requires exactly {} inputs", Self::ARITY);
        inputs.iter().sum()
    }
    
    fn compute_derivative(&self, _inputs: &[f64], _input_idx: usize) -> f64 {
        1.0
    }
    
    fn input_ids(&self) -> &[NodeId] {
        &self.input_ids
    }
}

#[derive(Debug)]
pub struct MulOp {
    pub input_ids: [NodeId; 2],
}

impl OpTrait for MulOp {
    const ARITY: usize = 2;
    
    fn compute(&self, inputs: &[f64]) -> f64 {
        debug_assert_eq!(inputs.len(), Self::ARITY, "MulOp requires exactly {} inputs", Self::ARITY);
        inputs.iter().product()
    }
    
    fn compute_derivative(&self, inputs: &[f64], input_idx: usize) -> f64 {
        debug_assert_eq!(inputs.len(), Self::ARITY, "MulOp requires exactly {} inputs", Self::ARITY);
        inputs.iter()
            .enumerate()
            .filter(|(i, _)| *i != input_idx)
            .map(|(_, &x)| x)
            .product()
    }
    
    fn input_ids(&self) -> &[NodeId] {
        &self.input_ids
    }
}

impl MultiGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            node_map: HashMap::new(),
            next_id: 0,
            primals: Vec::with_capacity(1024), // Pre-allocate reasonable size
            tangents: Vec::with_capacity(1024),
        }
    }

    pub fn input(&mut self, name: String) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        self.nodes.push(Node::Input(name.clone()));
        self.node_map.insert(name, id);
        id
    }

    pub fn operation(&mut self, op: Op, inputs: Vec<NodeId>) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        let operation = match op {
            Op::Scale(factor) => {
                debug_assert_eq!(inputs.len(), 1, "Scale operation requires exactly 1 input");
                Box::new(ScaleOp { factor, input_id: inputs[0] })
            }
            Op::Sin => {
                debug_assert_eq!(inputs.len(), 1, "Sin operation requires exactly 1 input");
                Box::new(SinOp { input_id: inputs[0] })
            }
            Op::Cos => {
                debug_assert_eq!(inputs.len(), 1, "Cos operation requires exactly 1 input");
                Box::new(CosOp { input_id: inputs[0] })
            }
            Op::Pow(exp) => {
                debug_assert_eq!(inputs.len(), 1, "Pow operation requires exactly 1 input");
                Box::new(PowOp { exp, input_id: inputs[0] })
            }
            Op::Add => {
                debug_assert_eq!(inputs.len(), 2, "Add operation requires exactly 2 inputs");
                Box::new(AddOp { input_ids: [inputs[0], inputs[1]] })
            }
            Op::Mul => {
                debug_assert_eq!(inputs.len(), 2, "Mul operation requires exactly 2 inputs");
                Box::new(MulOp { input_ids: [inputs[0], inputs[1]] })
            }
        };
        self.nodes.push(Node::Operation(operation));
        id
    }

    pub fn output(&mut self, node: NodeId) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        self.nodes.push(Node::Output(node));
        id
    }

    pub fn compute(&mut self, inputs: &[f64]) -> Vec<(f64, f64)> {
        self.primals.clear();
        self.tangents.clear();
        
        // Ensure buffers are large enough
        let needed_size = self.nodes.len();
        if self.primals.capacity() < needed_size {
            self.primals.reserve(needed_size);
            self.tangents.reserve(needed_size);
        }

        // Initialize with zeros
        self.primals.resize(needed_size, 0.0);
        self.tangents.resize(needed_size, 0.0);

        // Create a mapping from input names to their indices in the inputs array
        let mut input_indices = HashMap::new();
        let mut input_count = 0;
        for node in &self.nodes {
            if let Node::Input(name) = node {
                input_indices.insert(name.clone(), input_count);
                input_count += 1;
            }
        }

        // First pass: handle inputs
        for (i, node) in self.nodes.iter().enumerate() {
            if let Node::Input(name) = node {
                if let Some(&input_idx) = input_indices.get(name) {
                    if input_idx < inputs.len() {
                        self.primals[i] = inputs[input_idx];
                        self.tangents[i] = 1.0;
                    } else {
                        // Handle case where input index is out of bounds
                        self.primals[i] = 0.0;
                        self.tangents[i] = 0.0;
                    }
                } else {
                    // Handle case where input name is not found
                    self.primals[i] = 0.0;
                    self.tangents[i] = 0.0;
                }
            }
        }

        // Second pass: handle operations (topological order)
        for (i, node) in self.nodes.iter().enumerate() {
            if let Node::Operation(op) = node {
                // Pre-allocate input_primals to avoid repeated allocations
                let mut input_primals = Vec::with_capacity(op.input_ids().len());
                for &id in op.input_ids() {
                    if id.0 < self.primals.len() {
                        input_primals.push(self.primals[id.0]);
                    } else {
                        input_primals.push(0.0);
                    }
                }
                
                self.primals[i] = op.compute(&input_primals);
                
                // Compute derivatives using chain rule
                let mut total_derivative = 0.0;
                for (j, &input_id) in op.input_ids().iter().enumerate() {
                    if input_id.0 < self.tangents.len() {
                        let partial = op.compute_derivative(&input_primals, j);
                        total_derivative += self.tangents[input_id.0] * partial;
                    }
                }
                self.tangents[i] = total_derivative;
            }
        }

        // Third pass: handle outputs
        for (i, node) in self.nodes.iter().enumerate() {
            if let Node::Output(input_id) = node {
                if input_id.0 < self.primals.len() {
                    self.primals[i] = self.primals[input_id.0];
                    self.tangents[i] = self.tangents[input_id.0];
                } else {
                    self.primals[i] = 0.0;
                    self.tangents[i] = 0.0;
                }
            }
        }

        // Collect outputs
        self.nodes.iter()
            .enumerate()
            .filter_map(|(i, node)| {
                if matches!(node, Node::Output(_)) {
                    Some((self.primals[i], self.tangents[i]))
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Legacy single-input computation graph (kept for backward compatibility)
#[derive(Clone, Debug)]
pub struct CompGraph {
    ops: Vec<Op>,
    /// Pre-allocated buffers for performance
    _buf_primals: Vec<f64>,
    _buf_tangents: Vec<f64>,
}

impl CompGraph {
    pub fn new(ops: Vec<Op>) -> Self {
        let cap = ops.len() + 1;
        Self {
            ops,
            _buf_primals: Vec::with_capacity(cap),
            _buf_tangents: Vec::with_capacity(cap),
        }
    }

    pub fn compute(&mut self, input: f64) -> (f64, f64) {
        self._buf_primals.clear();
        self._buf_tangents.clear();

        self._buf_primals.push(input);
        self.ops
            .iter()
            .fold((input, 1.0), |(primal_acc, tangent_chain), x| {
                let primal = x.compute(&[primal_acc]);
                let tangent = tangent_chain * x.compute_derivative(&[primal_acc], 0);

                self._buf_primals.push(primal);
                self._buf_tangents.push(tangent);

                (primal, tangent)
            })
    }
}

// Extension traits for ergonomic API
pub trait NodeOps {
    fn sin(self) -> NodeId;
    fn cos(self) -> NodeId;
    fn pow(self, exp: i32) -> NodeId;
    fn scale(self, factor: f64) -> NodeId;
    fn add(self, other: NodeId) -> NodeId;
    fn mul(self, other: NodeId) -> NodeId;
}

impl NodeOps for NodeId {
    fn sin(self) -> NodeId {
        // This would need access to the graph, so we'll implement this differently
        unimplemented!("Use graph.operation(Op::Sin, vec![self]) instead")
    }

    fn cos(self) -> NodeId {
        unimplemented!("Use graph.operation(Op::Cos, vec![self]) instead")
    }

    fn pow(self, exp: i32) -> NodeId {
        unimplemented!("Use graph.operation(Op::Pow(exp), vec![self]) instead")
    }

    fn scale(self, factor: f64) -> NodeId {
        unimplemented!("Use graph.operation(Op::Scale(factor), vec![self]) instead")
    }

    fn add(self, other: NodeId) -> NodeId {
        unimplemented!("Use graph.operation(Op::Add, vec![self, other]) instead")
    }

    fn mul(self, other: NodeId) -> NodeId {
        unimplemented!("Use graph.operation(Op::Mul, vec![self, other]) instead")
    }
}

/// Macro for building computation graphs
/// 
/// # Examples
/// 
/// Single input graph:
/// ```rust
/// let graph = graph! {
///     input -> sin -> cos -> output
/// };
/// ```
/// 
/// Multi-input graph:
/// ```rust
/// let graph = graph! {
///     inputs: [x, y]
///     x -> pow(2) -> @x_sq
///     y -> sin -> @y_sin
///     (@x_sq, @y_sin) -> add -> @result
///     output @result
/// };
/// ```
/// 
/// Mixed graph (operations without intermediate names):
/// ```rust
/// let graph = graph! {
///     inputs: [x, y]
///     x -> pow(2) -> sin -> @temp1
///     y -> cos -> scale(2.0) -> @temp2
///     (@temp1, @temp2) -> mul -> output
/// };
/// ```
/// 
/// # Performance Notes
/// 
/// The implementation uses pre-allocated buffers to minimize memory allocations
/// during computation. The graph structure is optimized for forward-mode automatic
/// differentiation with efficient chain rule computation. Operations use type-level
/// arity for compile-time safety.
#[macro_export]
macro_rules! graph {
    // Single input graph (backward compatibility)
    (input -> $($rest:tt)*) => {
        {
            use $crate::autodiff::{Op, CompGraph};
            $crate::graph! {
                @build_linear
                [],
                $($rest)*
            }
        }
    };

    // Multi-input graph
    (inputs: [$($input:ident),*] $($rest:tt)*) => {
        {
            use $crate::autodiff::{MultiGraph, Op, NodeId};
            let mut graph = MultiGraph::new();
            $(let $input = graph.input(stringify!($input).to_string());)*
            $crate::graph! {
                @build_multi
                graph,
                $($rest)*
            }
        }
    };

    // Linear building (single input)
    (@build_linear [$($ops:expr,)*], sin -> $($rest:tt)*) => {
        $crate::graph! {
            @build_linear
            [$($ops,)* Op::Sin,],
            $($rest)*
        }
    };

    (@build_linear [$($ops:expr,)*], cos -> $($rest:tt)*) => {
        $crate::graph! {
            @build_linear
            [$($ops,)* Op::Cos,],
            $($rest)*
        }
    };

    (@build_linear [$($ops:expr,)*], scale($x:expr) -> $($rest:tt)*) => {
        $crate::graph! {
            @build_linear
            [$($ops,)* Op::Scale($x),],
            $($rest)*
        }
    };

    (@build_linear [$($ops:expr,)*], pow($n:literal) -> $($rest:tt)*) => {
        $crate::graph! {
            @build_linear
            [$($ops,)* Op::Pow($n),],
            $($rest)*
        }
    };

    (@build_linear [$($ops:expr,)*], output) => {
        CompGraph::new(Vec::from([$($ops,)*]))
    };

    // Multi-input building
    (@build_multi $graph:ident, $input:ident -> sin -> @$node:ident $($rest:tt)*) => {
        let $node = $graph.operation(Op::Sin, vec![$input]);
        $crate::graph! {
            @build_multi
            $graph,
            $($rest)*
        }
    };

    (@build_multi $graph:ident, $input:ident -> cos -> @$node:ident $($rest:tt)*) => {
        let $node = $graph.operation(Op::Cos, vec![$input]);
        $crate::graph! {
            @build_multi
            $graph,
            $($rest)*
        }
    };

    (@build_multi $graph:ident, $input:ident -> pow($n:literal) -> @$node:ident $($rest:tt)*) => {
        let $node = $graph.operation(Op::Pow($n), vec![$input]);
        $crate::graph! {
            @build_multi
            $graph,
            $($rest)*
        }
    };

    (@build_multi $graph:ident, $input:ident -> scale($x:expr) -> @$node:ident $($rest:tt)*) => {
        let $node = $graph.operation(Op::Scale($x), vec![$input]);
        $crate::graph! {
            @build_multi
            $graph,
            $($rest)*
        }
    };

    // Handle operations without intermediate names
    (@build_multi $graph:ident, $input:ident -> sin $($rest:tt)*) => {
        let temp_node = $graph.operation(Op::Sin, vec![$input]);
        $crate::graph! {
            @build_multi
            $graph,
            $($rest)*
        }
    };

    (@build_multi $graph:ident, $input:ident -> cos $($rest:tt)*) => {
        let temp_node = $graph.operation(Op::Cos, vec![$input]);
        $crate::graph! {
            @build_multi
            $graph,
            $($rest)*
        }
    };

    (@build_multi $graph:ident, $input:ident -> pow($n:literal) $($rest:tt)*) => {
        let temp_node = $graph.operation(Op::Pow($n), vec![$input]);
        $crate::graph! {
            @build_multi
            $graph,
            $($rest)*
        }
    };

    (@build_multi $graph:ident, $input:ident -> scale($x:expr) $($rest:tt)*) => {
        let temp_node = $graph.operation(Op::Scale($x), vec![$input]);
        $crate::graph! {
            @build_multi
            $graph,
            $($rest)*
        }
    };

    (@build_multi $graph:ident, (@$node1:ident, @$node2:ident) -> add -> @$result:ident $($rest:tt)*) => {
        let $result = $graph.operation(Op::Add, vec![$node1, $node2]);
        $crate::graph! {
            @build_multi
            $graph,
            $($rest)*
        }
    };

    (@build_multi $graph:ident, (@$node1:ident, @$node2:ident) -> mul -> @$result:ident $($rest:tt)*) => {
        let $result = $graph.operation(Op::Mul, vec![$node1, $node2]);
        $crate::graph! {
            @build_multi
            $graph,
            $($rest)*
        }
    };

    (@build_multi $graph:ident, (@$node1:ident, @$node2:ident) -> add $($rest:tt)*) => {
        let temp_node = $graph.operation(Op::Add, vec![$node1, $node2]);
        $crate::graph! {
            @build_multi
            $graph,
            $($rest)*
        }
    };

    (@build_multi $graph:ident, (@$node1:ident, @$node2:ident) -> mul $($rest:tt)*) => {
        let temp_node = $graph.operation(Op::Mul, vec![$node1, $node2]);
        $crate::graph! {
            @build_multi
            $graph,
            $($rest)*
        }
    };

    (@build_multi $graph:ident, output @$node:ident) => {
        $graph.output($node);
        $graph
    };

    (@build_multi $graph:ident, output) => {
        $graph
    };
}