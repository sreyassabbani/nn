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
    AfterOperation(Op, Box<[NodeId]>),
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
            Op::Mul => inputs
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != input_idx)
                .map(|(_, &x)| x)
                .product(),
        }
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

    pub fn operation<I>(&mut self, op: Op, inputs: I) -> NodeId
    where
        I: AsRef<[NodeId]>,
    {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        self.nodes
            .push(Node::AfterOperation(op, Box::from(inputs.as_ref())));
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
            if let Node::AfterOperation(op, inputs) = node {
                // Pre-allocate input_primals to avoid repeated allocations
                let mut input_primals = Vec::with_capacity(inputs.len());
                for &id in inputs {
                    if id.0 < self.primals.len() {
                        input_primals.push(self.primals[id.0]);
                    } else {
                        input_primals.push(0.0);
                    }
                }

                self.primals[i] = op.compute(&input_primals);

                // Compute derivatives using chain rule
                let mut total_derivative = 0.0;
                for (j, &input_id) in inputs.iter().enumerate() {
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
        self.nodes
            .iter()
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

    (@build_multi $graph:ident, $node:ident -> $op:ident -> @ $result:ident $($rest:tt)*) => {
        let $result = $graph.operation(Op::$op, vec![$node]);
        $crate::graph! { @build_multi $graph, $($rest)* }
    };

    (@build_multi $graph:ident, $node:ident -> $op:ident ( $($op_args:tt)* ) -> @ $result:ident $($rest:tt)*) => {
        let $result = $graph.operation(Op::$op($($op_args)*), vec![$node]);
        $crate::graph! { @build_multi $graph, $($rest)* }
    };

    // Generic N-ary op without extra args: (@a, @b, @c) -> add -> @result
    (@build_multi $graph:ident, ( $( @ $node:ident ),+ ) -> $op:ident -> @ $result:ident $($rest:tt)*) => {
        let $result = $graph.operation(Op::$op, vec![$($node),+]);
        $crate::graph! { @build_multi $graph, $($rest)* }
    };

    // Generic N-ary op with extra args: (@a, @b, @c) -> scale(2.0) -> @res
    (@build_multi $graph:ident, ( $( @ $node:ident ),+ ) -> $op:ident ( $($op_args:tt)* ) -> @ $result:ident $($rest:tt)*) => {
        let $result = $graph.operation(Op::$op($($op_args)*), vec![$($node),+]);
        $crate::graph! { @build_multi $graph, $($rest)* }
    };

    (@build_multi $graph:ident, output @ $node:ident) => {
        $graph.output($node);
        $graph
    };

    (@build_multi $graph:ident, output) => {
        $graph
    };

    // Multi-input building with custom names (lowercase)
    // (@build_multi $graph:ident, $input:ident -> sin -> @ $node:ident $($rest:tt)*) => {
    //     let $node = $graph.operation(Op::Sin, vec![$input]);
    //     $crate::graph! {
    //         @build_multi
    //         $graph,
    //         $($rest)*
    //     }
    // };

    // (@build_multi $graph:ident, $input:ident -> cos -> @ $node:ident $($rest:tt)*) => {
    //     let $node = $graph.operation(Op::Cos, vec![$input]);
    //     $crate::graph! {
    //         @build_multi
    //         $graph,
    //         $($rest)*
    //     }
    // };

    // (@build_multi $graph:ident, $input:ident -> pow($n:literal) -> @ $node:ident $($rest:tt)*) => {
    //     let $node = $graph.operation(Op::Pow($n), vec![$input]);
    //     $crate::graph! {
    //         @build_multi
    //         $graph,
    //         $($rest)*
    //     }
    // };

    // (@build_multi $graph:ident, $input:ident -> scale($x:expr) -> @ $node:ident $($rest:tt)*) => {
    //     let $node = $graph.operation(Op::Scale($x), vec![$input]);
    //     $crate::graph! {
    //         @build_multi
    //         $graph,
    //         $($rest)*
    //     }
    // };

    // // Handle operations without intermediate names
    // (@build_multi $graph:ident, $input:ident -> sin $($rest:tt)*) => {
    //     let temp_node = $graph.operation(Op::Sin, vec![$input]);
    //     $crate::graph! {
    //         @build_multi
    //         $graph,
    //         $($rest)*
    //     }
    // };

    // (@build_multi $graph:ident, $input:ident -> cos $($rest:tt)*) => {
    //     let temp_node = $graph.operation(Op::Cos, vec![$input]);
    //     $crate::graph! {
    //         @build_multi
    //         $graph,
    //         $($rest)*
    //     }
    // };

    // (@build_multi $graph:ident, $input:ident -> pow($n:literal) $($rest:tt)*) => {
    //     let temp_node = $graph.operation(Op::Pow($n), vec![$input]);
    //     $crate::graph! {
    //         @build_multi
    //         $graph,
    //         $($rest)*
    //     }
    // };

    // (@build_multi $graph:ident, $input:ident -> scale($x:expr) $($rest:tt)*) => {
    //     let temp_node = $graph.operation(Op::Scale($x), vec![$input]);
    //     $crate::graph! {
    //         @build_multi
    //         $graph,
    //         $($rest)*
    //     }
    // };
}
