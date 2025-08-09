/// This is our computation graph
#[derive(Clone, Debug)]
pub struct CompGraph {
    ops: Vec<Op>,
    /// Intermediate calculations in finding composite f
    _buf_primals: Vec<f64>,
    /// Intermediate calculations in finding the derivative, f', of the composite f
    _buf_tangents: Vec<f64>,
}

/// Very rudimentary operations that can be combined via the chain rule to make a more complex function
#[derive(Copy, Clone, Debug)]
pub enum Op {
    Scale(f64),
    Sin,
    Cos,
    Pow(i32),
}

impl Op {
    fn compute(self, input: f64) -> f64 {
        match self {
            Op::Scale(factor) => input * factor,
            Op::Sin => input.sin(),
            Op::Cos => input.cos(),
            Op::Pow(exp) => input.powi(exp),
        }
    }
    fn compute_derivative(self, input: f64) -> f64 {
        match self {
            Op::Scale(factor) => factor,
            Op::Sin => input.cos(),
            Op::Cos => -input.sin(),
            Op::Pow(exp) => exp as f64 * input.powi(exp - 1),
        }
    }
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
                let primal = x.compute(primal_acc);
                let tangent = tangent_chain * x.compute_derivative(primal_acc);

                // actually inserting at position i+1 due to input
                self._buf_primals.push(primal);
                self._buf_tangents.push(tangent);

                return (primal, tangent);
            })
    }
}

// ------------------------------
// Multi-input forward-mode autodiff
// ------------------------------
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Copy, Clone, Debug)]
pub enum BinaryOp {
    Add,
    Mul,
}

#[derive(Clone, Debug)]
struct GraphInner {
    nodes: Vec<NodeKind>,
    num_inputs: usize,
    input_names: Vec<String>,
}

impl GraphInner {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            num_inputs: 0,
            input_names: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
enum NodeKind {
    Input { name: String, position: usize },
    Unary { op: Op, parent: usize },
    Binary { op: BinaryOp, left: usize, right: usize },
}

#[derive(Clone, Debug)]
pub struct MultiGraph {
    inner: Rc<RefCell<GraphInner>>,
}

impl MultiGraph {
    pub fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(GraphInner::new())),
        }
    }

    pub fn input(&self, name: &str) -> Node {
        let mut inner = self.inner.borrow_mut();
        let id = inner.nodes.len();
        let position = inner.num_inputs;
        inner.num_inputs += 1;
        inner.input_names.push(name.to_string());
        inner
            .nodes
            .push(NodeKind::Input { name: name.to_string(), position });
        Node { inner: Rc::clone(&self.inner), id }
    }

    pub fn output(&self, node: Node) -> MultiGraphExecutable {
        let inner = self.inner.borrow();
        let num_nodes = inner.nodes.len();
        let num_inputs = inner.num_inputs;
        MultiGraphExecutable {
            inner: Rc::clone(&self.inner),
            outputs: vec![node.id],
            _buf_primals: Vec::with_capacity(num_nodes),
            _buf_tangents: Vec::with_capacity(num_nodes * num_inputs),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    inner: Rc<RefCell<GraphInner>>,
    id: usize,
}

impl Node {
    fn unary(&self, op: Op) -> Node {
        let mut inner = self.inner.borrow_mut();
        let id = inner.nodes.len();
        inner.nodes.push(NodeKind::Unary { op, parent: self.id });
        Node { inner: Rc::clone(&self.inner), id }
    }

    fn binary(&self, op: BinaryOp, other: Node) -> Node {
        let mut inner = self.inner.borrow_mut();
        let id = inner.nodes.len();
        inner
            .nodes
            .push(NodeKind::Binary { op, left: self.id, right: other.id });
        Node { inner: Rc::clone(&self.inner), id }
    }

    pub fn sin(&self) -> Node { self.unary(Op::Sin) }
    pub fn cos(&self) -> Node { self.unary(Op::Cos) }
    pub fn scale(&self, factor: f64) -> Node { self.unary(Op::Scale(factor)) }
    pub fn pow(&self, exp: i32) -> Node { self.unary(Op::Pow(exp)) }

    pub fn add(&self, other: Node) -> Node { self.binary(BinaryOp::Add, other) }
    pub fn mul(&self, other: Node) -> Node { self.binary(BinaryOp::Mul, other) }
}

#[derive(Clone, Debug)]
pub struct MultiGraphExecutable {
    inner: Rc<RefCell<GraphInner>>,
    outputs: Vec<usize>,
    _buf_primals: Vec<f64>,
    _buf_tangents: Vec<f64>,
}

impl MultiGraphExecutable {
    pub fn compute(&mut self, inputs: &[f64]) -> (f64, Vec<f64>) {
        let inner = self.inner.borrow();
        assert_eq!(inputs.len(), inner.num_inputs, "expected {} inputs, got {}", inner.num_inputs, inputs.len());

        let num_nodes = inner.nodes.len();
        let num_inputs = inner.num_inputs;

        if self._buf_primals.len() != num_nodes {
            self._buf_primals.resize(num_nodes, 0.0);
        }
        if self._buf_tangents.len() != num_nodes * num_inputs {
            self._buf_tangents.resize(num_nodes * num_inputs, 0.0);
        }

        let primals = &mut self._buf_primals;
        let tangents = &mut self._buf_tangents;

        for (node_index, node) in inner.nodes.iter().enumerate() {
            match *node {
                NodeKind::Input { position, .. } => {
                    primals[node_index] = inputs[position];
                    let base = node_index * num_inputs;
                    for j in 0..num_inputs {
                        tangents[base + j] = if j == position { 1.0 } else { 0.0 };
                    }
                }
                NodeKind::Unary { op, parent } => {
                    let x = primals[parent];
                    let y = op.compute(x);
                    let dy_dx = op.compute_derivative(x);
                    primals[node_index] = y;

                    let base = node_index * num_inputs;
                    let parent_base = parent * num_inputs;
                    for j in 0..num_inputs {
                        tangents[base + j] = tangents[parent_base + j] * dy_dx;
                    }
                }
                NodeKind::Binary { op, left, right } => {
                    let xl = primals[left];
                    let xr = primals[right];
                    let base = node_index * num_inputs;
                    let left_base = left * num_inputs;
                    let right_base = right * num_inputs;

                    match op {
                        BinaryOp::Add => {
                            primals[node_index] = xl + xr;
                            for j in 0..num_inputs {
                                tangents[base + j] = tangents[left_base + j] + tangents[right_base + j];
                            }
                        }
                        BinaryOp::Mul => {
                            primals[node_index] = xl * xr;
                            for j in 0..num_inputs {
                                tangents[base + j] = tangents[left_base + j] * xr + tangents[right_base + j] * xl;
                            }
                        }
                    }
                }
            }
        }

        let out_id = self.outputs[0];
        let primal_out = primals[out_id];
        let mut grad = vec![0.0_f64; num_inputs];
        let base = out_id * num_inputs;
        for j in 0..num_inputs {
            grad[j] = tangents[base + j];
        }
        (primal_out, grad)
    }
}

#[macro_export]
macro_rules! graph {
    (input -> $($rest:tt)*) => {
        {
            use $crate::autodiff::{Op, CompGraph};
            $crate::graph! {
                @build
                [],
                $($rest)*
            }
        }
    };

    // Multi-input entrypoint
    (inputs: [$($input:ident),*] $($rest:tt)*) => {
        {
            use $crate::autodiff::{MultiGraph, Node};
            let graph = MultiGraph::new();
            $(let $input = graph.input(stringify!($input));)*
            $crate::graph! {
                @build_multi
                graph,
                $($rest)*
            }
        }
    };

    // Linear building (existing code)
    (@build [$($ops:expr,)*], sin -> $($rest:tt)*) => {
        $crate::graph! {
            @build
            [$($ops,)* Op::Sin,],
            $($rest)*
        }
    };

    (@build [$($ops:expr,)*], cos -> $($rest:tt)*) => {
        $crate::graph! {
            @build
            [$($ops,)* Op::Cos,],
            $($rest)*
        }
    };

    (@build [$($ops:expr,)*], scale($x:expr) -> $($rest:tt)*) => {
        $crate::graph! {
            @build
            [$($ops,)* Op::Scale($x),],
            $($rest)*
        }
    };

    (@build [$($ops:expr,)*], pow($n:literal) -> $($rest:tt)*) => {
        $crate::graph! {
            @build
            [$($ops,)* Op::Pow($n),],
            $($rest)*
        }
    };

    (@build [$($ops:expr,)*], output) => {
        CompGraph::new(Vec::from([$($ops,)*]))
    };

    // Multi-input building
    // unary op with arg from input ident
    (@build_multi $graph:ident, $start:ident -> $op:ident($arg:expr) -> @$name:ident $($rest:tt)*) => {
        let $name = $start.$op($arg);
        $crate::graph! { @build_multi $graph, $($rest)* }
    };

    // unary op no-arg from input ident
    (@build_multi $graph:ident, $start:ident -> $op:ident -> @$name:ident $($rest:tt)*) => {
        let $name = $start.$op();
        $crate::graph! { @build_multi $graph, $($rest)* }
    };

    // unary op with arg from existing node ident
    (@build_multi $graph:ident, @$start:ident -> $op:ident($arg:expr) -> @$name:ident $($rest:tt)*) => {
        let $name = $start.$op($arg);
        $crate::graph! { @build_multi $graph, $($rest)* }
    };

    // unary op no-arg from existing node ident
    (@build_multi $graph:ident, @$start:ident -> $op:ident -> @$name:ident $($rest:tt)*) => {
        let $name = $start.$op();
        $crate::graph! { @build_multi $graph, $($rest)* }
    };

    // binary op between two nodes
    (@build_multi $graph:ident, (@$left:ident, @$right:ident) -> $op:ident -> @$name:ident $($rest:tt)*) => {
        let $name = $left.$op($right);
        $crate::graph! { @build_multi $graph, $($rest)* }
    };

    // binary op directly to output
    (@build_multi $graph:ident, (@$left:ident, @$right:ident) -> $op:ident -> output) => {
        let __tmp = $left.$op($right);
        $graph.output(__tmp)
    };

    // unary op directly to output from input ident
    (@build_multi $graph:ident, $start:ident -> $op:ident($arg:expr) -> output) => {
        let __tmp = $start.$op($arg);
        $graph.output(__tmp)
    };

    (@build_multi $graph:ident, $start:ident -> $op:ident -> output) => {
        let __tmp = $start.$op();
        $graph.output(__tmp)
    };

    // unary op directly to output from node ident
    (@build_multi $graph:ident, @$start:ident -> $op:ident($arg:expr) -> output) => {
        let __tmp = $start.$op($arg);
        $graph.output(__tmp)
    };

    (@build_multi $graph:ident, @$start:ident -> $op:ident -> output) => {
        let __tmp = $start.$op();
        $graph.output(__tmp)
    };

    // finalize multi graph with named node
    (@build_multi $graph:ident, output @$node:ident) => {
        $graph.output($node)
    };
}
