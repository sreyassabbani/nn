use std::{cell::RefCell, rc::Rc};

use crate::Float;

use super::graph::{ExprGraph, NodeId, Op};

/// Named gradients for a single output.
#[derive(Debug, Clone)]
pub struct Gradients {
    /// The output value.
    pub value: Float,
    /// Per-input gradients paired with their names.
    pub grads: Vec<(String, Float)>,
}

impl Gradients {
    /// Looks up a gradient by input name.
    pub fn get(&self, name: &str) -> Option<Float> {
        self.grads
            .iter()
            .find_map(|(key, value)| (key == name).then_some(*value))
    }
}

/// Errors that can occur when mutating [`Tape`] inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TapeError {
    InputLengthMismatch { expected: usize, got: usize },
    UnknownInput(String),
}

impl std::fmt::Display for TapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputLengthMismatch { expected, got } => {
                write!(f, "expected {expected} inputs, got {got}")
            }
            Self::UnknownInput(name) => write!(f, "unknown input name: {name}"),
        }
    }
}

impl std::error::Error for TapeError {}

/// Rust-like autodiff tape with operator overloading.
#[derive(Debug, Clone)]
pub struct Tape {
    inner: Rc<RefCell<TapeInner>>,
}

#[derive(Debug)]
struct TapeInner {
    graph: ExprGraph,
    values: Vec<Float>,
}

/// A node handle tied to a [`Tape`].
#[derive(Debug, Clone)]
pub struct Var {
    id: NodeId,
    inner: Rc<RefCell<TapeInner>>,
}

impl Tape {
    /// Creates a new empty tape.
    pub fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(TapeInner {
                graph: ExprGraph::new(),
                values: Vec::new(),
            })),
        }
    }

    /// Creates a named differentiable input.
    pub fn input(&mut self, name: impl Into<String>, value: Float) -> Var {
        let mut inner = self.inner.borrow_mut();
        let id = inner.graph.input(name.into());
        inner.values.push(value);
        Var {
            id,
            inner: self.inner.clone(),
        }
    }

    /// Creates an unnamed input using an auto-generated name.
    pub fn input_unnamed(&mut self, value: Float) -> Var {
        let idx = self.inner.borrow().values.len();
        self.input(format!("_{}", idx), value)
    }

    /// Creates a constant node on the same tape.
    pub fn constant(&mut self, value: Float) -> Var {
        let mut inner = self.inner.borrow_mut();
        let id = inner.graph.constant(value);
        Var {
            id,
            inner: self.inner.clone(),
        }
    }

    /// Overwrites all input values in declaration order.
    pub fn set_inputs(&mut self, values: &[Float]) {
        self.try_set_inputs(values)
            .expect("input length mismatch for Tape::set_inputs");
    }

    /// Tries to overwrite all input values in declaration order.
    pub fn try_set_inputs(&mut self, values: &[Float]) -> Result<(), TapeError> {
        let mut inner = self.inner.borrow_mut();
        let expected = inner.values.len();
        if values.len() != expected {
            return Err(TapeError::InputLengthMismatch {
                expected,
                got: values.len(),
            });
        }
        inner.values.copy_from_slice(values);
        Ok(())
    }

    /// Sets a named input.
    pub fn set(&mut self, name: &str, value: Float) {
        self.try_set(name, value)
            .expect("unknown input name for Tape::set");
    }

    /// Tries to set a named input.
    pub fn try_set(&mut self, name: &str, value: Float) -> Result<(), TapeError> {
        let mut inner = self.inner.borrow_mut();
        let Some(idx) = inner.graph.input_names().iter().position(|n| n == name) else {
            return Err(TapeError::UnknownInput(name.to_string()));
        };
        inner.values[idx] = value;
        Ok(())
    }

    /// Returns the current input names in declaration order.
    pub fn input_names(&self) -> Vec<String> {
        self.inner.borrow().graph.input_names().to_vec()
    }

    /// Computes named gradients for a single output variable.
    pub fn gradients(&self, output: &Var) -> Gradients {
        output.assert_same_tape(self);
        let inner = self.inner.borrow();
        let results = inner.graph.eval_named_for(&inner.values, &[output.id]);
        let (value, grads) = results.into_iter().next().expect("missing output");
        Gradients { value, grads }
    }

    /// Computes named gradients for several output variables.
    pub fn gradients_for(&self, outputs: &[Var]) -> Vec<Gradients> {
        if outputs.is_empty() {
            return Vec::new();
        }
        outputs[0].assert_same_tape(self);
        for var in outputs.iter().skip(1) {
            var.assert_same_tape(self);
        }

        let inner = self.inner.borrow();
        let ids = outputs.iter().map(|var| var.id).collect::<Vec<_>>();
        inner
            .graph
            .eval_named_for(&inner.values, &ids)
            .into_iter()
            .map(|(value, grads)| Gradients { value, grads })
            .collect()
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

impl Var {
    fn assert_same_tape(&self, tape: &Tape) {
        assert!(
            Rc::ptr_eq(&self.inner, &tape.inner),
            "cannot mix Vars from different tapes"
        );
    }

    fn assert_same_var_tape(&self, other: &Var) {
        assert!(
            Rc::ptr_eq(&self.inner, &other.inner),
            "cannot mix Vars from different tapes"
        );
    }

    fn unary_op(&self, op: Op) -> Var {
        let mut inner = self.inner.borrow_mut();
        let id = inner.graph.operation(op, vec![self.id]);
        Var {
            id,
            inner: self.inner.clone(),
        }
    }

    fn binary_op(&self, rhs: &Var, op: Op) -> Var {
        self.assert_same_var_tape(rhs);
        let mut inner = self.inner.borrow_mut();
        let id = inner.graph.operation(op, vec![self.id, rhs.id]);
        Var {
            id,
            inner: self.inner.clone(),
        }
    }

    fn konst(&self, value: Float) -> Var {
        let mut inner = self.inner.borrow_mut();
        let id = inner.graph.constant(value);
        Var {
            id,
            inner: self.inner.clone(),
        }
    }

    /// Applies `sin` to this variable.
    pub fn sin(&self) -> Var {
        self.unary_op(Op::Sin)
    }

    /// Applies `cos` to this variable.
    pub fn cos(&self) -> Var {
        self.unary_op(Op::Cos)
    }

    /// Raises this variable to an integer power.
    pub fn powi(&self, exp: i32) -> Var {
        self.unary_op(Op::Pow(exp))
    }

    /// Scales this variable by a constant.
    pub fn scale(&self, factor: Float) -> Var {
        self.unary_op(Op::Scale(factor))
    }
}

impl std::ops::Add for Var {
    type Output = Var;
    fn add(self, rhs: Var) -> Self::Output {
        self.binary_op(&rhs, Op::Add)
    }
}

impl std::ops::Add<Float> for Var {
    type Output = Var;
    fn add(self, rhs: Float) -> Self::Output {
        let rhs = self.konst(rhs);
        self.binary_op(&rhs, Op::Add)
    }
}

impl std::ops::Sub for Var {
    type Output = Var;
    fn sub(self, rhs: Var) -> Self::Output {
        self + (-rhs)
    }
}

impl std::ops::Sub<Float> for Var {
    type Output = Var;
    fn sub(self, rhs: Float) -> Self::Output {
        self + (-rhs)
    }
}

impl std::ops::Mul for Var {
    type Output = Var;
    fn mul(self, rhs: Var) -> Self::Output {
        self.binary_op(&rhs, Op::Mul)
    }
}

impl std::ops::Mul<Float> for Var {
    type Output = Var;
    fn mul(self, rhs: Float) -> Self::Output {
        self.scale(rhs)
    }
}

impl std::ops::Div for Var {
    type Output = Var;
    fn div(self, rhs: Var) -> Self::Output {
        self * rhs.powi(-1)
    }
}

impl std::ops::Div<Float> for Var {
    type Output = Var;
    fn div(self, rhs: Float) -> Self::Output {
        self.scale(1.0 / rhs)
    }
}

impl std::ops::Neg for Var {
    type Output = Var;
    fn neg(self) -> Self::Output {
        self.scale(-1.0)
    }
}
