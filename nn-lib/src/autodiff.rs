/// This is our computation graph
pub struct CompGraph {
    ops: Vec<Op>,
    /// Intermediate calculations in finding composite f
    _buf_primals: Vec<f64>,
    /// Intermediate calculations in finding the derivative, f', of the composite f
    _buf_tangents: Vec<f64>,
}

/// Very rudimentary operations that can be combined via the chain rule to make a more complex function
pub enum Op {
    Scale(f64),
    Sin,
    Cos,
    Pow(i32),
}

impl Op {
    fn compute(&self, input: f64) -> f64 {
        match self {
            Op::Scale(factor) => input * factor,
            Op::Sin => input.sin(),
            Op::Cos => input.cos(),
            Op::Pow(exp) => input.powi(*exp),
        }
    }
}

impl CompGraph {
    pub fn new(ops: Vec<Op>) -> Self {
        let cap = ops.len();
        Self {
            ops,
            _buf_primals: Vec::with_capacity(cap),
            _buf_tangents: Vec::with_capacity(cap),
        }
    }

    pub fn compute(&self, input: f64) -> f64 {
        self.ops.iter().fold(input, |acc, x| {
            return x.compute(acc);
        })
    }
}

#[macro_export]
macro_rules! graph {
    (input -> $($rest:tt)*) => {
        {
            let mut _v = Vec::new();

            use $crate::autodiff::{Op, CompGraph};
            $crate::graph! {
                @build
                _v,
                $($rest)*
            }
        }
    };

    (@build $ops:ident, sin -> $($rest:tt)*) => {
        $ops.push(Op::Sin);
        $crate::graph! {
            @build
            $ops,
            $($rest)*
        }
    };

    (@build $ops:ident, cos -> $($rest:tt)*) => {
        $ops.push(Op::Cos);
        $crate::graph! {
            @build
            $ops,
            $($rest)*
        }
    };

    (@build $ops:ident, scale($x:expr) -> $($rest:tt)*) => {
        $ops.push(Op::Scale($x));
        $crate::graph! {
            @build
            $ops,
            $($rest)*
        }
    };

    (@build $ops:ident, pow($n:literal) -> $($rest:tt)*) => {
        $ops.push(Op::Pow($n));
        $crate::graph! {
            @build
            $ops,
            $($rest)*
        }
    };

    (@build $ops:expr, output) => {
        CompGraph::new($ops)
    };
}
