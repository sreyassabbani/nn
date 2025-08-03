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
}
