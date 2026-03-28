/// Macro for building differentiable expressions.
///
/// # Examples
///
/// Single input expression:
/// ```rust,ignore
/// let expr = expr! {
///     input -> Sin -> Cos -> output
/// };
/// ```
///
/// Multi-input expression:
/// ```rust,ignore
/// let expr = expr! {
///     inputs: [x, y]
///     x -> Pow(2) -> @x_sq
///     y -> Sin -> @y_sin
///     (@x_sq, @y_sin) -> Add -> @result
///     output @result
/// };
/// ```
///
/// Mixed expression (operations without intermediate names):
/// ```rust,ignore
/// let expr = expr! {
///     inputs: [x, y]
///     x -> Pow(2) -> @temp1
///     y -> Cos -> @temp2
///     (@temp1, @temp2) -> Mul -> @res
///     output @res
/// };
/// ```
///
/// # Performance Notes
///
/// The default `eval` path allocates a fresh [`ReverseTape`](crate::autodiff::ReverseTape)
/// each call for purity. When you need to reuse buffers, create a tape with
/// `expr.tape()` (or `expr.reverse_tape()`) and call `eval_with_tape` to keep
/// allocations off the hot path. Operation arity is validated at runtime.
#[macro_export]
macro_rules! expr {
    (input -> $($rest:tt)*) => {
        {
            use $crate::autodiff::{ExprGraph, Op};
            let mut graph = ExprGraph::new();
            let __input = graph.input("input".to_string());
            $crate::expr! {
                @build_single
                graph,
                __input,
                $($rest)*
            }
        }
    };

    (inputs: [$($input:ident),*] $($rest:tt)*) => {
        {
            use $crate::autodiff::{ExprGraph, Op};
            let mut graph = ExprGraph::new();
            $(let $input = graph.input(stringify!($input).to_string());)*
            $crate::expr! {
                @build_multi
                graph,
                $($rest)*
            }
        }
    };

    (@build_single $graph:ident, $node:ident, Add -> $($rest:tt)*) => {
        compile_error!("Add is n-ary; use `inputs: [...]` and `(@a, @b, ...) -> Add`");
    };

    (@build_single $graph:ident, $node:ident, Mul -> $($rest:tt)*) => {
        compile_error!("Mul is n-ary; use `inputs: [...]` and `(@a, @b, ...) -> Mul`");
    };

    (@build_single $graph:ident, $node:ident, $op:ident -> $($rest:tt)*) => {
        let __next = $graph.operation(Op::$op, vec![$node]);
        $crate::expr! {
            @build_single
            $graph,
            __next,
            $($rest)*
        }
    };

    (@build_single $graph:ident, $node:ident, $op:ident ( $($op_args:tt)* ) -> $($rest:tt)*) => {
        let __next = $graph.operation(Op::$op($($op_args)*), vec![$node]);
        $crate::expr! {
            @build_single
            $graph,
            __next,
            $($rest)*
        }
    };

    (@build_single $graph:ident, $node:ident, output) => {
        $graph.output($node);
        $graph
    };

    (@build_multi $graph:ident, $node:ident -> Add -> @ $result:ident $($rest:tt)*) => {
        compile_error!("Add is n-ary; use (@a, @b, ...) -> Add");
    };

    (@build_multi $graph:ident, $node:ident -> Mul -> @ $result:ident $($rest:tt)*) => {
        compile_error!("Mul is n-ary; use (@a, @b, ...) -> Mul");
    };

    (@build_multi $graph:ident, $node:ident -> Add ( $($op_args:tt)* ) -> @ $result:ident $($rest:tt)*) => {
        compile_error!("Add takes no arguments and is n-ary; use (@a, @b, ...) -> Add");
    };

    (@build_multi $graph:ident, $node:ident -> Mul ( $($op_args:tt)* ) -> @ $result:ident $($rest:tt)*) => {
        compile_error!("Mul takes no arguments and is n-ary; use (@a, @b, ...) -> Mul");
    };

    (@build_multi $graph:ident, $node:ident -> $op:ident -> @ $result:ident $($rest:tt)*) => {
        let $result = $graph.operation(Op::$op, vec![$node]);
        $crate::expr! { @build_multi $graph, $($rest)* }
    };

    (@build_multi $graph:ident, $node:ident -> $op:ident ( $($op_args:tt)* ) -> @ $result:ident $($rest:tt)*) => {
        let $result = $graph.operation(Op::$op($($op_args)*), vec![$node]);
        $crate::expr! { @build_multi $graph, $($rest)* }
    };

    (@build_multi $graph:ident, ( $( @ $node:ident ),+ ) -> Sin -> @ $result:ident $($rest:tt)*) => {
        compile_error!("Sin is unary; use x -> Sin");
    };

    (@build_multi $graph:ident, ( $( @ $node:ident ),+ ) -> Cos -> @ $result:ident $($rest:tt)*) => {
        compile_error!("Cos is unary; use x -> Cos");
    };

    (@build_multi $graph:ident, ( $( @ $node:ident ),+ ) -> Scale ( $($op_args:tt)* ) -> @ $result:ident $($rest:tt)*) => {
        compile_error!("Scale is unary; use x -> Scale(factor)");
    };

    (@build_multi $graph:ident, ( $( @ $node:ident ),+ ) -> Pow ( $($op_args:tt)* ) -> @ $result:ident $($rest:tt)*) => {
        compile_error!("Pow is unary; use x -> Pow(exp)");
    };

    (@build_multi $graph:ident, ( @ $node:ident ) -> Add -> @ $result:ident $($rest:tt)*) => {
        compile_error!("Add requires at least 2 inputs");
    };

    (@build_multi $graph:ident, ( @ $node:ident ) -> Mul -> @ $result:ident $($rest:tt)*) => {
        compile_error!("Mul requires at least 2 inputs");
    };

    (@build_multi $graph:ident, ( $( @ $node:ident ),+ ) -> $op:ident -> @ $result:ident $($rest:tt)*) => {
        let $result = $graph.operation(Op::$op, vec![$($node),+]);
        $crate::expr! { @build_multi $graph, $($rest)* }
    };

    (@build_multi $graph:ident, ( $( @ $node:ident ),+ ) -> $op:ident ( $($op_args:tt)* ) -> @ $result:ident $($rest:tt)*) => {
        let $result = $graph.operation(Op::$op($($op_args)*), vec![$($node),+]);
        $crate::expr! { @build_multi $graph, $($rest)* }
    };

    (@build_multi $graph:ident, output @ $node:ident) => {
        $graph.output($node);
        $graph
    };

    (@build_multi $graph:ident, output) => {
        $graph
    };
}
