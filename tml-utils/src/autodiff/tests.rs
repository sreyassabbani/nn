use crate::Float;

use super::{ExprGraph, Op, Tape};

fn approx_eq(a: Float, b: Float, eps: Float) {
    let diff = (a - b).abs();
    assert!(diff <= eps, "expected {a} ~= {b} (diff={diff}, eps={eps})");
}

#[test]
fn reverse_matches_forward_and_finite_difference() {
    let mut g = ExprGraph::new();
    let x = g.input("x".to_string());
    let z = g.input("z".to_string());
    let x_sq = g.operation(Op::Pow(2), [x]);
    let z_cos = g.operation(Op::Cos, [z]);
    let sum = g.operation(Op::Add, [x_sq, z_cos]);
    let out = g.operation(Op::Sin, [sum]);
    g.output(out);

    let base = [1.3, -0.7];
    let (fwd_val, fwd_grad) = g.eval_fwd_one(&base);
    let (rev_val, rev_grad) = g.eval_one(&base);

    approx_eq(fwd_val, rev_val, 1e-12);
    approx_eq(fwd_grad[0], rev_grad[0], 1e-10);
    approx_eq(fwd_grad[1], rev_grad[1], 1e-10);

    let eps = 1e-7;
    for i in 0..base.len() {
        let mut plus = base;
        let mut minus = base;
        plus[i] += eps;
        minus[i] -= eps;
        let f_plus = g.eval_fwd_one(&plus).0;
        let f_minus = g.eval_fwd_one(&minus).0;
        let numeric = (f_plus - f_minus) / (2.0 * eps);
        approx_eq(rev_grad[i], numeric, 1e-6);
    }
}

#[test]
fn output_rejects_foreign_node_id() {
    let mut g1 = ExprGraph::new();
    let foreign = g1.input("x".to_string());

    let mut g2 = ExprGraph::new();
    let _ = g2.input("y".to_string());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        g2.output(foreign);
    }));
    assert!(result.is_err());
}

#[test]
fn tape_try_set_variants() {
    let mut tape = Tape::new();
    let x = tape.input("x", 1.0);
    let y = tape.input("y", 2.0);
    let out = x + y;

    tape.try_set_inputs(&[3.0, 4.0])
        .expect("valid input update");
    let grads = tape.gradients(&out);
    approx_eq(grads.value, 7.0, 1e-12);

    let err = tape
        .try_set_inputs(&[1.0])
        .expect_err("length mismatch should fail");
    assert!(matches!(
        err,
        super::TapeError::InputLengthMismatch {
            expected: 2,
            got: 1
        }
    ));

    tape.try_set("x", 5.0).expect("known input should be set");
    let err = tape
        .try_set("missing", 0.0)
        .expect_err("unknown input should fail");
    assert!(matches!(err, super::TapeError::UnknownInput(_)));
}

#[test]
fn pow_zero_has_zero_gradient_at_zero() {
    let mut g = ExprGraph::new();
    let x = g.input("x".to_string());
    let out = g.operation(Op::Pow(0), [x]);
    g.output(out);

    let (value, grads) = g.eval_one(&[0.0]);
    approx_eq(value, 1.0, 1e-12);
    approx_eq(grads[0], 0.0, 1e-12);
    assert!(grads[0].is_finite());
}
