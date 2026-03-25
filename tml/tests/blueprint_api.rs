#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use tml::{Float, InitConfig, network};

fn assert_close<const N: usize>(left: [Float; N], right: [Float; N]) {
    for (lhs, rhs) in left.into_iter().zip(right) {
        assert!(
            (lhs - rhs).abs() < 1e-9,
            "expected {lhs} ~= {rhs} within 1e-9"
        );
    }
}

#[test]
fn blueprint_inspection_reports_summary_trace_and_params() {
    let arch = network! {
        input(channels: 1, height: 8, width: 8)
            -> conv(2, kernel: 3, pad: 1)
            -> relu
            -> flatten
            -> dense(3)
    };

    let summary = arch.summary();
    assert!(summary.contains("input (channels: 1, height: 8, width: 8)"));
    assert!(summary.contains("conv(2, kernel: 3, stride: 1, pad: 1)"));
    assert!(summary.contains("flatten"));
    assert!(summary.contains("dense(3)"));

    let trace = arch.shape_trace();
    assert!(
        trace.contains("(channels: 1, height: 8, width: 8) -> (channels: 2, height: 8, width: 8)")
    );
    assert!(trace.contains("(channels: 2, height: 8, width: 8) -> (features: 128)"));
    assert!(trace.contains("(features: 128) -> (features: 3)"));

    assert_eq!(arch.parameter_count(), 407);
}

#[test]
fn materialize_is_reproducible_with_the_same_seed() {
    let arch = network! {
        input(features: 1) -> dense(8) -> relu -> dense(1)
    };

    let model_a = arch.materialize(InitConfig::new().seed(42));
    let model_b = arch.materialize(InitConfig::new().seed(42));
    let out_a = model_a.predict(&[0.5]);
    let out_b = model_b.predict(&[0.5]);
    assert_close(out_a, out_b);
}

#[test]
fn sharing_changes_parameter_count_and_reuses_the_same_transform() {
    let stem = network! {
        dense(2) -> relu
    };

    let unshared = network! {
        input(features: 2) -> stem -> stem
    };
    let shared = network! {
        input(features: 2) -> share(stem) -> share(stem)
    };
    let single = network! {
        input(features: 2) -> share(stem)
    };

    assert_eq!(unshared.parameter_count(), 12);
    assert_eq!(shared.parameter_count(), 6);

    let single_model = single.materialize(InitConfig::new().seed(7));
    let shared_model = shared.materialize(InitConfig::new().seed(7));
    let input = [0.25, -0.75];

    let mid = single_model.predict(&input);
    let expected = single_model.predict(&mid);
    let actual = shared_model.predict(&input);
    assert_close(actual, expected);
}

#[test]
fn heads_materialize_into_named_outputs() {
    let arch = network! {
        input(features: 2)
            -> dense(3)
            -> relu
            -> heads {
                logits: dense(2),
                embedding: dense(4),
            }
    };

    let summary = arch.summary();
    assert!(summary.contains("heads"));
    assert!(summary.contains("head logits"));
    assert!(summary.contains("head embedding"));

    let model = arch.materialize_heads(InitConfig::new().seed(9));
    let out = model.predict(&[0.5, -1.0]);
    assert_eq!(out.logits.len(), 2);
    assert_eq!(out.embedding.len(), 4);
}

#[test]
fn tagged_volume_input_can_flatten_into_dense_output() {
    let arch = network! {
        input(channels: 1, depth: 2, height: 3, width: 4)
            -> flatten
            -> dense(5)
    };

    let model = arch.materialize(InitConfig::new().seed(5));
    let out = model.predict(&[0.0; 24]);
    assert_eq!(out.len(), 5);
}

#[test]
fn repeat_accepts_shape_preserving_blocks() {
    let arch = network! {
        input(features: 2) -> repeat(3, relu) -> dense(1)
    };

    let model = arch.materialize(InitConfig::new().seed(3));
    let out = model.predict(&[1.0, -2.0]);
    assert_eq!(out.len(), 1);
}
