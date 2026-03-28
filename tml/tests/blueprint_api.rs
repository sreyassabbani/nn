#![allow(incomplete_features)]
#![feature(generic_const_exprs, adt_const_params, unsized_const_params)]

use tml::{Float, FragmentExt, InitConfig, network, vision};

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

#[test]
fn saved_sources_can_be_summed_back_into_the_pipeline() {
    let arch = network! {
        input(features: 2)
            -> dense(2)
            -> relu
            -> save(skip)
            -> dense(2)
            -> sum_from(skip)
            -> dense(1)
    };

    let model = arch.materialize(InitConfig::new().seed(11));
    let out = model.predict(&[0.25, -0.5]);
    assert_eq!(out.len(), 1);
}

#[test]
fn saved_sources_can_be_concatenated_back_into_the_pipeline() {
    let arch = network! {
        input(channels: 1, height: 8, width: 8)
            -> conv(2, kernel: 3, pad: 1)
            -> relu
            -> save(skip)
            -> conv(2, kernel: 3, pad: 1)
            -> concat_from(skip, channels)
            -> flatten
            -> dense(3)
    };

    let model = arch.materialize(InitConfig::new().seed(17));
    let out = model.predict(&[0.5; 64]);
    assert_eq!(out.len(), 3);
}

fn user_defined_stem() -> vision::common::Stem<2, 2> {
    vision::common::stem::<2, 2>()
}

#[derive(Clone)]
struct NamedStem;

impl tml::Fragment for NamedStem {
    type Spec = vision::common::StemSpec<2, 2>;

    fn into_blueprint(self) -> tml::Blueprint<Self::Spec> {
        tml::into_blueprint(vision::common::stem::<2, 2>())
    }
}

#[test]
fn rust_defined_fragment_values_compose_inside_network_macro() {
    let tower = vision::common::stem::<2, 2>().then_fragment(vision::common::residual_block::<2>());

    let arch = network! {
        input(channels: 2, height: 8, width: 8) -> tower -> flatten -> dense(4)
    };

    let summary = arch.summary();
    assert!(summary.contains("conv(2, kernel: 3, stride: 1, pad: 1)"));
    assert!(summary.contains("residual"));

    let model = arch.materialize(InitConfig::new().seed(13));
    let out = model.predict(&[0.5; 128]);
    assert_eq!(out.len(), 4);
}

#[test]
fn named_rust_defined_fragments_can_be_shared_by_the_macro() {
    let block = vision::common::residual_block::<2>();

    let unshared = network! {
        input(channels: 2, height: 8, width: 8) -> block -> block -> flatten -> dense(1)
    };
    let shared = network! {
        input(channels: 2, height: 8, width: 8) -> share(block) -> share(block) -> flatten -> dense(1)
    };

    assert!(shared.parameter_count() < unshared.parameter_count());
}

#[test]
fn distinct_fragment_bindings_do_not_accidentally_share_parameters() {
    let left = vision::common::stem::<2, 2>();
    let right = vision::common::stem::<2, 2>();

    let unshared = network! {
        input(channels: 2, height: 8, width: 8) -> left -> right -> flatten -> dense(1)
    };
    let separately_shared = network! {
        input(channels: 2, height: 8, width: 8) -> share(left) -> share(right) -> flatten -> dense(1)
    };

    assert_eq!(
        separately_shared.parameter_count(),
        unshared.parameter_count()
    );
}

#[test]
fn user_defined_fragment_values_can_be_bound_without_the_macro() {
    let stem = user_defined_stem();
    let arch = network! {
        input(channels: 2, height: 8, width: 8) -> stem -> flatten -> dense(2)
    };

    let model = arch.materialize(InitConfig::new().seed(23));
    let out = model.predict(&[1.0; 128]);
    assert_eq!(out.len(), 2);
}

#[test]
fn named_fragment_types_can_be_bound_and_composed_inside_network_macro() {
    let stem = NamedStem;
    let arch = network! {
        input(channels: 2, height: 8, width: 8) -> stem -> flatten -> dense(2)
    };

    let model = arch.materialize(InitConfig::new().seed(29));
    let out = model.predict(&[1.0; 128]);
    assert_eq!(out.len(), 2);
}

#[test]
fn rust_defined_fragment_factories_can_be_called_inside_network_macro() {
    let arch = network! {
        input(channels: 2, height: 8, width: 8) -> user_defined_stem() -> flatten -> dense(2)
    };

    let model = arch.materialize(InitConfig::new().seed(31));
    let out = model.predict(&[0.25; 128]);
    assert_eq!(out.len(), 2);
}

#[test]
fn built_in_fragment_factories_can_be_called_inside_network_macro() {
    let arch = network! {
        input(channels: 2, height: 8, width: 8) -> vision::common::stem::<2, 2>() -> flatten -> dense(2)
    };

    let model = arch.materialize(InitConfig::new().seed(37));
    let out = model.predict(&[0.75; 128]);
    assert_eq!(out.len(), 2);
}
