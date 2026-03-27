#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

#[test]
fn invalid_programs_fail_to_compile() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/dense_on_image_without_flatten.rs");
    t.compile_fail("tests/compile_fail/invalid_reshape_size.rs");
    t.compile_fail("tests/compile_fail/invalid_tensor_rank.rs");
    t.compile_fail("tests/compile_fail/missing_saved_source.rs");
    t.compile_fail("tests/compile_fail/duplicate_saved_source.rs");
    t.compile_fail("tests/compile_fail/ragged_tensor_literal.rs");
    t.compile_fail("tests/compile_fail/repeat_requires_shape_preserving_block.rs");
    t.compile_fail("tests/compile_fail/residual_requires_matching_shape.rs");
    t.compile_fail("tests/compile_fail/share_requires_named_blueprint.rs");
    t.compile_fail("tests/compile_fail/sum_requires_matching_branch_shapes.rs");

    #[cfg(not(target_vendor = "apple"))]
    t.compile_fail("tests/compile_fail/invalid_conv_geometry_linux.rs");

    #[cfg(target_vendor = "apple")]
    t.compile_fail("tests/compile_fail/invalid_conv_geometry_macos.rs");
}
