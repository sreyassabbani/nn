#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

#[test]
fn invalid_programs_fail_to_compile() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}
