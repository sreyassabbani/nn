#![allow(incomplete_features)]
#![feature(generic_const_exprs, adt_const_params, unsized_const_params)]

use tml::{Axis, InitConfig, concat, dense, identity, root, validate_blueprint};

type Tokens = tml::shape!(tokens: 3);

fn main() {
    let spec = concat(Axis::new("tokens"), identity(), identity()).then(dense::<2>());
    let arch = validate_blueprint(root::<Tokens, _>(spec, Vec::new()));

    println!("summary:\n{}", arch.summary());
    println!("shape trace:\n{}", arch.shape_trace());

    let model = arch.materialize(InitConfig::new().seed(33));
    let output = model.predict(&[1.0, 0.0, -1.0]);
    println!("prediction: {output:?}");
}
