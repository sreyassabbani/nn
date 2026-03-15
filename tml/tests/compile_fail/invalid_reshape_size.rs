use tml::{shape, tensor};

fn main() {
    let _ = tensor![1.0, 2.0].reshape::<shape!(3)>();
}
