use tml::{Float, Sample, TrainConfig, network};

// y = 3x + 2
fn magic(x: Float) -> Float {
    3. * x + 2.
}

fn main() {
    let mut model = network! {
        input(1) -> dense(1) -> output
    };

    let samples = (-50..=50)
        .map(|i| {
            let x = i as Float / 10.0;
            let y = magic(x);
            Sample::new([x], [y])
        })
        .collect::<Vec<_>>();

    let config = TrainConfig::sgd(0.05)
        .epochs(1500)
        .batch_size(16)
        .shuffle_seed(7);
    let loss = model.fit(&samples, config);
    println!("final loss: {loss}");

    for x in [-2.0, 0.0, 1.5, 4.0] {
        let target = magic(x);
        let pred = model.predict(&[x]);
        println!("x = {x:>5.2} -> y = {:.4} (target {:.4})", pred[0], target);
    }
}
