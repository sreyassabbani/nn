use tml::{Float, Sample, Sgd, TrainConfig, network};

fn main() {
    // y = 3x + 2
    let mut model = network! {
        input(1) -> dense(1) -> output
    };

    let samples = (-50..=50)
        .map(|i| {
            let x = i as Float / 10.0;
            let y = 3.0 * x + 2.0;
            Sample::new([x], [y])
        })
        .collect::<Vec<_>>();

    let mut optimizer = Sgd::new(0.05);
    let config = TrainConfig {
        epochs: 1500,
        batch_size: 16,
        shuffle_seed: Some(7),
    };
    let loss = model.fit(&samples, &mut optimizer, config);
    println!("final loss: {loss}");

    for x in [-2.0, 0.0, 1.5, 4.0] {
        let pred = model.predict(&[x]);
        println!("x = {x:>5.2} -> y = {:.4}", pred[0]);
    }
}
