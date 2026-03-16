use tml::{Float, ModelBuilder, Sample, TrainConfig};

fn main() {
    let mut model = ModelBuilder::new()
        .input::<1>()
        .dense::<8>()
        .relu()
        .dense::<1>()
        .build();

    let samples = (-20..=20)
        .map(|i| {
            let x = i as Float / 10.0;
            Sample::new([x], [2.0 * x - 0.5])
        })
        .collect::<Vec<_>>();

    let config = TrainConfig::adam(0.03)
        .epochs(250)
        .batch_size(8)
        .shuffle_seed(9);

    let loss = model.fit(&samples, config);
    println!("final loss: {loss}");

    for x in [-2.0, 0.0, 1.5, 4.0] {
        let pred = model.predict(&[x]);
        println!("x = {x:>5.2} -> y = {:.4}", pred[0]);
    }
}
