use tml::{Adam, Chain, DenseLayer, End, Float, ReLU, Sample, Sequential, TrainConfig};

fn main() {
    let layers = Chain::<_, _, 8>::new(
        DenseLayer::<1, 8>::seeded(3),
        Chain::<_, _, 8>::new(
            ReLU::<8>::init(),
            Chain::<_, _, 1>::new(DenseLayer::<8, 1>::seeded(4), End),
        ),
    );
    let mut model = Sequential::new(layers);

    let samples = (-20..=20)
        .map(|i| {
            let x = i as Float / 10.0;
            Sample::new([x], [2.0 * x - 0.5])
        })
        .collect::<Vec<_>>();

    let mut optimizer = Adam::new(0.03);
    let config = TrainConfig {
        epochs: 250,
        batch_size: 8,
        shuffle_seed: Some(9),
    };

    let loss = model.fit(&samples, &mut optimizer, config);
    println!("final loss: {loss}");

    for x in [-2.0, 0.0, 1.5, 4.0] {
        let pred = model.predict(&[x]);
        println!("x = {x:>5.2} -> y = {:.4}", pred[0]);
    }
}
