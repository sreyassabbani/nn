use nn::*;

fn main() {
    let input = Layer::from([1.0]);
    let empty = Layer::from([0.0]);
    let weights = vec![Weight::from([[1.0]])];
    let biases = vec![vec![Bias(2.0)]];

    let mut network = Network::new(vec![empty], weights, biases);

    network.train(
        &[
            Input {
                layer: input.clone(),
                expect: 4.0,
            },
            Input {
                layer: Layer::from([2.0]),
                expect: 6.0,
            },
            Input {
                layer: input,
                expect: 4.0,
            },
        ],
        0.2,
        2000,
    );

    let cost = &network.run(&Input {
        layer: Layer::from([3.0]),
        expect: 8.0,
    });

    dbg!(cost);
}
