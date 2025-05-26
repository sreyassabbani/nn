#![feature(generic_arg_infer)]
#![allow(incomplete_features)]

pub mod activation;

#[macro_use]
pub mod layer;

pub mod network;
mod tensor;

#[derive(Debug, Clone)]
pub struct Node(f64);

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
    weights: Vec<Weight>,
    biases: Vec<Vec<Bias>>,
}

#[derive(Debug)]
pub struct Input {
    pub layer: Layer,
    pub expect: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Bias(pub f64);

#[derive(Debug, Clone)]
pub struct Layer {
    nodes: Vec<Node>,
}

impl Layer {
    pub fn dot(&self, other: &Vec<f64>) -> f64 {
        self.nodes
            .iter()
            .zip(other.iter())
            .map(|(&Node(value), &other)| value * other)
            .sum()
    }

    pub fn new() -> Self {
        Layer { nodes: Vec::new() }
    }
}

impl<const N: usize> From<[f64; N]> for Layer {
    fn from(value: [f64; N]) -> Self {
        Layer {
            nodes: Vec::from(value.map(Node::new)),
        }
    }
}

impl Network {
    pub fn new(layers: Vec<Layer>, weights: Vec<Weight>, biases: Vec<Vec<Bias>>) -> Self {
        Self {
            layers,
            weights,
            biases,
        }
    }

    pub fn run(&self, input: &Input) -> f64 {
        let output_layer =
            self.layers
                .iter()
                .enumerate()
                .fold(input.layer.clone(), |accumulation, (l, layer)| Layer {
                    nodes: (0..layer.nodes.len())
                        .map(|i| {
                            Node(accumulation.dot(&self.weights[l].0.0[i]) + self.biases[l][i].0)
                        })
                        .collect::<Vec<_>>(),
                });

        output_layer
            .nodes
            .iter()
            .fold(0.0, |accumulation, &Node(val)| {
                accumulation + (val - input.expect).powi(2)
            })

        // (0..self.layers.len())
        //     .map(|i| self.next(i))
        //     .last() // select final layer
        //     .unwrap()
        //     .nodes
        //     .iter()
        //     .map(|Node(value)| (value - self.input.expect).exp2())
        //     .sum()
    }

    fn _run(&mut self, input: &Input) -> f64 {
        if self.layers.len() == 0 {
            let mut acc = 0.0;
            for i in 0..input.layer.nodes.len() {
                acc += (input.layer.dot(&self.weights[0].0.0[i]) + self.biases[0][i].0
                    - input.expect)
                    .powi(2);
            }
            return acc;
        }
        for (i, Node(value)) in self.layers[0].nodes.iter_mut().enumerate() {
            *value = input.layer.dot(&self.weights[0].0.0[i]) + self.biases[0][i].0;
        }
        for l in 1..self.layers.len() {
            let weight_rows = &self.weights[l].0.0;
            let bias_row = &self.biases[l];

            let (lower, upper) = self.layers.split_at_mut(l);
            let prev_layer = &lower[l - 1]; // immutable borrow of layers[l-1]
            let curr_layer = &mut upper[0]; // mutable borrow of layers[l]

            for (i, Node(value)) in curr_layer.nodes.iter_mut().enumerate() {
                *value = prev_layer.dot(&weight_rows[i]) + bias_row[i].0;
            }
        }

        self.layers
            .last()
            .unwrap()
            .nodes
            .iter()
            .fold(0.0, |accumulation, &Node(val)| {
                accumulation + (val - input.expect).powi(2)
            })
    }

    pub fn train(&mut self, training_data: &[Input], eta: f64, epochs: usize) {
        // dbg!(&self.layers);
        // dbg!(&self.weights);
        for _ in 0..epochs {
            for data in training_data {
                self._run(&data);
                for (l, layer) in self.layers.iter().enumerate() {
                    let inputs = if l == 0 {
                        &data.layer.nodes
                    } else {
                        &self.layers[l - 1].nodes
                    };
                    for (i, row) in self.weights[l].0.0.iter_mut().enumerate() {
                        let error = layer.nodes[i].0 - data.expect;
                        for (j, weight) in row.iter_mut().enumerate() {
                            let sd = -2.0 * inputs[j].0 * (*weight);
                            // dbg!(sd);
                            *weight -= eta * 2.0 * error * inputs[j].0 / (sd.abs() + 1.0);
                            // *weight -= eta * 2.0 * error * inputs[j].0;
                        }
                    }
                }
                // dbg!(&self.weights);
                for (l, bias_row) in self.biases.iter_mut().enumerate() {
                    for (i, Bias(b)) in bias_row.iter_mut().enumerate() {
                        let error = self.layers[l].nodes[i].0 - data.expect;
                        *b -= eta * 2.0 * error;
                    }
                }
            }
        }
    }
}

impl FromIterator<f64> for Layer {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        let mut layer = Layer::new();
        for item in iter {
            layer.nodes.push(Node(item));
        }
        layer
    }
}

impl Node {
    pub fn new(value: f64) -> Self {
        Node(value)
    }
}

#[derive(Debug, Clone)]
pub struct Weight(Matrix);

impl<const M: usize, const N: usize> From<[[f64; M]; N]> for Weight {
    fn from(value: [[f64; M]; N]) -> Self {
        Self(Matrix(Vec::from(value.map(Vec::from))))
    }
}

#[derive(Debug, Clone)]
pub struct Matrix(Vec<Vec<f64>>);

impl<const M: usize, const N: usize> From<[[f64; M]; N]> for Matrix {
    fn from(value: [[f64; M]; N]) -> Self {
        Self(Vec::from(value.map(Vec::from)))
    }
}

impl Weight {
    pub fn new(value: Matrix) -> Self {
        Weight(value)
    }
}
