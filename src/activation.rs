pub trait Activation<T> {
    fn forward(&self, input: T) -> T;
    fn backward(&self, input: T) -> T;
}

pub struct Sigmoid;

impl Activation<f64> for Sigmoid {
    fn forward(&self, input: f64) -> f64 {
        1.0 / (1.0 + (-input).exp())
    }

    fn backward(&self, input: f64) -> f64 {
        // TODO: optimize this expression
        -(-input).exp() * self.forward(input)
    }
}

pub struct ReLU;

impl Activation<f64> for ReLU {
    fn forward(&self, input: f64) -> f64 {
        input.max(0.0)
    }

    fn backward(&self, input: f64) -> f64 {
        if input > 0.0 { 1.0 } else { 0.0 }
    }
}

pub struct SiLU {
    beta: f64,
}

impl Activation<f64> for SiLU {
    fn forward(&self, input: f64) -> f64 {
        input / (1.0 + (-self.beta * input).exp())
    }

    fn backward(&self, input: f64) -> f64 {
        // TODO: optimize this expression

        let bx = self.beta * input;
        let exp_nbx = (-bx).exp();

        (1.0 + (1.0 + bx) * exp_nbx) / (1.0 + exp_nbx).powi(2)
    }
}
