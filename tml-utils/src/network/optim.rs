use crate::Float;
use rand::Rng;
use std::fmt;

pub trait Initializer {
    fn fill<R: Rng + ?Sized>(
        &self,
        values: &mut [Float],
        fan_in: usize,
        fan_out: usize,
        rng: &mut R,
    );
}

#[derive(Debug, Clone, Copy)]
pub struct Uniform {
    pub low: Float,
    pub high: Float,
}

impl Uniform {
    pub const fn new(low: Float, high: Float) -> Self {
        Self { low, high }
    }
}

impl Initializer for Uniform {
    fn fill<R: Rng + ?Sized>(
        &self,
        values: &mut [Float],
        _fan_in: usize,
        _fan_out: usize,
        rng: &mut R,
    ) {
        for value in values {
            *value = rng.random_range(self.low..self.high);
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct XavierUniform;

impl Initializer for XavierUniform {
    fn fill<R: Rng + ?Sized>(
        &self,
        values: &mut [Float],
        fan_in: usize,
        fan_out: usize,
        rng: &mut R,
    ) {
        let denom = (fan_in + fan_out).max(1) as Float;
        let bound = (6.0 / denom).sqrt();
        Uniform::new(-bound, bound).fill(values, fan_in, fan_out, rng);
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct KaimingUniform;

impl Initializer for KaimingUniform {
    fn fill<R: Rng + ?Sized>(
        &self,
        values: &mut [Float],
        fan_in: usize,
        fan_out: usize,
        rng: &mut R,
    ) {
        let denom = fan_in.max(1) as Float;
        let bound = (6.0 / denom).sqrt();
        Uniform::new(-bound, bound).fill(values, fan_in, fan_out, rng);
    }
}

pub trait Optimizer: fmt::Debug {
    fn begin_step(&mut self) {}
    fn update_parameter(
        &mut self,
        slot: usize,
        params: &mut [Float],
        grads: &[Float],
        scale: Float,
    );
}

#[derive(Debug, Clone, Copy)]
pub struct Sgd {
    pub lr: Float,
    pub weight_decay: Float,
}

impl Sgd {
    pub const fn new(lr: Float) -> Self {
        Self {
            lr,
            weight_decay: 0.0,
        }
    }

    pub const fn with_weight_decay(mut self, weight_decay: Float) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for Sgd {
    fn update_parameter(
        &mut self,
        _slot: usize,
        params: &mut [Float],
        grads: &[Float],
        scale: Float,
    ) {
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            let update = *grad * scale + self.weight_decay * *param;
            *param -= self.lr * update;
        }
    }
}

#[derive(Debug, Clone)]
pub struct Adam {
    pub lr: Float,
    pub beta1: Float,
    pub beta2: Float,
    pub epsilon: Float,
    pub weight_decay: Float,
    step: usize,
    first_moment: Vec<Box<[Float]>>,
    second_moment: Vec<Box<[Float]>>,
}

impl Adam {
    pub fn new(lr: Float) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            step: 0,
            first_moment: Vec::new(),
            second_moment: Vec::new(),
        }
    }

    pub const fn with_weight_decay(mut self, weight_decay: Float) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    fn ensure_slot(&mut self, slot: usize, len: usize) {
        while self.first_moment.len() <= slot {
            self.first_moment.push(Vec::new().into_boxed_slice());
            self.second_moment.push(Vec::new().into_boxed_slice());
        }
        if self.first_moment[slot].len() != len {
            self.first_moment[slot] = vec![0.0; len].into_boxed_slice();
            self.second_moment[slot] = vec![0.0; len].into_boxed_slice();
        }
    }
}

impl Optimizer for Adam {
    fn begin_step(&mut self) {
        self.step += 1;
    }

    fn update_parameter(
        &mut self,
        slot: usize,
        params: &mut [Float],
        grads: &[Float],
        scale: Float,
    ) {
        self.ensure_slot(slot, params.len());
        let bias_correction1 = 1.0 - self.beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step as i32);

        let first = &mut self.first_moment[slot];
        let second = &mut self.second_moment[slot];
        for i in 0..params.len() {
            let grad = grads[i] * scale + self.weight_decay * params[i];
            first[i] = self.beta1 * first[i] + (1.0 - self.beta1) * grad;
            second[i] = self.beta2 * second[i] + (1.0 - self.beta2) * grad * grad;
            let m_hat = first[i] / bias_correction1.max(f64::EPSILON);
            let v_hat = second[i] / bias_correction2.max(f64::EPSILON);
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}

pub struct TrainConfig {
    optimizer: Box<dyn Optimizer>,
    pub epochs: usize,
    pub batch_size: usize,
    pub shuffle_seed: Option<u64>,
}

impl fmt::Debug for TrainConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrainConfig")
            .field("optimizer", &self.optimizer)
            .field("epochs", &self.epochs)
            .field("batch_size", &self.batch_size)
            .field("shuffle_seed", &self.shuffle_seed)
            .finish()
    }
}

impl TrainConfig {
    pub fn new<O: Optimizer + 'static>(optimizer: O) -> Self {
        Self {
            optimizer: Box::new(optimizer),
            epochs: 1,
            batch_size: 1,
            shuffle_seed: None,
        }
    }

    pub fn sgd(lr: Float) -> Self {
        Self::new(Sgd::new(lr))
    }

    pub fn adam(lr: Float) -> Self {
        Self::new(Adam::new(lr))
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    pub fn shuffle_seed(mut self, shuffle_seed: u64) -> Self {
        self.shuffle_seed = Some(shuffle_seed);
        self
    }

    pub(crate) fn optimizer_mut(&mut self) -> &mut dyn Optimizer {
        self.optimizer.as_mut()
    }
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self::adam(1e-3)
    }
}
