// Define the DenseLayer struct with weights and biases
#[derive(Debug)]
pub struct DenseLayer<const IN: usize, const OUT: usize> {
    weights: Box<[[f32; IN]; OUT]>,
    biases: Box<[f32; OUT]>,
}

// Rectified Linear Unit
#[derive(Debug)]
pub struct ReLU<const N: usize>;

// Sigmoid
#[derive(Debug)]
pub struct Sigmoid<const N: usize>;

// Conv
#[derive(Debug)]
pub struct Conv<const C: usize, const H: usize, const W: usize>;

// Forward pass implementation for ReLU
impl<const N: usize> ReLU<N> {
    pub fn init() -> Self {
        ReLU
    }

    pub fn forward(&self, input: &[f32], output: &mut [f32])
    // where
    //     I: AsRef<[f32; N]>,
    {
        for i in 0..N {
            output[i] = input.as_ref()[i].max(0.0);
        }
    }
}

// Forward pass implementation for Sigmoid
impl<const N: usize> Sigmoid<N> {
    pub fn init() -> Self {
        Sigmoid
    }

    /// You can pass a reference to owned values in &Box<>
    pub fn forward(&self, input: &[f32], output: &mut [f32])
    // where
    //     I: AsRef<[f32; N]>,
    {
        for i in 0..N {
            output[i] = 1.0 / (1.0 + (-input.as_ref()[i]).exp());
        }
    }
}

// Initialize DenseLayer (simplified; real init would use proper randomization)
impl<const IN: usize, const OUT: usize> DenseLayer<IN, OUT> {
    pub fn init() -> Self {
        Self {
            weights: Box::new([[0.0; IN]; OUT]),
            biases: Box::new([0.0; OUT]),
        }
    }

    // Forward pass for DenseLayer (basic implementation)
    //
    // used to be forward<I: AsRef<[f32; IN]>>(... input: I, ...)
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        for o in 0..OUT {
            let mut sum = self.biases[o];
            for i in 0..IN {
                sum += self.weights[o][i] * input.as_ref()[i];
            }
            output[o] = sum;
        }
    }
}
