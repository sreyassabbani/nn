// Define the DenseLayer struct with weights and biases
pub struct DenseLayer<const IN: usize, const OUT: usize> {
    weights: [[f32; IN]; OUT],
    biases: [f32; OUT],
}

// Define activation structs
pub struct ReLU;
pub struct Sigmoid;

// Forward pass implementation for ReLU
impl ReLU {
    pub fn forward<const N: usize>(&self, input: &[f32; N], output: &mut [f32; N]) {
        for i in 0..N {
            output[i] = input[i].max(0.0);
        }
    }
}

// Forward pass implementation for Sigmoid
impl Sigmoid {
    pub fn forward<const N: usize>(&self, input: &[f32; N], output: &mut [f32; N]) {
        for i in 0..N {
            output[i] = 1.0 / (1.0 + (-input[i]).exp());
        }
    }
}

// Trait for initializing layers
pub trait LayerInit {
    fn init() -> Self;
}

// Initialize DenseLayer (simplified; real init would use proper randomization)
impl<const IN: usize, const OUT: usize> LayerInit for DenseLayer<IN, OUT> {
    fn init() -> Self {
        Self {
            weights: [[0.0; IN]; OUT],
            biases: [0.0; OUT],
        }
    }
}

// Initialize ReLU
impl LayerInit for ReLU {
    fn init() -> Self {
        ReLU
    }
}

// Initialize Sigmoid
impl LayerInit for Sigmoid {
    fn init() -> Self {
        Sigmoid
    }
}

// Trait for network functionality
pub trait NetworkTrait<const IN: usize, const OUT: usize> {
    fn forward(&self, input: &[f32; IN]) -> [f32; OUT];
    fn train(&mut self, data: &[[f32; IN]], targets: &[[f32; OUT]]);
}

// Forward pass for DenseLayer (basic implementation)
impl<const IN: usize, const OUT: usize> DenseLayer<IN, OUT> {
    pub fn forward(&self, input: &[f32; IN], output: &mut [f32; OUT]) {
        for o in 0..OUT {
            let mut sum = self.biases[o];
            for i in 0..IN {
                sum += self.weights[o][i] * input[i];
            }
            output[o] = sum;
        }
    }
}

#[macro_export]
macro_rules! __network {
    // strip off the leading `input(N) ->`
    (input($in:expr) -> $($rest:tt)*) => {
        // call the recursive helper with:
        //  – an empty accumulator `()` for the tuple‐types
        //  – the current “in size” = $in
        $crate::__network!(@accumulate (), $in, $($rest)*);
    };

    // end recursion
    (@accumulate ( $($types:ty,)* ), $cur_in:literal, output) => {
        {
            struct Net(($($types,)*));

            impl Net {
                pub fn new() -> Self {
                    Net ((
                        $(<$types as $crate::network::LayerInit>::init(),)*
                    ))
                }
            }

            Net::new()
        }
    };

    (@accumulate ( $($types:ty,)* ), $cur_in:expr, dense($mid:expr) -> $($rest:tt)* ) => {
        $crate::__network!(@accumulate
            ( $($types,)* $crate::network::DenseLayer<$cur_in,$mid>, ),
            $mid,
            $($rest)*
        );
    };

    (@accumulate ( $($types:ty,)* ), $cur_in:expr, relu -> $($rest:tt)* ) => {
        $crate::__network!(@accumulate
            ( $($types,)* $crate::network::ReLU, ),
            $cur_in,
            $($rest)*
        );
    };

    (@accumulate ( $($types:ty,)* ), $cur_in:expr, sigmoid -> $($rest:tt)* ) => {
        $crate::__network!(@accumulate
            ( $($types,)* Sigmoid, ),
            $cur_in,
            $($rest)*
        );
    };
}

pub use __network as network;
