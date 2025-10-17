use std::{array, marker::PhantomData, ops};

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

#[derive(Debug, Clone)]
pub struct Tensor<const N: usize, Shape> {
    data: [f64; N],
    _shape_marker: PhantomData<Shape>,
}

impl<const N: usize, Shape> Tensor<N, Shape> {
    pub fn new() -> Self {
        Self {
            data: [0.; N],
            _shape_marker: PhantomData,
        }
    }
}

impl<const N: usize, Shape> Default for Tensor<N, Shape> {
    fn default() -> Self {
        Self::new()
    }
}

#[macro_export]
macro_rules! shape_ty {
    ($d:expr) => {
        [f64; $d]
    };
    ($first:expr, $($rest:expr),+ $(,)?) => {
        [$crate::shape_ty!($($rest),+); $first]
    };
}

// left-fold shape_ty
//
// #[macro_export]
// macro_rules! shape_ty {
//     ($d:expr) => {
//         [f64; $d]
//     };

//     ($first:expr, $($rest:expr),+ $(,)?) => {
//         $crate::shape_ty!(@acc [f64; $first] $(, $rest)+)
//     };

//     (@acc $acc:ty, $next:expr) => {
//         [$acc; $next]
//     };

//     (@acc $acc:ty, $next:expr, $($rest:expr),+ $(,)?) => {
//         $crate::shape_ty!(@acc [$acc; $next] $(, $rest)+)
//     };
// }

#[macro_export]
macro_rules! tensor {
    ($first:expr $(, $rest:expr)* $(,)?) => {
        {
            const N: usize = $first $( * $rest )*;
            type Shape = $crate::shape_ty!($first $(, $rest)*);

            <$crate::network::Tensor::<N, Shape>>::new()
        }
    };
}

// impl<D> ops::Index<I> for Tensor<D> {
//     type Output = f64;
//     fn index(&self, index: I) -> &Self::Output {
//         &3.
//     }
// }

// impl<D> Tensor<D> {
//     pub const fn reshape<IR>(self, dim: IR) -> Tensor<N, IR>
//     where IR: IntoIterator
//     {

//         let Tensor { data, .. } = self;
//         assert_eq!(dim.into_iter().product(), I.iter().product());
//         Tensor {
//             data,
//             _index_marker: PhantomData,
//         }
//     }
// }

// pub struct Tensor<const N: usize, const D: usize> {
//     data: [f64; N],
//     shape: [usize; D] // i would really like this to be part of the type...
// }

// impl<const N: usize, const D: usize> ops::Index<[usize; D]> for Tensor<N, D> {
//     type Output = f64;
//     fn index(&self, index: [usize; D]) -> &Self::Output {
//         &3.
//     }
// }

// impl<const N: usize, const D: usize> Tensor<N, D> {
//     fn reshape(&mut self, shape: [usize; D]) {
//         self.shape = shape;
//     }
// }

/// A convolutional layer
///
/// `H` - filter/kernel height
/// `W` - filter/kernel width
/// `IC` - number of input channels
/// `OC` - number of output channels (equivalently, number of kernels/filters)
/// `S` - stride
/// `P` - padding
#[derive(Debug)]
pub struct Conv<
    const IW: usize,
    const IH: usize,
    const IC: usize,
    const H: usize,
    const W: usize,
    const OC: usize,
    const S: usize,
    const P: usize,
> where
    [(); H * W * IC]:,
{
    data: [Filter<H, W, IC>; OC],
}

// height, width, and depth (input channel size)
// pub struct Filter<const H: usize, const W: usize, const D: usize>([[[f32; H]; W]; D]);
#[derive(Debug, Clone, Default)]
pub struct Filter<const H: usize, const W: usize, const D: usize>(
    pub Tensor<{ H * W * D }, crate::shape_ty!(H, W, D)>,
)
where
    [(); H * W * D]:;

// Forward pass implementation for Conv
impl<
    const IW: usize,
    const IH: usize,
    const IC: usize,
    const H: usize,
    const W: usize,
    const OC: usize,
    const S: usize,
    const P: usize,
> Conv<IW, IH, IC, H, W, OC, S, P>
where
    [(); H * W * IC]:,
{
    pub fn init() -> Self {
        Conv {
            // data: [Filter([[[0.; H]; W]; IC]); OC],
            // data: [const { Filter(tensor!(H, W, IC)) }; OC],
            data: array::from_fn(|_| Filter::default()),
        }
    }

    // pub fn forward(&self, input: &[f32], output: &mut [f32])
    // // where
    // //     I: AsRef<[f32; N]>,
    // {
    //     let out_h = IH - H + 1;
    //     let out_w = IW - W + 1;

    //     for oc in 0..OC {
    //         for ic in 0..IC {
    //             for x in 0..out_w {
    //                 for y in 0..out_h {
    //                     let mut val = 0.;
    //                     for ix in (x * S + W / 2)..() {
    //                         for iy in (y * S + H / 2)..() {
    //                             val += self.data[oc].0[ic][x][y] * input[];
    //                         }
    //                     }
    //                     output[oc * IC * out_w * out_h + ic * out_w * out_h + x * out_h + y] =
    //                         val;
    //                 }
    //             }
    //         }
    //     }
    // }
}

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
