/// Computes the output extent of one convolution axis.
#[doc(hidden)]
pub const fn conv_out_dim(input: usize, pad: usize, kernel: usize, stride: usize) -> usize {
    if stride == 0 {
        return 0;
    }
    let padded = input + 2 * pad;
    if padded < kernel {
        return 0;
    }
    let numer = padded - kernel;
    if !numer.is_multiple_of(stride) {
        return 0;
    }
    numer / stride + 1
}
