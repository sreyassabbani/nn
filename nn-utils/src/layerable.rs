#[derive(Debug, Clone)]
pub enum LayerKind {
    Dense { output: usize },
    ReLU { width: usize },
    Sigmoid { width: usize },
    Conv {
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    },
}

pub trait Layerable {
    fn input(&self) -> usize;
    fn kind(&self) -> LayerKind;
}
