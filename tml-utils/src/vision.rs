pub mod common {
    use crate::blueprint::{Blueprint, ConvSpec, ReLUSpec, ResidualSpec, SeqSpec};
    use crate::{conv, relu, residual};

    pub type StemSpec<const C1: usize, const C2: usize> = SeqSpec<
        SeqSpec<SeqSpec<ConvSpec<C1, 3, 3, 1, 1>, ReLUSpec>, ConvSpec<C2, 3, 3, 1, 1>>,
        ReLUSpec,
    >;

    pub fn stem<const C1: usize, const C2: usize>() -> Blueprint<StemSpec<C1, C2>> {
        conv::<C1, 3, 3, 1, 1>()
            .then(relu())
            .then(conv::<C2, 3, 3, 1, 1>())
            .then(relu())
    }

    pub type ResidualBlockBodySpec<const WIDTH: usize> =
        SeqSpec<SeqSpec<ConvSpec<WIDTH, 3, 3, 1, 1>, ReLUSpec>, ConvSpec<WIDTH, 3, 3, 1, 1>>;

    pub type ResidualBlockSpec<const WIDTH: usize> = ResidualSpec<ResidualBlockBodySpec<WIDTH>>;

    pub fn residual_block<const WIDTH: usize>() -> Blueprint<ResidualBlockSpec<WIDTH>> {
        residual(
            conv::<WIDTH, 3, 3, 1, 1>()
                .then(relu())
                .then(conv::<WIDTH, 3, 3, 1, 1>()),
        )
    }
}
