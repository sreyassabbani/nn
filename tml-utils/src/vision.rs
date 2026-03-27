pub mod common {
    use crate::blueprint::{Blueprint, ConvSpec, Fragment, ReLUSpec, ResidualSpec, SeqSpec};
    use crate::{conv, relu, residual};

    pub type StemSpec<const C1: usize, const C2: usize> = SeqSpec<
        SeqSpec<SeqSpec<ConvSpec<C1, 3, 3, 1, 1>, ReLUSpec>, ConvSpec<C2, 3, 3, 1, 1>>,
        ReLUSpec,
    >;

    #[derive(Clone, Copy, Debug, Default)]
    pub struct Stem<const C1: usize, const C2: usize>;

    impl<const C1: usize, const C2: usize> Fragment for Stem<C1, C2> {
        type Spec = StemSpec<C1, C2>;

        fn into_blueprint(self) -> Blueprint<Self::Spec> {
            conv::<C1, 3, 3, 1, 1>()
                .then(relu())
                .then(conv::<C2, 3, 3, 1, 1>())
                .then(relu())
        }
    }

    pub fn stem<const C1: usize, const C2: usize>() -> Stem<C1, C2> {
        Stem
    }

    pub type ResidualBlockBodySpec<const WIDTH: usize> =
        SeqSpec<SeqSpec<ConvSpec<WIDTH, 3, 3, 1, 1>, ReLUSpec>, ConvSpec<WIDTH, 3, 3, 1, 1>>;

    pub type ResidualBlockSpec<const WIDTH: usize> = ResidualSpec<ResidualBlockBodySpec<WIDTH>>;

    #[derive(Clone, Copy, Debug, Default)]
    pub struct ResidualBlock<const WIDTH: usize>;

    impl<const WIDTH: usize> Fragment for ResidualBlock<WIDTH> {
        type Spec = ResidualBlockSpec<WIDTH>;

        fn into_blueprint(self) -> Blueprint<Self::Spec> {
            residual(
                conv::<WIDTH, 3, 3, 1, 1>()
                    .then(relu())
                    .then(conv::<WIDTH, 3, 3, 1, 1>()),
            )
        }
    }

    pub fn residual_block<const WIDTH: usize>() -> ResidualBlock<WIDTH> {
        ResidualBlock
    }
}
