use super::{Blueprint, SeqSpec, SharedSpec, TransformSpec};
use crate::shape::TensorShape;

pub trait Fragment: Clone {
    type Spec: Clone;

    fn into_blueprint(self) -> Blueprint<Self::Spec>;
}

impl<Spec> Fragment for Blueprint<Spec>
where
    Spec: Clone,
{
    type Spec = Spec;

    fn into_blueprint(self) -> Blueprint<Self::Spec> {
        self
    }
}

pub fn into_blueprint<F>(fragment: F) -> Blueprint<F::Spec>
where
    F: Fragment,
{
    fragment.into_blueprint()
}

pub fn share_fragment<F>(fragment: &F) -> Blueprint<SharedSpec<F::Spec>>
where
    F: Fragment,
{
    let blueprint = fragment.clone().into_blueprint();
    Blueprint::new(SharedSpec {
        id: fragment as *const F as *const () as usize,
        inner: blueprint.into_inner(),
    })
}

pub trait FragmentExt: Fragment + Sized {
    fn then_fragment<Next>(self, next: Next) -> Blueprint<SeqSpec<Self::Spec, Next::Spec>>
    where
        Next: Fragment,
    {
        self.into_blueprint().then(next.into_blueprint())
    }

    fn then_fragment_checked<InputShape, Next>(
        self,
        next: Next,
    ) -> Blueprint<SeqSpec<Self::Spec, Next::Spec>>
    where
        InputShape: TensorShape + 'static,
        Self::Spec: TransformSpec<InputShape>,
        Next: Fragment,
        Next::Spec: TransformSpec<<Self::Spec as TransformSpec<InputShape>>::OutputShape>,
    {
        self.into_blueprint().then(next.into_blueprint())
    }
}

impl<F> FragmentExt for F where F: Fragment {}
