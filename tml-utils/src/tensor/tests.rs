use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::Float;
use crate::shape::TensorShape;

use super::Tensor;

type T3 = crate::shape!(2, 3, 4);

#[test]
fn indexing_borrows_and_owned_get_match_layout() {
    let mut t = Tensor::<T3>::zeros();
    let mut value = 0.0;
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                t.set([i, j, k], value);
                value += 1.0;
            }
        }
    }

    assert_eq!(*t.at([1, 2, 3]), 23.0);

    let row = t.get_ref(1);
    assert_eq!(*row.at([2, 3]), 23.0);

    let owned = t.get(1);
    assert_eq!(*owned.at([2, 3]), 23.0);

    let mut tmut = t.as_mut();
    let mut row_mut = tmut.get_mut(0);
    row_mut.set([0, 0], 99.0);
    assert_eq!(*t.at([0, 0, 0]), 99.0);
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn get_ref_panics_on_oob_index() {
    let t = Tensor::<T3>::zeros();
    let _ = t.get_ref(2);
}

#[test]
fn reshape_changes_shape_type_without_reordering_data() {
    let flat = Tensor::<crate::shape!(6)>::from_flat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let reshaped = flat.reshape::<crate::shape!(2, 3)>();
    assert_eq!(reshaped.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(*reshaped.at([1, 2]), 6.0);
}

#[test]
fn tensor_literal_infers_shape_and_layout() {
    let t = crate::tensor![[1.0, 2.0], [3.0, 4.0]];
    assert_eq!(t.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(*t.at([1, 0]), 3.0);
}

#[test]
fn labeled_tensor_literal_preserves_axis_names() {
    let t = crate::tensor! { shape: crate::shape!(row: 2, col: 2); [[1.0, 2.0], [3.0, 4.0]] };
    assert_eq!(
        <crate::shape!(row: 2, col: 2) as TensorShape>::axis_names(),
        vec![Some("row"), Some("col")]
    );
    let row = t.get_ref(1);
    assert_eq!(
        <crate::shape!(col: 2) as TensorShape>::axis_names(),
        vec![Some("col")]
    );
    assert_eq!(row.as_slice(), &[3.0, 4.0]);
}

#[test]
fn tensor_debug_uses_public_type_names() {
    let tensor = crate::tensor![[1.0, 2.0], [3.0, 4.0]];
    assert!(format!("{tensor:?}").starts_with("Tensor {"));
    let row = tensor.get_ref(1);
    assert!(format!("{row:?}").starts_with("TensorRef {"));
}

#[test]
fn elementwise_ops_and_reductions_work() {
    let lhs = crate::tensor![1.0, 2.0, 3.0];
    let rhs = crate::tensor![4.0, 5.0, 6.0];

    assert_eq!((&lhs + &rhs).as_slice(), &[5.0, 7.0, 9.0]);
    assert_eq!((&rhs - &lhs).as_slice(), &[3.0, 3.0, 3.0]);
    assert_eq!((&lhs * &rhs).as_slice(), &[4.0, 10.0, 18.0]);
    assert_eq!((lhs.clone() + 1.0).as_slice(), &[2.0, 3.0, 4.0]);
    assert_eq!(lhs.sum(), 6.0);
    assert_eq!(lhs.mean(), 2.0);
}

#[test]
fn dot_transpose_and_matmul_work() {
    let vec_a = crate::tensor![1.0, 2.0, 3.0];
    let vec_b = crate::tensor![4.0, 5.0, 6.0];
    assert_eq!(vec_a.dot(&vec_b), 32.0);

    let lhs = crate::tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let rhs = crate::tensor![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];
    let product = lhs.matmul(&rhs);
    assert_eq!(product.as_slice(), &[58.0, 64.0, 139.0, 154.0]);

    let transposed = lhs.transpose();
    assert_eq!(transposed.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn seeded_random_is_reproducible() {
    let a = Tensor::<crate::shape!(2, 3)>::random_with_seed(7);
    let b = Tensor::<crate::shape!(2, 3)>::random_with_seed(7);
    let c = Tensor::<crate::shape!(2, 3)>::random_with_seed(9);

    assert_eq!(a.as_slice(), b.as_slice());
    assert_ne!(a.as_slice(), c.as_slice());
}

#[test]
fn randomized_shape_stress_preserves_row_major_layout() {
    let mut tensor = Tensor::<crate::shape!(2, 3, 4)>::zeros();
    let mut rng = StdRng::seed_from_u64(42);

    for index in 0..tensor.len() {
        tensor.as_mut_slice()[index] = rng.random::<Float>();
    }

    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                let flat_index = i * 12 + j * 4 + k;
                assert_eq!(*tensor.at([i, j, k]), tensor.as_slice()[flat_index]);
            }
        }
    }

    let reshaped = tensor.clone().reshape::<crate::shape!(4, 3, 2)>();
    assert_eq!(tensor.as_slice(), reshaped.as_slice());
}

#[test]
fn borrowed_reshape_preserves_view_semantics() {
    let mut tensor = crate::tensor![[1.0, 2.0], [3.0, 4.0]];

    let flat_ref = tensor.as_ref().reshape::<crate::shape!(4)>();
    assert_eq!(flat_ref.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(*flat_ref.at([2]), 3.0);

    {
        let mut flat_mut = tensor.as_mut().reshape::<crate::shape!(4)>();
        flat_mut.set([3], 9.0);
    }

    assert_eq!(tensor.as_slice(), &[1.0, 2.0, 3.0, 9.0]);
}
