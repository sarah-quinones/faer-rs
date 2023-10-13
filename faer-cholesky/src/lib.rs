#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
use assert2::assert;
use core::cmp::Ordering;
use faer_core::{permutation::PermutationMut, ComplexField, MatRef};

pub mod ldlt_diagonal;
pub mod llt;

/// Computes a permutation that reduces the chance of numerical errors during the $LDL^H$
/// factorization with diagonal $D$, then stores the result in `perm_indices` and
/// `perm_inv_indices`.
#[track_caller]
pub fn compute_cholesky_permutation<'a, E: ComplexField>(
    perm_indices: &'a mut [usize],
    perm_inv_indices: &'a mut [usize],
    matrix: MatRef<'_, E>,
) -> PermutationMut<'a> {
    let n = matrix.nrows();
    assert!(
        matrix.nrows() == matrix.ncols(),
        "input matrix must be square",
    );
    assert!(
        perm_indices.len() == n,
        "length of permutation must be equal to the matrix dimension",
    );
    assert!(
        perm_inv_indices.len() == n,
        "length of inverse permutation must be equal to the matrix dimension",
    );

    for (i, p) in perm_indices.iter_mut().enumerate() {
        *p = i;
    }

    perm_indices.sort_unstable_by(move |&i, &j| {
        let lhs = matrix.read(i, i).faer_abs();
        let rhs = matrix.read(j, j).faer_abs();
        let cmp = rhs.partial_cmp(&lhs);
        if let Some(cmp) = cmp {
            cmp
        } else {
            Ordering::Equal
        }
    });

    for (i, p) in perm_indices.iter().copied().enumerate() {
        perm_inv_indices[p] = i;
    }

    unsafe { PermutationMut::new_unchecked(perm_indices, perm_inv_indices) }
}
