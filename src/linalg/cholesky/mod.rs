//! Low level implementation of the various Cholesky-like decompositions.

use crate::{assert, perm::PermRef, ComplexField, Index, MatRef, SignedIndex};
use core::cmp::Ordering;

pub mod bunch_kaufman;
pub mod ldlt_diagonal;
pub mod llt;

#[cfg(feature = "unstable")]
#[cfg_attr(docsrs, doc(cfg(feature = "unstable")))]
pub(crate) mod piv_llt;

/// Computes a permutation that reduces the chance of numerical errors during the $LDL^H$
/// factorization with diagonal $D$, then stores the result in `perm_indices` and
/// `perm_inv_indices`.
#[track_caller]
pub fn compute_cholesky_permutation<'a, E: ComplexField, I: Index>(
    perm_indices: &'a mut [I],
    perm_inv_indices: &'a mut [I],
    matrix: MatRef<'_, E>,
) -> PermRef<'a, I> {
    let n = matrix.nrows();
    let truncate = <I::Signed as SignedIndex>::truncate;
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
        *p = I::from_signed(truncate(i));
    }

    perm_indices.sort_unstable_by(move |&i, &j| {
        let lhs = matrix
            .read(i.to_signed().zx(), i.to_signed().zx())
            .faer_abs();
        let rhs = matrix
            .read(j.to_signed().zx(), j.to_signed().zx())
            .faer_abs();
        let cmp = rhs.partial_cmp(&lhs);
        if let Some(cmp) = cmp {
            cmp
        } else {
            Ordering::Equal
        }
    });

    for (i, p) in perm_indices.iter().copied().enumerate() {
        perm_inv_indices[p.to_signed().zx()] = I::from_signed(truncate(i));
    }

    unsafe { PermRef::new_unchecked(perm_indices, perm_inv_indices, n) }
}
