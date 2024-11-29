use crate::{assert, internal_prelude::*};
use linalg::householder::apply_block_householder_sequence_transpose_on_the_right_in_place_scratch;

pub fn inverse_scratch<I: Index, T: ComplexField>(
    dim: usize,
    blocksize: usize,
    par: Par,
) -> Result<StackReq, SizeOverflow> {
    _ = par;
    StackReq::try_or(
        apply_block_householder_sequence_transpose_on_the_right_in_place_scratch::<T>(
            dim, blocksize, dim,
        )?,
        crate::perm::permute_cols_in_place_scratch::<I, T>(dim, dim)?,
    )
}

#[track_caller]
pub fn inverse<I: Index, T: ComplexField>(
    out: MatMut<'_, T>,
    QR: MatRef<'_, T>,
    H: MatRef<'_, T>,
    col_perm: PermRef<'_, I>,
    par: Par,
    stack: &mut DynStack,
) {
    // A P^-1 = Q R
    // A^-1 = P^-1 R^-1 Q^-1

    let n = QR.ncols();
    assert!(all(
        QR.nrows() == n,
        QR.ncols() == n,
        out.nrows() == n,
        out.ncols() == n,
        H.ncols() == n,
    ));

    let mut out = out;
    out.fill(zero());
    linalg::triangular_inverse::invert_upper_triangular(out.rb_mut(), QR, par);
    linalg::householder::apply_block_householder_sequence_transpose_on_the_right_in_place_with_conj(
        QR,
        H,
        Conj::Yes,
        out.rb_mut(),
        par,
        stack,
    );
    crate::perm::permute_rows_in_place(out.rb_mut(), col_perm.inverse(), stack);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, stats::prelude::*, utils::approx::*};
    use dyn_stack::GlobalMemBuffer;
    use linalg::qr::col_pivoting::*;

    #[test]
    fn test_inverse() {
        let rng = &mut StdRng::seed_from_u64(0);
        let n = 50;
        let A = CwiseMatDistribution {
            nrows: n,
            ncols: n,
            dist: ComplexDistribution::new(StandardNormal, StandardNormal),
        }
        .rand::<Mat<c64>>(rng);

        let mut QR = A.to_owned();
        let mut H = Mat::zeros(4, n);
        let col_perm_fwd = &mut *vec![0usize; n];
        let col_perm_bwd = &mut *vec![0usize; n];

        let (_, col_perm) = factor::qr_in_place(
            QR.as_mut(),
            H.as_mut(),
            col_perm_fwd,
            col_perm_bwd,
            Par::Seq,
            DynStack::new(&mut {
                GlobalMemBuffer::new(
                    factor::qr_in_place_scratch::<usize, c64>(n, n, 4, Par::Seq, auto!(c64))
                        .unwrap(),
                )
            }),
            auto!(c64),
        );

        let approx_eq = CwiseMat(ApproxEq::<c64>::eps() * (n as f64));

        let mut A_inv = Mat::zeros(n, n);
        inverse::inverse(
            A_inv.as_mut(),
            QR.as_ref(),
            H.as_ref(),
            col_perm,
            Par::Seq,
            DynStack::new(&mut GlobalMemBuffer::new(
                inverse::inverse_scratch::<usize, c64>(n, 4, Par::Seq).unwrap(),
            )),
        );

        assert!(A_inv * A ~ Mat::identity(n, n));
    }
}
