use crate::{assert, internal_prelude::*};

pub fn reconstruct_scratch<I: Index, T: ComplexField>(
    nrows: usize,
    ncols: usize,
    blocksize: usize,
    par: Par,
) -> Result<StackReq, SizeOverflow> {
    _ = par;
    StackReq::try_or(
        linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
            nrows, blocksize, ncols,
        )?,
        crate::perm::permute_cols_in_place_scratch::<I, T>(nrows, ncols)?,
    )
}

#[track_caller]
pub fn reconstruct<I: Index, T: ComplexField>(
    out: MatMut<'_, T>,
    QR: MatRef<'_, T>,
    H: MatRef<'_, T>,
    col_perm: PermRef<'_, I>,
    par: Par,
    stack: &mut DynStack,
) {
    let (m, n) = QR.shape();
    let size = Ord::min(m, n);
    assert!(all(out.nrows() == m, out.ncols() == n, H.ncols() == size));

    let mut out = out;
    out.fill(zero());
    out.copy_from_triangular_upper(QR);

    linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
        QR,
        H,
        Conj::No,
        out.rb_mut(),
        par,
        stack,
    );
    crate::perm::permute_cols_in_place(out.rb_mut(), col_perm.inverse(), stack);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, stats::prelude::*, utils::approx::*};
    use dyn_stack::GlobalMemBuffer;
    use linalg::qr::col_pivoting::*;

    #[test]
    fn test_reconstruct() {
        let rng = &mut StdRng::seed_from_u64(0);
        for (m, n) in [(100, 50), (50, 100)] {
            let A = CwiseMatDistribution {
                nrows: m,
                ncols: n,
                dist: ComplexDistribution::new(StandardNormal, StandardNormal),
            }
            .rand::<Mat<c64>>(rng);

            let mut QR = A.to_owned();
            let mut H = Mat::zeros(4, Ord::min(m, n));
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
                        factor::qr_in_place_scratch::<usize, c64>(m, n, 4, Par::Seq, auto!(c64))
                            .unwrap(),
                    )
                }),
                auto!(c64),
            );

            let approx_eq = CwiseMat(ApproxEq::<c64>::eps() * (n as f64));

            let mut A_rec = Mat::zeros(m, n);
            reconstruct::reconstruct(
                A_rec.as_mut(),
                QR.as_ref(),
                H.as_ref(),
                col_perm,
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    reconstruct::reconstruct_scratch::<usize, c64>(m, n, 4, Par::Seq).unwrap(),
                )),
            );

            assert!(A_rec ~ A);
        }
    }
}
