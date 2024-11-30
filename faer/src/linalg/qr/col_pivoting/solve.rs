use crate::{assert, internal_prelude::*};

pub fn solve_lstsq_in_place_scratch<I: Index, T: ComplexField>(
    qr_nrows: usize,
    qr_ncols: usize,
    qr_blocksize: usize,
    rhs_ncols: usize,
    par: Par,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_and(
        linalg::qr::no_pivoting::solve::solve_lstsq_in_place_scratch::<T>(
            qr_nrows,
            qr_ncols,
            qr_blocksize,
            rhs_ncols,
            par,
        )?,
        crate::perm::permute_rows_in_place_scratch::<I, T>(qr_ncols, rhs_ncols)?,
    )
}

pub fn solve_in_place_scratch<I: Index, T: ComplexField>(
    qr_dim: usize,
    qr_blocksize: usize,
    rhs_ncols: usize,
    par: Par,
) -> Result<StackReq, SizeOverflow> {
    solve_lstsq_in_place_scratch::<I, T>(qr_dim, qr_dim, qr_blocksize, rhs_ncols, par)
}

pub fn solve_transpose_in_place_scratch<I: Index, T: ComplexField>(
    qr_dim: usize,
    qr_blocksize: usize,
    rhs_ncols: usize,
    par: Par,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_and(
        linalg::qr::no_pivoting::solve::solve_transpose_in_place_scratch::<T>(
            qr_dim,
            qr_blocksize,
            rhs_ncols,
            par,
        )?,
        crate::perm::permute_rows_in_place_scratch::<I, T>(qr_dim, rhs_ncols)?,
    )
}

#[track_caller]
pub fn solve_lstsq_in_place_with_conj<I: Index, T: ComplexField>(
    Q_basis: MatRef<'_, T>,
    Q_coeff: MatRef<'_, T>,
    R: MatRef<'_, T>,
    col_perm: PermRef<'_, I>,
    conj_QR: Conj,
    rhs: MatMut<'_, T>,
    par: Par,
    stack: &mut DynStack,
) {
    let m = Q_basis.nrows();
    let n = Q_basis.ncols();
    let size = Ord::min(m, n);
    let mut rhs = rhs;

    linalg::qr::no_pivoting::solve::solve_lstsq_in_place_with_conj(
        Q_basis,
        Q_coeff,
        R,
        conj_QR,
        rhs.rb_mut(),
        par,
        stack,
    );

    crate::perm::permute_rows_in_place(rhs.get_mut(..size, ..), col_perm.inverse(), stack);
}

#[track_caller]
pub fn solve_lstsq_in_place<I: Index, T: ComplexField, C: Conjugate<Canonical = T>>(
    Q_basis: MatRef<'_, C>,
    Q_coeff: MatRef<'_, C>,
    R: MatRef<'_, C>,
    col_perm: PermRef<'_, I>,
    rhs: MatMut<'_, T>,
    par: Par,
    stack: &mut DynStack,
) {
    solve_lstsq_in_place_with_conj(
        Q_basis.canonical(),
        Q_coeff.canonical(),
        R.canonical(),
        col_perm,
        Conj::get::<C>(),
        rhs,
        par,
        stack,
    );
}

#[track_caller]
pub fn solve_in_place_with_conj<I: Index, T: ComplexField>(
    Q_basis: MatRef<'_, T>,
    Q_coeff: MatRef<'_, T>,
    R: MatRef<'_, T>,
    col_perm: PermRef<'_, I>,
    conj_QR: Conj,
    rhs: MatMut<'_, T>,
    par: Par,
    stack: &mut DynStack,
) {
    let n = Q_basis.nrows();
    let blocksize = Q_coeff.nrows();

    assert!(all(
        blocksize > 0,
        rhs.nrows() == n,
        Q_basis.nrows() == n,
        Q_basis.ncols() == n,
        Q_coeff.ncols() == n,
        R.nrows() == n,
        R.ncols() == n,
    ));

    solve_lstsq_in_place_with_conj(Q_basis, Q_coeff, R, col_perm, conj_QR, rhs, par, stack);
}

#[track_caller]
pub fn solve_in_place<I: Index, T: ComplexField, C: Conjugate<Canonical = T>>(
    Q_basis: MatRef<'_, C>,
    Q_coeff: MatRef<'_, C>,
    R: MatRef<'_, C>,
    col_perm: PermRef<'_, I>,
    rhs: MatMut<'_, T>,
    par: Par,
    stack: &mut DynStack,
) {
    solve_in_place_with_conj(
        Q_basis.canonical(),
        Q_coeff.canonical(),
        R.canonical(),
        col_perm,
        Conj::get::<C>(),
        rhs,
        par,
        stack,
    );
}

#[track_caller]
pub fn solve_transpose_in_place_with_conj<I: Index, T: ComplexField>(
    Q_basis: MatRef<'_, T>,
    Q_coeff: MatRef<'_, T>,
    R: MatRef<'_, T>,
    col_perm: PermRef<'_, I>,
    conj_QR: Conj,
    rhs: MatMut<'_, T>,
    par: Par,
    stack: &mut DynStack,
) {
    let n = Q_basis.nrows();
    let blocksize = Q_coeff.nrows();

    assert!(all(
        blocksize > 0,
        rhs.nrows() == n,
        Q_basis.nrows() == n,
        Q_basis.ncols() == n,
        Q_coeff.ncols() == n,
        R.nrows() == n,
        R.ncols() == n,
    ));

    let mut rhs = rhs;

    crate::perm::permute_rows_in_place(rhs.rb_mut(), col_perm, stack);
    linalg::qr::no_pivoting::solve::solve_transpose_in_place_with_conj(
        Q_basis, Q_coeff, R, conj_QR, rhs, par, stack,
    );
}

#[track_caller]
pub fn solve_transpose_in_place<I: Index, T: ComplexField, C: Conjugate<Canonical = T>>(
    Q_basis: MatRef<'_, C>,
    Q_coeff: MatRef<'_, C>,
    R: MatRef<'_, C>,
    col_perm: PermRef<'_, I>,
    rhs: MatMut<'_, T>,
    par: Par,
    stack: &mut DynStack,
) {
    solve_transpose_in_place_with_conj(
        Q_basis.canonical(),
        Q_coeff.canonical(),
        R.canonical(),
        col_perm,
        Conj::get::<C>(),
        rhs,
        par,
        stack,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, stats::prelude::*, utils::approx::*};
    use dyn_stack::GlobalMemBuffer;
    use linalg::qr::col_pivoting::*;

    #[test]
    fn test_lstsq() {
        let rng = &mut StdRng::seed_from_u64(0);
        let m = 100;
        let n = 50;
        let k = 3;

        let A = CwiseMatDistribution {
            nrows: m,
            ncols: n,
            dist: ComplexDistribution::new(StandardNormal, StandardNormal),
        }
        .rand::<Mat<c64>>(rng);

        let B = CwiseMatDistribution {
            nrows: m,
            ncols: k,
            dist: ComplexDistribution::new(StandardNormal, StandardNormal),
        }
        .rand::<Mat<c64>>(rng);

        let mut QR = A.to_owned();
        let mut Q_coeff = Mat::zeros(4, n);

        let col_perm_fwd = &mut *vec![0usize; n];
        let col_perm_bwd = &mut *vec![0usize; n];

        let (_, col_perm) = factor::qr_in_place(
            QR.as_mut(),
            Q_coeff.as_mut(),
            col_perm_fwd,
            col_perm_bwd,
            Par::Seq,
            DynStack::new(&mut GlobalMemBuffer::new(
                factor::qr_in_place_scratch::<usize, c64>(m, n, 4, Par::Seq, auto!(c64)).unwrap(),
            )),
            auto!(c64),
        );

        let approx_eq = CwiseMat(ApproxEq::<c64>::eps() * (n as f64));

        {
            let mut X = B.to_owned();
            solve::solve_lstsq_in_place(
                QR.as_ref(),
                Q_coeff.as_ref(),
                QR.as_ref(),
                col_perm,
                X.as_mut(),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    solve::solve_lstsq_in_place_scratch::<usize, c64>(m, n, 4, k, Par::Seq)
                        .unwrap(),
                )),
            );

            let X = X.get(..n, ..);

            assert!(A.adjoint() * &A * &X ~ A.adjoint() * &B);
        }

        {
            let mut X = B.to_owned();
            solve::solve_lstsq_in_place(
                QR.conjugate(),
                Q_coeff.conjugate(),
                QR.conjugate(),
                col_perm,
                X.as_mut(),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    solve::solve_lstsq_in_place_scratch::<usize, c64>(m, n, 4, k, Par::Seq)
                        .unwrap(),
                )),
            );

            let X = X.get(..n, ..);
            assert!(A.transpose() * A.conjugate() * &X ~ A.transpose() * &B);
        }
    }

    #[test]
    fn test_solve() {
        let rng = &mut StdRng::seed_from_u64(0);
        let n = 50;
        let k = 3;

        let A = CwiseMatDistribution {
            nrows: n,
            ncols: n,
            dist: ComplexDistribution::new(StandardNormal, StandardNormal),
        }
        .rand::<Mat<c64>>(rng);

        let B = CwiseMatDistribution {
            nrows: n,
            ncols: k,
            dist: ComplexDistribution::new(StandardNormal, StandardNormal),
        }
        .rand::<Mat<c64>>(rng);

        let mut QR = A.to_owned();
        let mut Q_coeff = Mat::zeros(4, n);

        let col_perm_fwd = &mut *vec![0usize; n];
        let col_perm_bwd = &mut *vec![0usize; n];

        let (_, col_perm) = factor::qr_in_place(
            QR.as_mut(),
            Q_coeff.as_mut(),
            col_perm_fwd,
            col_perm_bwd,
            Par::Seq,
            DynStack::new(&mut GlobalMemBuffer::new(
                factor::qr_in_place_scratch::<usize, c64>(n, n, 4, Par::Seq, auto!(c64)).unwrap(),
            )),
            auto!(c64),
        );

        let approx_eq = CwiseMat(ApproxEq::<c64>::eps() * (n as f64));

        {
            let mut X = B.to_owned();
            solve::solve_in_place(
                QR.as_ref(),
                Q_coeff.as_ref(),
                QR.as_ref(),
                col_perm,
                X.as_mut(),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    solve::solve_in_place_scratch::<usize, c64>(n, 4, k, Par::Seq).unwrap(),
                )),
            );

            assert!(&A * &X ~ B);
        }

        {
            let mut X = B.to_owned();
            solve::solve_in_place(
                QR.conjugate(),
                Q_coeff.conjugate(),
                QR.conjugate(),
                col_perm,
                X.as_mut(),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    solve::solve_in_place_scratch::<usize, c64>(n, 4, k, Par::Seq).unwrap(),
                )),
            );

            assert!(A.conjugate() * &X ~ B);
        }

        {
            let mut X = B.to_owned();
            solve::solve_transpose_in_place(
                QR.as_ref(),
                Q_coeff.as_ref(),
                QR.as_ref(),
                col_perm,
                X.as_mut(),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    solve::solve_transpose_in_place_scratch::<usize, c64>(n, 4, k, Par::Seq)
                        .unwrap(),
                )),
            );

            assert!(A.transpose() * &X ~ B);
        }

        {
            let mut X = B.to_owned();
            solve::solve_transpose_in_place(
                QR.conjugate(),
                Q_coeff.conjugate(),
                QR.conjugate(),
                col_perm,
                X.as_mut(),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    solve::solve_transpose_in_place_scratch::<usize, c64>(n, 4, k, Par::Seq)
                        .unwrap(),
                )),
            );

            assert!(A.adjoint() * &X ~ B);
        }
    }
}
