use crate::{
    assert,
    internal_prelude::*,
    perm::{permute_rows_in_place, permute_rows_in_place_scratch},
};

pub fn solve_in_place_scratch<I: Index, T: ComplexField>(
    LU_dim: usize,
    rhs_ncols: usize,
    par: Par,
) -> Result<StackReq, SizeOverflow> {
    _ = par;
    permute_rows_in_place_scratch::<I, T>(LU_dim, rhs_ncols)
}

pub fn solve_transpose_in_place_scratch<I: Index, T: ComplexField>(
    LU_dim: usize,
    rhs_ncols: usize,
    par: Par,
) -> Result<StackReq, SizeOverflow> {
    _ = par;
    permute_rows_in_place_scratch::<I, T>(LU_dim, rhs_ncols)
}

#[track_caller]
pub fn solve_in_place_with_conj<I: Index, T: ComplexField>(
    LU: MatRef<'_, T>,
    row_perm: PermRef<'_, I>,
    col_perm: PermRef<'_, I>,
    conj_LU: Conj,
    rhs: MatMut<'_, T>,
    par: Par,
    stack: &mut DynStack,
) {
    // LU = P A Q^-1
    // P^-1 LU Q = A
    // A^-1 = Q^-1 U^-1 L^-1 P

    let n = LU.nrows();

    assert!(all(
        LU.nrows() == n,
        LU.ncols() == n,
        row_perm.len() == n,
        col_perm.len() == n,
        rhs.nrows() == n,
    ));

    let mut rhs = rhs;
    permute_rows_in_place(rhs.rb_mut(), row_perm, stack);
    linalg::triangular_solve::solve_unit_lower_triangular_in_place_with_conj(
        LU,
        conj_LU,
        rhs.rb_mut(),
        par,
    );
    linalg::triangular_solve::solve_upper_triangular_in_place_with_conj(
        LU,
        conj_LU,
        rhs.rb_mut(),
        par,
    );
    permute_rows_in_place(rhs.rb_mut(), col_perm.inverse(), stack);
}

#[track_caller]
pub fn solve_transpose_in_place_with_conj<I: Index, T: ComplexField>(
    LU: MatRef<'_, T>,
    row_perm: PermRef<'_, I>,
    col_perm: PermRef<'_, I>,
    conj_LU: Conj,
    rhs: MatMut<'_, T>,
    par: Par,
    stack: &mut DynStack,
) {
    // LU = P A Q^-1
    // A^-1 = Q^-1 U^-1 L^-1 P
    // A^-T = P^-1 L^-T U^-T Q

    let n = LU.nrows();

    assert!(all(
        LU.nrows() == n,
        LU.ncols() == n,
        row_perm.len() == n,
        col_perm.len() == n,
        rhs.nrows() == n,
    ));

    let mut rhs = rhs;
    permute_rows_in_place(rhs.rb_mut(), col_perm, stack);
    linalg::triangular_solve::solve_lower_triangular_in_place_with_conj(
        LU.transpose(),
        conj_LU,
        rhs.rb_mut(),
        par,
    );
    linalg::triangular_solve::solve_unit_upper_triangular_in_place_with_conj(
        LU.transpose(),
        conj_LU,
        rhs.rb_mut(),
        par,
    );
    permute_rows_in_place(rhs.rb_mut(), row_perm.inverse(), stack);
}

#[track_caller]
pub fn solve_in_place<I: Index, T: ComplexField, C: Conjugate<Canonical = T>>(
    LU: MatRef<'_, C>,
    row_perm: PermRef<'_, I>,
    col_perm: PermRef<'_, I>,
    rhs: MatMut<'_, T>,
    par: Par,
    stack: &mut DynStack,
) {
    solve_in_place_with_conj(
        LU.canonical(),
        row_perm,
        col_perm,
        Conj::get::<C>(),
        rhs,
        par,
        stack,
    )
}

#[track_caller]
pub fn solve_transpose_in_place<I: Index, T: ComplexField, C: Conjugate<Canonical = T>>(
    LU: MatRef<'_, C>,
    row_perm: PermRef<'_, I>,
    col_perm: PermRef<'_, I>,
    rhs: MatMut<'_, T>,
    par: Par,
    stack: &mut DynStack,
) {
    solve_transpose_in_place_with_conj(
        LU.canonical(),
        row_perm,
        col_perm,
        Conj::get::<C>(),
        rhs,
        par,
        stack,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, stats::prelude::*, utils::approx::*};
    use dyn_stack::GlobalMemBuffer;
    use linalg::lu::full_pivoting::*;

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

        let mut LU = A.to_owned();
        let row_perm_fwd = &mut *vec![0usize; n];
        let row_perm_bwd = &mut *vec![0usize; n];
        let col_perm_fwd = &mut *vec![0usize; n];
        let col_perm_bwd = &mut *vec![0usize; n];

        let (_, row_perm, col_perm) = factor::lu_in_place(
            LU.as_mut(),
            row_perm_fwd,
            row_perm_bwd,
            col_perm_fwd,
            col_perm_bwd,
            Par::Seq,
            DynStack::new(&mut {
                GlobalMemBuffer::new(
                    factor::lu_in_place_scratch::<usize, c64>(n, n, Par::Seq, auto!(c64)).unwrap(),
                )
            }),
            auto!(c64),
        );

        let approx_eq = CwiseMat(ApproxEq::<c64>::eps() * (n as f64));

        {
            let mut X = B.to_owned();
            solve::solve_in_place(
                LU.as_ref(),
                row_perm,
                col_perm,
                X.as_mut(),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    solve::solve_in_place_scratch::<usize, c64>(n, k, Par::Seq).unwrap(),
                )),
            );

            assert!(&A * &X ~ B);
        }

        {
            let mut X = B.to_owned();
            solve::solve_in_place(
                LU.as_ref(),
                row_perm,
                col_perm,
                X.as_mut(),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    solve::solve_in_place_scratch::<usize, c64>(n, k, Par::Seq).unwrap(),
                )),
            );

            assert!(&A * &X ~ B);
        }
        {
            let mut X = B.to_owned();
            solve::solve_transpose_in_place(
                LU.as_ref(),
                row_perm,
                col_perm,
                X.as_mut(),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    solve::solve_in_place_scratch::<usize, c64>(n, k, Par::Seq).unwrap(),
                )),
            );

            assert!(A.transpose() * &X ~ B);
        }
        {
            let mut X = B.to_owned();
            solve::solve_in_place(
                LU.conjugate(),
                row_perm,
                col_perm,
                X.as_mut(),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    solve::solve_in_place_scratch::<usize, c64>(n, k, Par::Seq).unwrap(),
                )),
            );

            assert!(A.conjugate() * &X ~ B);
        }
        {
            let mut X = B.to_owned();
            solve::solve_transpose_in_place(
                LU.conjugate(),
                row_perm,
                col_perm,
                X.as_mut(),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    solve::solve_in_place_scratch::<usize, c64>(n, k, Par::Seq).unwrap(),
                )),
            );

            assert!(A.adjoint() * &X ~ B);
        }
    }
}
