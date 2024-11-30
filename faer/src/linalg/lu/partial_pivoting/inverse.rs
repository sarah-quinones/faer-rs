use crate::{assert, internal_prelude::*};
use linalg::matmul::triangular::BlockStructure;

pub fn inverse_scratch<I: Index, T: ComplexField>(
    dim: usize,
    par: Par,
) -> Result<StackReq, SizeOverflow> {
    _ = par;
    temp_mat_scratch::<T>(dim, dim)
}

#[track_caller]
pub fn inverse<I: Index, T: ComplexField>(
    out: MatMut<'_, T>,
    L: MatRef<'_, T>,
    U: MatRef<'_, T>,
    row_perm: PermRef<'_, I>,
    par: Par,
    stack: &mut DynStack,
) {
    // A = P^-1 L U
    // A^-1 = U^-1 L^-1 P

    let n = L.ncols();
    assert!(all(
        L.nrows() == n,
        L.ncols() == n,
        U.nrows() == n,
        U.ncols() == n,
        out.nrows() == n,
        out.ncols() == n,
        row_perm.len() == n,
    ));

    let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
    let mut tmp = tmp.as_mat_mut();
    let mut out = out;

    linalg::triangular_inverse::invert_unit_lower_triangular(out.rb_mut(), L, par);
    linalg::triangular_inverse::invert_upper_triangular(out.rb_mut(), U, par);

    linalg::matmul::triangular::matmul(
        tmp.rb_mut(),
        BlockStructure::Rectangular,
        Accum::Replace,
        out.rb(),
        BlockStructure::TriangularUpper,
        out.rb(),
        BlockStructure::UnitTriangularLower,
        one(),
        par,
    );
    crate::perm::permute_cols(out.rb_mut(), tmp.rb(), row_perm.inverse());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, stats::prelude::*, utils::approx::*};
    use dyn_stack::GlobalMemBuffer;
    use linalg::lu::partial_pivoting::*;

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

        let mut LU = A.to_owned();
        let perm_fwd = &mut *vec![0usize; n];
        let perm_bwd = &mut *vec![0usize; n];

        let (_, perm) = factor::lu_in_place(
            LU.as_mut(),
            perm_fwd,
            perm_bwd,
            Par::Seq,
            DynStack::new(&mut {
                GlobalMemBuffer::new(
                    factor::lu_in_place_scratch::<usize, c64>(n, n, Par::Seq, auto!(c64)).unwrap(),
                )
            }),
            auto!(c64),
        );

        let approx_eq = CwiseMat(ApproxEq::<c64>::eps() * (n as f64));

        let mut A_inv = Mat::zeros(n, n);
        inverse::inverse(
            A_inv.as_mut(),
            LU.as_ref(),
            LU.as_ref(),
            perm,
            Par::Seq,
            DynStack::new(&mut GlobalMemBuffer::new(
                inverse::inverse_scratch::<usize, c64>(n, Par::Seq).unwrap(),
            )),
        );

        assert!(&A_inv * &A ~ Mat::identity(n, n));
    }
}
