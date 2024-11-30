use crate::{assert, internal_prelude::*};
use linalg::matmul::triangular::BlockStructure;

pub fn reconstruct_scratch<I: Index, T: ComplexField>(
    nrows: usize,
    ncols: usize,
    par: Par,
) -> Result<StackReq, SizeOverflow> {
    _ = par;
    temp_mat_scratch::<T>(nrows, ncols)
}

#[track_caller]
pub fn reconstruct<I: Index, T: ComplexField>(
    out: MatMut<'_, T>,
    L: MatRef<'_, T>,
    U: MatRef<'_, T>,
    row_perm: PermRef<'_, I>,
    par: Par,
    stack: &mut DynStack,
) {
    let m = L.nrows();
    let n = U.ncols();
    let size = Ord::min(m, n);
    assert!(all(out.nrows() == m, out.ncols() == n, row_perm.len() == m));

    let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(m, n, stack) };
    let mut tmp = tmp.as_mat_mut();
    let mut out = out;

    linalg::matmul::triangular::matmul(
        tmp.rb_mut().get_mut(..size, ..size),
        BlockStructure::Rectangular,
        Accum::Replace,
        L.get(..size, ..size),
        BlockStructure::UnitTriangularLower,
        U.get(..size, ..size),
        BlockStructure::TriangularUpper,
        one(),
        par,
    );

    if m > n {
        linalg::matmul::triangular::matmul(
            tmp.rb_mut().get_mut(size.., ..size),
            BlockStructure::Rectangular,
            Accum::Replace,
            L.get(size.., ..size),
            BlockStructure::Rectangular,
            U.get(..size, ..size),
            BlockStructure::TriangularUpper,
            one(),
            par,
        );
    }
    if m < n {
        linalg::matmul::triangular::matmul(
            tmp.rb_mut().get_mut(..size, size..),
            BlockStructure::Rectangular,
            Accum::Replace,
            L.get(..size, ..size),
            BlockStructure::UnitTriangularLower,
            U.get(..size, size..),
            BlockStructure::Rectangular,
            one(),
            par,
        );
    }

    crate::perm::permute_rows(out.rb_mut(), tmp.rb(), row_perm.inverse());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, stats::prelude::*, utils::approx::*};
    use dyn_stack::GlobalMemBuffer;
    use linalg::lu::partial_pivoting::*;

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

            let mut LU = A.to_owned();
            let perm_fwd = &mut *vec![0usize; m];
            let perm_bwd = &mut *vec![0usize; m];

            let (_, perm) = factor::lu_in_place(
                LU.as_mut(),
                perm_fwd,
                perm_bwd,
                Par::Seq,
                DynStack::new(&mut {
                    GlobalMemBuffer::new(
                        factor::lu_in_place_scratch::<usize, c64>(m, n, Par::Seq, auto!(c64))
                            .unwrap(),
                    )
                }),
                auto!(c64),
            );

            let approx_eq = CwiseMat(ApproxEq::<c64>::eps() * (n as f64));

            let mut A_rec = Mat::zeros(m, n);
            reconstruct::reconstruct(
                A_rec.as_mut(),
                LU.as_ref(),
                LU.as_ref(),
                perm,
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    reconstruct::reconstruct_scratch::<usize, c64>(m, n, Par::Seq).unwrap(),
                )),
            );

            assert!(A_rec ~ A);
        }
    }
}
