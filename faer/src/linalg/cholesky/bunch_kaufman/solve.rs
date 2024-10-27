use linalg::triangular_solve::{
    solve_unit_lower_triangular_in_place_with_conj, solve_unit_upper_triangular_in_place_with_conj,
};

use crate::{assert, internal_prelude::*, perm::permute_rows};

/// Solving a linear system using the decomposition.

/// Computes the size and alignment of required workspace for solving a linear system defined by
/// a matrix in place, given its Bunch-Kaufman decomposition.
#[track_caller]
pub fn solve_in_place_scratch<I: Index, T: ComplexField>(
    dim: usize,
    rhs_ncols: usize,
    parallelism: Par,
) -> Result<StackReq, SizeOverflow> {
    let _ = parallelism;
    temp_mat_scratch::<T>(dim, rhs_ncols)
}

/// Given the Bunch-Kaufman factors of a matrix $A$ and a matrix $B$ stored in `rhs`, this
/// function computes the solution of the linear system:
/// $$\text{Op}_A(A)X = B.$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of
/// `conj`.
///
/// The solution of the linear system is stored in `rhs`.
///
/// # Panics
///
/// - Panics if `lb_factors` is not a square matrix.
/// - Panics if `subdiag` is not a column vector with the same number of rows as the dimension of
///   `lb_factors`.
/// - Panics if `rhs` doesn't have the same number of rows as the dimension of `lb_factors`.
/// - Panics if the provided memory in `stack` is insufficient (see [`solve_in_place_scratch`]).
#[track_caller]
#[math]
pub fn solve_in_place_with_conj<I: Index, T: ComplexField>(
    lb_factors: MatRef<'_, T>,
    subdiag: ColRef<'_, T>,
    conj_lb: Conj,
    perm: PermRef<'_, I>,
    rhs: MatMut<'_, T>,
    parallelism: Par,
    stack: &mut DynStack,
) {
    let n = lb_factors.nrows();
    let k = rhs.ncols();

    assert!(all(
        lb_factors.nrows() == lb_factors.ncols(),
        rhs.nrows() == n,
        subdiag.nrows() == n,
        perm.len() == n
    ));

    let a = lb_factors;
    let par = parallelism;
    let not_conj = conj_lb.compose(Conj::Yes);

    let mut rhs = rhs;
    let mut x = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack).0 };
    let mut x = x.as_mat_mut();

    permute_rows(x.rb_mut(), rhs.rb(), perm);
    solve_unit_lower_triangular_in_place_with_conj(a, conj_lb, x.rb_mut(), par);

    let mut i = 0;
    while i < n {
        let i0 = i;
        let i1 = i + 1;

        if subdiag[i] == zero() {
            let d_inv = recip(real(a[(i, i)]));
            for j in 0..k {
                x[(i, j)] = mul_real(x[(i, j)], d_inv);
            }
            i += 1;
        } else {
            let mut akp1k = copy(subdiag[i0]);
            if matches!(conj_lb, Conj::Yes) {
                akp1k = conj(akp1k);
            }
            akp1k = recip(akp1k);
            let (ak, akp1) = (
                mul_real(conj(akp1k), real(a[(i0, i0)])),
                mul_real(akp1k, real(a[(i1, i1)])),
            );

            let denom = real(recip(ak * akp1 - one()));

            for j in 0..k {
                let (xk, xkp1) = (
                    //
                    x[(i0, j)] * conj(akp1k),
                    x[(i1, j)] * akp1k,
                );

                let (xk, xkp1) = (
                    mul_real((akp1 * xk - xkp1), denom),
                    mul_real((ak * xkp1 - xk), denom),
                );

                x[(i, j)] = xk;
                x[(i + 1, j)] = xkp1;
            }

            i += 2;
        }
    }

    solve_unit_upper_triangular_in_place_with_conj(a.transpose(), not_conj, x.rb_mut(), par);
    permute_rows(rhs.rb_mut(), x.rb(), perm.inverse());
}
