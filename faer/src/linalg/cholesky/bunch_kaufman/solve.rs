use linalg::triangular_solve::{
    solve_unit_lower_triangular_in_place_with_conj, solve_unit_upper_triangular_in_place_with_conj,
};

use crate::{assert, internal_prelude::*, perm::permute_rows};

/// Solving a linear system using the decomposition.

/// Computes the size and alignment of required workspace for solving a linear system defined by
/// a matrix in place, given its Bunch-Kaufman decomposition.
#[track_caller]
pub fn solve_in_place_scratch<I: Index, C: ComplexContainer, T: ComplexField<C>>(
    dim: usize,
    rhs_ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = parallelism;
    temp_mat_scratch::<C, T>(dim, rhs_ncols)
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
pub fn solve_in_place_with_conj<I: Index, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    lb_factors: MatRef<'_, C, T>,
    subdiag: ColRef<'_, C, T>,
    conj: Conj,
    perm: PermRef<'_, I>,
    rhs: MatMut<'_, C, T>,
    parallelism: Parallelism,
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
    let not_conj = conj.compose(Conj::Yes);

    let mut rhs = rhs;
    let mut x = unsafe { temp_mat_uninit::<C, T, _, _>(ctx, n, k, stack).0 };
    let mut x = x.as_mat_mut();

    permute_rows(ctx, x.rb_mut(), rhs.rb(), perm);
    solve_unit_lower_triangular_in_place_with_conj(ctx, a, conj, x.rb_mut(), par);
    help!(C);

    let mut i = 0;
    while i < n {
        let i0 = i;
        let i1 = i + 1;

        if math(subdiag[i] == zero()) {
            let d_inv = math(re.recip(real(a[(i, i)])));
            for j in 0..k {
                write1!(x[(i, j)] = math(mul_real(x[(i, j)], d_inv)));
            }
            i += 1;
        } else {
            let mut akp1k = math(copy(subdiag[i0]));
            if matches!(conj, Conj::Yes) {
                akp1k = math.conj(akp1k);
            }
            akp1k = math.recip(akp1k);
            let (ak, akp1) = math((
                mul_real(conj(akp1k), real(a[(i0, i0)])),
                mul_real(akp1k, real(a[(i1, i1)])),
            ));

            let denom = math(real(recip(ak * akp1 - one())));

            for j in 0..k {
                let (xk, xkp1) = math((
                    //
                    x[(i0, j)] * conj(akp1k),
                    x[(i1, j)] * akp1k,
                ));

                let (xk, xkp1) = math((
                    //
                    mul_real((akp1 * xk - xkp1), denom),
                    mul_real((ak * xkp1 - xk), denom),
                ));

                write1!(x[(i, j)] = xk);
                write1!(x[(i + 1, j)] = xkp1);
            }

            i += 2;
        }
    }

    solve_unit_upper_triangular_in_place_with_conj(ctx, a.transpose(), not_conj, x.rb_mut(), par);
    permute_rows(ctx, rhs.rb_mut(), x.rb(), perm.inverse());
}
