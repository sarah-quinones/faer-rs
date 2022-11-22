use assert2::assert as fancy_assert;
use dyn_stack::DynStack;
use faer_core::{
    permutation::{permute_rows, PermutationIndicesRef},
    solve::*,
    temp_mat_uninit, ComplexField, Conj, MatMut, MatRef, Parallelism,
};
use reborrow::*;

fn solve_impl<T: ComplexField>(
    lu_factors: MatRef<'_, T>,
    conj_lhs: Conj,
    row_perm: PermutationIndicesRef<'_>,
    col_perm: PermutationIndicesRef<'_>,
    dst: MatMut<'_, T>,
    rhs: Option<MatRef<'_, T>>,
    conj_rhs: Conj,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    // LU = P(row_fwd) × A × P(col_fwd)

    // P(row_inv) ConjA?(LU) P(col_inv) X = ConjB?(B)
    // X = P(col_fwd) ConjA?(U)^-1 ConjA?(L)^-1 P(row_fwd) ConjB?(B)

    let n = lu_factors.ncols();
    let k = dst.ncols();

    temp_mat_uninit! {
        let (mut temp, _) = unsafe { temp_mat_uninit::<T>(n, k, stack) };
    }

    // temp <- P(row_fwd) B
    let src = match rhs {
        Some(rhs) => rhs,
        None => dst.rb(),
    };
    permute_rows(temp.rb_mut(), src, row_perm);

    // temp <- ConjA?(L)^-1 P(row_fwd) ConjB?(B)
    solve_unit_lower_triangular_in_place(
        lu_factors,
        conj_lhs,
        temp.rb_mut(),
        conj_rhs,
        parallelism,
    );

    // temp <- ConjA?(U)^-1 ConjA?(L)^-1 P(row_fwd) B
    solve_upper_triangular_in_place(lu_factors, conj_lhs, temp.rb_mut(), Conj::No, parallelism);

    permute_rows(dst, temp.rb(), col_perm);
}

pub fn solve_to<T: ComplexField>(
    dst: MatMut<'_, T>,
    lu_factors: MatRef<'_, T>,
    conj_lhs: Conj,
    row_perm: PermutationIndicesRef<'_>,
    col_perm: PermutationIndicesRef<'_>,
    rhs: MatRef<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    solve_impl(
        lu_factors,
        conj_lhs,
        row_perm,
        col_perm,
        dst,
        Some(rhs),
        conj_rhs,
        parallelism,
        stack,
    )
}

pub fn solve_in_place<T: ComplexField>(
    lu_factors: MatRef<'_, T>,
    conj_lhs: Conj,
    row_perm: PermutationIndicesRef<'_>,
    col_perm: PermutationIndicesRef<'_>,
    rhs: MatMut<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    solve_impl(
        lu_factors,
        conj_lhs,
        row_perm,
        col_perm,
        rhs,
        None,
        conj_rhs,
        parallelism,
        stack,
    );
}

#[cfg(test)]
mod tests {}
