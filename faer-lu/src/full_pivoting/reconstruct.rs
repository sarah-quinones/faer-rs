use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{
    mul::triangular, permutation::PermutationIndicesRef, temp_mat_req, temp_mat_uninit,
    ComplexField, Conj, MatMut, MatRef, Parallelism,
};
use reborrow::*;
use triangular::BlockStructure;

fn reconstruct_impl<T: ComplexField>(
    mut dst: MatMut<'_, T>,
    lu_factors: Option<MatRef<'_, T>>,
    row_perm: PermutationIndicesRef<'_>,
    col_perm: PermutationIndicesRef<'_>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let lu_factors = match lu_factors {
        Some(lu_factors) => lu_factors,
        None => dst.rb(),
    };

    let m = lu_factors.nrows();
    let n = lu_factors.ncols();
    let size = m.min(n);

    temp_mat_uninit! {
        let (mut lu, _) = unsafe { temp_mat_uninit::<T>(m, n, stack) };
    }

    let (l_top, _, l_bot, _) = lu_factors.split_at(size, size);
    let (u_left, u_right, _, _) = lu_factors.split_at(size, size);

    let (lu_topleft, lu_topright, lu_botleft, _) = lu.rb_mut().split_at(size, size);

    triangular::matmul(
        lu_topleft,
        BlockStructure::Rectangular,
        Conj::No,
        l_top,
        BlockStructure::UnitTriangularLower,
        Conj::No,
        u_left,
        BlockStructure::TriangularUpper,
        Conj::No,
        None,
        T::one(),
        parallelism,
    );
    triangular::matmul(
        lu_topright,
        BlockStructure::Rectangular,
        Conj::No,
        l_top,
        BlockStructure::UnitTriangularLower,
        Conj::No,
        u_right,
        BlockStructure::Rectangular,
        Conj::No,
        None,
        T::one(),
        parallelism,
    );
    triangular::matmul(
        lu_botleft,
        BlockStructure::Rectangular,
        Conj::No,
        l_bot,
        BlockStructure::Rectangular,
        Conj::No,
        u_left,
        BlockStructure::TriangularUpper,
        Conj::No,
        None,
        T::one(),
        parallelism,
    );

    let row_inv = row_perm.into_arrays().1;
    let col_inv = col_perm.into_arrays().1;
    if dst.row_stride().abs() <= dst.col_stride().abs() {
        for j in 0..n {
            for i in 0..m {
                unsafe {
                    *dst.rb_mut().ptr_in_bounds_at_unchecked(i, j) = *lu
                        .rb()
                        .get_unchecked(*row_inv.get_unchecked(i), *col_inv.get_unchecked(j));
                }
            }
        }
    } else {
        for i in 0..m {
            for j in 0..n {
                unsafe {
                    *dst.rb_mut().ptr_in_bounds_at_unchecked(i, j) = *lu
                        .rb()
                        .get_unchecked(*row_inv.get_unchecked(i), *col_inv.get_unchecked(j));
                }
            }
        }
    }
}

pub fn reconstruct_to<T: ComplexField>(
    dst: MatMut<'_, T>,
    lu_factors: MatRef<'_, T>,
    row_perm: PermutationIndicesRef<'_>,
    col_perm: PermutationIndicesRef<'_>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    reconstruct_impl(
        dst,
        Some(lu_factors),
        row_perm,
        col_perm,
        parallelism,
        stack,
    )
}

pub fn reconstruct_in_place<T: ComplexField>(
    lu_factors: MatMut<'_, T>,
    row_perm: PermutationIndicesRef<'_>,
    col_perm: PermutationIndicesRef<'_>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    reconstruct_impl(lu_factors, None, row_perm, col_perm, parallelism, stack)
}

pub fn reconstruct_req<T: 'static>(
    nrows: usize,
    ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = parallelism;
    temp_mat_req::<T>(nrows, ncols)
}
