use crate::{internal_prelude::*, Idx};
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use linalg::zip::{Last, Zip};
use reborrow::*;

#[track_caller]
#[inline]
#[math]
pub fn swap_cols<N: Shape, C: Container, T>(a: ColMut<'_, C, T, N>, b: ColMut<'_, C, T, N>) {
    help!(C);

    fn swap<C: Container, T>() -> impl FnMut(Zip<C::Of<&mut T>, Last<C::Of<&mut T>>>) {
        |unzipped!(mut a, mut b)| faer_traits::utils::swap::<C, _>(&mut a, &mut b)
    }

    zipped!(a, b).for_each(swap::<C, T>());
}

#[track_caller]
#[inline]
pub fn swap_rows<N: Shape, C: Container, T>(a: RowMut<'_, C, T, N>, b: RowMut<'_, C, T, N>) {
    swap_cols(a.transpose_mut(), b.transpose_mut())
}

#[track_caller]
#[inline]
pub fn swap_rows_idx<M: Shape, N: Shape, C: Container, T>(
    mat: MatMut<'_, C, T, M, N>,
    a: Idx<M>,
    b: Idx<M>,
) {
    if a != b {
        let (a, b) = mat.two_rows_mut(a, b);
        swap_rows(a, b);
    }
}

#[track_caller]
#[inline]
pub fn swap_cols_idx<M: Shape, N: Shape, C: Container, T>(
    mat: MatMut<'_, C, T, M, N>,
    a: Idx<N>,
    b: Idx<N>,
) {
    if a != b {
        let (a, b) = mat.two_cols_mut(a, b);
        swap_cols(a, b);
    }
}

mod permown;
mod permref;

pub use permown::Perm;
pub use permref::PermRef;

use self::linalg::temp_mat_scratch;

/// Computes a permutation of the columns of the source matrix using the given permutation, and
/// stores the result in the destination matrix.
///
/// # Panics
///
/// - Panics if the matrices do not have the same shape.
/// - Panics if the size of the permutation doesn't match the number of columns of the matrices.
#[inline]
#[track_caller]
pub fn permute_cols<I: Index, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    dst: MatMut<'_, C, T>,
    src: MatRef<'_, C, T>,
    perm_indices: PermRef<'_, I>,
) {
    Assert!(all(
        src.nrows() == dst.nrows(),
        src.ncols() == dst.ncols(),
        perm_indices.arrays().0.len() == src.ncols(),
    ));

    permute_rows(
        ctx,
        dst.transpose_mut(),
        src.transpose(),
        perm_indices.canonicalized(),
    );
}

/// Computes a permutation of the rows of the source matrix using the given permutation, and
/// stores the result in the destination matrix.
///
/// # Panics
///
/// - Panics if the matrices do not have the same shape.
/// - Panics if the size of the permutation doesn't match the number of rows of the matrices.
#[inline]
#[track_caller]
pub fn permute_rows<I: Index, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    dst: MatMut<'_, C, T>,
    src: MatRef<'_, C, T>,
    perm_indices: PermRef<'_, I>,
) {
    #[track_caller]
    #[math]
    fn implementation<I: Index, C: ComplexContainer, T: ComplexField<C>>(
        ctx: &Ctx<C, T>,
        dst: MatMut<'_, C, T>,
        src: MatRef<'_, C, T>,
        perm_indices: PermRef<'_, I>,
    ) {
        Assert!(all(
            src.nrows() == dst.nrows(),
            src.ncols() == dst.ncols(),
            perm_indices.len() == src.nrows(),
        ));

        with_dim!(m, src.nrows());
        with_dim!(n, src.ncols());
        let mut dst = dst.as_shape_mut(m, n);
        let src = src.as_shape(m, n);
        let perm = perm_indices.as_shape(m).bound_arrays().0;

        help!(C);

        if dst.rb().row_stride().unsigned_abs() < dst.rb().col_stride().unsigned_abs() {
            for j in n.indices() {
                for i in m.indices() {
                    math(write1!(dst.write(i, j), copy(src[(perm[i].zx(), j)])));
                }
            }
        } else {
            for i in m.indices() {
                let src_i = src.row(perm[i].zx());
                let mut dst_i = dst.rb_mut().row_mut(i);

                dst_i.copy_from_with(ctx, src_i);
            }
        }
    }

    implementation(ctx, dst, src, perm_indices.canonicalized())
}

/// Computes the size and alignment of required workspace for applying a row permutation to a
/// matrix in place.
pub fn permute_rows_in_place_scratch<I: Index, C: ComplexContainer, T: ComplexField<C>>(
    nrows: usize,
    ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    temp_mat_scratch::<C, T>(nrows, ncols)
}

/// Computes the size and alignment of required workspace for applying a column permutation to a
/// matrix in place.
pub fn permute_cols_in_place_scratch<I: Index, C: ComplexContainer, T: ComplexField<C>>(
    nrows: usize,
    ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    temp_mat_scratch::<C, T>(nrows, ncols)
}

/// Computes a permutation of the rows of the matrix using the given permutation, and
/// stores the result in the same matrix.
///
/// # Panics
///
/// - Panics if the size of the permutation doesn't match the number of rows of the matrix.
#[inline]
#[track_caller]
pub fn permute_rows_in_place<I: Index, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    matrix: MatMut<'_, C, T>,
    perm_indices: PermRef<'_, I>,
    stack: &mut DynStack,
) {
    #[inline]
    #[track_caller]
    fn implementation<C: ComplexContainer, T: ComplexField<C>, I: Index>(
        ctx: &Ctx<C, T>,
        matrix: MatMut<'_, C, T>,
        perm_indices: PermRef<'_, I>,
        stack: &mut DynStack,
    ) {
        let mut matrix = matrix;
        let (mut tmp, _) = unsafe { temp_mat_uninit(ctx, matrix.nrows(), matrix.ncols(), stack) };
        let mut tmp = tmp.as_mat_mut();
        tmp.copy_from_with(ctx, matrix.rb());
        permute_rows(ctx, matrix.rb_mut(), tmp.rb(), perm_indices);
    }

    implementation(ctx, matrix, perm_indices.canonicalized(), stack)
}

/// Computes a permutation of the columns of the matrix using the given permutation, and
/// stores the result in the same matrix.
///
/// # Panics
///
/// - Panics if the size of the permutation doesn't match the number of columns of the matrix.
#[inline]
#[track_caller]
pub fn permute_cols_in_place<I: Index, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    matrix: MatMut<'_, C, T>,
    perm_indices: PermRef<'_, I>,
    stack: &mut DynStack,
) {
    #[inline]
    #[track_caller]
    fn implementation<I: Index, C: ComplexContainer, T: ComplexField<C>>(
        ctx: &Ctx<C, T>,
        matrix: MatMut<'_, C, T>,
        perm_indices: PermRef<'_, I>,
        stack: &mut DynStack,
    ) {
        let mut matrix = matrix;
        let (mut tmp, _) = unsafe { temp_mat_uninit(ctx, matrix.nrows(), matrix.ncols(), stack) };
        let mut tmp = tmp.as_mat_mut();
        tmp.copy_from_with(ctx, matrix.rb());
        permute_cols(ctx, matrix.rb_mut(), tmp.rb(), perm_indices);
    }

    implementation(ctx, matrix, perm_indices.canonicalized(), stack)
}
