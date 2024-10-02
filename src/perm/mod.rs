use crate::{assert, col::*, linalg::temp_mat_uninit, mat::*, row::*, *};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use reborrow::*;

/// Swaps the values in the columns `a` and `b`.
///
/// # Panics
///
/// Panics if `a` and `b` don't have the same number of columns.
///
/// # Example
///
/// ```
/// use faer::{mat, perm::swap_cols};
///
/// let mut m = mat![
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [7.0, 8.0, 9.0],
///     [10.0, 14.0, 12.0],
/// ];
///
/// let (a, b) = m.as_mut().two_cols_mut(0, 2);
/// swap_cols(a, b);
///
/// let swapped = mat![
///     [3.0, 2.0, 1.0],
///     [6.0, 5.0, 4.0],
///     [9.0, 8.0, 7.0],
///     [12.0, 14.0, 10.0],
/// ];
///
/// assert_eq!(m, swapped);
/// ```
#[track_caller]
#[inline]
pub fn swap_cols<E: ComplexField>(a: ColMut<'_, E>, b: ColMut<'_, E>) {
    zipped!(__rw, a, b).for_each(|unzipped!(mut a, mut b)| {
        let (a_read, b_read) = (a.read(), b.read());
        a.write(b_read);
        b.write(a_read);
    });
}

/// Swaps the values in the rows `a` and `b`.
///
/// # Panics
///
/// Panics if `a` and `b` don't have the same number of columns.
///
/// # Example
///
/// ```
/// use faer::{mat, perm::swap_rows};
///
/// let mut m = mat![
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [7.0, 8.0, 9.0],
///     [10.0, 14.0, 12.0],
/// ];
///
/// let (a, b) = m.as_mut().two_rows_mut(0, 2);
/// swap_rows(a, b);
///
/// let swapped = mat![
///     [7.0, 8.0, 9.0],
///     [4.0, 5.0, 6.0],
///     [1.0, 2.0, 3.0],
///     [10.0, 14.0, 12.0],
/// ];
///
/// assert_eq!(m, swapped);
/// ```
#[track_caller]
#[inline]
pub fn swap_rows<E: ComplexField>(a: RowMut<'_, E>, b: RowMut<'_, E>) {
    swap_cols(a.transpose_mut(), b.transpose_mut())
}

/// Swaps the two rows at indices `a` and `b` in the given matrix.
///
/// # Panics
///
/// Panics if either `a` or `b` is out of bounds.
///
/// # Example
///
/// ```
/// use faer::{mat, perm::swap_rows_idx};
///
/// let mut m = mat![
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [7.0, 8.0, 9.0],
///     [10.0, 14.0, 12.0],
/// ];
///
/// swap_rows_idx(m.as_mut(), 0, 2);
///
/// let swapped = mat![
///     [7.0, 8.0, 9.0],
///     [4.0, 5.0, 6.0],
///     [1.0, 2.0, 3.0],
///     [10.0, 14.0, 12.0],
/// ];
///
/// assert_eq!(m, swapped);
/// ```
#[track_caller]
#[inline]
pub fn swap_rows_idx<E: ComplexField>(mat: MatMut<'_, E>, a: usize, b: usize) {
    if a != b {
        let (a, b) = mat.two_rows_mut(a, b);
        swap_rows(a, b);
    }
}

/// Swaps the two columns at indices `a` and `b` in the given matrix.
///
/// # Panics
///
/// Panics if either `a` or `b` is out of bounds.
///
/// # Example
///
/// ```
/// use faer::{mat, perm::swap_cols_idx};
///
/// let mut m = mat![
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [7.0, 8.0, 9.0],
///     [10.0, 14.0, 12.0],
/// ];
///
/// swap_cols_idx(m.as_mut(), 0, 2);
///
/// let swapped = mat![
///     [3.0, 2.0, 1.0],
///     [6.0, 5.0, 4.0],
///     [9.0, 8.0, 7.0],
///     [12.0, 14.0, 10.0],
/// ];
///
/// assert_eq!(m, swapped);
/// ```
#[track_caller]
#[inline]
pub fn swap_cols_idx<E: ComplexField>(mat: MatMut<'_, E>, a: usize, b: usize) {
    if a != b {
        let (a, b) = mat.two_cols_mut(a, b);
        swap_cols(a, b);
    }
}

mod permown;
mod permref;

pub use permown::Perm;
pub use permref::PermRef;

use self::linalg::temp_mat_req;

/// Computes a permutation of the columns of the source matrix using the given permutation, and
/// stores the result in the destination matrix.
///
/// # Panics
///
/// - Panics if the matrices do not have the same shape.
/// - Panics if the size of the permutation doesn't match the number of columns of the matrices.
#[inline]
#[track_caller]
pub fn permute_cols<I: Index, E: ComplexField>(
    dst: MatMut<'_, E>,
    src: MatRef<'_, E>,
    perm_indices: PermRef<'_, I>,
) {
    assert!(all(
        src.nrows() == dst.nrows(),
        src.ncols() == dst.ncols(),
        perm_indices.arrays().0.len() == src.ncols(),
    ));

    permute_rows(
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
pub fn permute_rows<I: Index, E: ComplexField>(
    dst: MatMut<'_, E>,
    src: MatRef<'_, E>,
    perm_indices: PermRef<'_, I>,
) {
    #[track_caller]
    fn implementation<I: Index, E: ComplexField>(
        dst: MatMut<'_, E>,
        src: MatRef<'_, E>,
        perm_indices: PermRef<'_, I>,
    ) {
        assert!(all(
            src.nrows() == dst.nrows(),
            src.ncols() == dst.ncols(),
            perm_indices.len() == src.nrows(),
        ));

        with_dim!(m, src.nrows());
        with_dim!(n, src.ncols());
        let mut dst = dst.as_shape_mut(m, n);
        let src = src.as_shape(m, n);
        let perm = perm_indices.as_shape(m).bound_arrays().0;

        if dst.rb().row_stride().unsigned_abs() < dst.rb().col_stride().unsigned_abs() {
            for j in n.indices() {
                for i in m.indices() {
                    dst.rb_mut().write(i, j, src.read(perm[i].zx(), j));
                }
            }
        } else {
            for i in m.indices() {
                let src_i = src.row(perm[i].zx());
                let mut dst_i = dst.rb_mut().row_mut(i);

                dst_i.copy_from(src_i);
            }
        }
    }

    implementation(dst, src, perm_indices.canonicalized())
}

/// Computes the size and alignment of required workspace for applying a row permutation to a
/// matrix in place.
pub fn permute_rows_in_place_req<I: Index, E: Entity>(
    nrows: usize,
    ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    temp_mat_req::<E>(nrows, ncols)
}

/// Computes the size and alignment of required workspace for applying a column permutation to a
/// matrix in place.
pub fn permute_cols_in_place_req<I: Index, E: Entity>(
    nrows: usize,
    ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    temp_mat_req::<E>(nrows, ncols)
}

/// Computes a permutation of the rows of the matrix using the given permutation, and
/// stores the result in the same matrix.
///
/// # Panics
///
/// - Panics if the size of the permutation doesn't match the number of rows of the matrix.
#[inline]
#[track_caller]
pub fn permute_rows_in_place<I: Index, E: ComplexField>(
    matrix: MatMut<'_, E>,
    perm_indices: PermRef<'_, I>,
    stack: &mut PodStack,
) {
    #[inline]
    #[track_caller]
    fn implementation<E: ComplexField, I: Index>(
        matrix: MatMut<'_, E>,
        perm_indices: PermRef<'_, I>,
        stack: &mut PodStack,
    ) {
        let mut matrix = matrix;
        let (mut tmp, _) = temp_mat_uninit::<E>(matrix.nrows(), matrix.ncols(), stack);
        tmp.rb_mut().copy_from(matrix.rb());
        permute_rows(matrix.rb_mut(), tmp.rb(), perm_indices);
    }

    implementation(matrix, perm_indices.canonicalized(), stack)
}

/// Computes a permutation of the columns of the matrix using the given permutation, and
/// stores the result in the same matrix.
///
/// # Panics
///
/// - Panics if the size of the permutation doesn't match the number of columns of the matrix.
#[inline]
#[track_caller]
pub fn permute_cols_in_place<I: Index, E: ComplexField>(
    matrix: MatMut<'_, E>,
    perm_indices: PermRef<'_, I>,
    stack: &mut PodStack,
) {
    #[inline]
    #[track_caller]
    fn implementation<I: Index, E: ComplexField>(
        matrix: MatMut<'_, E>,
        perm_indices: PermRef<'_, I>,
        stack: &mut PodStack,
    ) {
        let mut matrix = matrix;
        let (mut tmp, _) = temp_mat_uninit::<E>(matrix.nrows(), matrix.ncols(), stack);
        tmp.rb_mut().copy_from(matrix.rb());
        permute_cols(matrix.rb_mut(), tmp.rb(), perm_indices);
    }

    implementation(matrix, perm_indices.canonicalized(), stack)
}
