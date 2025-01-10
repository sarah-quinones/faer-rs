use crate::Idx;
use crate::internal_prelude::*;
use dyn_stack::StackReq;
use linalg::zip::{Last, Zip};
use reborrow::*;

/// swaps the values in the columns `a` and `b`
///
/// # panics
///
/// panics if `a` and `b` don't have the same number of columns
///
/// # example
///
/// ```
/// use faer::{mat, perm};
///
/// let mut m = mat![
/// 	[1.0, 2.0, 3.0], //
/// 	[4.0, 5.0, 6.0],
/// 	[7.0, 8.0, 9.0],
/// 	[10.0, 14.0, 12.0],
/// ];
///
/// let (a, b) = m.two_cols_mut(0, 2);
/// perm::swap_cols(a, b);
///
/// let swapped = mat![
/// 	[3.0, 2.0, 1.0], //
/// 	[6.0, 5.0, 4.0],
/// 	[9.0, 8.0, 7.0],
/// 	[12.0, 14.0, 10.0],
/// ];
///
/// assert_eq!(m, swapped);
/// ```
#[track_caller]
#[inline]
pub fn swap_cols<N: Shape, T>(a: ColMut<'_, T, N>, b: ColMut<'_, T, N>) {
	fn swap<T>() -> impl FnMut(Zip<&mut T, Last<&mut T>>) {
		|unzip!(a, b)| core::mem::swap(a, b)
	}

	zip!(a, b).for_each(swap::<T>());
}

/// swaps the values in the rows `a` and `b`
///
/// # panics
///
/// panics if `a` and `b` don't have the same number of columns
///
/// # example
///
/// ```
/// use faer::{mat, perm};
///
/// let mut m = mat![
/// 	[1.0, 2.0, 3.0], //
/// 	[4.0, 5.0, 6.0],
/// 	[7.0, 8.0, 9.0],
/// 	[10.0, 14.0, 12.0],
/// ];
///
/// let (a, b) = m.two_rows_mut(0, 2);
/// perm::swap_rows(a, b);
///
/// let swapped = mat![
/// 	[7.0, 8.0, 9.0], //
/// 	[4.0, 5.0, 6.0],
/// 	[1.0, 2.0, 3.0],
/// 	[10.0, 14.0, 12.0],
/// ];
///
/// assert_eq!(m, swapped);
/// ```
#[track_caller]
#[inline]
pub fn swap_rows<N: Shape, T>(a: RowMut<'_, T, N>, b: RowMut<'_, T, N>) {
	swap_cols(a.transpose_mut(), b.transpose_mut())
}

/// swaps the two rows at indices `a` and `b` in the given matrix
///
/// # panics
///
/// panics if either `a` or `b` is out of bounds
///
/// # example
///
/// ```
/// use faer::{mat, perm};
///
/// let mut m = mat![
/// 	[1.0, 2.0, 3.0], //
/// 	[4.0, 5.0, 6.0],
/// 	[7.0, 8.0, 9.0],
/// 	[10.0, 14.0, 12.0],
/// ];
///
/// perm::swap_rows_idx(m.as_mut(), 0, 2);
///
/// let swapped = mat![
/// 	[7.0, 8.0, 9.0], //
/// 	[4.0, 5.0, 6.0],
/// 	[1.0, 2.0, 3.0],
/// 	[10.0, 14.0, 12.0],
/// ];
///
/// assert_eq!(m, swapped);
/// ```
#[track_caller]
#[inline]
pub fn swap_rows_idx<M: Shape, N: Shape, T>(mat: MatMut<'_, T, M, N>, a: Idx<M>, b: Idx<M>) {
	if a != b {
		let (a, b) = mat.two_rows_mut(a, b);
		swap_rows(a, b);
	}
}

/// swaps the two columns at indices `a` and `b` in the given matrix
///
/// # panics
///
/// panics if either `a` or `b` is out of bounds
///
/// # example
///
/// ```
/// use faer::{mat, perm};
///
/// let mut m = mat![
/// 	[1.0, 2.0, 3.0], //
/// 	[4.0, 5.0, 6.0],
/// 	[7.0, 8.0, 9.0],
/// 	[10.0, 14.0, 12.0],
/// ];
///
/// perm::swap_cols_idx(m.as_mut(), 0, 2);
///
/// let swapped = mat![
/// 	[3.0, 2.0, 1.0], //
/// 	[6.0, 5.0, 4.0],
/// 	[9.0, 8.0, 7.0],
/// 	[12.0, 14.0, 10.0],
/// ];
///
/// assert_eq!(m, swapped);
/// ```
#[track_caller]
#[inline]
pub fn swap_cols_idx<M: Shape, N: Shape, T>(mat: MatMut<'_, T, M, N>, a: Idx<N>, b: Idx<N>) {
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

/// computes a permutation of the columns of the source matrix using the given permutation, and
/// stores the result in the destination matrix
///
/// # panics
///
/// - panics if the matrices do not have the same shape
/// - panics if the size of the permutation doesn't match the number of columns of the matrices
#[inline]
#[track_caller]
pub fn permute_cols<I: Index, T: ComplexField>(dst: MatMut<'_, T>, src: MatRef<'_, T>, perm_indices: PermRef<'_, I>) {
	Assert!(all(
		src.nrows() == dst.nrows(),
		src.ncols() == dst.ncols(),
		perm_indices.arrays().0.len() == src.ncols(),
	));

	permute_rows(dst.transpose_mut(), src.transpose(), perm_indices.canonicalized());
}

/// computes a permutation of the rows of the source matrix using the given permutation, and
/// stores the result in the destination matrix
///
/// # panics
///
/// - panics if the matrices do not have the same shape
/// - panics if the size of the permutation doesn't match the number of rows of the matrices
#[inline]
#[track_caller]
pub fn permute_rows<I: Index, T: ComplexField>(dst: MatMut<'_, T>, src: MatRef<'_, T>, perm_indices: PermRef<'_, I>) {
	#[track_caller]
	#[math]
	fn implementation<I: Index, T: ComplexField>(dst: MatMut<'_, T>, src: MatRef<'_, T>, perm_indices: PermRef<'_, I>) {
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

		if dst.rb().row_stride().unsigned_abs() < dst.rb().col_stride().unsigned_abs() {
			for j in n.indices() {
				for i in m.indices() {
					dst[(i, j)] = copy(src[(perm[i].zx(), j)]);
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

/// computes the size and alignment of required workspace for applying a row permutation to a
/// matrix in place
pub fn permute_rows_in_place_scratch<I: Index, T: ComplexField>(nrows: usize, ncols: usize) -> StackReq {
	temp_mat_scratch::<T>(nrows, ncols)
}

/// computes the size and alignment of required workspace for applying a column permutation to a
/// matrix in place
pub fn permute_cols_in_place_scratch<I: Index, T: ComplexField>(nrows: usize, ncols: usize) -> StackReq {
	temp_mat_scratch::<T>(nrows, ncols)
}

/// computes a permutation of the rows of the matrix using the given permutation, and
/// stores the result in the same matrix
///
/// # panics
///
/// - panics if the size of the permutation doesn't match the number of rows of the matrix
#[inline]
#[track_caller]
pub fn permute_rows_in_place<I: Index, T: ComplexField>(matrix: MatMut<'_, T>, perm_indices: PermRef<'_, I>, stack: &mut MemStack) {
	#[inline]
	#[track_caller]
	fn implementation<T: ComplexField, I: Index>(matrix: MatMut<'_, T>, perm_indices: PermRef<'_, I>, stack: &mut MemStack) {
		let mut matrix = matrix;
		let (mut tmp, _) = unsafe { temp_mat_uninit(matrix.nrows(), matrix.ncols(), stack) };
		let mut tmp = tmp.as_mat_mut();
		tmp.copy_from(matrix.rb());
		permute_rows(matrix.rb_mut(), tmp.rb(), perm_indices);
	}

	implementation(matrix, perm_indices.canonicalized(), stack)
}

/// computes a permutation of the columns of the matrix using the given permutation, and
/// stores the result in the same matrix.
///
/// # panics
///
/// - panics if the size of the permutation doesn't match the number of columns of the matrix
#[inline]
#[track_caller]
pub fn permute_cols_in_place<I: Index, T: ComplexField>(matrix: MatMut<'_, T>, perm_indices: PermRef<'_, I>, stack: &mut MemStack) {
	#[inline]
	#[track_caller]
	fn implementation<I: Index, T: ComplexField>(matrix: MatMut<'_, T>, perm_indices: PermRef<'_, I>, stack: &mut MemStack) {
		let mut matrix = matrix;
		let (mut tmp, _) = unsafe { temp_mat_uninit(matrix.nrows(), matrix.ncols(), stack) };
		let mut tmp = tmp.as_mat_mut();
		tmp.copy_from(matrix.rb());
		permute_cols(matrix.rb_mut(), tmp.rb(), perm_indices);
	}

	implementation(matrix, perm_indices.canonicalized(), stack)
}
