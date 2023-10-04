//! Permutation matrices.
#![allow(clippy::len_without_is_empty)]

use crate::{temp_mat_req, temp_mat_uninit, zipped, ComplexField, Entity, MatMut, MatRef};
use assert2::{assert, debug_assert};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use reborrow::*;

/// Swaps the two columns at indices `a` and `b` in the given matrix.
///
/// # Panics
///
/// Panics if either `a` or `b` is out of bounds.
///
/// # Example
///
/// ```
/// use faer_core::{mat, permutation::swap_cols};
///
/// let mut m = mat![
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [7.0, 8.0, 9.0],
///     [10.0, 14.0, 12.0],
/// ];
///
/// swap_cols(m.as_mut(), 0, 2);
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
pub fn swap_cols<E: ComplexField>(mat: MatMut<'_, E>, a: usize, b: usize) {
    assert!(a < mat.ncols());
    assert!(b < mat.ncols());

    if a == b {
        return;
    }

    let mat = mat.into_const();
    let mat_a = mat.subcols(a, 1);
    let mat_b = mat.subcols(b, 1);

    unsafe {
        zipped!(mat_a.const_cast(), mat_b.const_cast()).for_each(|mut a, mut b| {
            let (a_read, b_read) = (a.read(), b.read());
            a.write(b_read);
            b.write(a_read);
        });
    }
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
/// use faer_core::{mat, permutation::swap_rows};
///
/// let mut m = mat![
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [7.0, 8.0, 9.0],
///     [10.0, 14.0, 12.0],
/// ];
///
/// swap_rows(m.as_mut(), 0, 2);
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
pub fn swap_rows<E: ComplexField>(mat: MatMut<'_, E>, a: usize, b: usize) {
    swap_cols(mat.transpose(), a, b)
}

#[derive(Clone, Copy, Debug)]
pub struct PermutationRef<'a> {
    forward: &'a [usize],
    inverse: &'a [usize],
}

impl<'a> PermutationRef<'a> {
    /// Returns the permutation as an array.
    #[inline]
    pub fn into_arrays(self) -> (&'a [usize], &'a [usize]) {
        (self.forward, self.inverse)
    }

    #[inline]
    pub fn len(&self) -> usize {
        debug_assert!(self.inverse.len() == self.forward.len());
        self.forward.len()
    }

    /// Returns the inverse permutation.
    #[inline]
    pub fn inverse(self) -> Self {
        Self {
            forward: self.inverse,
            inverse: self.forward,
        }
    }

    /// Creates a new permutation reference, without checking the validity of the inputs.
    ///
    /// # Safety
    ///
    /// `forward` and `inverse` must have the same length, be valid permutations, and be inverse
    /// permutations of each other.
    #[inline]
    pub unsafe fn new_unchecked(forward: &'a [usize], inverse: &'a [usize]) -> Self {
        Self { forward, inverse }
    }
}

impl<'a> PermutationMut<'a> {
    /// Returns the permutation as an array.
    ///
    /// # Safety
    ///
    /// The behavior is undefined if the arrays are no longer inverse permutations of each other by
    /// the end of lifetime `'a`.
    #[inline]
    pub unsafe fn into_arrays(self) -> (&'a mut [usize], &'a mut [usize]) {
        (self.forward, self.inverse)
    }

    #[inline]
    pub fn len(&self) -> usize {
        debug_assert!(self.inverse.len() == self.forward.len());
        self.forward.len()
    }

    /// Returns the inverse permutation.
    #[inline]
    pub fn inverse(self) -> Self {
        Self {
            forward: self.inverse,
            inverse: self.forward,
        }
    }

    /// Creates a new permutation mutable reference, without checking the validity of the inputs.
    ///
    /// # Safety
    ///
    /// `forward` and `inverse` must have the same length, be valid permutations, and be inverse
    /// permutations of each other.
    #[inline]
    pub unsafe fn new_unchecked(forward: &'a mut [usize], inverse: &'a mut [usize]) -> Self {
        Self { forward, inverse }
    }
}

#[derive(Debug)]
pub struct PermutationMut<'a> {
    forward: &'a mut [usize],
    inverse: &'a mut [usize],
}

impl<'short, 'a> Reborrow<'short> for PermutationRef<'a> {
    type Target = PermutationRef<'short>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}

impl<'short, 'a> ReborrowMut<'short> for PermutationRef<'a> {
    type Target = PermutationRef<'short>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}

impl<'short, 'a> Reborrow<'short> for PermutationMut<'a> {
    type Target = PermutationRef<'short>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        PermutationRef {
            forward: &*self.forward,
            inverse: &*self.inverse,
        }
    }
}

impl<'short, 'a> ReborrowMut<'short> for PermutationMut<'a> {
    type Target = PermutationMut<'short>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        PermutationMut {
            forward: &mut *self.forward,
            inverse: &mut *self.inverse,
        }
    }
}

/// Computes a permutation of the columns of the source matrix using the given permutation, and
/// stores the result in the destination matrix.
///
/// # Panics
///
/// - Panics if the matrices do not have the same shape.
/// - Panics if the size of the permutation doesn't match the number of columns of the matrices.
#[inline]
#[track_caller]
pub fn permute_cols<E: ComplexField>(
    dst: MatMut<'_, E>,
    src: MatRef<'_, E>,
    perm_indices: PermutationRef<'_>,
) {
    assert!((src.nrows(), src.ncols()) == (dst.nrows(), dst.ncols()));
    assert!(perm_indices.into_arrays().0.len() == src.ncols());

    permute_rows(dst.transpose(), src.transpose(), perm_indices);
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
pub fn permute_rows<E: ComplexField>(
    dst: MatMut<'_, E>,
    src: MatRef<'_, E>,
    perm_indices: PermutationRef<'_>,
) {
    assert!((src.nrows(), src.ncols()) == (dst.nrows(), dst.ncols()));
    assert!(perm_indices.into_arrays().0.len() == src.nrows());

    let mut dst = dst;
    let m = src.nrows();
    let n = src.ncols();

    let perm = perm_indices.into_arrays().0;

    if dst.row_stride().abs() < dst.col_stride().abs() {
        for j in 0..n {
            for i in 0..m {
                unsafe {
                    dst.rb_mut().write_unchecked(
                        i,
                        j,
                        src.read_unchecked(*perm.get_unchecked(i), j),
                    );
                }
            }
        }
    } else {
        for i in 0..m {
            unsafe {
                let src_i = src.subrows(*perm.get_unchecked(i), 1);
                let dst_i = dst.rb_mut().subrows(i, 1);

                zipped!(dst_i, src_i).for_each(|mut dst, src| dst.write(src.read()));
            }
        }
    }
}

/// Computes the size and alignment of required workspace for applying a row permutation to a
/// matrix in place.
pub fn permute_rows_in_place_req<E: Entity>(
    nrows: usize,
    ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    temp_mat_req::<E>(nrows, ncols)
}

/// Computes the size and alignment of required workspace for applying a column permutation to a
/// matrix in place.
pub fn permute_cols_in_place_req<E: Entity>(
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
pub fn permute_rows_in_place<E: ComplexField>(
    matrix: MatMut<'_, E>,
    perm_indices: PermutationRef<'_>,
    stack: PodStack<'_>,
) {
    let mut matrix = matrix;
    let (mut tmp, _) = temp_mat_uninit::<E>(matrix.nrows(), matrix.ncols(), stack);
    let mut tmp = tmp.as_mut();
    zipped!(tmp.rb_mut(), matrix.rb()).for_each(|mut dst, src| dst.write(src.read()));
    permute_rows(matrix.rb_mut(), tmp.rb(), perm_indices);
}

/// Computes a permutation of the columns of the matrix using the given permutation, and
/// stores the result in the same matrix.
///
/// # Panics
///
/// - Panics if the size of the permutation doesn't match the number of columns of the matrix.
#[inline]
#[track_caller]
pub fn permute_cols_in_place<E: ComplexField>(
    matrix: MatMut<'_, E>,
    perm_indices: PermutationRef<'_>,
    stack: PodStack<'_>,
) {
    let mut matrix = matrix;
    let (mut tmp, _) = temp_mat_uninit::<E>(matrix.nrows(), matrix.ncols(), stack);
    let mut tmp = tmp.as_mut();
    zipped!(tmp.rb_mut(), matrix.rb()).for_each(|mut dst, src| dst.write(src.read()));
    permute_cols(matrix.rb_mut(), tmp.rb(), perm_indices);
}
