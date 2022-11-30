//! Permutation matrices.

use assert2::{assert as fancy_assert, debug_assert as fancy_debug_assert};
use reborrow::*;

use crate::{MatMut, MatRef};

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
pub fn swap_cols<T>(mat: MatMut<'_, T>, a: usize, b: usize) {
    let m = mat.nrows();
    let n = mat.ncols();
    fancy_assert!(a < n);
    fancy_assert!(b < n);

    if a == b {
        return;
    }

    let rs = mat.row_stride();
    let cs = mat.col_stride();
    let ptr = mat.as_ptr();

    let ptr_a = ptr.wrapping_offset(cs * a as isize);
    let ptr_b = ptr.wrapping_offset(cs * b as isize);

    if rs == 1 {
        unsafe {
            core::ptr::swap_nonoverlapping(ptr_a, ptr_b, m);
        }
    } else {
        for i in 0..m {
            let offset = rs * i as isize;
            unsafe {
                core::ptr::swap_nonoverlapping(
                    ptr_a.wrapping_offset(offset),
                    ptr_b.wrapping_offset(offset),
                    1,
                );
            }
        }
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
pub fn swap_rows<T>(mat: MatMut<'_, T>, a: usize, b: usize) {
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
        fancy_debug_assert!(self.inverse.len() == self.forward.len());
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
    #[inline]
    pub unsafe fn into_arrays(self) -> (&'a mut [usize], &'a mut [usize]) {
        (self.forward, self.inverse)
    }

    #[inline]
    pub fn len(&self) -> usize {
        fancy_debug_assert!(self.inverse.len() == self.forward.len());
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

/// Computes a symmetric permutation of the rows and columns of the source matrix using the given
/// permutation, and stores the result in the destination matrix.
///
/// Both the source and the destination are interpreted as symmetric matrices, and only their lower
/// triangular part is accessed.
///
/// # Panics
///
/// - Panics if the matrices are not square or if they do not have the same shape.
/// - Panics if the size of the permutation doesn't match the dimension of the matrices.
#[track_caller]
pub fn permute_rows_and_cols_symmetric_lower<T: Copy>(
    dst: MatMut<'_, T>,
    src: MatRef<'_, T>,
    perm_indices: PermutationRef<'_>,
) {
    let mut dst = dst;
    let n = src.nrows();
    fancy_assert!(src.nrows() == src.ncols());
    fancy_assert!((src.nrows(), src.ncols()) == (dst.nrows(), dst.ncols()));
    fancy_assert!(perm_indices.into_arrays().0.len() == n);

    let perm = perm_indices.into_arrays().0;
    let src_tril = |i, j| unsafe {
        if i > j {
            src.get_unchecked(i, j)
        } else {
            src.get_unchecked(j, i)
        }
    };
    for j in 0..n {
        for i in j..n {
            unsafe {
                *dst.rb_mut().ptr_in_bounds_at_unchecked(i, j) =
                    *src_tril(*perm.get_unchecked(i), *perm.get_unchecked(j));
            }
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
pub fn permute_cols<T: Copy>(
    dst: MatMut<'_, T>,
    src: MatRef<'_, T>,
    perm_indices: PermutationRef<'_>,
) {
    fancy_assert!((src.nrows(), src.ncols()) == (dst.nrows(), dst.ncols()));
    fancy_assert!(perm_indices.into_arrays().0.len() == src.ncols());

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
pub fn permute_rows<T: Copy>(
    dst: MatMut<'_, T>,
    src: MatRef<'_, T>,
    perm_indices: PermutationRef<'_>,
) {
    fancy_assert!((src.nrows(), src.ncols()) == (dst.nrows(), dst.ncols()));
    fancy_assert!(perm_indices.into_arrays().0.len() == src.nrows());

    let src = src;
    let perm_indices = perm_indices;
    let mut dst = dst;
    let m = src.nrows();
    let n = src.ncols();

    let perm = perm_indices.into_arrays().0;

    if dst.row_stride().abs() < dst.col_stride().abs() {
        for j in 0..n {
            for i in 0..m {
                unsafe {
                    *dst.rb_mut().ptr_in_bounds_at_unchecked(i, j) =
                        *src.get_unchecked(*perm.get_unchecked(i), j);
                }
            }
        }
    } else {
        for i in 0..m {
            unsafe {
                let src_i = src.row_unchecked(*perm.get_unchecked(i));
                let dst_i = dst.rb_mut().row_unchecked(i);

                dst_i.cwise().zip_unchecked(src_i).for_each(|dst, src| {
                    *dst = *src;
                });
            }
        }
    }
}
