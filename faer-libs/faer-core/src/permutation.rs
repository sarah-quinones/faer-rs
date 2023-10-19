//! Permutation matrices.
#![allow(clippy::len_without_is_empty)]

use crate::{
    inner::{PermMut, PermOwn, PermRef},
    seal::Seal,
    temp_mat_req, temp_mat_uninit, zipped, ComplexField, Entity, MatMut, MatRef, Matrix,
};
#[cfg(feature = "std")]
use assert2::{assert, debug_assert};
use bytemuck::Pod;
use core::fmt::Debug;
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use reborrow::*;

impl Seal for i32 {}
impl Seal for i64 {}
impl Seal for u32 {}
impl Seal for u64 {}
impl Seal for usize {}

pub trait Index: Seal + core::fmt::Debug + Pod + Eq + Ord + Send + Sync {
    type Unsigned: Index;
    type Signed: SignedIndex;

    #[doc(hidden)]
    #[inline(always)]
    fn canonicalize(slice: &[Self]) -> &[Self::Unsigned] {
        bytemuck::cast_slice(slice)
    }

    #[doc(hidden)]
    #[inline(always)]
    fn canonicalize_mut(slice: &mut [Self]) -> &mut [Self::Unsigned] {
        bytemuck::cast_slice_mut(slice)
    }

    #[inline(always)]
    fn from_unsigned(value: Self::Unsigned) -> Self {
        pulp::cast(value)
    }
    #[inline(always)]
    fn from_signed(value: Self::Signed) -> Self {
        pulp::cast(value)
    }

    #[inline(always)]
    fn to_unsigned(self) -> Self::Unsigned {
        pulp::cast(self)
    }
    #[inline(always)]
    fn to_signed(self) -> Self::Signed {
        pulp::cast(self)
    }
}

pub trait SignedIndex:
    Seal
    + core::fmt::Debug
    + core::ops::Neg<Output = Self>
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::AddAssign
    + core::ops::SubAssign
    + Pod
    + Eq
    + Ord
    + Send
    + Sync
{
    const MAX: Self;

    #[must_use]
    fn truncate(value: usize) -> Self;

    /// zero extend
    #[must_use]
    fn zx(self) -> usize;
    /// sign extend
    #[must_use]
    fn sx(self) -> usize;
}

#[cfg(any(
    target_pointer_width = "32",
    target_pointer_width = "64",
    target_pointer_width = "128",
))]
impl Index for u32 {
    type Unsigned = u32;
    type Signed = i32;
}
#[cfg(any(target_pointer_width = "64", target_pointer_width = "128"))]
impl Index for u64 {
    type Unsigned = u64;
    type Signed = i64;
}
#[cfg(target_pointer_width = "128")]
impl Index for u128 {
    type Unsigned = u128;
    type Signed = i128;
}

impl Index for usize {
    #[cfg(target_pointer_width = "32")]
    type Unsigned = u32;
    #[cfg(target_pointer_width = "64")]
    type Unsigned = u64;
    #[cfg(target_pointer_width = "128")]
    type Unsigned = u128;

    #[cfg(target_pointer_width = "32")]
    type Signed = i32;
    #[cfg(target_pointer_width = "64")]
    type Signed = i64;
    #[cfg(target_pointer_width = "128")]
    type Signed = i128;
}

#[cfg(any(
    target_pointer_width = "32",
    target_pointer_width = "64",
    target_pointer_width = "128",
))]
impl SignedIndex for i32 {
    const MAX: Self = Self::MAX;

    #[inline(always)]
    fn truncate(value: usize) -> Self {
        #[allow(clippy::assertions_on_constants)]
        const _: () = {
            core::assert!(i32::BITS <= usize::BITS);
        };
        value as isize as Self
    }

    #[inline(always)]
    fn zx(self) -> usize {
        self as u32 as usize
    }

    #[inline(always)]
    fn sx(self) -> usize {
        self as isize as usize
    }
}

#[cfg(any(target_pointer_width = "64", target_pointer_width = "128"))]
impl SignedIndex for i64 {
    const MAX: Self = Self::MAX;

    #[inline(always)]
    fn truncate(value: usize) -> Self {
        #[allow(clippy::assertions_on_constants)]
        const _: () = {
            core::assert!(i64::BITS <= usize::BITS);
        };
        value as isize as Self
    }

    #[inline(always)]
    fn zx(self) -> usize {
        self as u64 as usize
    }

    #[inline(always)]
    fn sx(self) -> usize {
        self as isize as usize
    }
}

#[cfg(target_pointer_width = "128")]
impl SignedIndex for i128 {
    const MAX: Self = Self::MAX;

    #[inline(always)]
    fn truncate(value: usize) -> Self {
        #[allow(clippy::assertions_on_constants)]
        const _: () = {
            core::assert!(i128::BITS <= usize::BITS);
        };
        value as isize as Self
    }

    #[inline(always)]
    fn zx(self) -> usize {
        self as u128 as usize
    }

    #[inline(always)]
    fn sx(self) -> usize {
        self as isize as usize
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

pub type PermutationRef<'a, I> = Matrix<PermRef<'a, I>>;
pub type PermutationMut<'a, I> = Matrix<PermMut<'a, I>>;
pub type Permutation<I> = Matrix<PermOwn<I>>;

impl<I: Index> Permutation<I> {
    #[inline]
    pub fn as_ref(&self) -> PermutationRef<'_, I> {
        PermutationRef {
            inner: PermRef {
                forward: &self.inner.forward,
                inverse: &self.inner.inverse,
            },
        }
    }

    #[inline]
    pub fn as_mut(&mut self) -> PermutationMut<'_, I> {
        PermutationMut {
            inner: PermMut {
                forward: &mut self.inner.forward,
                inverse: &mut self.inner.inverse,
            },
        }
    }
}

impl<I: Index> Permutation<I> {
    /// Returns the permutation as an array.
    #[inline]
    pub fn into_arrays(self) -> (Box<[I]>, Box<[I]>) {
        (self.inner.forward, self.inner.inverse)
    }

    #[inline]
    pub fn len(&self) -> usize {
        debug_assert!(self.inner.inverse.len() == self.inner.forward.len());
        self.inner.forward.len()
    }

    /// Returns the inverse permutation.
    #[inline]
    pub fn inverse(self) -> Self {
        Self {
            inner: PermOwn {
                forward: self.inner.inverse,
                inverse: self.inner.forward,
            },
        }
    }

    /// Creates a new permutation reference, without checking the validity of the inputs.
    ///
    /// # Safety
    ///
    /// `forward` and `inverse` must have the same length, be valid permutations, and be inverse
    /// permutations of each other.
    #[inline]
    pub unsafe fn new_unchecked(forward: Box<[I]>, inverse: Box<[I]>) -> Self {
        Self {
            inner: PermOwn { forward, inverse },
        }
    }
}

impl<'a, I: Index> PermutationRef<'a, I> {
    /// Returns the permutation as an array.
    #[inline]
    pub fn into_arrays(self) -> (&'a [I], &'a [I]) {
        (self.inner.forward, self.inner.inverse)
    }

    #[inline]
    pub fn len(&self) -> usize {
        debug_assert!(self.inner.inverse.len() == self.inner.forward.len());
        self.inner.forward.len()
    }

    /// Returns the inverse permutation.
    #[inline]
    pub fn inverse(self) -> Self {
        Self {
            inner: PermRef {
                forward: self.inner.inverse,
                inverse: self.inner.forward,
            },
        }
    }

    /// Creates a new permutation reference, without checking the validity of the inputs.
    ///
    /// # Safety
    ///
    /// `forward` and `inverse` must have the same length, be valid permutations, and be inverse
    /// permutations of each other.
    #[inline]
    pub unsafe fn new_unchecked(forward: &'a [I], inverse: &'a [I]) -> Self {
        Self {
            inner: PermRef { forward, inverse },
        }
    }

    #[inline(always)]
    pub fn canonicalize(self) -> PermutationRef<'a, I::Unsigned> {
        PermutationRef {
            inner: PermRef {
                forward: I::canonicalize(self.inner.forward),
                inverse: I::canonicalize(self.inner.inverse),
            },
        }
    }

    #[inline(always)]
    pub fn uncanonicalize<J: Index>(self) -> PermutationRef<'a, J> {
        assert!(core::mem::size_of::<J>() == core::mem::size_of::<I>());
        PermutationRef {
            inner: PermRef {
                forward: bytemuck::cast_slice(self.inner.forward),
                inverse: bytemuck::cast_slice(self.inner.inverse),
            },
        }
    }
}

impl<'a, I: Index> PermutationMut<'a, I> {
    /// Returns the permutation as an array.
    ///
    /// # Safety
    ///
    /// The behavior is undefined if the arrays are no longer inverse permutations of each other by
    /// the end of lifetime `'a`.
    #[inline]
    pub unsafe fn into_arrays(self) -> (&'a mut [I], &'a mut [I]) {
        (self.inner.forward, self.inner.inverse)
    }

    #[inline]
    pub fn len(&self) -> usize {
        debug_assert!(self.inner.inverse.len() == self.inner.forward.len());
        self.inner.forward.len()
    }

    /// Returns the inverse permutation.
    #[inline]
    pub fn inverse(self) -> Self {
        Self {
            inner: PermMut {
                forward: self.inner.inverse,
                inverse: self.inner.forward,
            },
        }
    }

    /// Creates a new permutation mutable reference, without checking the validity of the inputs.
    ///
    /// # Safety
    ///
    /// `forward` and `inverse` must have the same length, be valid permutations, and be inverse
    /// permutations of each other.
    #[inline]
    pub unsafe fn new_unchecked(forward: &'a mut [I], inverse: &'a mut [I]) -> Self {
        Self {
            inner: PermMut { forward, inverse },
        }
    }

    #[inline(always)]
    pub fn canonicalize(self) -> PermutationMut<'a, I::Unsigned> {
        PermutationMut {
            inner: PermMut {
                forward: I::canonicalize_mut(self.inner.forward),
                inverse: I::canonicalize_mut(self.inner.inverse),
            },
        }
    }

    #[inline(always)]
    pub fn uncanonicalize<J: Index>(self) -> PermutationMut<'a, J> {
        assert!(core::mem::size_of::<J>() == core::mem::size_of::<I>());
        PermutationMut {
            inner: PermMut {
                forward: bytemuck::cast_slice_mut(self.inner.forward),
                inverse: bytemuck::cast_slice_mut(self.inner.inverse),
            },
        }
    }
}

impl<'short, 'a, I> Reborrow<'short> for PermutationRef<'a, I> {
    type Target = PermutationRef<'short, I>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}

impl<'short, 'a, I> ReborrowMut<'short> for PermutationRef<'a, I> {
    type Target = PermutationRef<'short, I>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}

impl<'short, 'a, I> Reborrow<'short> for PermutationMut<'a, I> {
    type Target = PermutationRef<'short, I>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        PermutationRef {
            inner: PermRef {
                forward: &*self.inner.forward,
                inverse: &*self.inner.inverse,
            },
        }
    }
}

impl<'short, 'a, I> ReborrowMut<'short> for PermutationMut<'a, I> {
    type Target = PermutationMut<'short, I>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        PermutationMut {
            inner: PermMut {
                forward: &mut *self.inner.forward,
                inverse: &mut *self.inner.inverse,
            },
        }
    }
}

impl<'a, I: Debug> Debug for PermutationRef<'a, I> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}
impl<'a, I: Debug> Debug for PermutationMut<'a, I> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
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
pub fn permute_cols<E: ComplexField, I: Index>(
    dst: MatMut<'_, E>,
    src: MatRef<'_, E>,
    perm_indices: PermutationRef<'_, I>,
) {
    assert!((src.nrows(), src.ncols()) == (dst.nrows(), dst.ncols()));
    assert!(perm_indices.into_arrays().0.len() == src.ncols());

    permute_rows(
        dst.transpose(),
        src.transpose(),
        perm_indices.canonicalize(),
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
pub fn permute_rows<E: ComplexField, I: Index>(
    dst: MatMut<'_, E>,
    src: MatRef<'_, E>,
    perm_indices: PermutationRef<'_, I>,
) {
    #[track_caller]
    fn implementation<E: ComplexField, I: Index>(
        dst: MatMut<'_, E>,
        src: MatRef<'_, E>,
        perm_indices: PermutationRef<'_, I>,
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
                            src.read_unchecked(perm.get_unchecked(i).to_signed().zx(), j),
                        );
                    }
                }
            }
        } else {
            for i in 0..m {
                unsafe {
                    let src_i = src.subrows(perm.get_unchecked(i).to_signed().zx(), 1);
                    let dst_i = dst.rb_mut().subrows(i, 1);

                    zipped!(dst_i, src_i).for_each(|mut dst, src| dst.write(src.read()));
                }
            }
        }
    }

    implementation(dst, src, perm_indices.canonicalize())
}

/// Computes the size and alignment of required workspace for applying a row permutation to a
/// matrix in place.
pub fn permute_rows_in_place_req<E: Entity, I: Index>(
    nrows: usize,
    ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    temp_mat_req::<E>(nrows, ncols)
}

/// Computes the size and alignment of required workspace for applying a column permutation to a
/// matrix in place.
pub fn permute_cols_in_place_req<E: Entity, I: Index>(
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
pub fn permute_rows_in_place<E: ComplexField, I: Index>(
    matrix: MatMut<'_, E>,
    perm_indices: PermutationRef<'_, I>,
    stack: PodStack<'_>,
) {
    #[inline]
    #[track_caller]
    fn implementation<E: ComplexField, I: Index>(
        matrix: MatMut<'_, E>,
        perm_indices: PermutationRef<'_, I>,
        stack: PodStack<'_>,
    ) {
        let mut matrix = matrix;
        let (mut tmp, _) = temp_mat_uninit::<E>(matrix.nrows(), matrix.ncols(), stack);
        tmp.rb_mut().clone_from(matrix.rb());
        permute_rows(matrix.rb_mut(), tmp.rb(), perm_indices);
    }

    implementation(matrix, perm_indices.canonicalize(), stack)
}

/// Computes a permutation of the columns of the matrix using the given permutation, and
/// stores the result in the same matrix.
///
/// # Panics
///
/// - Panics if the size of the permutation doesn't match the number of columns of the matrix.
#[inline]
#[track_caller]
pub fn permute_cols_in_place<E: ComplexField, I: Index>(
    matrix: MatMut<'_, E>,
    perm_indices: PermutationRef<'_, I>,
    stack: PodStack<'_>,
) {
    #[inline]
    #[track_caller]
    fn implementation<E: ComplexField, I: Index>(
        matrix: MatMut<'_, E>,
        perm_indices: PermutationRef<'_, I>,
        stack: PodStack<'_>,
    ) {
        let mut matrix = matrix;
        let (mut tmp, _) = temp_mat_uninit::<E>(matrix.nrows(), matrix.ncols(), stack);
        tmp.rb_mut().clone_from(matrix.rb());
        permute_cols(matrix.rb_mut(), tmp.rb(), perm_indices);
    }

    implementation(matrix, perm_indices.canonicalize(), stack)
}
