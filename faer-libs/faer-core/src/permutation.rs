//! Permutation matrices.
#![allow(clippy::len_without_is_empty)]

use crate::{
    assert, constrained, debug_assert,
    inner::{PermMut, PermOwn, PermRef},
    seal::Seal,
    temp_mat_req, temp_mat_uninit, unzipped, zipped, ComplexField, Entity, MatMut, MatRef, Matrix,
};
use bytemuck::Pod;
use core::fmt::Debug;
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use reborrow::*;

impl Seal for i32 {}
impl Seal for i64 {}
impl Seal for i128 {}
impl Seal for isize {}
impl Seal for u32 {}
impl Seal for u64 {}
impl Seal for u128 {}
impl Seal for usize {}

/// Trait for unsigned integers that can be indexed with.
///
/// Always smaller than or equal to `usize`.
pub trait Index:
    Seal
    + core::fmt::Debug
    + core::ops::Not<Output = Self>
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
    type FixedWidth: Index;
    type Signed: SignedIndex;

    #[must_use]
    #[inline(always)]
    fn truncate(value: usize) -> Self {
        Self::from_signed(<Self::Signed as SignedIndex>::truncate(value))
    }

    /// zero extend
    #[must_use]
    #[inline(always)]
    fn zx(self) -> usize {
        self.to_signed().zx()
    }

    #[inline(always)]
    fn canonicalize(slice: &[Self]) -> &[Self::FixedWidth] {
        bytemuck::cast_slice(slice)
    }

    #[inline(always)]
    fn canonicalize_mut(slice: &mut [Self]) -> &mut [Self::FixedWidth] {
        bytemuck::cast_slice_mut(slice)
    }

    #[inline(always)]
    fn from_signed(value: Self::Signed) -> Self {
        pulp::cast(value)
    }

    #[inline(always)]
    fn to_signed(self) -> Self::Signed {
        pulp::cast(self)
    }

    #[inline]
    fn sum_nonnegative(slice: &[Self]) -> Option<Self> {
        Self::Signed::sum_nonnegative(bytemuck::cast_slice(slice)).map(Self::from_signed)
    }
}

/// Trait for signed integers corresponding to the ones satisfying [`Index`].
///
/// Always smaller than or equal to `isize`.
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

    fn sum_nonnegative(slice: &[Self]) -> Option<Self> {
        let mut acc = Self::zeroed();
        for &i in slice {
            if Self::MAX - i < acc {
                return None;
            }
            acc += i;
        }
        Some(acc)
    }
}

#[cfg(any(
    target_pointer_width = "32",
    target_pointer_width = "64",
    target_pointer_width = "128",
))]
impl Index for u32 {
    type FixedWidth = u32;
    type Signed = i32;
}
#[cfg(any(target_pointer_width = "64", target_pointer_width = "128"))]
impl Index for u64 {
    type FixedWidth = u64;
    type Signed = i64;
}
#[cfg(target_pointer_width = "128")]
impl Index for u128 {
    type FixedWidth = u128;
    type Signed = i128;
}

impl Index for usize {
    #[cfg(target_pointer_width = "32")]
    type FixedWidth = u32;
    #[cfg(target_pointer_width = "64")]
    type FixedWidth = u64;
    #[cfg(target_pointer_width = "128")]
    type FixedWidth = u128;

    type Signed = isize;
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

impl SignedIndex for isize {
    const MAX: Self = Self::MAX;

    #[inline(always)]
    fn truncate(value: usize) -> Self {
        value as isize
    }

    #[inline(always)]
    fn zx(self) -> usize {
        self as usize
    }

    #[inline(always)]
    fn sx(self) -> usize {
        self as usize
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
    let mat_a = mat.col(a);
    let mat_b = mat.col(b);

    unsafe {
        zipped!(
            mat_a.const_cast().as_2d_mut(),
            mat_b.const_cast().as_2d_mut(),
        )
    }
    .for_each(|unzipped!(mut a, mut b)| {
        let (a_read, b_read) = (a.read(), b.read());
        a.write(b_read);
        b.write(a_read);
    });
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
    swap_cols(mat.transpose_mut(), a, b)
}

pub type PermutationRef<'a, I, E> = Matrix<PermRef<'a, I, E>>;
pub type PermutationMut<'a, I, E> = Matrix<PermMut<'a, I, E>>;
pub type Permutation<I, E> = Matrix<PermOwn<I, E>>;

impl<I, E: Entity> Permutation<I, E> {
    #[inline]
    pub fn as_ref(&self) -> PermutationRef<'_, I, E> {
        PermutationRef {
            inner: PermRef {
                forward: &self.inner.forward,
                inverse: &self.inner.inverse,
                __marker: core::marker::PhantomData,
            },
        }
    }

    #[inline]
    pub fn as_mut(&mut self) -> PermutationMut<'_, I, E> {
        PermutationMut {
            inner: PermMut {
                forward: &mut self.inner.forward,
                inverse: &mut self.inner.inverse,
                __marker: core::marker::PhantomData,
            },
        }
    }
}

impl<I: Index, E: Entity> Permutation<I, E> {
    /// Creates a new permutation, by checking the validity of the inputs.
    ///
    /// # Panics
    ///
    /// The function panics if any of the following conditions are violated:
    /// `forward` and `inverse` must have the same length which must be less than or equal to
    /// `I::Signed::MAX`, be valid permutations, and be inverse permutations of each other.
    #[inline]
    #[track_caller]
    pub fn new_checked(forward: alloc::boxed::Box<[I]>, inverse: alloc::boxed::Box<[I]>) -> Self {
        PermutationRef::<'_, I, E>::new_checked(&forward, &inverse);
        Self {
            inner: PermOwn {
                forward,
                inverse,
                __marker: core::marker::PhantomData,
            },
        }
    }

    /// Creates a new permutation reference, without checking the validity of the inputs.
    ///
    /// # Safety
    ///
    /// `forward` and `inverse` must have the same length which must be less than or equal to
    /// `I::Signed::MAX`, be valid permutations, and be inverse permutations of each other.
    #[inline]
    #[track_caller]
    pub unsafe fn new_unchecked(
        forward: alloc::boxed::Box<[I]>,
        inverse: alloc::boxed::Box<[I]>,
    ) -> Self {
        let n = forward.len();
        assert!(forward.len() == inverse.len());
        assert!(n <= I::Signed::MAX.zx());
        Self {
            inner: PermOwn {
                forward,
                inverse,
                __marker: core::marker::PhantomData,
            },
        }
    }

    /// Returns the permutation as an array.
    #[inline]
    pub fn into_arrays(self) -> (alloc::boxed::Box<[I]>, alloc::boxed::Box<[I]>) {
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
                __marker: core::marker::PhantomData,
            },
        }
    }

    #[inline]
    pub fn cast<T: Entity>(self) -> Permutation<I, T> {
        Permutation {
            inner: PermOwn {
                forward: self.inner.forward,
                inverse: self.inner.inverse,
                __marker: core::marker::PhantomData,
            },
        }
    }
}

impl<'a, I: Index, E: Entity> PermutationRef<'a, I, E> {
    /// Creates a new permutation reference, by checking the validity of the inputs.
    ///
    /// # Panics
    ///
    /// The function panics if any of the following conditions are violated:
    /// `forward` and `inverse` must have the same length which must be less than or equal to
    /// `I::Signed::MAX`, be valid permutations, and be inverse permutations of each other.
    #[inline]
    #[track_caller]
    pub fn new_checked(forward: &'a [I], inverse: &'a [I]) -> Self {
        #[track_caller]
        fn check<I: Index>(forward: &[I], inverse: &[I]) {
            let n = forward.len();
            assert!(forward.len() == inverse.len());
            assert!(n <= I::Signed::MAX.zx());
            for (i, &p) in forward.iter().enumerate() {
                let p = p.to_signed().zx();
                assert!(p < n);
                assert!(inverse[p].to_signed().zx() == i);
            }
        }

        check(I::canonicalize(forward), I::canonicalize(inverse));
        Self {
            inner: PermRef {
                forward,
                inverse,
                __marker: core::marker::PhantomData,
            },
        }
    }

    /// Creates a new permutation reference, without checking the validity of the inputs.
    ///
    /// # Safety
    ///
    /// `forward` and `inverse` must have the same length which must be less than or equal to
    /// `I::Signed::MAX`, be valid permutations, and be inverse permutations of each other.
    #[inline]
    #[track_caller]
    pub unsafe fn new_unchecked(forward: &'a [I], inverse: &'a [I]) -> Self {
        let n = forward.len();
        assert!(forward.len() == inverse.len());
        assert!(n <= I::Signed::MAX.zx());

        Self {
            inner: PermRef {
                forward,
                inverse,
                __marker: core::marker::PhantomData,
            },
        }
    }

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
                __marker: core::marker::PhantomData,
            },
        }
    }

    #[inline]
    pub fn cast<T: Entity>(self) -> PermutationRef<'a, I, T> {
        PermutationRef {
            inner: PermRef {
                forward: self.inner.forward,
                inverse: self.inner.inverse,
                __marker: core::marker::PhantomData,
            },
        }
    }

    #[inline(always)]
    pub fn canonicalize(self) -> PermutationRef<'a, I::FixedWidth, E> {
        PermutationRef {
            inner: PermRef {
                forward: I::canonicalize(self.inner.forward),
                inverse: I::canonicalize(self.inner.inverse),
                __marker: core::marker::PhantomData,
            },
        }
    }

    #[inline(always)]
    pub fn uncanonicalize<J: Index>(self) -> PermutationRef<'a, J, E> {
        assert!(core::mem::size_of::<J>() == core::mem::size_of::<I>());
        PermutationRef {
            inner: PermRef {
                forward: bytemuck::cast_slice(self.inner.forward),
                inverse: bytemuck::cast_slice(self.inner.inverse),
                __marker: core::marker::PhantomData,
            },
        }
    }
}

impl<'a, I: Index, E: Entity> PermutationMut<'a, I, E> {
    /// Creates a new permutation mutable reference, by checking the validity of the inputs.
    ///
    /// # Panics
    ///
    /// The function panics if any of the following conditions are violated:
    /// `forward` and `inverse` must have the same length which must be less than or equal to
    /// `I::Signed::MAX`, be valid permutations, and be inverse permutations of each other.
    #[inline]
    #[track_caller]
    pub fn new_checked(forward: &'a mut [I], inverse: &'a mut [I]) -> Self {
        PermutationRef::<'_, I, E>::new_checked(forward, inverse);
        Self {
            inner: PermMut {
                forward,
                inverse,
                __marker: core::marker::PhantomData,
            },
        }
    }

    /// Creates a new permutation mutable reference, without checking the validity of the inputs.
    ///
    /// # Safety
    ///
    /// `forward` and `inverse` must have the same length which must be less than or equal to
    /// `I::Signed::MAX`, be valid permutations, and be inverse permutations of each other.
    #[inline]
    #[track_caller]
    pub unsafe fn new_unchecked(forward: &'a mut [I], inverse: &'a mut [I]) -> Self {
        let n = forward.len();
        assert!(forward.len() == inverse.len());
        assert!(n <= I::Signed::MAX.zx());

        Self {
            inner: PermMut {
                forward,
                inverse,
                __marker: core::marker::PhantomData,
            },
        }
    }

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
                __marker: core::marker::PhantomData,
            },
        }
    }

    #[inline]
    pub fn cast<T: Entity>(self) -> PermutationMut<'a, I, T> {
        PermutationMut {
            inner: PermMut {
                forward: self.inner.forward,
                inverse: self.inner.inverse,
                __marker: core::marker::PhantomData,
            },
        }
    }

    #[inline(always)]
    pub fn canonicalize(self) -> PermutationMut<'a, I::FixedWidth, E> {
        PermutationMut {
            inner: PermMut {
                forward: I::canonicalize_mut(self.inner.forward),
                inverse: I::canonicalize_mut(self.inner.inverse),
                __marker: core::marker::PhantomData,
            },
        }
    }

    #[inline(always)]
    pub fn uncanonicalize<J: Index>(self) -> PermutationMut<'a, J, E> {
        assert!(core::mem::size_of::<J>() == core::mem::size_of::<I>());
        PermutationMut {
            inner: PermMut {
                forward: bytemuck::cast_slice_mut(self.inner.forward),
                inverse: bytemuck::cast_slice_mut(self.inner.inverse),
                __marker: core::marker::PhantomData,
            },
        }
    }
}

impl<'short, 'a, I, E: Entity> Reborrow<'short> for PermutationRef<'a, I, E> {
    type Target = PermutationRef<'short, I, E>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}

impl<'short, 'a, I, E: Entity> ReborrowMut<'short> for PermutationRef<'a, I, E> {
    type Target = PermutationRef<'short, I, E>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}

impl<'short, 'a, I, E: Entity> Reborrow<'short> for PermutationMut<'a, I, E> {
    type Target = PermutationRef<'short, I, E>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        PermutationRef {
            inner: PermRef {
                forward: &*self.inner.forward,
                inverse: &*self.inner.inverse,
                __marker: core::marker::PhantomData,
            },
        }
    }
}

impl<'short, 'a, I, E: Entity> ReborrowMut<'short> for PermutationMut<'a, I, E> {
    type Target = PermutationMut<'short, I, E>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        PermutationMut {
            inner: PermMut {
                forward: &mut *self.inner.forward,
                inverse: &mut *self.inner.inverse,
                __marker: core::marker::PhantomData,
            },
        }
    }
}

impl<'a, I: Debug, E: Entity> Debug for PermutationRef<'a, I, E> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}
impl<'a, I: Debug, E: Entity> Debug for PermutationMut<'a, I, E> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}
impl<'a, I: Debug, E: Entity> Debug for Permutation<I, E> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
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
pub fn permute_cols<I: Index, E: ComplexField>(
    dst: MatMut<'_, E>,
    src: MatRef<'_, E>,
    perm_indices: PermutationRef<'_, I, E>,
) {
    assert!((src.nrows(), src.ncols()) == (dst.nrows(), dst.ncols()));
    assert!(perm_indices.into_arrays().0.len() == src.ncols());

    permute_rows(
        dst.transpose_mut(),
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
pub fn permute_rows<I: Index, E: ComplexField>(
    dst: MatMut<'_, E>,
    src: MatRef<'_, E>,
    perm_indices: PermutationRef<'_, I, E>,
) {
    #[track_caller]
    fn implementation<I: Index, E: ComplexField>(
        dst: MatMut<'_, E>,
        src: MatRef<'_, E>,
        perm_indices: PermutationRef<'_, I, E>,
    ) {
        assert!((src.nrows(), src.ncols()) == (dst.nrows(), dst.ncols()));
        assert!(perm_indices.into_arrays().0.len() == src.nrows());

        constrained::Size::with2(src.nrows(), src.ncols(), |m, n| {
            let mut dst = constrained::MatMut::new(dst, m, n);
            let src = constrained::MatRef::new(src, m, n);
            let perm = constrained::permutation::PermutationRef::new(perm_indices, m)
                .into_arrays()
                .0;

            if dst.rb().into_inner().row_stride().unsigned_abs()
                < dst.rb().into_inner().col_stride().unsigned_abs()
            {
                for j in n.indices() {
                    for i in m.indices() {
                        dst.rb_mut().write(i, j, src.read(perm[i].zx(), j));
                    }
                }
            } else {
                for i in m.indices() {
                    let src_i = src.into_inner().row(perm[i].zx().into_inner());
                    let mut dst_i = dst.rb_mut().into_inner().row_mut(i.into_inner());

                    dst_i.copy_from(src_i);
                }
            }
        });
    }

    implementation(dst, src, perm_indices.canonicalize())
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
    perm_indices: PermutationRef<'_, I, E>,
    stack: PodStack<'_>,
) {
    #[inline]
    #[track_caller]
    fn implementation<E: ComplexField, I: Index>(
        matrix: MatMut<'_, E>,
        perm_indices: PermutationRef<'_, I, E>,
        stack: PodStack<'_>,
    ) {
        let mut matrix = matrix;
        let (mut tmp, _) = temp_mat_uninit::<E>(matrix.nrows(), matrix.ncols(), stack);
        tmp.rb_mut().copy_from(matrix.rb());
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
pub fn permute_cols_in_place<I: Index, E: ComplexField>(
    matrix: MatMut<'_, E>,
    perm_indices: PermutationRef<'_, I, E>,
    stack: PodStack<'_>,
) {
    #[inline]
    #[track_caller]
    fn implementation<I: Index, E: ComplexField>(
        matrix: MatMut<'_, E>,
        perm_indices: PermutationRef<'_, I, E>,
        stack: PodStack<'_>,
    ) {
        let mut matrix = matrix;
        let (mut tmp, _) = temp_mat_uninit::<E>(matrix.nrows(), matrix.ncols(), stack);
        tmp.rb_mut().copy_from(matrix.rb());
        permute_cols(matrix.rb_mut(), tmp.rb(), perm_indices);
    }

    implementation(matrix, perm_indices.canonicalize(), stack)
}
