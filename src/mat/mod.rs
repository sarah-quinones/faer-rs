use crate::{assert, col::*, row::*, utils::slice::*, Conj, Shape};
use coe::Coerce;
use core::{marker::PhantomData, ptr::NonNull};
use faer_entity::*;
use reborrow::*;

#[repr(C)]
struct MatImpl<E: Entity, R: Shape = usize, C: Shape = usize> {
    ptr: GroupCopyFor<E, NonNull<E::Unit>>,
    nrows: R,
    ncols: C,
    row_stride: isize,
    col_stride: isize,
}
#[repr(C)]
struct MatOwnImpl<E: Entity> {
    ptr: GroupCopyFor<E, NonNull<E::Unit>>,
    nrows: usize,
    ncols: usize,
}

unsafe impl<E: Entity, R: Shape, C: Shape> Sync for MatImpl<E, R, C> {}
unsafe impl<E: Entity, R: Shape, C: Shape> Send for MatImpl<E, R, C> {}
unsafe impl<E: Entity> Sync for MatOwnImpl<E> {}
unsafe impl<E: Entity> Send for MatOwnImpl<E> {}

impl<E: Entity, R: Shape, C: Shape> Copy for MatImpl<E, R, C> {}
impl<E: Entity, R: Shape, C: Shape> Clone for MatImpl<E, R, C> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

/// Represents a type that can be used to slice a matrix, such as an index or a range of indices.
pub trait MatIndex<RowRange, ColRange>: crate::seal::Seal + Sized {
    /// Resulting type of the indexing operation.
    type Target;

    /// Index the matrix at `(row, col)`, without bound checks.
    #[allow(clippy::missing_safety_doc)]
    unsafe fn get_unchecked(this: Self, row: RowRange, col: ColRange) -> Self::Target {
        <Self as MatIndex<RowRange, ColRange>>::get(this, row, col)
    }
    /// Index the matrix at `(row, col)`.
    fn get(this: Self, row: RowRange, col: ColRange) -> Self::Target;
}

/// Trait for types that can be converted to a matrix view.
///
/// This trait is implemented for types of the matrix family, like [`Mat`],
/// [`MatRef`], and [`MatMut`], but not for types like [`Col`], [`Row`], or
/// their families. For a more general trait, see [`As2D`].
pub trait AsMatRef<E: Entity> {
    /// Row dimension of the matrix.
    type R: Shape;
    /// Column dimension of the matrix.
    type C: Shape;

    /// Convert to a matrix view.
    fn as_mat_ref(&self) -> MatRef<'_, E, Self::R, Self::C>;
}
/// Trait for types that can be converted to a mutable matrix view.
///
/// This trait is implemented for types of the matrix family, like [`Mat`],
/// [`MatRef`], and [`MatMut`], but not for types like [`Col`], [`Row`], or
/// their families. For a more general trait, see [`As2D`].
pub trait AsMatMut<E: Entity>: AsMatRef<E> {
    /// Convert to a mutable matrix view.
    fn as_mat_mut(&mut self) -> MatMut<'_, E, Self::R, Self::C>;
}

/// Trait for types that can be converted to a 2D matrix view.
///
/// This trait is implemented for any type that can be represented as a
/// 2D matrix view, like [`Mat`], [`Row`], [`Col`], and their respective
/// references and mutable references. For a trait specific to the matrix
/// family, see [`AsMatRef`] or [`AsMatMut`].
pub trait As2D<E: Entity> {
    /// Convert to a 2D matrix view.
    fn as_2d_ref(&self) -> MatRef<'_, E>;
}

impl<E: Entity, T: As2D<E>> As2D<E> for &T {
    fn as_2d_ref(&self) -> MatRef<'_, E> {
        (**self).as_2d_ref()
    }
}
impl<E: Entity, T: As2D<E>> As2D<E> for &mut T {
    fn as_2d_ref(&self) -> MatRef<'_, E> {
        (**self).as_2d_ref()
    }
}

/// Trait for types that can be converted to a mutable 2D matrix view.
///
/// This trait is implemented for any type that can be represented as a
/// 2D matrix view, like [`Mat`], [`Row`], [`Col`], and their respective
/// references and mutable references. For a trait specific to the matrix
/// family, see [`AsMatRef`] or [`AsMatMut`].
pub trait As2DMut<E: Entity>: As2D<E> {
    /// Convert to a mutable 2D matrix view.
    fn as_2d_mut(&mut self) -> MatMut<'_, E>;
}

impl<E: Entity, T: As2DMut<E>> As2DMut<E> for &mut T {
    fn as_2d_mut(&mut self) -> MatMut<'_, E> {
        (**self).as_2d_mut()
    }
}

impl<E: Entity, T: AsMatRef<E>> AsMatRef<E> for &T {
    type R = T::R;
    type C = T::C;

    fn as_mat_ref(&self) -> MatRef<'_, E, Self::R, Self::C> {
        (**self).as_mat_ref()
    }
}
impl<E: Entity, T: AsMatRef<E>> AsMatRef<E> for &mut T {
    type R = T::R;
    type C = T::C;

    fn as_mat_ref(&self) -> MatRef<'_, E, Self::R, Self::C> {
        (**self).as_mat_ref()
    }
}

impl<E: Entity, T: AsMatMut<E>> AsMatMut<E> for &mut T {
    fn as_mat_mut(&mut self) -> MatMut<'_, E, Self::R, Self::C> {
        (**self).as_mat_mut()
    }
}

impl<'a, FromE: Entity, ToE: Entity> Coerce<MatRef<'a, ToE>> for MatRef<'a, FromE> {
    #[inline(always)]
    fn coerce(self) -> MatRef<'a, ToE> {
        assert!(coe::is_same::<FromE, ToE>());
        unsafe { transmute_unchecked::<MatRef<'a, FromE>, MatRef<'a, ToE>>(self) }
    }
}
impl<'a, FromE: Entity, ToE: Entity> Coerce<MatMut<'a, ToE>> for MatMut<'a, FromE> {
    #[inline(always)]
    fn coerce(self) -> MatMut<'a, ToE> {
        assert!(coe::is_same::<FromE, ToE>());
        unsafe { transmute_unchecked::<MatMut<'a, FromE>, MatMut<'a, ToE>>(self) }
    }
}

mod mat_index;

mod matref;
pub use matref::{
    from_column_major_slice, from_column_major_slice_generic, from_column_major_slice_with_stride,
    from_column_major_slice_with_stride_generic, from_raw_parts, from_ref, from_ref_generic,
    from_repeated_col, from_repeated_ref, from_repeated_ref_generic, from_repeated_row,
    from_row_major_slice, from_row_major_slice_generic, from_row_major_slice_with_stride,
    from_row_major_slice_with_stride_generic, MatRef,
};

mod matmut;
pub use matmut::{
    from_column_major_slice_mut, from_column_major_slice_mut_generic,
    from_column_major_slice_with_stride_mut, from_column_major_slice_with_stride_mut_generic,
    from_mut, from_mut_generic, from_raw_parts_mut, from_row_major_slice_mut,
    from_row_major_slice_mut_generic, from_row_major_slice_with_stride_mut,
    from_row_major_slice_with_stride_mut_generic, MatMut,
};

mod matown;
pub use matown::Mat;

pub(crate) mod matalloc;

#[track_caller]
#[inline]
fn from_slice_assert(nrows: usize, ncols: usize, len: usize) {
    // we don't have to worry about size == usize::MAX == slice.len(), because the length of a
    // slice can never exceed isize::MAX in bytes, unless the type is zero sized, in which case
    // we don't care
    let size = usize::checked_mul(nrows, ncols).unwrap_or(usize::MAX);
    assert!(size == len);
}

#[track_caller]
#[inline]
fn from_strided_column_major_slice_assert(
    nrows: usize,
    ncols: usize,
    col_stride: usize,
    len: usize,
) {
    if nrows > 0 && ncols > 0 {
        // we don't have to worry about size == usize::MAX == slice.len(), because the length of a
        // slice can never exceed isize::MAX in bytes, unless the type is zero sized, in which case
        // we don't care
        let last = usize::checked_mul(col_stride, ncols - 1)
            .and_then(|last_col| last_col.checked_add(nrows - 1))
            .unwrap_or(usize::MAX);
        assert!(last < len);
    }
}

#[track_caller]
#[inline]
fn from_strided_column_major_slice_mut_assert(
    nrows: usize,
    ncols: usize,
    col_stride: usize,
    len: usize,
) {
    if nrows > 0 && ncols > 0 {
        // we don't have to worry about size == usize::MAX == slice.len(), because the length of a
        // slice can never exceed isize::MAX in bytes, unless the type is zero sized, in which case
        // we don't care
        let last = usize::checked_mul(col_stride, ncols - 1)
            .and_then(|last_col| last_col.checked_add(nrows - 1))
            .unwrap_or(usize::MAX);
        assert!(all(col_stride >= nrows, last < len));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use equator::assert;

    #[test]
    fn test_from_ref() {
        let x = crate::mat![[1.0, 2.0], [3.0, 4.0]];
        let c = 100.0;

        let sum = crate::mat![[101.0, 102.0], [103.0, 104.0]];
        crate::dbgf!("6.2?", &sum);

        // this converts a &f64 to a MatRef<'_, f64> without allocating
        assert!(&x + from_repeated_ref::<f64>(&c, x.nrows(), x.ncols()) == sum);
    }

    #[test]
    fn test_from_mut() {
        let mut c = 100.0;
        from_mut::<f64>(&mut c).fill(3.0);
        assert!(c == 3.0);
    }

    #[test]
    fn test_alloc() {
        let mut a = crate::Mat::<f64>::zeros(2, 2);

        a.reserve_exact(128, 0);
        a.reserve_exact(129, 1);
    }
}
