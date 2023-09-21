//! Matrix zipping module.

use crate::{seal::Seal, Entity, MatMut, MatRef};
use assert2::{assert, debug_assert};
use core::mem::MaybeUninit;
use reborrow::*;

/// Specifies whether the main diagonal should be traversed, when iterating over a triangular chunk
/// of the matrix.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Diag {
    /// Do not include diagonal of matrix
    Skip,
    /// Include diagonal of matrix
    Include,
}

/// Read only view over a single matrix element.
pub struct Read<'a, E: Entity> {
    ptr: E::Group<&'a MaybeUninit<E::Unit>>,
}
/// Read-write view over a single matrix element.
pub struct ReadWrite<'a, E: Entity> {
    ptr: E::Group<&'a mut MaybeUninit<E::Unit>>,
}

impl<E: Entity> Read<'_, E> {
    #[inline(always)]
    pub fn read(&self) -> E {
        E::from_units(E::map(
            E::as_ref(&self.ptr),
            #[inline(always)]
            |ptr| unsafe { ptr.assume_init_read() },
        ))
    }
}

impl<E: Entity> ReadWrite<'_, E> {
    #[inline(always)]
    pub fn read(&self) -> E {
        E::from_units(E::map(
            E::as_ref(&self.ptr),
            #[inline(always)]
            |ptr| unsafe { ptr.assume_init_ref().clone() },
        ))
    }

    #[inline(always)]
    pub fn write(&mut self, value: E) {
        let value = E::into_units(value);
        E::map(
            E::zip(E::as_mut(&mut self.ptr), value),
            #[inline(always)]
            |(ptr, value)| unsafe { *ptr.assume_init_mut() = value },
        );
    }
}

/// Internal trait for abstracting over [`MatRef`] and [`MatMut`].
pub trait Mat<'short, Outlives = &'short Self>: Seal {
    type Item;
    type RawSlice;

    fn transpose(self) -> Self;
    fn reverse_rows(self) -> Self;
    fn reverse_cols(self) -> Self;
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn row_stride(&self) -> isize;
    fn col_stride(&self) -> isize;
    /// # Safety
    /// `i` and `j` must be within bounds.
    unsafe fn get(&'short mut self, i: usize, j: usize) -> Self::Item;
    /// # Safety
    /// The row stride must be `1`.
    /// `[i, i + n_elems)` and `j` must be within bounds.
    unsafe fn get_column_slice(
        &'short mut self,
        i: usize,
        j: usize,
        n_elems: usize,
    ) -> Self::RawSlice;

    #[doc(hidden)]
    // this is a bad api since it needs to extend the lifetime of slice, but this is somewhat fine
    // since we only use it internally in this module
    unsafe fn get_slice_elem(slice: &mut Self::RawSlice, idx: usize) -> Self::Item;
}

impl<'a, E: Entity> Seal for MatRef<'a, E> {}
impl<'a, 'short, E: Entity> Mat<'short> for MatRef<'a, E> {
    type Item = Read<'short, E>;
    type RawSlice = E::Group<&'a [MaybeUninit<E::Unit>]>;

    #[inline(always)]
    fn transpose(self) -> Self {
        self.transpose()
    }
    #[inline(always)]
    fn reverse_rows(self) -> Self {
        self.reverse_rows()
    }
    #[inline(always)]
    fn reverse_cols(self) -> Self {
        self.reverse_cols()
    }

    #[inline(always)]
    fn nrows(&self) -> usize {
        (*self).nrows()
    }

    #[inline(always)]
    fn ncols(&self) -> usize {
        (*self).ncols()
    }

    #[inline(always)]
    fn row_stride(&self) -> isize {
        (*self).row_stride()
    }

    #[inline(always)]
    fn col_stride(&self) -> isize {
        (*self).col_stride()
    }

    #[inline(always)]
    unsafe fn get(&mut self, i: usize, j: usize) -> Self::Item {
        Read {
            ptr: E::map(
                self.ptr_inbounds_at(i, j),
                #[inline(always)]
                |ptr| &*(ptr as *const MaybeUninit<E::Unit>),
            ),
        }
    }

    #[inline(always)]
    unsafe fn get_column_slice(&mut self, i: usize, j: usize, n_elems: usize) -> Self::RawSlice {
        E::map(
            self.ptr_at(i, j),
            #[inline(always)]
            |ptr| core::slice::from_raw_parts(ptr as *const MaybeUninit<E::Unit>, n_elems),
        )
    }

    #[inline(always)]
    unsafe fn get_slice_elem(slice: &mut Self::RawSlice, idx: usize) -> Self::Item {
        Read {
            ptr: E::map(
                E::as_mut(slice),
                #[inline(always)]
                |slice| slice.get_unchecked(idx),
            ),
        }
    }
}

impl<'a, E: Entity> Seal for MatMut<'a, E> {}
impl<'a, 'short, E: Entity> Mat<'short> for MatMut<'a, E> {
    type Item = ReadWrite<'short, E>;
    type RawSlice = E::Group<&'a mut [MaybeUninit<E::Unit>]>;

    #[inline(always)]
    fn transpose(self) -> Self {
        self.transpose()
    }
    #[inline(always)]
    fn reverse_rows(self) -> Self {
        self.reverse_rows()
    }
    #[inline(always)]
    fn reverse_cols(self) -> Self {
        self.reverse_cols()
    }

    #[inline(always)]
    fn nrows(&self) -> usize {
        (*self).nrows()
    }

    #[inline(always)]
    fn ncols(&self) -> usize {
        (*self).ncols()
    }

    #[inline(always)]
    fn row_stride(&self) -> isize {
        (*self).row_stride()
    }

    #[inline(always)]
    fn col_stride(&self) -> isize {
        (*self).col_stride()
    }

    #[inline(always)]
    unsafe fn get(&mut self, i: usize, j: usize) -> Self::Item {
        ReadWrite {
            ptr: E::map(
                self.rb_mut().ptr_inbounds_at(i, j),
                #[inline(always)]
                |ptr| &mut *(ptr as *mut MaybeUninit<E::Unit>),
            ),
        }
    }

    #[inline(always)]
    unsafe fn get_column_slice(&mut self, i: usize, j: usize, n_elems: usize) -> Self::RawSlice {
        E::map(
            self.rb_mut().ptr_at(i, j),
            #[inline(always)]
            |ptr| core::slice::from_raw_parts_mut(ptr as *mut MaybeUninit<E::Unit>, n_elems),
        )
    }

    #[inline(always)]
    unsafe fn get_slice_elem(slice: &mut Self::RawSlice, idx: usize) -> Self::Item {
        ReadWrite {
            ptr: E::map(
                E::as_mut(slice),
                #[inline(always)]
                |slice| &mut *(slice.get_unchecked_mut(idx) as *mut _),
            ),
        }
    }
}

/// Structure holding matrix views with matching dimensions.
pub struct Zip<Tuple> {
    pub(crate) tuple: Tuple,
}

include!(concat!(env!("OUT_DIR"), "/zip.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{zipped, ComplexField, Mat};
    use assert2::assert;

    #[test]
    fn test_zip() {
        for (m, n) in [(2, 2), (4, 2), (2, 4)] {
            for rev_dst in [false, true] {
                for rev_src in [false, true] {
                    for transpose_dst in [false, true] {
                        for transpose_src in [false, true] {
                            for diag in [Diag::Include, Diag::Skip] {
                                let mut dst = Mat::from_fn(
                                    if transpose_dst { n } else { m },
                                    if transpose_dst { m } else { n },
                                    |_, _| f64::zero(),
                                );
                                let src = Mat::from_fn(
                                    if transpose_src { n } else { m },
                                    if transpose_src { m } else { n },
                                    |_, _| f64::one(),
                                );

                                let mut target = Mat::from_fn(m, n, |_, _| f64::zero());
                                let target_src = Mat::from_fn(m, n, |_, _| f64::one());

                                zipped!(target.as_mut(), target_src.as_ref())
                                    .for_each_triangular_lower(diag, |mut dst, src| {
                                        dst.write(src.read())
                                    });

                                let mut dst = dst.as_mut();
                                let mut src = src.as_ref();

                                if transpose_dst {
                                    dst = dst.transpose();
                                }
                                if rev_dst {
                                    dst = dst.reverse_rows();
                                }

                                if transpose_src {
                                    src = src.transpose();
                                }
                                if rev_src {
                                    src = src.reverse_rows();
                                }

                                zipped!(dst.rb_mut(), src)
                                    .for_each_triangular_lower(diag, |mut dst, src| {
                                        dst.write(src.read())
                                    });

                                assert!(dst.rb() == target.as_ref());
                            }
                        }
                    }
                }
            }
        }
    }
}
