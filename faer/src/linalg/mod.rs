use crate::internal_prelude::*;
use core::marker::PhantomData;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_traits::ComplexField;

use crate::{
    mat::{matown::align_for, AsMatMut, MatMut},
    Shape,
};

pub fn temp_mat_scratch<T: ComplexField>(
    nrows: usize,
    ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    let align = align_for(
        size_of::<T>(),
        align_of::<T>(),
        core::mem::needs_drop::<T>(),
    );

    let mut col_stride = nrows;
    if align > size_of::<T>() {
        col_stride = col_stride.next_multiple_of(align / size_of::<T>());
    }
    let len = col_stride.checked_mul(ncols).unwrap();
    StackReq::try_new_aligned::<T>(len, align)
}

struct DynMat<'a, T: ComplexField, Rows: Shape, Cols: Shape> {
    ptr: *mut T,
    nrows: Rows,
    ncols: Cols,
    col_stride: usize,
    __marker: PhantomData<(&'a T, T)>,
}

impl<'a, T: ComplexField, Rows: Shape, Cols: Shape> Drop for DynMat<'a, T, Rows, Cols> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            core::ptr::drop_in_place(core::slice::from_raw_parts_mut(
                self.ptr,
                self.col_stride * self.ncols.unbound(),
            ))
        };
    }
}

impl<'a, T: ComplexField, Rows: Shape, Cols: Shape> AsMatRef for DynMat<'a, T, Rows, Cols> {
    type T = T;
    type Rows = Rows;
    type Cols = Cols;

    fn as_mat_ref(&self) -> crate::mat::MatRef<T, Rows, Cols> {
        unsafe {
            MatRef::from_raw_parts(
                self.ptr as *const T,
                self.nrows,
                self.ncols,
                1,
                self.col_stride as isize,
            )
        }
    }
}

impl<'a, T: ComplexField, Rows: Shape, Cols: Shape> AsMatMut for DynMat<'a, T, Rows, Cols> {
    fn as_mat_mut(&mut self) -> crate::mat::MatMut<T, Rows, Cols> {
        unsafe {
            MatMut::from_raw_parts_mut(
                self.ptr,
                self.nrows,
                self.ncols,
                1,
                self.col_stride as isize,
            )
        }
    }
}

struct DropGuard<T> {
    ptr: *mut T,
    len: usize,
}
impl<T> Drop for DropGuard<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe { core::ptr::drop_in_place(core::slice::from_raw_parts_mut(self.ptr, self.len)) };
    }
}

#[track_caller]
pub unsafe fn temp_mat_uninit<'a, T: ComplexField + 'a, Rows: Shape + 'a, Cols: Shape + 'a>(
    nrows: Rows,
    ncols: Cols,
    stack: &'a mut DynStack,
) -> (
    impl 'a + AsMatMut<T = T, Rows = Rows, Cols = Cols>,
    &'a mut DynStack,
) {
    let align = align_for(
        size_of::<T>(),
        align_of::<T>(),
        core::mem::needs_drop::<T>(),
    );

    let mut col_stride = nrows.unbound();
    if align > size_of::<T>() {
        col_stride = col_stride.next_multiple_of(align / size_of::<T>());
    }
    let len = col_stride.checked_mul(ncols.unbound()).unwrap();

    let (uninit, stack) = stack.make_aligned_uninit::<T>(len, align);

    let ptr = uninit.as_mut_ptr() as *mut T;
    if core::mem::needs_drop::<T>() {
        unsafe {
            let mut guard = DropGuard { ptr, len: 0 };
            for j in 0..len {
                let ptr = ptr.add(j);
                let val = T::nan_impl();
                ptr.write(val);
                guard.len += 1;
            }
            core::mem::forget(guard);
        }
    }
    (
        DynMat {
            ptr,
            nrows,
            ncols,
            col_stride,
            __marker: PhantomData,
        },
        stack,
    )
}

#[track_caller]
pub fn temp_mat_zeroed<'a, T: ComplexField + 'a, Rows: Shape + 'a, Cols: Shape + 'a>(
    nrows: Rows,
    ncols: Cols,
    stack: &'a mut DynStack,
) -> (
    impl 'a + AsMatMut<T = T, Rows = Rows, Cols = Cols>,
    &'a mut DynStack,
) {
    let align = align_for(
        size_of::<T>(),
        align_of::<T>(),
        core::mem::needs_drop::<T>(),
    );

    let mut col_stride = nrows.unbound();
    if align > size_of::<T>() {
        col_stride = col_stride.next_multiple_of(align / size_of::<T>());
    }
    let len = col_stride.checked_mul(ncols.unbound()).unwrap();
    _ = stack.make_aligned_uninit::<T>(len, align);

    let (uninit, stack) = stack.make_aligned_uninit::<T>(len, align);

    let ptr = uninit.as_mut_ptr() as *mut T;

    unsafe {
        let mut guard = DropGuard { ptr, len: 0 };
        for j in 0..len {
            let ptr = ptr.add(j);
            let val = T::zero_impl();
            ptr.write(val);
            guard.len += 1;
        }
        core::mem::forget(guard);
    }

    (
        DynMat {
            ptr,
            nrows,
            ncols,
            col_stride,
            __marker: PhantomData,
        },
        stack,
    )
}

pub mod matmul;
pub mod triangular_inverse;
pub mod triangular_solve;

pub mod reductions;
pub mod zip;

pub mod householder;
pub mod jacobi;

pub mod mat_ops;

pub mod kron;

pub mod cholesky;
pub mod lu;
pub mod qr;

// pub mod evd;
// pub mod svd;
