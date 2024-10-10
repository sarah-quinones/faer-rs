use crate::internal_prelude::*;
use core::marker::PhantomData;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_traits::{help, ComplexContainer, ComplexField};

use crate::{
    mat::{matown::align_for, AsMatMut, MatMutGeneric},
    Shape,
};

pub fn temp_mat_scratch<C: ComplexContainer, T: ComplexField<C>>(
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
    let (_, scratch) = C::map_impl(
        C::NIL,
        Ok(StackReq::empty()),
        &mut |(), scratch| match scratch {
            Err(x) => ((), Err(x)),
            Ok(x) => match StackReq::try_new_aligned::<T>(len, align) {
                Ok(y) => ((), x.try_and(y)),
                Err(y) => ((), Err(y)),
            },
        },
    );
    scratch
}

#[track_caller]
pub unsafe fn temp_mat_uninit<
    'a,
    C: ComplexContainer,
    T: ComplexField<C> + 'a,
    Rows: Shape + 'a,
    Cols: Shape + 'a,
>(
    ctx: &Ctx<C, T>,
    nrows: Rows,
    ncols: Cols,
    stack: &'a mut DynStack,
) -> (impl 'a + AsMatMut<C, T, Rows, Cols>, &'a mut DynStack) {
    help!(C);

    struct DynMat<'a, C: ComplexContainer, T: ComplexField<C>, Rows: Shape, Cols: Shape> {
        ptr: C::Of<*mut T>,
        nrows: Rows,
        ncols: Cols,
        col_stride: usize,
        __marker: PhantomData<(&'a T, T)>,
    }

    impl<'a, C: ComplexContainer, T: ComplexField<C>, Rows: Shape, Cols: Shape> Drop
        for DynMat<'a, C, T, Rows, Cols>
    {
        #[inline]
        fn drop(&mut self) {
            unsafe {
                map!(
                    copy!(self.ptr),
                    ptr,
                    core::ptr::drop_in_place(core::slice::from_raw_parts_mut(
                        ptr,
                        self.col_stride * self.ncols.unbound()
                    ))
                )
            };
        }
    }

    impl<'a, C: ComplexContainer, T: ComplexField<C>, Rows: Shape, Cols: Shape>
        AsMatMut<C, T, Rows, Cols> for DynMat<'a, C, T, Rows, Cols>
    {
        fn as_mat_mut(&mut self) -> crate::mat::MatMutGeneric<C, T, Rows, Cols> {
            unsafe {
                MatMutGeneric::from_raw_parts_mut(
                    copy!(self.ptr),
                    self.nrows,
                    self.ncols,
                    1,
                    self.col_stride as isize,
                )
            }
        }
    }
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
    _ = stack.make_aligned_uninit::<T>(len * C::N_COMPONENTS, align);

    let (uninit, stack) = C::map_impl(C::NIL, stack, &mut |(), stack| {
        stack.make_aligned_uninit::<T>(len, align)
    });

    let ptr = map!(uninit, uninit, uninit.as_mut_ptr() as *mut T);
    if core::mem::needs_drop::<T>() {
        unsafe {
            struct DropGuard<T> {
                ptr: *mut T,
                len: usize,
            }
            impl<T> Drop for DropGuard<T> {
                #[inline]
                fn drop(&mut self) {
                    unsafe {
                        core::ptr::drop_in_place(core::slice::from_raw_parts_mut(
                            self.ptr, self.len,
                        ))
                    };
                }
            }

            let mut guard = map!(copy!(ptr), ptr, DropGuard { ptr, len: 0 });
            for j in 0..len {
                let ptr = map!(copy!(ptr), ptr, ptr.add(j));
                let val = T::nan_impl(ctx);
                map!(zip!(ptr, val), (ptr, val), ptr.write(val));
                map!(as_mut!(guard), guard, guard.len += 1);
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
