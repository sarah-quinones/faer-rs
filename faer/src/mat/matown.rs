use super::*;
use crate::{internal_prelude::*, Idx, IdxInc, TryReserveError};
use core::{
    alloc::Layout,
    ops::{Index, IndexMut},
};
use dyn_stack::StackReq;
use faer_traits::{ComplexField, RealValue};
use matmut::MatMut;
use matref::MatRef;

#[inline]
pub fn align_for(size: usize, align: usize, needs_drop: bool) -> usize {
    if needs_drop || !size.is_power_of_two() {
        align
    } else {
        Ord::max(align, 64)
    }
}

// CURSED: currently avoiding inlining to get noalias annotations in llvm
#[inline(never)]
unsafe fn noalias_annotate<C: Container, T, Rows: Shape, Cols: Shape>(
    mut iter: C::Of<&mut [core::mem::MaybeUninit<T>]>,
    new_nrows: IdxInc<Rows>,
    old_nrows: IdxInc<Rows>,
    f: &mut impl FnMut(Idx<Rows>, Idx<Cols>) -> C::Of<T>,
    j: Idx<Cols>,
) {
    help!(C);
    let ptr = map!(rb_mut!(iter), slice, (slice.as_mut_ptr()));
    let mut iter = map!(
        copy!(ptr),
        ptr,
        core::slice::from_raw_parts_mut(ptr, new_nrows.unbound() - old_nrows.unbound())
    );

    let mut row_guard = map!(copy!(ptr), ptr, DropCol { ptr, nrows: 0 });
    for i in Rows::indices(old_nrows, new_nrows) {
        let ptr = map!(
            as_mut!(iter),
            iter,
            iter.as_mut_ptr().add(i.unbound()) as *mut T
        );
        map!(zip!(ptr, (*f)(i, j)), (ptr, val), ptr.write(val));
        map!(as_mut!(row_guard), guard, guard.nrows += 1);
    }
    core::mem::forget(row_guard);
}

pub struct DropIter<I: Iterator>(pub I);
impl<I: Iterator> Drop for DropIter<I> {
    #[inline]
    fn drop(&mut self) {
        pub struct DropIterRetry<'a, I: Iterator>(pub &'a mut I);
        impl<I: Iterator> Drop for DropIterRetry<'_, I> {
            #[inline]
            fn drop(&mut self) {
                self.0.for_each(drop);
            }
        }

        let in_case_of_panic = DropIterRetry(&mut self.0);
        in_case_of_panic.0.for_each(drop);
        core::mem::forget(in_case_of_panic);
    }
}

extern crate alloc;

struct RawMatUnit<T> {
    ptr: NonNull<T>,
    row_capacity: usize,
    col_capacity: usize,
    layout: StackReq,
    __marker: PhantomData<T>,
}

struct RawMat<C: Container, T> {
    ptr: C::Of<NonNull<T>>,
    row_capacity: usize,
    col_capacity: usize,
    layout: StackReq,
    __marker: PhantomData<T>,
}

impl<T> RawMatUnit<T> {
    fn try_with_capacity(
        mut row_capacity: usize,
        col_capacity: usize,
    ) -> Result<Self, TryReserveError> {
        help!(C);
        let mut layout = StackReq::new::<T>(1);
        let size = layout.size_bytes();
        let prev_align = layout.align_bytes();
        let align = align_for(size, prev_align, core::mem::needs_drop::<T>());

        if align > size {
            row_capacity = row_capacity
                .checked_next_multiple_of(align / size)
                .ok_or(TryReserveError::CapacityOverflow)?;
        }

        let size = size
            .checked_mul(row_capacity)
            .and_then(|size| size.checked_mul(col_capacity))
            .ok_or(TryReserveError::CapacityOverflow)?;

        let ptr = if size == 0 {
            layout = StackReq::empty();
            core::ptr::null_mut::<u8>().wrapping_add(align)
        } else {
            let new_layout = Layout::from_size_align(size, align)
                .map_err(|_| TryReserveError::CapacityOverflow)?;
            layout = StackReq::new_aligned::<u8>(new_layout.size(), new_layout.align());
            let ptr = unsafe { alloc::alloc::alloc(new_layout) };
            if ptr.is_null() {
                return Err(TryReserveError::AllocError { layout: new_layout });
            }
            ptr
        };
        let ptr = ptr as *mut T;

        Ok(Self {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            row_capacity,
            col_capacity,
            layout,
            __marker: PhantomData,
        })
    }

    fn into_raw_parts(self) -> (NonNull<T>, usize, usize, StackReq) {
        let this = core::mem::ManuallyDrop::new(self);
        (this.ptr, this.row_capacity, this.col_capacity, this.layout)
    }
}

impl<T> Drop for RawMatUnit<T> {
    #[inline]
    fn drop(&mut self) {
        if self.layout.size_bytes() > 0 {
            unsafe {
                alloc::alloc::dealloc(
                    self.ptr.as_ptr() as *mut u8,
                    Layout::from_size_align_unchecked(
                        self.layout.size_bytes(),
                        self.layout.align_bytes(),
                    ),
                )
            };
        }
    }
}

impl<C: Container, T> RawMat<C, T> {
    #[cold]
    fn try_with_capacity(
        row_capacity: usize,
        col_capacity: usize,
    ) -> Result<Self, TryReserveError> {
        help!(C);
        let mut err = None;
        let alloc = map!(C::NIL, (), {
            let alloc = RawMatUnit::<T>::try_with_capacity(row_capacity, col_capacity);
            if let Err(alloc_err) = &alloc {
                err = Some(*alloc_err);
            }
            alloc
        });
        if let Some(err) = err {
            return Err(err);
        }

        let mut layout = StackReq::empty();
        let mut row_capacity = row_capacity;
        let ptr = map!(alloc, alloc, {
            let (ptr, new_row_capacity, _, unit_layout) = alloc.unwrap().into_raw_parts();
            row_capacity = new_row_capacity;
            layout = unit_layout;
            ptr
        });

        Ok(Self {
            ptr,
            row_capacity,
            col_capacity,
            layout,
            __marker: PhantomData,
        })
    }

    #[cold]
    fn do_reserve_with(
        &mut self,
        nrows: usize,
        ncols: usize,
        new_row_capacity: usize,
        new_col_capacity: usize,
    ) -> Result<(), TryReserveError> {
        let old_row_capacity = self.row_capacity;
        let size = size_of::<T>();

        let new = Self::try_with_capacity(new_row_capacity, new_col_capacity)?;
        help!(C);

        unsafe fn move_mat(
            mut new: *mut u8,
            mut old: *const u8,

            col_bytes: usize,
            ncols: usize,

            new_byte_stride: isize,
            old_byte_stride: isize,
        ) {
            for _ in 0..ncols {
                core::ptr::copy_nonoverlapping(old, new, col_bytes);
                new = new.wrapping_offset(new_byte_stride);
                old = old.wrapping_offset(old_byte_stride);
            }
        }

        map!(zip!(copy!(new.ptr), copy!(self.ptr)), (new, old), {
            let new = new.as_ptr() as *mut u8;
            let old = old.as_ptr() as *const u8;

            unsafe {
                move_mat(
                    new,
                    old,
                    nrows * size,
                    ncols,
                    (new_row_capacity * size) as isize,
                    (old_row_capacity * size) as isize,
                )
            };
        });
        *self = new;
        Ok(())
    }

    fn try_reserve(
        &mut self,
        nrows: usize,
        ncols: usize,
        new_row_capacity: usize,
        new_col_capacity: usize,
    ) -> Result<(), TryReserveError> {
        let new_row_capacity = Ord::max(new_row_capacity, nrows);
        let new_col_capacity = Ord::max(new_col_capacity, ncols);

        if new_row_capacity > self.row_capacity || new_col_capacity > self.col_capacity {
            self.do_reserve_with(nrows, ncols, new_row_capacity, new_col_capacity)?
        }
        Ok(())
    }
}
impl<C: Container, T> Drop for RawMat<C, T> {
    fn drop(&mut self) {
        help!(C);
        drop(map!(
            copy!(self.ptr),
            ptr,
            RawMatUnit {
                ptr,
                row_capacity: self.row_capacity,
                col_capacity: self.col_capacity,
                layout: self.layout,
                __marker: PhantomData
            }
        ));
    }
}

pub struct Mat<C: Container, T, Rows: Shape = usize, Cols: Shape = usize> {
    raw: RawMat<C, T>,
    nrows: Rows,
    ncols: Cols,
}

pub struct DropCol<T> {
    ptr: *mut T,
    nrows: usize,
}

pub struct DropMat<T> {
    ptr: *mut T,
    nrows: usize,
    ncols: usize,
    byte_col_stride: usize,
}

impl<T> Drop for DropCol<T> {
    #[inline]
    fn drop(&mut self) {
        if const { core::mem::needs_drop::<T>() } {
            unsafe {
                let slice = core::slice::from_raw_parts_mut(self.ptr, self.nrows);
                core::ptr::drop_in_place(slice);
            }
        }
    }
}

impl<T> Drop for DropMat<T> {
    #[inline]
    fn drop(&mut self) {
        if const { core::mem::needs_drop::<T>() } {
            let mut ptr = self.ptr;

            if self.nrows > 0 {
                DropIter((0..self.ncols).map(|_| {
                    DropCol {
                        ptr,
                        nrows: self.nrows,
                    };
                    ptr = ptr.wrapping_byte_add(self.byte_col_stride);
                }));
            }
        }
    }
}

impl<C: Container, T, Rows: Shape, Cols: Shape> Drop for Mat<C, T, Rows, Cols> {
    #[inline]
    fn drop(&mut self) {
        if const { core::mem::needs_drop::<T>() } {
            help!(C);
            if self.nrows.unbound() > 0 && self.ncols.unbound() > 0 {
                unsafe {
                    let mut size = 0;
                    let ptr = map!(copy!(self.raw.ptr), ptr, ptr.as_ptr());
                    map!(copy!(ptr), x, size = size_of_val(&*x));
                    let row_capacity = self.raw.row_capacity;
                    let stride = row_capacity * size;

                    map!(
                        ptr,
                        ptr,
                        DropMat {
                            ptr,
                            nrows: self.nrows.unbound(),
                            ncols: self.ncols.unbound(),
                            byte_col_stride: stride,
                        }
                    );
                }
            }
        }
    }
}

impl<C: Container, T, Rows: Shape, Cols: Shape> Mat<C, T, Rows, Cols> {
    unsafe fn init_with(
        ptr: C::Of<*mut T>,
        old_nrows: IdxInc<Rows>,
        old_ncols: IdxInc<Cols>,
        new_nrows: IdxInc<Rows>,
        new_ncols: IdxInc<Cols>,
        row_capacity: usize,
        f: &mut impl FnMut(Idx<Rows>, Idx<Cols>) -> C::Of<T>,
    ) {
        help!(C);

        let stride = row_capacity;

        let mut ptr = map!(
            ptr,
            ptr,
            ptr.wrapping_add(stride * old_ncols.unbound())
                .wrapping_add(old_nrows.unbound())
        );
        let mut col_guard = map!(
            copy!(ptr),
            ptr,
            DropMat {
                ptr,
                nrows: new_nrows.unbound() - old_nrows.unbound(),
                ncols: 0,
                byte_col_stride: stride,
            }
        );

        for j in Cols::indices(old_ncols, new_ncols) {
            let old = copy!(ptr);

            noalias_annotate::<C, T, Rows, Cols>(
                map!(
                    copy!(ptr),
                    ptr,
                    core::slice::from_raw_parts_mut(
                        ptr as *mut _,
                        new_nrows.unbound() - old_nrows.unbound()
                    )
                ),
                new_nrows,
                old_nrows,
                f,
                j,
            );

            map!(as_mut!(col_guard), guard, guard.ncols += 1);
            ptr = map!(old, ptr, ptr.wrapping_add(stride));
        }
        core::mem::forget(col_guard);
    }

    pub fn from_fn(
        nrows: Rows,
        ncols: Cols,
        f: impl FnMut(Idx<Rows>, Idx<Cols>) -> C::Of<T>,
    ) -> Self {
        unsafe {
            help!(C);
            let raw = RawMat::<C, T>::try_with_capacity(nrows.unbound(), ncols.unbound()).unwrap();

            let ptr = map!(copy!(raw.ptr), ptr, ptr.as_ptr());
            Self::init_with(
                ptr,
                Rows::start(),
                Cols::start(),
                nrows.end(),
                ncols.end(),
                raw.row_capacity,
                &mut { f },
            );

            Self { raw, nrows, ncols }
        }
    }

    #[inline]
    pub fn zeros_with_ctx(ctx: &Ctx<C, T>, nrows: Rows, ncols: Cols) -> Self
    where
        C: ComplexContainer,
        T: ComplexField<C>,
    {
        Self::from_fn(nrows, ncols, |_, _| T::zero_impl(ctx))
    }

    pub fn try_reserve(
        &mut self,
        new_row_capacity: usize,
        new_col_capacity: usize,
    ) -> Result<(), TryReserveError> {
        self.raw.try_reserve(
            self.nrows.unbound(),
            self.ncols.unbound(),
            new_row_capacity,
            new_col_capacity,
        )
    }

    #[track_caller]
    pub fn reserve(&mut self, new_row_capacity: usize, new_col_capacity: usize) {
        self.try_reserve(new_row_capacity, new_col_capacity)
            .unwrap()
    }

    pub fn resize_with(
        &mut self,
        new_nrows: Rows,
        new_ncols: Cols,
        f: impl FnMut(Idx<Rows>, Idx<Cols>) -> C::Of<T>,
    ) {
        unsafe {
            let this = &mut *self;
            help!(C);
            if new_nrows == this.nrows && new_ncols == this.ncols {
                return;
            }

            this.truncate(new_nrows, new_ncols);

            if new_nrows > this.nrows || new_ncols > this.ncols {
                this.reserve(new_nrows.unbound(), new_ncols.unbound());
            }

            let mut f = f;

            if new_nrows > this.nrows {
                Self::init_with(
                    map!(copy!(this.raw.ptr), ptr, ptr.as_ptr()),
                    this.nrows.end(),
                    Cols::start(),
                    new_nrows.end(),
                    this.ncols.end(),
                    this.raw.row_capacity,
                    &mut f,
                );
                this.nrows = new_nrows;
            }

            if new_ncols > this.ncols {
                Self::init_with(
                    map!(copy!(this.raw.ptr), ptr, ptr.as_ptr()),
                    Rows::start(),
                    this.ncols.end(),
                    new_nrows.end(),
                    new_ncols.end(),
                    this.raw.row_capacity,
                    &mut f,
                );
                this.ncols = new_ncols;
            }
        };
    }
    pub fn truncate(&mut self, new_nrows: Rows, new_ncols: Cols) {
        help!(C);
        if new_ncols < self.ncols {
            let stride = self.raw.row_capacity;

            drop(map!(
                copy!(self.raw.ptr),
                ptr,
                DropMat {
                    ptr: ptr.as_ptr().wrapping_add(stride * new_ncols.unbound()),
                    nrows: self.nrows.unbound(),
                    ncols: self.ncols.unbound() - new_ncols.unbound(),
                    byte_col_stride: stride,
                }
            ));
            self.ncols = new_ncols;
        }
        if new_nrows < self.nrows {
            let size = size_of::<T>();
            let stride = size * self.raw.row_capacity;

            drop(map!(
                copy!(self.raw.ptr),
                ptr,
                DropMat {
                    ptr: ptr.as_ptr().wrapping_add(new_nrows.unbound()),
                    nrows: self.nrows.unbound() - new_nrows.unbound(),
                    ncols: self.ncols.unbound(),
                    byte_col_stride: stride,
                }
            ));
            self.nrows = new_nrows;
        }
    }

    pub fn into_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> Mat<C, T, V, H> {
        help!(C);
        let this = core::mem::ManuallyDrop::new(self);

        Mat {
            raw: RawMat {
                ptr: copy!(this.raw.ptr),
                row_capacity: this.raw.row_capacity,
                col_capacity: this.raw.col_capacity,
                layout: this.raw.layout,
                __marker: PhantomData,
            },
            nrows,
            ncols,
        }
    }
}

impl<C: Container, T, Rows: Shape, Cols: Shape> Mat<C, T, Rows, Cols> {
    #[inline]
    pub fn nrows(&self) -> Rows {
        self.nrows
    }
    #[inline]
    pub fn ncols(&self) -> Cols {
        self.ncols
    }

    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, C, T, Rows, Cols> {
        help!(C);
        unsafe {
            MatRef::from_raw_parts(
                map!(copy!(self.raw.ptr), ptr, ptr.as_ptr() as *const T),
                self.nrows,
                self.ncols,
                1,
                self.raw.row_capacity as isize,
            )
        }
    }

    #[inline]
    pub fn as_mut(&mut self) -> MatMut<'_, C, T, Rows, Cols> {
        help!(C);
        unsafe {
            MatMut::from_raw_parts_mut(
                map!(copy!(self.raw.ptr), ptr, ptr.as_ptr()),
                self.nrows,
                self.ncols,
                1,
                self.raw.row_capacity as isize,
            )
        }
    }
}

impl<C: Container, T: Clone, Rows: Shape, Cols: Shape> Clone for Mat<C, T, Rows, Cols> {
    #[inline]
    fn clone(&self) -> Self {
        help!(C);
        with_dim!(M, self.nrows().unbound());
        with_dim!(N, self.ncols().unbound());
        let this = self.as_ref().as_shape(M, N);
        Mat::from_fn(this.nrows(), this.ncols(), |i, j| {
            map!(this.at(i, j), val, val.clone())
        })
        .into_shape(self.nrows(), self.ncols())
    }
}

impl<T, Rows: Shape, Cols: Shape> Index<(Idx<Rows>, Idx<Cols>)> for Mat<Unit, T, Rows, Cols> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (Idx<Rows>, Idx<Cols>)) -> &Self::Output {
        self.as_ref().at(row, col)
    }
}

impl<T, Rows: Shape, Cols: Shape> IndexMut<(Idx<Rows>, Idx<Cols>)> for Mat<Unit, T, Rows, Cols> {
    #[inline]
    fn index_mut(&mut self, (row, col): (Idx<Rows>, Idx<Cols>)) -> &mut Self::Output {
        self.as_mut().at_mut(row, col)
    }
}

impl<C: Container, T: core::fmt::Debug, Rows: Shape, Cols: Shape> core::fmt::Debug
    for Mat<C, T, Rows, Cols>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<C: Container, T, Rows: Shape, Cols: Shape> Mat<C, T, Rows, Cols> {
    #[inline(always)]
    pub fn as_ptr(&self) -> C::Of<*const T> {
        self.as_ref().as_ptr()
    }

    #[inline(always)]
    pub fn shape(&self) -> (Rows, Cols) {
        (self.nrows(), self.ncols())
    }

    #[inline(always)]
    pub fn row_stride(&self) -> isize {
        1
    }

    #[inline(always)]
    pub fn col_stride(&self) -> isize {
        self.raw.row_capacity as isize
    }

    #[inline(always)]
    pub fn ptr_at(&self, row: IdxInc<Rows>, col: IdxInc<Cols>) -> C::Of<*const T> {
        self.as_ref().ptr_at(row, col)
    }

    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at(&self, row: Idx<Rows>, col: Idx<Cols>) -> C::Of<*const T> {
        self.as_ref().ptr_inbounds_at(row, col)
    }

    #[inline]
    #[track_caller]
    pub fn split_at(
        &self,
        row: IdxInc<Rows>,
        col: IdxInc<Cols>,
    ) -> (
        MatRef<'_, C, T, usize, usize>,
        MatRef<'_, C, T, usize, usize>,
        MatRef<'_, C, T, usize, usize>,
        MatRef<'_, C, T, usize, usize>,
    ) {
        self.as_ref().split_at(row, col)
    }

    #[inline]
    #[track_caller]
    pub fn split_at_row(
        &self,
        row: IdxInc<Rows>,
    ) -> (MatRef<'_, C, T, usize, Cols>, MatRef<'_, C, T, usize, Cols>) {
        self.as_ref().split_at_row(row)
    }

    #[inline]
    #[track_caller]
    pub fn split_at_col(
        &self,
        col: IdxInc<Cols>,
    ) -> (MatRef<'_, C, T, Rows, usize>, MatRef<'_, C, T, Rows, usize>) {
        self.as_ref().split_at_col(col)
    }

    #[inline(always)]
    pub fn transpose(&self) -> MatRef<'_, C, T, Cols, Rows> {
        self.as_ref().transpose()
    }

    #[inline(always)]
    pub fn conjugate(&self) -> MatRef<'_, C::Conj, T::Conj, Rows, Cols>
    where
        T: ConjUnit,
    {
        self.as_ref().conjugate()
    }

    #[inline(always)]
    pub fn canonical(&self) -> MatRef<'_, C::Canonical, T::Canonical, Rows, Cols>
    where
        T: ConjUnit,
    {
        self.as_ref().canonical()
    }

    #[inline(always)]
    pub fn adjoint(&self) -> MatRef<'_, C::Conj, T::Conj, Cols, Rows>
    where
        T: ConjUnit,
    {
        self.as_ref().adjoint()
    }

    #[inline(always)]
    pub fn at(&self, row: Idx<Rows>, col: Idx<Cols>) -> C::Of<&'_ T> {
        self.as_ref().at(row, col)
    }

    #[inline(always)]
    pub unsafe fn at_unchecked(&self, row: Idx<Rows>, col: Idx<Cols>) -> C::Of<&'_ T> {
        self.as_ref().at_unchecked(row, col)
    }

    #[inline]
    pub fn reverse_rows(&self) -> MatRef<'_, C, T, Rows, Cols> {
        self.as_ref().reverse_rows()
    }

    #[inline]
    pub fn reverse_cols(&self) -> MatRef<'_, C, T, Rows, Cols> {
        self.as_ref().reverse_cols()
    }

    #[inline]
    pub fn reverse_rows_and_cols(&self) -> MatRef<'_, C, T, Rows, Cols> {
        self.as_ref().reverse_rows_and_cols()
    }

    #[inline]
    pub fn submatrix<V: Shape, H: Shape>(
        &self,
        row_start: IdxInc<Rows>,
        col_start: IdxInc<Cols>,
        nrows: V,
        ncols: H,
    ) -> MatRef<'_, C, T, V, H> {
        self.as_ref().submatrix(row_start, col_start, nrows, ncols)
    }

    #[inline]
    pub fn subrows<V: Shape>(
        &self,
        row_start: IdxInc<Rows>,
        nrows: V,
    ) -> MatRef<'_, C, T, V, Cols> {
        self.as_ref().subrows(row_start, nrows)
    }

    #[inline]
    pub fn subcols<H: Shape>(
        &self,
        col_start: IdxInc<Cols>,
        ncols: H,
    ) -> MatRef<'_, C, T, Rows, H> {
        self.as_ref().subcols(col_start, ncols)
    }

    #[inline]
    pub fn submatrix_range(
        &self,
        rows: (impl Into<IdxInc<Rows>>, impl Into<IdxInc<Rows>>),
        cols: (impl Into<IdxInc<Cols>>, impl Into<IdxInc<Cols>>),
    ) -> MatRef<'_, C, T, usize, usize> {
        self.as_ref().submatrix_range(rows, cols)
    }

    #[inline]
    pub fn subrows_range(
        &self,
        rows: (impl Into<IdxInc<Rows>>, impl Into<IdxInc<Rows>>),
    ) -> MatRef<'_, C, T, usize, Cols> {
        self.as_ref().subrows_range(rows)
    }

    #[inline]
    pub fn subcols_range(
        &self,
        cols: (impl Into<IdxInc<Cols>>, impl Into<IdxInc<Cols>>),
    ) -> MatRef<'_, C, T, Rows, usize> {
        self.as_ref().subcols_range(cols)
    }

    #[inline]
    pub fn as_shape<V: Shape, H: Shape>(&self, nrows: V, ncols: H) -> MatRef<'_, C, T, V, H> {
        self.as_ref().as_shape(nrows, ncols)
    }

    #[inline]
    pub fn as_row_shape<V: Shape>(&self, nrows: V) -> MatRef<'_, C, T, V, Cols> {
        self.as_ref().as_row_shape(nrows)
    }

    #[inline]
    pub fn as_col_shape<H: Shape>(&self, ncols: H) -> MatRef<'_, C, T, Rows, H> {
        self.as_ref().as_col_shape(ncols)
    }

    #[inline]
    pub fn as_dyn_stride(&self) -> MatRef<'_, C, T, Rows, Cols, isize, isize> {
        self.as_ref().as_dyn_stride()
    }

    #[inline]
    pub fn as_dyn(&self) -> MatRef<'_, C, T, usize, usize> {
        self.as_ref().as_dyn()
    }

    #[inline]
    pub fn as_dyn_rows(&self) -> MatRef<'_, C, T, usize, Cols> {
        self.as_ref().as_dyn_rows()
    }

    #[inline]
    pub fn as_dyn_cols(&self) -> MatRef<'_, C, T, Rows, usize> {
        self.as_ref().as_dyn_cols()
    }

    #[inline]
    pub fn row(&self, i: Idx<Rows>) -> RowRef<'_, C, T, Cols> {
        self.as_ref().row(i)
    }

    #[inline]
    #[track_caller]
    pub fn col(&self, j: Idx<Cols>) -> ColRef<'_, C, T, Rows> {
        self.as_ref().col(j)
    }

    #[inline]
    pub fn col_iter(
        &self,
    ) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = ColRef<'_, C, T, Rows>> {
        self.as_ref().col_iter()
    }

    #[inline]
    pub fn row_iter(
        &self,
    ) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = RowRef<'_, C, T, Cols>> {
        self.as_ref().row_iter()
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_col_iter(
        &self,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = ColRef<'_, C, T, Rows>>
    where
        T: Sync,
    {
        self.as_ref().par_col_iter()
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_row_iter(
        &self,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = RowRef<'_, C, T, Cols>>
    where
        T: Sync,
    {
        self.as_ref().par_row_iter()
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_col_chunks(
        &self,
        chunk_size: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatRef<'_, C, T, Rows, usize>>
    where
        T: Sync,
    {
        self.as_ref().par_col_chunks(chunk_size)
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_col_partition(
        &self,
        count: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatRef<'_, C, T, Rows, usize>>
    where
        T: Sync,
    {
        self.as_ref().par_col_partition(count)
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_row_chunks(
        &self,
        chunk_size: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatRef<'_, C, T, usize, Cols>>
    where
        T: Sync,
    {
        self.as_ref().par_row_chunks(chunk_size)
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_row_partition(
        &self,
        count: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatRef<'_, C, T, usize, Cols>>
    where
        T: Sync,
    {
        self.as_ref().par_row_partition(count)
    }

    #[inline]
    pub fn try_as_col_major(&self) -> Option<MatRef<'_, C, T, Rows, Cols, ContiguousFwd>> {
        self.as_ref().try_as_col_major()
    }

    #[inline]
    pub fn try_as_row_major(&self) -> Option<MatRef<'_, C, T, Rows, Cols, isize, ContiguousFwd>> {
        self.as_ref().try_as_row_major()
    }

    #[inline]
    pub fn norm_max_with(&self, ctx: &Ctx<C::Canonical, T::Canonical>) -> RealValue<C, T>
    where
        C: Container<Canonical: ComplexContainer>,
        T: ConjUnit<Canonical: ComplexField<C::Canonical>>,
    {
        self.as_ref().norm_max_with(ctx)
    }

    #[inline]
    pub fn norm_max(&self) -> RealValue<C, T>
    where
        C: Container<Canonical: ComplexContainer>,
        T: ConjUnit<Canonical: ComplexField<C::Canonical, MathCtx: Default>>,
    {
        self.as_ref().norm_max()
    }

    #[inline]
    pub fn norm_l2_with(&self, ctx: &Ctx<C::Canonical, T::Canonical>) -> RealValue<C, T>
    where
        C: Container<Canonical: ComplexContainer>,
        T: ConjUnit<Canonical: ComplexField<C::Canonical>>,
    {
        self.as_ref().norm_l2_with(ctx)
    }

    #[inline]
    pub fn norm_l2(&self) -> RealValue<C, T>
    where
        C: Container<Canonical: ComplexContainer>,
        T: ConjUnit<Canonical: ComplexField<C::Canonical, MathCtx: Default>>,
    {
        self.as_ref().norm_l2()
    }
}

impl<C: Container, T, Rows: Shape, Cols: Shape> Mat<C, T, Rows, Cols> {
    #[inline(always)]
    pub fn as_ptr_mut(&mut self) -> C::Of<*mut T> {
        self.as_mut().as_ptr_mut()
    }

    #[inline(always)]
    pub fn ptr_at_mut(&mut self, row: IdxInc<Rows>, col: IdxInc<Cols>) -> C::Of<*mut T> {
        self.as_mut().ptr_at_mut(row, col)
    }

    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at_mut(&mut self, row: Idx<Rows>, col: Idx<Cols>) -> C::Of<*mut T> {
        self.as_mut().ptr_inbounds_at_mut(row, col)
    }

    #[inline]
    #[track_caller]
    pub fn split_at_mut(
        &mut self,
        row: IdxInc<Rows>,
        col: IdxInc<Cols>,
    ) -> (
        MatMut<'_, C, T, usize, usize>,
        MatMut<'_, C, T, usize, usize>,
        MatMut<'_, C, T, usize, usize>,
        MatMut<'_, C, T, usize, usize>,
    ) {
        self.as_mut().split_at_mut(row, col)
    }

    #[inline]
    #[track_caller]
    pub fn split_at_row_mut(
        &mut self,
        row: IdxInc<Rows>,
    ) -> (MatMut<'_, C, T, usize, Cols>, MatMut<'_, C, T, usize, Cols>) {
        self.as_mut().split_at_row_mut(row)
    }

    #[inline]
    #[track_caller]
    pub fn split_at_col_mut(
        &mut self,
        col: IdxInc<Cols>,
    ) -> (MatMut<'_, C, T, Rows, usize>, MatMut<'_, C, T, Rows, usize>) {
        self.as_mut().split_at_col_mut(col)
    }

    #[inline(always)]
    pub fn transpose_mut(&mut self) -> MatMut<'_, C, T, Cols, Rows> {
        self.as_mut().transpose_mut()
    }

    #[inline(always)]
    pub fn conjugate_mut(&mut self) -> MatMut<'_, C::Conj, T::Conj, Rows, Cols>
    where
        T: ConjUnit,
    {
        self.as_mut().conjugate_mut()
    }

    #[inline(always)]
    pub fn canonical_mut(&mut self) -> MatMut<'_, C::Canonical, T::Canonical, Rows, Cols>
    where
        T: ConjUnit,
    {
        self.as_mut().canonical_mut()
    }

    #[inline(always)]
    pub fn adjoint_mut(&mut self) -> MatMut<'_, C::Conj, T::Conj, Cols, Rows>
    where
        T: ConjUnit,
    {
        self.as_mut().adjoint_mut()
    }

    #[inline(always)]
    #[track_caller]
    pub fn at_mut(&mut self, row: Idx<Rows>, col: Idx<Cols>) -> C::Of<&'_ mut T> {
        self.as_mut().at_mut(row, col)
    }

    #[inline(always)]
    pub unsafe fn at_mut_unchecked(&mut self, row: Idx<Rows>, col: Idx<Cols>) -> C::Of<&'_ mut T> {
        self.as_mut().at_mut_unchecked(row, col)
    }

    #[inline]
    pub fn reverse_rows_mut(&mut self) -> MatMut<'_, C, T, Rows, Cols> {
        self.as_mut().reverse_rows_mut()
    }

    #[inline]
    pub fn reverse_cols_mut(&mut self) -> MatMut<'_, C, T, Rows, Cols> {
        self.as_mut().reverse_cols_mut()
    }

    #[inline]
    pub fn reverse_rows_and_cols_mut(&mut self) -> MatMut<'_, C, T, Rows, Cols> {
        self.as_mut().reverse_rows_and_cols_mut()
    }

    #[inline]
    pub fn submatrix_mut<V: Shape, H: Shape>(
        &mut self,
        row_start: IdxInc<Rows>,
        col_start: IdxInc<Cols>,
        nrows: V,
        ncols: H,
    ) -> MatMut<'_, C, T, V, H> {
        self.as_mut()
            .submatrix_mut(row_start, col_start, nrows, ncols)
    }

    #[inline]
    pub fn subrows_mut<V: Shape>(
        &mut self,
        row_start: IdxInc<Rows>,
        nrows: V,
    ) -> MatMut<'_, C, T, V, Cols> {
        self.as_mut().subrows_mut(row_start, nrows)
    }

    #[inline]
    pub fn subcols_mut<H: Shape>(
        &mut self,
        col_start: IdxInc<Cols>,
        ncols: H,
    ) -> MatMut<'_, C, T, Rows, H> {
        self.as_mut().subcols_mut(col_start, ncols)
    }

    #[inline]
    pub fn submatrix_range_mut(
        &mut self,
        rows: (impl Into<IdxInc<Rows>>, impl Into<IdxInc<Rows>>),
        cols: (impl Into<IdxInc<Cols>>, impl Into<IdxInc<Cols>>),
    ) -> MatMut<'_, C, T, usize, usize> {
        self.as_mut().submatrix_range_mut(rows, cols)
    }

    #[inline]
    pub fn subrows_range_mut(
        &mut self,
        rows: (impl Into<IdxInc<Rows>>, impl Into<IdxInc<Rows>>),
    ) -> MatMut<'_, C, T, usize, Cols> {
        self.as_mut().subrows_range_mut(rows)
    }

    #[inline]
    pub fn subcols_range_mut(
        &mut self,
        cols: (impl Into<IdxInc<Cols>>, impl Into<IdxInc<Cols>>),
    ) -> MatMut<'_, C, T, Rows, usize> {
        self.as_mut().subcols_range_mut(cols)
    }

    #[inline]
    #[track_caller]
    pub fn as_shape_mut<V: Shape, H: Shape>(
        &mut self,
        nrows: V,
        ncols: H,
    ) -> MatMut<'_, C, T, V, H> {
        self.as_mut().as_shape_mut(nrows, ncols)
    }

    #[inline]
    pub fn as_row_shape_mut<V: Shape>(&mut self, nrows: V) -> MatMut<'_, C, T, V, Cols> {
        self.as_mut().as_row_shape_mut(nrows)
    }

    #[inline]
    pub fn as_col_shape_mut<H: Shape>(&mut self, ncols: H) -> MatMut<'_, C, T, Rows, H> {
        self.as_mut().as_col_shape_mut(ncols)
    }

    #[inline]
    pub fn as_dyn_stride_mut(&mut self) -> MatMut<'_, C, T, Rows, Cols, isize, isize> {
        self.as_mut().as_dyn_stride_mut()
    }

    #[inline]
    pub fn as_dyn_mut(&mut self) -> MatMut<'_, C, T, usize, usize> {
        self.as_mut().as_dyn_mut()
    }

    #[inline]
    pub fn as_dyn_rows_mut(&mut self) -> MatMut<'_, C, T, usize, Cols> {
        self.as_mut().as_dyn_rows_mut()
    }

    #[inline]
    pub fn as_dyn_cols_mut(&mut self) -> MatMut<'_, C, T, Rows, usize> {
        self.as_mut().as_dyn_cols_mut()
    }

    #[inline]
    pub fn row_mut(&mut self, i: Idx<Rows>) -> RowMut<'_, C, T, Cols> {
        self.as_mut().row_mut(i)
    }

    #[inline]
    pub fn col_mut(&mut self, j: Idx<Cols>) -> ColMut<'_, C, T, Rows> {
        self.as_mut().col_mut(j)
    }

    #[inline]
    pub fn col_iter_mut(
        &mut self,
    ) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = ColMut<'_, C, T, Rows>> {
        self.as_mut().col_iter_mut()
    }

    #[inline]
    pub fn row_iter_mut(
        &mut self,
    ) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = RowMut<'_, C, T, Cols>> {
        self.as_mut().row_iter_mut()
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_col_iter_mut(
        &mut self,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = ColMut<'_, C, T, Rows>>
    where
        T: Send,
    {
        self.as_mut().par_col_iter_mut()
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_row_iter_mut(
        &mut self,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = RowMut<'_, C, T, Cols>>
    where
        T: Send,
    {
        self.as_mut().par_row_iter_mut()
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_col_chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatMut<'_, C, T, Rows, usize>>
    where
        T: Send,
    {
        self.as_mut().par_col_chunks_mut(chunk_size)
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_col_partition_mut(
        &mut self,
        count: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatMut<'_, C, T, Rows, usize>>
    where
        T: Send,
    {
        self.as_mut().par_col_partition_mut(count)
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_row_chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatMut<'_, C, T, usize, Cols>>
    where
        T: Send,
    {
        self.as_mut().par_row_chunks_mut(chunk_size)
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_row_partition_mut(
        &mut self,
        count: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatMut<'_, C, T, usize, Cols>>
    where
        T: Send,
    {
        self.as_mut().par_row_partition_mut(count)
    }

    #[inline]
    pub fn split_first_row_mut(
        &mut self,
    ) -> Option<(RowMut<'_, C, T, Cols>, MatMut<'_, C, T, usize, Cols>)> {
        self.as_mut().split_first_row_mut()
    }

    #[inline]
    pub fn try_as_col_major_mut(&mut self) -> Option<MatMut<'_, C, T, Rows, Cols, ContiguousFwd>> {
        self.as_mut().try_as_col_major_mut()
    }

    #[inline]
    pub fn try_as_row_major_mut(
        &mut self,
    ) -> Option<MatMut<'_, C, T, Rows, Cols, isize, ContiguousFwd>> {
        self.as_mut().try_as_row_major_mut()
    }

    #[inline]
    #[track_caller]
    pub fn write(&mut self, i: Idx<Rows>, j: Idx<Cols>) -> C::Of<&'_ mut T> {
        self.as_mut().at_mut(i, j)
    }

    #[inline]
    #[track_caller]
    pub fn two_cols_mut(
        &mut self,
        i0: Idx<Cols>,
        i1: Idx<Cols>,
    ) -> (ColMut<'_, C, T, Rows>, ColMut<'_, C, T, Rows>) {
        self.as_mut().two_cols_mut(i0, i1)
    }
    #[inline]
    #[track_caller]
    pub fn two_rows_mut(
        &mut self,
        i0: Idx<Rows>,
        i1: Idx<Rows>,
    ) -> (RowMut<'_, C, T, Cols>, RowMut<'_, C, T, Cols>) {
        self.as_mut().two_rows_mut(i0, i1)
    }

    #[inline]
    pub fn copy_from_triangular_lower_with_ctx<
        RhsC: Container<Canonical = C>,
        RhsT: ConjUnit<Canonical = T>,
    >(
        &mut self,
        ctx: &Ctx<C, T>,
        other: impl AsMatRef<C = RhsC, T = RhsT, Rows = Rows, Cols = Cols>,
    ) where
        C: ComplexContainer,
        T: ComplexField<C>,
    {
        self.as_mut()
            .copy_from_triangular_lower_with_ctx(ctx, other)
    }

    #[inline]
    pub fn copy_from_with_ctx<RhsC: Container<Canonical = C>, RhsT: ConjUnit<Canonical = T>>(
        &mut self,
        ctx: &Ctx<C, T>,
        other: impl AsMatRef<C = RhsC, T = RhsT, Rows = Rows, Cols = Cols>,
    ) where
        C: ComplexContainer,
        T: ComplexField<C>,
    {
        self.as_mut().copy_from_with_ctx(ctx, other)
    }

    #[inline]
    pub fn copy_from_strict_triangular_lower_with_ctx<
        RhsC: Container<Canonical = C>,
        RhsT: ConjUnit<Canonical = T>,
    >(
        &mut self,
        ctx: &Ctx<C, T>,
        other: impl AsMatRef<C = RhsC, T = RhsT, Rows = Rows, Cols = Cols>,
    ) where
        C: ComplexContainer,
        T: ComplexField<C>,
    {
        self.as_mut().copy_from_with_ctx(ctx, other)
    }
}
