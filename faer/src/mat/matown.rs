use super::*;
use crate::internal_prelude::*;
use crate::{Idx, IdxInc, TryReserveError, assert};
use core::alloc::Layout;
use dyn_stack::StackReq;
use faer_traits::ComplexField;

#[inline]
pub(crate) fn align_for(size: usize, align: usize, needs_drop: bool) -> usize {
	if needs_drop || !size.is_power_of_two() {
		align
	} else {
		Ord::max(align, 64)
	}
}

// CURSED: currently avoiding inlining to get noalias annotations in llvm
#[inline(never)]
unsafe fn noalias_annotate<T, Rows: Shape, Cols: Shape>(
	iter: &mut [core::mem::MaybeUninit<T>],
	new_nrows: IdxInc<Rows>,
	old_nrows: IdxInc<Rows>,
	f: &mut impl FnMut(Idx<Rows>, Idx<Cols>) -> T,
	j: Idx<Cols>,
) {
	let ptr = iter.as_mut_ptr();
	let iter = core::slice::from_raw_parts_mut(ptr, new_nrows.unbound() - old_nrows.unbound());

	let mut guard = DropCol {
		ptr: ptr as *mut T,
		nrows: 0,
	};
	for i in Rows::indices(old_nrows, new_nrows) {
		let ptr = iter.as_mut_ptr().add(i.unbound()) as *mut T;
		ptr.write((*f)(i, j));
		guard.nrows += 1;
	}
	core::mem::forget(guard);
}

pub(crate) struct DropIter<I: Iterator>(pub I);
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

struct RawMat<T> {
	ptr: NonNull<T>,
	row_capacity: usize,
	col_capacity: usize,
	layout: StackReq,
	__marker: PhantomData<T>,
}

impl<T> RawMatUnit<T> {
	fn try_with_capacity(mut row_capacity: usize, col_capacity: usize) -> Result<Self, TryReserveError> {
		let size = core::mem::size_of::<T>();
		let prev_align = core::mem::align_of::<T>();
		let align = align_for(size, prev_align, core::mem::needs_drop::<T>());

		if align > size {
			row_capacity = row_capacity
				.msrv_checked_next_multiple_of(align / size)
				.ok_or(TryReserveError::CapacityOverflow)?;
		}

		let size = size
			.checked_mul(row_capacity)
			.and_then(|size| size.checked_mul(col_capacity))
			.ok_or(TryReserveError::CapacityOverflow)?;

		let layout;
		let ptr = if size == 0 {
			layout = StackReq::empty();
			core::ptr::null_mut::<u8>().wrapping_add(align)
		} else {
			let new_layout = Layout::from_size_align(size, align).map_err(|_| TryReserveError::CapacityOverflow)?;
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
					Layout::from_size_align_unchecked(self.layout.size_bytes(), self.layout.align_bytes()),
				)
			};
		}
	}
}

impl<T> RawMat<T> {
	#[cold]
	fn try_with_capacity(row_capacity: usize, col_capacity: usize) -> Result<Self, TryReserveError> {
		let mut err = None;
		let alloc = {
			let alloc = RawMatUnit::<T>::try_with_capacity(row_capacity, col_capacity);
			if let Err(alloc_err) = &alloc {
				err = Some(*alloc_err);
			}
			alloc
		};
		if let Some(err) = err {
			return Err(err);
		}

		let layout;
		let row_capacity;
		let ptr = {
			let (ptr, new_row_capacity, _, unit_layout) = alloc.unwrap().into_raw_parts();
			row_capacity = new_row_capacity;
			layout = unit_layout;
			ptr
		};

		Ok(Self {
			ptr,
			row_capacity,
			col_capacity,
			layout,
			__marker: PhantomData,
		})
	}

	#[cold]
	fn do_reserve_with(&mut self, nrows: usize, ncols: usize, new_row_capacity: usize, new_col_capacity: usize) -> Result<(), TryReserveError> {
		let old_row_capacity = self.row_capacity;
		let size = core::mem::size_of::<T>();

		let new = Self::try_with_capacity(new_row_capacity, new_col_capacity)?;
		let new_row_capacity = new.row_capacity;

		unsafe fn move_mat(mut new: *mut u8, mut old: *const u8, col_bytes: usize, ncols: usize, new_byte_stride: isize, old_byte_stride: isize) {
			for _ in 0..ncols {
				core::ptr::copy_nonoverlapping(old, new, col_bytes);
				new = new.wrapping_offset(new_byte_stride);
				old = old.wrapping_offset(old_byte_stride);
			}
		}

		{
			let new = new.ptr;
			let old = self.ptr;

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
		};

		*self = new;
		Ok(())
	}

	fn try_reserve(&mut self, nrows: usize, ncols: usize, new_row_capacity: usize, new_col_capacity: usize) -> Result<(), TryReserveError> {
		let new_row_capacity = Ord::max(new_row_capacity, nrows);
		let new_col_capacity = Ord::max(new_col_capacity, ncols);

		if new_row_capacity > self.row_capacity || new_col_capacity > self.col_capacity {
			self.do_reserve_with(nrows, ncols, new_row_capacity, new_col_capacity)?
		}
		Ok(())
	}
}
impl<T> Drop for RawMat<T> {
	fn drop(&mut self) {
		let ptr = self.ptr;
		drop(RawMatUnit {
			ptr,
			row_capacity: self.row_capacity,
			col_capacity: self.col_capacity,
			layout: self.layout,
			__marker: PhantomData,
		});
	}
}

/// see [`super::Mat`]
pub struct Own<T, Rows: Shape = usize, Cols: Shape = usize> {
	raw: RawMat<T>,
	nrows: Rows,
	ncols: Cols,
}

unsafe impl<T: Send, Rows: Shape, Cols: Shape> Send for Own<T, Rows, Cols> {}
unsafe impl<T: Sync, Rows: Shape, Cols: Shape> Sync for Own<T, Rows, Cols> {}

pub(crate) struct DropCol<T> {
	ptr: *mut T,
	nrows: usize,
}

pub(crate) struct DropMat<T> {
	ptr: *mut T,
	nrows: usize,
	ncols: usize,
	byte_col_stride: usize,
}

impl<T> Drop for DropCol<T> {
	#[inline]
	fn drop(&mut self) {
		if try_const! { core::mem::needs_drop::<T>() } {
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
		if try_const! { core::mem::needs_drop::<T>() } {
			let mut ptr = self.ptr;

			if self.nrows > 0 {
				DropIter((0..self.ncols).map(|_| {
					DropCol { ptr, nrows: self.nrows };
					ptr = ptr.wrapping_byte_add(self.byte_col_stride);
				}));
			}
		}
	}
}

impl<T, Rows: Shape, Cols: Shape> Drop for Own<T, Rows, Cols> {
	#[inline]
	fn drop(&mut self) {
		if try_const! { core::mem::needs_drop::<T>() } {
			if self.nrows.unbound() > 0 && self.ncols.unbound() > 0 {
				let size = core::mem::size_of::<T>();
				let ptr = self.raw.ptr.as_ptr();
				let row_capacity = self.raw.row_capacity;
				let stride = row_capacity * size;

				drop(DropMat {
					ptr,
					nrows: self.nrows.unbound(),
					ncols: self.ncols.unbound(),
					byte_col_stride: stride,
				})
			}
		}
	}
}

impl<T> Mat<T> {
	/// returns an empty matrix of dimension `0×0`.
	#[inline]
	pub const fn new() -> Self {
		Self(Own {
			raw: RawMat {
				ptr: NonNull::dangling(),
				row_capacity: 0,
				col_capacity: 0,
				layout: StackReq::EMPTY,
				__marker: PhantomData,
			},
			nrows: 0,
			ncols: 0,
		})
	}

	/// reserves the minimum capacity for `row_capacity` rows and `col_capacity`
	/// columns without reallocating. does nothing if the capacity is already sufficient
	#[track_caller]
	pub fn with_capacity(row_capacity: usize, col_capacity: usize) -> Self {
		let mut me = Self::new();
		me.reserve(row_capacity, col_capacity);
		me
	}
}

impl<T, Rows: Shape, Cols: Shape> Mat<T, Rows, Cols> {
	unsafe fn init_with(
		ptr: *mut T,
		old_nrows: IdxInc<Rows>,
		old_ncols: IdxInc<Cols>,
		new_nrows: IdxInc<Rows>,
		new_ncols: IdxInc<Cols>,
		row_capacity: usize,
		f: &mut impl FnMut(Idx<Rows>, Idx<Cols>) -> T,
	) {
		let stride = row_capacity;

		let mut ptr = ptr.wrapping_add(stride * old_ncols.unbound());
		let mut col_guard = DropMat {
			ptr,
			nrows: new_nrows.unbound() - old_nrows.unbound(),
			ncols: 0,
			byte_col_stride: stride,
		};

		for j in Cols::indices(old_ncols, new_ncols) {
			let old = ptr;

			noalias_annotate::<T, Rows, Cols>(
				core::slice::from_raw_parts_mut(ptr as *mut _, new_nrows.unbound() - old_nrows.unbound()),
				new_nrows,
				old_nrows,
				f,
				j,
			);

			col_guard.ncols += 1;
			ptr = old.wrapping_add(stride);
		}
		core::mem::forget(col_guard);
	}

	/// returns a new matrix with dimensions `(nrows, ncols)`, filled with the provided function
	pub fn from_fn(nrows: Rows, ncols: Cols, f: impl FnMut(Idx<Rows>, Idx<Cols>) -> T) -> Self {
		unsafe {
			let raw = RawMat::<T>::try_with_capacity(nrows.unbound(), ncols.unbound()).unwrap();

			let ptr = raw.ptr.as_ptr();
			Self::init_with(ptr, Rows::start(), Cols::start(), nrows.end(), ncols.end(), raw.row_capacity, &mut { f });

			Self(Own { raw, nrows, ncols })
		}
	}

	/// returns a new matrix with dimensions `(nrows, ncols)`, filled with zeros
	#[inline]
	pub fn zeros(nrows: Rows, ncols: Cols) -> Self
	where
		T: ComplexField,
	{
		Self::from_fn(nrows, ncols, |_, _| T::zero_impl())
	}

	/// returns a new matrix with dimensions `(nrows, ncols)`, filled with ones
	#[inline]
	pub fn ones(nrows: Rows, ncols: Cols) -> Self
	where
		T: ComplexField,
	{
		Self::from_fn(nrows, ncols, |_, _| T::one_impl())
	}

	/// returns a new identity matrix, with ones on the diagonal and zeros everywhere else
	#[inline]
	pub fn identity(nrows: Rows, ncols: Cols) -> Self
	where
		T: ComplexField,
	{
		Self::from_fn(
			nrows,
			ncols,
			|i, j| if i.unbound() == j.unbound() { T::one_impl() } else { T::zero_impl() },
		)
	}

	/// returns a new matrix with dimensions `(nrows, ncols)`, filled with `value`
	#[inline]
	pub fn full(nrows: Rows, ncols: Cols, value: T) -> Self
	where
		T: Clone,
	{
		Self::from_fn(nrows, ncols, |_, _| value.clone())
	}

	/// reserves the minimum capacity for `new_row_capacity` rows and `new_col_capacity`
	/// columns without reallocating, or returns an error in case of failure. does nothing if the
	/// capacity is already sufficient
	pub fn try_reserve(&mut self, new_row_capacity: usize, new_col_capacity: usize) -> Result<(), TryReserveError> {
		self.0
			.raw
			.try_reserve(self.0.nrows.unbound(), self.0.ncols.unbound(), new_row_capacity, new_col_capacity)
	}

	/// reserves the minimum capacity for `new_row_capacity` rows and `new_col_capacity`
	/// columns without reallocating. does nothing if the capacity is already sufficient
	#[track_caller]
	pub fn reserve(&mut self, new_row_capacity: usize, new_col_capacity: usize) {
		self.try_reserve(new_row_capacity, new_col_capacity).unwrap()
	}

	/// resizes the matrix in-place so that the new dimensions are `(new_nrows, new_ncols)`.
	/// new elements are created with the given function `f`, so that elements at index `(i, j)`
	/// are created by calling `f(i, j)`.
	pub fn resize_with(&mut self, new_nrows: Rows, new_ncols: Cols, f: impl FnMut(Idx<Rows>, Idx<Cols>) -> T) {
		unsafe {
			let this = &mut *self;

			if new_nrows == this.0.nrows && new_ncols == this.0.ncols {
				return;
			}

			this.truncate(new_nrows, new_ncols);

			if new_nrows > this.0.nrows || new_ncols > this.0.ncols {
				this.reserve(new_nrows.unbound(), new_ncols.unbound());
			}

			let mut f = f;

			if new_nrows > this.0.nrows {
				Self::init_with(
					this.0.raw.ptr.as_ptr(),
					this.0.nrows.end(),
					Cols::start(),
					new_nrows.end(),
					this.0.ncols.end(),
					this.0.raw.row_capacity,
					&mut f,
				);
				this.0.nrows = new_nrows;
			}

			if new_ncols > this.0.ncols {
				Self::init_with(
					this.0.raw.ptr.as_ptr(),
					Rows::start(),
					this.0.ncols.end(),
					new_nrows.end(),
					new_ncols.end(),
					this.0.raw.row_capacity,
					&mut f,
				);
				this.0.ncols = new_ncols;
			}
		};
	}

	/// truncates the matrix so that its new dimensions are `new_nrows` and `new_ncols`.  
	/// both of the new dimensions must be smaller than or equal to the current dimensions
	///
	/// # panics
	/// the function panics if any of the following conditions are violated:
	/// - `new_nrows > self.nrows()`
	/// - `new_ncols > self.ncols()`
	pub fn truncate(&mut self, new_nrows: Rows, new_ncols: Cols) {
		if new_ncols < self.0.ncols {
			let stride = self.0.raw.row_capacity;

			drop(DropMat {
				ptr: self.0.raw.ptr.as_ptr().wrapping_add(stride * new_ncols.unbound()),
				nrows: self.0.nrows.unbound(),
				ncols: self.0.ncols.unbound() - new_ncols.unbound(),
				byte_col_stride: stride,
			});
			self.0.ncols = new_ncols;
		}
		if new_nrows < self.0.nrows {
			let size = core::mem::size_of::<T>();
			let stride = size * self.0.raw.row_capacity;

			drop(DropMat {
				ptr: self.0.raw.ptr.as_ptr().wrapping_add(new_nrows.unbound()),
				nrows: self.0.nrows.unbound() - new_nrows.unbound(),
				ncols: self.0.ncols.unbound(),
				byte_col_stride: stride,
			});
			self.0.nrows = new_nrows;
		}
	}

	/// see [`MatRef::as_shape`]
	#[track_caller]
	pub fn into_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> Mat<T, V, H> {
		assert!(all(self.nrows().unbound() == nrows.unbound(), self.ncols().unbound() == ncols.unbound()));
		let this = core::mem::ManuallyDrop::new(self);

		Mat {
			0: Own {
				raw: RawMat {
					ptr: this.0.raw.ptr,
					row_capacity: this.0.raw.row_capacity,
					col_capacity: this.0.raw.col_capacity,
					layout: this.0.raw.layout,
					__marker: PhantomData,
				},
				nrows,
				ncols,
			},
		}
	}

	/// set the dimensions of the matrix.
	///
	/// # safety
	/// the behavior is undefined if any of the following conditions are violated:
	/// - `nrows < self.row_capacity()`
	/// - `ncols < self.col_capacity()`
	/// - the elements that were previously out of bounds but are now in bounds must be
	/// initialized
	pub unsafe fn set_dims(&mut self, nrows: Rows, ncols: Cols) {
		self.0.nrows = nrows;
		self.0.ncols = ncols;
	}

	/// returns a reference to a slice over the column at the given index
	pub fn col_as_slice(&self, j: Idx<Cols>) -> &[T] {
		self.col(j).try_as_col_major().unwrap().as_slice()
	}

	/// returns a reference to a slice over the column at the given index
	pub fn col_as_slice_mut(&mut self, j: Idx<Cols>) -> &mut [T] {
		self.col_mut(j).try_as_col_major_mut().unwrap().as_slice_mut()
	}
}

impl<T, Rows: Shape, Cols: Shape> Mat<T, Rows, Cols> {
	/// returns the number of rows of the matrix
	#[inline]
	pub fn nrows(&self) -> Rows {
		self.0.nrows
	}

	/// returns the number of columns of the matrix
	#[inline]
	pub fn ncols(&self) -> Cols {
		self.0.ncols
	}
}

impl<T: Clone, Rows: Shape, Cols: Shape> Clone for Own<T, Rows, Cols> {
	#[inline]
	fn clone(&self) -> Self {
		let __self__ = Mat::from_inner_ref(self);
		with_dim!(M, __self__.nrows().unbound());
		with_dim!(N, __self__.ncols().unbound());
		let this = __self__.as_ref().as_shape(M, N);
		Mat::from_fn(this.nrows(), this.ncols(), |i, j| this.at(i, j).clone())
			.into_shape(__self__.nrows(), __self__.ncols())
			.0
	}
}

impl<T: core::fmt::Debug, Rows: Shape, Cols: Shape> core::fmt::Debug for Own<T, Rows, Cols> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		self.rb().fmt(f)
	}
}

impl<T, Rows: Shape, Cols: Shape> Mat<T, Rows, Cols> {
	/// returns a pointer to the matrix data
	#[inline(always)]
	pub fn as_ptr(&self) -> *const T {
		self.as_ref().as_ptr()
	}

	/// returns the number of rows and columns of the matrix
	#[inline(always)]
	pub fn shape(&self) -> (Rows, Cols) {
		(self.nrows(), self.ncols())
	}

	/// returns the row stride of the matrix, specified in number of elements, not in bytes
	#[inline(always)]
	pub fn row_stride(&self) -> isize {
		1
	}

	/// returns the column stride of the matrix, specified in number of elements, not in bytes
	#[inline(always)]
	pub fn col_stride(&self) -> isize {
		self.0.raw.row_capacity as isize
	}

	/// returns a raw pointer to the element at the given index
	#[inline(always)]
	pub fn ptr_at(&self, row: IdxInc<Rows>, col: IdxInc<Cols>) -> *const T {
		self.as_ref().ptr_at(row, col)
	}

	/// returns a raw pointer to the element at the given index, assuming the provided index
	/// is within the matrix bounds
	///
	/// # safety
	/// the behavior is undefined if any of the following conditions are violated:
	/// * `row < self.nrows()`
	/// * `col < self.ncols()`
	#[inline(always)]
	#[track_caller]
	pub unsafe fn ptr_inbounds_at(&self, row: Idx<Rows>, col: Idx<Cols>) -> *const T {
		self.as_ref().ptr_inbounds_at(row, col)
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::split_at`]
	pub fn split_at(
		&self,
		row: IdxInc<Rows>,
		col: IdxInc<Cols>,
	) -> (
		MatRef<'_, T, usize, usize>,
		MatRef<'_, T, usize, usize>,
		MatRef<'_, T, usize, usize>,
		MatRef<'_, T, usize, usize>,
	) {
		self.as_ref().split_at(row, col)
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::split_at_row`]
	pub fn split_at_row(&self, row: IdxInc<Rows>) -> (MatRef<'_, T, usize, Cols>, MatRef<'_, T, usize, Cols>) {
		self.as_ref().split_at_row(row)
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::split_at_col`]
	pub fn split_at_col(&self, col: IdxInc<Cols>) -> (MatRef<'_, T, Rows, usize>, MatRef<'_, T, Rows, usize>) {
		self.as_ref().split_at_col(col)
	}

	#[inline(always)]
	/// see [`MatRef::transpose`]
	pub fn transpose(&self) -> MatRef<'_, T, Cols, Rows> {
		self.as_ref().transpose()
	}

	#[inline(always)]
	/// see [`MatRef::conjugate`]
	pub fn conjugate(&self) -> MatRef<'_, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		self.as_ref().conjugate()
	}

	#[inline(always)]
	/// see [`MatRef::canonical`]
	pub fn canonical(&self) -> MatRef<'_, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		self.as_ref().canonical()
	}

	#[inline(always)]
	/// see [`MatRef::adjoint`]
	pub fn adjoint(&self) -> MatRef<'_, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.as_ref().adjoint()
	}

	#[inline]
	/// see [`MatRef::reverse_rows`]
	pub fn reverse_rows(&self) -> MatRef<'_, T, Rows, Cols> {
		self.as_ref().reverse_rows()
	}

	#[inline]
	/// see [`MatRef::reverse_cols`]
	pub fn reverse_cols(&self) -> MatRef<'_, T, Rows, Cols> {
		self.as_ref().reverse_cols()
	}

	#[inline]
	/// see [`MatRef::reverse_rows_and_cols`]
	pub fn reverse_rows_and_cols(&self) -> MatRef<'_, T, Rows, Cols> {
		self.as_ref().reverse_rows_and_cols()
	}

	#[inline]
	/// see [`MatRef::submatrix`]
	pub fn submatrix<V: Shape, H: Shape>(&self, row_start: IdxInc<Rows>, col_start: IdxInc<Cols>, nrows: V, ncols: H) -> MatRef<'_, T, V, H> {
		self.as_ref().submatrix(row_start, col_start, nrows, ncols)
	}

	#[inline]
	/// see [`MatRef::subrows`]
	pub fn subrows<V: Shape>(&self, row_start: IdxInc<Rows>, nrows: V) -> MatRef<'_, T, V, Cols> {
		self.as_ref().subrows(row_start, nrows)
	}

	#[inline]
	/// see [`MatRef::subcols`]
	pub fn subcols<H: Shape>(&self, col_start: IdxInc<Cols>, ncols: H) -> MatRef<'_, T, Rows, H> {
		self.as_ref().subcols(col_start, ncols)
	}

	#[inline]
	/// see [`MatRef::as_shape`]
	pub fn as_shape<V: Shape, H: Shape>(&self, nrows: V, ncols: H) -> MatRef<'_, T, V, H> {
		self.as_ref().as_shape(nrows, ncols)
	}

	#[inline]
	/// see [`MatRef::as_row_shape`]
	pub fn as_row_shape<V: Shape>(&self, nrows: V) -> MatRef<'_, T, V, Cols> {
		self.as_ref().as_row_shape(nrows)
	}

	#[inline]
	/// see [`MatRef::as_col_shape`]
	pub fn as_col_shape<H: Shape>(&self, ncols: H) -> MatRef<'_, T, Rows, H> {
		self.as_ref().as_col_shape(ncols)
	}

	#[inline]
	/// see [`MatRef::as_dyn_stride`]
	pub fn as_dyn_stride(&self) -> MatRef<'_, T, Rows, Cols, isize, isize> {
		self.as_ref().as_dyn_stride()
	}

	#[inline]
	/// see [`MatRef::as_dyn`]
	pub fn as_dyn(&self) -> MatRef<'_, T, usize, usize> {
		self.as_ref().as_dyn()
	}

	#[inline]
	/// see [`MatRef::as_dyn_rows`]
	pub fn as_dyn_rows(&self) -> MatRef<'_, T, usize, Cols> {
		self.as_ref().as_dyn_rows()
	}

	#[inline]
	/// see [`MatRef::as_dyn_cols`]
	pub fn as_dyn_cols(&self) -> MatRef<'_, T, Rows, usize> {
		self.as_ref().as_dyn_cols()
	}

	#[inline]
	/// see [`MatRef::row`]
	pub fn row(&self, i: Idx<Rows>) -> RowRef<'_, T, Cols> {
		self.as_ref().row(i)
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::col`]
	pub fn col(&self, j: Idx<Cols>) -> ColRef<'_, T, Rows> {
		self.as_ref().col(j)
	}

	#[inline]
	/// see [`MatRef::col_iter`]
	pub fn col_iter(&self) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = ColRef<'_, T, Rows>> {
		self.as_ref().col_iter()
	}

	#[inline]
	/// see [`MatRef::row_iter`]
	pub fn row_iter(&self) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = RowRef<'_, T, Cols>> {
		self.as_ref().row_iter()
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`MatRef::par_col_iter`]
	pub fn par_col_iter(&self) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = ColRef<'_, T, Rows>>
	where
		T: Sync,
	{
		self.as_ref().par_col_iter()
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`MatRef::par_row_iter`]
	pub fn par_row_iter(&self) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = RowRef<'_, T, Cols>>
	where
		T: Sync,
	{
		self.as_ref().par_row_iter()
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`MatRef::par_col_chunks`]
	pub fn par_col_chunks(&self, chunk_size: usize) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatRef<'_, T, Rows, usize>>
	where
		T: Sync,
	{
		self.as_ref().par_col_chunks(chunk_size)
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`MatRef::par_col_partition`]
	pub fn par_col_partition(&self, count: usize) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatRef<'_, T, Rows, usize>>
	where
		T: Sync,
	{
		self.as_ref().par_col_partition(count)
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`MatRef::par_row_chunks`]
	pub fn par_row_chunks(&self, chunk_size: usize) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatRef<'_, T, usize, Cols>>
	where
		T: Sync,
	{
		self.as_ref().par_row_chunks(chunk_size)
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`MatRef::par_row_partition`]
	pub fn par_row_partition(&self, count: usize) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatRef<'_, T, usize, Cols>>
	where
		T: Sync,
	{
		self.as_ref().par_row_partition(count)
	}

	#[inline]
	/// see [`MatRef::try_as_col_major`]
	pub fn try_as_col_major(&self) -> Option<MatRef<'_, T, Rows, Cols, ContiguousFwd>> {
		self.as_ref().try_as_col_major()
	}

	#[inline]
	/// see [`MatRef::try_as_row_major`]
	pub fn try_as_row_major(&self) -> Option<MatRef<'_, T, Rows, Cols, isize, ContiguousFwd>> {
		self.as_ref().try_as_row_major()
	}

	#[track_caller]
	#[inline]
	/// see [`MatRef::get`]
	pub fn get<RowRange, ColRange>(&self, row: RowRange, col: ColRange) -> <MatRef<'_, T, Rows, Cols> as MatIndex<RowRange, ColRange>>::Target
	where
		for<'a> MatRef<'a, T, Rows, Cols>: MatIndex<RowRange, ColRange>,
	{
		<MatRef<'_, T, Rows, Cols> as MatIndex<RowRange, ColRange>>::get(self.as_ref(), row, col)
	}

	#[track_caller]
	#[inline]
	/// see [`MatRef::get_unchecked`]
	pub unsafe fn get_unchecked<RowRange, ColRange>(
		&self,
		row: RowRange,
		col: ColRange,
	) -> <MatRef<'_, T, Rows, Cols> as MatIndex<RowRange, ColRange>>::Target
	where
		for<'a> MatRef<'a, T, Rows, Cols>: MatIndex<RowRange, ColRange>,
	{
		unsafe { <MatRef<'_, T, Rows, Cols> as MatIndex<RowRange, ColRange>>::get_unchecked(self.as_ref(), row, col) }
	}

	#[track_caller]
	#[inline]
	/// see [`MatMut::get_mut`]
	pub fn get_mut<RowRange, ColRange>(&mut self, row: RowRange, col: ColRange) -> <MatMut<'_, T, Rows, Cols> as MatIndex<RowRange, ColRange>>::Target
	where
		for<'a> MatMut<'a, T, Rows, Cols>: MatIndex<RowRange, ColRange>,
	{
		<MatMut<'_, T, Rows, Cols> as MatIndex<RowRange, ColRange>>::get(self.as_mut(), row, col)
	}

	#[track_caller]
	#[inline]
	/// see [`MatMut::get_mut_unchecked`]
	pub unsafe fn get_mut_unchecked<RowRange, ColRange>(
		&mut self,
		row: RowRange,
		col: ColRange,
	) -> <MatMut<'_, T, Rows, Cols> as MatIndex<RowRange, ColRange>>::Target
	where
		for<'a> MatMut<'a, T, Rows, Cols>: MatIndex<RowRange, ColRange>,
	{
		unsafe { <MatMut<'_, T, Rows, Cols> as MatIndex<RowRange, ColRange>>::get_unchecked(self.as_mut(), row, col) }
	}
}

impl<T, Rows: Shape, Cols: Shape> Mat<T, Rows, Cols> {
	#[inline(always)]
	/// returns a pointer to the matrix data
	pub fn as_ptr_mut(&mut self) -> *mut T {
		self.as_mut().as_ptr_mut()
	}

	#[inline(always)]
	/// returns a raw pointer to the element at the given index
	pub fn ptr_at_mut(&mut self, row: IdxInc<Rows>, col: IdxInc<Cols>) -> *mut T {
		self.as_mut().ptr_at_mut(row, col)
	}

	#[inline(always)]
	#[track_caller]
	/// returns a raw pointer to the element at the given index, assuming the provided index
	/// is within the matrix bounds
	///
	/// # safety
	/// the behavior is undefined if any of the following conditions are violated:
	/// * `row < self.nrows()`
	/// * `col < self.ncols()`
	pub unsafe fn ptr_inbounds_at_mut(&mut self, row: Idx<Rows>, col: Idx<Cols>) -> *mut T {
		self.as_mut().ptr_inbounds_at_mut(row, col)
	}

	#[inline]
	#[track_caller]
	/// see [`MatMut::split_at_mut`]
	pub fn split_at_mut(
		&mut self,
		row: IdxInc<Rows>,
		col: IdxInc<Cols>,
	) -> (
		MatMut<'_, T, usize, usize>,
		MatMut<'_, T, usize, usize>,
		MatMut<'_, T, usize, usize>,
		MatMut<'_, T, usize, usize>,
	) {
		self.as_mut().split_at_mut(row, col)
	}

	#[inline]
	#[track_caller]
	/// see [`MatMut::split_at_row_mut`]
	pub fn split_at_row_mut(&mut self, row: IdxInc<Rows>) -> (MatMut<'_, T, usize, Cols>, MatMut<'_, T, usize, Cols>) {
		self.as_mut().split_at_row_mut(row)
	}

	#[inline]
	#[track_caller]
	/// see [`MatMut::split_at_col_mut`]
	pub fn split_at_col_mut(&mut self, col: IdxInc<Cols>) -> (MatMut<'_, T, Rows, usize>, MatMut<'_, T, Rows, usize>) {
		self.as_mut().split_at_col_mut(col)
	}

	#[inline(always)]
	/// see [`MatMut::transpose_mut`]
	pub fn transpose_mut(&mut self) -> MatMut<'_, T, Cols, Rows> {
		self.as_mut().transpose_mut()
	}

	#[inline(always)]
	/// see [`MatMut::conjugate_mut`]
	pub fn conjugate_mut(&mut self) -> MatMut<'_, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		self.as_mut().conjugate_mut()
	}

	#[inline(always)]
	/// see [`MatMut::canonical_mut`]
	pub fn canonical_mut(&mut self) -> MatMut<'_, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		self.as_mut().canonical_mut()
	}

	#[inline(always)]
	/// see [`MatMut::adjoint_mut`]
	pub fn adjoint_mut(&mut self) -> MatMut<'_, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.as_mut().adjoint_mut()
	}

	#[inline]
	/// see [`MatMut::reverse_rows_mut`]
	pub fn reverse_rows_mut(&mut self) -> MatMut<'_, T, Rows, Cols> {
		self.as_mut().reverse_rows_mut()
	}

	#[inline]
	/// see [`MatMut::reverse_cols_mut`]
	pub fn reverse_cols_mut(&mut self) -> MatMut<'_, T, Rows, Cols> {
		self.as_mut().reverse_cols_mut()
	}

	#[inline]
	/// see [`MatMut::reverse_rows_and_cols_mut`]
	pub fn reverse_rows_and_cols_mut(&mut self) -> MatMut<'_, T, Rows, Cols> {
		self.as_mut().reverse_rows_and_cols_mut()
	}

	#[inline]
	/// see [`MatMut::submatrix_mut`]
	pub fn submatrix_mut<V: Shape, H: Shape>(&mut self, row_start: IdxInc<Rows>, col_start: IdxInc<Cols>, nrows: V, ncols: H) -> MatMut<'_, T, V, H> {
		self.as_mut().submatrix_mut(row_start, col_start, nrows, ncols)
	}

	#[inline]
	/// see [`MatMut::subrows_mut`]
	pub fn subrows_mut<V: Shape>(&mut self, row_start: IdxInc<Rows>, nrows: V) -> MatMut<'_, T, V, Cols> {
		self.as_mut().subrows_mut(row_start, nrows)
	}

	#[inline]
	/// see [`MatMut::subcols_mut`]
	pub fn subcols_mut<H: Shape>(&mut self, col_start: IdxInc<Cols>, ncols: H) -> MatMut<'_, T, Rows, H> {
		self.as_mut().subcols_mut(col_start, ncols)
	}

	#[inline]
	#[track_caller]
	/// see [`MatMut::as_shape_mut`]
	pub fn as_shape_mut<V: Shape, H: Shape>(&mut self, nrows: V, ncols: H) -> MatMut<'_, T, V, H> {
		self.as_mut().as_shape_mut(nrows, ncols)
	}

	#[inline]
	/// see [`MatMut::as_row_shape_mut`]
	pub fn as_row_shape_mut<V: Shape>(&mut self, nrows: V) -> MatMut<'_, T, V, Cols> {
		self.as_mut().as_row_shape_mut(nrows)
	}

	#[inline]
	/// see [`MatMut::as_col_shape_mut`]
	pub fn as_col_shape_mut<H: Shape>(&mut self, ncols: H) -> MatMut<'_, T, Rows, H> {
		self.as_mut().as_col_shape_mut(ncols)
	}

	#[inline]
	/// see [`MatMut::as_dyn_stride_mut`]
	pub fn as_dyn_stride_mut(&mut self) -> MatMut<'_, T, Rows, Cols, isize, isize> {
		self.as_mut().as_dyn_stride_mut()
	}

	#[inline]
	/// see [`MatMut::as_dyn_mut`]
	pub fn as_dyn_mut(&mut self) -> MatMut<'_, T, usize, usize> {
		self.as_mut().as_dyn_mut()
	}

	#[inline]
	/// see [`MatMut::as_dyn_rows_mut`]
	pub fn as_dyn_rows_mut(&mut self) -> MatMut<'_, T, usize, Cols> {
		self.as_mut().as_dyn_rows_mut()
	}

	#[inline]
	/// see [`MatMut::as_dyn_cols_mut`]
	pub fn as_dyn_cols_mut(&mut self) -> MatMut<'_, T, Rows, usize> {
		self.as_mut().as_dyn_cols_mut()
	}

	#[inline]
	/// see [`MatMut::row_mut`]
	pub fn row_mut(&mut self, i: Idx<Rows>) -> RowMut<'_, T, Cols> {
		self.as_mut().row_mut(i)
	}

	#[inline]
	/// see [`MatMut::col_mut`]
	pub fn col_mut(&mut self, j: Idx<Cols>) -> ColMut<'_, T, Rows> {
		self.as_mut().col_mut(j)
	}

	#[inline]
	/// see [`MatMut::col_iter_mut`]
	pub fn col_iter_mut(&mut self) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = ColMut<'_, T, Rows>> {
		self.as_mut().col_iter_mut()
	}

	#[inline]
	/// see [`MatMut::row_iter_mut`]
	pub fn row_iter_mut(&mut self) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = RowMut<'_, T, Cols>> {
		self.as_mut().row_iter_mut()
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`MatMut::par_col_iter_mut`]
	pub fn par_col_iter_mut(&mut self) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = ColMut<'_, T, Rows>>
	where
		T: Send,
	{
		self.as_mut().par_col_iter_mut()
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`MatMut::par_row_iter_mut`]
	pub fn par_row_iter_mut(&mut self) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = RowMut<'_, T, Cols>>
	where
		T: Send,
	{
		self.as_mut().par_row_iter_mut()
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`MatMut::par_col_chunks_mut`]
	pub fn par_col_chunks_mut(&mut self, chunk_size: usize) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatMut<'_, T, Rows, usize>>
	where
		T: Send,
	{
		self.as_mut().par_col_chunks_mut(chunk_size)
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`MatMut::par_col_partition_mut`]
	pub fn par_col_partition_mut(&mut self, count: usize) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatMut<'_, T, Rows, usize>>
	where
		T: Send,
	{
		self.as_mut().par_col_partition_mut(count)
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`MatMut::par_row_chunks_mut`]
	pub fn par_row_chunks_mut(&mut self, chunk_size: usize) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatMut<'_, T, usize, Cols>>
	where
		T: Send,
	{
		self.as_mut().par_row_chunks_mut(chunk_size)
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`MatMut::par_row_partition_mut`]
	pub fn par_row_partition_mut(&mut self, count: usize) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatMut<'_, T, usize, Cols>>
	where
		T: Send,
	{
		self.as_mut().par_row_partition_mut(count)
	}

	#[inline]
	/// see [`MatMut::split_first_row_mut`]
	pub fn split_first_row_mut(&mut self) -> Option<(RowMut<'_, T, Cols>, MatMut<'_, T, usize, Cols>)> {
		self.as_mut().split_first_row_mut()
	}

	#[inline]
	/// see [`MatMut::try_as_col_major_mut`]
	pub fn try_as_col_major_mut(&mut self) -> Option<MatMut<'_, T, Rows, Cols, ContiguousFwd>> {
		self.as_mut().try_as_col_major_mut()
	}

	#[inline]
	/// see [`MatMut::try_as_row_major_mut`]
	pub fn try_as_row_major_mut(&mut self) -> Option<MatMut<'_, T, Rows, Cols, isize, ContiguousFwd>> {
		self.as_mut().try_as_row_major_mut()
	}

	#[inline]
	#[track_caller]
	/// see [`MatMut::two_cols_mut`]
	pub fn two_cols_mut(&mut self, i0: Idx<Cols>, i1: Idx<Cols>) -> (ColMut<'_, T, Rows>, ColMut<'_, T, Rows>) {
		self.as_mut().two_cols_mut(i0, i1)
	}

	#[inline]
	#[track_caller]
	/// see [`MatMut::two_rows_mut`]
	pub fn two_rows_mut(&mut self, i0: Idx<Rows>, i1: Idx<Rows>) -> (RowMut<'_, T, Cols>, RowMut<'_, T, Cols>) {
		self.as_mut().two_rows_mut(i0, i1)
	}
}

impl<T, Dim: Shape> Mat<T, Dim, Dim> {
	#[inline]
	/// see [`MatRef::diagonal`]
	pub fn diagonal(&self) -> DiagRef<'_, T, Dim, isize> {
		self.as_ref().diagonal()
	}

	#[inline]
	/// see [`MatMut::diagonal_mut`]
	pub fn diagonal_mut(&mut self) -> DiagMut<'_, T, Dim, isize> {
		self.as_mut().diagonal_mut()
	}
}

impl<'short, T, Rows: Shape, Cols: Shape> Reborrow<'short> for Own<T, Rows, Cols> {
	type Target = Ref<'short, T, Rows, Cols>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		Ref {
			imp: MatView {
				ptr: self.raw.ptr,
				nrows: self.nrows,
				ncols: self.ncols,
				row_stride: 1,
				col_stride: self.raw.row_capacity as isize,
			},
			__marker: PhantomData,
		}
	}
}
impl<'short, T, Rows: Shape, Cols: Shape> ReborrowMut<'short> for Own<T, Rows, Cols> {
	type Target = Mut<'short, T, Rows, Cols>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		Mut {
			imp: MatView {
				ptr: self.raw.ptr,
				nrows: self.nrows,
				ncols: self.ncols,
				row_stride: 1,
				col_stride: self.raw.row_capacity as isize,
			},
			__marker: PhantomData,
		}
	}
}

impl<T, Cols: Shape> Mat<T, usize, Cols> {
	/// inserts a row at the end of the matrix
	/// # panics
	/// The function panics if the number of columns in the row does not match the number of columns
	/// in the matrix
	pub fn push_row(&mut self, row: RowRef<'_, T, Cols>)
	where
		T: Clone,
	{
		self::assert!(
			self.ncols() == row.ncols(),
			"row ncols ({:?}) must match matrix ncols ({:?})",
			row.ncols(),
			self.ncols()
		);

		self.resize_with(self.nrows() + 1, self.ncols(), |_, j| row[j].clone());
	}
}

impl<T, Rows: Shape> Mat<T, Rows, usize> {
	/// inserts a col at the end of the matrix
	/// # panics
	/// The function panics if the number of rows in the col does not match the number of rows in
	/// the matrix
	pub fn push_col(&mut self, col: ColRef<'_, T, Rows>)
	where
		T: Clone,
	{
		self::assert!(
			self.nrows() == col.nrows(),
			"col nrows ({:?}) must match matrix nrows ({:?})",
			col.nrows(),
			self.nrows()
		);

		self.resize_with(self.nrows(), self.ncols() + 1, |i, _| col[i].clone());
	}
}

impl<T, Rows: Shape, Cols: Shape> Mat<T, Rows, Cols>
where
	T: RealField,
{
	/// see [MatRef::min]
	pub fn min(self) -> Option<T> {
		MatRef::internal_min(self.as_dyn())
	}

	/// see [MatRef::min]
	pub fn max(self) -> Option<T> {
		MatRef::internal_max(self.as_dyn())
	}
}

#[cfg(test)]
mod tests {
	use crate::{assert, mat};

	#[test]
	fn test_resize() {
		// Create a matrix
		let mut m = mat![
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
			[10.0, 11.0, 12.0], //
		];

		m.resize_with(m.nrows() + 1, m.ncols(), |_, _| 99.9);

		let target = mat![
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
			[10.0, 11.0, 12.0],
			[99.9, 99.9, 99.9], //
		];

		assert!(m == target);
	}

	#[test]
	fn test_resize_5() {
		// Create a matrix
		let mut m = mat![
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
			[10.0, 11.0, 12.0], //
		];

		m.resize_with(m.nrows() + 5, m.ncols(), |_, _| 99.9);

		let target = mat![
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
			[10.0, 11.0, 12.0],
			[99.9, 99.9, 99.9], //
			[99.9, 99.9, 99.9], //
			[99.9, 99.9, 99.9], //
			[99.9, 99.9, 99.9], //
			[99.9, 99.9, 99.9], //
		];

		assert!(m == target);
	}

	#[test]
	fn test_resize_5_1() {
		// Create a matrix
		let mut m = mat![
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
			[10.0, 11.0, 12.0], //
		];

		m.resize_with(m.nrows() + 5, m.ncols() + 1, |_, _| 99.9);

		let target = mat![
			[1.0, 2.0, 3.0, 99.9],
			[4.0, 5.0, 6.0, 99.9],
			[7.0, 8.0, 9.0, 99.9],
			[10.0, 11.0, 12.0, 99.9],
			[99.9, 99.9, 99.9, 99.9], //
			[99.9, 99.9, 99.9, 99.9], //
			[99.9, 99.9, 99.9, 99.9], //
			[99.9, 99.9, 99.9, 99.9], //
			[99.9, 99.9, 99.9, 99.9], //
		];

		assert!(m == target);
	}

	#[test]
	fn test_push_row() {
		let mut m = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];

		let row = row![10.0, 11.0, 12.0];

		m.push_row(row.as_ref());

		let target = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0],];

		assert!(m == target);
	}

	#[test]
	#[should_panic]
	fn test_push_row_panic() {
		let mut m = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];

		// row has one too many columns
		let row = row![10.0, 11.0, 12.0, 13.0];

		m.push_row(row.as_ref());
	}

	#[test]
	fn test_push_col() {
		let mut m = mat![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0],];

		let col = col![4.0, 4.0, 4.0];

		m.push_col(col.as_ref());

		let target = mat![[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0],];

		assert!(m == target);
	}

	#[test]
	#[should_panic]
	fn test_push_col_panic() {
		let mut m = mat![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0],];

		let col = col![4.0, 4.0, 4.0, 4.0];

		m.push_col(col.as_ref());
	}

	#[test]
	fn test_min() {
		use crate::Mat;
		let m = mat![
			[1.0, 5.0, 3.0],
			[4.0, 2.0, 9.0],
			[7.0, 8.0, 6.0], //
		];

		assert_eq!(m.min(), Some(1.0));

		let empty: Mat<f64> = Mat::new();
		assert_eq!(empty.min(), None);
	}

	#[test]
	fn test_max() {
		use crate::Mat;
		let m = mat![
			[1.0, 5.0, 3.0],
			[4.0, 2.0, 9.0],
			[7.0, 8.0, 6.0], //
		];

		assert_eq!(m.max(), Some(9.0));

		let empty: Mat<f64> = Mat::new();
		assert_eq!(empty.max(), None);
	}
}
