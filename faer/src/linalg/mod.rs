//! linear algebra module
//!
//! contains low level routines and the implementation of their corresponding high level
//! wrappers
//!
//! # memory allocation
//! since most `faer` crates aim to expose a low level api for optimal performance, most algorithms
//! try to defer memory allocation to the user
//!
//! however, since a lot of algorithms need some form of temporary space for intermediate
//! computations, they may ask for a slice of memory for that purpose, by taking a [`stack:
//! MemStack`](dyn_stack::MemStack) parameter. a [`MemStack`] is a thin wrapper over a slice of
//! memory bytes. this memory may come from any valid source (heap allocation, fixed-size array on
//! the stack, etc.). the functions taking a [`MemStack`] parameter have a corresponding function
//! with a similar name ending in `_scratch` that returns the memory requirements of the algorithm.
//! for example:
//! [`householder::apply_block_householder_on_the_left_in_place_with_conj`] and
//! [`householder::apply_block_householder_on_the_left_in_place_scratch`]
//!
//! the memory stack may be reused in user-code to avoid repeated allocations, and it is also
//! possible to compute the sum ([`dyn_stack::StackReq::all_of`]) or union
//! ([`dyn_stack::StackReq::any_of`]) of multiple scratch requirements, in order to optimally
//! combine them into a single allocation
//!
//! after computing a [`dyn_stack::StackReq`], one can query its layout to allocate the
//! required memory. the simplest way to do so is through [`dyn_stack::MemBuffer::new`]

use crate::internal_prelude::*;
use core::marker::PhantomData;
use dyn_stack::StackReq;
use faer_traits::ComplexField;

use crate::Shape;
use crate::mat::matown::align_for;
use crate::mat::{AsMatMut, MatMut};

/// returns the stack requirements for creating a temporary matrix with the given dimensions.
pub fn temp_mat_scratch<T: ComplexField>(nrows: usize, ncols: usize) -> StackReq {
	let align = align_for(core::mem::size_of::<T>(), core::mem::align_of::<T>(), core::mem::needs_drop::<T>());

	let mut col_stride = nrows;
	if align > core::mem::size_of::<T>() {
		col_stride = col_stride.msrv_next_multiple_of(align / core::mem::size_of::<T>());
	}
	let len = col_stride.checked_mul(ncols).unwrap();
	StackReq::new_aligned::<T>(len, align)
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
		unsafe { core::ptr::drop_in_place(core::slice::from_raw_parts_mut(self.ptr, self.col_stride * self.ncols.unbound())) };
	}
}

impl<'a, T: ComplexField, Rows: Shape, Cols: Shape> AsMatRef for DynMat<'a, T, Rows, Cols> {
	type Cols = Cols;
	type Owned = Mat<T, Rows, Cols>;
	type Rows = Rows;
	type T = T;

	fn as_mat_ref(&self) -> crate::mat::MatRef<'_, T, Rows, Cols> {
		unsafe { MatRef::from_raw_parts(self.ptr as *const T, self.nrows, self.ncols, 1, self.col_stride as isize) }
	}
}

impl<'a, T: ComplexField, Rows: Shape, Cols: Shape> AsMatMut for DynMat<'a, T, Rows, Cols> {
	fn as_mat_mut(&mut self) -> crate::mat::MatMut<'_, T, Rows, Cols> {
		unsafe { MatMut::from_raw_parts_mut(self.ptr, self.nrows, self.ncols, 1, self.col_stride as isize) }
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

/// creates a temporary matrix of uninit values, from the given memory stack.
#[track_caller]
pub unsafe fn temp_mat_uninit<'a, T: ComplexField + 'a, Rows: Shape + 'a, Cols: Shape + 'a>(
	nrows: Rows,
	ncols: Cols,
	stack: &'a mut MemStack,
) -> (impl 'a + AsMatMut<T = T, Rows = Rows, Cols = Cols>, &'a mut MemStack) {
	let align = align_for(core::mem::size_of::<T>(), core::mem::align_of::<T>(), core::mem::needs_drop::<T>());

	let mut col_stride = nrows.unbound();
	if align > core::mem::size_of::<T>() {
		col_stride = col_stride.msrv_next_multiple_of(align / core::mem::size_of::<T>());
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

/// creates a temporary matrix of zero values, from the given memory stack.
#[track_caller]
pub fn temp_mat_zeroed<'a, T: ComplexField + 'a, Rows: Shape + 'a, Cols: Shape + 'a>(
	nrows: Rows,
	ncols: Cols,
	stack: &'a mut MemStack,
) -> (impl 'a + AsMatMut<T = T, Rows = Rows, Cols = Cols>, &'a mut MemStack) {
	let align = align_for(core::mem::size_of::<T>(), core::mem::align_of::<T>(), core::mem::needs_drop::<T>());

	let mut col_stride = nrows.unbound();
	if align > core::mem::size_of::<T>() {
		col_stride = col_stride.msrv_next_multiple_of(align / core::mem::size_of::<T>());
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
/// triangular matrix inverse
pub mod triangular_inverse;
/// triangular matrix solve
pub mod triangular_solve;

pub(crate) mod reductions;
/// matrix zipping implementation
pub mod zip;

pub mod householder;
/// jacobi rotation matrix
pub mod jacobi;

/// kronecker product
pub mod kron;

pub mod cholesky;
pub mod lu;
pub mod qr;

pub mod evd;
pub mod gevd;
pub mod svd;

mod mat_ops;

/// high level solvers
pub mod solvers;
