use super::*;
use crate::internal_prelude::*;
use crate::{Idx, MaybeIdx, assert, debug_assert};
use core::ops::Range;
use core::{fmt, iter};

/// implementation of symbolic representation
pub mod symbolic {
	/// generic `SymbolicSparseRowMat` wrapper
	pub mod generic {
		use core::fmt::Debug;
		use reborrow::*;

		/// generic `SymbolicSparseRowMat` wrapper
		#[derive(Copy, Clone)]
		#[repr(transparent)]
		pub struct SymbolicSparseRowMat<Inner>(pub Inner);

		impl<Inner: Debug> Debug for SymbolicSparseRowMat<Inner> {
			#[inline(always)]
			fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
				self.0.fmt(f)
			}
		}

		impl<Inner> SymbolicSparseRowMat<Inner> {
			/// wrap by reference
			#[inline(always)]
			pub fn from_inner_ref(inner: &Inner) -> &Self {
				unsafe { &*(inner as *const Inner as *const Self) }
			}

			/// wrap by mutable reference
			#[inline(always)]
			pub fn from_inner_mut(inner: &mut Inner) -> &mut Self {
				unsafe { &mut *(inner as *mut Inner as *mut Self) }
			}
		}

		impl<Inner> core::ops::Deref for SymbolicSparseRowMat<Inner> {
			type Target = Inner;

			#[inline(always)]
			fn deref(&self) -> &Self::Target {
				&self.0
			}
		}

		impl<Inner> core::ops::DerefMut for SymbolicSparseRowMat<Inner> {
			#[inline(always)]
			fn deref_mut(&mut self) -> &mut Self::Target {
				&mut self.0
			}
		}

		impl<'short, Inner: Reborrow<'short>> Reborrow<'short> for SymbolicSparseRowMat<Inner> {
			type Target = SymbolicSparseRowMat<Inner::Target>;

			#[inline(always)]
			fn rb(&'short self) -> Self::Target {
				SymbolicSparseRowMat(self.0.rb())
			}
		}

		impl<'short, Inner: ReborrowMut<'short>> ReborrowMut<'short> for SymbolicSparseRowMat<Inner> {
			type Target = SymbolicSparseRowMat<Inner::Target>;

			#[inline(always)]
			fn rb_mut(&'short mut self) -> Self::Target {
				SymbolicSparseRowMat(self.0.rb_mut())
			}
		}

		impl<Inner: IntoConst> IntoConst for SymbolicSparseRowMat<Inner> {
			type Target = SymbolicSparseRowMat<Inner::Target>;

			#[inline(always)]
			fn into_const(self) -> Self::Target {
				SymbolicSparseRowMat(self.0.into_const())
			}
		}
	}

	/// see [`super::SymbolicSparseRowMatRef`]
	pub struct Ref<'a, I, Rows = usize, Cols = usize> {
		pub(crate) nrows: Rows,
		pub(crate) ncols: Cols,
		pub(crate) row_ptr: &'a [I],
		pub(crate) row_nnz: Option<&'a [I]>,
		pub(crate) col_idx: &'a [I],
	}

	/// see [`super::SymbolicSparseRowMat`]
	#[derive(Clone)]
	pub struct Own<I, Rows = usize, Cols = usize> {
		pub(crate) nrows: Rows,
		pub(crate) ncols: Cols,
		pub(crate) row_ptr: alloc::vec::Vec<I>,
		pub(crate) row_nnz: Option<alloc::vec::Vec<I>>,
		pub(crate) col_idx: alloc::vec::Vec<I>,
	}
}

/// implementation of numeric representation
pub mod numeric {
	/// generic `SparseRowMat` wrapper
	pub mod generic {
		use core::fmt::Debug;
		use reborrow::*;

		/// generic `SparseRowMat` wrapper
		#[derive(Copy, Clone)]
		#[repr(transparent)]
		pub struct SparseRowMat<Inner>(pub Inner);

		impl<Inner: Debug> Debug for SparseRowMat<Inner> {
			#[inline(always)]
			fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
				self.0.fmt(f)
			}
		}

		impl<Inner> SparseRowMat<Inner> {
			/// wrap by reference
			#[inline(always)]
			pub fn from_inner_ref(inner: &Inner) -> &Self {
				unsafe { &*(inner as *const Inner as *const Self) }
			}

			/// wrap by mutable reference
			#[inline(always)]
			pub fn from_inner_mut(inner: &mut Inner) -> &mut Self {
				unsafe { &mut *(inner as *mut Inner as *mut Self) }
			}
		}

		impl<'short, Inner: Reborrow<'short>> Reborrow<'short> for SparseRowMat<Inner> {
			type Target = SparseRowMat<Inner::Target>;

			#[inline(always)]
			fn rb(&'short self) -> Self::Target {
				SparseRowMat(self.0.rb())
			}
		}

		impl<'short, Inner: ReborrowMut<'short>> ReborrowMut<'short> for SparseRowMat<Inner> {
			type Target = SparseRowMat<Inner::Target>;

			#[inline(always)]
			fn rb_mut(&'short mut self) -> Self::Target {
				SparseRowMat(self.0.rb_mut())
			}
		}

		impl<Inner: IntoConst> IntoConst for SparseRowMat<Inner> {
			type Target = SparseRowMat<Inner::Target>;

			#[inline(always)]
			fn into_const(self) -> Self::Target {
				SparseRowMat(self.0.into_const())
			}
		}
	}

	/// see [`super::SparseRowMatRef`]
	pub struct Ref<'a, I, T, Rows = usize, Cols = usize> {
		pub(crate) symbolic: super::SymbolicSparseRowMatRef<'a, I, Rows, Cols>,
		pub(crate) val: &'a [T],
	}

	/// see [`super::SparseRowMatMut`]
	pub struct Mut<'a, I, T, Rows = usize, Cols = usize> {
		pub(crate) symbolic: super::SymbolicSparseRowMatRef<'a, I, Rows, Cols>,
		pub(crate) val: &'a mut [T],
	}

	/// see [`super::SparseRowMat`]
	#[derive(Clone)]
	pub struct Own<I, T, Rows = usize, Cols = usize> {
		pub(crate) symbolic: super::SymbolicSparseRowMat<I, Rows, Cols>,
		pub(crate) val: alloc::vec::Vec<T>,
	}
}

/// symbolic view structure of sparse matrix in row format, either compressed or uncompressed
///
/// # invariants
/// - `nrows <= I::Signed::MAX` (always checked)
/// - `ncols <= I::Signed::MAX` (always checked)
/// - `row_ptrs` has length `nrows + 1` (always checked)
/// - `row_ptrs` is increasing
/// - `row_ptrs[0]..row_ptrs[nrows]` is a valid range in row_indices (always checked, assuming
///   increasing)
/// - if `nnz_per_row` is `none`, elements of `col_indices[row_ptrs[i]..row_ptrs[i + 1]]` are less
///   than `ncols`
///
/// - `nnz_per_row[i] <= row_ptrs[i+1] - row_ptrs[i]`
/// - if `nnz_per_row` is `some(_)`, elements of `col_indices[row_ptrs[i]..][..nnz_per_row[i]]` are
///   less than `ncols`
///
/// # soft invariants
/// - within each row, column indices are sorted in increasing order
///
/// # note
/// some algorithms allow working with matrices containing unsorted row indices per column
///
/// passing such a matrix to an algorithm that does not explicitly permit this is unspecified
/// (though not undefined) behavior
pub type SymbolicSparseRowMatRef<'a, I, Rows = usize, Cols = usize> = symbolic::generic::SymbolicSparseRowMat<symbolic::Ref<'a, I, Rows, Cols>>;

/// owning symbolic structure of sparse matrix in row format, either compressed or
/// uncompressed
///
/// see [`SymbolicSparseRowMatRef`]
pub type SymbolicSparseRowMat<I, Rows = usize, Cols = usize> = symbolic::generic::SymbolicSparseRowMat<symbolic::Own<I, Rows, Cols>>;

/// view over sparse row major matrix
///
/// see [`SymbolicSparseRowMatRef`]
pub type SparseRowMatRef<'a, I, T, Rows = usize, Cols = usize> = numeric::generic::SparseRowMat<numeric::Ref<'a, I, T, Rows, Cols>>;

/// view over sparse row major matrix
///
/// see [`SymbolicSparseRowMatRef`]
pub type SparseRowMatMut<'a, I, T, Rows = usize, Cols = usize> = numeric::generic::SparseRowMat<numeric::Mut<'a, I, T, Rows, Cols>>;

/// owning sparse row major matrix
///
/// see [`SymbolicSparseRowMatRef`]
pub type SparseRowMat<I, T, Rows = usize, Cols = usize> = numeric::generic::SparseRowMat<numeric::Own<I, T, Rows, Cols>>;

impl<'a, I, Rows: Copy, Cols: Copy> Copy for symbolic::Ref<'a, I, Rows, Cols> {}
impl<'a, I, T, Rows: Copy, Cols: Copy> Copy for numeric::Ref<'a, I, T, Rows, Cols> {}

impl<'a, I, Rows: Copy, Cols: Copy> Clone for symbolic::Ref<'a, I, Rows, Cols> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}
impl<'a, I, T, Rows: Copy, Cols: Copy> Clone for numeric::Ref<'a, I, T, Rows, Cols> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}

impl<'a, I, Rows: Copy, Cols: Copy> IntoConst for symbolic::Ref<'a, I, Rows, Cols> {
	type Target = symbolic::Ref<'a, I, Rows, Cols>;

	#[inline]
	fn into_const(self) -> Self::Target {
		self
	}
}

impl<'short, 'a, I, Rows: Copy, Cols: Copy> ReborrowMut<'short> for symbolic::Ref<'a, I, Rows, Cols> {
	type Target = symbolic::Ref<'short, I, Rows, Cols>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		*self
	}
}

impl<'short, 'a, I, Rows: Copy, Cols: Copy> Reborrow<'short> for symbolic::Ref<'a, I, Rows, Cols> {
	type Target = symbolic::Ref<'short, I, Rows, Cols>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		*self
	}
}

impl<'a, I, T, Rows: Copy, Cols: Copy> IntoConst for numeric::Ref<'a, I, T, Rows, Cols> {
	type Target = numeric::Ref<'a, I, T, Rows, Cols>;

	#[inline]
	fn into_const(self) -> Self::Target {
		self
	}
}

impl<'short, 'a, I, T, Rows: Copy, Cols: Copy> ReborrowMut<'short> for numeric::Ref<'a, I, T, Rows, Cols> {
	type Target = numeric::Ref<'short, I, T, Rows, Cols>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		*self
	}
}

impl<'short, 'a, I, T, Rows: Copy, Cols: Copy> Reborrow<'short> for numeric::Ref<'a, I, T, Rows, Cols> {
	type Target = numeric::Ref<'short, I, T, Rows, Cols>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		*self
	}
}

impl<'a, I, T, Rows: Copy, Cols: Copy> IntoConst for numeric::Mut<'a, I, T, Rows, Cols> {
	type Target = numeric::Ref<'a, I, T, Rows, Cols>;

	#[inline]
	fn into_const(self) -> Self::Target {
		numeric::Ref {
			symbolic: self.symbolic,
			val: self.val,
		}
	}
}

impl<'short, 'a, I, T, Rows: Copy, Cols: Copy> ReborrowMut<'short> for numeric::Mut<'a, I, T, Rows, Cols> {
	type Target = numeric::Mut<'short, I, T, Rows, Cols>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		numeric::Mut {
			symbolic: self.symbolic,
			val: self.val,
		}
	}
}

impl<'short, 'a, I, T, Rows: Copy, Cols: Copy> Reborrow<'short> for numeric::Mut<'a, I, T, Rows, Cols> {
	type Target = numeric::Ref<'short, I, T, Rows, Cols>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		numeric::Ref {
			symbolic: self.symbolic,
			val: self.val,
		}
	}
}

impl<'short, I, T, Rows: Copy, Cols: Copy> ReborrowMut<'short> for numeric::Own<I, T, Rows, Cols> {
	type Target = numeric::Mut<'short, I, T, Rows, Cols>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		numeric::Mut {
			symbolic: self.symbolic.rb(),
			val: &mut self.val,
		}
	}
}

impl<'short, I, T, Rows: Copy, Cols: Copy> Reborrow<'short> for numeric::Own<I, T, Rows, Cols> {
	type Target = numeric::Ref<'short, I, T, Rows, Cols>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		numeric::Ref {
			symbolic: self.symbolic.rb(),
			val: &self.val,
		}
	}
}

impl<'short, I, Rows: Copy, Cols: Copy> Reborrow<'short> for symbolic::Own<I, Rows, Cols> {
	type Target = symbolic::Ref<'short, I, Rows, Cols>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		symbolic::Ref {
			nrows: self.nrows,
			ncols: self.ncols,
			row_ptr: &self.row_ptr,
			row_nnz: self.row_nnz.as_deref(),
			col_idx: &self.col_idx,
		}
	}
}

#[inline(always)]
#[track_caller]
fn assume_row_ptr<I: Index>(nrows: usize, ncols: usize, row_ptr: &[I], row_nnz: Option<&[I]>, col_idx: &[I]) {
	assert!(all(ncols <= I::Signed::MAX.zx(), nrows <= I::Signed::MAX.zx(),));
	assert!(row_ptr.len() == nrows + 1);
	assert!(row_ptr[nrows].zx() <= col_idx.len());
	if let Some(row_nnz) = row_nnz {
		assert!(row_nnz.len() == nrows);
	}
}

#[track_caller]
fn check_row_ptr<I: Index>(nrows: usize, ncols: usize, row_ptr: &[I], row_nnz: Option<&[I]>, col_idx: &[I]) {
	assert!(all(ncols <= I::Signed::MAX.zx(), nrows <= I::Signed::MAX.zx(),));
	assert!(row_ptr.len() == nrows + 1);
	if let Some(row_nnz) = row_nnz {
		assert!(row_nnz.len() == nrows);
		for (&nnz_i, &[row, row_next]) in iter::zip(row_nnz, windows2(row_ptr)) {
			assert!(row <= row_next);
			assert!(nnz_i <= row_next - row);
		}
	} else {
		for &[row, row_next] in windows2(row_ptr) {
			assert!(row <= row_next);
		}
	}
	assert!(row_ptr[nrows].zx() <= col_idx.len());
}

#[track_caller]
fn check_col_idx<I: Index>(nrows: usize, ncols: usize, row_ptr: &[I], row_nnz: Option<&[I]>, col_idx: &[I]) {
	_ = nrows;
	if let Some(row_nnz) = row_nnz {
		for (nnz, &r) in iter::zip(row_nnz, row_ptr) {
			let r = r.zx();
			let nnz = nnz.zx();
			let col_idx = &col_idx[r..r + nnz];
			if !col_idx.is_empty() {
				let mut j = col_idx[0].zx();
				for &j_next in &col_idx[1..] {
					let j_next = j_next.zx();
					assert!(j < j_next);
					j = j_next;
				}
				assert!(j < ncols);
			}
		}
	} else {
		for &[r, r_next] in windows2(row_ptr) {
			let col_idx = &col_idx[r.zx()..r_next.zx()];
			if !col_idx.is_empty() {
				let mut j = col_idx[0].zx();
				for &j_next in &col_idx[1..] {
					let j_next = j_next.zx();
					assert!(j < j_next);
					j = j_next;
				}
				assert!(j < ncols);
			}
		}
	}
}

#[track_caller]
fn check_col_idx_unsorted<I: Index>(nrows: usize, ncols: usize, row_ptr: &[I], row_nnz: Option<&[I]>, col_idx: &[I]) {
	_ = nrows;

	if let Some(row_nnz) = row_nnz {
		for (&nnz, &r) in iter::zip(row_nnz, row_ptr) {
			let r = r.zx();
			let nnz = nnz.zx();
			for &j in &col_idx[r..r + nnz] {
				let j = j.zx();
				assert!(j < ncols);
			}
		}
	} else {
		for &[r, r_next] in windows2(row_ptr) {
			for &j in &col_idx[r.zx()..r_next.zx()] {
				let j = j.zx();
				assert!(j < ncols);
			}
		}
	}
}

impl<'a, Rows: Shape, Cols: Shape, I: Index> SymbolicSparseRowMatRef<'a, I, Rows, Cols> {
	/// creates a new symbolic matrix view without checking its invariants
	///
	/// # safety
	/// see type level documentation.
	#[inline]
	#[track_caller]
	pub unsafe fn new_unchecked(nrows: Rows, ncols: Cols, row_ptr: &'a [I], row_nnz: Option<&'a [I]>, col_idx: &'a [I]) -> Self {
		assume_row_ptr(nrows.unbound(), ncols.unbound(), row_ptr, row_nnz, col_idx);

		Self {
			0: symbolic::Ref {
				nrows,
				ncols,
				row_ptr,
				row_nnz,
				col_idx,
			},
		}
	}

	/// creates a new symbolic matrix view after checking its invariants
	///
	/// # safety
	/// see type level documentation.
	#[inline]
	#[track_caller]
	pub fn new_checked(nrows: Rows, ncols: Cols, row_ptr: &'a [I], row_nnz: Option<&'a [I]>, col_idx: &'a [I]) -> Self {
		check_row_ptr(nrows.unbound(), ncols.unbound(), row_ptr, row_nnz, col_idx);
		check_col_idx(nrows.unbound(), ncols.unbound(), row_ptr, row_nnz, col_idx);

		Self {
			0: symbolic::Ref {
				nrows,
				ncols,
				row_ptr,
				row_nnz,
				col_idx,
			},
		}
	}

	/// creates a new symbolic matrix view after checking its invariants (excluding soft invariants)
	///
	/// # safety
	/// see type level documentation.
	#[inline]
	#[track_caller]
	pub fn new_unsorted_checked(nrows: Rows, ncols: Cols, row_ptr: &'a [I], row_nnz: Option<&'a [I]>, col_idx: &'a [I]) -> Self {
		check_row_ptr(nrows.unbound(), ncols.unbound(), row_ptr, row_nnz, col_idx);
		check_col_idx_unsorted(nrows.unbound(), ncols.unbound(), row_ptr, row_nnz, col_idx);

		Self {
			0: symbolic::Ref {
				nrows,
				ncols,
				row_ptr,
				row_nnz,
				col_idx,
			},
		}
	}

	/// returns the components of the sparse matrix
	/// - number of rows
	/// - number of columns
	/// - row pointers
	/// - row non-zero counts
	/// - column indices
	#[inline]
	pub fn parts(self) -> (Rows, Cols, &'a [I], Option<&'a [I]>, &'a [I]) {
		(self.nrows, self.ncols, self.0.row_ptr, self.0.row_nnz, self.0.col_idx)
	}

	/// returns the number of rows of the matrix
	#[inline]
	pub fn nrows(&self) -> Rows {
		self.nrows
	}

	/// returns the number of columns of the matrix
	#[inline]
	pub fn ncols(&self) -> Cols {
		self.ncols
	}

	/// returns the number of rows and columns of the matrix
	#[inline]
	pub fn shape(&self) -> (Rows, Cols) {
		(self.nrows, self.ncols)
	}

	/// returns a view over the transpose of `self`
	#[inline]
	pub fn transpose(self) -> SymbolicSparseColMatRef<'a, I, Cols, Rows> {
		SymbolicSparseColMatRef {
			0: super::csc::symbolic::Ref {
				nrows: self.ncols,
				ncols: self.nrows,
				col_ptr: self.0.row_ptr,
				col_nnz: self.0.row_nnz,
				row_idx: self.0.col_idx,
			},
		}
	}

	/// returns a newly allocated matrix holding the values of `self`
	#[inline]
	pub fn to_owned(&self) -> Result<SymbolicSparseRowMat<I, Rows, Cols>, FaerError> {
		Ok(self.transpose().to_owned()?.into_transpose())
	}

	/// returns a newly allocated matrix holding the values of `self` in column major format
	#[inline]
	pub fn to_col_major(&self) -> Result<SymbolicSparseColMat<I, Rows, Cols>, FaerError> {
		Ok(self.transpose().to_row_major()?.into_transpose())
	}

	/// returns the number of non-zero elements in the matrix
	#[inline]
	pub fn compute_nnz(&self) -> usize {
		self.transpose().compute_nnz()
	}

	/// returns the row pointers
	#[inline]
	pub fn row_ptr(&self) -> &'a [I] {
		self.0.row_ptr
	}

	/// returns the row non-zero counts
	#[inline]
	pub fn row_nnz(&self) -> Option<&'a [I]> {
		self.0.row_nnz
	}

	/// returns the column indices
	#[inline]
	pub fn col_idx(&self) -> &'a [I] {
		self.0.col_idx
	}

	/// returns the range specifying the indices of row `i`
	#[inline]
	#[track_caller]
	pub fn row_range(&self, i: Idx<Rows>) -> Range<usize> {
		assert!(i < self.nrows());
		unsafe { self.row_range_unchecked(i) }
	}

	/// returns the range specifying the indices of row `i`, without bound checks
	#[inline]
	#[track_caller]
	pub unsafe fn row_range_unchecked(&self, i: Idx<Rows>) -> Range<usize> {
		debug_assert!(i < self.nrows());

		self.transpose().col_range_unchecked(i)
	}

	/// returns the column indices of row `i`
	#[inline]
	#[track_caller]
	pub fn col_idx_of_row_raw(&self, i: Idx<Rows>) -> &'a [Idx<Cols, I>] {
		unsafe {
			let slice = self.0.col_idx.get_unchecked(self.row_range(i));
			let len = slice.len();
			core::slice::from_raw_parts(slice.as_ptr() as *const Idx<Cols, I>, len)
		}
	}

	/// returns the column indices of row `i`
	#[inline]
	#[track_caller]
	pub fn col_idx_of_row(&self, i: Idx<Rows>) -> impl 'a + Clone + ExactSizeIterator + DoubleEndedIterator<Item = Idx<Cols>>
	where
		Rows: 'a,
		Cols: 'a,
	{
		self.col_idx_of_row_raw(i)
			.iter()
			.map(|&j| unsafe { Idx::<Cols>::new_unbound(j.unbound().zx()) })
	}

	/// returns the input matrix with the given shape after checking that it matches the
	/// current shape
	#[inline]
	#[track_caller]
	pub fn as_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SymbolicSparseRowMatRef<'a, I, V, H> {
		assert!(all(self.nrows.unbound() == nrows.unbound(), self.ncols.unbound() == ncols.unbound()));

		SymbolicSparseRowMatRef {
			0: symbolic::Ref {
				nrows,
				ncols,
				row_ptr: self.0.row_ptr,
				row_nnz: self.0.row_nnz,
				col_idx: self.0.col_idx,
			},
		}
	}

	/// returns the input matrix with dynamic shape
	#[inline]
	pub fn as_dyn(self) -> SymbolicSparseRowMatRef<'a, I> {
		SymbolicSparseRowMatRef {
			0: symbolic::Ref {
				nrows: self.nrows.unbound(),
				ncols: self.ncols.unbound(),
				row_ptr: self.0.row_ptr,
				row_nnz: self.0.row_nnz,
				col_idx: self.0.col_idx,
			},
		}
	}

	/// Returns a view over the symbolic structure of `self`.
	#[inline]
	pub fn as_ref(self) -> SymbolicSparseRowMatRef<'a, I, Rows, Cols> {
		SymbolicSparseRowMatRef {
			0: symbolic::Ref {
				nrows: self.nrows,
				ncols: self.ncols,
				row_ptr: self.0.row_ptr,
				row_nnz: self.0.row_nnz,
				col_idx: self.0.col_idx,
			},
		}
	}
}

impl<Rows: Shape, Cols: Shape, I: Index> SymbolicSparseRowMat<I, Rows, Cols> {
	#[inline]
	#[track_caller]
	/// see [`SymbolicSparseRowMatRef::new_unchecked`]
	pub unsafe fn new_unchecked(
		nrows: Rows,
		ncols: Cols,
		row_ptr: alloc::vec::Vec<I>,
		row_nnz: Option<alloc::vec::Vec<I>>,
		col_idx: alloc::vec::Vec<I>,
	) -> Self {
		assume_row_ptr(nrows.unbound(), ncols.unbound(), &row_ptr, row_nnz.as_deref(), &col_idx);

		Self {
			0: symbolic::Own {
				nrows,
				ncols,
				row_ptr,
				row_nnz,
				col_idx,
			},
		}
	}

	#[inline]
	#[track_caller]
	/// see [`SymbolicSparseRowMatRef::new_checked`]
	pub fn new_checked(
		nrows: Rows,
		ncols: Cols,
		row_ptr: alloc::vec::Vec<I>,
		row_nnz: Option<alloc::vec::Vec<I>>,
		col_idx: alloc::vec::Vec<I>,
	) -> Self {
		check_row_ptr(nrows.unbound(), ncols.unbound(), &row_ptr, row_nnz.as_deref(), &col_idx);
		check_col_idx(nrows.unbound(), ncols.unbound(), &row_ptr, row_nnz.as_deref(), &col_idx);

		Self {
			0: symbolic::Own {
				nrows,
				ncols,
				row_ptr,
				row_nnz,
				col_idx,
			},
		}
	}

	#[inline]
	#[track_caller]
	/// see [`SymbolicSparseRowMatRef::new_unsorted_checked`]
	pub fn new_unsorted_checked(
		nrows: Rows,
		ncols: Cols,
		row_ptr: alloc::vec::Vec<I>,
		row_nnz: Option<alloc::vec::Vec<I>>,
		col_idx: alloc::vec::Vec<I>,
	) -> Self {
		check_row_ptr(nrows.unbound(), ncols.unbound(), &row_ptr, row_nnz.as_deref(), &col_idx);
		check_col_idx_unsorted(nrows.unbound(), ncols.unbound(), &row_ptr, row_nnz.as_deref(), &col_idx);

		Self {
			0: symbolic::Own {
				nrows,
				ncols,
				row_ptr,
				row_nnz,
				col_idx,
			},
		}
	}

	#[inline]
	/// see [`SymbolicSparseRowMatRef::parts`]
	pub fn parts(&self) -> (Rows, Cols, &'_ [I], Option<&'_ [I]>, &'_ [I]) {
		(self.nrows, self.ncols, &self.0.row_ptr, self.0.row_nnz.as_deref(), &self.0.col_idx)
	}

	#[inline]
	/// see [`SymbolicSparseRowMatRef::parts`]
	pub fn into_parts(self) -> (Rows, Cols, alloc::vec::Vec<I>, Option<alloc::vec::Vec<I>>, alloc::vec::Vec<I>) {
		(self.nrows, self.ncols, self.0.row_ptr, self.0.row_nnz, self.0.col_idx)
	}

	#[inline]
	/// see [`SymbolicSparseRowMatRef::nrows`]
	pub fn nrows(&self) -> Rows {
		self.nrows
	}

	#[inline]
	/// see [`SymbolicSparseRowMatRef::ncols`]
	pub fn ncols(&self) -> Cols {
		self.ncols
	}

	#[inline]
	/// see [`SymbolicSparseRowMatRef::as_shape`]
	pub fn shape(&self) -> (Rows, Cols) {
		(self.nrows, self.ncols)
	}

	#[inline]
	/// see [`SymbolicSparseRowMatRef::transpose`]
	pub fn transpose(&self) -> SymbolicSparseColMatRef<'_, I, Cols, Rows> {
		self.rb().transpose()
	}

	#[inline]
	/// see [`SymbolicSparseRowMatRef::transpose`]
	pub fn into_transpose(self) -> SymbolicSparseColMat<I, Cols, Rows> {
		SymbolicSparseColMat {
			0: super::csc::symbolic::Own {
				nrows: self.ncols,
				ncols: self.nrows,
				col_ptr: self.0.row_ptr,
				col_nnz: self.0.row_nnz,
				row_idx: self.0.col_idx,
			},
		}
	}

	#[inline]
	/// see [`SymbolicSparseRowMatRef::to_owned`]
	pub fn to_owned(&self) -> Result<SymbolicSparseRowMat<I, Rows, Cols>, FaerError> {
		self.rb().to_owned()
	}

	#[inline]
	/// see [`SymbolicSparseRowMatRef::to_col_major`]
	pub fn to_col_major(&self) -> Result<SymbolicSparseColMat<I, Rows, Cols>, FaerError> {
		self.rb().to_col_major()
	}

	#[inline]
	/// see [`SymbolicSparseRowMatRef::compute_nnz`]
	pub fn compute_nnz(&self) -> usize {
		self.rb().compute_nnz()
	}

	#[inline]
	/// see [`SymbolicSparseRowMatRef::row_ptr`]
	pub fn row_ptr(&self) -> &'_ [I] {
		&self.0.row_ptr
	}

	#[inline]
	/// see [`SymbolicSparseRowMatRef::row_nnz`]
	pub fn row_nnz(&self) -> Option<&'_ [I]> {
		self.0.row_nnz.as_deref()
	}

	#[inline]
	/// see [`SymbolicSparseRowMatRef::col_idx`]
	pub fn col_idx(&self) -> &'_ [I] {
		&self.0.col_idx
	}

	#[inline]
	#[track_caller]
	/// see [`SymbolicSparseRowMatRef::row_range`]
	pub fn row_range(&self, i: Idx<Rows>) -> Range<usize> {
		self.rb().row_range(i)
	}

	#[inline]
	#[track_caller]
	/// see [`SymbolicSparseRowMatRef::row_range_unchecked`]
	pub unsafe fn row_range_unchecked(&self, i: Idx<Rows>) -> Range<usize> {
		self.rb().row_range_unchecked(i)
	}

	#[inline]
	#[track_caller]
	/// see [`SymbolicSparseRowMatRef::col_idx_of_row_raw`]
	pub fn col_idx_of_row_raw(&self, i: Idx<Rows>) -> &'_ [Idx<Cols, I>] {
		self.rb().col_idx_of_row_raw(i)
	}

	#[inline]
	#[track_caller]
	/// see [`SymbolicSparseRowMatRef::col_idx_of_row`]
	pub fn col_idx_of_row(&self, i: Idx<Rows>) -> impl '_ + Clone + ExactSizeIterator + DoubleEndedIterator<Item = Idx<Cols>> {
		self.rb().col_idx_of_row(i)
	}

	#[inline]
	#[track_caller]
	/// see [`SymbolicSparseRowMatRef::as_shape`]
	pub fn as_shape<V: Shape, H: Shape>(&self, nrows: V, ncols: H) -> SymbolicSparseRowMatRef<'_, I, V, H> {
		self.rb().as_shape(nrows, ncols)
	}

	#[inline]
	#[track_caller]
	/// see [`SymbolicSparseRowMatRef::as_shape`]
	pub fn into_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SymbolicSparseRowMat<I, V, H> {
		assert!(all(self.nrows().unbound() == nrows.unbound(), self.ncols().unbound() == ncols.unbound()));
		SymbolicSparseRowMat {
			0: symbolic::Own {
				nrows,
				ncols,
				row_ptr: self.0.row_ptr,
				row_nnz: self.0.row_nnz,
				col_idx: self.0.col_idx,
			},
		}
	}

	#[inline]
	/// see [`SymbolicSparseRowMatRef::as_dyn`]
	pub fn as_dyn(&self) -> SymbolicSparseRowMatRef<'_, I> {
		self.rb().as_dyn()
	}

	#[inline]
	/// see [`SymbolicSparseRowMatRef::as_dyn`]
	pub fn into_dyn(self) -> SymbolicSparseRowMat<I> {
		SymbolicSparseRowMat {
			0: symbolic::Own {
				nrows: self.nrows.unbound(),
				ncols: self.ncols.unbound(),
				row_ptr: self.0.row_ptr,
				row_nnz: self.0.row_nnz,
				col_idx: self.0.col_idx,
			},
		}
	}

	#[inline]
	/// Returns a view over the symbolic structure of `self`.
	pub fn as_ref(&self) -> SymbolicSparseRowMatRef<'_, I, Rows, Cols> {
		SymbolicSparseRowMatRef {
			0: symbolic::Ref {
				nrows: self.nrows,
				ncols: self.ncols,
				row_ptr: &self.0.row_ptr,
				row_nnz: self.0.row_nnz.as_deref(),
				col_idx: &self.0.col_idx,
			},
		}
	}

	#[inline]
	/// create a new symbolic structure, and the corresponding order for the numerical values
	/// from pairs of indices
	pub fn try_new_from_indices(nrows: Rows, ncols: Cols, idx: &[Pair<Idx<Rows, I>, Idx<Cols, I>>]) -> Result<(Self, Argsort<I>), CreationError> {
		let (symbolic, argsort) = SymbolicSparseColMat::try_new_from_indices_impl(
			ncols,
			nrows,
			|i| Pair {
				row: idx[i].col,
				col: idx[i].row,
			},
			|_, _| false,
			idx.len(),
		)?;

		Ok((symbolic.into_transpose(), argsort))
	}

	#[inline]
	/// create a new symbolic structure, and the corresponding order for the numerical values
	/// from pairs of indices
	///
	/// negative indices are ignored
	pub fn try_new_from_nonnegative_indices(
		nrows: Rows,
		ncols: Cols,
		idx: &[Pair<MaybeIdx<Rows, I>, MaybeIdx<Cols, I>>],
	) -> Result<(Self, Argsort<I>), CreationError> {
		let (symbolic, argsort) = SymbolicSparseColMat::try_new_from_indices_impl(
			ncols,
			nrows,
			|i| Pair {
				row: unsafe { Idx::<Cols, I>::new_unbound(I::from_signed(idx[i].col.unbound())) },
				col: unsafe { Idx::<Rows, I>::new_unbound(I::from_signed(idx[i].row.unbound())) },
			},
			|row, col| {
				let row = row.unbound().to_signed();
				let col = col.unbound().to_signed();
				let zero = I::Signed::truncate(0);

				row < zero || col < zero
			},
			idx.len(),
		)?;
		Ok((symbolic.into_transpose(), argsort))
	}
}

impl<'a, Rows: Shape, Cols: Shape, I: Index, T> SparseRowMatRef<'a, I, T, Rows, Cols> {
	/// creates a new sparse matrix view.
	///
	/// # panics
	/// panics if the length of `values` is not equal to the length of
	/// `symbolic.col_idx()`
	#[inline]
	#[track_caller]
	pub fn new(symbolic: SymbolicSparseRowMatRef<'a, I, Rows, Cols>, val: &'a [T]) -> Self {
		assert!(symbolic.col_idx().len() == val.len());
		Self {
			0: numeric::Ref { symbolic, val },
		}
	}

	/// returns the symbolic and numeric components of the sparse matrix
	#[inline]
	pub fn parts(self) -> (SymbolicSparseRowMatRef<'a, I, Rows, Cols>, &'a [T]) {
		(self.0.symbolic, self.0.val)
	}

	/// returns the symbolic component of the sparse matrix
	#[inline]
	pub fn symbolic(&self) -> SymbolicSparseRowMatRef<'a, I, Rows, Cols> {
		self.0.symbolic
	}

	/// returns the numeric component of the sparse matrix
	#[inline]
	pub fn val(self) -> &'a [T] {
		self.0.val
	}

	/// returns a view over the values of row `i`
	#[inline]
	#[track_caller]
	pub fn val_of_row(self, i: Idx<Rows>) -> &'a [T] {
		unsafe { self.0.val.get_unchecked(self.row_range(i)) }
	}

	/// returns the input matrix with the given shape after checking that it matches the
	/// current shape
	#[inline]
	#[track_caller]
	pub fn as_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SparseRowMatRef<'a, I, T, V, H> {
		SparseRowMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic.as_shape(nrows, ncols),
				val: self.0.val,
			},
		}
	}

	/// returns a reference to the value at the given index, or `None` if the symbolic structure
	/// doesn't contain it, or contains multiple indices with the given index
	///
	/// # panics
	/// - panics if `row >= self.nrows()`
	/// - panics if `col >= self.ncols()`
	#[track_caller]
	pub fn get(self, row: Idx<Rows>, col: Idx<Cols>) -> Option<&'a T> {
		assert!(row < self.nrows());
		assert!(col < self.ncols());
		let col = I::truncate(col.unbound());
		let rowl = row.unbound();
		let start = self
			.symbolic()
			.as_dyn()
			.col_idx_of_row_raw(rowl)
			.partition_point(super::csc::partition_by_lt(col));
		let end = start + self.symbolic().as_dyn().col_idx_of_row_raw(rowl)[start..].partition_point(super::csc::partition_by_le(col));

		if end == start + 1 { Some(&self.val_of_row(row)[start]) } else { None }
	}

	/// returns the input matrix with dynamic shape
	#[inline]
	pub fn as_dyn(self) -> SparseRowMatRef<'a, I, T> {
		SparseRowMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic.as_dyn(),
				val: self.0.val,
			},
		}
	}

	/// returns a view over `self`
	#[inline]
	pub fn as_ref(self) -> SparseRowMatRef<'a, I, T, Rows, Cols> {
		SparseRowMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic,
				val: self.0.val,
			},
		}
	}

	/// returns a view over the transpose of `self`
	#[inline]
	pub fn transpose(self) -> SparseColMatRef<'a, I, T, Cols, Rows> {
		SparseColMatRef {
			0: super::csc::numeric::Ref {
				symbolic: self.0.symbolic.transpose(),
				val: self.0.val,
			},
		}
	}

	/// returns a view over the conjugate of `self`
	#[inline]
	pub fn conjugate(self) -> SparseRowMatRef<'a, I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		self.transpose().conjugate().transpose()
	}

	/// returns a view over the adjoint of `self`
	#[inline]
	pub fn adjoint(self) -> SparseColMatRef<'a, I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.transpose().conjugate()
	}

	/// returns an unconjugated view over `self`
	#[inline]
	pub fn canonical(self) -> SparseRowMatRef<'a, I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		self.transpose().canonical().transpose()
	}

	/// returns a newly allocated matrix holding the (possibly conjugated) values of `self` in
	/// column major format
	#[inline]
	pub fn to_col_major(&self) -> Result<SparseColMat<I, T::Canonical, Rows, Cols>, FaerError>
	where
		T: Conjugate,
	{
		Ok(self.transpose().to_row_major()?.into_transpose())
	}

	/// returns a newly allocated dense matrix holding the (possibly conjugated) values of `self`
	#[inline]
	pub fn to_dense(&self) -> Mat<T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		fn imp<'ROWS, 'COLS, I: Index, T: Conjugate>(
			src: SparseRowMatRef<'_, I, T, Dim<'ROWS>, Dim<'COLS>>,
		) -> Mat<T::Canonical, Dim<'ROWS>, Dim<'COLS>> {
			let src = src.canonical();

			let mut out = Mat::zeros(src.nrows(), src.ncols());
			let M = src.nrows();

			for i in M.indices() {
				for (j, val) in iter::zip(src.col_idx_of_row(i), src.val_of_row(i)) {
					if try_const! { Conj::get::<T>().is_conj() } {
						out[(i, j)] = add(&out[(i, j)], &conj(val));
					} else {
						out[(i, j)] = add(&out[(i, j)], val);
					}
				}
			}

			out
		}
		with_dim!(ROWS, self.nrows().unbound());
		with_dim!(COLS, self.ncols().unbound());
		let this = self.as_shape(ROWS, COLS);

		imp(this).into_shape(self.nrows(), self.ncols())
	}

	/// returns an iterator over the entries of the matrix.
	pub fn triplet_iter(self) -> impl 'a + Iterator<Item = Triplet<Idx<Rows>, Idx<Cols>, &'a T>>
	where
		Rows: 'a,
		Cols: 'a,
	{
		self.transpose().triplet_iter().map(
			#[inline(always)]
			|Triplet { row, col, val }| Triplet { row: col, col: row, val },
		)
	}
}

impl<'a, Rows: Shape, Cols: Shape, I: Index, T> SparseRowMatMut<'a, I, T, Rows, Cols> {
	#[inline]
	#[track_caller]
	/// see [`SparseRowMatRef::new`]
	pub fn new(symbolic: SymbolicSparseRowMatRef<'a, I, Rows, Cols>, val: &'a mut [T]) -> Self {
		assert!(symbolic.col_idx().len() == val.len());
		Self {
			0: numeric::Mut { symbolic, val },
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::parts`]
	pub fn parts(self) -> (SymbolicSparseRowMatRef<'a, I, Rows, Cols>, &'a [T]) {
		(self.0.symbolic, self.0.val)
	}

	#[inline]
	/// see [`SparseRowMatRef::parts`]
	pub fn parts_mut(self) -> (SymbolicSparseRowMatRef<'a, I, Rows, Cols>, &'a mut [T]) {
		(self.0.symbolic, self.0.val)
	}

	#[inline]
	/// see [`SparseRowMatRef::symbolic`]
	pub fn symbolic(&self) -> SymbolicSparseRowMatRef<'a, I, Rows, Cols> {
		self.0.symbolic
	}

	#[inline]
	/// see [`SparseRowMatRef::val`]
	pub fn val(self) -> &'a [T] {
		self.0.val
	}

	#[inline]
	/// see [`SparseRowMatRef::val`]
	pub fn val_mut(self) -> &'a mut [T] {
		self.0.val
	}

	#[inline]
	#[track_caller]
	/// see [`SparseRowMatRef::val_of_row`]
	pub fn val_of_row(self, i: Idx<Rows>) -> &'a [T] {
		unsafe { self.0.val.get_unchecked(self.row_range(i)) }
	}

	#[inline]
	#[track_caller]
	/// see [`SparseRowMatRef::val_of_row`]
	pub fn val_of_row_mut(self, j: Idx<Rows>) -> &'a mut [T] {
		unsafe { self.0.val.get_unchecked_mut(self.row_range(j)) }
	}

	#[inline]
	#[track_caller]
	/// see [`SparseRowMatRef::as_shape`]
	pub fn as_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SparseRowMatRef<'a, I, T, V, H> {
		SparseRowMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic.as_shape(nrows, ncols),
				val: self.0.val,
			},
		}
	}

	#[inline]
	#[track_caller]
	/// see [`SparseRowMatRef::as_shape`]
	pub fn as_shape_mut<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SparseRowMatMut<'a, I, T, V, H> {
		SparseRowMatMut {
			0: numeric::Mut {
				symbolic: self.0.symbolic.as_shape(nrows, ncols),
				val: self.0.val,
			},
		}
	}

	/// see [`SparseRowMatRef::get`]
	#[track_caller]
	#[inline]
	pub fn get(self, row: Idx<Rows>, col: Idx<Cols>) -> Option<&'a T> {
		self.into_const().get(row, col)
	}

	/// returns a reference to the value at the given index, or `None` if the symbolic structure
	/// doesn't contain it, or contains multiple indices with the given index
	///
	/// # panics
	/// - panics if `row >= self.nrows()`
	/// - panics if `col >= self.ncols()`
	#[track_caller]
	/// see [`SparseRowMatRef::get`]
	pub fn get_mut(self, row: Idx<Rows>, col: Idx<Cols>) -> Option<&'a mut T> {
		assert!(row < self.nrows());
		assert!(col < self.ncols());
		let col = I::truncate(col.unbound());
		let rowl = row.unbound();
		let start = self
			.symbolic()
			.as_dyn()
			.col_idx_of_row_raw(rowl)
			.partition_point(super::csc::partition_by_lt(col));
		let end = start + self.symbolic().as_dyn().col_idx_of_row_raw(rowl)[start..].partition_point(super::csc::partition_by_le(col));

		if end == start + 1 {
			Some(&mut self.val_of_row_mut(row)[start])
		} else {
			None
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::as_dyn`]
	pub fn as_dyn(self) -> SparseRowMatRef<'a, I, T> {
		SparseRowMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic.as_dyn(),
				val: self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::as_dyn`]
	pub fn as_dyn_mut(self) -> SparseRowMatMut<'a, I, T> {
		SparseRowMatMut {
			0: numeric::Mut {
				symbolic: self.0.symbolic.as_dyn(),
				val: self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::transpose`]
	pub fn transpose(self) -> SparseColMatRef<'a, I, T, Cols, Rows> {
		SparseColMatRef {
			0: super::csc::numeric::Ref {
				symbolic: self.0.symbolic.transpose(),
				val: self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::transpose`]
	pub fn transpose_mut(self) -> SparseColMatMut<'a, I, T, Cols, Rows> {
		SparseColMatMut {
			0: super::csc::numeric::Mut {
				symbolic: self.0.symbolic.transpose(),
				val: self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::conjugate`]
	pub fn conjugate(self) -> SparseRowMatRef<'a, I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.0.val.len();
		SparseRowMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic,
				val: unsafe { core::slice::from_raw_parts(self.0.val.as_ptr() as *const T::Conj, len) },
			},
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::conjugate`]
	pub fn conjugate_mut(self) -> SparseRowMatMut<'a, I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.0.val.len();
		SparseRowMatMut {
			0: numeric::Mut {
				symbolic: self.0.symbolic,
				val: unsafe { core::slice::from_raw_parts_mut(self.0.val.as_mut_ptr() as *mut T::Conj, len) },
			},
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::adjoint`]
	pub fn adjoint(self) -> SparseColMatRef<'a, I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.conjugate().transpose()
	}

	#[inline]
	/// see [`SparseRowMatRef::adjoint`]
	pub fn adjoint_mut(self) -> SparseColMatMut<'a, I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.conjugate_mut().transpose_mut()
	}

	#[inline]
	/// see [`SparseRowMatRef::canonical`]
	pub fn canonical(self) -> SparseRowMatRef<'a, I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.0.val.len();
		SparseRowMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic,
				val: unsafe { core::slice::from_raw_parts(self.0.val.as_ptr() as *const T::Canonical, len) },
			},
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::canonical`]
	pub fn canonical_mut(self) -> SparseRowMatMut<'a, I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.0.val.len();
		SparseRowMatMut {
			0: numeric::Mut {
				symbolic: self.0.symbolic,
				val: unsafe { core::slice::from_raw_parts_mut(self.0.val.as_mut_ptr() as *mut T::Canonical, len) },
			},
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::to_col_major`]
	pub fn to_col_major(&self) -> Result<SparseColMat<I, T::Canonical, Rows, Cols>, FaerError>
	where
		T: Conjugate,
	{
		self.rb().to_col_major()
	}

	#[inline]
	/// see [`SparseRowMatRef::to_dense`]
	pub fn to_dense(&self) -> Mat<T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		self.rb().to_dense()
	}

	/// see [`SparseRowMatRef::triplet_iter`]
	#[inline]
	pub fn triplet_iter(self) -> impl 'a + Iterator<Item = Triplet<Idx<Rows>, Idx<Cols>, &'a T>>
	where
		Rows: 'a,
		Cols: 'a,
	{
		self.into_const().triplet_iter()
	}
}

impl<Rows: Shape, Cols: Shape, I: Index, T> SparseRowMat<I, T, Rows, Cols> {
	#[inline]
	#[track_caller]
	/// see [`SparseRowMatRef::new`]
	pub fn new(symbolic: SymbolicSparseRowMat<I, Rows, Cols>, val: alloc::vec::Vec<T>) -> Self {
		assert!(symbolic.col_idx().len() == val.len());
		Self {
			0: numeric::Own { symbolic, val },
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::parts`]
	pub fn parts(&self) -> (SymbolicSparseRowMatRef<'_, I, Rows, Cols>, &'_ [T]) {
		(self.0.symbolic.rb(), &self.0.val)
	}

	#[inline]
	/// see [`SparseRowMatRef::parts`]
	pub fn parts_mut(&mut self) -> (SymbolicSparseRowMatRef<'_, I, Rows, Cols>, &'_ mut [T]) {
		(self.0.symbolic.rb(), &mut self.0.val)
	}

	#[inline]
	/// see [`SparseRowMatRef::parts`]
	pub fn into_parts(self) -> (SymbolicSparseRowMat<I, Rows, Cols>, alloc::vec::Vec<T>) {
		(self.0.symbolic, self.0.val)
	}

	#[inline]
	/// see [`SparseRowMatRef::symbolic`]
	pub fn symbolic(&self) -> SymbolicSparseRowMatRef<'_, I, Rows, Cols> {
		self.0.symbolic.rb()
	}

	#[inline]
	/// see [`SparseRowMatRef::val`]
	pub fn val(&self) -> &'_ [T] {
		&self.0.val
	}

	#[inline]
	/// see [`SparseRowMatRef::val`]
	pub fn val_mut(&mut self) -> &'_ mut [T] {
		&mut self.0.val
	}

	#[inline]
	#[track_caller]
	/// see [`SparseRowMatRef::val_of_row`]
	pub fn val_of_row(&self, j: Idx<Rows>) -> &'_ [T] {
		unsafe { self.0.val.get_unchecked(self.row_range(j)) }
	}

	#[inline]
	#[track_caller]
	/// see [`SparseRowMatRef::val_of_row`]
	pub fn val_of_row_mut(&mut self, j: Idx<Rows>) -> &'_ mut [T] {
		unsafe { self.0.val.get_unchecked_mut(self.0.symbolic.row_range(j)) }
	}

	#[inline]
	#[track_caller]
	/// see [`SparseRowMatRef::as_shape`]
	pub fn as_shape<V: Shape, H: Shape>(&self, nrows: V, ncols: H) -> SparseRowMatRef<'_, I, T, V, H> {
		SparseRowMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic.as_shape(nrows, ncols),
				val: &self.0.val,
			},
		}
	}

	#[inline]
	#[track_caller]
	/// see [`SparseRowMatRef::as_shape`]
	pub fn as_shape_mut<V: Shape, H: Shape>(&mut self, nrows: V, ncols: H) -> SparseRowMatMut<'_, I, T, V, H> {
		SparseRowMatMut {
			0: numeric::Mut {
				symbolic: self.0.symbolic.as_shape(nrows, ncols),
				val: &mut self.0.val,
			},
		}
	}

	#[inline]
	#[track_caller]
	/// see [`SparseRowMatRef::as_shape`]
	pub fn into_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SparseRowMat<I, T, V, H> {
		SparseRowMat {
			0: numeric::Own {
				symbolic: self.0.symbolic.into_shape(nrows, ncols),
				val: self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::as_dyn`]
	pub fn as_dyn(&self) -> SparseRowMatRef<'_, I, T> {
		SparseRowMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic.as_dyn(),
				val: &self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::as_dyn`]
	pub fn as_dyn_mut(&mut self) -> SparseRowMatMut<'_, I, T> {
		SparseRowMatMut {
			0: numeric::Mut {
				symbolic: self.0.symbolic.as_dyn(),
				val: &mut self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::as_dyn`]
	pub fn into_dyn(self) -> SparseRowMat<I, T> {
		SparseRowMat {
			0: numeric::Own {
				symbolic: self.0.symbolic.into_dyn(),
				val: self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::as_ref`]
	pub fn as_ref(&self) -> SparseRowMatRef<'_, I, T, Rows, Cols> {
		SparseRowMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic.as_ref(),
				val: &self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::transpose`]
	pub fn transpose(&self) -> SparseColMatRef<'_, I, T, Cols, Rows> {
		SparseColMatRef {
			0: super::csc::numeric::Ref {
				symbolic: self.0.symbolic.transpose(),
				val: &self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::transpose`]
	pub fn transpose_mut(&mut self) -> SparseColMatMut<'_, I, T, Cols, Rows> {
		SparseColMatMut {
			0: super::csc::numeric::Mut {
				symbolic: self.0.symbolic.transpose(),
				val: &mut self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::transpose`]
	pub fn into_transpose(self) -> SparseColMat<I, T, Cols, Rows> {
		SparseColMat {
			0: super::csc::numeric::Own {
				symbolic: self.0.symbolic.into_transpose(),
				val: self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::conjugate`]
	pub fn conjugate(&self) -> SparseRowMatRef<'_, I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		self.rb().conjugate()
	}

	#[inline]
	/// see [`SparseRowMatRef::conjugate`]
	pub fn conjugate_mut(&mut self) -> SparseRowMatMut<'_, I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		self.rb_mut().conjugate_mut()
	}

	#[inline]
	/// see [`SparseRowMatRef::conjugate`]
	pub fn into_conjugate(self) -> SparseRowMat<I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		let mut vec = core::mem::ManuallyDrop::new(self.0.val);
		let len = vec.len();
		let cap = vec.capacity();
		let ptr = vec.as_mut_ptr();

		SparseRowMat {
			0: numeric::Own {
				symbolic: self.0.symbolic,
				val: unsafe { alloc::vec::Vec::from_raw_parts(ptr as *mut T::Conj, len, cap) },
			},
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::adjoint`]
	pub fn adjoint(&self) -> SparseColMatRef<'_, I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.conjugate().transpose()
	}

	#[inline]
	/// see [`SparseRowMatRef::adjoint`]
	pub fn adjoint_mut(&mut self) -> SparseColMatMut<'_, I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.conjugate_mut().transpose_mut()
	}

	#[inline]
	/// see [`SparseRowMatRef::adjoint`]
	pub fn into_adjoint(self) -> SparseColMat<I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.into_conjugate().into_transpose()
	}

	#[inline]
	/// see [`SparseRowMatRef::canonical`]
	pub fn canonical(&self) -> SparseRowMatRef<'_, I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.0.val.len();
		SparseRowMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic.rb(),
				val: unsafe { core::slice::from_raw_parts(self.0.val.as_ptr() as *const T::Canonical, len) },
			},
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::canonical`]
	pub fn canonical_mut(&mut self) -> SparseRowMatMut<'_, I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.0.val.len();
		SparseRowMatMut {
			0: numeric::Mut {
				symbolic: self.0.symbolic.rb(),
				val: unsafe { core::slice::from_raw_parts_mut(self.0.val.as_mut_ptr() as *mut T::Canonical, len) },
			},
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::canonical`]
	pub fn into_canonical(self) -> SparseRowMat<I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let mut vec = core::mem::ManuallyDrop::new(self.0.val);
		let len = vec.len();
		let cap = vec.capacity();
		let ptr = vec.as_mut_ptr();

		SparseRowMat {
			0: numeric::Own {
				symbolic: self.0.symbolic,
				val: unsafe { alloc::vec::Vec::from_raw_parts(ptr as *mut T::Canonical, len, cap) },
			},
		}
	}

	#[inline]
	/// see [`SparseRowMatRef::to_col_major`]
	pub fn to_col_major(&self) -> Result<SparseColMat<I, T::Canonical, Rows, Cols>, FaerError>
	where
		T: Conjugate,
	{
		self.rb().to_col_major()
	}

	#[inline]
	/// see [`SparseRowMatRef::to_dense`]
	pub fn to_dense(&self) -> Mat<T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		self.rb().to_dense()
	}

	#[track_caller]
	/// create a new matrix from a previously created symbolic structure and value order
	///
	/// the provided values must correspond to the same indices that were provided in the
	/// function call from which the order was created
	pub fn new_from_argsort(symbolic: SymbolicSparseRowMat<I, Rows, Cols>, argsort: &Argsort<I>, val: &[T]) -> Result<Self, FaerError>
	where
		T: ComplexField,
	{
		Ok(SparseColMat::new_from_argsort(symbolic.into_transpose(), argsort, val)?.into_transpose())
	}

	#[track_caller]
	/// create a new matrix from triplets
	pub fn try_new_from_triplets(nrows: Rows, ncols: Cols, entries: &[Triplet<Idx<Rows, I>, Idx<Cols, I>, T>]) -> Result<Self, CreationError>
	where
		T: ComplexField,
	{
		let (symbolic, argsort) = SymbolicSparseColMat::try_new_from_indices_impl(
			ncols,
			nrows,
			|i| Pair {
				row: entries[i].col,
				col: entries[i].row,
			},
			|_, _| false,
			entries.len(),
		)?;

		Ok(SparseColMat::new_from_argsort_impl(symbolic, &argsort, |i| entries[i].val.clone(), entries.len())?.into_transpose())
	}

	#[track_caller]
	/// create a new matrix from triplets
	///
	/// negative indices are ignored
	pub fn try_new_from_nonnegative_triplets(
		nrows: Rows,
		ncols: Cols,
		entries: &[Triplet<MaybeIdx<Rows, I>, MaybeIdx<Cols, I>, T>],
	) -> Result<Self, CreationError>
	where
		T: ComplexField,
	{
		let (symbolic, argsort) = SymbolicSparseColMat::try_new_from_indices_impl(
			ncols,
			nrows,
			|i| Pair {
				row: unsafe { Idx::<Cols, I>::new_unbound(I::from_signed(entries[i].col.unbound())) },
				col: unsafe { Idx::<Rows, I>::new_unbound(I::from_signed(entries[i].row.unbound())) },
			},
			|row, col| {
				let row = row.unbound().to_signed();
				let col = col.unbound().to_signed();
				let zero = I::Signed::truncate(0);
				row < zero || col < zero
			},
			entries.len(),
		)?;

		Ok(SparseColMat::new_from_argsort_impl(symbolic, &argsort, |i| entries[i].val.clone(), entries.len())?.into_transpose())
	}

	/// see [`SparseRowMatRef::get`]
	#[track_caller]
	#[inline]
	pub fn get(&self, row: Idx<Rows>, col: Idx<Cols>) -> Option<&T> {
		self.rb().get(row, col)
	}

	/// see [`SparseRowMatRef::get`]
	#[track_caller]
	#[inline]
	pub fn get_mut(&mut self, row: Idx<Rows>, col: Idx<Cols>) -> Option<&mut T> {
		self.rb_mut().get_mut(row, col)
	}

	/// see [`SparseRowMatRef::triplet_iter`]
	#[inline]
	pub fn triplet_iter(&self) -> impl '_ + Iterator<Item = Triplet<Idx<Rows>, Idx<Cols>, &'_ T>> {
		self.rb().triplet_iter()
	}
}

impl<'a, Rows: Shape, Cols: Shape, I: Index, T> core::ops::Deref for SparseRowMatRef<'a, I, T, Rows, Cols> {
	type Target = SymbolicSparseRowMatRef<'a, I, Rows, Cols>;

	#[inline]
	fn deref(&self) -> &Self::Target {
		&self.0.symbolic
	}
}

impl<'a, Rows: Shape, Cols: Shape, I: Index, T> core::ops::Deref for SparseRowMatMut<'a, I, T, Rows, Cols> {
	type Target = SymbolicSparseRowMatRef<'a, I, Rows, Cols>;

	#[inline]
	fn deref(&self) -> &Self::Target {
		&self.0.symbolic
	}
}

impl<Rows: Shape, Cols: Shape, I: Index, T> core::ops::Deref for SparseRowMat<I, T, Rows, Cols> {
	type Target = SymbolicSparseRowMat<I, Rows, Cols>;

	#[inline]
	fn deref(&self) -> &Self::Target {
		&self.0.symbolic
	}
}

impl<Rows: Shape, Cols: Shape, I: Index> fmt::Debug for symbolic::Ref<'_, I, Rows, Cols> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		fn imp<'ROWS, 'COLS, I: Index>(mat: SymbolicSparseRowMatRef<'_, I, Dim<'ROWS>, Dim<'COLS>>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
			struct Entries<'a, 'ROWS, 'COLS, I>(SymbolicSparseRowMatRef<'a, I, Dim<'ROWS>, Dim<'COLS>>);

			impl<'ROWS, 'COLS, I: Index> fmt::Debug for Entries<'_, 'ROWS, 'COLS, I> {
				fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
					let mat = self.0;

					f.debug_list()
						.entries(
							mat.nrows()
								.indices()
								.flat_map(|row| mat.col_idx_of_row(row).map(move |col| Pair { row, col })),
						)
						.finish()
				}
			}

			f.debug_struct("SymbolicSparseRowMat")
				.field("nrows", &mat.nrows)
				.field("ncols", &mat.ncols)
				.field("entries", &Entries(mat))
				.finish()
		}
		let this = symbolic::generic::SymbolicSparseRowMat::from_inner_ref(self);
		with_dim!(ROWS, this.nrows().unbound());
		with_dim!(COLS, this.ncols().unbound());

		imp(this.as_shape(ROWS, COLS), f)
	}
}

impl<Rows: Shape, Cols: Shape, I: Index> fmt::Debug for symbolic::Own<I, Rows, Cols> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.rb().fmt(f)
	}
}

impl<Rows: Shape, Cols: Shape, I: Index, T: fmt::Debug> fmt::Debug for SparseRowMatRef<'_, I, T, Rows, Cols> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		fn imp<'ROWS, 'COLS, I: Index, T: fmt::Debug>(
			mat: SparseRowMatRef<'_, I, T, Dim<'ROWS>, Dim<'COLS>>,
			f: &mut fmt::Formatter<'_>,
		) -> fmt::Result {
			struct Entries<'a, 'ROWS, 'COLS, I, T>(SparseRowMatRef<'a, I, T, Dim<'ROWS>, Dim<'COLS>>);

			impl<'ROWS, 'COLS, I: Index, T: fmt::Debug> fmt::Debug for Entries<'_, 'ROWS, 'COLS, I, T> {
				fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
					let mat = self.0;

					f.debug_list()
						.entries(mat.nrows().indices().flat_map(|row| {
							let col_idx = mat.col_idx_of_row(row);
							let val = mat.val_of_row(row);

							iter::zip(col_idx, val).map(move |(col, val)| Triplet {
								row,
								col,
								val: crate::hacks::hijack_debug(val),
							})
						}))
						.finish()
				}
			}

			f.debug_struct("SparseRowMat")
				.field("nrows", &mat.nrows)
				.field("ncols", &mat.ncols)
				.field("entries", &Entries(mat))
				.finish()
		}

		with_dim!(ROWS, self.nrows().unbound());
		with_dim!(COLS, self.ncols().unbound());

		imp(self.as_shape(ROWS, COLS), f)
	}
}

impl<Rows: Shape, Cols: Shape, I: Index, T: fmt::Debug> fmt::Debug for SparseRowMatMut<'_, I, T, Rows, Cols> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.rb().fmt(f)
	}
}

impl<Rows: Shape, Cols: Shape, I: Index, T: fmt::Debug> fmt::Debug for SparseRowMat<I, T, Rows, Cols> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.rb().fmt(f)
	}
}
