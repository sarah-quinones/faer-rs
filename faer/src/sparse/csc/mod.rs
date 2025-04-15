use super::*;
use crate::internal_prelude::*;
use crate::{Idx, MaybeIdx, assert, debug_assert};
use core::ops::Range;
use core::{fmt, iter};

/// implementation of symbolic representation
pub mod symbolic {
	/// generic `SymbolicSparseColMat` wrapper
	pub mod generic {
		use core::fmt::Debug;
		use reborrow::*;

		/// generic `SymbolicSparseColMat` wrapper
		#[derive(Copy, Clone)]
		#[repr(transparent)]
		pub struct SymbolicSparseColMat<Inner>(pub Inner);

		impl<Inner: Debug> Debug for SymbolicSparseColMat<Inner> {
			#[inline(always)]
			fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
				self.0.fmt(f)
			}
		}

		impl<Inner> SymbolicSparseColMat<Inner> {
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

		impl<Inner> core::ops::Deref for SymbolicSparseColMat<Inner> {
			type Target = Inner;

			#[inline(always)]
			fn deref(&self) -> &Self::Target {
				&self.0
			}
		}

		impl<Inner> core::ops::DerefMut for SymbolicSparseColMat<Inner> {
			#[inline(always)]
			fn deref_mut(&mut self) -> &mut Self::Target {
				&mut self.0
			}
		}

		impl<'short, Inner: Reborrow<'short>> Reborrow<'short> for SymbolicSparseColMat<Inner> {
			type Target = SymbolicSparseColMat<Inner::Target>;

			#[inline(always)]
			fn rb(&'short self) -> Self::Target {
				SymbolicSparseColMat(self.0.rb())
			}
		}

		impl<'short, Inner: ReborrowMut<'short>> ReborrowMut<'short> for SymbolicSparseColMat<Inner> {
			type Target = SymbolicSparseColMat<Inner::Target>;

			#[inline(always)]
			fn rb_mut(&'short mut self) -> Self::Target {
				SymbolicSparseColMat(self.0.rb_mut())
			}
		}

		impl<Inner: IntoConst> IntoConst for SymbolicSparseColMat<Inner> {
			type Target = SymbolicSparseColMat<Inner::Target>;

			#[inline(always)]
			fn into_const(self) -> Self::Target {
				SymbolicSparseColMat(self.0.into_const())
			}
		}
	}

	/// see [`super::SymbolicSparseColMatRef`]
	pub struct Ref<'a, I, Rows = usize, Cols = usize> {
		pub(crate) nrows: Rows,
		pub(crate) ncols: Cols,
		pub(crate) col_ptr: &'a [I],
		pub(crate) col_nnz: Option<&'a [I]>,
		pub(crate) row_idx: &'a [I],
	}

	/// see [`super::SymbolicSparseColMat`]
	#[derive(Clone)]
	pub struct Own<I, Rows = usize, Cols = usize> {
		pub(crate) nrows: Rows,
		pub(crate) ncols: Cols,
		pub(crate) col_ptr: alloc::vec::Vec<I>,
		pub(crate) col_nnz: Option<alloc::vec::Vec<I>>,
		pub(crate) row_idx: alloc::vec::Vec<I>,
	}
}

/// implementation of numeric representation
pub mod numeric {
	/// generic `SparseColMat` wrapper
	pub mod generic {
		use core::fmt::Debug;
		use reborrow::*;

		/// generic `SparseColMat` wrapper
		#[derive(Copy, Clone)]
		#[repr(transparent)]
		pub struct SparseColMat<Inner>(pub Inner);

		impl<Inner: Debug> Debug for SparseColMat<Inner> {
			#[inline(always)]
			fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
				self.0.fmt(f)
			}
		}

		impl<Inner> SparseColMat<Inner> {
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

		impl<'short, Inner: Reborrow<'short>> Reborrow<'short> for SparseColMat<Inner> {
			type Target = SparseColMat<Inner::Target>;

			#[inline(always)]
			fn rb(&'short self) -> Self::Target {
				SparseColMat(self.0.rb())
			}
		}

		impl<'short, Inner: ReborrowMut<'short>> ReborrowMut<'short> for SparseColMat<Inner> {
			type Target = SparseColMat<Inner::Target>;

			#[inline(always)]
			fn rb_mut(&'short mut self) -> Self::Target {
				SparseColMat(self.0.rb_mut())
			}
		}

		impl<Inner: IntoConst> IntoConst for SparseColMat<Inner> {
			type Target = SparseColMat<Inner::Target>;

			#[inline(always)]
			fn into_const(self) -> Self::Target {
				SparseColMat(self.0.into_const())
			}
		}
	}

	/// see [`super::SparseColMatRef`]
	pub struct Ref<'a, I, T, Rows = usize, Cols = usize> {
		pub(crate) symbolic: super::SymbolicSparseColMatRef<'a, I, Rows, Cols>,
		pub(crate) val: &'a [T],
	}

	/// see [`super::SparseColMatMut`]
	pub struct Mut<'a, I, T, Rows = usize, Cols = usize> {
		pub(crate) symbolic: super::SymbolicSparseColMatRef<'a, I, Rows, Cols>,
		pub(crate) val: &'a mut [T],
	}

	/// see [`super::SparseColMat`]
	#[derive(Clone)]
	pub struct Own<I, T, Rows = usize, Cols = usize> {
		pub(crate) symbolic: super::SymbolicSparseColMat<I, Rows, Cols>,
		pub(crate) val: alloc::vec::Vec<T>,
	}
}

/// symbolic view structure of sparse matrix in column format, either compressed or uncompressed
///
/// # invariants
/// - `nrows <= I::Signed::MAX` (always checked)
/// - `ncols <= I::Signed::MAX` (always checked)
/// - `col_ptrs` has length `ncols + 1` (always checked)
/// - `col_ptrs` is increasing
/// - `col_ptrs[0]..col_ptrs[ncols]` is a valid range in row_indices (always checked, assuming it's
///   increasing)
/// - if `nnz_per_col` is `none`, elements of `row_indices[col_ptrs[j]..col_ptrs[j + 1]]` are less
///   than `nrows`
///
/// - `nnz_per_col[j] <= col_ptrs[j+1] - col_ptrs[j]`
/// - if `nnz_per_col` is `some(_)`, elements of `row_indices[col_ptrs[j]..][..nnz_per_col[j]]` are
///   less than `nrows`
///
/// # soft invariants
/// - within each column, row indices are sorted in increasing order
///
/// # note
/// some algorithms allow working with matrices containing unsorted row indices per column
///
/// passing such a matrix to an algorithm that does not explicitly permit this is unspecified
/// (though not undefined) behavior
pub type SymbolicSparseColMatRef<'a, I, Rows = usize, Cols = usize> = symbolic::generic::SymbolicSparseColMat<symbolic::Ref<'a, I, Rows, Cols>>;

/// owning symbolic structure of sparse matrix in column format, either compressed or
/// uncompressed
///
/// see [`SymbolicSparseColMatRef`]
pub type SymbolicSparseColMat<I, Rows = usize, Cols = usize> = symbolic::generic::SymbolicSparseColMat<symbolic::Own<I, Rows, Cols>>;

/// view over sparse column major matrix
///
/// see [`SymbolicSparseColMatRef`]
pub type SparseColMatRef<'a, I, T, Rows = usize, Cols = usize> = numeric::generic::SparseColMat<numeric::Ref<'a, I, T, Rows, Cols>>;

/// view over sparse column major matrix
///
/// see [`SymbolicSparseColMatRef`]
pub type SparseColMatMut<'a, I, T, Rows = usize, Cols = usize> = numeric::generic::SparseColMat<numeric::Mut<'a, I, T, Rows, Cols>>;

/// owning sparse column major matrix
///
/// see [`SymbolicSparseColMatRef`]
pub type SparseColMat<I, T, Rows = usize, Cols = usize> = numeric::generic::SparseColMat<numeric::Own<I, T, Rows, Cols>>;

pub(crate) fn partition_by_lt<I: Index>(upper: I) -> impl Fn(&I) -> bool {
	move |&p| p < upper
}
pub(crate) fn partition_by_le<I: Index>(upper: I) -> impl Fn(&I) -> bool {
	move |&p| p <= upper
}

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
			col_ptr: &self.col_ptr,
			col_nnz: self.col_nnz.as_deref(),
			row_idx: &self.row_idx,
		}
	}
}

#[inline(always)]
#[track_caller]
fn assume_col_ptr<I: Index>(nrows: usize, ncols: usize, col_ptr: &[I], col_nnz: Option<&[I]>, row_idx: &[I]) {
	assert!(all(ncols <= I::Signed::MAX.zx(), nrows <= I::Signed::MAX.zx(),));
	assert!(col_ptr.len() == ncols + 1);
	assert!(col_ptr[ncols].zx() <= row_idx.len());
	if let Some(col_nnz) = col_nnz {
		assert!(col_nnz.len() == ncols);
	}
}

#[track_caller]
fn check_col_ptr<I: Index>(nrows: usize, ncols: usize, col_ptr: &[I], col_nnz: Option<&[I]>, row_idx: &[I]) {
	assert!(all(ncols <= I::Signed::MAX.zx(), nrows <= I::Signed::MAX.zx(),));
	assert!(col_ptr.len() == ncols + 1);
	if let Some(nnz_per_col) = col_nnz {
		assert!(nnz_per_col.len() == ncols);
		for (&nnz_j, &[col, col_next]) in iter::zip(nnz_per_col, windows2(col_ptr)) {
			assert!(col <= col_next);
			assert!(nnz_j <= col_next - col);
		}
	} else {
		for &[col, col_next] in windows2(col_ptr) {
			assert!(col <= col_next);
		}
	}
	assert!(col_ptr[ncols].zx() <= row_idx.len());
}

#[track_caller]
fn check_row_idx<I: Index>(nrows: usize, ncols: usize, col_ptr: &[I], col_nnz: Option<&[I]>, row_idx: &[I]) {
	_ = ncols;
	if let Some(col_nnz) = col_nnz {
		for (nnz, &c) in iter::zip(col_nnz, col_ptr) {
			let c = c.zx();
			let nnz = nnz.zx();
			let row_idx = &row_idx[c..c + nnz];
			if !row_idx.is_empty() {
				let mut i = row_idx[0].zx();
				for &i_next in &row_idx[1..] {
					let i_next = i_next.zx();
					assert!(i < i_next);
					i = i_next;
				}
				assert!(i < nrows);
			}
		}
	} else {
		for &[c, c_next] in windows2(col_ptr) {
			let row_idx = &row_idx[c.zx()..c_next.zx()];
			if !row_idx.is_empty() {
				let mut i = row_idx[0].zx();
				for &i_next in &row_idx[1..] {
					let i_next = i_next.zx();
					assert!(i < i_next);
					i = i_next;
				}
				assert!(i < nrows);
			}
		}
	}
}

#[track_caller]
fn check_row_idx_unsorted<I: Index>(nrows: usize, ncols: usize, col_ptr: &[I], col_nnz: Option<&[I]>, row_idx: &[I]) {
	_ = ncols;

	if let Some(col_nnz) = col_nnz {
		for (&nnz, &c) in iter::zip(col_nnz, col_ptr) {
			let c = c.zx();
			let nnz = nnz.zx();
			for &i in &row_idx[c..c + nnz] {
				let i = i.zx();
				assert!(i < nrows);
			}
		}
	} else {
		for &[c, c_next] in windows2(col_ptr) {
			for &i in &row_idx[c.zx()..c_next.zx()] {
				let i = i.zx();
				assert!(i < nrows);
			}
		}
	}
}

impl<'a, Rows: Shape, Cols: Shape, I: Index> SymbolicSparseColMatRef<'a, I, Rows, Cols> {
	/// creates a new symbolic matrix view without checking its invariants
	///
	/// # safety
	/// see type level documentation.
	#[inline]
	#[track_caller]
	pub unsafe fn new_unchecked(nrows: Rows, ncols: Cols, col_ptr: &'a [I], col_nnz: Option<&'a [I]>, row_idx: &'a [I]) -> Self {
		assume_col_ptr(nrows.unbound(), ncols.unbound(), col_ptr, col_nnz, row_idx);

		Self {
			0: symbolic::Ref {
				nrows,
				ncols,
				col_ptr,
				col_nnz,
				row_idx,
			},
		}
	}

	/// creates a new symbolic matrix view after checking its invariants
	///
	/// # safety
	/// see type level documentation.
	#[inline]
	#[track_caller]
	pub fn new_checked(nrows: Rows, ncols: Cols, col_ptr: &'a [I], col_nnz: Option<&'a [I]>, row_idx: &'a [I]) -> Self {
		check_col_ptr(nrows.unbound(), ncols.unbound(), col_ptr, col_nnz, row_idx);
		check_row_idx(nrows.unbound(), ncols.unbound(), col_ptr, col_nnz, row_idx);

		Self {
			0: symbolic::Ref {
				nrows,
				ncols,
				col_ptr,
				col_nnz,
				row_idx,
			},
		}
	}

	/// creates a new symbolic matrix view after checking its invariants (excluding soft invariants)
	///
	/// # safety
	/// see type level documentation.
	#[inline]
	#[track_caller]
	pub fn new_unsorted_checked(nrows: Rows, ncols: Cols, col_ptr: &'a [I], col_nnz: Option<&'a [I]>, row_idx: &'a [I]) -> Self {
		check_col_ptr(nrows.unbound(), ncols.unbound(), col_ptr, col_nnz, row_idx);
		check_row_idx_unsorted(nrows.unbound(), ncols.unbound(), col_ptr, col_nnz, row_idx);

		Self {
			0: symbolic::Ref {
				nrows,
				ncols,
				col_ptr,
				col_nnz,
				row_idx,
			},
		}
	}

	/// returns the components of the sparse matrix
	/// - number of rows
	/// - number of columns
	/// - column pointers
	/// - column non-zero counts
	/// - row indices
	#[inline]
	pub fn parts(self) -> (Rows, Cols, &'a [I], Option<&'a [I]>, &'a [I]) {
		(self.nrows, self.ncols, self.col_ptr, self.col_nnz, self.row_idx)
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
	pub fn transpose(self) -> SymbolicSparseRowMatRef<'a, I, Cols, Rows> {
		SymbolicSparseRowMatRef {
			0: super::csr::symbolic::Ref {
				nrows: self.ncols,
				ncols: self.nrows,
				row_ptr: self.col_ptr,
				row_nnz: self.col_nnz,
				col_idx: self.row_idx,
			},
		}
	}

	/// returns a newly allocated matrix holding the values of `self`
	#[inline]
	pub fn to_owned(&self) -> Result<SymbolicSparseColMat<I, Rows, Cols>, FaerError> {
		Ok(SymbolicSparseColMat {
			0: symbolic::Own {
				nrows: self.nrows,
				ncols: self.ncols,
				col_ptr: try_collect(self.col_ptr.iter().copied())?,
				col_nnz: self.col_nnz.map(|col_nnz| try_collect(col_nnz.iter().copied())).transpose()?,
				row_idx: try_collect(self.row_idx.iter().copied())?,
			},
		})
	}

	/// returns a newly allocated matrix holding the values of `self` in row major format
	#[inline]
	pub fn to_row_major(&self) -> Result<SymbolicSparseRowMat<I, Rows, Cols>, FaerError> {
		let mat = SparseColMatRef::new(*self, Symbolic::materialize(self.row_idx.len()));
		Ok(mat.to_row_major()?.0.symbolic)
	}

	/// returns the number of non-zero elements in the matrix
	#[inline]
	pub fn compute_nnz(&self) -> usize {
		fn imp<I: Index>(col_ptr: &[I], col_nnz: Option<&[I]>, ncols: usize) -> usize {
			match col_nnz {
				Some(col_nnz) => {
					let mut nnz = 0usize;
					for &nnz_j in col_nnz {
						// can't overflow
						nnz += nnz_j.zx();
					}
					nnz
				},
				None => col_ptr[ncols].zx() - col_ptr[0].zx(),
			}
		}
		imp(self.col_ptr, self.col_nnz, self.ncols.unbound())
	}

	/// returns the column pointers
	#[inline]
	pub fn col_ptr(&self) -> &'a [I] {
		self.col_ptr
	}

	/// returns the column non-zero counts
	#[inline]
	pub fn col_nnz(&self) -> Option<&'a [I]> {
		self.col_nnz
	}

	/// returns the row indices
	#[inline]
	pub fn row_idx(&self) -> &'a [I] {
		self.row_idx
	}

	/// returns the range specifying the indices of column `j`
	#[inline]
	#[track_caller]
	pub fn col_range(&self, j: Idx<Cols>) -> Range<usize> {
		assert!(j < self.ncols());
		unsafe { self.col_range_unchecked(j) }
	}

	/// returns the range specifying the indices of column `j`, without bound checks
	#[inline]
	#[track_caller]
	pub unsafe fn col_range_unchecked(&self, j: Idx<Cols>) -> Range<usize> {
		debug_assert!(j < self.ncols());
		let j = j.unbound();

		let start = self.col_ptr.get_unchecked(j).zx();
		let end = self
			.col_nnz
			.map(|col_nnz| (col_nnz.get_unchecked(j).zx() + start))
			.unwrap_or(self.col_ptr.get_unchecked(j + 1).zx());

		start..end
	}

	/// returns the row indices of column `j`
	#[inline]
	#[track_caller]
	pub fn row_idx_of_col_raw(&self, j: Idx<Cols>) -> &'a [Idx<Rows, I>] {
		unsafe {
			let slice = self.row_idx.get_unchecked(self.col_range(j));
			let len = slice.len();
			core::slice::from_raw_parts(slice.as_ptr() as *const Idx<Rows, I>, len)
		}
	}

	/// returns the row indices of column `j`
	#[inline]
	#[track_caller]
	pub fn row_idx_of_col(&self, j: Idx<Cols>) -> impl 'a + Clone + ExactSizeIterator + DoubleEndedIterator<Item = Idx<Rows>>
	where
		Rows: 'a,
		Cols: 'a,
	{
		self.row_idx_of_col_raw(j)
			.iter()
			.map(|&i| unsafe { Idx::<Rows>::new_unbound(i.unbound().zx()) })
	}

	/// returns the input matrix with the given shape after checking that it matches the
	/// current shape
	#[inline]
	#[track_caller]
	pub fn as_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SymbolicSparseColMatRef<'a, I, V, H> {
		assert!(all(self.nrows.unbound() == nrows.unbound(), self.ncols.unbound() == ncols.unbound()));

		SymbolicSparseColMatRef {
			0: symbolic::Ref {
				nrows,
				ncols,
				col_ptr: self.col_ptr,
				col_nnz: self.col_nnz,
				row_idx: self.row_idx,
			},
		}
	}

	/// returns the input matrix with dynamic shape
	#[inline]
	pub fn as_dyn(self) -> SymbolicSparseColMatRef<'a, I> {
		SymbolicSparseColMatRef {
			0: symbolic::Ref {
				nrows: self.nrows.unbound(),
				ncols: self.ncols.unbound(),
				col_ptr: self.col_ptr,
				col_nnz: self.col_nnz,
				row_idx: self.row_idx,
			},
		}
	}

	/// Returns a view over the symbolic structure of `self`.
	#[inline]
	pub fn as_ref(self) -> SymbolicSparseColMatRef<'a, I, Rows, Cols> {
		SymbolicSparseColMatRef {
			0: symbolic::Ref {
				nrows: self.nrows,
				ncols: self.ncols,
				col_ptr: self.col_ptr,
				col_nnz: self.col_nnz,
				row_idx: self.row_idx,
			},
		}
	}
}

impl<Rows: Shape, Cols: Shape, I: Index> SymbolicSparseColMat<I, Rows, Cols> {
	/// creates a new symbolic matrix view without checking its invariants
	///
	/// # safety
	/// see type level documentation.
	#[inline]
	#[track_caller]
	pub unsafe fn new_unchecked(
		nrows: Rows,
		ncols: Cols,
		col_ptr: alloc::vec::Vec<I>,
		col_nnz: Option<alloc::vec::Vec<I>>,
		row_idx: alloc::vec::Vec<I>,
	) -> Self {
		assume_col_ptr(nrows.unbound(), ncols.unbound(), &col_ptr, col_nnz.as_deref(), &row_idx);

		Self {
			0: symbolic::Own {
				nrows,
				ncols,
				col_ptr,
				col_nnz,
				row_idx,
			},
		}
	}

	/// creates a new symbolic matrix view after checking its invariants
	///
	/// # safety
	/// see type level documentation.
	#[inline]
	#[track_caller]
	pub fn new_checked(
		nrows: Rows,
		ncols: Cols,
		col_ptr: alloc::vec::Vec<I>,
		col_nnz: Option<alloc::vec::Vec<I>>,
		row_idx: alloc::vec::Vec<I>,
	) -> Self {
		check_col_ptr(nrows.unbound(), ncols.unbound(), &col_ptr, col_nnz.as_deref(), &row_idx);
		check_row_idx(nrows.unbound(), ncols.unbound(), &col_ptr, col_nnz.as_deref(), &row_idx);

		Self {
			0: symbolic::Own {
				nrows,
				ncols,
				col_ptr,
				col_nnz,
				row_idx,
			},
		}
	}

	/// creates a new symbolic matrix view after checking its invariants (excluding soft invariants)
	///
	/// # safety
	/// see type level documentation.
	#[inline]
	#[track_caller]
	pub fn new_unsorted_checked(
		nrows: Rows,
		ncols: Cols,
		col_ptr: alloc::vec::Vec<I>,
		col_nnz: Option<alloc::vec::Vec<I>>,
		row_idx: alloc::vec::Vec<I>,
	) -> Self {
		check_col_ptr(nrows.unbound(), ncols.unbound(), &col_ptr, col_nnz.as_deref(), &row_idx);
		check_row_idx_unsorted(nrows.unbound(), ncols.unbound(), &col_ptr, col_nnz.as_deref(), &row_idx);

		Self {
			0: symbolic::Own {
				nrows,
				ncols,
				col_ptr,
				col_nnz,
				row_idx,
			},
		}
	}

	#[inline]
	/// see [`SymbolicSparseColMatRef::parts`]
	pub fn parts(&self) -> (Rows, Cols, &'_ [I], Option<&'_ [I]>, &'_ [I]) {
		(self.nrows, self.ncols, &self.col_ptr, self.col_nnz.as_deref(), &self.row_idx)
	}

	#[inline]
	/// see [`SymbolicSparseColMatRef::parts`]
	pub fn into_parts(self) -> (Rows, Cols, alloc::vec::Vec<I>, Option<alloc::vec::Vec<I>>, alloc::vec::Vec<I>) {
		(self.0.nrows, self.0.ncols, self.0.col_ptr, self.0.col_nnz, self.0.row_idx)
	}

	#[inline]
	/// returns the number of rows of the matrix
	pub fn nrows(&self) -> Rows {
		self.nrows
	}

	#[inline]
	/// returns the number of columns of the matrix
	pub fn ncols(&self) -> Cols {
		self.ncols
	}

	#[inline]
	/// returns the number of rows and columns of the matrix
	pub fn shape(&self) -> (Rows, Cols) {
		(self.nrows, self.ncols)
	}

	#[inline]
	/// returns a view over the transpose of `self`
	pub fn transpose(&self) -> SymbolicSparseRowMatRef<'_, I, Cols, Rows> {
		self.rb().transpose()
	}

	#[inline]
	/// returns the transpose of `self`
	pub fn into_transpose(self) -> SymbolicSparseRowMat<I, Cols, Rows> {
		SymbolicSparseRowMat {
			0: super::csr::symbolic::Own {
				nrows: self.0.ncols,
				ncols: self.0.nrows,
				row_ptr: self.0.col_ptr,
				row_nnz: self.0.col_nnz,
				col_idx: self.0.row_idx,
			},
		}
	}

	#[inline]
	/// see [`SymbolicSparseColMatRef::to_owned`]
	pub fn to_owned(&self) -> Result<SymbolicSparseColMat<I, Rows, Cols>, FaerError> {
		self.rb().to_owned()
	}

	#[inline]
	/// see [`SymbolicSparseColMatRef::to_row_major`]
	pub fn to_row_major(&self) -> Result<SymbolicSparseRowMat<I, Rows, Cols>, FaerError> {
		self.rb().to_row_major()
	}

	#[inline]
	/// see [`SymbolicSparseColMatRef::compute_nnz`]
	pub fn compute_nnz(&self) -> usize {
		self.rb().compute_nnz()
	}

	#[inline]
	/// see [`SymbolicSparseColMatRef::col_ptr`]
	pub fn col_ptr(&self) -> &'_ [I] {
		&self.col_ptr
	}

	#[inline]
	/// see [`SymbolicSparseColMatRef::col_nnz`]
	pub fn col_nnz(&self) -> Option<&'_ [I]> {
		self.col_nnz.as_deref()
	}

	#[inline]
	/// see [`SymbolicSparseColMatRef::row_idx`]
	pub fn row_idx(&self) -> &'_ [I] {
		&self.row_idx
	}

	#[inline]
	#[track_caller]
	/// see [`SymbolicSparseColMatRef::col_range`]
	pub fn col_range(&self, j: Idx<Cols>) -> Range<usize> {
		self.rb().col_range(j)
	}

	#[inline]
	#[track_caller]
	/// see [`SymbolicSparseColMatRef::col_range_unchecked`]
	pub unsafe fn col_range_unchecked(&self, j: Idx<Cols>) -> Range<usize> {
		self.rb().col_range_unchecked(j)
	}

	#[inline]
	#[track_caller]
	/// see [`SymbolicSparseColMatRef::row_idx_of_col_raw`]
	pub fn row_idx_of_col_raw(&self, j: Idx<Cols>) -> &'_ [Idx<Rows, I>] {
		self.rb().row_idx_of_col_raw(j)
	}

	#[inline]
	#[track_caller]
	/// see [`SymbolicSparseColMatRef::row_idx_of_col`]
	pub fn row_idx_of_col(&self, j: Idx<Cols>) -> impl '_ + Clone + ExactSizeIterator + DoubleEndedIterator<Item = Idx<Rows>> {
		self.rb().row_idx_of_col(j)
	}

	#[inline]
	#[track_caller]
	/// see [`SymbolicSparseColMatRef::as_shape`]
	pub fn as_shape<V: Shape, H: Shape>(&self, nrows: V, ncols: H) -> SymbolicSparseColMatRef<'_, I, V, H> {
		self.rb().as_shape(nrows, ncols)
	}

	#[inline]
	#[track_caller]
	/// see [`SymbolicSparseColMatRef::as_shape`]
	pub fn into_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SymbolicSparseColMat<I, V, H> {
		assert!(all(self.nrows().unbound() == nrows.unbound(), self.ncols().unbound() == ncols.unbound()));
		SymbolicSparseColMat {
			0: symbolic::Own {
				nrows,
				ncols,
				col_ptr: self.0.col_ptr,
				col_nnz: self.0.col_nnz,
				row_idx: self.0.row_idx,
			},
		}
	}

	#[inline]
	/// see [`SymbolicSparseColMatRef::as_dyn`]
	pub fn as_dyn(&self) -> SymbolicSparseColMatRef<'_, I> {
		self.rb().as_dyn()
	}

	#[inline]
	/// see [`SymbolicSparseColMatRef::as_dyn`]
	pub fn into_dyn(self) -> SymbolicSparseColMat<I> {
		SymbolicSparseColMat {
			0: symbolic::Own {
				nrows: self.0.nrows.unbound(),
				ncols: self.0.ncols.unbound(),
				col_ptr: self.0.col_ptr,
				col_nnz: self.0.col_nnz,
				row_idx: self.0.row_idx,
			},
		}
	}

	/// Returns a view over the symbolic structure of `self`.
	#[inline]
	pub fn as_ref(&self) -> SymbolicSparseColMatRef<'_, I, Rows, Cols> {
		SymbolicSparseColMatRef {
			0: symbolic::Ref {
				nrows: self.nrows,
				ncols: self.ncols,
				col_ptr: &self.col_ptr,
				col_nnz: self.col_nnz.as_deref(),
				row_idx: &self.row_idx,
			},
		}
	}

	#[inline]
	pub(crate) fn try_new_from_indices_impl(
		nrows: Rows,
		ncols: Cols,
		idx: impl Fn(usize) -> Pair<Idx<Rows, I>, Idx<Cols, I>>,
		ignore: impl Fn(Idx<Rows, I>, Idx<Cols, I>) -> bool,
		all_nnz: usize,
	) -> Result<(Self, Argsort<I>), CreationError> {
		if nrows.unbound() > I::Signed::MAX.zx() || ncols.unbound() > I::Signed::MAX.zx() || all_nnz > I::Signed::MAX.zx() {
			return Err(CreationError::Generic(FaerError::IndexOverflow));
		}

		if all_nnz == 0 {
			return Ok((
				Self {
					0: symbolic::Own {
						nrows,
						ncols,
						col_ptr: try_zeroed(ncols.unbound() + 1)?,
						col_nnz: None,
						row_idx: alloc::vec::Vec::new(),
					},
				},
				Argsort {
					idx: alloc::vec::Vec::new(),
					all_nnz,
					nnz: 0,
				},
			));
		}

		let mut argsort = try_collect((0..all_nnz).map(I::truncate))?;

		argsort.sort_unstable_by_key(|&i| {
			let Pair { row, col } = idx(i.zx());
			let ignore = ignore(row, col);
			(ignore, col, row)
		});

		let all_nnz = argsort.partition_point(|&i| {
			let Pair { row, col } = idx(i.zx());
			!ignore(row, col)
		});

		let mut n_dup = 0usize;
		let mut prev = (I::truncate(usize::MAX), I::truncate(usize::MAX));

		let top_bit = I::truncate(1 << (I::BITS - 1));

		for i in 0..all_nnz {
			let idx = idx(argsort[i].zx());
			let idx @ (row, col) = (idx.row.unbound(), idx.col.unbound());

			let valid_row = if try_const! { Rows::IS_BOUND } {
				true
			} else {
				row.zx() < nrows.unbound()
			};
			let valid_col = if try_const! { Cols::IS_BOUND } {
				true
			} else {
				col.zx() < ncols.unbound()
			};

			if !(valid_row && valid_col) {
				return Err(CreationError::OutOfBounds {
					row: row.zx(),
					col: col.zx(),
				});
			}

			let same = idx == prev;

			argsort[i] = argsort[i] | (if same { top_bit } else { I::truncate(0) });

			n_dup += same as usize;
			prev = idx;
		}

		let nnz = all_nnz - n_dup;
		let mut col_ptr = try_zeroed::<I>(ncols.unbound() + 1)?;
		let mut row_idx = try_zeroed::<I>(nnz)?;

		let mut reader = 0usize;
		let mut writer = 0usize;

		for j in Cols::indices(Cols::start(), ncols.end()) {
			let jj = I::truncate(j.unbound());
			let mut n_unique = 0usize;

			let mut prev = I::truncate(usize::MAX);

			while reader < all_nnz {
				let Pair { row, col } = idx((argsort[reader] & !top_bit).zx());

				if col.unbound() != jj {
					break;
				}

				reader += 1;

				let row = row.unbound();

				if row == prev {
					continue;
				}

				prev = row;

				row_idx[writer] = row;
				writer += 1;

				n_unique += 1;
			}

			col_ptr[j.unbound() + 1] = col_ptr[j.unbound()] + I::truncate(n_unique);
		}

		Ok((
			unsafe { Self::new_unchecked(nrows, ncols, col_ptr, None, row_idx) },
			Argsort { idx: argsort, all_nnz, nnz },
		))
	}

	/// create a new symbolic structure, and the corresponding order for the numerical values
	/// from pairs of indices
	#[inline]
	pub fn try_new_from_indices(nrows: Rows, ncols: Cols, idx: &[Pair<Idx<Rows, I>, Idx<Cols, I>>]) -> Result<(Self, Argsort<I>), CreationError> {
		Self::try_new_from_indices_impl(nrows, ncols, |i| idx[i], |_, _| false, idx.len())
	}

	/// create a new symbolic structure, and the corresponding order for the numerical values
	/// from pairs of indices
	///
	/// negative indices are ignored
	#[inline]
	pub fn try_new_from_nonnegative_indices(
		nrows: Rows,
		ncols: Cols,
		idx: &[Pair<MaybeIdx<Rows, I>, MaybeIdx<Cols, I>>],
	) -> Result<(Self, Argsort<I>), CreationError> {
		Self::try_new_from_indices_impl(
			nrows,
			ncols,
			|i| Pair {
				row: unsafe { Idx::<Rows, I>::new_unbound(I::from_signed(idx[i].row.unbound())) },
				col: unsafe { Idx::<Cols, I>::new_unbound(I::from_signed(idx[i].col.unbound())) },
			},
			|row, col| {
				let row = row.unbound().to_signed();
				let col = col.unbound().to_signed();
				let zero = I::Signed::truncate(0);

				row < zero || col < zero
			},
			idx.len(),
		)
	}
}

impl<'a, Rows: Shape, Cols: Shape, I: Index, T> SparseColMatRef<'a, I, T, Rows, Cols> {
	/// creates a new sparse matrix view.
	///
	/// # panics
	/// panics if the length of `values` is not equal to the length of
	/// `symbolic.row_idx()`
	#[inline]
	#[track_caller]
	pub fn new(symbolic: SymbolicSparseColMatRef<'a, I, Rows, Cols>, val: &'a [T]) -> Self {
		assert!(symbolic.row_idx().len() == val.len());
		Self {
			0: numeric::Ref { symbolic, val },
		}
	}

	/// returns the symbolic and numeric components of the sparse matrix
	#[inline]
	pub fn parts(self) -> (SymbolicSparseColMatRef<'a, I, Rows, Cols>, &'a [T]) {
		(self.0.symbolic, self.0.val)
	}

	/// returns the symbolic component of the sparse matrix
	#[inline]
	pub fn symbolic(&self) -> SymbolicSparseColMatRef<'a, I, Rows, Cols> {
		self.0.symbolic
	}

	/// returns the numeric component of the sparse matrix
	#[inline]
	pub fn val(self) -> &'a [T] {
		self.0.val
	}

	/// returns a view over the values of column `j`
	#[inline]
	#[track_caller]
	pub fn val_of_col(self, j: Idx<Cols>) -> &'a [T] {
		unsafe { self.0.val.get_unchecked(self.col_range(j)) }
	}

	/// returns the input matrix with the given shape after checking that it matches the
	/// current shape
	#[inline]
	#[track_caller]
	pub fn as_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SparseColMatRef<'a, I, T, V, H> {
		SparseColMatRef {
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
		let row = I::truncate(row.unbound());
		let coll = col.unbound();
		let start = self.symbolic().as_dyn().row_idx_of_col_raw(coll).partition_point(partition_by_lt(row));
		let end = start + self.symbolic().as_dyn().row_idx_of_col_raw(coll)[start..].partition_point(partition_by_le(row));

		if end == start + 1 { Some(&self.val_of_col(col)[start]) } else { None }
	}

	/// returns the input matrix with dynamic shape
	#[inline]
	pub fn as_dyn(self) -> SparseColMatRef<'a, I, T> {
		SparseColMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic.as_dyn(),
				val: self.0.val,
			},
		}
	}

	/// returns a view over `self`
	#[inline]
	pub fn as_ref(self) -> SparseColMatRef<'a, I, T, Rows, Cols> {
		SparseColMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic,
				val: self.0.val,
			},
		}
	}

	/// returns a view over the transpose of `self`
	#[inline]
	pub fn transpose(self) -> SparseRowMatRef<'a, I, T, Cols, Rows> {
		SparseRowMatRef {
			0: super::csr::numeric::Ref {
				symbolic: self.0.symbolic.transpose(),
				val: self.0.val,
			},
		}
	}

	/// returns a view over the conjugate of `self`
	#[inline]
	pub fn conjugate(self) -> SparseColMatRef<'a, I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.0.val.len();
		SparseColMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic,
				val: unsafe { core::slice::from_raw_parts(self.0.val.as_ptr() as *const T::Conj, len) },
			},
		}
	}

	/// returns a view over the adjoint of `self`
	#[inline]
	pub fn adjoint(self) -> SparseRowMatRef<'a, I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.conjugate().transpose()
	}

	/// returns an unconjugated view over `self`
	#[inline]
	pub fn canonical(self) -> SparseColMatRef<'a, I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.0.val.len();
		SparseColMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic,
				val: unsafe { core::slice::from_raw_parts(self.0.val.as_ptr() as *const T::Canonical, len) },
			},
		}
	}

	/// returns a newly allocated matrix holding the (possibly conjugated) values of `self` in row
	/// major format
	#[inline]
	pub fn to_row_major(&self) -> Result<SparseRowMat<I, T::Canonical, Rows, Cols>, FaerError>
	where
		T: Conjugate,
	{
		let max = self.row_idx().len();
		let mut new_col_ptr = try_zeroed::<I>(self.nrows().unbound() + 1)?;
		let mut new_row_idx = try_zeroed::<I>(max)?;
		let mut new_val = try_collect(repeat_n!(zero::<T::Canonical>(), max))?;
		let nnz = utils::transpose_dedup(
			&mut new_val,
			&mut new_col_ptr,
			&mut new_row_idx,
			*self,
			MemStack::new(&mut dyn_stack::MemBuffer::try_new(utils::transpose_dedup_scratch::<I>(
				self.nrows().unbound(),
				self.ncols().unbound(),
			))?),
		)
		.compute_nnz(); // O(1) since it's compressed

		new_val.truncate(nnz);
		new_row_idx.truncate(nnz);

		Ok(SparseRowMat::new(
			unsafe { SymbolicSparseRowMat::new_unchecked(self.nrows(), self.ncols(), new_col_ptr, None, new_row_idx) },
			new_val,
		))
	}

	/// returns a newly allocated dense matrix holding the (possibly conjugated) values of `self`
	#[inline]
	pub fn to_dense(&self) -> Mat<T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		fn imp<'ROWS, 'COLS, I: Index, T: Conjugate>(
			src: SparseColMatRef<'_, I, T, Dim<'ROWS>, Dim<'COLS>>,
		) -> Mat<T::Canonical, Dim<'ROWS>, Dim<'COLS>> {
			let src = src.canonical();

			let mut out = Mat::zeros(src.nrows(), src.ncols());
			let N = src.ncols();

			for j in N.indices() {
				for (i, val) in iter::zip(src.row_idx_of_col(j), src.val_of_col(j)) {
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
		(0..self.ncols.unbound()).flat_map(
			#[inline(always)]
			move |j| {
				let j = unsafe { self.ncols.unchecked_idx(j) };
				let range = self.col_range(j);
				iter::zip(
					iter::zip(
						self.0.symbolic.row_idx[range.clone()].iter().map(
							#[inline(always)]
							move |i| unsafe { self.nrows.unchecked_idx(i.zx()) },
						),
						iter::repeat_n(j, range.len()),
					),
					&self.0.val[range],
				)
				.map(
					#[inline(always)]
					move |((i, j), v)| Triplet::new(i, j, v),
				)
			},
		)
	}
}

impl<'a, Rows: Shape, Cols: Shape, I: Index, T> SparseColMatMut<'a, I, T, Rows, Cols> {
	#[inline]
	#[track_caller]
	/// see [`SparseColMatRef::new`]
	pub fn new(symbolic: SymbolicSparseColMatRef<'a, I, Rows, Cols>, val: &'a mut [T]) -> Self {
		assert!(symbolic.row_idx().len() == val.len());
		Self {
			0: numeric::Mut { symbolic, val },
		}
	}

	#[inline]
	/// see [`SparseColMatRef::parts`]
	pub fn parts(self) -> (SymbolicSparseColMatRef<'a, I, Rows, Cols>, &'a [T]) {
		(self.0.symbolic, self.0.val)
	}

	#[inline]
	/// see [`SparseColMatRef::parts`]
	pub fn parts_mut(self) -> (SymbolicSparseColMatRef<'a, I, Rows, Cols>, &'a mut [T]) {
		(self.0.symbolic, self.0.val)
	}

	#[inline]
	/// see [`SparseColMatRef::symbolic`]
	pub fn symbolic(&self) -> SymbolicSparseColMatRef<'a, I, Rows, Cols> {
		self.0.symbolic
	}

	#[inline]
	/// see [`SparseColMatRef::val`]
	pub fn val(self) -> &'a [T] {
		self.0.val
	}

	#[inline]
	/// see [`SparseColMatRef::val`]
	pub fn val_mut(self) -> &'a mut [T] {
		self.0.val
	}

	#[inline]
	#[track_caller]
	/// see [`SparseColMatRef::val_of_col`]
	pub fn val_of_col(self, j: Idx<Cols>) -> &'a [T] {
		unsafe { self.0.val.get_unchecked(self.col_range(j)) }
	}

	#[inline]
	#[track_caller]
	/// see [`SparseColMatRef::val_of_col`]
	pub fn val_of_col_mut(self, j: Idx<Cols>) -> &'a mut [T] {
		unsafe { self.0.val.get_unchecked_mut(self.col_range(j)) }
	}

	#[inline]
	#[track_caller]
	/// see [`SymbolicSparseColMatRef::row_idx_of_col`] and [`SparseColMatRef::val_of_col`]
	pub fn idx_val_of_col_mut(self, j: Idx<Cols>) -> (impl 'a + Clone + ExactSizeIterator + DoubleEndedIterator<Item = Idx<Rows>>, &'a mut [T])
	where
		Rows: 'a,
		Cols: 'a,
	{
		let range = self.col_range(j);
		unsafe { (self.0.symbolic.row_idx_of_col(j), self.0.val.get_unchecked_mut(range)) }
	}

	#[inline]
	#[track_caller]
	/// see [`SparseColMatRef::as_shape`]
	pub fn as_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SparseColMatRef<'a, I, T, V, H> {
		SparseColMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic.as_shape(nrows, ncols),
				val: self.0.val,
			},
		}
	}

	#[inline]
	#[track_caller]
	/// see [`SparseColMatRef::as_shape`]
	pub fn as_shape_mut<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SparseColMatMut<'a, I, T, V, H> {
		SparseColMatMut {
			0: numeric::Mut {
				symbolic: self.0.symbolic.as_shape(nrows, ncols),
				val: self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseColMatRef::as_dyn`]
	pub fn as_dyn(self) -> SparseColMatRef<'a, I, T> {
		SparseColMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic.as_dyn(),
				val: self.0.val,
			},
		}
	}

	/// see [`SparseColMatRef::get`]
	#[track_caller]
	#[inline]
	pub fn get(self, row: Idx<Rows>, col: Idx<Cols>) -> Option<&'a T> {
		self.into_const().get(row, col)
	}

	#[track_caller]
	/// see [`SparseColMatRef::get`]
	pub fn get_mut(self, row: Idx<Rows>, col: Idx<Cols>) -> Option<&'a mut T> {
		assert!(row < self.nrows());
		assert!(col < self.ncols());
		let row = I::truncate(row.unbound());
		let coll = col.unbound();
		let start = self.symbolic().as_dyn().row_idx_of_col_raw(coll).partition_point(partition_by_lt(row));
		let end = start + self.symbolic().as_dyn().row_idx_of_col_raw(coll)[start..].partition_point(partition_by_le(row));

		if end == start + 1 {
			Some(&mut self.val_of_col_mut(col)[start])
		} else {
			None
		}
	}

	#[inline]
	/// see [`SparseColMatRef::as_dyn`]
	pub fn as_dyn_mut(self) -> SparseColMatMut<'a, I, T> {
		SparseColMatMut {
			0: numeric::Mut {
				symbolic: self.0.symbolic.as_dyn(),
				val: self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseColMatRef::transpose`]
	pub fn transpose(self) -> SparseRowMatRef<'a, I, T, Cols, Rows> {
		SparseRowMatRef {
			0: super::csr::numeric::Ref {
				symbolic: self.0.symbolic.transpose(),
				val: self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseColMatRef::transpose`]
	pub fn transpose_mut(self) -> SparseRowMatMut<'a, I, T, Cols, Rows> {
		SparseRowMatMut {
			0: super::csr::numeric::Mut {
				symbolic: self.0.symbolic.transpose(),
				val: self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseColMatRef::conjugate`]
	pub fn conjugate(self) -> SparseColMatRef<'a, I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.0.val.len();
		SparseColMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic,
				val: unsafe { core::slice::from_raw_parts(self.0.val.as_ptr() as *const T::Conj, len) },
			},
		}
	}

	#[inline]
	/// see [`SparseColMatRef::conjugate`]
	pub fn conjugate_mut(self) -> SparseColMatMut<'a, I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.0.val.len();
		SparseColMatMut {
			0: numeric::Mut {
				symbolic: self.0.symbolic,
				val: unsafe { core::slice::from_raw_parts_mut(self.0.val.as_mut_ptr() as *mut T::Conj, len) },
			},
		}
	}

	#[inline]
	/// see [`SparseColMatRef::adjoint`]
	pub fn adjoint(self) -> SparseRowMatRef<'a, I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.conjugate().transpose()
	}

	#[inline]
	/// see [`SparseColMatRef::adjoint`]
	pub fn adjoint_mut(self) -> SparseRowMatMut<'a, I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.conjugate_mut().transpose_mut()
	}

	#[inline]
	/// see [`SparseColMatRef::canonical`]
	pub fn canonical(self) -> SparseColMatRef<'a, I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.0.val.len();
		SparseColMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic,
				val: unsafe { core::slice::from_raw_parts(self.0.val.as_ptr() as *const T::Canonical, len) },
			},
		}
	}

	#[inline]
	/// see [`SparseColMatRef::canonical`]
	pub fn canonical_mut(self) -> SparseColMatMut<'a, I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.0.val.len();
		SparseColMatMut {
			0: numeric::Mut {
				symbolic: self.0.symbolic,
				val: unsafe { core::slice::from_raw_parts_mut(self.0.val.as_mut_ptr() as *mut T::Canonical, len) },
			},
		}
	}

	#[inline]
	/// see [`SparseColMatRef::to_row_major`]
	pub fn to_row_major(&self) -> Result<SparseRowMat<I, T::Canonical, Rows, Cols>, FaerError>
	where
		T: Conjugate,
	{
		self.rb().to_row_major()
	}

	#[inline]
	/// see [`SparseColMatRef::to_dense`]
	pub fn to_dense(&self) -> Mat<T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		self.rb().to_dense()
	}

	/// see [`SparseColMatRef::triplet_iter`]
	#[inline]
	pub fn triplet_iter(self) -> impl 'a + Iterator<Item = Triplet<Idx<Rows>, Idx<Cols>, &'a T>>
	where
		Rows: 'a,
		Cols: 'a,
	{
		self.into_const().triplet_iter()
	}
}

impl<Rows: Shape, Cols: Shape, I: Index, T> SparseColMat<I, T, Rows, Cols> {
	#[inline]
	#[track_caller]
	/// see [`SparseColMatRef::new`]
	pub fn new(symbolic: SymbolicSparseColMat<I, Rows, Cols>, val: alloc::vec::Vec<T>) -> Self {
		assert!(symbolic.row_idx().len() == val.len());
		Self {
			0: numeric::Own { symbolic, val },
		}
	}

	#[inline]
	/// see [`SparseColMatRef::parts`]
	pub fn parts(&self) -> (SymbolicSparseColMatRef<'_, I, Rows, Cols>, &'_ [T]) {
		(self.0.symbolic.rb(), &self.0.val)
	}

	#[inline]
	/// see [`SparseColMatRef::parts`]
	pub fn parts_mut(&mut self) -> (SymbolicSparseColMatRef<'_, I, Rows, Cols>, &'_ mut [T]) {
		(self.0.symbolic.rb(), &mut self.0.val)
	}

	#[inline]
	/// see [`SparseColMatRef::parts`]
	pub fn into_parts(self) -> (SymbolicSparseColMat<I, Rows, Cols>, alloc::vec::Vec<T>) {
		(self.0.symbolic, self.0.val)
	}

	#[inline]
	/// see [`SparseColMatRef::symbolic`]
	pub fn symbolic(&self) -> SymbolicSparseColMatRef<'_, I, Rows, Cols> {
		self.0.symbolic.rb()
	}

	#[inline]
	/// see [`SparseColMatRef::val`]
	pub fn val(&self) -> &'_ [T] {
		&self.0.val
	}

	#[inline]
	/// see [`SparseColMatRef::val`]
	pub fn val_mut(&mut self) -> &'_ mut [T] {
		&mut self.0.val
	}

	#[inline]
	#[track_caller]
	/// see [`SparseColMatRef::val_of_col`]
	pub fn val_of_col(&self, j: Idx<Cols>) -> &'_ [T] {
		unsafe { self.0.val.get_unchecked(self.col_range(j)) }
	}

	#[inline]
	#[track_caller]
	/// see [`SparseColMatRef::val_of_col`]
	pub fn val_of_col_mut(&mut self, j: Idx<Cols>) -> &'_ mut [T] {
		unsafe { self.0.val.get_unchecked_mut(self.0.symbolic.col_range(j)) }
	}

	#[inline]
	#[track_caller]
	/// see [`SparseColMatRef::as_shape`]
	pub fn as_shape<V: Shape, H: Shape>(&self, nrows: V, ncols: H) -> SparseColMatRef<'_, I, T, V, H> {
		SparseColMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic.as_shape(nrows, ncols),
				val: &self.0.val,
			},
		}
	}

	#[inline]
	#[track_caller]
	/// see [`SparseColMatRef::as_shape`]
	pub fn as_shape_mut<V: Shape, H: Shape>(&mut self, nrows: V, ncols: H) -> SparseColMatMut<'_, I, T, V, H> {
		SparseColMatMut {
			0: numeric::Mut {
				symbolic: self.0.symbolic.as_shape(nrows, ncols),
				val: &mut self.0.val,
			},
		}
	}

	#[inline]
	#[track_caller]
	/// see [`SparseColMatRef::as_shape`]
	pub fn into_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SparseColMat<I, T, V, H> {
		SparseColMat {
			0: numeric::Own {
				symbolic: self.0.symbolic.into_shape(nrows, ncols),
				val: self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseColMatRef::as_dyn`]
	pub fn as_dyn(&self) -> SparseColMatRef<'_, I, T> {
		SparseColMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic.as_dyn(),
				val: &self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseColMatRef::as_dyn`]
	pub fn as_dyn_mut(&mut self) -> SparseColMatMut<'_, I, T> {
		SparseColMatMut {
			0: numeric::Mut {
				symbolic: self.0.symbolic.as_dyn(),
				val: &mut self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseColMatRef::as_dyn`]
	pub fn into_dyn(self) -> SparseColMat<I, T> {
		SparseColMat {
			0: numeric::Own {
				symbolic: self.0.symbolic.into_dyn(),
				val: self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseColMatRef::as_ref`]
	pub fn as_ref(&self) -> SparseColMatRef<'_, I, T, Rows, Cols> {
		SparseColMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic.as_ref(),
				val: &self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseColMatRef::transpose`]
	pub fn transpose(&self) -> SparseRowMatRef<'_, I, T, Cols, Rows> {
		SparseRowMatRef {
			0: super::csr::numeric::Ref {
				symbolic: self.0.symbolic.transpose(),
				val: &self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseColMatRef::transpose`]
	pub fn transpose_mut(&mut self) -> SparseRowMatMut<'_, I, T, Cols, Rows> {
		SparseRowMatMut {
			0: super::csr::numeric::Mut {
				symbolic: self.0.symbolic.transpose(),
				val: &mut self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseColMatRef::transpose`]
	pub fn into_transpose(self) -> SparseRowMat<I, T, Cols, Rows> {
		SparseRowMat {
			0: super::csr::numeric::Own {
				symbolic: self.0.symbolic.into_transpose(),
				val: self.0.val,
			},
		}
	}

	#[inline]
	/// see [`SparseColMatRef::conjugate`]
	pub fn conjugate(&self) -> SparseColMatRef<'_, I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		self.rb().conjugate()
	}

	#[inline]
	/// see [`SparseColMatRef::conjugate`]
	pub fn conjugate_mut(&mut self) -> SparseColMatMut<'_, I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		self.rb_mut().conjugate_mut()
	}

	#[inline]
	/// see [`SparseColMatRef::conjugate`]
	pub fn into_conjugate(self) -> SparseColMat<I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		let mut vec = core::mem::ManuallyDrop::new(self.0.val);
		let len = vec.len();
		let cap = vec.capacity();
		let ptr = vec.as_mut_ptr();

		SparseColMat {
			0: numeric::Own {
				symbolic: self.0.symbolic,
				val: unsafe { alloc::vec::Vec::from_raw_parts(ptr as *mut T::Conj, len, cap) },
			},
		}
	}

	#[inline]
	/// see [`SparseColMatRef::adjoint`]
	pub fn adjoint(&self) -> SparseRowMatRef<'_, I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.conjugate().transpose()
	}

	#[inline]
	/// see [`SparseColMatRef::adjoint`]
	pub fn adjoint_mut(&mut self) -> SparseRowMatMut<'_, I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.conjugate_mut().transpose_mut()
	}

	#[inline]
	/// see [`SparseColMatRef::adjoint`]
	pub fn into_adjoint(self) -> SparseRowMat<I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.into_conjugate().into_transpose()
	}

	#[inline]
	/// see [`SparseColMatRef::canonical`]
	pub fn canonical(&self) -> SparseColMatRef<'_, I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.0.val.len();
		SparseColMatRef {
			0: numeric::Ref {
				symbolic: self.0.symbolic.rb(),
				val: unsafe { core::slice::from_raw_parts(self.0.val.as_ptr() as *const T::Canonical, len) },
			},
		}
	}

	#[inline]
	/// see [`SparseColMatRef::canonical`]
	pub fn canonical_mut(&mut self) -> SparseColMatMut<'_, I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.0.val.len();
		SparseColMatMut {
			0: numeric::Mut {
				symbolic: self.0.symbolic.rb(),
				val: unsafe { core::slice::from_raw_parts_mut(self.0.val.as_mut_ptr() as *mut T::Canonical, len) },
			},
		}
	}

	#[inline]
	/// see [`SparseColMatRef::canonical`]
	pub fn into_canonical(self) -> SparseColMat<I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let mut vec = core::mem::ManuallyDrop::new(self.0.val);
		let len = vec.len();
		let cap = vec.capacity();
		let ptr = vec.as_mut_ptr();

		SparseColMat {
			0: numeric::Own {
				symbolic: self.0.symbolic,
				val: unsafe { alloc::vec::Vec::from_raw_parts(ptr as *mut T::Canonical, len, cap) },
			},
		}
	}

	#[inline]
	/// see [`SparseColMatRef::to_row_major`]
	pub fn to_row_major(&self) -> Result<SparseRowMat<I, T::Canonical, Rows, Cols>, FaerError>
	where
		T: Conjugate,
	{
		self.rb().to_row_major()
	}

	#[inline]
	/// see [`SparseColMatRef::to_dense`]
	pub fn to_dense(&self) -> Mat<T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		self.rb().to_dense()
	}

	#[track_caller]
	pub(crate) fn new_from_argsort_impl(
		symbolic: SymbolicSparseColMat<I, Rows, Cols>,
		argsort: &Argsort<I>,
		all_values: impl Fn(usize) -> T,
		values_len: usize,
	) -> Result<Self, FaerError>
	where
		T: ComplexField,
	{
		{
			let nnz = argsort.idx.len();
			assert!(values_len == nnz);
		}

		let all_nnz = argsort.all_nnz;
		let mut val = alloc::vec::Vec::new();

		if val.try_reserve_exact(argsort.nnz).is_err() {
			return Err(FaerError::OutOfMemory);
		}

		let mut pos = 0usize;
		let mut pos_unique = usize::MAX;
		let top_bit = I::truncate(1 << (I::BITS - 1));

		while pos < all_nnz {
			let argsort_pos = argsort.idx[pos];
			let extracted_bit = argsort_pos & top_bit;
			let argsort_pos = (argsort_pos & !top_bit).zx();

			let v = all_values(argsort_pos);
			if extracted_bit != I::truncate(0) {
				val[pos_unique] = add(&val[pos_unique], &v);
			} else {
				val.push(v);
				pos_unique = pos_unique.wrapping_add(1);
			}

			pos += 1;
		}

		Ok(Self {
			0: numeric::Own { symbolic, val },
		})
	}

	/// create a new matrix from a previously created symbolic structure and value order
	///
	/// the provided values must correspond to the same indices that were provided in the
	/// function call from which the order was created
	#[track_caller]
	pub fn new_from_argsort(symbolic: SymbolicSparseColMat<I, Rows, Cols>, argsort: &Argsort<I>, val: &[T]) -> Result<Self, FaerError>
	where
		T: ComplexField,
	{
		Self::new_from_argsort_impl(symbolic, argsort, |i| val[i].clone(), val.len())
	}

	/// create a new matrix from triplets
	#[track_caller]
	pub fn try_new_from_triplets(nrows: Rows, ncols: Cols, entries: &[Triplet<Idx<Rows, I>, Idx<Cols, I>, T>]) -> Result<Self, CreationError>
	where
		T: ComplexField,
	{
		let (symbolic, argsort) = SymbolicSparseColMat::try_new_from_indices_impl(
			nrows,
			ncols,
			|i| Pair {
				row: entries[i].row,
				col: entries[i].col,
			},
			|_, _| false,
			entries.len(),
		)?;

		Ok(Self::new_from_argsort_impl(
			symbolic,
			&argsort,
			|i| entries[i].val.clone(),
			entries.len(),
		)?)
	}

	/// create a new matrix from triplets
	///
	/// negative indices are ignored
	#[track_caller]
	pub fn try_new_from_nonnegative_triplets(
		nrows: Rows,
		ncols: Cols,
		entries: &[Triplet<MaybeIdx<Rows, I>, MaybeIdx<Cols, I>, T>],
	) -> Result<Self, CreationError>
	where
		T: ComplexField,
	{
		let (symbolic, argsort) = SymbolicSparseColMat::try_new_from_indices_impl(
			nrows,
			ncols,
			|i| Pair {
				row: unsafe { Idx::<Rows, I>::new_unbound(I::from_signed(entries[i].row.unbound())) },
				col: unsafe { Idx::<Cols, I>::new_unbound(I::from_signed(entries[i].col.unbound())) },
			},
			|row, col| {
				let row = row.unbound().to_signed();
				let col = col.unbound().to_signed();
				let zero = I::Signed::truncate(0);

				row < zero || col < zero
			},
			entries.len(),
		)?;

		Ok(Self::new_from_argsort_impl(
			symbolic,
			&argsort,
			|i| entries[i].val.clone(),
			entries.len(),
		)?)
	}

	/// see [`SparseColMatRef::get`]
	#[track_caller]
	#[inline]
	pub fn get(&self, row: Idx<Rows>, col: Idx<Cols>) -> Option<&T> {
		self.rb().get(row, col)
	}

	/// see [`SparseColMatRef::get`]
	#[track_caller]
	#[inline]
	pub fn get_mut(&mut self, row: Idx<Rows>, col: Idx<Cols>) -> Option<&mut T> {
		self.rb_mut().get_mut(row, col)
	}

	/// see [`SparseColMatRef::triplet_iter`]
	#[inline]
	pub fn triplet_iter(&self) -> impl '_ + Iterator<Item = Triplet<Idx<Rows>, Idx<Cols>, &'_ T>> {
		self.rb().triplet_iter()
	}
}

impl<'a, Rows: Shape, Cols: Shape, I: Index, T> core::ops::Deref for SparseColMatRef<'a, I, T, Rows, Cols> {
	type Target = SymbolicSparseColMatRef<'a, I, Rows, Cols>;

	#[inline]
	fn deref(&self) -> &Self::Target {
		&self.0.symbolic
	}
}

impl<'a, Rows: Shape, Cols: Shape, I: Index, T> core::ops::Deref for SparseColMatMut<'a, I, T, Rows, Cols> {
	type Target = SymbolicSparseColMatRef<'a, I, Rows, Cols>;

	#[inline]
	fn deref(&self) -> &Self::Target {
		&self.0.symbolic
	}
}

impl<Rows: Shape, Cols: Shape, I: Index, T> core::ops::Deref for SparseColMat<I, T, Rows, Cols> {
	type Target = SymbolicSparseColMat<I, Rows, Cols>;

	#[inline]
	fn deref(&self) -> &Self::Target {
		&self.0.symbolic
	}
}

impl<Rows: Shape, Cols: Shape, I: Index> fmt::Debug for symbolic::Ref<'_, I, Rows, Cols> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		fn imp<'ROWS, 'COLS, I: Index>(mat: SymbolicSparseColMatRef<'_, I, Dim<'ROWS>, Dim<'COLS>>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
			struct Entries<'a, 'ROWS, 'COLS, I>(SymbolicSparseColMatRef<'a, I, Dim<'ROWS>, Dim<'COLS>>);

			impl<'ROWS, 'COLS, I: Index> fmt::Debug for Entries<'_, 'ROWS, 'COLS, I> {
				fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
					let mat = self.0;

					f.debug_list()
						.entries(
							mat.ncols()
								.indices()
								.flat_map(|col| mat.row_idx_of_col(col).map(move |row| Pair { row, col })),
						)
						.finish()
				}
			}

			f.debug_struct("SymbolicSparseColMat")
				.field("nrows", &mat.nrows)
				.field("ncols", &mat.ncols)
				.field("entries", &Entries(mat))
				.finish()
		}

		let this = symbolic::generic::SymbolicSparseColMat::from_inner_ref(self);
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

impl<Rows: Shape, Cols: Shape, I: Index, T: fmt::Debug> fmt::Debug for numeric::Ref<'_, I, T, Rows, Cols> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		fn imp<'ROWS, 'COLS, I: Index, T: fmt::Debug>(
			mat: SparseColMatRef<'_, I, T, Dim<'ROWS>, Dim<'COLS>>,
			f: &mut fmt::Formatter<'_>,
		) -> fmt::Result {
			struct Entries<'a, 'ROWS, 'COLS, I, T>(SparseColMatRef<'a, I, T, Dim<'ROWS>, Dim<'COLS>>);

			impl<'ROWS, 'COLS, I: Index, T: fmt::Debug> fmt::Debug for Entries<'_, 'ROWS, 'COLS, I, T> {
				fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
					let mat = self.0;

					f.debug_list()
						.entries(mat.ncols().indices().flat_map(|col| {
							let row_idx = mat.row_idx_of_col(col);
							let val = mat.val_of_col(col);

							iter::zip(row_idx, val).map(move |(row, val)| Triplet {
								row,
								col,
								val: crate::hacks::hijack_debug(val),
							})
						}))
						.finish()
				}
			}

			f.debug_struct("SparseColMat")
				.field("nrows", &mat.nrows)
				.field("ncols", &mat.ncols)
				.field("entries", &Entries(mat))
				.finish()
		}

		let this = numeric::generic::SparseColMat::from_inner_ref(self);
		with_dim!(ROWS, this.nrows().unbound());
		with_dim!(COLS, this.ncols().unbound());

		imp(this.as_shape(ROWS, COLS), f)
	}
}

impl<Rows: Shape, Cols: Shape, I: Index, T: fmt::Debug> fmt::Debug for SparseColMatMut<'_, I, T, Rows, Cols> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.rb().fmt(f)
	}
}

impl<Rows: Shape, Cols: Shape, I: Index, T: fmt::Debug> fmt::Debug for SparseColMat<I, T, Rows, Cols> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.rb().fmt(f)
	}
}
