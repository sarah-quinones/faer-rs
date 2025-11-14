use crate::{Index, Shape, ShapeIdx, SignedIndex, Unbind, assert};
use core::fmt;
use core::marker::PhantomData;
use core::ops::Range;
use generativity::Guard;
type Invariant<'a> = fn(&'a ()) -> &'a ();
/// splits a range into two segments.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Partition<'head, 'tail, 'n> {
	/// size of the first half.
	pub head: Dim<'head>,
	/// size of the second half.
	pub tail: Dim<'tail>,
	__marker: PhantomData<Invariant<'n>>,
}
impl<'head, 'tail, 'n> Partition<'head, 'tail, 'n> {
	/// returns the midpoint of the partition.
	#[inline]
	pub const fn midpoint(&self) -> IdxInc<'n> {
		unsafe { IdxInc::new_unbound(self.head.unbound) }
	}

	/// returns the midpoint of the partition.
	#[inline]
	pub const fn flip(&self) -> Partition<'tail, 'head, 'n> {
		Partition {
			head: self.tail,
			tail: self.head,
			__marker: PhantomData,
		}
	}
}
/// lifetime branded length
/// # safety
/// the type's safety invariant is that all instances of this type with the same
/// lifetime correspond to the same length.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Dim<'n> {
	unbound: usize,
	__marker: PhantomData<Invariant<'n>>,
}
impl PartialEq for Dim<'_> {
	#[inline(always)]
	fn eq(&self, other: &Self) -> bool {
		equator::debug_assert!(self.unbound == other.unbound);
		true
	}
}
impl Eq for Dim<'_> {}
impl PartialOrd for Dim<'_> {
	#[inline(always)]
	fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
		equator::debug_assert!(self.unbound == other.unbound);
		Some(core::cmp::Ordering::Equal)
	}
}
impl Ord for Dim<'_> {
	#[inline(always)]
	fn cmp(&self, other: &Self) -> core::cmp::Ordering {
		equator::debug_assert!(self.unbound == other.unbound);
		core::cmp::Ordering::Equal
	}
}
/// lifetime branded index.
/// # safety
/// the type's safety invariant is that all instances of this type are valid
/// indices for [`Dim<'n>`] and less than or equal to `i::signed::max`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Idx<'n, I: Index = usize> {
	unbound: I,
	__marker: PhantomData<Invariant<'n>>,
}
/// lifetime branded partition index.
/// # safety
/// the type's safety invariant is that all instances of this type are valid
/// partition places for [`Dim<'n>`] and less than or equal to `i::signed::max`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct IdxInc<'n, I: Index = usize> {
	unbound: I,
	__marker: PhantomData<Invariant<'n>>,
}
impl fmt::Debug for Dim<'_> {
	#[inline]
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.unbound.fmt(f)
	}
}
impl<I: Index> fmt::Debug for Idx<'_, I> {
	#[inline]
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.unbound.fmt(f)
	}
}
impl<I: Index> fmt::Debug for IdxInc<'_, I> {
	#[inline]
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.unbound.fmt(f)
	}
}
impl<I: Index> fmt::Debug for MaybeIdx<'_, I> {
	#[inline]
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		if self.unbound.to_signed() >= I::Signed::truncate(0) {
			self.unbound.fmt(f)
		} else {
			f.write_str("None")
		}
	}
}
impl<'n, I: Index> PartialEq<Dim<'n>> for Idx<'n, I> {
	#[inline(always)]
	fn eq(&self, other: &Dim<'n>) -> bool {
		equator::debug_assert!(self.unbound.zx() < other.unbound);
		false
	}
}
impl<'n, I: Index> PartialOrd<Dim<'n>> for Idx<'n, I> {
	#[inline(always)]
	fn partial_cmp(&self, other: &Dim<'n>) -> Option<core::cmp::Ordering> {
		equator::debug_assert!(self.unbound.zx() < other.unbound);
		Some(core::cmp::Ordering::Less)
	}
}
impl<'n, I: Index> PartialEq<Dim<'n>> for IdxInc<'n, I> {
	#[inline(always)]
	fn eq(&self, other: &Dim<'n>) -> bool {
		equator::debug_assert!(self.unbound.zx() <= other.unbound);
		self.unbound.zx() == other.unbound
	}
}
impl<'n, I: Index> PartialOrd<Dim<'n>> for IdxInc<'n, I> {
	#[inline(always)]
	fn partial_cmp(&self, other: &Dim<'n>) -> Option<core::cmp::Ordering> {
		equator::debug_assert!(self.unbound.zx() <= other.unbound);
		Some(if self.unbound.zx() == other.unbound {
			core::cmp::Ordering::Equal
		} else {
			core::cmp::Ordering::Less
		})
	}
}
impl<'n> Dim<'n> {
	/// create new branded value with the value `dim`.
	#[inline(always)]
	pub fn with<R>(dim: usize, f: impl for<'dim> FnOnce(Dim<'dim>) -> R) -> R {
		f(unsafe { Self::new_unbound(dim) })
	}

	/// create new branded value with an arbitrary brand.
	/// # safety
	/// see struct safety invariant.
	#[inline(always)]
	pub const unsafe fn new_unbound(dim: usize) -> Self {
		Self {
			unbound: dim,
			__marker: PhantomData,
		}
	}

	/// create new branded value with a unique brand.
	#[inline(always)]
	pub fn new(dim: usize, guard: Guard<'n>) -> Self {
		_ = guard;
		Self {
			unbound: dim,
			__marker: PhantomData,
		}
	}

	/// returns the unconstrained value.
	#[inline(always)]
	pub const fn unbound(self) -> usize {
		self.unbound
	}

	/// partitions `self` into two segments as specifiedd by the midpoint.
	#[inline]
	pub const fn partition<'head, 'tail>(
		self,
		midpoint: IdxInc<'n>,
		head: Guard<'head>,
		tail: Guard<'tail>,
	) -> Partition<'head, 'tail, 'n> {
		_ = (head, tail);
		unsafe {
			Partition {
				head: Dim::new_unbound(midpoint.unbound),
				tail: Dim::new_unbound(self.unbound - midpoint.unbound),
				__marker: PhantomData,
			}
		}
	}

	/// partitions `self` into two segments.
	#[inline]
	#[track_caller]
	pub fn head_partition<'head, 'tail>(
		self,
		head: Dim<'head>,
		tail: Guard<'tail>,
	) -> Partition<'head, 'tail, 'n> {
		_ = (head, tail);
		let midpoint = IdxInc::new_checked(head.unbound(), self);
		unsafe {
			Partition {
				head,
				tail: Dim::new_unbound(self.unbound - midpoint.unbound),
				__marker: PhantomData,
			}
		}
	}

	/// returns `start` advanced by `len` units, saturated to `self`
	#[inline]
	pub fn advance(self, start: Idx<'n>, len: usize) -> IdxInc<'n> {
		let len = Ord::min(self.unbound.saturating_sub(start.unbound), len);
		IdxInc {
			unbound: start.unbound + len,
			__marker: PhantomData,
		}
	}

	/// returns an iterator over the indices between `0` and `self`.
	#[inline]
	pub fn indices(
		self,
	) -> impl Clone + ExactSizeIterator + DoubleEndedIterator<Item = Idx<'n>> {
		(0..self.unbound).map(|i| unsafe { Idx::new_unbound(i) })
	}

	/// returns an iterator over the indices between `0` and `self`.
	#[inline]
	#[cfg(feature = "rayon")]
	pub fn par_indices(
		self,
	) -> impl rayon::iter::IndexedParallelIterator<Item = Idx<'n>> {
		use rayon::prelude::*;
		(0..self.unbound)
			.into_par_iter()
			.map(|i| unsafe { Idx::new_unbound(i) })
	}
}
impl<'n, I: Index> Idx<'n, I> {
	/// create new branded value with an arbitrary brand.
	/// # safety
	/// see struct safety invariant.
	#[inline(always)]
	pub const unsafe fn new_unbound(idx: I) -> Self {
		Self {
			unbound: idx,
			__marker: PhantomData,
		}
	}

	/// create new branded value with the same brand as `Dim`.
	/// # safety
	/// the behavior is undefined unless `idx < dim` and `idx <=
	/// i::signed::max`.
	#[inline(always)]
	pub unsafe fn new_unchecked(idx: I, dim: Dim<'n>) -> Self {
		equator::debug_assert!(all(
			idx.zx() < dim.unbound,
			idx <= I::from_signed(I::Signed::MAX),
		));
		Self {
			unbound: idx,
			__marker: PhantomData,
		}
	}

	/// create new branded value with the same brand as `Dim`.
	/// # panics
	/// panics unless `idx < dim` and `idx <= i::signed::max`.
	#[inline(always)]
	#[track_caller]
	pub fn new_checked(idx: I, dim: Dim<'n>) -> Self {
		equator::assert!(all(
			idx.zx() < dim.unbound,
			idx <= I::from_signed(I::Signed::MAX),
		));
		Self {
			unbound: idx,
			__marker: PhantomData,
		}
	}

	/// returns the unconstrained value.
	#[inline(always)]
	pub const fn unbound(self) -> I {
		self.unbound
	}

	/// zero-extends the internal value into a `usize`.
	#[inline(always)]
	pub fn zx(self) -> Idx<'n> {
		Idx {
			unbound: self.unbound.zx(),
			__marker: PhantomData,
		}
	}
}
impl<'n> IdxInc<'n> {
	/// zero index
	pub const ZERO: Self = unsafe { Self::new_unbound(0) };
}
impl<'n, I: Index> IdxInc<'n, I> {
	/// create new branded value with an arbitrary brand.
	/// # safety
	/// see struct safety invariant.
	#[inline(always)]
	pub const unsafe fn new_unbound(idx: I) -> Self {
		Self {
			unbound: idx,
			__marker: PhantomData,
		}
	}

	/// create new branded value with the same brand as `Dim`.
	/// # safety
	/// the behavior is undefined unless `idx <= dim`.
	#[inline(always)]
	pub unsafe fn new_unchecked(idx: I, dim: Dim<'n>) -> Self {
		equator::debug_assert!(all(
			idx.zx() <= dim.unbound,
			idx <= I::from_signed(I::Signed::MAX),
		));
		Self {
			unbound: idx,
			__marker: PhantomData,
		}
	}

	/// create new branded value with the same brand as `Dim`.
	/// # panics
	/// panics unless `idx <= dim`.
	#[inline(always)]
	#[track_caller]
	pub fn new_checked(idx: I, dim: Dim<'n>) -> Self {
		equator::assert!(all(
			idx.zx() <= dim.unbound,
			idx <= I::from_signed(I::Signed::MAX),
		));
		Self {
			unbound: idx,
			__marker: PhantomData,
		}
	}

	/// returns the unconstrained value.
	#[inline(always)]
	pub const fn unbound(self) -> I {
		self.unbound
	}

	/// zero-extends the internal value into a `usize`.
	#[inline(always)]
	pub fn zx(self) -> IdxInc<'n> {
		IdxInc {
			unbound: self.unbound.zx(),
			__marker: PhantomData,
		}
	}
}
impl<'n> IdxInc<'n> {
	/// returns an iterator over the indices between `self` and `to`.
	#[inline]
	pub fn to(
		self,
		upper: IdxInc<'n>,
	) -> impl Clone + ExactSizeIterator + DoubleEndedIterator<Item = Idx<'n>> {
		(self.unbound..upper.unbound).map(|i| unsafe { Idx::new_unbound(i) })
	}

	/// returns an iterator over the indices between `self` and `to`.
	#[inline]
	pub fn range_to(
		self,
		upper: IdxInc<'n>,
	) -> impl Clone + ExactSizeIterator + DoubleEndedIterator<Item = Idx<'n>> {
		(self.unbound..upper.unbound).map(|i| unsafe { Idx::new_unbound(i) })
	}
}
impl Unbind for Dim<'_> {
	#[inline(always)]
	unsafe fn new_unbound(idx: usize) -> Self {
		Self::new_unbound(idx)
	}

	#[inline(always)]
	fn unbound(self) -> usize {
		self.unbound
	}
}
impl<I: Index> Unbind<I> for Idx<'_, I> {
	#[inline(always)]
	unsafe fn new_unbound(idx: I) -> Self {
		Self::new_unbound(idx)
	}

	#[inline(always)]
	fn unbound(self) -> I {
		self.unbound
	}
}
impl<I: Index> Unbind<I> for IdxInc<'_, I> {
	#[inline(always)]
	unsafe fn new_unbound(idx: I) -> Self {
		Self::new_unbound(idx)
	}

	#[inline(always)]
	fn unbound(self) -> I {
		self.unbound
	}
}
impl<I: Index> Unbind<I::Signed> for MaybeIdx<'_, I> {
	#[inline(always)]
	unsafe fn new_unbound(idx: I::Signed) -> Self {
		Self::new_unbound(I::from_signed(idx))
	}

	#[inline(always)]
	fn unbound(self) -> I::Signed {
		self.unbound.to_signed()
	}
}
impl<'dim> ShapeIdx for Dim<'dim> {
	type Idx<I: Index> = Idx<'dim, I>;
	type IdxInc<I: Index> = IdxInc<'dim, I>;
	type MaybeIdx<I: Index> = MaybeIdx<'dim, I>;
}
impl<'dim> Shape for Dim<'dim> {}
impl<'n, I: Index> From<Idx<'n, I>> for IdxInc<'n, I> {
	#[inline(always)]
	fn from(value: Idx<'n, I>) -> Self {
		Self {
			unbound: value.unbound,
			__marker: PhantomData,
		}
	}
}
impl<'n> From<Dim<'n>> for IdxInc<'n> {
	#[inline(always)]
	fn from(value: Dim<'n>) -> Self {
		Self {
			unbound: value.unbound,
			__marker: PhantomData,
		}
	}
}
impl<'n, I: Index> From<Idx<'n, I>> for MaybeIdx<'n, I> {
	#[inline(always)]
	fn from(value: Idx<'n, I>) -> Self {
		Self {
			unbound: value.unbound,
			__marker: PhantomData,
		}
	}
}
impl<'size> Dim<'size> {
	/// check that the index is bounded by `self`, or panic otherwise.
	#[track_caller]
	#[inline]
	pub fn check<I: Index>(self, idx: I) -> Idx<'size, I> {
		Idx::new_checked(idx, self)
	}

	/// check that the index is bounded by `self`, or panic otherwise.
	#[track_caller]
	#[inline]
	pub fn idx<I: Index>(self, idx: I) -> Idx<'size, I> {
		Idx::new_checked(idx, self)
	}

	/// check that the index is bounded by `self`, or panic otherwise.
	#[track_caller]
	#[inline]
	pub fn idx_inc<I: Index>(self, idx: I) -> IdxInc<'size, I> {
		IdxInc::new_checked(idx, self)
	}

	/// check that the index is bounded by `self`, or return `none` otherwise.
	#[inline]
	pub fn try_check<I: Index>(self, idx: I) -> Option<Idx<'size, I>> {
		if idx.zx() < self.unbound() {
			Some(unsafe { Idx::new_unbound(idx) })
		} else {
			None
		}
	}
}
impl<'n> Idx<'n> {
	/// truncate `self` to a smaller type `i`.
	pub fn truncate<I: Index>(self) -> Idx<'n, I> {
		unsafe { Idx::new_unbound(I::truncate(self.unbound())) }
	}
}
impl<'n, I: Index> Idx<'n, I> {
	/// returns the index, bounded inclusively by the value tied to `'n`.
	#[inline]
	pub const fn to_incl(self) -> IdxInc<'n, I> {
		unsafe { IdxInc::new_unbound(self.unbound()) }
	}

	/// returns the next index, bounded inclusively by the value tied to `'n`.
	#[inline]
	pub fn next(self) -> IdxInc<'n, I> {
		unsafe { IdxInc::new_unbound(self.unbound() + I::truncate(1)) }
	}

	/// returns the index, bounded inclusively by the value tied to `'n`.
	#[inline]
	pub fn excl(self) -> IdxInc<'n, I> {
		unsafe { IdxInc::new_unbound(self.unbound()) }
	}

	/// assert that the values of `slice` are all bounded by `size`.
	#[track_caller]
	#[inline]
	pub fn from_slice_mut_checked<'a>(
		slice: &'a mut [I],
		size: Dim<'n>,
	) -> &'a mut [Idx<'n, I>] {
		Self::from_slice_ref_checked(slice, size);
		unsafe { &mut *(slice as *mut _ as *mut _) }
	}

	/// assume that the values of `slice` are all bounded by the value tied to
	/// `'n`.
	#[track_caller]
	#[inline]
	pub unsafe fn from_slice_mut_unchecked<'a>(
		slice: &'a mut [I],
	) -> &'a mut [Idx<'n, I>] {
		unsafe { &mut *(slice as *mut _ as *mut _) }
	}

	/// assert that the values of `slice` are all bounded by `size`.
	#[track_caller]
	pub fn from_slice_ref_checked<'a>(
		slice: &'a [I],
		size: Dim<'n>,
	) -> &'a [Idx<'n, I>] {
		for &idx in slice {
			Self::new_checked(idx, size);
		}
		unsafe { &*(slice as *const _ as *const _) }
	}

	/// assume that the values of `slice` are all bounded by the value tied to
	/// `'n`.
	#[track_caller]
	#[inline]
	pub unsafe fn from_slice_ref_unchecked<'a>(
		slice: &'a [I],
	) -> &'a [Idx<'n, I>] {
		unsafe { &*(slice as *const _ as *const _) }
	}
}
/// `i` value smaller than the size corresponding to the lifetime `'n`, or
/// `none`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct MaybeIdx<'n, I: Index = usize> {
	unbound: I,
	__marker: PhantomData<Invariant<'n>>,
}
impl<'n, I: Index> MaybeIdx<'n, I> {
	/// returns an index value.
	#[inline]
	pub fn from_index(idx: Idx<'n, I>) -> Self {
		unsafe { Self::new_unbound(idx.unbound()) }
	}

	/// returns a `none` value.
	#[inline]
	pub fn none() -> Self {
		unsafe { Self::new_unbound(I::truncate(usize::MAX)) }
	}

	/// returns a constrained index value if `idx` is nonnegative, `none`
	/// otherwise.
	#[inline]
	#[track_caller]
	pub fn new_checked(idx: I::Signed, size: Dim<'n>) -> Self {
		assert!((idx.sx() as isize) < size.unbound() as isize);
		Self {
			unbound: I::from_signed(idx),
			__marker: PhantomData,
		}
	}

	/// returns a constrained index value if `idx` is nonnegative, `none`
	/// otherwise.
	#[inline]
	pub unsafe fn new_unchecked(idx: I::Signed, size: Dim<'n>) -> Self {
		debug_assert!((idx.sx() as isize) < size.unbound() as isize);
		Self {
			unbound: I::from_signed(idx),
			__marker: PhantomData,
		}
	}

	/// returns a constrained index value if `idx` is nonnegative, `none`
	/// otherwise.
	#[inline]
	pub unsafe fn new_unbound(idx: I) -> Self {
		Self {
			unbound: idx,
			__marker: PhantomData,
		}
	}

	/// returns the inner value.
	#[inline]
	pub fn unbound(self) -> I {
		self.unbound
	}

	/// returns the index if available, or `none` otherwise.
	#[inline]
	pub fn idx(self) -> Option<Idx<'n, I>> {
		if self.unbound.to_signed() >= I::Signed::truncate(0) {
			Some(unsafe { Idx::new_unbound(self.unbound()) })
		} else {
			None
		}
	}

	/// sign extend the value.
	#[inline]
	pub fn sx(self) -> MaybeIdx<'n> {
		unsafe { MaybeIdx::new_unbound(self.unbound.to_signed().sx()) }
	}

	/// assert that the values of `slice` are all bounded by `size`.
	#[track_caller]
	#[inline]
	pub fn from_slice_mut_checked<'a>(
		slice: &'a mut [I::Signed],
		size: Dim<'n>,
	) -> &'a mut [MaybeIdx<'n, I>] {
		Self::from_slice_ref_checked(slice, size);
		unsafe { &mut *(slice as *mut _ as *mut _) }
	}

	/// assume that the values of `slice` are all bounded by the value tied to
	/// `'n`.
	#[track_caller]
	#[inline]
	pub unsafe fn from_slice_mut_unchecked<'a>(
		slice: &'a mut [I::Signed],
	) -> &'a mut [MaybeIdx<'n, I>] {
		unsafe { &mut *(slice as *mut _ as *mut _) }
	}

	/// assert that the values of `slice` are all bounded by `size`.
	#[track_caller]
	pub fn from_slice_ref_checked<'a>(
		slice: &'a [I::Signed],
		size: Dim<'n>,
	) -> &'a [MaybeIdx<'n, I>] {
		for &idx in slice {
			Self::new_checked(idx, size);
		}
		unsafe { &*(slice as *const _ as *const _) }
	}

	/// convert a constrained slice to an unconstrained one.
	#[track_caller]
	pub fn as_slice_ref<'a>(slice: &'a [MaybeIdx<'n, I>]) -> &'a [I::Signed] {
		unsafe { &*(slice as *const _ as *const _) }
	}

	/// assume that the values of `slice` are all bounded by the value tied to
	/// `'n`.
	#[track_caller]
	#[inline]
	pub unsafe fn from_slice_ref_unchecked<'a>(
		slice: &'a [I::Signed],
	) -> &'a [MaybeIdx<'n, I>] {
		unsafe { &*(slice as *const _ as *const _) }
	}
}
impl core::ops::Deref for Dim<'_> {
	type Target = usize;

	#[inline]
	fn deref(&self) -> &Self::Target {
		&self.unbound
	}
}
impl<I: Index> core::ops::Deref for MaybeIdx<'_, I> {
	type Target = I::Signed;

	#[inline]
	fn deref(&self) -> &Self::Target {
		bytemuck::cast_ref(&self.unbound)
	}
}
impl<I: Index> core::ops::Deref for Idx<'_, I> {
	type Target = I;

	#[inline]
	fn deref(&self) -> &Self::Target {
		&self.unbound
	}
}
impl<I: Index> core::ops::Deref for IdxInc<'_, I> {
	type Target = I;

	#[inline]
	fn deref(&self) -> &Self::Target {
		&self.unbound
	}
}
/// array of length equal to the value tied to `'n`.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Array<'n, T> {
	__marker: PhantomData<Invariant<'n>>,
	unbound: [T],
}
impl<'n, T> Array<'n, T> {
	/// returns a constrained array after checking that its length matches
	/// `size`.
	#[inline]
	#[track_caller]
	pub fn from_ref<'a>(slice: &'a [T], size: Dim<'n>) -> &'a Self {
		assert!(slice.len() == size.unbound());
		unsafe { &*(slice as *const [T] as *const Self) }
	}

	/// returns a constrained array after checking that its length matches
	/// `size`.
	#[inline]
	#[track_caller]
	pub fn from_mut<'a>(slice: &'a mut [T], size: Dim<'n>) -> &'a mut Self {
		assert!(slice.len() == size.unbound());
		unsafe { &mut *(slice as *mut [T] as *mut Self) }
	}

	/// returns the unconstrained slice.
	#[inline]
	#[track_caller]
	pub fn as_ref(&self) -> &[T] {
		unsafe { &*(self as *const _ as *const _) }
	}

	/// returns the unconstrained slice.
	#[inline]
	#[track_caller]
	pub fn as_mut<'a>(&mut self) -> &'a mut [T] {
		unsafe { &mut *(self as *mut _ as *mut _) }
	}

	/// returns the length of `self`.
	#[inline]
	pub fn len(&self) -> Dim<'n> {
		unsafe { Dim::new_unbound(self.unbound.len()) }
	}
}
impl<T: core::fmt::Debug> core::fmt::Debug for Array<'_, T> {
	#[inline]
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		self.unbound.fmt(f)
	}
}
impl<'n, T> core::ops::Index<Range<IdxInc<'n>>> for Array<'n, T> {
	type Output = [T];

	#[track_caller]
	fn index(&self, idx: Range<IdxInc<'n>>) -> &Self::Output {
		#[cfg(debug_assertions)]
		{
			&self.unbound[idx.start.unbound()..idx.end.unbound()]
		}
		#[cfg(not(debug_assertions))]
		unsafe {
			self.unbound
				.get_unchecked(idx.start.unbound()..idx.end.unbound())
		}
	}
}
impl<'n, T> core::ops::IndexMut<Range<IdxInc<'n>>> for Array<'n, T> {
	#[track_caller]
	fn index_mut(&mut self, idx: Range<IdxInc<'n>>) -> &mut Self::Output {
		#[cfg(debug_assertions)]
		{
			&mut self.unbound[idx.start.unbound()..idx.end.unbound()]
		}
		#[cfg(not(debug_assertions))]
		unsafe {
			self.unbound
				.get_unchecked_mut(idx.start.unbound()..idx.end.unbound())
		}
	}
}
impl<'n, T> core::ops::Index<Idx<'n>> for Array<'n, T> {
	type Output = T;

	#[track_caller]
	fn index(&self, idx: Idx<'n>) -> &Self::Output {
		#[cfg(debug_assertions)]
		{
			&self.unbound[idx.unbound()]
		}
		#[cfg(not(debug_assertions))]
		unsafe {
			self.unbound.get_unchecked(idx.unbound())
		}
	}
}
impl<'n, T> core::ops::IndexMut<Idx<'n>> for Array<'n, T> {
	#[track_caller]
	fn index_mut(&mut self, idx: Idx<'n>) -> &mut Self::Output {
		#[cfg(debug_assertions)]
		{
			&mut self.unbound[idx.unbound()]
		}
		#[cfg(not(debug_assertions))]
		unsafe {
			self.unbound.get_unchecked_mut(idx.unbound())
		}
	}
}
/// dimension equal to one
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct One;
/// index equal to zero
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Zero;
/// index equal to zero ro one
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct IdxIncOne<I: Index = usize> {
	inner: I,
}
/// index equal to zero ro one, or a sentinel value
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MaybeIdxOne<I: Index = usize> {
	inner: I,
}
impl<I: Index> Unbind<I> for IdxIncOne<I> {
	#[inline]
	unsafe fn new_unbound(idx: I) -> Self {
		Self { inner: idx }
	}

	#[inline]
	fn unbound(self) -> I {
		self.inner
	}
}
impl<I: Index> Unbind<I::Signed> for MaybeIdxOne<I> {
	#[inline]
	unsafe fn new_unbound(idx: I::Signed) -> Self {
		Self {
			inner: I::from_signed(idx),
		}
	}

	#[inline]
	fn unbound(self) -> I::Signed {
		self.inner.to_signed()
	}
}
impl<I: Index> Unbind<I> for Zero {
	#[inline]
	unsafe fn new_unbound(idx: I) -> Self {
		equator::debug_assert!(idx.zx() == 0);
		Zero
	}

	#[inline]
	fn unbound(self) -> I {
		I::truncate(0)
	}
}
impl Unbind for One {
	#[inline]
	unsafe fn new_unbound(idx: usize) -> Self {
		equator::debug_assert!(idx == 1);
		One
	}

	#[inline]
	fn unbound(self) -> usize {
		1
	}
}
impl<I: Index> From<Zero> for IdxIncOne<I> {
	fn from(_: Zero) -> Self {
		Self {
			inner: I::truncate(0),
		}
	}
}
impl ShapeIdx for One {
	type Idx<I: Index> = Zero;
	type IdxInc<I: Index> = IdxIncOne<I>;
	type MaybeIdx<I: Index> = MaybeIdxOne<I>;
}
impl PartialEq<One> for IdxIncOne {
	#[inline]
	fn eq(&self, _: &One) -> bool {
		self.inner == 1
	}
}
impl PartialOrd<One> for IdxIncOne {
	#[inline]
	fn partial_cmp(&self, _: &One) -> Option<core::cmp::Ordering> {
		if self.inner == 1 {
			Some(core::cmp::Ordering::Equal)
		} else {
			Some(core::cmp::Ordering::Less)
		}
	}
}
impl PartialEq<One> for Zero {
	#[inline]
	fn eq(&self, _: &One) -> bool {
		false
	}
}
impl PartialOrd<One> for Zero {
	#[inline]
	fn partial_cmp(&self, _: &One) -> Option<core::cmp::Ordering> {
		Some(core::cmp::Ordering::Less)
	}
}
impl Shape for One {
	const IS_BOUND: bool = true;
}
