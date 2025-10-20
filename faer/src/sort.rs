use crate::debug_assert;
use core::mem::MaybeUninit;

pub unsafe trait Ptr: Sized + Copy {
	type Item;

	fn get_ptr(ptr: *mut Self::Item) -> Self;

	unsafe fn offset_from(self, origin: Self) -> isize;

	unsafe fn add(self, offset: usize) -> Self;

	unsafe fn sub(self, offset: usize) -> Self;

	unsafe fn read(self) -> Self::Item;

	unsafe fn write(self, item: Self::Item);

	unsafe fn copy_nonoverlapping(src: Self, dst: Self, len: usize);

	unsafe fn reverse(ptr: Self, len: usize);

	unsafe fn swap(a: Self, b: Self) {
		let a_item = a.read();

		let b_item = b.read();

		a.write(b_item);

		b.write(a_item);
	}

	unsafe fn swap_idx(self, i: usize, j: usize) {
		Self::swap(self.add(i), self.add(j));
	}
}

unsafe impl<T> Ptr for *mut T {
	type Item = MaybeUninit<T>;

	#[inline]

	fn get_ptr(ptr: *mut Self::Item) -> Self {
		ptr as *mut T
	}

	#[inline]

	unsafe fn offset_from(self, origin: Self) -> isize {
		self.offset_from(origin)
	}

	#[inline]

	unsafe fn add(self, offset: usize) -> Self {
		self.add(offset)
	}

	#[inline]

	unsafe fn sub(self, offset: usize) -> Self {
		self.sub(offset)
	}

	#[inline]

	unsafe fn read(self) -> Self::Item {
		(self as *mut MaybeUninit<T>).read()
	}

	#[inline]

	unsafe fn write(self, item: Self::Item) {
		(self as *mut MaybeUninit<T>).write(item)
	}

	#[inline]

	unsafe fn copy_nonoverlapping(src: Self, dst: Self, len: usize) {
		core::ptr::copy_nonoverlapping(src, dst, len);
	}

	#[inline]

	unsafe fn reverse(ptr: Self, len: usize) {
		core::slice::from_raw_parts_mut(ptr, len).reverse()
	}
}

unsafe impl<P: Ptr, Q: Ptr> Ptr for (P, Q) {
	type Item = (P::Item, Q::Item);

	#[inline]

	fn get_ptr(ptr: *mut Self::Item) -> Self {
		unsafe {
			(
				P::get_ptr(core::ptr::addr_of_mut!((*ptr).0)),
				Q::get_ptr(core::ptr::addr_of_mut!((*ptr).1)),
			)
		}
	}

	#[inline]

	unsafe fn offset_from(self, origin: Self) -> isize {
		self.0.offset_from(origin.0)
	}

	#[inline]

	unsafe fn add(self, offset: usize) -> Self {
		(self.0.add(offset), self.1.add(offset))
	}

	#[inline]

	unsafe fn sub(self, offset: usize) -> Self {
		(self.0.sub(offset), self.1.sub(offset))
	}

	#[inline]

	unsafe fn read(self) -> Self::Item {
		(self.0.read(), self.1.read())
	}

	#[inline]

	unsafe fn write(self, item: Self::Item) {
		self.0.write(item.0);

		self.1.write(item.1);
	}

	#[inline]

	unsafe fn copy_nonoverlapping(src: Self, dst: Self, len: usize) {
		P::copy_nonoverlapping(src.0, dst.0, len);

		Q::copy_nonoverlapping(src.1, dst.1, len);
	}

	#[inline]

	unsafe fn reverse(ptr: Self, len: usize) {
		P::reverse(ptr.0, len);

		Q::reverse(ptr.1, len);
	}
}

struct InsertionHole<P: Ptr> {
	src: P,
	dest: P,
}

impl<P: Ptr> Drop for InsertionHole<P> {
	#[inline(always)]

	fn drop(&mut self) {
		unsafe {
			P::copy_nonoverlapping(self.src, self.dest, 1);
		}
	}
}

unsafe fn insert_tail<P: Ptr, F>(v: P, v_len: usize, is_less: &mut F)
where
	F: FnMut(P, P) -> bool,
{
	debug_assert!(v_len >= 2);

	let arr_ptr = v;

	let i = v_len - 1;

	unsafe {
		let i_ptr = arr_ptr.add(i);

		if is_less(i_ptr, i_ptr.sub(1)) {
			let tmp = core::mem::ManuallyDrop::new(P::read(i_ptr));

			let tmp = P::get_ptr((&*tmp) as *const P::Item as *mut P::Item);

			let mut hole = InsertionHole {
				src: tmp,
				dest: i_ptr.sub(1),
			};

			P::copy_nonoverlapping(hole.dest, i_ptr, 1);

			for j in (0..(i - 1)).rev() {
				let j_ptr = arr_ptr.add(j);

				if !is_less(tmp, j_ptr) {
					break;
				}

				P::copy_nonoverlapping(j_ptr, hole.dest, 1);

				hole.dest = j_ptr;
			}
		}
	}
}

unsafe fn insert_head<P: Ptr, F>(v: P, v_len: usize, is_less: &mut F)
where
	F: FnMut(P, P) -> bool,
{
	debug_assert!(v_len >= 2);

	unsafe {
		if is_less(v.add(1), v.add(0)) {
			let arr_ptr = v;

			let tmp = core::mem::ManuallyDrop::new(P::read(arr_ptr));

			let tmp = P::get_ptr((&*tmp) as *const P::Item as *mut P::Item);

			let mut hole = InsertionHole {
				src: tmp,
				dest: arr_ptr.add(1),
			};

			P::copy_nonoverlapping(arr_ptr.add(1), arr_ptr.add(0), 1);

			for i in 2..v_len {
				if !is_less(v.add(i), tmp) {
					break;
				}

				P::copy_nonoverlapping(arr_ptr.add(i), arr_ptr.add(i - 1), 1);

				hole.dest = arr_ptr.add(i);
			}
		}
	}
}

#[inline(never)]

pub(super) fn insertion_sort_shift_left<P: Ptr, F: FnMut(P, P) -> bool>(v: P, v_len: usize, offset: usize, is_less: &mut F) {
	let len = v_len;

	core::assert!(offset != 0 && offset <= len);

	for i in offset..len {
		unsafe {
			insert_tail(v, i + 1, is_less);
		}
	}
}

#[inline(never)]

fn insertion_sort_shift_right<P: Ptr, F: FnMut(P, P) -> bool>(v: P, v_len: usize, offset: usize, is_less: &mut F) {
	let len = v_len;

	core::assert!(offset != 0 && offset <= len && len >= 2);

	for i in (0..offset).rev() {
		unsafe {
			insert_head(v.add(i), len - i, is_less);
		}
	}
}

#[cold]

unsafe fn partial_insertion_sort<P: Ptr, F: FnMut(P, P) -> bool>(v: P, v_len: usize, is_less: &mut F) -> bool {
	const MAX_STEPS: usize = 5;

	const SHORTEST_SHIFTING: usize = 50;

	let len = v_len;

	let mut i = 1;

	for _ in 0..MAX_STEPS {
		unsafe {
			while i < len && !is_less(v.add(i), v.add(i - 1)) {
				i += 1;
			}
		}

		if i == len {
			return true;
		}

		if len < SHORTEST_SHIFTING {
			return false;
		}

		v.swap_idx(i - 1, i);

		if i >= 2 {
			insertion_sort_shift_left(v, i, i - 1, is_less);

			insertion_sort_shift_right(v, i, 1, is_less);
		}
	}

	false
}

#[cold]

pub unsafe fn heapsort<P: Ptr, F: FnMut(P, P) -> bool>(v: P, v_len: usize, mut is_less: F) {
	let mut sift_down = |v: P, v_len: usize, mut node| {
		loop {
			let mut child = 2 * node + 1;

			if child >= v_len {
				break;
			}

			if child + 1 < v_len {
				child += is_less(v.add(child), v.add(child + 1)) as usize;
			}

			if !is_less(v.add(node), v.add(child)) {
				break;
			}

			v.swap_idx(node, child);

			node = child;
		}
	};

	for i in (0..v_len / 2).rev() {
		sift_down(v, v_len, i);
	}

	for i in (1..v_len).rev() {
		v.swap_idx(0, i);

		sift_down(v, i, 0);
	}
}

unsafe fn partition_in_blocks<P: Ptr, F: FnMut(P, P) -> bool>(v: P, v_len: usize, pivot: P, is_less: &mut F) -> usize {
	const BLOCK: usize = 128;

	let mut l = v;

	let mut block_l = BLOCK;

	let mut start_l = core::ptr::null_mut();

	let mut end_l = core::ptr::null_mut();

	let mut offsets_l = [core::mem::MaybeUninit::<u8>::uninit(); BLOCK];

	let mut r = unsafe { l.add(v_len) };

	let mut block_r = BLOCK;

	let mut start_r = core::ptr::null_mut();

	let mut end_r = core::ptr::null_mut();

	let mut offsets_r = [core::mem::MaybeUninit::<u8>::uninit(); BLOCK];

	unsafe fn width<P: Ptr>(l: P, r: P) -> usize {
		r.offset_from(l) as usize
	}

	loop {
		let is_done = width(l, r) <= 2 * BLOCK;

		if is_done {
			let mut rem = width(l, r);

			if start_l < end_l || start_r < end_r {
				rem -= BLOCK;
			}

			if start_l < end_l {
				block_r = rem;
			} else if start_r < end_r {
				block_l = rem;
			} else {
				block_l = rem / 2;

				block_r = rem - block_l;
			}

			debug_assert!(block_l <= BLOCK && block_r <= BLOCK);

			debug_assert!(width(l, r) == block_l + block_r);
		}

		if start_l == end_l {
			start_l = offsets_l.as_mut_ptr() as *mut u8;

			end_l = start_l;

			let mut elem = l;

			for i in 0..block_l {
				unsafe {
					*end_l = i as u8;

					end_l = end_l.add(!is_less(elem, pivot) as usize);

					elem = elem.add(1);
				}
			}
		}

		if start_r == end_r {
			start_r = offsets_r.as_mut_ptr() as *mut u8;

			end_r = start_r;

			let mut elem = r;

			for i in 0..block_r {
				unsafe {
					elem = elem.sub(1);

					*end_r = i as u8;

					end_r = end_r.add(is_less(elem, pivot) as usize);
				}
			}
		}

		let count = Ord::min(width(start_l, end_l), width(start_r, end_r));

		if count > 0 {
			macro_rules! left {
				() => {
					l.add(usize::from(*start_l))
				};
			}

			macro_rules! right {
				() => {
					r.sub(usize::from(*start_r) + 1)
				};
			}

			unsafe {
				let tmp = P::read(left!());

				let tmp_ptr = P::get_ptr(&tmp as *const P::Item as *mut P::Item);

				P::copy_nonoverlapping(right!(), left!(), 1);

				for _ in 1..count {
					start_l = start_l.add(1);

					P::copy_nonoverlapping(left!(), right!(), 1);

					start_r = start_r.add(1);

					P::copy_nonoverlapping(right!(), left!(), 1);
				}

				P::copy_nonoverlapping(tmp_ptr, right!(), 1);

				start_l = start_l.add(1);

				start_r = start_r.add(1);
			}
		}

		if start_l == end_l {
			l = unsafe { l.add(block_l) };
		}

		if start_r == end_r {
			r = unsafe { r.sub(block_r) };
		}

		if is_done {
			break;
		}
	}

	if start_l < end_l {
		debug_assert_eq!(width(l, r), block_l);

		while start_l < end_l {
			unsafe {
				end_l = end_l.sub(1);

				P::swap(l.add(usize::from(*end_l)), r.sub(1));

				r = r.sub(1);
			}
		}

		width(v, r)
	} else if start_r < end_r {
		debug_assert_eq!(width(l, r), block_r);

		while start_r < end_r {
			unsafe {
				end_r = end_r.sub(1);

				P::swap(l, r.sub(usize::from(*end_r) + 1));

				l = l.add(1);
			}
		}

		width(v, l)
	} else {
		width(v, l)
	}
}

pub(super) unsafe fn partition<P: Ptr, F>(v: P, v_len: usize, pivot: usize, is_less: &mut F) -> (usize, bool)
where
	F: FnMut(P, P) -> bool,
{
	let (mid, was_partitioned) = {
		v.swap_idx(0, pivot);

		let pivot = v;

		let v = v.add(1);

		let v_len = v_len - 1;

		let tmp = core::mem::ManuallyDrop::new(unsafe { P::read(pivot) });

		let tmp = P::get_ptr((&*tmp) as *const P::Item as *mut P::Item);

		let _pivot_guard = InsertionHole { src: tmp, dest: pivot };

		let pivot = tmp;

		let mut l = 0;

		let mut r = v_len;

		unsafe {
			while l < r && is_less(v.add(l), pivot) {
				l += 1;
			}

			while l < r && !is_less(v.add(r - 1), pivot) {
				r -= 1;
			}
		}

		(l + partition_in_blocks(v.add(l), r - l, pivot, is_less), l >= r)
	};

	v.swap_idx(0, mid);

	(mid, was_partitioned)
}

pub(super) unsafe fn partition_equal<P: Ptr, F>(v: P, v_len: usize, pivot: usize, is_less: &mut F) -> usize
where
	F: FnMut(P, P) -> bool,
{
	v.swap_idx(0, pivot);

	let pivot = v;

	let v = v.add(1);

	let v_len = v_len - 1;

	let tmp = core::mem::ManuallyDrop::new(unsafe { P::read(pivot) });

	let tmp = P::get_ptr((&*tmp) as *const P::Item as *mut P::Item);

	let _pivot_guard = InsertionHole { src: tmp, dest: pivot };

	let pivot = tmp;

	let len = v_len;

	if len == 0 {
		return 0;
	}

	let mut l = 0;

	let mut r = len;

	loop {
		unsafe {
			while l < r && !is_less(pivot, v.add(l)) {
				l += 1;
			}

			loop {
				r -= 1;

				if l >= r || !is_less(pivot, v.add(r)) {
					break;
				}
			}

			if l >= r {
				break;
			}

			let ptr = v;

			P::swap(ptr.add(l), ptr.add(r));

			l += 1;
		}
	}

	l + 1
}

#[cold]

pub(super) unsafe fn break_patterns<P: Ptr>(v: P, v_len: usize) {
	let len = v_len;

	if len >= 8 {
		let mut seed = len;

		let mut gen_usize = || {
			if usize::BITS <= 32 {
				let mut r = seed as u32;

				r ^= r << 13;

				r ^= r >> 17;

				r ^= r << 5;

				seed = r as usize;

				seed
			} else {
				let mut r = seed as u64;

				r ^= r << 13;

				r ^= r >> 7;

				r ^= r << 17;

				seed = r as usize;

				seed
			}
		};

		let modulus = len.next_power_of_two();

		let pos = len / 4 * 2;

		for i in 0..3 {
			let mut other = gen_usize() & (modulus - 1);

			if other >= len {
				other -= len;
			}

			v.swap_idx(pos - 1 + i, other);
		}
	}
}

pub(super) unsafe fn choose_pivot<P: Ptr, F>(v: P, v_len: usize, is_less: &mut F) -> (usize, bool)
where
	F: FnMut(P, P) -> bool,
{
	const SHORTEST_MEDIAN_OF_MEDIANS: usize = 50;

	const MAX_SWAPS: usize = 4 * 3;

	let len = v_len;

	let mut a = len / 4;

	let mut b = len / 4 * 2;

	let mut c = len / 4 * 3;

	let mut swaps = 0;

	if len >= 8 {
		let mut sort2 = |a: &mut usize, b: &mut usize| unsafe {
			if is_less(v.add(*b), v.add(*a)) {
				core::ptr::swap(a, b);

				swaps += 1;
			}
		};

		let mut sort3 = |a: &mut usize, b: &mut usize, c: &mut usize| {
			sort2(a, b);

			sort2(b, c);

			sort2(a, b);
		};

		if len >= SHORTEST_MEDIAN_OF_MEDIANS {
			let mut sort_adjacent = |a: &mut usize| {
				let tmp = *a;

				sort3(&mut (tmp - 1), a, &mut (tmp + 1));
			};

			sort_adjacent(&mut a);

			sort_adjacent(&mut b);

			sort_adjacent(&mut c);
		}

		sort3(&mut a, &mut b, &mut c);
	}

	if swaps < MAX_SWAPS {
		(b, swaps == 0)
	} else {
		P::reverse(v, v_len);

		(len - 1 - b, true)
	}
}

/// sorts `v` using pattern-defeating quicksort, which is *O*(*n* \* log(*n*)) worst-case

pub unsafe fn quicksort<P: Ptr, F>(v: P, v_len: usize, mut is_less: F)
where
	F: FnMut(P, P) -> bool,
{
	if core::mem::size_of::<P::Item>() == 0 {
		return;
	}

	let limit = usize::BITS - v_len.leading_zeros();

	recurse(v, v_len, &mut is_less, None, limit);
}

unsafe fn recurse<P: Ptr, F: FnMut(P, P) -> bool>(mut v: P, mut v_len: usize, is_less: &mut F, mut pred: Option<P>, mut limit: u32) {
	const MAX_INSERTION: usize = 20;

	let mut was_balanced = true;

	let mut was_partitioned = true;

	loop {
		let len = v_len;

		if len <= MAX_INSERTION {
			if len >= 2 {
				insertion_sort_shift_left(v, v_len, 1, is_less);
			}

			return;
		}

		if limit == 0 {
			heapsort(v, v_len, is_less);

			return;
		}

		if !was_balanced {
			break_patterns(v, v_len);

			limit -= 1;
		}

		let (pivot, likely_sorted) = choose_pivot(v, v_len, is_less);

		if was_balanced && was_partitioned && likely_sorted {
			if partial_insertion_sort(v, v_len, is_less) {
				return;
			}
		}

		if let Some(p) = pred {
			if !is_less(p, v.add(pivot)) {
				let mid = partition_equal(v, v_len, pivot, is_less);

				v = v.add(mid);

				v_len -= mid;

				continue;
			}
		}

		let (mid, was_p) = partition(v, v_len, pivot, is_less);

		was_balanced = Ord::min(mid, len - mid) >= len / 8;

		was_partitioned = was_p;

		let left = v;

		let left_len = mid;

		let right = v.add(mid);

		let right_len = v_len - mid;

		let pivot = right;

		let right = right.add(1);

		let right_len = right_len - 1;

		if left_len < right_len {
			recurse(left, left_len, is_less, pred, limit);

			v = right;

			v_len = right_len;

			pred = Some(pivot);
		} else {
			recurse(right, right_len, is_less, Some(pivot), limit);

			v = left;

			v_len = left_len;
		}
	}
}

pub unsafe fn sort_unstable_by<P: Ptr>(ptr: P, len: usize, compare: impl FnMut(P, P) -> core::cmp::Ordering) {
	let mut compare = compare;

	quicksort(
		ptr,
		len,
		#[inline(always)]
		|a, b| compare(a, b) == core::cmp::Ordering::Less,
	);
}

pub unsafe fn sort_indices<I: crate::Index, T>(indices: &mut [I], values: &mut [T]) {
	let len = indices.len();

	debug_assert!(values.len() == len);

	sort_unstable_by((indices.as_mut_ptr(), values.as_mut_ptr()), len, |(i, _), (j, _)| (*i).cmp(&*j));
}

#[cfg(test)]

mod tests {

	use super::*;
	use crate::assert;
	use crate::internal_prelude::*;
	use rand::rngs::StdRng;
	use rand::{Rng, SeedableRng};

	#[test]

	fn test_quicksort() {
		let mut a = [3, 2, 2, 4, 1];

		let mut b = [1.0, 2.0, 3.0, 4.0, 5.0];

		let len = a.len();

		unsafe { quicksort((a.as_mut_ptr(), b.as_mut_ptr()), len, |p, q| *p.0 < *q.0) };

		assert!(a == [1, 2, 2, 3, 4]);

		assert!(b == [5.0, 2.0, 3.0, 1.0, 4.0]);
	}

	#[test]

	fn test_quicksort_big() {
		let rng = &mut StdRng::seed_from_u64(0);

		let a = &mut *(0..1000).map(|_| rng.random::<u32>()).collect::<Vec<_>>();

		let b = &mut *(0..1000).map(|_| rng.random::<f64>()).collect::<Vec<_>>();

		let a_orig = &*a.to_vec();

		let b_orig = &*b.to_vec();

		let mut perm = (0..1000).collect::<Vec<_>>();

		perm.sort_unstable_by_key(|&i| a[i]);

		let len = a.len();

		unsafe { quicksort((a.as_mut_ptr(), b.as_mut_ptr()), len, |p, q| *p.0 < *q.0) };

		for i in 0..1000 {
			assert!(a_orig[perm[i]] == a[i]);

			assert!(b_orig[perm[i]] == b[i]);
		}
	}
}
