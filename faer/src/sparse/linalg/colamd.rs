//! approximate minimum degree column ordering.

// COLAMD, Copyright (c) 1998-2022, Timothy A. Davis and Stefan Larimore,
// All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause
//
//     Redistribution and use in source and binary forms, with or without
//     modification, are permitted provided that the following conditions are met:
//         * Redistributions of source code must retain the above copyright notice, this list of
//           conditions and the following disclaimer.
//         * Redistributions in binary form must reproduce the above copyright notice, this list of
//           conditions and the following disclaimer in the documentation and/or other materials
//           provided with the distribution.
//         * Neither the name of the organizations to which the authors are affiliated, nor the
//           names of its contributors may be used to endorse or promote products derived from this
//           software without specific prior written permission.
//
//     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//     AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//     IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//     ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
//     DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//     (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//     SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//     LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
//     OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//     DAMAGE.

use crate::assert;
use crate::internal_prelude_sp::*;

impl<I: Index> ColamdCol<I> {
	fn is_dead_principal(&self) -> bool {
		self.start == I::Signed::truncate(NONE)
	}

	fn is_dead(&self) -> bool {
		self.start < I::Signed::truncate(0)
	}

	fn is_alive(&self) -> bool {
		!self.is_dead()
	}

	fn kill_principal(&mut self) {
		self.start = I::Signed::truncate(NONE)
	}

	fn kill_non_principal(&mut self) {
		self.start = I::Signed::truncate(NONE - 1)
	}
}

impl<I: Index> ColamdRow<I> {
	fn is_dead(&self) -> bool {
		self.shared2 < I::Signed::truncate(0)
	}

	fn is_alive(&self) -> bool {
		!self.is_dead()
	}

	fn kill(&mut self) {
		self.shared2 = I::Signed::truncate(NONE)
	}
}

fn clear_mark<I: Index>(tag_mark: I, max_mark: I, row: &mut [ColamdRow<I>]) -> I {
	let I = I::truncate;
	let SI = I::Signed::truncate;
	if tag_mark == I(0) || tag_mark >= max_mark {
		for r in row {
			if r.is_alive() {
				r.shared2 = SI(0);
			}
		}
		I(1)
	} else {
		tag_mark
	}
}

/// computes the layout of required workspace for computing the colamd ordering of a
/// matrix
pub fn order_scratch<I: Index>(nrows: usize, ncols: usize, A_nnz: usize) -> StackReq {
	let m = nrows;
	let n = ncols;
	let n_scratch = StackReq::new::<I>(n);
	let m_scratch = StackReq::new::<I>(m);
	let np1_scratch = StackReq::new::<I>(n + 1);
	let size = StackReq::new::<I>(
		A_nnz
			.checked_mul(2)
			.and_then(|x| x.checked_add(A_nnz / 5))
			.and_then(|p| p.checked_add(n))
			.unwrap_or(usize::MAX),
	);

	StackReq::or(
		StackReq::all_of(&[
			StackReq::new::<ColamdCol<I>>(n + 1),
			StackReq::new::<ColamdRow<I>>(m + 1),
			np1_scratch,
			size,
		]),
		StackReq::all_of(&[
			n_scratch,
			n_scratch,
			StackReq::or(StackReq::and(n_scratch, m_scratch), StackReq::all_of(&[n_scratch; 3])),
		]),
	)
}

/// computes the approximate minimum degree ordering for reducing the fill-in during the sparse
/// qr factorization of a matrix with the sparsity pattern of $A$
///
/// # note
/// allows unsorted matrices
pub fn order<I: Index>(
	perm: &mut [I],
	perm_inv: &mut [I],
	A: SymbolicSparseColMatRef<'_, I>,
	control: Control,
	stack: &mut MemStack,
) -> Result<(), FaerError> {
	let m = A.nrows();
	let n = A.ncols();
	let I = I::truncate;
	let SI = I::Signed::truncate;

	{
		let (col, stack) = unsafe { stack.make_raw::<ColamdCol<I>>(n + 1) };
		let (row, stack) = unsafe { stack.make_raw::<ColamdRow<I>>(m + 1) };

		let nnz = A.compute_nnz();
		let (p, stack) = unsafe { stack.make_raw::<I>(n + 1) };

		let size = (2 * nnz).checked_add(nnz / 5).and_then(|p| p.checked_add(n));
		let (new_row_idx, _) = unsafe { stack.make_raw::<I>(size.ok_or(FaerError::IndexOverflow)?) };

		p[0] = I(0);
		for j in 0..n {
			let row_idx = A.row_idx_of_col_raw(j);
			p[j + 1] = p[j] + I(row_idx.len());
			new_row_idx[p[j].zx()..p[j + 1].zx()].copy_from_slice(&*row_idx);
		}
		let A = new_row_idx;

		for c in 0..n {
			col[c] = ColamdCol::<I> {
				start: p[c].to_signed(),
				length: p[c + 1].to_signed() - p[c].to_signed(),
				shared1: SI(1),
				shared2: SI(0),
				shared3: SI(NONE),
				shared4: SI(NONE),
			};
		}

		col[n] = ColamdCol::<I> {
			start: SI(NONE),
			length: SI(NONE),
			shared1: SI(NONE),
			shared2: SI(NONE),
			shared3: SI(NONE),
			shared4: SI(NONE),
		};

		for r in 0..m {
			row[r] = ColamdRow::<I> {
				start: SI(0),
				length: SI(0),
				shared1: SI(NONE),
				shared2: SI(NONE),
			};
		}

		row[m] = ColamdRow::<I> {
			start: SI(NONE),
			length: SI(NONE),
			shared1: SI(NONE),
			shared2: SI(NONE),
		};

		let mut jumbled = false;

		for c in 0..n {
			let mut last_row = SI(NONE);
			let mut cp = p[c].zx();
			let cp_end = p[c + 1].zx();

			while cp < cp_end {
				let r = A[cp].zx();
				cp += 1;
				if SI(r) <= last_row || row[r].shared2 == SI(c) {
					jumbled = true;
				}

				if row[r].shared2 != SI(c) {
					row[r].length += SI(1);
				} else {
					col[c].length -= SI(1);
				}

				row[r].shared2 = SI(c);
				last_row = SI(r);
			}
		}

		row[0].start = p[n].to_signed();
		row[0].shared1 = row[0].start;
		row[0].shared2 = -SI(1);
		for r in 1..m {
			row[r].start = row[r - 1].start + row[r - 1].length;
			row[r].shared1 = row[r].start;
			row[r].shared2 = -SI(1);
		}

		if jumbled {
			for c in 0..n {
				let mut cp = p[c].zx();
				let cp_end = p[c + 1].zx();
				while cp < cp_end {
					let r = A[cp].zx();
					cp += 1;

					if row[r].shared2 != SI(c) {
						A[row[r].shared1.zx()] = I(c);
						row[r].shared1 += SI(1);
						row[r].shared2 = SI(c);
					}
				}
			}
		} else {
			for c in 0..n {
				let mut cp = p[c].zx();
				let cp_end = p[c + 1].zx();
				while cp < cp_end {
					let r = A[cp].zx();
					cp += 1;

					A[row[r].shared1.zx()] = I(c);
					row[r].shared1 += SI(1);
				}
			}
		}

		for r in 0..m {
			row[r].shared2 = SI(0);
			row[r].shared1 = row[r].length;
		}

		if jumbled {
			col[0].start = SI(0);
			p[0] = I::from_signed(col[0].start);
			for c in 1..n {
				col[c].start = col[c - 1].start + col[c - 1].length;
				p[c] = I::from_signed(col[c].start);
			}

			for r in 0..m {
				let mut rp = row[r].start.zx();
				let rp_end = rp + row[r].length.zx();
				while rp < rp_end {
					A[p[rp].zx()] = I(r);
					p[rp] += I(1);
					rp += 1;
				}
			}
		}

		let dense_row_count = if control.dense_row < 0.0 {
			n - 1
		} else {
			Ord::max(16, (control.dense_row * n as f64) as usize)
		};

		let dense_col_count = if control.dense_col < 0.0 {
			m - 1
		} else {
			Ord::max(16, (control.dense_col * Ord::min(m, n) as f64) as usize)
		};

		let mut max_deg = 0;
		let mut ncol2 = n;
		let mut nrow2 = m;
		let _ = nrow2;

		let head = &mut *p;
		for c in (0..n).rev() {
			let deg = col[c].length;
			if deg == SI(0) {
				ncol2 -= 1;
				col[c].shared2 = SI(ncol2);
				col[c].kill_principal();
			}
		}

		for c in (0..n).rev() {
			if col[c].is_dead() {
				continue;
			}
			let deg = col[c].length;
			if deg.zx() > dense_col_count {
				ncol2 -= 1;
				col[c].shared2 = SI(ncol2);

				let mut cp = col[c].start.zx();
				let cp_end = cp + col[c].length.zx();

				while cp < cp_end {
					row[A[cp].zx()].shared1 -= SI(1);
					cp += 1;
				}
				col[c].kill_principal();
			}
		}

		for r in 0..m {
			let deg = row[r].shared1.zx();
			assert!(deg <= n);
			if deg > dense_row_count || deg == 0 {
				row[r].kill();
				nrow2 -= 1;
			} else {
				max_deg = Ord::max(deg, max_deg);
			}
		}

		for c in (0..n).rev() {
			if col[c].is_dead() {
				continue;
			}

			let mut score = 0;
			let mut cp = col[c].start.zx();
			let mut new_cp = cp;
			let cp_end = cp + col[c].length.zx();

			while cp < cp_end {
				let r = A[cp].zx();
				cp += 1;
				if row[r].is_dead() {
					continue;
				}

				A[new_cp] = I(r);
				new_cp += 1;

				score += row[r].shared1.zx() - 1;
				score = Ord::min(score, n);
			}

			let col_length = new_cp - col[c].start.zx();
			if col_length == 0 {
				ncol2 -= 1;
				col[c].shared2 = SI(ncol2);
				col[c].kill_principal();
			} else {
				assert!(score <= n);
				col[c].length = SI(col_length);
				col[c].shared2 = SI(score);
			}
		}

		head.fill(I(NONE));
		let mut min_score = n;
		for c in (0..n).rev() {
			if col[c].is_alive() {
				let score = col[c].shared2.zx();

				assert!(min_score <= n);
				assert!(score <= n);
				assert!(head[score].to_signed() >= SI(NONE));

				let next_col = head[score];
				col[c].shared3 = SI(NONE);
				col[c].shared4 = next_col.to_signed();

				if next_col != I(NONE) {
					col[next_col.zx()].shared3 = SI(c);
				}
				head[score] = I(c);

				min_score = Ord::min(score, min_score);
			}
		}

		let max_mark = I::from_signed(I::Signed::MAX) - I(n);
		let mut tag_mark = clear_mark(I(0), max_mark, row);
		let mut min_score = 0;
		let mut pfree = 2 * nnz;

		let mut k = 0;
		while k < ncol2 {
			assert!(min_score <= n);
			assert!(head[min_score].to_signed() >= SI(NONE));
			while head[min_score] == I(NONE) && min_score < n {
				min_score += 1;
			}

			let pivot_col = head[min_score].zx();
			let mut next_col = col[pivot_col].shared4;
			head[min_score] = I::from_signed(next_col);
			if next_col != SI(NONE) {
				col[next_col.zx()].shared3 = SI(NONE);
			}

			assert!(!col[pivot_col].is_dead());

			let pivot_col_score = col[pivot_col].shared2;
			col[pivot_col].shared2 = SI(k);

			let pivot_col_thickness = col[pivot_col].shared1;
			assert!(pivot_col_thickness > SI(0));
			k += pivot_col_thickness.zx();

			let needed_memory = Ord::min(pivot_col_score, SI(n - k));

			if pfree as isize + needed_memory.sx() as isize >= A.len() as isize {
				pfree = garbage_collection(row, col, A, pfree);
				assert!((pfree as isize + needed_memory.sx() as isize) < A.len() as isize);
				tag_mark = clear_mark(I(0), max_mark, row);
			}
			let pivot_row_start = pfree;
			let mut pivot_row_degree = 0;
			col[pivot_col].shared1 = -pivot_col_thickness;
			let mut cp = col[pivot_col].start.zx();
			let cp_end = cp + col[pivot_col].length.zx();

			while cp < cp_end {
				let r = A[cp].zx();
				cp += 1;
				if row[r].is_alive() {
					let mut rp = row[r].start.zx();
					let rp_end = rp + row[r].length.zx();
					while rp < rp_end {
						let c = A[rp].zx();
						rp += 1;

						let col_thickness = col[c].shared1;
						if col_thickness > SI(0) && col[c].is_alive() {
							col[c].shared1 = -col_thickness;
							A[pfree] = I(c);
							pfree += 1;
							pivot_row_degree += col_thickness.zx();
						}
					}
				}
			}

			col[pivot_col].shared1 = pivot_col_thickness;
			max_deg = Ord::max(max_deg, pivot_row_degree);

			let mut cp = col[pivot_col].start.zx();
			let cp_end = cp + col[pivot_col].length.zx();
			while cp < cp_end {
				let r = A[cp].zx();
				cp += 1;
				row[r].kill();
			}

			let pivot_row_length = pfree - pivot_row_start;
			let pivot_row = if pivot_row_length > 0 { A[col[pivot_col].start.zx()] } else { I(NONE) };

			let mut rp = pivot_row_start;
			let rp_end = rp + pivot_row_length;
			while rp < rp_end {
				let c = A[rp].zx();
				rp += 1;

				assert!(col[c].is_alive());

				let col_thickness = -col[c].shared1;
				assert!(col_thickness > SI(0));

				col[c].shared1 = col_thickness;

				let cur_score = col[c].shared2.zx();
				let prev_col = col[c].shared3;
				let next_col = col[c].shared4;

				assert!(cur_score <= n);

				if prev_col == SI(NONE) {
					head[cur_score] = I::from_signed(next_col);
				} else {
					col[prev_col.zx()].shared4 = next_col;
				}
				if next_col != SI(NONE) {
					col[next_col.zx()].shared3 = prev_col;
				}

				let mut cp = col[c].start.zx();
				let cp_end = cp + col[c].length.zx();
				while cp < cp_end {
					let r = A[cp].zx();
					cp += 1;
					let row_mark = row[r].shared2;
					if row[r].is_dead() {
						continue;
					}
					assert!(I(r) != pivot_row);
					let mut set_difference = row_mark - tag_mark.to_signed();
					if set_difference < SI(0) {
						assert!(row[r].shared1 <= SI(max_deg));
						set_difference = row[r].shared1;
					}
					set_difference -= col_thickness;
					assert!(set_difference >= SI(0));
					if set_difference == SI(0) && control.aggressive {
						row[r].kill();
					} else {
						row[r].shared2 = set_difference + tag_mark.to_signed();
					}
				}
			}

			let mut rp = pivot_row_start;
			let rp_end = rp + pivot_row_length;
			while rp < rp_end {
				let c = A[rp].zx();
				rp += 1;

				assert!(col[c].is_alive());

				let mut hash = 0;
				let mut cur_score = 0;
				let mut cp = col[c].start.zx();
				let mut new_cp = cp;
				let cp_end = cp + col[c].length.zx();

				while cp < cp_end {
					let r = A[cp].zx();
					cp += 1;
					let row_mark = row[r].shared2;
					if row[r].is_dead() {
						continue;
					}
					assert!(row_mark >= tag_mark.to_signed());
					A[new_cp] = I(r);
					new_cp += 1;
					hash += r;

					cur_score += (row_mark - tag_mark.to_signed()).zx();
					cur_score = Ord::min(cur_score, n);
				}

				col[c].length = SI(new_cp - col[c].start.zx());

				if col[c].length.zx() == 0 {
					col[c].kill_principal();
					pivot_row_degree -= col[c].shared1.zx();
					col[c].shared2 = SI(k);
					k += col[c].shared1.zx();
				} else {
					col[c].shared2 = SI(cur_score);
					hash %= n + 1;

					let head_column = head[hash];
					let first_col;
					if head_column.to_signed() > SI(NONE) {
						first_col = col[head_column.zx()].shared3;
						col[head_column.zx()].shared3 = SI(c);
					} else {
						first_col = -(head_column.to_signed() + SI(2));
						head[hash] = I::from_signed(-SI(c + 2));
					}
					col[c].shared4 = first_col;
					col[c].shared3 = SI(hash);
					assert!(col[c].is_alive());
				}
			}

			detect_super_cols(col, A, head, pivot_row_start, pivot_row_length);

			col[pivot_col].kill_principal();
			tag_mark = clear_mark(tag_mark + I(max_deg) + I(1), max_mark, row);

			let mut rp = pivot_row_start;
			let mut new_rp = rp;
			let rp_end = rp + pivot_row_length;
			while rp < rp_end {
				let c = A[rp].zx();
				rp += 1;
				if col[c].is_dead() {
					continue;
				}

				A[new_rp] = I(c);
				new_rp += 1;

				A[(col[c].start + col[c].length).zx()] = pivot_row;
				col[c].length += SI(1);

				let mut cur_score = col[c].shared2.zx() + pivot_row_degree;
				let max_score = n - k - col[c].shared1.zx();
				cur_score -= col[c].shared1.zx();
				cur_score = Ord::min(cur_score, max_score);
				col[c].shared2 = SI(cur_score);

				next_col = head[cur_score].to_signed();
				col[c].shared4 = next_col;
				col[c].shared3 = SI(NONE);
				if next_col != SI(NONE) {
					col[next_col.zx()].shared3 = SI(c);
				}
				head[cur_score] = I(c);

				min_score = Ord::min(min_score, cur_score);
			}

			if pivot_row_degree > 0 {
				let pivot_row = pivot_row.zx();
				row[pivot_row].start = SI(pivot_row_start);
				row[pivot_row].length = SI(new_rp - pivot_row_start);
				row[pivot_row].shared1 = SI(pivot_row_degree);
				row[pivot_row].shared2 = SI(0);
			}
		}

		for i in 0..n {
			assert!(col[i].is_dead());
			if !col[i].is_dead_principal() && col[i].shared2 == SI(NONE) {
				let mut parent = i;
				loop {
					parent = col[parent].shared1.zx();
					if col[parent].is_dead_principal() {
						break;
					}
				}

				let mut c = i;
				let mut order = col[parent].shared2.zx();

				loop {
					assert!(col[c].shared2 == SI(NONE));
					col[c].shared2 = SI(order);
					order += 1;
					col[c].shared1 = SI(parent);
					c = col[c].shared1.zx();

					if col[c].shared2 != SI(NONE) {
						break;
					}
				}

				col[parent].shared2 = SI(order);
			}
		}

		for c in 0..n {
			perm[col[c].shared2.zx()] = I(c);
		}
		for c in 0..n {
			perm_inv[perm[c].zx()] = I(c);
		}
	}

	let mut etree = alloc::vec![I(0); n];
	let mut post = alloc::vec![I(0); n];
	let etree = super::qr::col_etree::<I>(A, Some(PermRef::<'_, I>::new_checked(perm, perm_inv, n)), &mut etree, stack);
	super::qr::postorder(&mut post, etree, stack);
	for i in 0..n {
		perm[post[i].zx()] = I(i);
	}
	for i in 0..n {
		perm_inv[i] = perm[perm_inv[i].zx()];
	}
	for i in 0..n {
		perm[perm_inv[i].zx()] = I(i);
	}

	Ok(())
}

fn detect_super_cols<I: Index>(col: &mut [ColamdCol<I>], A: &mut [I], head: &mut [I], row_start: usize, row_length: usize) {
	let I = I::truncate;
	let SI = I::Signed::truncate;

	let mut rp = row_start;
	let rp_end = rp + row_length;
	while rp < rp_end {
		let c = A[rp].zx();
		rp += 1;
		if col[c].is_dead() {
			continue;
		}

		let hash = col[c].shared3.zx();
		let head_column = head[hash].to_signed();
		let first_col = if head_column > SI(NONE) {
			col[head_column.zx()].shared3
		} else {
			-(head_column + SI(2))
		};

		let mut super_c_ = first_col;
		while super_c_ != SI(NONE) {
			let super_c = super_c_.zx();
			let length = col[super_c].length;
			let mut prev_c = super_c;

			let mut c_ = col[super_c].shared4;
			while c_ != SI(NONE) {
				let c = c_.zx();
				if col[c].length != length || col[c].shared2 != col[super_c].shared2 {
					prev_c = c;
					c_ = col[c].shared4;
					continue;
				}

				let mut cp1 = col[super_c].start.zx();
				let mut cp2 = col[c].start.zx();

				let mut i = 0;
				while i < length.zx() {
					if A[cp1] != A[cp2] {
						break;
					}
					cp1 += 1;
					cp2 += 1;
					i += 1;
				}

				if i != length.zx() {
					prev_c = c;
					c_ = col[c].shared4;
					continue;
				}

				col[super_c].shared1 += col[c].shared1;
				col[c].shared1 = SI(super_c);
				col[c].kill_non_principal();
				col[c].shared2 = SI(NONE);
				col[prev_c].shared4 = col[c].shared4;

				c_ = col[c].shared4;
			}
			super_c_ = col[super_c].shared4;
		}

		if head_column > SI(NONE) {
			col[head_column.zx()].shared3 = SI(NONE);
		} else {
			head[hash] = I(NONE);
		}
	}
}

fn garbage_collection<I: Index>(row: &mut [ColamdRow<I>], col: &mut [ColamdCol<I>], A: &mut [I], pfree: usize) -> usize {
	let I = I::truncate;
	let SI = I::Signed::truncate;

	let mut pdest = 0usize;
	let m = row.len() - 1;
	let n = col.len() - 1;
	for c in 0..n {
		if !col[c].is_dead() {
			let mut psrc = col[c].start.zx();
			col[c].start = SI(pdest);
			let length = col[c].length.zx();

			for _ in 0..length {
				let r = A[psrc].zx();
				psrc += 1;
				if !row[r].is_dead() {
					A[pdest] = I(r);
					pdest += 1;
				}
			}
			col[c].length = SI(pdest) - col[c].start;
		}
	}
	for r in 0..m {
		if row[r].is_dead() || row[r].length == SI(0) {
			row[r].kill();
		} else {
			let psrc = row[r].start.zx();
			row[r].shared2 = A[psrc].to_signed();
			A[psrc] = !I(r);
		}
	}

	let mut psrc = pdest;
	while psrc < pfree {
		let psrc_ = psrc;
		psrc += 1;
		if A[psrc_].to_signed() < SI(0) {
			psrc -= 1;
			let r = (!A[psrc]).zx();
			A[psrc] = I::from_signed(row[r].shared2);
			row[r].start = SI(pdest);
			let length = row[r].length.zx();
			for _ in 0..length {
				let c = A[psrc].zx();
				psrc += 1;
				if !col[c].is_dead() {
					A[pdest] = I(c);
					pdest += 1;
				}
			}

			row[r].length = SI(pdest) - row[r].start;
		}
	}

	pdest
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct ColamdRow<I: Index> {
	start: I::Signed,
	length: I::Signed,
	shared1: I::Signed,
	shared2: I::Signed,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct ColamdCol<I: Index> {
	start: I::Signed,
	length: I::Signed,
	shared1: I::Signed,
	shared2: I::Signed,
	shared3: I::Signed,
	shared4: I::Signed,
}

unsafe impl<I: Index> bytemuck::Zeroable for ColamdCol<I> {}
unsafe impl<I: Index> bytemuck::Pod for ColamdCol<I> {}
unsafe impl<I: Index> bytemuck::Zeroable for ColamdRow<I> {}
unsafe impl<I: Index> bytemuck::Pod for ColamdRow<I> {}

/// tuning parameters for the amd implementation
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Control {
	/// "dense" if degree > dense_row * sqrt(ncols)
	pub dense_row: f64,
	/// "dense" if degree > dense_col * sqrt(min(nrows, ncols))
	pub dense_col: f64,
	/// do aggressive absorption
	pub aggressive: bool,
}

impl Default for Control {
	#[inline]
	fn default() -> Self {
		Self {
			dense_row: 0.5,
			dense_col: 0.5,
			aggressive: true,
		}
	}
}
