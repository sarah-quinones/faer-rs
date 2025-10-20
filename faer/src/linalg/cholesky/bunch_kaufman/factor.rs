use crate::internal_prelude::*;
use crate::{assert, perm};
use linalg::matmul::triangular::BlockStructure;

const TOP_BIT: usize = 1 << (usize::BITS - 1);

/// pivoting strategy for choosing the pivots
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]

pub enum PivotingStrategy {
	/// deprecated, corresponds to partial pivoting
	#[deprecated]
	Diagonal,
	/// searches for the k-th pivot in the k-th column
	Partial,
	/// searches for the k-th pivot in the k-th column, as well as the tail of the diagonal of the
	/// matrix
	PartialDiag,
	/// searches for pivots that are locally optimal
	Rook,
	/// searches for pivots that are locally optimal, as well as the tail of the diagonal of the
	/// matrix
	RookDiag,
	/// searches for pivots that are globally optimal
	Full,
}

/// tuning parameters for the decomposition
#[derive(Copy, Clone, Debug)]

pub struct LbltParams {
	/// pivoting strategy
	pub pivoting: PivotingStrategy,
	/// block size of the algorithm
	pub block_size: usize,
	/// threshold at which size parallelism should be disabled
	pub par_threshold: usize,
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

fn swap_self_adjoint<T: ComplexField>(A: MatMut<'_, T>, i: usize, j: usize) {
	assert_ne!(i, j);

	let mut A = A;

	let (i, j) = (Ord::min(i, j), Ord::max(i, j));

	perm::swap_cols_idx(A.rb_mut().get_mut(j + 1.., ..), i, j);

	perm::swap_rows_idx(A.rb_mut().get_mut(.., ..i), i, j);

	let tmp = A[(i, i)].real();

	A[(i, i)] = A[(j, j)].as_real();

	A[(j, j)] = tmp.to_cplx();

	A[(j, i)] = A[(j, i)].conj();

	let (Ai, Aj) = A.split_at_row_mut(j);

	let Ai = Ai.get_mut(i + 1..j, i);

	let Aj = Aj.get_mut(0, i + 1..j).transpose_mut();

	zip!(Ai, Aj).for_each(|unzip!(x, y): Zip!(&mut _, &mut _)| {
		(*x, *y) = (y.conj(), x.conj());
	});
}

fn rank_1_update_and_argmax_fallback<'M, 'N, T: ComplexField>(
	A: MatMut<'_, T, Dim<'N>, Dim<'N>>,
	L: ColRef<'_, T, Dim<'N>>,
	d: T::Real,
	start: IdxInc<'N>,
	end: IdxInc<'N>,
) -> (usize, usize, T::Real) {
	let mut A = A;

	let n = A.nrows();

	let mut max_j = n.idx(0);

	let mut max_i = n.idx(0);

	let mut max_offdiag = zero();

	for j in start.to(end) {
		for i in j.next().to(n.end()) {
			A[(i, j)] = &A[(i, j)] - (&L[i] * L[j].conj()).mul_real(&d);

			let val = A[(i, j)].abs2();

			if val > max_offdiag {
				max_offdiag = val;

				max_i = i;

				max_j = j;
			}
		}
	}

	(*max_i, *max_j, max_offdiag)
}

fn rank_2_update_and_argmax_fallback<'N, T: ComplexField>(
	A: MatMut<'_, T, Dim<'N>, Dim<'N>>,
	L0: ColRef<'_, T, Dim<'N>>,
	L1: ColRef<'_, T, Dim<'N>>,
	d: T::Real,
	d00: T::Real,
	d11: T::Real,
	d10: T,
	start: IdxInc<'N>,
	end: IdxInc<'N>,
) -> (usize, usize, T::Real) {
	let mut A = A;

	let n = A.nrows();

	let mut max_j = n.idx(0);

	let mut max_i = n.idx(0);

	let mut max_offdiag = zero();

	for j in start.to(end) {
		let x0 = L0[j].copy();

		let x1 = L1[j].copy();

		let w0 = (x0.mul_real(&d11) - &x1 * &d10).mul_real(&d);

		let w1 = (x1.mul_real(&d00) - &x0 * d10.conj()).mul_real(&d);

		for i in j.next().to(n.end()) {
			A[(i, j)] = &A[(i, j)] - &L0[i] * w0.conj() - &L1[i] * w1.conj();

			let val = A[(i, j)].abs2();

			if val > max_offdiag {
				max_offdiag = val;

				max_i = i;

				max_j = j;
			}
		}
	}

	(*max_i, *max_j, max_offdiag)
}

fn rank_1_update_and_argmax_seq<'M, 'N, T: ComplexField>(
	A: MatMut<'_, T, Dim<'N>, Dim<'N>>,
	L: ColRef<'_, T, Dim<'N>>,
	d: T::Real,
	start: IdxInc<'N>,
	end: IdxInc<'N>,
) -> (usize, usize, T::Real) {
	rank_1_update_and_argmax_fallback(A, L, d, start, end)
}

fn rank_2_update_and_argmax_seq<'N, T: ComplexField>(
	A: MatMut<'_, T, Dim<'N>, Dim<'N>>,
	L0: ColRef<'_, T, Dim<'N>>,
	L1: ColRef<'_, T, Dim<'N>>,
	d: T::Real,
	d00: T::Real,
	d11: T::Real,
	d10: T,
	start: IdxInc<'N>,
	end: IdxInc<'N>,
) -> (usize, usize, T::Real) {
	rank_2_update_and_argmax_fallback(A, L0, L1, d, d00, d11, d10, start, end)
}

fn rank_1_update_and_argmax<T: ComplexField>(A: MatMut<'_, T>, L: ColRef<'_, T>, d: T::Real, par: Par) -> (usize, usize, T::Real) {
	with_dim!(N, A.nrows());

	match par {
		Par::Seq => rank_1_update_and_argmax_seq(A.as_shape_mut(N, N), L.as_row_shape(N), d, IdxInc::ZERO, N.end()),
		#[cfg(feature = "rayon")]
		Par::Rayon(nthreads) => {
			use rayon::prelude::*;

			let nthreads = nthreads.get();

			let n = *N;

			assert!((n as u64) < (1u64 << 50));

			let idx_to_col_start = |idx: usize| {
				let idx_as_percent = idx as f64 / nthreads as f64;

				let col_start_percent = 1.0f64 - libm::sqrt(1.0f64 - idx_as_percent);

				(col_start_percent * n as f64) as usize
			};

			let mut r = alloc::vec![(0usize, 0usize, zero::< T::Real > ()); nthreads];

			spindle::for_each(nthreads, r.par_iter_mut().enumerate(), |(idx, out)| {
				let A = unsafe { A.rb().const_cast() };

				let start = N.idx_inc(idx_to_col_start(idx));

				let end = N.idx_inc(idx_to_col_start(idx + 1));

				*out = rank_1_update_and_argmax_seq(A.as_shape_mut(N, N), L.as_row_shape(N), d.copy(), start, end);
			});

			r.into_iter()
				.max_by(|(_, _, a), (_, _, b)| {
					if a == b {
						core::cmp::Ordering::Equal
					} else if a > b {
						core::cmp::Ordering::Greater
					} else {
						core::cmp::Ordering::Less
					}
				})
				.unwrap()
		},
	}
}

fn rank_2_update_and_argmax<'N, T: ComplexField>(
	A: MatMut<'_, T>,
	L0: ColRef<'_, T>,
	L1: ColRef<'_, T>,
	d: T::Real,
	d00: T::Real,
	d11: T::Real,
	d10: T,
	par: Par,
) -> (usize, usize, T::Real) {
	with_dim!(N, A.nrows());

	match par {
		Par::Seq => rank_2_update_and_argmax_seq(
			A.as_shape_mut(N, N),
			L0.as_row_shape(N),
			L1.as_row_shape(N),
			d,
			d00,
			d11,
			d10,
			IdxInc::ZERO,
			N.end(),
		),
		#[cfg(feature = "rayon")]
		Par::Rayon(nthreads) => {
			use rayon::prelude::*;

			let nthreads = nthreads.get();

			let n = *N;

			assert!((n as u64) < (1u64 << 50));

			let idx_to_col_start = |idx: usize| {
				let idx_as_percent = idx as f64 / nthreads as f64;

				let col_start_percent = 1.0f64 - libm::sqrt(1.0f64 - idx_as_percent);

				(col_start_percent * n as f64) as usize
			};

			let mut r = alloc::vec![(0usize, 0usize, zero::< T::Real > ()); nthreads];

			spindle::for_each(nthreads, r.par_iter_mut().enumerate(), |(idx, out)| {
				let A = unsafe { A.rb().const_cast() };

				let start = N.idx_inc(idx_to_col_start(idx));

				let end = N.idx_inc(idx_to_col_start(idx + 1));

				*out = rank_2_update_and_argmax_seq(
					A.as_shape_mut(N, N),
					L0.as_row_shape(N),
					L1.as_row_shape(N),
					d.copy(),
					d00.copy(),
					d11.copy(),
					d10.copy(),
					start,
					end,
				);
			});

			r.into_iter()
				.max_by(|(_, _, a), (_, _, b)| {
					if a == b {
						core::cmp::Ordering::Equal
					} else if a < b {
						core::cmp::Ordering::Less
					} else {
						core::cmp::Ordering::Greater
					}
				})
				.unwrap()
		},
	}
}

fn lblt_full_piv<T: ComplexField>(A: MatMut<'_, T>, subdiag: DiagMut<'_, T>, pivots: &mut [usize], par: Par, params: LbltParams) {
	let alpha = ((one::<T::Real>() + from_f64::<T::Real>(17.0).sqrt()) * from_f64::<T::Real>(0.125)).abs2();

	let mut A = A;

	let mut subdiag = subdiag.column_vector_mut();

	let mut par = par;

	let n = A.nrows();

	let scale_fwd = &A.norm_max();

	let scale_bwd = &scale_fwd.recip();

	zip!(A.rb_mut()).for_each(|unzip!(x)| *x = x.mul_real(scale_bwd));

	let mut max_i = 0;

	let mut max_j = 0;

	let mut max_offdiag = zero();

	for j in 0..n {
		for i in j + 1..n {
			let val = A[(i, j)].abs2();

			if val > max_offdiag {
				max_offdiag = val;

				max_i = i;

				max_j = j;
			}
		}
	}

	let mut k = 0;

	while k < n {
		if max_offdiag == zero() {
			break;
		}

		let (mut Aprev, mut A) = A.rb_mut().get_mut(k.., ..).split_at_col_mut(k);

		let mut subdiag = subdiag.rb_mut().get_mut(k..);

		let pivots = &mut pivots[k..];

		let n = A.nrows();

		let mut max_s = 0;

		let mut max_diag = zero();

		for s in 0..n {
			let val = A[(s, s)].abs2();

			if val > max_diag {
				max_diag = val;

				max_s = s;
			}
		}

		let npiv;

		let i0;

		let i1;

		if max_diag >= &alpha * &max_offdiag {
			npiv = 1;

			i0 = max_s;

			i1 = usize::MAX;
		} else {
			npiv = 2;

			i0 = max_j;

			i1 = max_i;
		}

		let rem = n - npiv;

		if rem * rem < params.par_threshold {
			par = Par::Seq;
		}

		if i0 != 0 {
			swap_self_adjoint(A.rb_mut(), 0, i0);

			perm::swap_rows_idx(Aprev.rb_mut(), 0, i0);
		}

		if npiv == 2 && i1 != 1 {
			swap_self_adjoint(A.rb_mut(), 1, i1);

			perm::swap_rows_idx(Aprev.rb_mut(), 1, i1);
		}

		if npiv == 1 {
			let diag = A[(0, 0)].real();

			let diag_inv = &diag.recip();

			subdiag[0] = zero();

			let (_, _, L, mut A) = A.rb_mut().split_at_mut(1, 1);

			let n = A.nrows();

			let mut L = L.col_mut(0);

			zip!(L.rb_mut()).for_each(|unzip!(x)| *x = x.mul_real(diag_inv));

			for i in 0..n {
				A[(i, i)] = (A[(i, i)].real() - &diag * L[i].abs2()).to_cplx();
			}

			if n < params.par_threshold {}

			if n != 0 {
				(max_i, max_j, max_offdiag) = rank_1_update_and_argmax(A.rb_mut(), L.rb(), diag, par);
			}
		} else {
			let a00 = A[(0, 0)].real();

			let a11 = A[(1, 1)].real();

			let a10 = A[(1, 0)].copy();

			subdiag[0] = a10.copy();

			subdiag[1] = zero();

			A[(1, 0)] = zero();

			let d10 = a10.abs();

			let d10_inv = &d10.recip();

			let d00 = a00 * d10_inv;

			let d11 = a11 * d10_inv;

			let t = (&d00 * &d11 - one::<T::Real>()).recip();

			let d10 = a10.mul_real(d10_inv);

			let d = t * d10_inv;

			let (_, _, L, mut A) = A.rb_mut().split_at_mut(2, 2);

			let (mut L0, mut L1) = L.two_cols_mut(0, 1);

			let n = A.nrows();

			if n != 0 {
				(max_i, max_j, max_offdiag) =
					rank_2_update_and_argmax(A.rb_mut(), L0.rb(), L1.rb(), d.copy(), d00.copy(), d11.copy(), d10.copy(), par);
			}

			for j in 0..n {
				let x0 = &L0[j].copy();

				let x1 = &L1[j].copy();

				let w0 = (x0.mul_real(&d11) - x1 * &d10).mul_real(&d);

				let w1 = (x1.mul_real(&d00) - x0 * d10.conj()).mul_real(&d);

				A[(j, j)] = (&A[(j, j)] - &L0[j] * w0.conj() - &L1[j] * w1.conj()).as_real();

				L0[j] = w0;

				L1[j] = w1;
			}
		}

		if npiv == 2 {
			pivots[0] = (i0 + k) | TOP_BIT;

			pivots[1] = (i1 + k) | TOP_BIT;
		} else {
			pivots[0] = i0 + k;
		}

		k += npiv;
	}

	while k < n {
		let (mut Aprev, mut A) = A.rb_mut().get_mut(k.., ..).split_at_col_mut(k);

		let mut subdiag = subdiag.rb_mut().get_mut(k..);

		let pivots = &mut pivots[k..];

		let n = A.nrows();

		let mut max_s = 0;

		let mut max_diag = zero();

		for s in 0..n {
			let val = A[(s, s)].abs2();

			if val > max_diag {
				max_diag = val;

				max_s = s;
			}
		}

		if max_s != 0 {
			let (mut A0, mut As) = A.rb_mut().two_cols_mut(0, max_s);

			core::mem::swap(&mut A0[0], &mut As[max_s]);

			perm::swap_rows_idx(Aprev.rb_mut(), 0, max_s);
		}

		subdiag[0] = zero();

		pivots[0] = max_s + k;

		k += 1;
	}

	zip!(A.rb_mut().diagonal_mut().column_vector_mut()).for_each(|unzip!(x)| *x = x.mul_real(scale_fwd));

	zip!(subdiag.rb_mut()).for_each(|unzip!(x)| *x = x.mul_real(scale_fwd));
}

#[track_caller]

fn l1_argmax<T: ComplexField>(col: ColRef<'_, T>) -> (Option<usize>, T::Real) {
	let n = col.nrows();

	if n == 0 {
		return (None, zero());
	}

	let mut i = 0;

	let mut best = zero();

	for j in 0..n {
		let val = col[j].abs1();

		if val > best {
			best = val;

			i = j;
		}
	}

	(Some(i), best)
}

#[track_caller]

fn offdiag_argmax<T: ComplexField>(A: MatRef<'_, T>, idx: usize) -> (Option<usize>, T::Real) {
	let (mut col_argmax, col_max) = l1_argmax(A.rb().get(idx + 1.., idx));

	col_argmax.as_mut().map(|col_argmax| *col_argmax += idx + 1);

	let (row_argmax, row_max) = l1_argmax(A.rb().get(idx, ..idx).transpose());

	if col_max > row_max {
		(col_argmax, col_max)
	} else {
		(row_argmax, row_max)
	}
}

fn update_and_offdiag_argmax<T: ComplexField>(
	mut dst: ColMut<'_, T>,
	Wl: MatRef<'_, T>,
	Al: MatRef<'_, T>,
	Ar: MatRef<'_, T>,
	i0: usize,
	par: Par,
) -> (Option<usize>, T::Real) {
	let n = Al.nrows();

	for j in 0..i0 {
		dst[j] = Ar[(i0, j)].conj();
	}

	dst[i0] = zero();

	for j in i0 + 1..n {
		dst[j] = Ar[(j, i0)].copy();
	}

	linalg::matmul::matmul(dst.rb_mut(), Accum::Add, Al.rb(), Wl.row(i0).adjoint(), -one::<T>(), par);

	dst[i0] = zero();

	let ret = l1_argmax(dst.rb());

	dst[i0] = Ar[(i0, i0)].as_real();

	if n == 1 { (None, zero()) } else { ret }
}

#[inline(never)]

fn lblt_blocked_step<T: ComplexField>(
	alpha: T::Real,
	W: MatMut<'_, T>,
	A_left: MatMut<'_, T>,
	A: MatMut<'_, T>,
	subdiag: DiagMut<'_, T>,
	pivots: &mut [usize],
	rook: bool,
	diagonal: bool,
	par: Par,
) -> usize {
	let mut A = A;

	let mut A_left = A_left;

	let mut subdiag = subdiag;

	let mut W = W;

	let n = A.nrows();

	let block_size = W.ncols();

	assert!(all(A.nrows() == n, A.ncols() == n, W.nrows() == n, subdiag.dim() == n, block_size >= 2,));

	let kmax = Ord::min(block_size - 1, n);

	let mut k = 0usize;

	while k < kmax {
		let mut A = A.rb_mut();

		let mut W = W.rb_mut();

		let mut subdiag = subdiag.rb_mut().column_vector_mut().get_mut(k..);

		let A_left = A_left.rb_mut().get_mut(k.., ..);

		let (mut Wl, mut Wr) = W.rb_mut().get_mut(k.., ..).split_at_col_mut(k);

		let (mut Al, mut Ar) = A.rb_mut().get_mut(k.., ..).split_at_col_mut(k);

		let mut Al = Al.rb_mut();

		let mut Wr = Wr.rb_mut().get_mut(.., ..2);

		let npiv;

		let mut i0 = if diagonal {
			l1_argmax(Ar.rb().diagonal().column_vector()).0.unwrap()
		} else {
			0
		};

		let mut i1 = usize::MAX;

		let mut nothing_to_do = false;

		let (mut Wr0, mut Wr1) = Wr.rb_mut().two_cols_mut(0, 1);

		let (r, mut gamma_i) = update_and_offdiag_argmax(Wr0.rb_mut(), Wl.rb(), Al.rb(), Ar.rb(), i0, par);

		if k + 1 == n || gamma_i == zero() {
			nothing_to_do = true;

			npiv = 1;
		} else if Ar[(i0, i0)].real().abs() >= &alpha * &gamma_i {
			npiv = 1;
		} else {
			i1 = r.unwrap();

			if rook {
				loop {
					let (s, gamma_r) = update_and_offdiag_argmax(Wr1.rb_mut(), Wl.rb(), Al.rb(), Ar.rb(), i1, par);

					if Ar[(i1, i1)].abs1() >= &alpha * &gamma_r {
						npiv = 1;

						i0 = i1;

						i1 = usize::MAX;

						Wr0.copy_from(&Wr1);

						break;
					} else if s == Some(i0) || gamma_i == gamma_r {
						npiv = 2;

						break;
					} else {
						i0 = i1;

						i1 = s.unwrap();

						gamma_i = gamma_r;

						Wr0.copy_from(&Wr1);
					}
				}
			} else {
				let (_, gamma_r) = update_and_offdiag_argmax(Wr1.rb_mut(), Wl.rb(), Al.rb(), Ar.rb(), i1, par);

				if Ar[(i0, i0)].real().abs() >= (&alpha * &gamma_r) * (&gamma_r / &gamma_i) {
					npiv = 1;
				} else if Ar[(i1, i1)].real().abs() >= &alpha * &gamma_r {
					npiv = 1;

					i0 = i1;

					i1 = usize::MAX;

					Wr0.copy_from(&Wr1);
				} else {
					npiv = 2;
				}
			}
		}

		if npiv == 2 && i0 > i1 {
			perm::swap_cols_idx(Wr.rb_mut(), 0, 1);

			(i0, i1) = (i1, i0);
		}

		let mut Wr = Wr.rb_mut().get_mut(.., ..npiv);

		'next_iter: {
			if i0 != 0 {
				swap_self_adjoint(Ar.rb_mut(), 0, i0);

				perm::swap_rows_idx(Al.rb_mut(), 0, i0);

				perm::swap_rows_idx(Wl.rb_mut(), 0, i0);

				perm::swap_rows_idx(Wr.rb_mut(), 0, i0);
			}

			if npiv == 2 && i1 != 1 {
				swap_self_adjoint(Ar.rb_mut(), 1, i1);

				perm::swap_rows_idx(Al.rb_mut(), 1, i1);

				perm::swap_rows_idx(Wl.rb_mut(), 1, i1);

				perm::swap_rows_idx(Wr.rb_mut(), 1, i1);
			}

			if nothing_to_do {
				break 'next_iter;
			}

			if npiv == 1 {
				let W0 = Wr.rb_mut().col_mut(0);

				let diag = &W0[0].real();

				let diag_inv = &diag.recip();

				subdiag[0] = zero();

				let (_, _, L, mut A) = Ar.rb_mut().split_at_mut(1, 1);

				let W0 = W0.rb().get(1..);

				let n = A.nrows();

				let mut L = L.col_mut(0);

				zip!(W0, L.rb_mut()).for_each(|unzip!(w, a): Zip!(&T, &mut T)| {
					*a = w.mul_real(diag_inv);
				});

				for j in 0..n {
					A[(j, j)] = (A[(j, j)].real() - diag * &L[j].abs2()).to_cplx();
				}
			} else {
				let a00 = Wr[(0, 0)].real();

				let a11 = Wr[(1, 1)].real();

				let a10 = Wr[(1, 0)].copy();

				subdiag[0] = a10.copy();

				subdiag[1] = zero();

				Wr[(1, 0)] = zero();

				Ar[(1, 0)] = zero();

				let d10 = a10.abs();

				let d10_inv = &d10.recip();

				let d00 = a00 * d10_inv;

				let d11 = a11 * d10_inv;

				let t = (&d00 * &d11 - one::<T::Real>()).recip();

				let d10 = a10.mul_real(d10_inv);

				let d = t * d10_inv;

				let (_, _, L, mut A) = Ar.rb_mut().split_at_mut(2, 2);

				let (mut L0, mut L1) = L.two_cols_mut(0, 1);

				let Wr = Wr.rb().get(2.., ..);

				let W0 = Wr.col(0);

				let W1 = Wr.col(1);

				let n = A.nrows();

				for j in 0..n {
					let x0 = &W0[j];

					let x1 = &W1[j];

					let w0 = (x0.mul_real(&d11) - x1 * &d10).mul_real(&d);

					let w1 = (x1.mul_real(&d00) - x0 * d10.conj()).mul_real(&d);

					A[(j, j)] = (&A[(j, j)] - &W0[j] * w0.conj() - &W1[j] * w1.conj()).as_real();

					L0[j] = w0;

					L1[j] = w1;
				}
			}
		}

		let offset = A_left.ncols();

		if npiv == 2 {
			pivots[k] = (offset + i0 + k) | TOP_BIT;

			pivots[k + 1] = (offset + i1 + k) | TOP_BIT;
		} else {
			pivots[k] = offset + i0 + k;
		}

		k += npiv;
	}

	let W = W.rb().get(k.., ..k);

	let (_, _, Al, mut Ar) = A.rb_mut().split_at_mut(k, k);

	let Al = Al.rb();

	linalg::matmul::triangular::matmul(
		Ar.rb_mut(),
		BlockStructure::StrictTriangularLower,
		Accum::Add,
		W,
		BlockStructure::Rectangular,
		Al.adjoint(),
		BlockStructure::Rectangular,
		-one::<T>(),
		par,
	);

	for j in 0..n - k {
		Ar[(j, j)] = Ar[(j, j)].as_real();
	}

	k
}

fn lblt_blocked<T: ComplexField>(
	A: MatMut<'_, T>,
	subdiag: DiagMut<'_, T>,
	pivots: &mut [usize],
	block_size: usize,
	rook: bool,
	diagonal: bool,
	par: Par,
	stack: &mut MemStack,
) {
	let ref alpha = (one::<T::Real>() + from_f64::<T::Real>(17.0).sqrt()) * from_f64::<T::Real>(0.125);

	let mut A = A;

	let mut subdiag = subdiag.column_vector_mut();

	let n = A.nrows();

	let mut k = 0;

	while k < n {
		let (_, _, mut A_left, A_right) = A.rb_mut().split_at_mut(k, k);

		let (mut W, _) = unsafe { temp_mat_uninit::<T, _, _>(n - k, block_size, stack) };

		let W = W.as_mat_mut();

		let next;

		if block_size < 2 || n - k <= block_size {
			lblt_unblocked(
				alpha.copy(),
				A_left.rb_mut(),
				A_right,
				subdiag.rb_mut().get_mut(k..).as_diagonal_mut(),
				&mut pivots[k..],
				rook,
				diagonal,
				par,
			);

			next = n;
		} else {
			let block_size = lblt_blocked_step(
				alpha.copy(),
				W,
				A_left.rb_mut(),
				A_right,
				subdiag.rb_mut().get_mut(k..).as_diagonal_mut(),
				&mut pivots[k..],
				rook,
				diagonal,
				par,
			);

			next = k + block_size;
		}

		let pivots = &pivots[k..next];

		let A_left = A.rb_mut().get_mut(.., ..k);

		if A_left.ncols() > 0 {
			match par {
				Par::Seq => {
					for mut col in A_left.col_iter_mut() {
						for (i, &j) in core::iter::zip(k..next, pivots) {
							let j = j & !TOP_BIT;

							linalg::lu::partial_pivoting::factor::swap_elems(col.rb_mut(), i, j);
						}
					}
				},
				#[cfg(feature = "rayon")]
				Par::Rayon(nthreads) => {
					let nthreads = nthreads.get();

					spindle::for_each(nthreads, A_left.par_col_iter_mut(), |mut col| {
						for (i, &j) in core::iter::zip(k..next, pivots) {
							let j = j & !TOP_BIT;

							linalg::lu::partial_pivoting::factor::swap_elems(col.rb_mut(), i, j);
						}
					});
				},
			}
		}

		k = next;
	}
}

#[inline(never)]

fn lblt_unblocked<T: ComplexField>(
	alpha: T::Real,
	A_left: MatMut<'_, T>,
	A: MatMut<'_, T>,
	subdiag: DiagMut<'_, T>,
	pivots: &mut [usize],
	rook: bool,
	diagonal: bool,
	par: Par,
) {
	let _ = par;

	let mut A = A;

	let mut A_left = A_left;

	let mut subdiag = subdiag;

	let n = A.nrows();

	assert!(all(A.nrows() == n, A.ncols() == n, subdiag.dim() == n));

	let mut k = 0usize;

	while k < n {
		let (_, _, mut L_prev, mut A) = A.rb_mut().split_at_mut(k, k);

		let mut subdiag = subdiag.rb_mut().column_vector_mut().get_mut(k..);

		let A_left = A_left.rb_mut().get_mut(k.., ..);

		let npiv;

		let mut i0 = if diagonal {
			l1_argmax(A.rb().diagonal().column_vector()).0.unwrap()
		} else {
			0
		};

		let mut i1 = usize::MAX;

		let (r, mut gamma_i) = offdiag_argmax(A.rb(), i0);

		let mut nothing_to_do = false;

		if k + 1 == n || gamma_i == zero() {
			nothing_to_do = true;

			npiv = 1;
		} else if A[(i0, i0)].real().abs() >= &alpha * &gamma_i {
			npiv = 1;
		} else {
			i1 = r.unwrap();

			if rook {
				loop {
					let (s, gamma_r) = offdiag_argmax(A.rb(), i1);

					if A[(i1, i1)].abs1() >= &alpha * &gamma_r {
						npiv = 1;

						i0 = i1;

						i1 = usize::MAX;

						break;
					} else if gamma_i == gamma_r {
						npiv = 2;

						break;
					} else {
						i0 = i1;

						i1 = s.unwrap();

						gamma_i = gamma_r;
					}
				}
			} else {
				let (_, gamma_r) = offdiag_argmax(A.rb(), i1);

				if A[(i0, i0)].real().abs() >= (&alpha * &gamma_r) * (&gamma_r / &gamma_i) {
					npiv = 1;
				} else if A[(i1, i1)].real().abs() >= &alpha * &gamma_r {
					npiv = 1;

					i0 = i1;
				} else {
					npiv = 2;
				}
			}
		}

		if npiv == 2 && i0 > i1 {
			(i0, i1) = (i1, i0);
		}

		'next_iter: {
			if i0 != 0 {
				swap_self_adjoint(A.rb_mut(), 0, i0);

				perm::swap_rows_idx(L_prev.rb_mut(), 0, i0);
			}

			if npiv == 2 && i1 != 1 {
				swap_self_adjoint(A.rb_mut(), 1, i1);

				perm::swap_rows_idx(L_prev.rb_mut(), 1, i1);
			}

			if nothing_to_do {
				break 'next_iter;
			}

			if npiv == 1 {
				let diag = A[(0, 0)].real();

				let diag_inv = diag.recip();

				subdiag[0] = zero();

				let (_, _, L, A) = A.rb_mut().split_at_mut(1, 1);

				let L = L.col_mut(0);

				rank1_update(A, L, diag_inv);
			} else {
				let a00 = A[(0, 0)].real();

				let a11 = A[(1, 1)].real();

				let a10 = A[(1, 0)].copy();

				subdiag[0] = a10.copy();

				subdiag[1] = zero();

				A[(1, 0)] = zero();

				let d10 = a10.abs();

				let d10_inv = &d10.recip();

				let d00 = a00 * d10_inv;

				let d11 = a11 * d10_inv;

				let t = (&d00 * &d11 - one::<T::Real>()).recip();

				let d10 = a10.mul_real(d10_inv);

				let d = t * d10_inv;

				let (_, _, L, A) = A.rb_mut().split_at_mut(2, 2);

				let (L0, L1) = L.two_cols_mut(0, 1);

				rank2_update(A, L0, L1, d, d00, d10, d11);
			}
		}

		let offset = A_left.ncols();

		if npiv == 2 {
			pivots[k] = (offset + i0 + k) | TOP_BIT;

			pivots[k + 1] = (offset + i1 + k) | TOP_BIT;
		} else {
			pivots[k] = offset + i0 + k;
		}

		k += npiv;
	}
}

impl<T: ComplexField> Auto<T> for LbltParams {
	fn auto() -> Self {
		Self {
			pivoting: PivotingStrategy::PartialDiag,
			block_size: 64,
			par_threshold: 128 * 128,
			non_exhaustive: NonExhaustive(()),
		}
	}
}

pub fn rank2_update<'a, T: ComplexField>(
	mut A: MatMut<'a, T>,
	mut L0: ColMut<'a, T>,
	mut L1: ColMut<'a, T>,
	d: T::Real,
	d00: T::Real,
	d10: T,
	d11: T::Real,
) {
	if const { T::SIMD_CAPABILITIES.is_simd() } {
		if let (Some(A), Some(L0), Some(L1)) = (
			A.rb_mut().try_as_col_major_mut(),
			L0.rb_mut().try_as_col_major_mut(),
			L1.rb_mut().try_as_col_major_mut(),
		) {
			rank2_update_simd(A, L0, L1, d, d00, d10, d11);
		} else {
			rank2_update_fallback(A, L0, L1, d, d00, d10, d11);
		}
	} else {
		rank2_update_fallback(A, L0, L1, d, d00, d10, d11);
	}
}

pub fn rank2_update_simd<'a, T: ComplexField>(
	A: MatMut<'a, T, usize, usize, ContiguousFwd>,
	L0: ColMut<'a, T, usize, ContiguousFwd>,
	L1: ColMut<'a, T, usize, ContiguousFwd>,
	d: T::Real,
	d00: T::Real,
	d10: T,
	d11: T::Real,
) {
	struct Impl<'a, T: ComplexField> {
		A: MatMut<'a, T, usize, usize, ContiguousFwd>,
		L0: ColMut<'a, T, usize, ContiguousFwd>,
		L1: ColMut<'a, T, usize, ContiguousFwd>,
		d: T::Real,
		d00: T::Real,
		d10: T,
		d11: T::Real,
	}

	impl<T: ComplexField> pulp::WithSimd for Impl<'_, T> {
		type Output = ();

		#[inline(always)]

		fn with_simd<S: pulp::Simd>(self, simd: S) {
			let Self {
				mut A,
				mut L0,
				mut L1,
				d,
				d00,
				d10,
				d11,
			} = self;

			let n = A.nrows();

			for j in 0..n {
				let x0 = &L0[j];

				let x1 = &L1[j];

				let w0 = (x0.mul_real(&d11) - x1 * &d10).mul_real(&d);

				let w1 = (x1.mul_real(&d00) - x0 * d10.conj()).mul_real(&d);

				with_dim!({
					let subrange_len = n - j;
				});

				{
					let mut A = A.rb_mut().get_mut(j.., j).as_row_shape_mut(subrange_len);

					let L0 = L0.rb().get(j..).as_row_shape(subrange_len);

					let L1 = L1.rb().get(j..).as_row_shape(subrange_len);

					let simd = SimdCtx::<T, S>::new(T::simd_ctx(simd), subrange_len);

					let (head, body, tail) = simd.indices();

					let w0_conj = w0.conj();

					let w1_conj = w1.conj();

					let w0_conj_neg = -w0_conj;

					let w1_conj_neg = -w1_conj;

					let w0_splat = simd.splat(&w0_conj_neg);

					let w1_splat = simd.splat(&w1_conj_neg);

					if let Some(i) = head {
						let mut acc = simd.read(A.rb(), i);

						let l0_val = simd.read(L0, i);

						let l1_val = simd.read(L1, i);

						acc = simd.mul_add(l0_val, w0_splat, acc);

						acc = simd.mul_add(l1_val, w1_splat, acc);

						simd.write(A.rb_mut(), i, acc);
					}

					for i in body.clone() {
						let mut acc = simd.read(A.rb(), i);

						let l0_val = simd.read(L0, i);

						let l1_val = simd.read(L1, i);

						acc = simd.mul_add(l0_val, w0_splat, acc);

						acc = simd.mul_add(l1_val, w1_splat, acc);

						simd.write(A.rb_mut(), i, acc);
					}

					if let Some(i) = tail {
						let mut acc = simd.read(A.rb(), i);

						let l0_val = simd.read(L0, i);

						let l1_val = simd.read(L1, i);

						acc = simd.mul_add(l0_val, w0_splat, acc);

						acc = simd.mul_add(l1_val, w1_splat, acc);

						simd.write(A.rb_mut(), i, acc);
					}
				}

				A[(j, j)] = A[(j, j)].real().to_cplx();

				L0[j] = w0;

				L1[j] = w1;
			}
		}
	}

	dispatch!(Impl { A, L0, L1, d, d00, d10, d11 }, Impl, T)
}

pub fn rank2_update_fallback<'a, T: ComplexField>(
	mut A: MatMut<'a, T>,
	mut L0: ColMut<'a, T>,
	mut L1: ColMut<'a, T>,
	d: T::Real,
	d00: T::Real,
	d10: T,
	d11: T::Real,
) {
	let n = A.nrows();

	for j in 0..n {
		let x0 = &L0[j].copy();

		let x1 = &L1[j].copy();

		let w0 = (x0.mul_real(&d11) - x1 * &d10).mul_real(&d);

		let w1 = (x1.mul_real(&d00) - x0 * d10.conj()).mul_real(&d);

		for i in j..n {
			A[(i, j)] = &A[(i, j)] - &L0[i] * w0.conj() - &L1[i] * w1.conj();
		}

		A[(j, j)] = A[(j, j)].as_real();

		L0[j] = w0;

		L1[j] = w1;
	}
}

pub fn rank1_update<'a, T: ComplexField>(mut A: MatMut<'a, T>, mut L0: ColMut<'a, T>, d: T::Real) {
	if const { T::SIMD_CAPABILITIES.is_simd() } {
		if let (Some(A), Some(L0)) = (A.rb_mut().try_as_col_major_mut(), L0.rb_mut().try_as_col_major_mut()) {
			rank1_update_simd(A, L0, d);
		} else {
			rank1_update_fallback(A, L0, d);
		}
	} else {
		rank1_update_fallback(A, L0, d);
	}
}

pub fn rank1_update_simd<'a, T: ComplexField>(A: MatMut<'a, T, usize, usize, ContiguousFwd>, L0: ColMut<'a, T, usize, ContiguousFwd>, d: T::Real) {
	struct Impl<'a, T: ComplexField> {
		A: MatMut<'a, T, usize, usize, ContiguousFwd>,
		L0: ColMut<'a, T, usize, ContiguousFwd>,
		d: T::Real,
	}

	impl<T: ComplexField> pulp::WithSimd for Impl<'_, T> {
		type Output = ();

		#[inline(always)]

		fn with_simd<S: pulp::Simd>(self, simd: S) {
			let Self { mut A, mut L0, d } = self;

			let n = A.nrows();

			for j in 0..n {
				let x0 = &L0[j];

				let w0 = x0.mul_real(&d);

				with_dim!({
					let subrange_len = n - j;
				});

				{
					let mut A = A.rb_mut().get_mut(j.., j).as_row_shape_mut(subrange_len);

					let L0 = L0.rb().get(j..).as_row_shape(subrange_len);

					let simd = SimdCtx::<T, S>::new(T::simd_ctx(simd), subrange_len);

					let (head, body, tail) = simd.indices();

					let w0_conj = w0.conj();

					let w0_conj_neg = -w0_conj;

					let w0_splat = simd.splat(&w0_conj_neg);

					if let Some(i) = head {
						let mut acc = simd.read(A.rb(), i);

						let l0_val = simd.read(L0, i);

						acc = simd.mul_add(l0_val, w0_splat, acc);

						simd.write(A.rb_mut(), i, acc);
					}

					for i in body.clone() {
						let mut acc = simd.read(A.rb(), i);

						let l0_val = simd.read(L0, i);

						acc = simd.mul_add(l0_val, w0_splat, acc);

						simd.write(A.rb_mut(), i, acc);
					}

					if let Some(i) = tail {
						let mut acc = simd.read(A.rb(), i);

						let l0_val = simd.read(L0, i);

						acc = simd.mul_add(l0_val, w0_splat, acc);

						simd.write(A.rb_mut(), i, acc);
					}
				}

				A[(j, j)] = A[(j, j)].real().to_cplx();

				L0[j] = w0;
			}
		}
	}

	dispatch!(Impl { A, L0, d }, Impl, T)
}

pub fn rank1_update_fallback<'a, T: ComplexField>(mut A: MatMut<'a, T>, mut L0: ColMut<'a, T>, d: T::Real) {
	let n = A.nrows();

	for j in 0..n {
		let x0 = &L0[j];

		let w0 = x0.mul_real(&d);

		for i in j..n {
			A[(i, j)] = &A[(i, j)] - &L0[i] * w0.conj();
		}

		A[(j, j)] = A[(j, j)].real().to_cplx();

		L0[j] = w0;
	}
}

/// computes the layout of required workspace for performing an $LBL^\top$
/// decomposition

pub fn cholesky_in_place_scratch<I: Index, T: ComplexField>(dim: usize, par: Par, params: Spec<LbltParams, T>) -> StackReq {
	let params = params.config;

	let _ = par;

	let mut bs = params.block_size;

	if bs < 2 || dim <= bs {
		bs = 0;
	}

	StackReq::new::<usize>(dim).and(temp_mat_scratch::<T>(dim, bs))
}

/// info about the result of the $LBL^\top$ factorization
#[derive(Copy, Clone, Debug)]

pub struct LbltInfo {
	/// number of pivoting transpositions
	pub transposition_count: usize,
}

/// computes the $LBL^\top$ factorization of $A$ and stores the factorization in `matrix` and
/// `subdiag`
///
/// the diagonal of the block diagonal matrix is stored on the diagonal
/// of `matrix`, while the subdiagonal elements of the blocks are stored in `subdiag`
///
/// # panics
///
/// panics if the input matrix is not square
///
/// this can also panic if the provided memory in `stack` is insufficient (see
/// [`cholesky_in_place_scratch`]).
#[track_caller]

pub fn cholesky_in_place<'out, I: Index, T: ComplexField>(
	A: MatMut<'_, T>,
	subdiag: DiagMut<'_, T>,
	perm: &'out mut [I],
	perm_inv: &'out mut [I],
	par: Par,
	stack: &mut MemStack,
	params: Spec<LbltParams, T>,
) -> (LbltInfo, PermRef<'out, I>) {
	let params = params.config;

	let truncate = <I::Signed as SignedIndex>::truncate;

	let n = A.nrows();

	assert!(all(A.nrows() == A.ncols(), subdiag.dim() == n, perm.len() == n, perm_inv.len() == n));

	#[cfg(feature = "perf-warn")]
	if A.row_stride().unsigned_abs() != 1 && crate::__perf_warn!(CHOLESKY_WARN) {
		if A.col_stride().unsigned_abs() == 1 {
			log::warn!(
				target : "faer_perf",
				"$LBL^\top$ decomposition prefers column-major
			matrix. Found row-major matrix."
			);
		} else {
			log::warn!(
				target : "faer_perf",
				"$LBL^\top$ decomposition prefers column-major
			matrix. Found matrix with generic strides."
			);
		}
	}

	let (mut pivots, stack) = stack.make_with::<usize>(n, |_| 0);

	let pivots = &mut *pivots;

	let mut bs = params.block_size;

	if bs < 2 || n <= bs {
		bs = 0;
	}

	let (rook, diagonal) = match params.pivoting {
		PivotingStrategy::Partial => (false, false),
		PivotingStrategy::PartialDiag => (false, true),
		PivotingStrategy::Rook => (true, false),
		PivotingStrategy::RookDiag => (true, true),
		_ => (false, false),
	};

	if params.pivoting == PivotingStrategy::Full {
		lblt_full_piv(A, subdiag, pivots, par, params);
	} else {
		lblt_blocked(A, subdiag, pivots, bs, rook, diagonal, par, stack);
	}

	for (i, p) in perm.iter_mut().enumerate() {
		*p = I::from_signed(truncate(i));
	}

	let mut transposition_count = 0usize;

	for i in 0..n {
		let p = pivots[i] & !TOP_BIT;

		if i != p {
			transposition_count += 1;
		}

		perm.swap(i, p);
	}

	for (i, &p) in perm.iter().enumerate() {
		perm_inv[p.to_signed().zx()] = I::from_signed(truncate(i));
	}

	(LbltInfo { transposition_count }, unsafe { PermRef::new_unchecked(perm, perm_inv, n) })
}
