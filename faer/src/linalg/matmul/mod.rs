//! matrix multiplication

use super::temp_mat_scratch;
use crate::col::ColRef;
use crate::internal_prelude::*;
use crate::mat::{MatMut, MatRef};
use crate::row::RowRef;
use crate::utils::bound::Dim;
use crate::utils::simd::SimdCtx;
use crate::{Conj, ContiguousFwd, Par, Shape};
use core::mem::MaybeUninit;
use dyn_stack::{MemBuffer, MemStack};
use equator::assert;
use faer_macros::math;
use faer_traits::{ByRef, ComplexField, Conjugate};
use pulp::Simd;
use reborrow::*;

const NANO_GEMM_THRESHOLD: usize = 16 * 16 * 16;

pub(crate) mod internal;

/// triangular matrix multiplication module, where some of the operands are treated as triangular
/// matrices
pub mod triangular;

mod matmul_shared {
	use super::*;

	pub const NC: usize = 2048;
	pub const KC: usize = 128;

	pub struct SimdLaneCount<T: ComplexField> {
		pub __marker: core::marker::PhantomData<fn() -> T>,
	}
	impl<T: ComplexField> pulp::WithSimd for SimdLaneCount<T> {
		type Output = usize;

		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
			let _ = simd;
			core::mem::size_of::<T::SimdVec<S>>() / core::mem::size_of::<T>()
		}
	}

	pub struct MicroKernelShape<T: ComplexField> {
		pub __marker: core::marker::PhantomData<fn() -> T>,
	}

	impl<T: ComplexField> MicroKernelShape<T> {
		pub const IS_1X1: bool = Self::MAX_MR_DIV_N == 2 && Self::MAX_NR == 1;
		pub const IS_2X1: bool = Self::MAX_MR_DIV_N == 2 && Self::MAX_NR == 1;
		pub const IS_2X2: bool = Self::MAX_MR_DIV_N == 2 && Self::MAX_NR == 2;
		pub const MAX_MR_DIV_N: usize = Self::SHAPE.0;
		pub const MAX_NR: usize = Self::SHAPE.1;
		pub const SHAPE: (usize, usize) = {
			if const { size_of::<T>() / size_of::<T::Unit>() <= 2 } {
				(2, 2)
			} else if const { size_of::<T>() / size_of::<T::Unit>() == 4 } {
				(2, 1)
			} else {
				(1, 1)
			}
		};
	}
}

mod matmul_vertical {
	use super::*;
	use matmul_shared::*;

	struct Ukr<'a, const MR_DIV_N: usize, const NR: usize, T: ComplexField> {
		dst: MatMut<'a, T, usize, usize, ContiguousFwd>,
		a: MatRef<'a, T, usize, usize, ContiguousFwd>,
		b: MatRef<'a, T, usize, usize>,
		conj_lhs: Conj,
		conj_rhs: Conj,
		alpha: &'a T,
		beta: Accum,
	}

	impl<const MR_DIV_N: usize, const NR: usize, T: ComplexField> pulp::WithSimd for Ukr<'_, MR_DIV_N, NR, T> {
		type Output = ();

		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
			let Self {
				dst,
				a,
				b,
				conj_lhs,
				conj_rhs,
				alpha,
				beta,
			} = self;

			with_dim!(M, a.nrows());
			with_dim!(N, b.ncols());
			with_dim!(K, a.ncols());
			let a = a.as_shape(M, K);
			let b = b.as_shape(K, N);
			let mut dst = dst.as_shape_mut(M, N);

			let simd = SimdCtx::<T, S>::new_force_mask(T::simd_ctx(simd), M);
			let (_, body, tail) = simd.indices();
			let tail = tail.unwrap();

			let mut local_acc = [[simd.zero(); MR_DIV_N]; NR];

			if conj_lhs == conj_rhs {
				for depth in K.indices() {
					let mut a_uninit = [MaybeUninit::<T::SimdVec<S>>::uninit(); MR_DIV_N];

					for (dst, src) in core::iter::zip(&mut a_uninit, body.clone()) {
						*dst = MaybeUninit::new(simd.read(a.col(depth), src));
					}
					a_uninit[MR_DIV_N - 1] = MaybeUninit::new(simd.read(a.col(depth), tail));

					let a: [T::SimdVec<S>; MR_DIV_N] =
						unsafe { crate::hacks::transmute::<[MaybeUninit<T::SimdVec<S>>; MR_DIV_N], [T::SimdVec<S>; MR_DIV_N]>(a_uninit) };

					for j in N.indices() {
						let b = simd.splat(&b[(depth, j)]);

						for i in 0..MR_DIV_N {
							let local_acc = &mut local_acc[*j][i];
							*local_acc = simd.mul_add(b, a[i], *local_acc);
						}
					}
				}
			} else {
				for depth in K.indices() {
					let mut a_uninit = [MaybeUninit::<T::SimdVec<S>>::uninit(); MR_DIV_N];

					for (dst, src) in core::iter::zip(&mut a_uninit, body.clone()) {
						*dst = MaybeUninit::new(simd.read(a.col(depth), src));
					}
					a_uninit[MR_DIV_N - 1] = MaybeUninit::new(simd.read(a.col(depth), tail));

					let a: [T::SimdVec<S>; MR_DIV_N] =
						unsafe { crate::hacks::transmute::<[MaybeUninit<T::SimdVec<S>>; MR_DIV_N], [T::SimdVec<S>; MR_DIV_N]>(a_uninit) };

					for j in N.indices() {
						let b = simd.splat(&b[(depth, j)]);

						for i in 0..MR_DIV_N {
							let local_acc = &mut local_acc[*j][i];
							*local_acc = simd.conj_mul_add(b, a[i], *local_acc);
						}
					}
				}
			}

			if conj_lhs.is_conj() {
				for x in &mut local_acc {
					for x in x {
						*x = simd.conj(*x);
					}
				}
			}

			let alpha = simd.splat(alpha);

			match beta {
				Accum::Add => {
					for (result, j) in core::iter::zip(&local_acc, N.indices()) {
						for (result, i) in core::iter::zip(result, body.clone()) {
							let mut val = simd.read(dst.rb().col(j), i);
							val = simd.mul_add(alpha, *result, val);
							simd.write(dst.rb_mut().col_mut(j), i, val);
						}
						let i = tail;
						let result = &result[MR_DIV_N - 1];

						let mut val = simd.read(dst.rb().col(j), i);
						val = simd.mul_add(alpha, *result, val);
						simd.write(dst.rb_mut().col_mut(j), i, val);
					}
				},
				Accum::Replace => {
					for (result, j) in core::iter::zip(&local_acc, N.indices()) {
						for (result, i) in core::iter::zip(result, body.clone()) {
							let val = simd.mul(alpha, *result);
							simd.write(dst.rb_mut().col_mut(j), i, val);
						}

						let i = tail;
						let result = &result[MR_DIV_N - 1];

						let val = simd.mul(alpha, *result);
						simd.write(dst.rb_mut().col_mut(j), i, val);
					}
				},
			}
		}
	}

	#[math]
	pub fn matmul_simd<'M, 'N, 'K, T: ComplexField>(
		dst: MatMut<'_, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
		beta: Accum,
		lhs: MatRef<'_, T, Dim<'M>, Dim<'K>, ContiguousFwd>,
		conj_lhs: Conj,
		rhs: MatRef<'_, T, Dim<'K>, Dim<'N>>,
		conj_rhs: Conj,
		alpha: &T,
		par: Par,
	) {
		let dst = dst.as_dyn_mut();
		let lhs = lhs.as_dyn();
		let rhs = rhs.as_dyn();

		let (m, n) = dst.shape();
		let k = lhs.ncols();

		let arch = T::Arch::default();

		let lane_count = arch.dispatch(SimdLaneCount::<T> {
			__marker: core::marker::PhantomData,
		});

		let nr = MicroKernelShape::<T>::MAX_NR;
		let mr_div_n = MicroKernelShape::<T>::MAX_MR_DIV_N;
		let mr = mr_div_n * lane_count;

		let mut col_outer = 0;
		while col_outer < n {
			let n_chunk = Ord::min(n - col_outer, NC);
			let mut beta = beta;

			let mut depth = 0;
			while depth < k {
				let k_chunk = Ord::min(k - depth, KC);

				let job = |row: usize, col_inner: usize| {
					let nrows = Ord::min(m - row, mr);
					let ukr_i = nrows.div_ceil(lane_count);
					let ncols = Ord::min(n_chunk - col_inner, nr);
					let ukr_j = ncols;

					let dst = unsafe { dst.rb().const_cast() }.submatrix_mut(row, col_outer + col_inner, nrows, ncols);
					let a = lhs.submatrix(row, depth, nrows, k_chunk);
					let b = rhs.submatrix(depth, col_outer + col_inner, k_chunk, ncols);

					macro_rules! call {
						($M: expr, $N: expr) => {
							arch.dispatch(Ukr::<'_, $M, $N, T> {
								dst,
								a,
								b,
								conj_lhs,
								conj_rhs,
								alpha,
								beta,
							})
						};
					}
					if const { MicroKernelShape::<T>::IS_2X2 } {
						match (ukr_i, ukr_j) {
							(2, 2) => call!(2, 2),
							(1, 2) => call!(1, 2),
							(2, 1) => call!(2, 1),
							(1, 1) => call!(1, 1),
							_ => unreachable!(),
						}
					} else if const { MicroKernelShape::<T>::IS_2X1 } {
						match (ukr_i, ukr_j) {
							(2, 1) => call!(2, 1),
							(1, 1) => call!(1, 1),
							_ => unreachable!(),
						}
					} else if const { MicroKernelShape::<T>::IS_1X1 } {
						call!(1, 1)
					} else {
						unreachable!()
					}
				};

				let job_count = m.div_ceil(mr) * n_chunk.div_ceil(nr);
				let d = n_chunk.div_ceil(nr);
				match par {
					Par::Seq => {
						for job_idx in 0..job_count {
							let col_inner = nr * (job_idx % d);
							let row = mr * (job_idx / d);
							job(row, col_inner);
						}
					},
					#[cfg(feature = "rayon")]
					Par::Rayon(nthreads) => {
						let nthreads = nthreads.get();
						use rayon::prelude::*;

						let job_idx = core::sync::atomic::AtomicUsize::new(0);

						(0..nthreads).into_par_iter().for_each(|_| {
							loop {
								let job_idx = job_idx.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
								if job_idx < job_count {
									let col_inner = nr * (job_idx % d);
									let row = mr * (job_idx / d);
									job(row, col_inner);
								} else {
									return;
								}
							}
						});
					},
				}

				beta = Accum::Add;
				depth += k_chunk;
			}
			col_outer += n_chunk;
		}
	}
}

mod matmul_horizontal {
	use super::*;
	use matmul_shared::*;

	struct Ukr<'a, const MR: usize, const NR: usize, T: ComplexField> {
		dst: MatMut<'a, T, usize, usize>,
		a: MatRef<'a, T, usize, usize, isize, ContiguousFwd>,
		b: MatRef<'a, T, usize, usize, ContiguousFwd, isize>,
		conj_lhs: Conj,
		conj_rhs: Conj,
		alpha: &'a T,
		beta: Accum,
	}

	impl<const MR: usize, const NR: usize, T: ComplexField> pulp::WithSimd for Ukr<'_, MR, NR, T> {
		type Output = ();

		#[math]
		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
			let Self {
				dst,
				a,
				b,
				conj_lhs,
				conj_rhs,
				alpha,
				beta,
			} = self;

			with_dim!(M, a.nrows());
			with_dim!(N, b.ncols());
			with_dim!(K, a.ncols());
			let a = a.as_shape(M, K);
			let b = b.as_shape(K, N);
			let mut dst = dst.as_shape_mut(M, N);

			let simd = SimdCtx::<T, S>::new(T::simd_ctx(simd), K);
			let (_, body, tail) = simd.indices();

			let mut local_acc = [[simd.zero(); MR]; NR];
			let mut is = [M.idx(0usize); MR];
			let mut js = [N.idx(0usize); NR];

			for (idx, i) in is.iter_mut().enumerate() {
				*i = M.idx(idx);
			}
			for (idx, j) in js.iter_mut().enumerate() {
				*j = N.idx(idx);
			}

			if conj_lhs == conj_rhs {
				macro_rules! do_it {
					($depth: expr) => {{
						let depth = $depth;
						let a = is.map(
							#[inline(always)]
							|i| simd.read(a.row(i).transpose(), depth),
						);
						let b = js.map(
							#[inline(always)]
							|j| simd.read(b.col(j), depth),
						);

						for i in 0..MR {
							for j in 0..NR {
								local_acc[j][i] = simd.mul_add(b[j], a[i], local_acc[j][i]);
							}
						}
					}};
				}
				for depth in body {
					do_it!(depth);
				}
				if let Some(depth) = tail {
					do_it!(depth);
				}
			} else {
				macro_rules! do_it {
					($depth: expr) => {{
						let depth = $depth;
						let a = is.map(
							#[inline(always)]
							|i| simd.read(a.row(i).transpose(), depth),
						);
						let b = js.map(
							#[inline(always)]
							|j| simd.read(b.col(j), depth),
						);

						for i in 0..MR {
							for j in 0..NR {
								local_acc[j][i] = simd.conj_mul_add(b[j], a[i], local_acc[j][i]);
							}
						}
					}};
				}
				for depth in body {
					do_it!(depth);
				}
				if let Some(depth) = tail {
					do_it!(depth);
				}
			}

			if conj_lhs.is_conj() {
				for x in &mut local_acc {
					for x in x {
						*x = simd.conj(*x);
					}
				}
			}
			let result = local_acc;
			let result = result.map(
				#[inline(always)]
				|result| {
					result.map(
						#[inline(always)]
						|result| simd.reduce_sum(result),
					)
				},
			);

			let alpha = copy(*alpha);
			match beta {
				Accum::Add => {
					for (result, j) in core::iter::zip(&result, js) {
						for (result, i) in core::iter::zip(result, is) {
							dst[(i, j)] = alpha * *result + dst[(i, j)];
						}
					}
				},
				Accum::Replace => {
					for (result, j) in core::iter::zip(&result, js) {
						for (result, i) in core::iter::zip(result, is) {
							dst[(i, j)] = alpha * *result;
						}
					}
				},
			}
		}
	}

	#[math]
	pub fn matmul_simd<'M, 'N, 'K, T: ComplexField>(
		dst: MatMut<'_, T, Dim<'M>, Dim<'N>>,
		beta: Accum,
		lhs: MatRef<'_, T, Dim<'M>, Dim<'K>, isize, ContiguousFwd>,
		conj_lhs: Conj,
		rhs: MatRef<'_, T, Dim<'K>, Dim<'N>, ContiguousFwd, isize>,
		conj_rhs: Conj,
		alpha: &T,
		par: Par,
	) {
		let dst = dst.as_dyn_mut();
		let lhs = lhs.as_dyn();
		let rhs = rhs.as_dyn();

		let (m, n) = dst.shape();
		let k = lhs.ncols();

		let nr = MicroKernelShape::<T>::MAX_NR;
		let mr = MicroKernelShape::<T>::MAX_MR_DIV_N;

		let arch = T::Arch::default();

		let lane_count = arch.dispatch(SimdLaneCount::<T> {
			__marker: core::marker::PhantomData,
		});
		let kc = KC * lane_count;

		let mut col_outer = 0;
		while col_outer < n {
			let n_chunk = Ord::min(n - col_outer, NC);

			let mut beta = beta;
			let mut depth = 0;
			while depth < k {
				let k_chunk = Ord::min(k - depth, kc);

				let job = |row: usize, col_inner: usize| {
					let nrows = Ord::min(m - row, mr);
					let ukr_i = nrows;
					let ncols = Ord::min(n_chunk - col_inner, nr);
					let ukr_j = ncols;

					let dst = unsafe { dst.rb().const_cast() }.submatrix_mut(row, col_outer + col_inner, nrows, ncols);
					let a = lhs.submatrix(row, depth, nrows, k_chunk);
					let b = rhs.submatrix(depth, col_outer + col_inner, k_chunk, ncols);

					macro_rules! call {
						($M: expr, $N: expr) => {
							arch.dispatch(Ukr::<'_, $M, $N, T> {
								dst,
								a,
								b,
								conj_lhs,
								conj_rhs,
								alpha,
								beta,
							})
						};
					}
					if const { MicroKernelShape::<T>::IS_2X2 } {
						match (ukr_i, ukr_j) {
							(2, 2) => call!(2, 2),
							(1, 2) => call!(1, 2),
							(2, 1) => call!(2, 1),
							(1, 1) => call!(1, 1),
							_ => unreachable!(),
						}
					} else if const { MicroKernelShape::<T>::IS_2X1 } {
						match (ukr_i, ukr_j) {
							(2, 1) => call!(2, 1),
							(1, 1) => call!(1, 1),
							_ => unreachable!(),
						}
					} else if const { MicroKernelShape::<T>::IS_1X1 } {
						call!(1, 1)
					} else {
						unreachable!()
					}
				};

				let job_count = m.div_ceil(mr) * n.div_ceil(nr);
				let d = n.div_ceil(nr);
				match par {
					Par::Seq => {
						for job_idx in 0..job_count {
							let col_inner = nr * (job_idx % d);
							let row = mr * (job_idx / d);
							job(row, col_inner);
						}
					},
					#[cfg(feature = "rayon")]
					Par::Rayon(nthreads) => {
						let nthreads = nthreads.get();
						use rayon::prelude::*;

						let job_idx = core::sync::atomic::AtomicUsize::new(0);

						(0..nthreads).into_par_iter().for_each(|_| {
							loop {
								let job_idx = job_idx.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
								if job_idx < job_count {
									let col_inner = nr * (job_idx % d);
									let row = mr * (job_idx / d);
									job(row, col_inner);
								} else {
									return;
								}
							}
						});
					},
				}

				beta = Accum::Add;
				depth += k_chunk;
			}
			col_outer += n_chunk;
		}
	}
}

/// dot product
pub mod dot {
	use super::*;
	use faer_traits::SimdArch;

	/// returns `lhs * rhs`, implicitly conjugating the operands if needed
	pub fn inner_prod<K: Shape, T: ComplexField>(lhs: RowRef<T, K>, conj_lhs: Conj, rhs: ColRef<T, K>, conj_rhs: Conj) -> T {
		#[math]
		pub fn imp<'K, T: ComplexField>(lhs: RowRef<T, Dim<'K>>, conj_lhs: Conj, rhs: ColRef<T, Dim<'K>>, conj_rhs: Conj) -> T {
			if try_const! { T::SIMD_CAPABILITIES.is_simd() } {
				if let (Some(lhs), Some(rhs)) = (lhs.try_as_row_major(), rhs.try_as_col_major()) {
					inner_prod_slice::<T>(lhs.ncols(), lhs.transpose(), conj_lhs, rhs, conj_rhs)
				} else {
					inner_prod_schoolbook(lhs, conj_lhs, rhs, conj_rhs)
				}
			} else {
				inner_prod_schoolbook(lhs, conj_lhs, rhs, conj_rhs)
			}
		}

		with_dim!(K, lhs.ncols().unbound());

		imp(lhs.as_col_shape(K), conj_lhs, rhs.as_row_shape(K), conj_rhs)
	}

	#[inline(always)]
	#[math]
	fn inner_prod_slice<'K, T: ComplexField>(
		len: Dim<'K>,
		lhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
		conj_lhs: Conj,
		rhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
		conj_rhs: Conj,
	) -> T {
		struct Impl<'a, 'K, T: ComplexField> {
			len: Dim<'K>,
			lhs: ColRef<'a, T, Dim<'K>, ContiguousFwd>,
			conj_lhs: Conj,
			rhs: ColRef<'a, T, Dim<'K>, ContiguousFwd>,
			conj_rhs: Conj,
		}
		impl<'a, 'K, T: ComplexField> pulp::WithSimd for Impl<'_, '_, T> {
			type Output = T;

			#[inline(always)]
			fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
				let Self {
					len,
					lhs,
					conj_lhs,
					rhs,
					conj_rhs,
				} = self;

				let simd = SimdCtx::new(T::simd_ctx(simd), len);

				let mut tmp = if conj_lhs == conj_rhs {
					inner_prod_no_conj_simd::<T, S>(simd, lhs, rhs)
				} else {
					inner_prod_conj_lhs_simd::<T, S>(simd, lhs, rhs)
				};

				if conj_rhs == Conj::Yes {
					tmp = conj(tmp);
				}
				tmp
			}
		}

		dispatch!(
			Impl {
				len,
				lhs,
				rhs,
				conj_lhs,
				conj_rhs
			},
			Impl,
			T
		)
	}

	#[inline(always)]
	pub(crate) fn inner_prod_no_conj_simd<'K, T: ComplexField, S: Simd>(
		simd: SimdCtx<'K, T, S>,
		lhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
		rhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
	) -> T {
		let mut acc0 = simd.zero();
		let mut acc1 = simd.zero();
		let mut acc2 = simd.zero();
		let mut acc3 = simd.zero();

		let (head, idx4, idx, tail) = simd.batch_indices::<4>();

		if let Some(i0) = head {
			let l0 = simd.read(lhs, i0);
			let r0 = simd.read(rhs, i0);

			acc0 = simd.mul_add(l0, r0, acc0);
		}
		for [i0, i1, i2, i3] in idx4 {
			let l0 = simd.read(lhs, i0);
			let l1 = simd.read(lhs, i1);
			let l2 = simd.read(lhs, i2);
			let l3 = simd.read(lhs, i3);

			let r0 = simd.read(rhs, i0);
			let r1 = simd.read(rhs, i1);
			let r2 = simd.read(rhs, i2);
			let r3 = simd.read(rhs, i3);

			acc0 = simd.mul_add(l0, r0, acc0);
			acc1 = simd.mul_add(l1, r1, acc1);
			acc2 = simd.mul_add(l2, r2, acc2);
			acc3 = simd.mul_add(l3, r3, acc3);
		}
		for i0 in idx {
			let l0 = simd.read(lhs, i0);
			let r0 = simd.read(rhs, i0);

			acc0 = simd.mul_add(l0, r0, acc0);
		}
		if let Some(i0) = tail {
			let l0 = simd.read(lhs, i0);
			let r0 = simd.read(rhs, i0);

			acc0 = simd.mul_add(l0, r0, acc0);
		}
		acc0 = simd.add(acc0, acc1);
		acc2 = simd.add(acc2, acc3);
		acc0 = simd.add(acc0, acc2);

		simd.reduce_sum(acc0)
	}

	#[inline(always)]
	pub(crate) fn inner_prod_conj_lhs_simd<'K, T: ComplexField, S: Simd>(
		simd: SimdCtx<'K, T, S>,
		lhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
		rhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
	) -> T {
		let mut acc0 = simd.zero();
		let mut acc1 = simd.zero();
		let mut acc2 = simd.zero();
		let mut acc3 = simd.zero();

		let (head, idx4, idx, tail) = simd.batch_indices::<4>();

		if let Some(i0) = head {
			let l0 = simd.read(lhs, i0);
			let r0 = simd.read(rhs, i0);

			acc0 = simd.conj_mul_add(l0, r0, acc0);
		}
		for [i0, i1, i2, i3] in idx4 {
			let l0 = simd.read(lhs, i0);
			let l1 = simd.read(lhs, i1);
			let l2 = simd.read(lhs, i2);
			let l3 = simd.read(lhs, i3);

			let r0 = simd.read(rhs, i0);
			let r1 = simd.read(rhs, i1);
			let r2 = simd.read(rhs, i2);
			let r3 = simd.read(rhs, i3);

			acc0 = simd.conj_mul_add(l0, r0, acc0);
			acc1 = simd.conj_mul_add(l1, r1, acc1);
			acc2 = simd.conj_mul_add(l2, r2, acc2);
			acc3 = simd.conj_mul_add(l3, r3, acc3);
		}
		for i0 in idx {
			let l0 = simd.read(lhs, i0);
			let r0 = simd.read(rhs, i0);

			acc0 = simd.conj_mul_add(l0, r0, acc0);
		}
		if let Some(i0) = tail {
			let l0 = simd.read(lhs, i0);
			let r0 = simd.read(rhs, i0);

			acc0 = simd.conj_mul_add(l0, r0, acc0);
		}
		acc0 = simd.add(acc0, acc1);
		acc2 = simd.add(acc2, acc3);
		acc0 = simd.add(acc0, acc2);

		simd.reduce_sum(acc0)
	}

	#[math]
	pub(crate) fn inner_prod_schoolbook<'K, T: ComplexField>(
		lhs: RowRef<'_, T, Dim<'K>>,
		conj_lhs: Conj,
		rhs: ColRef<'_, T, Dim<'K>>,
		conj_rhs: Conj,
	) -> T {
		let mut acc = zero();

		for k in lhs.ncols().indices() {
			if try_const! { T::IS_REAL } {
				acc = lhs[k] * rhs[k] + acc;
			} else {
				match (conj_lhs, conj_rhs) {
					(Conj::No, Conj::No) => {
						acc = lhs[k] * rhs[k] + acc;
					},
					(Conj::No, Conj::Yes) => {
						acc = lhs[k] * conj(rhs[k]) + acc;
					},
					(Conj::Yes, Conj::No) => {
						acc = conj(lhs[k]) * rhs[k] + acc;
					},
					(Conj::Yes, Conj::Yes) => {
						acc = conj(lhs[k] * rhs[k]) + acc;
					},
				}
			}
		}

		acc
	}
}

mod matvec_rowmajor {
	use super::*;
	use crate::col::ColMut;
	use faer_traits::SimdArch;

	#[math]
	pub fn matvec<'M, 'K, T: ComplexField>(
		dst: ColMut<'_, T, Dim<'M>>,
		beta: Accum,
		lhs: MatRef<'_, T, Dim<'M>, Dim<'K>, isize, ContiguousFwd>,
		conj_lhs: Conj,
		rhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
		conj_rhs: Conj,
		alpha: &T,
		par: Par,
	) {
		core::assert!(try_const! { T::SIMD_CAPABILITIES.is_simd() });
		let size = *lhs.nrows() * *lhs.ncols();
		let par = if size < 256 * 256usize { Par::Seq } else { par };

		match par {
			Par::Seq => {
				pub struct Impl<'a, 'M, 'K, T: ComplexField> {
					dst: ColMut<'a, T, Dim<'M>>,
					beta: Accum,
					lhs: MatRef<'a, T, Dim<'M>, Dim<'K>, isize, ContiguousFwd>,
					conj_lhs: Conj,
					rhs: ColRef<'a, T, Dim<'K>, ContiguousFwd>,
					conj_rhs: Conj,
					alpha: &'a T,
				}

				impl<'a, 'M, 'K, T: ComplexField> pulp::WithSimd for Impl<'a, 'M, 'K, T> {
					type Output = ();

					#[inline(always)]
					fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
						let Self {
							dst,
							beta,
							lhs,
							conj_lhs,
							rhs,
							conj_rhs,
							alpha,
						} = self;
						let simd = T::simd_ctx(simd);
						let mut dst = dst;

						let K = lhs.ncols();
						let simd = SimdCtx::new(simd, K);
						for i in lhs.nrows().indices() {
							let dst = &mut dst[i];
							let lhs = lhs.row(i);
							let rhs = rhs;
							let mut tmp = if conj_lhs == conj_rhs {
								dot::inner_prod_no_conj_simd::<T, S>(simd, lhs.transpose(), rhs)
							} else {
								dot::inner_prod_conj_lhs_simd::<T, S>(simd, lhs.transpose(), rhs)
							};

							if conj_rhs == Conj::Yes {
								tmp = conj(tmp);
							}
							tmp = *alpha * tmp;
							if let Accum::Add = beta {
								tmp = *dst + tmp;
							}
							*dst = tmp;
						}
					}
				}

				dispatch!(
					Impl {
						dst,
						beta,
						lhs,
						conj_lhs,
						rhs,
						conj_rhs,
						alpha,
					},
					Impl,
					T
				);
			},
			#[cfg(feature = "rayon")]
			Par::Rayon(nthreads) => {
				let nthreads = nthreads.get();

				use rayon::prelude::*;
				dst.par_partition_mut(nthreads)
					.zip_eq(lhs.par_row_partition(nthreads))
					.for_each(|(dst, lhs)| {
						make_guard!(M);
						let nrows = dst.nrows().bind(M);
						let dst = dst.as_row_shape_mut(nrows);
						let lhs = lhs.as_row_shape(nrows);

						matvec(dst, beta, lhs, conj_lhs, rhs, conj_rhs, alpha, Par::Seq);
					})
			},
		}
	}
}

mod matvec_colmajor {
	use super::*;
	use crate::col::ColMut;
	use crate::linalg::temp_mat_uninit;
	use crate::mat::AsMatMut;
	use crate::utils::bound::IdxInc;
	use crate::{unzip, zip};
	use faer_traits::SimdArch;

	#[math]
	pub fn matvec<'M, 'K, T: ComplexField>(
		dst: ColMut<'_, T, Dim<'M>, ContiguousFwd>,
		beta: Accum,
		lhs: MatRef<'_, T, Dim<'M>, Dim<'K>, ContiguousFwd, isize>,
		conj_lhs: Conj,
		rhs: ColRef<'_, T, Dim<'K>>,
		conj_rhs: Conj,
		alpha: &T,
		par: Par,
	) {
		core::assert!(try_const! { T::SIMD_CAPABILITIES.is_simd() });
		let size = *lhs.nrows() * *lhs.ncols();
		let par = if size < 256 * 256usize { Par::Seq } else { par };

		match par {
			Par::Seq => {
				pub struct Impl<'a, 'M, 'K, T: ComplexField> {
					dst: ColMut<'a, T, Dim<'M>, ContiguousFwd>,
					beta: Accum,
					lhs: MatRef<'a, T, Dim<'M>, Dim<'K>, ContiguousFwd, isize>,
					conj_lhs: Conj,
					rhs: ColRef<'a, T, Dim<'K>>,
					conj_rhs: Conj,
					alpha: &'a T,
				}

				impl<'a, 'M, 'K, T: ComplexField> pulp::WithSimd for Impl<'a, 'M, 'K, T> {
					type Output = ();

					#[inline(always)]
					fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
						let Self {
							dst,
							beta,
							lhs,
							conj_lhs,
							rhs,
							conj_rhs,
							alpha,
						} = self;

						let simd = T::simd_ctx(simd);

						let M = lhs.nrows();
						let simd = SimdCtx::<T, S>::new(simd, M);
						let (head, body, tail) = simd.indices();

						let mut dst = dst;
						match beta {
							Accum::Add => {},
							Accum::Replace => {
								let mut dst = dst.rb_mut();
								if let Some(i) = head {
									simd.write(dst.rb_mut(), i, simd.zero());
								}
								for i in body.clone() {
									simd.write(dst.rb_mut(), i, simd.zero());
								}
								if let Some(i) = tail {
									simd.write(dst.rb_mut(), i, simd.zero());
								}
							},
						}

						for j in lhs.ncols().indices() {
							let mut dst = dst.rb_mut();
							let lhs = lhs.col(j);
							let rhs = &rhs[j];
							let rhs = if conj_rhs == Conj::Yes { conj(*rhs) } else { copy(*rhs) };
							let rhs = rhs * *alpha;

							let vrhs = simd.splat(&rhs);
							if conj_lhs == Conj::Yes {
								if let Some(i) = head {
									let y = simd.read(dst.rb(), i);
									let x = simd.read(lhs, i);
									simd.write(dst.rb_mut(), i, simd.conj_mul_add(x, vrhs, y));
								}
								for i in body.clone() {
									let y = simd.read(dst.rb(), i);
									let x = simd.read(lhs, i);
									simd.write(dst.rb_mut(), i, simd.conj_mul_add(x, vrhs, y));
								}
								if let Some(i) = tail {
									let y = simd.read(dst.rb(), i);
									let x = simd.read(lhs, i);
									simd.write(dst.rb_mut(), i, simd.conj_mul_add(x, vrhs, y));
								}
							} else {
								if let Some(i) = head {
									let y = simd.read(dst.rb(), i);
									let x = simd.read(lhs, i);
									simd.write(dst.rb_mut(), i, simd.mul_add(x, vrhs, y));
								}
								for i in body.clone() {
									let y = simd.read(dst.rb(), i);
									let x = simd.read(lhs, i);
									simd.write(dst.rb_mut(), i, simd.mul_add(x, vrhs, y));
								}
								if let Some(i) = tail {
									let y = simd.read(dst.rb(), i);
									let x = simd.read(lhs, i);
									simd.write(dst.rb_mut(), i, simd.mul_add(x, vrhs, y));
								}
							}
						}
					}
				}

				dispatch!(
					Impl {
						dst,
						lhs,
						conj_lhs,
						rhs,
						conj_rhs,
						beta,
						alpha,
					},
					Impl,
					T
				)
			},
			#[cfg(feature = "rayon")]
			Par::Rayon(nthreads) => {
				use rayon::prelude::*;
				let nthreads = nthreads.get();
				let mut mem = MemBuffer::new(temp_mat_scratch::<T>(dst.nrows().unbound(), nthreads));
				let stack = MemStack::new(&mut mem);

				let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(dst.nrows(), nthreads, stack) };
				let mut tmp = tmp.as_mat_mut().try_as_col_major_mut().unwrap();

				let mut dst = dst;
				make_guard!(Z);
				let Z = 0usize.bind(Z);
				let z = IdxInc::new_checked(0, lhs.ncols());

				tmp.rb_mut()
					.par_col_iter_mut()
					.zip_eq(lhs.par_col_partition(nthreads))
					.zip_eq(rhs.par_partition(nthreads))
					.for_each(|((dst, lhs), rhs)| {
						make_guard!(K);
						let K = lhs.ncols().bind(K);
						let lhs = lhs.as_col_shape(K);
						let rhs = rhs.as_row_shape(K);

						matvec(dst, Accum::Replace, lhs, conj_lhs, rhs, conj_rhs, alpha, Par::Seq);
					});

				matvec(
					dst.rb_mut(),
					beta,
					lhs.subcols(z, Z),
					conj_lhs,
					rhs.subrows(z, Z),
					conj_rhs,
					&zero(),
					Par::Seq,
				);
				for j in 0..nthreads {
					zip!(dst.rb_mut(), tmp.rb().col(j)).for_each(|unzip!(dst, src)| *dst = *dst + *src)
				}
			},
		}
	}
}

mod rank_update {
	use super::*;
	use crate::assert;

	#[math]
	fn rank_update_imp<'M, 'N, T: ComplexField>(
		dst: MatMut<'_, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
		beta: Accum,
		lhs: ColRef<'_, T, Dim<'M>, ContiguousFwd>,
		conj_lhs: Conj,
		rhs: RowRef<'_, T, Dim<'N>>,
		conj_rhs: Conj,
		alpha: &T,
	) {
		assert!(T::SIMD_CAPABILITIES.is_simd());

		struct Impl<'a, 'M, 'N, T: ComplexField> {
			dst: MatMut<'a, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
			beta: Accum,
			lhs: ColRef<'a, T, Dim<'M>, ContiguousFwd>,
			conj_lhs: Conj,
			rhs: RowRef<'a, T, Dim<'N>>,
			conj_rhs: Conj,
			alpha: &'a T,
		}

		impl<T: ComplexField> pulp::WithSimd for Impl<'_, '_, '_, T> {
			type Output = ();

			#[inline(always)]
			fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
				let Self {
					mut dst,
					beta,
					lhs,
					conj_lhs,
					rhs,
					conj_rhs,
					alpha,
				} = self;

				let (m, n) = dst.shape();
				let simd = SimdCtx::<T, S>::new(T::simd_ctx(simd), m);

				let (head, body, tail) = simd.indices();

				for j in n.indices() {
					let mut dst = dst.rb_mut().col_mut(j);

					let rhs = *alpha * conj_rhs.apply_rt(&rhs[j]);
					let rhs = simd.splat(&rhs);

					if conj_lhs.is_conj() {
						match beta {
							Accum::Add => {
								if let Some(i) = head {
									let mut acc = simd.read(dst.rb(), i);
									acc = simd.conj_mul_add(simd.read(lhs, i), rhs, acc);
									simd.write(dst.rb_mut(), i, acc);
								}
								for i in body.clone() {
									let mut acc = simd.read(dst.rb(), i);
									acc = simd.conj_mul_add(simd.read(lhs, i), rhs, acc);
									simd.write(dst.rb_mut(), i, acc);
								}
								if let Some(i) = tail {
									let mut acc = simd.read(dst.rb(), i);
									acc = simd.conj_mul_add(simd.read(lhs, i), rhs, acc);
									simd.write(dst.rb_mut(), i, acc);
								}
							},
							Accum::Replace => {
								if let Some(i) = head {
									let acc = simd.conj_mul(simd.read(lhs, i), rhs);
									simd.write(dst.rb_mut(), i, acc);
								}
								for i in body.clone() {
									let acc = simd.conj_mul(simd.read(lhs, i), rhs);
									simd.write(dst.rb_mut(), i, acc);
								}
								if let Some(i) = tail {
									let acc = simd.conj_mul(simd.read(lhs, i), rhs);
									simd.write(dst.rb_mut(), i, acc);
								}
							},
						}
					} else {
						match beta {
							Accum::Add => {
								if let Some(i) = head {
									let mut acc = simd.read(dst.rb(), i);
									acc = simd.mul_add(simd.read(lhs, i), rhs, acc);
									simd.write(dst.rb_mut(), i, acc);
								}
								for i in body.clone() {
									let mut acc = simd.read(dst.rb(), i);
									acc = simd.mul_add(simd.read(lhs, i), rhs, acc);
									simd.write(dst.rb_mut(), i, acc);
								}
								if let Some(i) = tail {
									let mut acc = simd.read(dst.rb(), i);
									acc = simd.mul_add(simd.read(lhs, i), rhs, acc);
									simd.write(dst.rb_mut(), i, acc);
								}
							},
							Accum::Replace => {
								if let Some(i) = head {
									let acc = simd.mul(simd.read(lhs, i), rhs);
									simd.write(dst.rb_mut(), i, acc);
								}
								for i in body.clone() {
									let acc = simd.mul(simd.read(lhs, i), rhs);
									simd.write(dst.rb_mut(), i, acc);
								}
								if let Some(i) = tail {
									let acc = simd.mul(simd.read(lhs, i), rhs);
									simd.write(dst.rb_mut(), i, acc);
								}
							},
						}
					}
				}
			}
		}

		dispatch!(
			Impl {
				dst,
				lhs,
				conj_lhs,
				rhs,
				conj_rhs,
				beta,
				alpha,
			},
			Impl,
			T
		)
	}

	#[math]
	pub fn rank_update<'M, 'N, T: ComplexField>(
		dst: MatMut<'_, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
		beta: Accum,
		lhs: ColRef<'_, T, Dim<'M>, ContiguousFwd>,
		conj_lhs: Conj,
		rhs: RowRef<'_, T, Dim<'N>>,
		conj_rhs: Conj,
		alpha: &T,
		par: Par,
	) {
		match par {
			Par::Seq => {
				rank_update_imp(dst, beta, lhs, conj_lhs, rhs, conj_rhs, alpha);
			},
			#[cfg(feature = "rayon")]
			Par::Rayon(nthreads) => {
				let nthreads = nthreads.get();
				use rayon::prelude::*;
				dst.par_col_partition_mut(nthreads)
					.zip(rhs.par_partition(nthreads))
					.for_each(|(dst, rhs)| {
						with_dim!(N, dst.ncols());
						rank_update_imp(dst.as_col_shape_mut(N), beta, lhs, conj_lhs, rhs.as_col_shape(N), conj_rhs, alpha);
					});
			},
		}
	}
}

#[math]
fn matmul_imp<'M, 'N, 'K, T: ComplexField>(
	dst: MatMut<'_, T, Dim<'M>, Dim<'N>>,
	beta: Accum,
	lhs: MatRef<'_, T, Dim<'M>, Dim<'K>>,
	conj_lhs: Conj,
	rhs: MatRef<'_, T, Dim<'K>, Dim<'N>>,
	conj_rhs: Conj,
	alpha: &T,
	par: Par,
) {
	let mut dst = dst;

	let M = dst.nrows();
	let N = dst.ncols();
	let K = lhs.ncols();
	if *M == 0 || *N == 0 {
		return;
	}
	if *K == 0 {
		if beta == Accum::Replace {
			dst.fill(zero());
		}
		return;
	}

	let mut lhs = lhs;
	let mut rhs = rhs;

	if try_const! { T::SIMD_CAPABILITIES.is_simd() } {
		if dst.row_stride() < 0 {
			dst = dst.reverse_rows_mut();
			lhs = lhs.reverse_rows();
		}
		if dst.col_stride() < 0 {
			dst = dst.reverse_cols_mut();
			rhs = rhs.reverse_cols();
		}
		if lhs.col_stride() < 0 {
			lhs = lhs.reverse_cols();
			rhs = rhs.reverse_rows();
		}

		if dst.ncols().unbound() == 1 {
			let first = dst.ncols().check(0);
			if let (Some(dst), Some(lhs)) = (dst.rb_mut().try_as_col_major_mut(), lhs.try_as_col_major()) {
				matvec_colmajor::matvec(dst.col_mut(first), beta, lhs, conj_lhs, rhs.col(first), conj_rhs, alpha, par);
				return;
			}

			if let (Some(rhs), Some(lhs)) = (rhs.try_as_col_major(), lhs.try_as_row_major()) {
				matvec_rowmajor::matvec(dst.col_mut(first), beta, lhs, conj_lhs, rhs.col(first), conj_rhs, alpha, par);
				return;
			}
		}
		if dst.nrows().unbound() == 1 {
			let mut dst = dst.rb_mut().transpose_mut();
			let (rhs, lhs) = (lhs.transpose(), rhs.transpose());
			let (conj_rhs, conj_lhs) = (conj_lhs, conj_rhs);

			let first = dst.ncols().check(0);
			if let (Some(dst), Some(lhs)) = (dst.rb_mut().try_as_col_major_mut(), lhs.try_as_col_major()) {
				matvec_colmajor::matvec(dst.col_mut(first), beta, lhs, conj_lhs, rhs.col(first), conj_rhs, alpha, par);
				return;
			}

			if let (Some(rhs), Some(lhs)) = (rhs.try_as_col_major(), lhs.try_as_row_major()) {
				matvec_rowmajor::matvec(dst.col_mut(first), beta, lhs, conj_lhs, rhs.col(first), conj_rhs, alpha, par);
				return;
			}
		}
		if *K == 1 {
			let z = K.idx(0);

			if let (Some(dst), Some(lhs)) = (dst.rb_mut().try_as_col_major_mut(), lhs.try_as_col_major()) {
				rank_update::rank_update(dst, beta, lhs.col(z), conj_lhs, rhs.row(z), conj_rhs, alpha, par);
				return;
			}

			if let (Some(dst), Some(rhs)) = (dst.rb_mut().try_as_row_major_mut(), rhs.try_as_row_major()) {
				let dst = dst.transpose_mut();
				let rhs = rhs.row(z).transpose();
				let lhs = lhs.col(z).transpose();
				rank_update::rank_update(dst, beta, rhs, conj_rhs, lhs, conj_lhs, alpha, par);
				return;
			}
		}
		macro_rules! gemm_call {
			($kind: ident, $ty: ty, $nanogemm: ident) => {
				unsafe {
					let dst = core::mem::transmute_copy::<MatMut<'_, T, Dim<'M>, Dim<'N>>, MatMut<'_, $ty, Dim<'M>, Dim<'N>>>(&dst);
					let lhs = core::mem::transmute_copy::<MatRef<'_, T, Dim<'M>, Dim<'K>>, MatRef<'_, $ty, Dim<'M>, Dim<'K>>>(&lhs);
					let rhs = core::mem::transmute_copy::<MatRef<'_, T, Dim<'K>, Dim<'N>>, MatRef<'_, $ty, Dim<'K>, Dim<'N>>>(&rhs);
					let alpha = *core::mem::transmute_copy::<&T, &$ty>(&alpha);

					if (*M).saturating_mul(*N).saturating_mul(*K) <= NANO_GEMM_THRESHOLD {
						nano_gemm::planless::$nanogemm(
							*M,
							*N,
							*K,
							dst.as_ptr_mut(),
							dst.row_stride(),
							dst.col_stride(),
							lhs.as_ptr(),
							lhs.row_stride(),
							lhs.col_stride(),
							rhs.as_ptr(),
							rhs.row_stride(),
							rhs.col_stride(),
							match beta {
								Accum::Replace => core::mem::zeroed(),
								Accum::Add => 1.0.into(),
							},
							alpha,
							conj_lhs == Conj::Yes,
							conj_rhs == Conj::Yes,
						);
						return;
					} else {
						#[cfg(all(target_arch = "x86_64", feature = "std"))]
						{
							use private_gemm_x86::*;

							let feat = if std::arch::is_x86_feature_detected!("avx512f") {
								Some(InstrSet::Avx512)
							} else if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
								Some(InstrSet::Avx256)
							} else {
								None
							};

							if let Some(feat) = feat {
								gemm(
									DType::$kind,
									IType::U64,
									feat,
									*M,
									*N,
									*K,
									dst.as_ptr_mut() as *mut (),
									dst.row_stride(),
									dst.col_stride(),
									core::ptr::null(),
									core::ptr::null(),
									DstKind::Full,
									match beta {
										$crate::Accum::Replace => Accum::Replace,
										$crate::Accum::Add => Accum::Add,
									},
									lhs.as_ptr() as *const (),
									lhs.row_stride(),
									lhs.col_stride(),
									conj_lhs == Conj::Yes,
									core::ptr::null(),
									0,
									rhs.as_ptr() as *const (),
									rhs.row_stride(),
									rhs.col_stride(),
									conj_rhs == Conj::Yes,
									&raw const alpha as *const (),
									par.degree(),
								);
								return;
							}
						}

						{
							gemm::gemm(
								M.unbound(),
								N.unbound(),
								K.unbound(),
								dst.as_ptr_mut(),
								dst.col_stride(),
								dst.row_stride(),
								beta != Accum::Replace,
								lhs.as_ptr(),
								lhs.col_stride(),
								lhs.row_stride(),
								rhs.as_ptr(),
								rhs.col_stride(),
								rhs.row_stride(),
								match beta {
									Accum::Replace => core::mem::zeroed(),
									Accum::Add => 1.0.into(),
								},
								alpha,
								false,
								conj_lhs == Conj::Yes,
								conj_rhs == Conj::Yes,
								match par {
									Par::Seq => gemm::Parallelism::None,
									#[cfg(feature = "rayon")]
									Par::Rayon(nthreads) => gemm::Parallelism::Rayon(nthreads.get()),
								},
							);

							return;
						}
					}
				};
			};
		}

		if try_const! { T::IS_NATIVE_F64 } {
			gemm_call!(F64, f64, execute_f64);
		}
		if try_const! { T::IS_NATIVE_C64 } {
			gemm_call!(C64, num_complex::Complex<f64>, execute_c64);
		}
		if try_const! { T::IS_NATIVE_F32 } {
			gemm_call!(F32, f32, execute_f32);
		}
		if try_const! { T::IS_NATIVE_C32 } {
			gemm_call!(C32, num_complex::Complex<f32>, execute_c32);
		}

		if const { !(T::IS_NATIVE_F64 || T::IS_NATIVE_F32 || T::IS_NATIVE_C64 || T::IS_NATIVE_C32) } {
			if let (Some(dst), Some(lhs)) = (dst.rb_mut().try_as_col_major_mut(), lhs.try_as_col_major()) {
				matmul_vertical::matmul_simd(dst, beta, lhs, conj_lhs, rhs, conj_rhs, alpha, par);
				return;
			}
			if let (Some(dst), Some(rhs)) = (dst.rb_mut().try_as_row_major_mut(), rhs.try_as_row_major()) {
				matmul_vertical::matmul_simd(
					dst.transpose_mut(),
					beta,
					rhs.transpose(),
					conj_rhs,
					lhs.transpose(),
					conj_lhs,
					alpha,
					par,
				);
				return;
			}
			if let (Some(lhs), Some(rhs)) = (lhs.try_as_row_major(), rhs.try_as_col_major()) {
				matmul_horizontal::matmul_simd(dst, beta, lhs, conj_lhs, rhs, conj_rhs, alpha, par);
				return;
			}
		}
	}

	match par {
		Par::Seq => {
			for j in dst.ncols().indices() {
				for i in dst.nrows().indices() {
					let dst = &mut dst[(i, j)];

					let mut acc = dot::inner_prod_schoolbook(lhs.row(i), conj_lhs, rhs.col(j), conj_rhs);
					acc = *alpha * acc;
					if let Accum::Add = beta {
						acc = *dst + acc;
					}
					*dst = acc;
				}
			}
		},
		#[cfg(feature = "rayon")]
		Par::Rayon(nthreads) => {
			use rayon::prelude::*;
			let nthreads = nthreads.get();

			let m = *dst.nrows();
			let n = *dst.ncols();
			let task_count = m * n;
			let task_per_thread = task_count.msrv_div_ceil(nthreads);

			let dst = dst.rb();
			(0..nthreads).into_par_iter().for_each(|tid| {
				let task_idx = tid * task_per_thread;
				if task_idx >= task_count {
					return;
				}
				let ntasks = Ord::min(task_per_thread, task_count - task_idx);

				for ij in 0..ntasks {
					let ij = task_idx + ij;
					let i = dst.nrows().check(ij % m);
					let j = dst.ncols().check(ij / m);

					let mut dst = unsafe { dst.const_cast() };
					let dst = &mut dst[(i, j)];

					let mut acc = dot::inner_prod_schoolbook(lhs.row(i), conj_lhs, rhs.col(j), conj_rhs);
					acc = *alpha * acc;

					if let Accum::Add = beta {
						acc = *dst + acc;
					}
					*dst = acc;
				}
			});
		},
	}
}

#[track_caller]
fn precondition<M: Shape, N: Shape, K: Shape>(dst_nrows: M, dst_ncols: N, lhs_nrows: M, lhs_ncols: K, rhs_nrows: K, rhs_ncols: N) {
	assert!(all(dst_nrows == lhs_nrows, dst_ncols == rhs_ncols, lhs_ncols == rhs_nrows,));
}

/// computes the matrix product `[beta * acc] + alpha * lhs * rhs` and stores the result in `acc`
///
/// performs the operation:
/// - `acc = alpha * lhs * rhs` if `beta` is `Accum::Replace` (in this case, the preexisting
/// values in `acc` are not read)
/// - `acc = acc + alpha * lhs * rhs` if `beta` is `Accum::Add`
///
/// # panics
///
/// panics if the matrix dimensions are not compatible for matrix multiplication.
/// i.e.  
///  - `acc.nrows() == lhs.nrows()`
///  - `acc.ncols() == rhs.ncols()`
///  - `lhs.ncols() == rhs.nrows()`
///
/// # Example
///
/// ```
/// use faer::linalg::matmul::matmul;
/// use faer::{Accum, Conj, Mat, Par, mat, unzip, zip};
///
/// let lhs = mat![[0.0, 2.0], [1.0, 3.0]];
/// let rhs = mat![[4.0, 6.0], [5.0, 7.0]];
///
/// let mut acc = Mat::<f64>::zeros(2, 2);
/// let target = mat![
/// 	[
/// 		2.5 * (lhs[(0, 0)] * rhs[(0, 0)] + lhs[(0, 1)] * rhs[(1, 0)]),
/// 		2.5 * (lhs[(0, 0)] * rhs[(0, 1)] + lhs[(0, 1)] * rhs[(1, 1)]),
/// 	],
/// 	[
/// 		2.5 * (lhs[(1, 0)] * rhs[(0, 0)] + lhs[(1, 1)] * rhs[(1, 0)]),
/// 		2.5 * (lhs[(1, 0)] * rhs[(0, 1)] + lhs[(1, 1)] * rhs[(1, 1)]),
/// 	],
/// ];
///
/// matmul(&mut acc, Accum::Replace, &lhs, &rhs, 2.5, Par::Seq);
///
/// zip!(&acc, &target).for_each(|unzip!(acc, target)| assert!((acc - target).abs() < 1e-10));
/// ```
#[track_caller]
#[inline]
pub fn matmul<T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, M: Shape, N: Shape, K: Shape>(
	dst: impl AsMatMut<T = T, Rows = M, Cols = N>,
	beta: Accum,
	lhs: impl AsMatRef<T = LhsT, Rows = M, Cols = K>,
	rhs: impl AsMatRef<T = RhsT, Rows = K, Cols = N>,
	alpha: T,
	par: Par,
) {
	let mut dst = dst;
	let dst = dst.as_mat_mut();
	let lhs = lhs.as_mat_ref();
	let rhs = rhs.as_mat_ref();

	precondition(dst.nrows(), dst.ncols(), lhs.nrows(), lhs.ncols(), rhs.nrows(), rhs.ncols());

	make_guard!(M);
	make_guard!(N);
	make_guard!(K);
	let M = dst.nrows().bind(M);
	let N = dst.ncols().bind(N);
	let K = lhs.ncols().bind(K);

	matmul_imp(
		dst.as_dyn_stride_mut().as_shape_mut(M, N),
		beta,
		lhs.as_dyn_stride().canonical().as_shape(M, K),
		try_const! { Conj::get::<LhsT>() },
		rhs.as_dyn_stride().canonical().as_shape(K, N),
		try_const! { Conj::get::<RhsT>() },
		&alpha,
		par,
	);
}

/// computes the matrix product `[beta * acc] + alpha * lhs * rhs` (implicitly conjugating the
/// operands if needed) and stores the result in `acc`
///
/// performs the operation:
/// - `acc = alpha * lhs * rhs` if `beta` is `Accum::Replace` (in this case, the preexisting
/// values in `acc` are not read)
/// - `acc = acc + alpha * lhs * rhs` if `beta` is `Accum::Add`
///
/// # panics
///
/// panics if the matrix dimensions are not compatible for matrix multiplication.
/// i.e.  
///  - `acc.nrows() == lhs.nrows()`
///  - `acc.ncols() == rhs.ncols()`
///  - `lhs.ncols() == rhs.nrows()`
///
/// # example
///
/// ```
/// use faer::linalg::matmul::matmul_with_conj;
/// use faer::{Accum, Conj, Mat, Par, mat, unzip, zip};
///
/// let lhs = mat![[0.0, 2.0], [1.0, 3.0]];
/// let rhs = mat![[4.0, 6.0], [5.0, 7.0]];
///
/// let mut acc = Mat::<f64>::zeros(2, 2);
/// let target = mat![
/// 	[
/// 		2.5 * (lhs[(0, 0)] * rhs[(0, 0)] + lhs[(0, 1)] * rhs[(1, 0)]),
/// 		2.5 * (lhs[(0, 0)] * rhs[(0, 1)] + lhs[(0, 1)] * rhs[(1, 1)]),
/// 	],
/// 	[
/// 		2.5 * (lhs[(1, 0)] * rhs[(0, 0)] + lhs[(1, 1)] * rhs[(1, 0)]),
/// 		2.5 * (lhs[(1, 0)] * rhs[(0, 1)] + lhs[(1, 1)] * rhs[(1, 1)]),
/// 	],
/// ];
///
/// matmul_with_conj(
/// 	&mut acc,
/// 	Accum::Replace,
/// 	&lhs,
/// 	Conj::No,
/// 	&rhs,
/// 	Conj::No,
/// 	2.5,
/// 	Par::Seq,
/// );
///
/// zip!(&acc, &target).for_each(|unzip!(acc, target)| assert!((acc - target).abs() < 1e-10));
/// ```
#[track_caller]
#[inline]
pub fn matmul_with_conj<T: ComplexField, M: Shape, N: Shape, K: Shape>(
	dst: impl AsMatMut<T = T, Rows = M, Cols = N>,
	beta: Accum,
	lhs: impl AsMatRef<T = T, Rows = M, Cols = K>,
	conj_lhs: Conj,
	rhs: impl AsMatRef<T = T, Rows = K, Cols = N>,
	conj_rhs: Conj,
	alpha: T,
	par: Par,
) {
	let mut dst = dst;
	let dst = dst.as_mat_mut();
	let lhs = lhs.as_mat_ref();
	let rhs = rhs.as_mat_ref();

	precondition(dst.nrows(), dst.ncols(), lhs.nrows(), lhs.ncols(), rhs.nrows(), rhs.ncols());

	make_guard!(M);
	make_guard!(N);
	make_guard!(K);
	let M = dst.nrows().bind(M);
	let N = dst.ncols().bind(N);
	let K = lhs.ncols().bind(K);

	matmul_imp(
		dst.as_dyn_stride_mut().as_shape_mut(M, N),
		beta,
		lhs.as_dyn_stride().canonical().as_shape(M, K),
		conj_lhs,
		rhs.as_dyn_stride().canonical().as_shape(K, N),
		conj_rhs,
		&alpha,
		par,
	);
}

#[cfg(test)]
mod tests {
	use crate::c32;
	use std::num::NonZeroUsize;

	use super::triangular::{BlockStructure, DiagonalKind};
	use super::*;
	use crate::assert;
	use crate::mat::{Mat, MatMut, MatRef};
	use crate::stats::prelude::*;

	#[test]
	#[ignore = "takes too long"]
	fn test_matmul() {
		let rng = &mut StdRng::seed_from_u64(0);

		if option_env!("CI") == Some("true") {
			// too big for CI
			return;
		}

		let betas = [Accum::Replace, Accum::Add];

		#[cfg(not(miri))]
		let bools = [false, true];
		#[cfg(not(miri))]
		let alphas = [c32::ONE, c32::ZERO, c32::new(21.04, -12.13)];
		#[cfg(not(miri))]
		let par = [Par::Seq, Par::Rayon(NonZeroUsize::new(4).unwrap())];
		#[cfg(not(miri))]
		let conjs = [Conj::Yes, Conj::No];

		#[cfg(miri)]
		let bools = [true];
		#[cfg(miri)]
		let alphas = [c32::new(0.3218, -1.217489)];
		#[cfg(miri)]
		let par = [Par::Seq];
		#[cfg(miri)]
		let conjs = [Conj::Yes];

		let big0 = 127;
		let big1 = 128;
		let big2 = 129;

		let mid0 = 15;
		let mid1 = 16;
		let mid2 = 17;
		for (m, n, k) in [
			(big0, big1, 5),
			(big1, big0, 5),
			(big0, big2, 5),
			(big2, big0, 5),
			(mid0, mid0, 5),
			(mid1, mid1, 5),
			(mid2, mid2, 5),
			(mid0, mid1, 5),
			(mid1, mid0, 5),
			(mid0, mid2, 5),
			(mid2, mid0, 5),
			(mid0, 1, 1),
			(1, mid0, 1),
			(1, 1, mid0),
			(1, mid0, mid0),
			(mid0, 1, mid0),
			(mid0, mid0, 1),
			(1, 1, 1),
		] {
			let distribution = ComplexDistribution::new(StandardNormal, StandardNormal);
			let a = CwiseMatDistribution {
				nrows: m,
				ncols: k,
				dist: distribution,
			}
			.rand::<Mat<c32>>(rng);
			let b = CwiseMatDistribution {
				nrows: k,
				ncols: n,
				dist: distribution,
			}
			.rand::<Mat<c32>>(rng);
			let mut acc_init = CwiseMatDistribution {
				nrows: m,
				ncols: n,
				dist: distribution,
			}
			.rand::<Mat<c32>>(rng);

			let a = a.as_ref();
			let b = b.as_ref();

			for reverse_acc_cols in bools {
				for reverse_acc_rows in bools {
					for reverse_b_cols in bools {
						for reverse_b_rows in bools {
							for reverse_a_cols in bools {
								for reverse_a_rows in bools {
									for a_colmajor in bools {
										for b_colmajor in bools {
											for acc_colmajor in bools {
												let a = if a_colmajor { a } else { a.transpose() };
												let mut a = if a_colmajor { a } else { a.transpose() };

												let b = if b_colmajor { b } else { b.transpose() };
												let mut b = if b_colmajor { b } else { b.transpose() };

												if reverse_a_rows {
													a = a.reverse_rows();
												}
												if reverse_a_cols {
													a = a.reverse_cols();
												}
												if reverse_b_rows {
													b = b.reverse_rows();
												}
												if reverse_b_cols {
													b = b.reverse_cols();
												}
												for conj_a in conjs {
													for conj_b in conjs {
														for par in par {
															for beta in betas {
																for alpha in alphas {
																	test_matmul_impl(
																		reverse_acc_cols,
																		reverse_acc_rows,
																		acc_colmajor,
																		m,
																		n,
																		conj_a,
																		conj_b,
																		par,
																		beta,
																		alpha,
																		acc_init.as_mut(),
																		a,
																		b,
																	);
																}
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

	#[math]
	fn matmul_with_conj_fallback<T: Copy + ComplexField>(
		acc: MatMut<'_, T>,
		a: MatRef<'_, T>,
		conj_a: Conj,
		b: MatRef<'_, T>,
		conj_b: Conj,
		beta: Accum,
		alpha: T,
	) {
		let m = acc.nrows();
		let n = acc.ncols();
		let k = a.ncols();

		let job = |idx: usize| {
			let i = idx % m;
			let j = idx / m;
			let acc = acc.rb().submatrix(i, j, 1, 1);
			let mut acc = unsafe { acc.const_cast() };

			let mut local_acc = zero::<T>();
			for depth in 0..k {
				let a = &a[(i, depth)];
				let b = &b[(depth, j)];
				local_acc = local_acc
					+ match conj_a {
						Conj::Yes => conj(*a),
						Conj::No => copy(*a),
					} * match conj_b {
						Conj::Yes => conj(*b),
						Conj::No => copy(*b),
					}
			}
			match beta {
				Accum::Add => acc[(0, 0)] = acc[(0, 0)] + local_acc * alpha,
				Accum::Replace => acc[(0, 0)] = local_acc * alpha,
			}
		};

		for i in 0..m * n {
			job(i);
		}
	}

	#[math]
	fn test_matmul_impl(
		reverse_acc_cols: bool,
		reverse_acc_rows: bool,
		acc_colmajor: bool,
		m: usize,
		n: usize,
		conj_a: Conj,
		conj_b: Conj,
		par: Par,
		beta: Accum,
		alpha: c32,
		acc_init: MatMut<c32>,
		a: MatRef<c32>,
		b: MatRef<c32>,
	) {
		let acc = if acc_colmajor { acc_init } else { acc_init.transpose_mut() };

		let mut acc = if acc_colmajor { acc } else { acc.transpose_mut() };
		if reverse_acc_rows {
			acc = acc.reverse_rows_mut();
		}
		if reverse_acc_cols {
			acc = acc.reverse_cols_mut();
		}

		let mut target = acc.rb().to_owned();
		matmul_with_conj_fallback(target.as_mut(), a, conj_a, b, conj_b, beta, alpha);
		let target = target.rb();

		{
			let mut acc = acc.cloned();
			let a = a.cloned();

			{
				with_dim!(M, a.nrows());
				with_dim!(N, b.ncols());
				with_dim!(K, a.ncols());
				let mut acc = acc.rb_mut().as_shape_mut(M, N);
				let a = a.as_shape(M, K);
				let b = b.as_shape(K, N);

				matmul_vertical::matmul_simd(
					acc.rb_mut().try_as_col_major_mut().unwrap(),
					beta,
					a.try_as_col_major().unwrap(),
					conj_a,
					b,
					conj_b,
					&alpha,
					par,
				);
			}
			for j in 0..n {
				for i in 0..m {
					let acc = acc[(i, j)];
					let target = target[(i, j)];
					assert!(abs(acc.re - target.re) < 1e-3);
					assert!(abs(acc.im - target.im) < 1e-3);
				}
			}
		}
		{
			let mut acc = acc.cloned();
			let a = a.transpose().cloned();
			let a = a.transpose();

			let b = b.cloned();

			{
				with_dim!(M, a.nrows());
				with_dim!(N, b.ncols());
				with_dim!(K, a.ncols());
				let mut acc = acc.rb_mut().as_shape_mut(M, N);
				let a = a.as_shape(M, K);
				let b = b.as_shape(K, N);

				matmul_horizontal::matmul_simd(
					acc.rb_mut(),
					beta,
					a.try_as_row_major().unwrap(),
					conj_a,
					b.try_as_col_major().unwrap(),
					conj_b,
					&alpha,
					par,
				);
			}
			for j in 0..n {
				for i in 0..m {
					let acc = acc[(i, j)];
					let target = target[(i, j)];
					assert!(abs(acc.re - target.re) < 1e-3);
					assert!(abs(acc.im - target.im) < 1e-3);
				}
			}
		}

		matmul_with_conj(acc.rb_mut(), beta, a, conj_a, b, conj_b, alpha, par);
		for j in 0..n {
			for i in 0..m {
				let acc = acc[(i, j)];
				let target = target[(i, j)];
				assert!(abs(acc.re - target.re) < 1e-3);
				assert!(abs(acc.im - target.im) < 1e-3);
			}
		}
	}

	fn generate_structured_matrix(is_dst: bool, nrows: usize, ncols: usize, structure: BlockStructure) -> Mat<f64> {
		let rng = &mut StdRng::seed_from_u64(0);
		let mut mat = CwiseMatDistribution {
			nrows,
			ncols,
			dist: StandardNormal,
		}
		.rand::<Mat<f64>>(rng);

		if !is_dst {
			let kind = structure.diag_kind();
			if structure.is_lower() {
				for j in 0..ncols {
					for i in 0..j {
						mat[(i, j)] = 0.0;
					}
				}
			} else if structure.is_upper() {
				for j in 0..ncols {
					for i in j + 1..nrows {
						mat[(i, j)] = 0.0;
					}
				}
			}

			match kind {
				triangular::DiagonalKind::Zero => {
					for i in 0..nrows {
						mat[(i, i)] = 0.0;
					}
				},
				triangular::DiagonalKind::Unit => {
					for i in 0..nrows {
						mat[(i, i)] = 1.0;
					}
				},
				triangular::DiagonalKind::Generic => (),
			}
		}
		mat
	}

	fn run_test_problem(m: usize, n: usize, k: usize, dst_structure: BlockStructure, lhs_structure: BlockStructure, rhs_structure: BlockStructure) {
		let mut dst = generate_structured_matrix(true, m, n, dst_structure);
		let mut dst_target = dst.as_ref().to_owned();
		let dst_orig = dst.as_ref().to_owned();
		let lhs = generate_structured_matrix(false, m, k, lhs_structure);
		let rhs = generate_structured_matrix(false, k, n, rhs_structure);

		for par in [Par::Seq, Par::rayon(8)] {
			triangular::matmul_with_conj(
				dst.as_mut(),
				dst_structure,
				Accum::Replace,
				lhs.as_ref(),
				lhs_structure,
				Conj::No,
				rhs.as_ref(),
				rhs_structure,
				Conj::No,
				2.5,
				par,
			);

			matmul_with_conj(
				dst_target.as_mut(),
				Accum::Replace,
				lhs.as_ref(),
				Conj::No,
				rhs.as_ref(),
				Conj::No,
				2.5,
				par,
			);

			if dst_structure.is_dense() {
				for j in 0..n {
					for i in 0..m {
						assert!((dst[(i, j)] - dst_target[(i, j)]).abs() < 1e-10);
					}
				}
			} else if dst_structure.is_lower() {
				for j in 0..n {
					if matches!(dst_structure.diag_kind(), DiagonalKind::Generic) {
						for i in 0..j {
							assert!((dst[(i, j)] - dst_orig[(i, j)]).abs() < 1e-10);
						}
						for i in j..n {
							assert!((dst[(i, j)] - dst_target[(i, j)]).abs() < 1e-10);
						}
					} else {
						for i in 0..=j {
							assert!((dst[(i, j)] - dst_orig[(i, j)]).abs() < 1e-10);
						}
						for i in j + 1..n {
							assert!((dst[(i, j)] - dst_target[(i, j)]).abs() < 1e-10);
						}
					}
				}
			} else {
				for j in 0..n {
					if matches!(dst_structure.diag_kind(), DiagonalKind::Generic) {
						for i in 0..=j {
							assert!((dst[(i, j)] - dst_target[(i, j)]).abs() < 1e-10);
						}
						for i in j + 1..n {
							assert!((dst[(i, j)] - dst_orig[(i, j)]).abs() < 1e-10);
						}
					} else {
						for i in 0..j {
							assert!((dst[(i, j)] - dst_target[(i, j)]).abs() < 1e-10);
						}
						for i in j..n {
							assert!((dst[(i, j)] - dst_orig[(i, j)]).abs() < 1e-10);
						}
					}
				}
			}
		}
	}

	#[test]
	fn test_triangular() {
		use BlockStructure::*;
		let structures = [
			Rectangular,
			TriangularLower,
			TriangularUpper,
			StrictTriangularLower,
			StrictTriangularUpper,
			UnitTriangularLower,
			UnitTriangularUpper,
		];

		for dst in structures {
			for lhs in structures {
				for rhs in structures {
					#[cfg(not(miri))]
					let big = 100;

					#[cfg(miri)]
					let big = 31;
					for _ in 0..3 {
						let m = rand::random::<usize>() % big;
						let mut n = rand::random::<usize>() % big;
						let mut k = rand::random::<usize>() % big;

						match (!dst.is_dense(), !lhs.is_dense(), !rhs.is_dense()) {
							(true, true, _) | (true, _, true) | (_, true, true) => {
								n = m;
								k = m;
							},
							_ => (),
						}

						if !dst.is_dense() {
							n = m;
						}

						if !lhs.is_dense() {
							k = m;
						}

						if !rhs.is_dense() {
							k = n;
						}

						run_test_problem(m, n, k, dst, lhs, rhs);
					}
				}
			}
		}
	}
}
