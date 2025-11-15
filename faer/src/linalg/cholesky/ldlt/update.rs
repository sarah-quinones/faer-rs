use crate::assert;
use crate::internal_prelude::*;
use pulp::Simd;
fn rank_update_step_simd<T: ComplexField>(
	L: ColMut<'_, T, usize, ContiguousFwd>,
	W: MatMut<'_, T, usize, usize, ContiguousFwd>,
	p: ColRef<'_, T>,
	beta: ColRef<'_, T>,
	align_offset: usize,
) {
	struct Impl<'a, 'N, 'R, T: ComplexField> {
		L: ColMut<'a, T, Dim<'N>, ContiguousFwd>,
		W: MatMut<'a, T, Dim<'N>, Dim<'R>, ContiguousFwd>,
		p: ColRef<'a, T, Dim<'R>>,
		beta: ColRef<'a, T, Dim<'R>>,
		align_offset: usize,
	}
	impl<'a, 'N, 'R, T: ComplexField> pulp::WithSimd for Impl<'a, 'N, 'R, T> {
		type Output = ();

		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) {
			let Self {
				L,
				W,
				p,
				beta,
				align_offset,
			} = self;
			let mut L = L;
			let mut W = W;
			let N = W.nrows();
			let R = W.ncols();
			let simd =
				SimdCtx::<T, S>::new_align(T::simd_ctx(simd), N, align_offset);
			let indices = simd.indices();
			let mut iter = R.indices();
			let (i0, i1, i2, i3) =
				(iter.next(), iter.next(), iter.next(), iter.next());
			match (i0, i1, i2, i3) {
				(Some(i0), None, None, None) => {
					let p0 = simd.splat(&p[i0]);
					let beta0 = simd.splat(&beta[i0]);
					simd_iter!(for i in [indices] {
						let mut l = simd.read(L.rb(), i);
						let mut w0 = simd.read(W.rb().col(i0), i);
						w0 = simd.mul_add(p0, l, w0);
						l = simd.mul_add(beta0, w0, l);
						simd.write(L.rb_mut(), i, l);
						simd.write(W.rb_mut().col_mut(i0), i, w0);
					});
				},
				(Some(i0), Some(i1), None, None) => {
					let (p0, p1) = (simd.splat(&p[i0]), simd.splat(&p[i1]));
					let (beta0, beta1) =
						(simd.splat(&beta[i0]), simd.splat(&beta[i1]));

					simd_iter!(for i in [indices] {
						let mut l = simd.read(L.rb(), i);
						let mut w0 = simd.read(W.rb().col(i0), i);
						let mut w1 = simd.read(W.rb().col(i1), i);
						w0 = simd.mul_add(p0, l, w0);
						l = simd.mul_add(beta0, w0, l);
						w1 = simd.mul_add(p1, l, w1);
						l = simd.mul_add(beta1, w1, l);
						simd.write(L.rb_mut(), i, l);
						simd.write(W.rb_mut().col_mut(i0), i, w0);
						simd.write(W.rb_mut().col_mut(i1), i, w1);
					});
				},
				(Some(i0), Some(i1), Some(i2), None) => {
					let (p0, p1, p2) = (
						simd.splat(&p[i0]),
						simd.splat(&p[i1]),
						simd.splat(&p[i2]),
					);
					let (beta0, beta1, beta2) = (
						simd.splat(&beta[i0]),
						simd.splat(&beta[i1]),
						simd.splat(&beta[i2]),
					);

					simd_iter!(for i in [indices] {
						let mut l = simd.read(L.rb(), i);
						let mut w0 = simd.read(W.rb().col(i0), i);
						let mut w1 = simd.read(W.rb().col(i1), i);
						let mut w2 = simd.read(W.rb().col(i2), i);
						w0 = simd.mul_add(p0, l, w0);
						l = simd.mul_add(beta0, w0, l);
						w1 = simd.mul_add(p1, l, w1);
						l = simd.mul_add(beta1, w1, l);
						w2 = simd.mul_add(p2, l, w2);
						l = simd.mul_add(beta2, w2, l);
						simd.write(L.rb_mut(), i, l);
						simd.write(W.rb_mut().col_mut(i0), i, w0);
						simd.write(W.rb_mut().col_mut(i1), i, w1);
						simd.write(W.rb_mut().col_mut(i2), i, w2);
					});
				},
				(Some(i0), Some(i1), Some(i2), Some(i3)) => {
					let (p0, p1, p2, p3) = (
						simd.splat(&p[i0]),
						simd.splat(&p[i1]),
						simd.splat(&p[i2]),
						simd.splat(&p[i3]),
					);
					let (beta0, beta1, beta2, beta3) = (
						simd.splat(&beta[i0]),
						simd.splat(&beta[i1]),
						simd.splat(&beta[i2]),
						simd.splat(&beta[i3]),
					);
					simd_iter!(for i in [indices] {
						let mut l = simd.read(L.rb(), i);
						let mut w0 = simd.read(W.rb().col(i0), i);
						let mut w1 = simd.read(W.rb().col(i1), i);
						let mut w2 = simd.read(W.rb().col(i2), i);
						let mut w3 = simd.read(W.rb().col(i3), i);
						w0 = simd.mul_add(p0, l, w0);
						l = simd.mul_add(beta0, w0, l);
						w1 = simd.mul_add(p1, l, w1);
						l = simd.mul_add(beta1, w1, l);
						w2 = simd.mul_add(p2, l, w2);
						l = simd.mul_add(beta2, w2, l);
						w3 = simd.mul_add(p3, l, w3);
						l = simd.mul_add(beta3, w3, l);
						simd.write(L.rb_mut(), i, l);
						simd.write(W.rb_mut().col_mut(i0), i, w0);
						simd.write(W.rb_mut().col_mut(i1), i, w1);
						simd.write(W.rb_mut().col_mut(i2), i, w2);
						simd.write(W.rb_mut().col_mut(i3), i, w3);
					});
				},
				_ => panic!(),
			}
		}
	}
	with_dim!(N, W.nrows());
	with_dim!(R, W.ncols());
	dispatch!(
		Impl {
			L: L.as_row_shape_mut(N),
			W: W.as_shape_mut(N, R),
			p: p.as_row_shape(R),
			beta: beta.as_row_shape(R),
			align_offset,
		},
		Impl,
		T
	)
}
fn rank_update_step_fallback<T: ComplexField>(
	L: ColMut<'_, T>,
	W: MatMut<'_, T>,
	p: ColRef<'_, T>,
	beta: ColRef<'_, T>,
) {
	let mut L = L;
	let mut W = W;
	let n = W.nrows();
	let r = W.ncols();
	let mut iter = 0..r;
	let (i0, i1, i2, i3) = (iter.next(), iter.next(), iter.next(), iter.next());
	match (i0, i1, i2, i3) {
		(Some(i0), None, None, None) => {
			let p0 = &p[i0];
			let beta0 = &beta[i0];
			for i in 0..n {
				let mut l = L[i].copy();
				let mut w0 = W[(i, i0)].copy();
				w0 = p0 * &l + w0;
				l = beta0 * &w0 + l;
				L[i] = l;
				W[(i, i0)] = w0;
			}
		},
		(Some(i0), Some(i1), None, None) => {
			let (p0, p1) = (&p[i0], &p[i1]);
			let (beta0, beta1) = (&beta[i0], &beta[i1]);
			for i in 0..n {
				let mut l = L[i].copy();
				let mut w0 = W[(i, i0)].copy();
				let mut w1 = W[(i, i1)].copy();
				w0 = p0 * &l + w0;
				l = beta0 * &w0 + l;
				w1 = p1 * &l + w1;
				l = beta1 * &w1 + l;
				L[i] = l;
				W[(i, i0)] = w0;
				W[(i, i1)] = w1;
			}
		},
		(Some(i0), Some(i1), Some(i2), None) => {
			let (p0, p1, p2) = (&p[i0], &p[i1], &p[i2]);
			let (beta0, beta1, beta2) = (&beta[i0], &beta[i1], &beta[i2]);
			for i in 0..n {
				let mut l = L[i].copy();
				let mut w0 = W[(i, i0)].copy();
				let mut w1 = W[(i, i1)].copy();
				let mut w2 = W[(i, i2)].copy();
				w0 = p0 * &l + w0;
				l = beta0 * &w0 + l;
				w1 = p1 * &l + w1;
				l = beta1 * &w1 + l;
				w2 = p2 * &l + w2;
				l = beta2 * &w2 + l;
				L[i] = l;
				W[(i, i0)] = w0;
				W[(i, i1)] = w1;
				W[(i, i2)] = w2;
			}
		},
		(Some(i0), Some(i1), Some(i2), Some(i3)) => {
			let (p0, p1, p2, p3) = (&p[i0], &p[i1], &p[i2], &p[i3]);
			let (beta0, beta1, beta2, beta3) =
				(&beta[i0], &beta[i1], &beta[i2], &beta[i3]);
			for i in 0..n {
				let mut l = L[i].copy();
				let mut w0 = W[(i, i0)].copy();
				let mut w1 = W[(i, i1)].copy();
				let mut w2 = W[(i, i2)].copy();
				let mut w3 = W[(i, i3)].copy();
				w0 = p0 * &l + w0;
				l = beta0 * &w0 + l;
				w1 = p1 * &l + w1;
				l = beta1 * &w1 + l;
				w2 = p2 * &l + w2;
				l = beta2 * &w2 + l;
				w3 = p3 * &l + w3;
				l = beta3 * &w3 + l;
				L[i] = l;
				W[(i, i0)] = w0;
				W[(i, i1)] = w1;
				W[(i, i2)] = w2;
				W[(i, i3)] = w3;
			}
		},
		_ => panic!(),
	}
}
struct RankRUpdate<'a, T: ComplexField> {
	ld: MatMut<'a, T>,
	w: MatMut<'a, T>,
	alpha: ColMut<'a, T>,
	r: &'a mut dyn FnMut() -> usize,
}
impl<T: ComplexField> RankRUpdate<'_, T> {
	fn run(self) {
		let Self {
			mut ld,
			mut w,
			mut alpha,
			r,
		} = self;
		let n = w.nrows();
		let k = w.ncols();
		for j in 0..n {
			let mut L_col = ld.rb_mut().col_mut(j);
			let r = Ord::min(r(), k);
			let mut W = w.rb_mut().subcols_mut(0, r);
			let mut alpha = alpha.rb_mut().subrows_mut(0, r);
			let R = r;
			const BLOCKSIZE: usize = 4;
			let mut r = 0;
			while r < R {
				let bs = Ord::min(BLOCKSIZE, R - r);
				stack_mat!(p, bs, 1, BLOCKSIZE, 1, T);
				stack_mat!(beta, bs, 1, BLOCKSIZE, 1, T);
				let mut p = p.rb_mut().col_mut(0);
				let mut beta = beta.rb_mut().col_mut(0);
				for k in 0..bs {
					let p = p.rb_mut().at_mut(k);
					let beta = beta.rb_mut().at_mut(k);
					let alpha = alpha.rb_mut().at_mut(r + k);
					let d = L_col.rb_mut().at_mut(j);
					let w = W.rb().col(r + k);
					*p = w[j].copy();
					let ref alpha_conj_p = &*alpha * p.conj();
					let ref new_d = d.real() + (alpha_conj_p * &*p).real();
					*beta = alpha_conj_p.mul_real(new_d.recip());
					*alpha = (alpha.real() - new_d * beta.abs2()).to_cplx();
					*d = new_d.to_cplx();
					*p = -&*p;
				}
				let mut L_col = L_col.rb_mut().get_mut(j + 1..);
				let mut W_col =
					W.rb_mut().subcols_mut(r, bs).get_mut(j + 1.., ..);
				if const { T::SIMD_CAPABILITIES.is_simd() } {
					if let (Some(L_col), Some(W_col)) = (
						L_col.rb_mut().try_as_col_major_mut(),
						W_col.rb_mut().try_as_col_major_mut(),
					) {
						rank_update_step_simd(
							L_col,
							W_col,
							p.rb(),
							beta.rb(),
							simd_align(j + 1),
						);
					} else {
						rank_update_step_fallback(
							L_col,
							W_col,
							p.rb(),
							beta.rb(),
						);
					}
				} else {
					rank_update_step_fallback(L_col, W_col, p.rb(), beta.rb());
				}
				r += bs;
			}
		}
	}
}
pub(crate) fn rank_update_indices(
	start_col: usize,
	indices: &[usize],
) -> impl FnMut() -> usize + '_ {
	let mut current_col = start_col;
	let mut current_r = 0;
	move || {
		if current_r == indices.len() {
			current_r
		} else {
			while current_col == indices[current_r] - current_r {
				current_r += 1;
				if current_r == indices.len() {
					return current_r;
				}
			}
			current_col += 1;
			current_r
		}
	}
}
pub(crate) fn delete_rows_and_cols_triangular<T: ComplexField>(
	A: MatMut<'_, T>,
	idx: &[usize],
) {
	let mut A = A;
	let n = A.nrows();
	let r = idx.len();
	let r1 = r + 1;
	for chunk_j in 0..r1 {
		let j_start = if chunk_j == 0 {
			0
		} else {
			idx[chunk_j - 1] + 1
		};
		let j_finish = if chunk_j == r { n } else { idx[chunk_j] };
		for j in j_start..j_finish {
			for chunk_i in chunk_j..r1 {
				let i_start = if chunk_i == chunk_j {
					j
				} else {
					idx[chunk_i - 1] + 1
				};
				let i_finish = if chunk_i == r { n } else { idx[chunk_i] };
				if chunk_i != 0 || chunk_j != 0 {
					for i in i_start..i_finish {
						A[(i - chunk_i, j - chunk_j)] = A[(i, j)].copy();
					}
				}
			}
		}
	}
}
#[track_caller]
pub fn rank_r_update_clobber<T: ComplexField>(
	LD: MatMut<'_, T>,
	w: MatMut<'_, T>,
	alpha: DiagMut<'_, T>,
) {
	let n = LD.nrows();
	let r = w.ncols();
	assert!(all(
		LD.nrows() == n,
		LD.ncols() == n,
		w.nrows() == n,
		w.ncols() == r,
		alpha.dim() == r
	));
	if n == 0 {
		return;
	}
	RankRUpdate {
		ld: LD,
		w,
		alpha: alpha.column_vector_mut(),
		r: &mut || r,
	}
	.run();
}
#[track_caller]
pub fn delete_rows_and_cols_clobber_scratch<T: ComplexField>(
	dim: usize,
	n_removed: usize,
) -> StackReq {
	temp_mat_scratch::<T>(dim, n_removed)
		.and(temp_mat_scratch::<T>(n_removed, 1))
}
#[track_caller]
pub fn insert_rows_and_cols_clobber_scratch<T: ComplexField>(
	n_inserted: usize,
	par: Par,
) -> StackReq {
	super::factor::cholesky_in_place_scratch::<T>(n_inserted, par, default())
}
#[track_caller]
pub fn delete_rows_and_cols_clobber<T: ComplexField>(
	LD: MatMut<'_, T>,
	indices: &mut [usize],
	par: Par,
	stack: &mut MemStack,
) {
	let n = LD.nrows();
	let r = indices.len();
	_ = par;
	assert!(all(LD.nrows() == n, LD.ncols() == n));
	if r == 0 {
		return;
	}
	indices.sort_unstable();
	for i in 0..r - 1 {
		assert!(indices[i + 1] > indices[i]);
	}
	assert!(indices[r - 1] < n);
	let first = indices[0];
	alloca!('stack: {
		let mut w = unsafe { mat![uninit::<T>, n - first - r, r] };
		let mut alpha = unsafe { col![uninit::<T>, r] };
	});
	for k in 0..r {
		let j = indices[k];
		alpha[k] = LD[(j, j)].copy();
		for chunk_i in k..r {
			let chunk_i = chunk_i + 1;
			let i_start = indices[chunk_i - 1] + 1;
			let i_finish = if chunk_i == r { n } else { indices[chunk_i] };
			for i in i_start..i_finish {
				w[(i - chunk_i - first, k)] = LD[(i, j)].copy();
			}
		}
	}
	let mut LD = LD;
	delete_rows_and_cols_triangular(LD.rb_mut(), indices);
	RankRUpdate {
		ld: LD.submatrix_mut(first, first, n - first - r, n - first - r),
		w,
		alpha,
		r: &mut rank_update_indices(first, indices),
	}
	.run();
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use crate::{Col, Mat, assert, c64};
	use dyn_stack::MemBuffer;
	#[test]
	fn test_rank_update() {
		let rng = &mut StdRng::seed_from_u64(0);
		let approx_eq = CwiseMat(ApproxEq {
			abs_tol: 1e-12,
			rel_tol: 1e-12,
		});
		for r in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10] {
			for n in [2, 4, 8, 15] {
				let A = CwiseMatDistribution {
					nrows: n,
					ncols: n,
					dist: ComplexDistribution::new(
						StandardNormal,
						StandardNormal,
					),
				}
				.rand::<Mat<c64>>(rng);
				let mut W = CwiseMatDistribution {
					nrows: n,
					ncols: r,
					dist: ComplexDistribution::new(
						StandardNormal,
						StandardNormal,
					),
				}
				.rand::<Mat<c64>>(rng);
				let mut alpha = CwiseColDistribution {
					nrows: r,
					dist: ComplexDistribution::new(
						StandardNormal,
						StandardNormal,
					),
				}
				.rand::<Col<c64>>(rng)
				.into_diagonal();
				for j in 0..r {
					alpha.column_vector_mut()[j].im = 0.0;
				}
				let A = &A * &A.adjoint();
				let A_new = &A + &W * &alpha * &W.adjoint();
				let A = A.as_ref();
				let A_new = A_new.as_ref();
				let mut L = A.cloned();
				let mut L = L.as_mut();
				linalg::cholesky::ldlt::factor::cholesky_in_place(
					L.rb_mut(),
					default(),
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(linalg::cholesky::ldlt::factor::cholesky_in_place_scratch::<c64>(
						n,
						Par::Seq,
						default(),
					))),
					default(),
				)
				.unwrap();
				linalg::cholesky::ldlt::update::rank_r_update_clobber(
					L.rb_mut(),
					W.as_mut(),
					alpha.as_mut(),
				);
				let D = L.as_mut().diagonal().column_vector().as_mat().cloned();
				let D = D.col(0).as_diagonal();
				for j in 0..n {
					for i in 0..j {
						L[(i, j)] = c64::ZERO;
					}
					L[(j, j)] = c64::ONE;
				}
				let L = L.as_ref();
				assert!(A_new ~ L * D * L.adjoint());
			}
		}
	}
}
