use super::*;
use crate::assert;
use crate::linalg::matmul::matmul;
use linalg::evd::schur;

const MIN_DIM: usize = 32;

/// partial eigendecomposition tuning parameters.
#[derive(Debug, Copy, Clone)]
pub struct PartialEigenParams {
	/// minimum projection subspace dimension.
	pub min_dim: usize,
	/// maximum projection subspace dimension.
	pub max_dim: usize,
	/// maximum number of algorithm restarts.
	pub max_restarts: usize,

	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

/// partial eigendecomposition tuning parameters.
#[derive(Debug, Copy, Clone)]
pub struct PartialEigenInfo {
	/// number of converged eigenvalues and eigenvectors.
	pub n_converged_eigen: usize,

	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

impl Default for PartialEigenParams {
	fn default() -> Self {
		Self {
			min_dim: 0,
			max_dim: 0,
			max_restarts: 1000,
			non_exhaustive: NonExhaustive(()),
		}
	}
}

#[math]
fn iterate_arnoldi<T: ComplexField>(A: &dyn LinOp<T>, H: MatMut<'_, T>, V: MatMut<'_, T>, start: usize, end: usize, par: Par, stack: &mut MemStack) {
	let mut V = V;
	let mut H = H;

	for j in start..end + 1 {
		let mut H = H.rb_mut().col_mut(j - 1);
		H.fill(zero());

		let (V, Vnext) = V.rb_mut().split_at_col_mut(j);
		let V = V.rb();

		let mut Vnext = Vnext.col_mut(0);
		A.apply(Vnext.rb_mut().as_mat_mut(), V.col(j - 1).as_mat(), par, stack);

		let (mut converged, _) = stack.collect(core::iter::repeat_n(false, j));

		let mut h = H.rb_mut().get_mut(..j);

		for i in 0..j {
			let r = V.col(i).adjoint() * Vnext.rb();
			zip!(Vnext.rb_mut(), V.col(i)).for_each(|unzip!(y, x)| *y = *y - r * *x);
			h[i] = r;
		}

		let f = from_f64::<T::Real>(Ord::max(j, 8) as f64) * eps::<T::Real>();

		loop {
			let mut all_true = true;
			for i in 0..j {
				if !converged[i] {
					all_true = false;

					let r = V.col(i).adjoint() * Vnext.rb();
					zip!(Vnext.rb_mut(), V.col(i)).for_each(|unzip!(y, x)| *y = *y - r * *x);
					h[i] = h[i] + r;

					converged[i] = abs(r) < f * Vnext.norm_l2();
				}
			}
			if all_true {
				break;
			}
		}

		let norm = Vnext.norm_l2();
		if norm > zero() {
			let norm_inv = recip(norm);
			zip!(&mut Vnext).for_each(|unzip!(v)| *v = mul_real(*v, norm_inv));
		}
		H[j] = from_real(norm);
	}
}

fn schur_swap<T: ComplexField>(a: MatMut<T>, q: Option<MatMut<T>>, j0: usize, n1: usize, n2: usize) -> isize {
	if const { T::IS_REAL } {
		unsafe { schur::real_schur::schur_swap::<T::Real>(core::mem::transmute(a), core::mem::transmute(q), j0, n1, n2) }
	} else {
		assert!(all(n1 == 1, n2 == 1));
		schur::complex_schur::schur_swap(a, q, j0)
	}
}

#[math]
fn reorder_schur<T: ComplexField>(mut A: MatMut<'_, T>, mut Q: Option<MatMut<'_, T>>, mut ifst: usize, mut ilst: usize) {
	let zero = zero::<T>();
	let n = A.nrows();

	// *
	// * Determine the first row of the specified block and find out
	// * if it is 1-by-1 or 2-by-2.
	// *
	let mut nbf = 1;
	if const { T::IS_REAL } {
		if ifst > 0 {
			if A[(ifst, ifst - 1)] != zero {
				ifst -= 1;
			}
		}

		if ifst < n - 1 {
			if A[(ifst + 1, ifst)] != zero {
				nbf = 2;
			}
		}
	}

	// *
	// * Determine the first row of the final block
	// * and find out if it is 1-by-1 or 2-by-2.
	// *
	let mut nbl = 1;
	if const { T::IS_REAL } {
		if ilst > 0 {
			if A[(ilst, ilst - 1)] != zero {
				ilst = ilst - 1;
			}
		}
		if ilst < n - 1 {
			if A[(ilst + 1, ilst)] != zero {
				nbl = 2
			}
		}
	}

	if ifst == ilst {
		return;
	}

	if ifst < ilst {
		if nbf == 2 && nbl == 1 {
			ilst -= 1;
		}
		if nbf == 1 && nbl == 2 {
			ilst += 1;
		}

		let mut here = ifst;
		// * Swap with next one below.
		loop {
			if nbf == 1 || nbf == 2 {
				// * Current block either 1-by-1 or 2-by-2.
				let mut nbnext = 1;
				if const { T::IS_REAL } {
					if here + nbf + 1 <= n - 1 {
						if A[(here + nbf + 1, here + nbf)] != zero {
							nbnext = 2;
						}
					}
				}

				schur_swap(A.rb_mut(), Q.rb_mut(), here, nbf, nbnext);

				here += nbnext;
				// * Test if 2-by-2 block breaks into two 1-by-1 blocks.
				if const { T::IS_REAL } {
					if nbf == 2 {
						if A[(here + 1, here)] == zero {
							nbf = 3;
						}
					}
				}
			} else if const { T::IS_REAL } {
				// * Current block consists of two 1-by-1 blocks, each of which
				// * must be swapped individually.
				let mut nbnext = 1;
				if here + 3 <= n - 1 {
					if A[(here + 3, here + 2)] != zero {
						nbnext = 2;
					}
				}

				schur_swap(A.rb_mut(), Q.rb_mut(), here + 1, 1, nbnext);

				if nbnext == 1 {
					// * Swap two 1-by-1 blocks.
					schur_swap(A.rb_mut(), Q.rb_mut(), here, 1, 1);
					here += 1;
				} else {
					// * Recompute NBNEXT in case of 2-by-2 split.
					if A[(here + 2, here + 1)] == zero {
						nbnext = 1;
					}

					if nbnext == 2 {
						// * 2-by-2 block did not split.
						schur_swap(A.rb_mut(), Q.rb_mut(), here, 1, nbnext);
						here += 2;
					} else {
						// * 2-by-2 block did split.
						schur_swap(A.rb_mut(), Q.rb_mut(), here, 1, 1);
						here += 1;
						schur_swap(A.rb_mut(), Q.rb_mut(), here, 1, 1);
						here += 1;
					}
				}
			}

			if here >= ilst {
				break;
			}
		}
	} else {
		let mut here = ifst;

		loop {
			// * Swap with next one below.
			if nbf == 1 || nbf == 2 {
				// * Current block either 1-by-1 or 2-by-2.
				let mut nbnext = 1;
				if const { T::IS_REAL } {
					if here >= 2 {
						if A[(here - 1, here - 2)] != zero {
							nbnext = 2;
						}
					}
				}
				schur_swap(A.rb_mut(), Q.rb_mut(), here - nbnext, nbnext, nbf);
				here -= nbnext;

				// * Test if 2-by-2 block breaks into two 1-by-1 blocks.
				if const { T::IS_REAL } {
					if nbf == 2 {
						if A[(here + 1, here)] == zero {
							nbf = 3;
						}
					}
				}
			} else if const { T::IS_REAL } {
				// * Current block consists of two 1-by-1 blocks, each of which
				// * must be swapped individually.
				let mut nbnext = 1;
				if here >= 2 {
					if A[(here - 1, here - 2)] != zero {
						nbnext = 2;
					}
				}

				schur_swap(A.rb_mut(), Q.rb_mut(), here - nbnext, nbnext, 1);
				if nbnext == 1 {
					// * Swap two 1-by-1 blocks.
					schur_swap(A.rb_mut(), Q.rb_mut(), here, nbnext, 1);
					here -= 1;
				} else {
					// * Recompute NBNEXT in case of 2-by-2 split.
					if A[(here, here - 1)] == zero {
						nbnext = 1;
					}
					if nbnext == 2 {
						// * 2-by-2 block did not split.
						schur_swap(A.rb_mut(), Q.rb_mut(), here - 1, 2, 1);
						here -= 2;
					} else {
						// * 2-by-2 block did split.
						schur_swap(A.rb_mut(), Q.rb_mut(), here, 1, 1);
						here -= 1;
						schur_swap(A.rb_mut(), Q.rb_mut(), here, 1, 1);
						here -= 1;
					}
				}
			}

			if here <= ilst {
				break;
			}
		}
	}
}

#[math]
fn partial_schur_real_imp<T: RealField>(
	eigvecs: MatMut<'_, Complex<T>>,
	eigvals: &mut [Complex<T>],

	A: &dyn LinOp<T>,
	v0: ColRef<'_, T>,
	min_dim: usize,
	max_dim: usize,
	n_eigval: usize,
	tol: T,
	restarts: usize,
	par: Par,
	stack: &mut MemStack,
) -> usize {
	let n = A.nrows();

	let (mut H, stack) = temp_mat_zeroed::<T, _, _>(max_dim + 1, max_dim, stack);
	let mut H = H.as_mat_mut();
	let (mut V, stack) = temp_mat_zeroed::<T, _, _>(n, max_dim + 1, stack);
	let mut V = V.as_mat_mut();
	let (mut X, stack) = temp_mat_zeroed::<T, _, _>(max_dim, max_dim, stack);
	let mut X = X.as_mat_mut();
	let (mut vecs, stack) = temp_mat_zeroed::<T, _, _>(max_dim, max_dim, stack);
	let mut vecs = vecs.as_mat_mut();
	let (mut tmp, stack) = temp_mat_zeroed::<T, _, _>(n, max_dim, stack);
	let mut tmp = tmp.as_mat_mut();
	let mut active = 0usize;
	if max_dim < n {
		let (mut Q, stack) = temp_mat_zeroed::<T, _, _>(max_dim, max_dim, stack);
		let mut Q = Q.as_mat_mut();

		let (mut residual, stack) = temp_mat_zeroed::<T, _, _>(max_dim, 1, stack);
		let mut residual = residual.as_mat_mut().col_mut(0);
		let (mut w, stack) = temp_mat_zeroed::<T, _, _>(max_dim, 2, stack);
		let mut w = w.as_mat_mut();

		let blocksize = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(max_dim, max_dim);
		let (mut householder, stack) = temp_mat_zeroed::<T, _, _>(blocksize, max_dim, stack);
		let mut householder = householder.as_mat_mut();

		let f = v0.norm_l2();
		if f > min_positive() {
			let f = recip(f);
			zip!(V.rb_mut().col_mut(0), v0).for_each(|unzip!(y, x)| *y = f * *x);
		} else {
			let n0 = n as u32;
			let n1 = (n >> 32) as u32;

			let n = from_f64::<T>(n0 as f64) + from_f64::<T>(n1 as f64);
			let f = recip(sqrt(n));

			zip!(V.rb_mut().col_mut(0)).for_each(|unzip!(y)| *y = copy(f));
		}

		iterate_arnoldi(A, H.as_mut(), V.as_mut(), 1, min_dim, par, stack);
		let mut k = min_dim;

		for iter in 0..restarts {
			_ = iter;

			iterate_arnoldi(A, H.as_mut(), V.as_mut(), k + 1, max_dim, par, stack);

			let Hmm = abs(H[(max_dim, max_dim - 1)]);

			let n = max_dim - active;
			let (mut w_re, mut w_im) = w.rb_mut().get_mut(active..max_dim, ..).two_cols_mut(0, 1);

			Q.fill(zero());
			Q.rb_mut().diagonal_mut().fill(one());

			let mut Q_slice = Q.rb_mut().get_mut(active..max_dim, active..max_dim);
			let mut H_slice = H.rb_mut().get_mut(active..max_dim, active..max_dim);

			{
				let n = max_dim - active;

				let blocksize = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(n, n);
				let mut householder = householder.rb_mut().get_mut(..blocksize, ..n - 1);

				linalg::evd::hessenberg::hessenberg_in_place(H_slice.rb_mut(), householder.rb_mut(), par, stack, default());

				linalg::householder::apply_block_householder_sequence_on_the_right_in_place_with_conj(
					H_slice.rb().submatrix(1, 0, n - 1, n - 1),
					householder.rb(),
					Conj::No,
					Q_slice.rb_mut().submatrix_mut(1, 1, n - 1, n - 1),
					par,
					stack,
				);

				for j in 0..n {
					for i in j + 2..n {
						H_slice[(i, j)] = zero();
					}
				}
			}

			schur::real_schur::multishift_qr(
				true,
				H_slice.rb_mut(),
				Some(Q_slice.rb_mut()),
				w_re.rb_mut(),
				w_im.rb_mut(),
				0,
				n,
				par,
				stack,
				auto!(T),
			);

			let mut j = 0usize;
			while j < n {
				let mut i = j;
				let mut idx = i;
				let mut max = zero::<T>();
				while i < n {
					let cplx = i + 1 < n && H_slice[(i + 1, i)] != zero::<T>();
					let (v, bs) = if cplx { (hypot(w_re[i], w_im[i]), 2) } else { (abs(w_re[i]), 1) };

					if v > max {
						max = v;
						idx = i;
					}

					i += bs;
				}

				let i = idx;
				let cplx = i + 1 < n && H_slice[(i + 1, i)] != zero::<T>();
				let bs = if cplx { 2 } else { 1 };
				if i != j {
					reorder_schur(H_slice.rb_mut(), Some(Q_slice.rb_mut()), i, j);

					let x_re = w_re.rb_mut().try_as_col_major_mut().unwrap().as_slice_mut();
					let x_im = w_im.rb_mut().try_as_col_major_mut().unwrap().as_slice_mut();

					for x in [x_re, x_im] {
						x[j..i + bs].rotate_right(bs)
					}
				}

				j += bs;
			}

			let mut X = X.rb_mut().get_mut(..n, ..n);
			linalg::evd::evd_from_real_schur_imp(H_slice.rb(), X.as_mut(), par, auto!(T));
			let mut vecs = vecs.rb_mut().get_mut(..n, ..n);
			matmul(vecs.rb_mut(), Accum::Replace, Q_slice.rb(), X.rb(), one(), par);
			let vecs = vecs.rb();

			let mut H_tmp = tmp.rb_mut().get_mut(..active, ..);
			matmul(H_tmp.rb_mut(), Accum::Replace, H.rb().get(..active, ..), Q.rb(), one(), par);
			H.rb_mut().get_mut(..active, ..).copy_from(&H_tmp);

			// AV = VH
			// x in span(V)
			// Ax = AV y = (VH + f e*) y = k V y + f e* y = kx + f * y[-1]
			let mut j = 0usize;
			while j < n {
				let re = &vecs[(max_dim - active - 1, j)];
				if w_im[j] != zero::<T>() {
					let im = &vecs[(max_dim - active - 1, j + 1)];
					let res = Hmm * hypot(*re, *im);

					residual[active + j] = copy(res);
					residual[active + j + 1] = res;
					j += 2;
				} else {
					residual[active + j] = Hmm * abs(*re);
					j += 1;
				}
			}

			#[derive(Copy, Clone, PartialEq, Eq, Debug)]
			enum Group {
				Lock,
				Retain,
				Purge,
			}

			let mut groups = alloc::vec![Group::Purge; max_dim];

			let mut nev = n_eigval;
			if w_im[nev - active - 1] > zero::<T>() {
				nev += 1;
			}

			let mut nlock = 0usize;
			for j in 0..nev {
				if residual[j] <= tol {
					groups[j] = Group::Lock;
					nlock += 1;
				} else {
					groups[j] = Group::Retain;
				}
			}

			let ideal_size = Ord::min(nlock + min_dim, (min_dim + max_dim) / 2);
			k = nev;

			let mut i = nev;
			while i < max_dim {
				let cplx = w[(i, 1)] != zero::<T>();
				let bs = if cplx { 2 } else { 1 };

				let group;
				if k < ideal_size && residual[i] > tol {
					group = Group::Retain;
					k += bs;
				} else {
					group = Group::Purge;
				}

				for k in 0..bs {
					groups[i + k] = group;
				}
				i += bs;
			}

			let mut purge = 0usize;
			while purge < active && groups[purge] == Group::Lock {
				purge += 1;
			}

			let mut lo = 0usize;
			let mut mi = 0usize;
			let mut hi = 0usize;

			while hi < max_dim {
				let cplx = hi + 1 < max_dim && H[(hi + 1, hi)] != zero::<T>();
				let bs = if cplx { 2 } else { 1 };

				match groups[hi] {
					Group::Lock => {
						reorder_schur(H.rb_mut().get_mut(..max_dim, ..max_dim), Some(Q.as_mut()), hi, lo);
						for k in 0..bs {
							residual[lo + k] = copy(residual[hi + k]);
						}

						lo += bs;
						mi += bs;
						hi += bs;
					},
					Group::Retain => {
						reorder_schur(H.rb_mut().get_mut(..max_dim, ..max_dim), Some(Q.as_mut()), hi, mi);
						mi += bs;
						hi += bs;
					},
					Group::Purge => {
						hi += bs;
					},
				}
			}

			let mut V_tmp = tmp.rb_mut().get_mut(.., purge..k);
			matmul(
				V_tmp.rb_mut(),
				Accum::Replace,
				V.rb().get(.., purge..max_dim),
				Q.rb().get(purge..max_dim, purge..k),
				one(),
				par,
			);
			V.rb_mut().get_mut(.., purge..k).copy_from(&V_tmp);

			let mut b_tmp = tmp.rb_mut().get_mut(0, ..);
			matmul(b_tmp.rb_mut(), Accum::Replace, H.rb().get(max_dim, ..), Q.rb(), one(), par);
			H.rb_mut().get_mut(max_dim, ..).copy_from(b_tmp);

			let (mut x, y) = V.rb_mut().two_cols_mut(k, max_dim);
			x.copy_from(&y);

			let (mut x, mut y) = H.rb_mut().two_rows_mut(k, max_dim);
			x.copy_from(&y);
			y.fill(zero());

			active = nlock;
			if nlock >= n_eigval {
				break;
			}
		}
	} else {
		let mut H = H.rb_mut().get_mut(..n, ..n);
		let mut V = V.rb_mut().get_mut(..n, ..n);
		V.rb_mut().diagonal_mut().fill(one());
		A.apply(H.rb_mut(), V.rb(), par, stack);

		let (mut w, stack) = temp_mat_zeroed::<T, _, _>(n, 2, stack);
		let mut w = w.as_mat_mut();
		let (mut w_re, mut w_im) = w.rb_mut().get_mut(active..max_dim, ..).two_cols_mut(0, 1);

		{
			let blocksize = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(n, n);
			let (mut householder, stack) = temp_mat_zeroed::<T, _, _>(blocksize, n - 1, stack);
			let mut householder = householder.as_mat_mut();

			linalg::evd::hessenberg::hessenberg_in_place(H.rb_mut(), householder.rb_mut(), par, stack, default());

			linalg::householder::apply_block_householder_sequence_on_the_right_in_place_with_conj(
				H.rb().submatrix(1, 0, n - 1, n - 1),
				householder.rb(),
				Conj::No,
				V.rb_mut().submatrix_mut(1, 1, n - 1, n - 1),
				par,
				stack,
			);

			for j in 0..n {
				for i in j + 2..n {
					H[(i, j)] = zero();
				}
			}
		}

		schur::real_schur::multishift_qr(
			true,
			H.rb_mut(),
			Some(V.rb_mut()),
			w_re.rb_mut(),
			w_im.rb_mut(),
			0,
			n,
			par,
			stack,
			auto!(T),
		);
		active = n;
	}

	let n = active;
	let H = H.rb().get(..n, ..n);
	let V = V.rb().get(.., ..n);
	let mut X = X.rb_mut().get_mut(..n, ..n);
	linalg::evd::evd_from_real_schur_imp(H, X.as_mut(), par, auto!(T));
	let mut vecs = tmp.rb_mut().get_mut(.., ..n);
	matmul(vecs.rb_mut(), Accum::Replace, V, X.rb(), one(), par);
	let V = vecs.rb();

	let (mut norms, stack) = stack.make_with(n, |_| zero::<T::Real>());
	let (mut perm, stack) = stack.make_with(n, |_| 0usize);
	let _ = stack;

	let perm = &mut *perm;
	let norms = &mut *norms;

	let mut j = 0usize;
	while j < n {
		let cplx = j + 1 < n && H[(j + 1, j)] != zero::<T>();
		let bs = if cplx { 2 } else { 1 };
		let re = &H[(j, j)];

		if cplx {
			let im = sqrt(abs(H[(j + 1, j)])) * sqrt(abs(H[(j, j + 1)]));
			norms[j] = hypot(*re, im);
			norms[j + 1] = copy(norms[j]);

			perm[j] = j;
			perm[j + 1] = j;
		} else {
			norms[j] = abs(*re);
			perm[j] = j;
		}
		j += bs;
	}
	perm.sort_unstable_by(|&i, &j| {
		if norms[i] > norms[j] {
			core::cmp::Ordering::Less
		} else if norms[i] < norms[j] {
			core::cmp::Ordering::Greater
		} else {
			core::cmp::Ordering::Equal
		}
	});

	let mut eigvecs = eigvecs;
	let limit = Ord::min(n, n_eigval);

	let mut idx = 0usize;
	while idx < limit {
		let j = perm[idx];

		let cplx = j + 1 < n && H[(j + 1, j)] != zero::<T>();
		let bs = if cplx { 2 } else { 1 };
		let re = &H[(j, j)];
		let v_re = V.col(j);

		if cplx {
			let v_im = V.col(j + 1);
			let im = sqrt(abs(H[(j + 1, j)])) * sqrt(abs(H[(j, j + 1)]));

			if idx + 1 < limit {
				eigvals[idx + 1] = Complex::new(copy(*re), -im);
			}
			eigvals[idx] = Complex::new(copy(*re), im);
			if idx + 1 < limit {
				let (ej, ej1) = eigvecs.rb_mut().two_cols_mut(idx, idx + 1);
				zip!(ej, ej1, v_re, v_im).for_each(|unzip!(y0, y1, re, im)| {
					*y0 = Complex::new(copy(*re), copy(*im));
					*y1 = Complex::new(copy(*re), -*im);
				});
			} else {
				let ej = eigvecs.rb_mut().col_mut(idx);
				zip!(ej, v_re, v_im).for_each(|unzip!(y0, re, im)| {
					*y0 = Complex::new(copy(*re), copy(*im));
				});
			}
		} else {
			eigvals[idx] = Complex::new(copy(*re), zero());
			zip!(eigvecs.rb_mut().col_mut(idx), v_re).for_each(|unzip!(y, x)| *y = Complex::new(copy(*x), zero()));
		}

		idx += bs;
	}
	limit
}

#[math]
fn partial_schur_cplx_imp<T: ComplexField>(
	eigvecs: MatMut<'_, Complex<T::Real>>,
	eigvals: &mut [Complex<T::Real>],

	A: &dyn LinOp<T>,
	v0: ColRef<'_, T>,
	min_dim: usize,
	max_dim: usize,
	n_eigval: usize,
	tol: T::Real,
	restarts: usize,
	par: Par,
	stack: &mut MemStack,
) -> usize {
	let n = A.nrows();

	let (mut H, stack) = temp_mat_zeroed::<T, _, _>(max_dim + 1, max_dim, stack);
	let mut H = H.as_mat_mut();
	let (mut V, stack) = temp_mat_zeroed::<T, _, _>(n, max_dim + 1, stack);
	let mut V = V.as_mat_mut();
	let (mut X, stack) = temp_mat_zeroed::<T, _, _>(max_dim, max_dim, stack);
	let mut X = X.as_mat_mut();
	let (mut vecs, stack) = temp_mat_zeroed::<T, _, _>(max_dim, max_dim, stack);
	let mut vecs = vecs.as_mat_mut();
	let (mut tmp, stack) = temp_mat_zeroed::<T, _, _>(n, max_dim, stack);
	let mut tmp = tmp.as_mat_mut();

	let mut active = 0usize;
	if max_dim < n {
		let (mut w, stack) = temp_mat_zeroed::<T, _, _>(max_dim, 1, stack);
		let mut w = w.as_mat_mut().col_mut(0);
		let (mut residual, stack) = temp_mat_zeroed::<T::Real, _, _>(max_dim, 1, stack);
		let mut residual = residual.as_mat_mut().col_mut(0);
		let (mut Q, stack) = temp_mat_zeroed::<T, _, _>(max_dim, max_dim, stack);
		let mut Q = Q.as_mat_mut();

		let blocksize = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(max_dim, max_dim);
		let (mut householder, stack) = temp_mat_zeroed::<T, _, _>(blocksize, max_dim, stack);
		let mut householder = householder.as_mat_mut();

		let f = v0.norm_l2();
		if f > min_positive() {
			let f = recip(f);
			zip!(V.rb_mut().col_mut(0), v0).for_each(|unzip!(y, x)| *y = mul_real(*x, f));
		} else {
			let n0 = n as u32;
			let n1 = (n >> 32) as u32;

			let n = from_f64::<T>(n0 as f64) + from_f64::<T>(n1 as f64);
			let f = recip(sqrt(n));

			zip!(V.rb_mut().col_mut(0)).for_each(|unzip!(y)| *y = copy(f));
		}

		iterate_arnoldi(A, H.as_mut(), V.as_mut(), 1, min_dim, par, stack);
		let mut k = min_dim;

		for iter in 0..restarts {
			_ = iter;

			iterate_arnoldi(A, H.as_mut(), V.as_mut(), k + 1, max_dim, par, stack);

			let Hmm = copy(H[(max_dim, max_dim - 1)]);

			let n = max_dim - active;
			let mut w = w.rb_mut().get_mut(active..max_dim);

			Q.fill(zero());
			Q.rb_mut().diagonal_mut().fill(one());

			let mut Q_slice = Q.rb_mut().get_mut(active..max_dim, active..max_dim);
			let mut H_slice = H.rb_mut().get_mut(active..max_dim, active..max_dim);

			{
				let n = max_dim - active;

				let blocksize = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(n, n);
				let mut householder = householder.rb_mut().get_mut(..blocksize, ..n - 1);

				linalg::evd::hessenberg::hessenberg_in_place(H_slice.rb_mut(), householder.rb_mut(), par, stack, default());

				linalg::householder::apply_block_householder_sequence_on_the_right_in_place_with_conj(
					H_slice.rb().submatrix(1, 0, n - 1, n - 1),
					householder.rb(),
					Conj::No,
					Q_slice.rb_mut().submatrix_mut(1, 1, n - 1, n - 1),
					par,
					stack,
				);

				for j in 0..n {
					for i in j + 2..n {
						H_slice[(i, j)] = zero();
					}
				}
			}

			schur::complex_schur::multishift_qr(true, H_slice.rb_mut(), Some(Q_slice.rb_mut()), w.rb_mut(), 0, n, par, stack, auto!(T));

			for j in 0..n {
				let mut idx = j;
				let mut max = zero::<T::Real>();
				for i in j..n {
					let v = abs(w[i]);

					if v > max {
						max = v;
						idx = i;
					}
				}

				let i = idx;
				if i != j {
					reorder_schur(H_slice.rb_mut(), Some(Q_slice.rb_mut()), i, j);
					w.rb_mut().try_as_col_major_mut().unwrap().as_slice_mut()[j..i + 1].rotate_right(1);
				}
			}

			let mut X = X.rb_mut().get_mut(..n, ..n);
			linalg::evd::evd_from_cplx_schur_imp(H_slice.rb(), Conj::No, X.as_mut(), par, auto!(T));
			let mut vecs = vecs.rb_mut().get_mut(..n, ..n);
			matmul(vecs.rb_mut(), Accum::Replace, Q_slice.rb(), X.rb(), one(), par);
			let vecs = vecs.rb();

			let mut H_tmp = tmp.rb_mut().get_mut(..active, ..);
			matmul(H_tmp.rb_mut(), Accum::Replace, H.rb().get(..active, ..), Q.rb(), one(), par);
			H.rb_mut().get_mut(..active, ..).copy_from(&H_tmp);

			// AV = VH
			// x in span(V)
			// Ax = AV y = (VH + f e*) y = k V y + f e* y = kx + f * y[-1]
			let Hmm_abs = abs(Hmm);
			for j in 0..n {
				residual[active + j] = Hmm_abs * abs(vecs[(max_dim - active - 1, j)]);
			}

			#[derive(Copy, Clone, PartialEq, Eq, Debug)]
			enum Group {
				Lock,
				Retain,
				Purge,
			}

			let mut groups = alloc::vec![Group::Purge; max_dim];

			let nev = n_eigval;

			let mut nlock = 0usize;
			for j in 0..nev {
				if residual[j] <= tol {
					groups[j] = Group::Lock;
					nlock += 1;
				} else {
					groups[j] = Group::Retain;
				}
			}

			let ideal_size = Ord::min(nlock + min_dim, (min_dim + max_dim) / 2);
			k = nev;

			for i in nev..max_dim {
				let group;
				if k < ideal_size && residual[i] > tol {
					group = Group::Retain;
					k += 1;
				} else {
					group = Group::Purge;
				}

				groups[i] = group;
			}

			let mut purge = 0usize;
			while purge < active && groups[purge] == Group::Lock {
				purge += 1;
			}

			let mut lo = 0usize;
			let mut mi = 0usize;
			let mut hi = 0usize;

			while hi < max_dim {
				match groups[hi] {
					Group::Lock => {
						reorder_schur(H.rb_mut().get_mut(..max_dim, ..max_dim), Some(Q.as_mut()), hi, lo);
						residual[lo] = copy(residual[hi]);

						lo += 1;
						mi += 1;
						hi += 1;
					},
					Group::Retain => {
						reorder_schur(H.rb_mut().get_mut(..max_dim, ..max_dim), Some(Q.as_mut()), hi, mi);
						mi += 1;
						hi += 1;
					},
					Group::Purge => {
						hi += 1;
					},
				}
			}

			let mut V_tmp = tmp.rb_mut().get_mut(.., purge..k);
			matmul(
				V_tmp.rb_mut(),
				Accum::Replace,
				V.rb().get(.., purge..max_dim),
				Q.rb().get(purge..max_dim, purge..k),
				one(),
				par,
			);
			V.rb_mut().get_mut(.., purge..k).copy_from(&V_tmp);

			let mut b_tmp = tmp.rb_mut().get_mut(0, ..);
			matmul(b_tmp.rb_mut(), Accum::Replace, H.rb().get(max_dim, ..), Q.rb(), one(), par);
			H.rb_mut().get_mut(max_dim, ..).copy_from(b_tmp);

			let (mut x, y) = V.rb_mut().two_cols_mut(k, max_dim);
			x.copy_from(&y);

			let (mut x, mut y) = H.rb_mut().two_rows_mut(k, max_dim);
			x.copy_from(&y);
			y.fill(zero());

			active = nlock;
			if nlock >= n_eigval {
				break;
			}
		}
	} else {
		let mut H = H.rb_mut().get_mut(..n, ..n);
		let mut V = V.rb_mut().get_mut(..n, ..n);
		V.rb_mut().diagonal_mut().fill(one());
		A.apply(H.rb_mut(), V.rb(), par, stack);

		let (mut w, stack) = temp_mat_zeroed::<T, _, _>(n, 1, stack);
		let mut w = w.as_mat_mut();

		{
			let blocksize = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(n, n);
			let (mut householder, stack) = temp_mat_zeroed::<T, _, _>(blocksize, n - 1, stack);
			let mut householder = householder.as_mat_mut();

			linalg::evd::hessenberg::hessenberg_in_place(H.rb_mut(), householder.rb_mut(), par, stack, default());

			linalg::householder::apply_block_householder_sequence_on_the_right_in_place_with_conj(
				H.rb().submatrix(1, 0, n - 1, n - 1),
				householder.rb(),
				Conj::No,
				V.rb_mut().submatrix_mut(1, 1, n - 1, n - 1),
				par,
				stack,
			);

			for j in 0..n {
				for i in j + 2..n {
					H[(i, j)] = zero();
				}
			}
		}

		schur::complex_schur::multishift_qr(true, H.rb_mut(), Some(V.rb_mut()), w.rb_mut().col_mut(0), 0, n, par, stack, auto!(T));
		active = n;
	}

	let n = active;
	let H = H.rb().get(..n, ..n);
	let V = V.rb().get(.., ..n);
	let mut X = X.rb_mut().get_mut(..n, ..n);
	linalg::evd::evd_from_cplx_schur_imp(H, Conj::No, X.as_mut(), par, auto!(T));

	let mut vecs = tmp.rb_mut().get_mut(.., ..n);
	matmul(vecs.rb_mut(), Accum::Replace, V, X.rb(), one(), par);

	let V = vecs.rb();

	let (mut norms, stack) = stack.make_with(n, |j| abs(H[(j, j)]));
	let (mut perm, stack) = stack.make_with(n, |j| j);
	let _ = stack;

	let perm = &mut *perm;
	let norms = &mut *norms;

	perm.sort_unstable_by(|&i, &j| {
		if norms[i] > norms[j] {
			core::cmp::Ordering::Less
		} else if norms[i] < norms[j] {
			core::cmp::Ordering::Greater
		} else {
			core::cmp::Ordering::Equal
		}
	});

	let mut eigvecs = eigvecs;
	let limit = Ord::min(n, n_eigval);

	for idx in 0..limit {
		let j = perm[idx];
		let w = &H[(j, j)];
		let v = V.col(j);

		eigvals[idx] = Complex::new(real(*w), imag(*w));
		zip!(eigvecs.rb_mut().col_mut(idx), v).for_each(|unzip!(y, x)| *y = Complex::new(real(*x), imag(*x)));
	}
	limit
}

/// computes the layout of required workspace for computing the `n_eigval` eigenvalues
/// (and corresponding eigenvectors) of $A$ with the largest magnitude.
pub fn partial_eigen_scratch<T: ComplexField>(A: &dyn LinOp<T>, n_eigval: usize, par: Par, params: PartialEigenParams) -> StackReq {
	let n = A.nrows();
	assert!(A.ncols() == n);
	if n == 0 {
		return StackReq::EMPTY;
	}

	let n_eigval = Ord::min(n_eigval, n);

	let max_dim = Ord::min(Ord::max(params.max_dim, Ord::max(2 * MIN_DIM, 2 * n_eigval)), n);

	let w = temp_mat_scratch::<T>(max_dim, if T::IS_REAL { 2 } else { 1 });
	let residual = temp_mat_scratch::<T::Real>(max_dim, 1);
	let H = temp_mat_scratch::<T>(max_dim + 1, max_dim);
	let V = temp_mat_scratch::<T>(n, max_dim + 1);
	let Q = temp_mat_scratch::<T>(max_dim, max_dim);
	let X = Q;
	let vecs = Q;
	let tmp = temp_mat_scratch::<T>(n, max_dim);

	let blocksize = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(max_dim, max_dim);
	let householder = temp_mat_scratch::<T>(blocksize, max_dim);
	let arnoldi = A.apply_scratch(1, par).or(StackReq::new::<bool>(max_dim));

	let hess = linalg::evd::hessenberg::hessenberg_in_place_scratch::<T>(max_dim, blocksize, par, default());
	let apply_house = linalg::householder::apply_block_householder_sequence_on_the_right_in_place_scratch::<T>(max_dim - 1, blocksize, max_dim - 1);
	let schur = schur::multishift_qr_scratch::<T>(max_dim, max_dim, true, true, par, auto!(T));

	StackReq::all_of(&[
		w,
		residual,
		H,
		V,
		Q,
		X,
		vecs,
		tmp,
		householder,
		StackReq::any_of(&[hess, apply_house, schur, arnoldi]),
	])
}

/// computes an estimate of the eigenvalues (and corresponding eigenvectors) of $A$ with the largest
/// magnitude until the provided outputs are full or the maximum number of algorithm restarts is
/// reached.
pub fn partial_eigen<T: ComplexField>(
	eigvecs: MatMut<'_, Complex<T::Real>>,
	eigvals: &mut [Complex<T::Real>],
	A: &dyn LinOp<T>,
	v0: ColRef<'_, T>,
	tolerance: T::Real,
	par: Par,
	stack: &mut MemStack,
	params: PartialEigenParams,
) -> PartialEigenInfo {
	let n = v0.nrows();
	assert!(all(
		eigvals.len() == eigvecs.ncols(),
		A.nrows() == n,
		A.ncols() == n,
		eigvecs.nrows() == n,
	));
	let n_eigval = eigvals.len();
	let n_eigval = Ord::min(n_eigval, n);

	if n == 0 {
		return PartialEigenInfo {
			n_converged_eigen: 0,
			non_exhaustive: NonExhaustive(()),
		};
	}

	let min_dim = Ord::min(Ord::max(params.min_dim, Ord::max(MIN_DIM, n_eigval)), n);
	let max_dim = Ord::min(Ord::max(params.max_dim, Ord::max(2 * MIN_DIM, 2 * n_eigval)), n);

	let n_eigval = if const { T::IS_REAL } {
		partial_schur_real_imp::<T::Real>(
			eigvecs,
			eigvals,
			unsafe { core::mem::transmute(A) },
			unsafe { core::mem::transmute(v0) },
			min_dim,
			max_dim,
			n_eigval,
			tolerance,
			params.max_restarts,
			par,
			stack,
		)
	} else {
		partial_schur_cplx_imp(
			eigvecs,
			eigvals,
			A,
			v0,
			min_dim,
			max_dim,
			n_eigval,
			tolerance,
			params.max_restarts,
			par,
			stack,
		)
	};

	PartialEigenInfo {
		n_converged_eigen: n_eigval,
		non_exhaustive: NonExhaustive(()),
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::stats::prelude::*;
	use crate::{Scale, assert};
	use rand::prelude::*;

	#[test]
	fn test_arnoldi_real() {
		let rng = &mut StdRng::seed_from_u64(1);
		let n = 100;
		let n_eigval = 20;
		let min_dim = 30;
		let max_dim = 60;
		let restarts = 1000;

		let mat = CwiseMatDistribution {
			nrows: n,
			ncols: n,
			dist: StandardNormal,
		};
		let col = CwiseColDistribution {
			nrows: n,
			dist: StandardNormal,
		};
		let A: Mat<f64> = mat.sample(rng);

		let mut v0: Col<f64> = col.sample(rng);
		v0 /= v0.norm_l2();
		let A = A.as_ref();
		let v0 = v0.as_ref();

		let par = Par::Seq;
		let mem = &mut MemBuffer::new(partial_eigen_scratch(
			&A,
			n_eigval,
			par,
			PartialEigenParams {
				min_dim,
				max_dim,
				max_restarts: restarts,
				..Default::default()
			},
		));
		let mut V = Mat::zeros(n, n_eigval);
		let mut w = vec![c64::ZERO; n_eigval];

		let info = partial_eigen(
			V.rb_mut(),
			&mut w,
			&A,
			v0,
			f64::EPSILON * 128.0,
			par,
			MemStack::new(mem),
			PartialEigenParams {
				min_dim,
				max_dim,
				max_restarts: restarts,
				..Default::default()
			},
		);
		assert!(w.iter().map(|x| x.norm()).is_sorted_by(|x, y| x >= y));

		let A = &zip!(A).map(|unzip!(x)| Complex::from(*x));
		for j in 0..info.n_converged_eigen {
			assert!((A * V.col(j) - Scale(w[j]) * V.col(j)).norm_l2() < 1e-10);
		}
	}

	#[test]
	fn test_arnoldi_cplx() {
		let rng = &mut StdRng::seed_from_u64(1);
		let n = 100;
		let n_eigval = 20;
		let min_dim = 30;
		let max_dim = 60;
		let restarts = 1000;

		let mat = CwiseMatDistribution {
			nrows: n,
			ncols: n,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		};
		let col = CwiseColDistribution {
			nrows: n,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		};
		let A: Mat<c64> = mat.sample(rng);

		let mut v0: Col<c64> = col.sample(rng);
		v0 /= v0.norm_l2();
		let A = A.as_ref();
		let v0 = v0.as_ref();

		let par = Par::Seq;
		let mem = &mut MemBuffer::new(partial_eigen_scratch(
			&A,
			n_eigval,
			par,
			PartialEigenParams {
				min_dim,
				max_dim,
				max_restarts: restarts,
				..Default::default()
			},
		));

		let mut V = Mat::zeros(n, n_eigval);
		let mut w = vec![c64::ZERO; n_eigval];

		let info = partial_eigen(
			V.rb_mut(),
			&mut w,
			&A,
			v0,
			f64::EPSILON * 128.0,
			par,
			MemStack::new(mem),
			PartialEigenParams {
				min_dim,
				max_dim,
				max_restarts: restarts,
				..Default::default()
			},
		);
		assert!(w.iter().map(|x| x.norm()).is_sorted_by(|x, y| x >= y));

		for j in 0..info.n_converged_eigen {
			assert!((A * V.col(j) - Scale(w[j]) * V.col(j)).norm_l2() < 1e-10);
		}
	}

	#[test]
	fn test_toeplitz() {
		let n = 10;
		let mut mat = Mat::<f64>::zeros(n, n);

		for i in 0..n {
			mat[(i, i)] = 6.0;
			if i + 1 < n {
				mat[(i, i + 1)] = 3.0;
			}
			if i > 0 {
				mat[(i, i - 1)] = 3.0;
			}
		}

		let rng = &mut StdRng::seed_from_u64(1);
		let col = CwiseColDistribution {
			nrows: n,
			dist: StandardNormal,
		};
		let mut v0: Col<f64> = col.sample(rng);
		v0 /= v0.norm_l2();
		let A = mat.as_ref();
		let v0 = v0.as_ref();

		let par = Par::Seq;
		let n_eigval = 5;
		let min_dim = 7;
		let max_dim = 9;
		let mem = &mut MemBuffer::new(partial_eigen_scratch(
			&A,
			n_eigval,
			par,
			PartialEigenParams {
				min_dim,
				max_dim,
				max_restarts: 20,
				..Default::default()
			},
		));

		let mut V = Mat::zeros(n, n_eigval);
		let mut w = vec![c64::ZERO; n_eigval];

		let info = partial_eigen(
			V.rb_mut(),
			&mut w,
			&A,
			v0,
			0.000001,
			par,
			MemStack::new(mem),
			PartialEigenParams {
				min_dim,
				max_dim,
				max_restarts: 20,
				..Default::default()
			},
		);
		assert!(w.iter().map(|x| x.norm()).is_sorted_by(|x, y| x >= y));

		let A = &zip!(A).map(|unzip!(x)| Complex::from(*x));
		for j in 0..info.n_converged_eigen {
			assert!((A * V.col(j) - Scale(w[j]) * V.col(j)).norm_l2() < 1e-10);
		}
	}

	#[test]
	fn test_small_real() {
		let rng = &mut StdRng::seed_from_u64(1);
		let n = 5;
		let n_eigval = 3;

		let mat = CwiseMatDistribution {
			nrows: n,
			ncols: n,
			dist: StandardNormal,
		};
		let col = CwiseColDistribution {
			nrows: n,
			dist: StandardNormal,
		};
		let A: Mat<f64> = mat.sample(rng);

		let mut v0: Col<f64> = col.sample(rng);
		v0 /= v0.norm_l2();
		let A = A.as_ref();
		let v0 = v0.as_ref();

		let par = Par::Seq;
		let mem = &mut MemBuffer::new(partial_eigen_scratch(&A, n_eigval, par, default()));
		let mut V = Mat::zeros(n, n_eigval);
		let mut w = vec![c64::ZERO; n_eigval];

		let info = partial_eigen(V.rb_mut(), &mut w, &A, v0, f64::EPSILON * 128.0, par, MemStack::new(mem), default());
		assert!(w.iter().map(|x| x.norm()).is_sorted_by(|x, y| x >= y));

		let A = &zip!(A).map(|unzip!(x)| Complex::from(*x));
		for j in 0..info.n_converged_eigen {
			assert!((A * V.col(j) - Scale(w[j]) * V.col(j)).norm_l2() < 1e-10);
		}
	}

	#[test]
	fn test_small_cplx() {
		let rng = &mut StdRng::seed_from_u64(1);
		let n = 5;
		let n_eigval = 3;

		let mat = CwiseMatDistribution {
			nrows: n,
			ncols: n,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		};
		let col = CwiseColDistribution {
			nrows: n,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		};
		let A: Mat<c64> = mat.sample(rng);

		let mut v0: Col<c64> = col.sample(rng);
		v0 /= v0.norm_l2();
		let A = A.as_ref();
		let v0 = v0.as_ref();

		let par = Par::Seq;
		let mem = &mut MemBuffer::new(partial_eigen_scratch(&A, n_eigval, par, default()));
		let mut V = Mat::zeros(n, n_eigval);
		let mut w = vec![c64::ZERO; n_eigval];

		let info = partial_eigen(V.rb_mut(), &mut w, &A, v0, f64::EPSILON * 128.0, par, MemStack::new(mem), default());
		assert!(w.iter().map(|x| x.norm()).is_sorted_by(|x, y| x >= y));

		let A = &zip!(A).map(|unzip!(x)| Complex::from(*x));
		for j in 0..info.n_converged_eigen {
			assert!((A * V.col(j) - Scale(w[j]) * V.col(j)).norm_l2() < 1e-10);
		}
	}
}
