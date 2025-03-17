use crate::assert;
use crate::internal_prelude::*;

pub use linalg::cholesky::llt::factor::LltError;
use linalg::matmul::triangular::BlockStructure;

#[derive(Copy, Clone, Debug)]
pub struct PivLltParams {
	pub blocksize: usize,

	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

impl Default for PivLltParams {
	#[inline]
	fn default() -> Self {
		Self {
			blocksize: 128,
			non_exhaustive: NonExhaustive(()),
		}
	}
}

#[derive(Copy, Clone, Debug)]
pub struct PivLltInfo {
	/// numerical rank of the matrix
	pub rank: usize,
	/// number of transpositions that make up the permutation
	pub transposition_count: usize,
}

#[inline]
pub fn cholesky_in_place_scratch<I: Index, T: ComplexField>(dim: usize, par: Par, params: PivLltParams) -> StackReq {
	_ = par;
	_ = params;
	temp_mat_scratch::<T::Real>(dim, 2)
}

#[track_caller]
#[math]
pub fn cholesky_in_place<'out, I: Index, T: ComplexField>(
	a: MatMut<'_, T>,
	perm: &'out mut [I],
	perm_inv: &'out mut [I],
	par: Par,
	stack: &mut MemStack,
	params: PivLltParams,
) -> Result<(PivLltInfo, PermRef<'out, I>), LltError> {
	assert!(a.nrows() == a.ncols());
	let n = a.nrows();
	assert!(n <= I::Signed::MAX.zx());
	let mut rank = n;
	let mut transposition_count = 0;

	'exit: {
		if n > 0 {
			let mut a = a;
			for (i, p) in perm.iter_mut().enumerate() {
				*p = I::truncate(i);
			}

			let (mut work1, stack) = unsafe { temp_mat_uninit::<T::Real, _, _>(n, 1, stack) };
			let (mut work2, _) = unsafe { temp_mat_uninit::<T::Real, _, _>(n, 1, stack) };
			let work1 = work1.as_mat_mut();
			let work2 = work2.as_mat_mut();

			let mut dot_products = work1.col_mut(0);
			let mut diagonals = work2.col_mut(0);

			let mut ajj = zero::<T::Real>();
			let mut pvt = 0usize;

			for i in 0..n {
				let aii = real(a[(i, i)]);
				if aii < zero::<T::Real>() || is_nan(aii) {
					return Err(LltError::NonPositivePivot { index: 0 });
				}
				if aii > ajj {
					ajj = aii;
					pvt = i;
				}
			}

			let tol = eps::<T::Real>() * from_f64::<T::Real>(n as f64) * ajj;

			let mut k = 0usize;
			while k < n {
				let bs = Ord::min(n - k, params.blocksize);

				for i in k..n {
					dot_products[i] = zero::<T::Real>();
				}

				for j in k..k + bs {
					if j == k {
						for i in j..n {
							diagonals[i] = real(a[(i, i)]);
						}
					} else {
						for i in j..n {
							dot_products[i] = dot_products[i] + abs2(a[(i, j - 1)]);
							diagonals[i] = real(a[(i, i)]) - dot_products[i];
						}
					}

					if j > 0 {
						pvt = j;
						ajj = zero::<T::Real>();
						for i in j..n {
							let aii = real(diagonals[i]);
							if is_nan(aii) {
								return Err(LltError::NonPositivePivot { index: j });
							}
							if aii > ajj {
								pvt = i;
								ajj = aii;
							}
						}
						if ajj < tol {
							rank = j;
							a[(j, j)] = from_real(ajj);
							break 'exit;
						}
					}

					if pvt != j {
						transposition_count += 1;

						a[(pvt, pvt)] = copy(a[(j, j)]);
						crate::perm::swap_rows_idx(a.rb_mut().get_mut(.., ..j), j, pvt);
						crate::perm::swap_cols_idx(a.rb_mut().get_mut(pvt + 1.., ..), j, pvt);
						unsafe {
							z!(
								a.rb().get(j + 1..pvt, j).const_cast(),
								a.rb().get(pvt, j + 1..pvt).const_cast().transpose_mut(),
							)
						}
						.for_each(|uz!(a, b)| (*a, *b) = (conj(*b), conj(*a)));
						a[(pvt, j)] = conj(a[(pvt, j)]);

						let tmp = copy(dot_products[j]);
						dot_products[j] = copy(dot_products[pvt]);
						dot_products[pvt] = tmp;
						perm.swap(j, pvt);
					}

					ajj = sqrt(ajj);
					a[(j, j)] = from_real(ajj);
					unsafe {
						linalg::matmul::matmul(
							a.rb().get(j + 1.., j).const_cast(),
							Accum::Add,
							a.rb().get(j + 1.., k..j),
							a.rb().get(j, k..j).adjoint(),
							-one::<T>(),
							par,
						);
					}
					let ajj = recip(ajj);
					z!(a.rb_mut().get_mut(j + 1.., j)).for_each(|uz!(x)| *x = mul_real(*x, ajj));
				}

				linalg::matmul::triangular::matmul(
					unsafe { a.rb().get(k + bs.., k + bs..).const_cast() },
					BlockStructure::TriangularLower,
					Accum::Add,
					a.rb().get(k + bs.., k..k + bs),
					BlockStructure::Rectangular,
					a.rb().get(k + bs.., k..k + bs).adjoint(),
					BlockStructure::Rectangular,
					-one::<T>(),
					par,
				);

				k += bs;
			}
			rank = n;
		}
	}

	for (i, p) in perm.iter().enumerate() {
		perm_inv[p.zx()] = I::truncate(i);
	}

	unsafe { Ok((PivLltInfo { rank, transposition_count }, PermRef::new_unchecked(perm, perm_inv, n))) }
}
