//! block householder transformations
//!
//! a householder reflection is linear transformation that describes a reflection about a
//! hyperplane that crosses the origin of the space
//!
//! let $v$ be a unit vector that is orthogonal to the hyperplane. then the corresponding
//! householder transformation in matrix form is $I - 2vv^H$, where $I$ is the identity matrix
//!
//! in practice, a non unit vector $v$ is used, so the transformation is written as
//! $$H = I - \frac{vv^H}{\tau}$$
//!
//! a block householder transformation is a sequence of such transformations
//! $H_0, H_1, \dots, H_{b -1 }$ applied one after the other, with the restriction that the first
//! $i$ components of the vector $v_i$ of the $i$-th transformation are zero, and the component at
//! index $i$ is one
//!
//! the matrix $V = [v_0\ v_1\ \dots\ v_{b-1}]$ is thus a lower trapezoidal matrix with unit
//! diagonal. we call it the householder basis
//!
//! there exists a unique upper triangular matrix $T$, that we call the householder factor, such
//! that $$H_0 \times \dots \times H_{b-1} = I - VT^{-1}V^H$$
//!
//! a block householder sequence is a sequence of such transformations, composed of two matrices:
//! - a lower trapezoidal matrix with unit diagonal, which is the horizontal concatenation of the
//! bases of each block householder transformation,
//! - a horizontal concatenation of the householder factors.
//!
//! examples on how to create and manipulate block householder sequences are provided in the
//! documentation of the $QR$ module.

use crate::assert;
use crate::internal_prelude::*;
use crate::linalg::matmul::triangular::{self, BlockStructure};
use crate::linalg::matmul::{dot, matmul, matmul_with_conj};
use crate::linalg::triangular_solve;
use crate::utils::simd::SimdCtx;
use crate::utils::thread::join_raw;

/// Householder information
#[derive(Clone, Debug)]
pub struct HouseholderInfo<T: ComplexField> {
	/// The tau value of the householder transformation
	pub tau: T::Real,

	/// The reciprocal of head with beta added
	pub head_with_beta_inv: T,

	/// The norm
	pub norm: T::Real,
}

/// computes the householder reflection $I - \frac{v v^H}{\tau}$ such that when multiplied by $x$
/// from the left, the result is $\beta e_0$. $\tau$ and $(\text{head} - \beta)^{-1}$ are returned
/// and $\tau$ is real-valued. $\beta$ is stored in `head`
///
/// $x$ is determined by $x_0$, contained in `head`, and $|x_{1\dots}|$, contained in `tail_norm`.
/// the vector $v$ is such that $v_0 = 1$ and $v_{1\dots}$ is stored in `essential` (when provided)
#[math]
fn make_householder_imp<T: ComplexField>(head: &mut T, out: ColMut<'_, T>, input: Option<ColRef<'_, T>>) -> HouseholderInfo<T> {
	let tail = input.unwrap_or(out.rb());
	let tail_norm = tail.norm_l2();

	let mut head_norm = abs(*head);
	if head_norm < min_positive() {
		*head = zero();
		head_norm = zero();
	}

	if tail_norm < min_positive() {
		return HouseholderInfo {
			tau: infinity::<T::Real>(),
			head_with_beta_inv: infinity(),
			norm: head_norm,
		};
	}

	let one_half = from_f64::<T::Real>(0.5);

	let norm = hypot(head_norm, tail_norm);

	let sign = if head_norm != zero() { mul_real(*head, recip(head_norm)) } else { one() };

	let signed_norm = sign * from_real(norm);
	let head_with_beta = *head + signed_norm;
	let head_with_beta_inv = recip(head_with_beta);

	match input {
		None => zip!(out).for_each(|unzip!(e)| {
			*e = *e * head_with_beta_inv;
		}),
		Some(input) => zip!(out, input).for_each(|unzip!(o, e)| {
			*o = *e * head_with_beta_inv;
		}),
	}

	*head = -signed_norm;

	let tau = one_half * (one::<T::Real>() + abs2(tail_norm * abs(head_with_beta_inv)));
	HouseholderInfo {
		tau,
		head_with_beta_inv,
		norm,
	}
}

/// computes the householder reflection $I - \frac{v v^H}{\tau}$ such that when multiplied by $x$
/// from the left, the result is $\beta e_0$. $\tau$ and $(\text{head} - \beta)^{-1}$ are returned
/// and $\tau$ is real-valued. $\beta$ is stored in `head`
///
/// $x$ is determined by $x_0$, contained in `head`, and $|x_{1\dots}|$, contained in `tail_norm`.
/// the vector $v$ is such that $v_0 = 1$ and $v_{1\dots}$ is stored in `essential` (when provided)
#[inline]
pub fn make_householder_in_place<T: ComplexField>(head: &mut T, tail: ColMut<'_, T>) -> HouseholderInfo<T> {
	make_householder_imp(head, tail, None)
}

#[inline]
pub(crate) fn make_householder_out_of_place<T: ComplexField>(head: &mut T, out: ColMut<'_, T>, tail: ColRef<'_, T>) -> HouseholderInfo<T> {
	make_householder_imp(head, out, Some(tail))
}

#[doc(hidden)]
#[math]
pub fn upgrade_householder_factor<T: ComplexField>(
	householder_factor: MatMut<'_, T>,
	essentials: MatRef<'_, T>,
	blocksize: usize,
	prev_blocksize: usize,
	par: Par,
) {
	assert!(all(
		householder_factor.nrows() == householder_factor.ncols(),
		essentials.ncols() == householder_factor.ncols(),
	));

	if blocksize == prev_blocksize || householder_factor.nrows().unbound() <= prev_blocksize {
		return;
	}

	let n = essentials.ncols();
	let mut householder_factor = householder_factor;
	let essentials = essentials;

	assert!(householder_factor.nrows() == householder_factor.ncols());

	let block_count = householder_factor.nrows().msrv_div_ceil(blocksize);
	if block_count > 1 {
		assert!(all(blocksize > prev_blocksize, blocksize % prev_blocksize == 0,));
		let mid = block_count / 2;

		let (tau_tl, _, _, tau_br) = householder_factor.split_at_mut(mid, mid);
		let (basis_left, basis_right) = essentials.split_at_col(mid);
		let basis_right = basis_right.split_at_row(mid).1;
		join_raw(
			|parallelism| upgrade_householder_factor(tau_tl, basis_left, blocksize, prev_blocksize, parallelism),
			|parallelism| upgrade_householder_factor(tau_br, basis_right, blocksize, prev_blocksize, parallelism),
			par,
		);
		return;
	}

	if prev_blocksize < 8 {
		// pretend that prev_blocksize == 1, recompute whole top half of matrix

		let (basis_top, basis_bot) = essentials.split_at_row(n);
		let acc_structure = BlockStructure::UnitTriangularUpper;

		triangular::matmul(
			householder_factor.rb_mut(),
			acc_structure,
			Accum::Replace,
			basis_top.adjoint(),
			BlockStructure::UnitTriangularUpper,
			basis_top,
			BlockStructure::UnitTriangularLower,
			one(),
			par,
		);
		triangular::matmul(
			householder_factor.rb_mut(),
			acc_structure,
			Accum::Add,
			basis_bot.adjoint(),
			BlockStructure::Rectangular,
			basis_bot,
			BlockStructure::Rectangular,
			one(),
			par,
		);
	} else {
		let prev_block_count = householder_factor.nrows().msrv_div_ceil(prev_blocksize);

		let mid = (prev_block_count / 2) * prev_blocksize;

		let (tau_tl, mut tau_tr, _, tau_br) = householder_factor.split_at_mut(mid, mid);
		let (basis_left, basis_right) = essentials.split_at_col(mid);
		let basis_right = basis_right.split_at_row(mid).1;

		join_raw(
			|parallelism| {
				join_raw(
					|parallelism| upgrade_householder_factor(tau_tl, basis_left, blocksize, prev_blocksize, parallelism),
					|parallelism| upgrade_householder_factor(tau_br, basis_right, blocksize, prev_blocksize, parallelism),
					parallelism,
				);
			},
			|parallelism| {
				let basis_left = basis_left.split_at_row(mid).1;
				let row_mid = basis_right.ncols();

				let (basis_left_top, basis_left_bot) = basis_left.split_at_row(row_mid);
				let (basis_right_top, basis_right_bot) = basis_right.split_at_row(row_mid);

				triangular::matmul(
					tau_tr.rb_mut(),
					BlockStructure::Rectangular,
					Accum::Replace,
					basis_left_top.adjoint(),
					BlockStructure::Rectangular,
					basis_right_top,
					BlockStructure::UnitTriangularLower,
					one(),
					parallelism,
				);
				matmul(tau_tr.rb_mut(), Accum::Add, basis_left_bot.adjoint(), basis_right_bot, one(), parallelism);
			},
			par,
		);
	}
}

/// computes the size and alignment of required workspace for applying a block householder
/// transformation to a right-hand-side matrix in place
pub fn apply_block_householder_on_the_left_in_place_scratch<T: ComplexField>(
	householder_basis_nrows: usize,
	blocksize: usize,
	rhs_ncols: usize,
) -> StackReq {
	let _ = householder_basis_nrows;
	temp_mat_scratch::<T>(blocksize, rhs_ncols)
}

/// computes the size and alignment of required workspace for applying the transpose of a block
/// householder transformation to a right-hand-side matrix in place
pub fn apply_block_householder_transpose_on_the_left_in_place_scratch<T: ComplexField>(
	householder_basis_nrows: usize,
	blocksize: usize,
	rhs_ncols: usize,
) -> StackReq {
	let _ = householder_basis_nrows;
	temp_mat_scratch::<T>(blocksize, rhs_ncols)
}

/// computes the size and alignment of required workspace for applying a block householder
/// transformation to a left-hand-side matrix in place
pub fn apply_block_householder_on_the_right_in_place_scratch<T: ComplexField>(
	householder_basis_nrows: usize,
	blocksize: usize,
	lhs_nrows: usize,
) -> StackReq {
	let _ = householder_basis_nrows;
	temp_mat_scratch::<T>(blocksize, lhs_nrows)
}

/// computes the size and alignment of required workspace for applying the transpose of a block
/// householder transformation to a left-hand-side matrix in place
pub fn apply_block_householder_transpose_on_the_right_in_place_scratch<T: ComplexField>(
	householder_basis_nrows: usize,
	blocksize: usize,
	lhs_nrows: usize,
) -> StackReq {
	let _ = householder_basis_nrows;
	temp_mat_scratch::<T>(blocksize, lhs_nrows)
}

/// computes the size and alignment of required workspace for applying the transpose of a sequence
/// of block householder transformations to a right-hand-side matrix in place
pub fn apply_block_householder_sequence_transpose_on_the_left_in_place_scratch<T: ComplexField>(
	householder_basis_nrows: usize,
	blocksize: usize,
	rhs_ncols: usize,
) -> StackReq {
	let _ = householder_basis_nrows;
	temp_mat_scratch::<T>(blocksize, rhs_ncols)
}

/// computes the size and alignment of required workspace for applying a sequence of block
/// householder transformations to a right-hand-side matrix in place
pub fn apply_block_householder_sequence_on_the_left_in_place_scratch<T: ComplexField>(
	householder_basis_nrows: usize,
	blocksize: usize,
	rhs_ncols: usize,
) -> StackReq {
	let _ = householder_basis_nrows;
	temp_mat_scratch::<T>(blocksize, rhs_ncols)
}

/// computes the size and alignment of required workspace for applying the transpose of a sequence
/// of block householder transformations to a left-hand-side matrix in place
pub fn apply_block_householder_sequence_transpose_on_the_right_in_place_scratch<T: ComplexField>(
	householder_basis_nrows: usize,
	blocksize: usize,
	lhs_nrows: usize,
) -> StackReq {
	let _ = householder_basis_nrows;
	temp_mat_scratch::<T>(blocksize, lhs_nrows)
}

/// computes the size and alignment of required workspace for applying a sequence of block
/// householder transformations to a left-hand-side matrix in place
pub fn apply_block_householder_sequence_on_the_right_in_place_scratch<T: ComplexField>(
	householder_basis_nrows: usize,
	blocksize: usize,
	lhs_nrows: usize,
) -> StackReq {
	let _ = householder_basis_nrows;
	temp_mat_scratch::<T>(blocksize, lhs_nrows)
}

#[track_caller]
#[math]
fn apply_block_householder_on_the_left_in_place_generic<'M, 'N, 'K, T: ComplexField>(
	householder_basis: MatRef<'_, T, Dim<'M>, Dim<'N>>,
	householder_factor: MatRef<'_, T, Dim<'N>, Dim<'N>>,
	conj_lhs: Conj,
	matrix: MatMut<'_, T, Dim<'M>, Dim<'K>>,
	forward: bool,
	par: Par,
	stack: &mut MemStack,
) {
	assert!(all(
		householder_factor.nrows() == householder_factor.ncols(),
		householder_basis.ncols() == householder_factor.nrows(),
		matrix.nrows() == householder_basis.nrows(),
	));

	let mut matrix = matrix;

	let M = householder_basis.nrows();
	let N = householder_basis.ncols();

	make_guard!(TAIL);
	let midpoint = M.head_partition(N, TAIL);

	if let (Some(householder_basis), Some(matrix), 1, true) = (
		householder_basis.try_as_col_major(),
		matrix.rb_mut().try_as_col_major_mut(),
		N.unbound(),
		T::SIMD_CAPABILITIES.is_simd(),
	) {
		struct ApplyOnLeft<'a, 'TAIL, 'K, T: ComplexField, const CONJ: bool> {
			tau_inv: &'a T,
			essential: ColRef<'a, T, Dim<'TAIL>, ContiguousFwd>,
			rhs0: RowMut<'a, T, Dim<'K>>,
			rhs: MatMut<'a, T, Dim<'TAIL>, Dim<'K>, ContiguousFwd>,
		}

		impl<'TAIL, 'K, T: ComplexField, const CONJ: bool> pulp::WithSimd for ApplyOnLeft<'_, 'TAIL, 'K, T, CONJ> {
			type Output = ();

			#[inline(always)]
			fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
				let Self {
					tau_inv,
					essential,
					mut rhs,
					mut rhs0,
				} = self;

				if rhs.nrows().unbound() == 0 {
					return;
				}

				let N = rhs.nrows();
				let K = rhs.ncols();
				let simd = SimdCtx::<T, S>::new(T::simd_ctx(simd), N);

				let (head, indices, tail) = simd.indices();

				for idx in K.indices() {
					let col0 = rhs0.rb_mut().at_mut(idx);
					let mut col = rhs.rb_mut().col_mut(idx);
					let essential = essential;

					let dot = if try_const! { CONJ } {
						*col0 + dot::inner_prod_no_conj_simd(simd, essential.rb(), col.rb())
					} else {
						*col0 + dot::inner_prod_conj_lhs_simd(simd, essential.rb(), col.rb())
					};

					let k = -dot * tau_inv;
					*col0 = *col0 + k;

					let k = simd.splat(&k);
					macro_rules! simd {
						($i: expr) => {{
							let i = $i;
							let mut a = simd.read(col.rb(), i);
							let b = simd.read(essential.rb(), i);

							if try_const! { CONJ } {
								a = simd.conj_mul_add(b, k, a);
							} else {
								a = simd.mul_add(b, k, a);
							}

							simd.write(col.rb_mut(), i, a);
						}};
					}

					if let Some(i) = head {
						simd!(i);
					}
					for i in indices.clone() {
						simd!(i);
					}
					if let Some(i) = tail {
						simd!(i);
					}
				}
			}
		}

		let N0 = N.check(0);

		let essential = householder_basis.col(N0).split_rows_with(midpoint).1;
		let (rhs0, rhs) = matrix.split_rows_with_mut(midpoint);
		let rhs0 = rhs0.row_mut(N0);

		let tau_inv: T = from_real(recip(real(householder_factor[(N0, N0)])));

		if try_const! { T::IS_REAL } {
			type Apply<'a, 'TAIL, 'K, T> = ApplyOnLeft<'a, 'TAIL, 'K, T, false>;

			dispatch!(
				Apply {
					tau_inv: &tau_inv,
					essential,
					rhs,
					rhs0,
				},
				Apply,
				T
			);
		} else if matches!(conj_lhs, Conj::No) {
			type Apply<'a, 'TAIL, 'K, T> = ApplyOnLeft<'a, 'TAIL, 'K, T, false>;

			dispatch!(
				Apply {
					tau_inv: &tau_inv,
					essential,
					rhs,
					rhs0,
				},
				Apply,
				T
			);
		} else {
			type Apply<'a, 'TAIL, 'K, T> = ApplyOnLeft<'a, 'TAIL, 'K, T, true>;

			dispatch!(
				Apply {
					tau_inv: &tau_inv,
					essential,
					rhs,
					rhs0,
				},
				Apply,
				T
			);
		}
	} else {
		let (essentials_top, essentials_bot) = householder_basis.split_rows_with(midpoint);
		let M = matrix.nrows();
		let K = matrix.ncols();

		// essentials* × mat
		let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(N, K, stack) };
		let mut tmp = tmp.as_mat_mut();

		let mut n_tasks = Ord::min(Ord::min(crate::utils::thread::parallelism_degree(par), K.unbound()), 4);
		if (M.unbound() * K.unbound()).saturating_mul(4 * M.unbound()) < gemm::get_threading_threshold() {
			n_tasks = 1;
		}

		let inner_parallelism = match par {
			Par::Seq => Par::Seq,
			#[cfg(feature = "rayon")]
			Par::Rayon(par) => {
				let par = par.get();

				if par >= 2 * n_tasks { Par::rayon(par / n_tasks) } else { Par::Seq }
			},
		};

		let func = |(mut tmp, mut matrix): (MatMut<'_, T, Dim<'N>>, MatMut<'_, T, Dim<'M>>)| {
			let (mut top, mut bot) = matrix.rb_mut().split_rows_with_mut(midpoint);

			triangular::matmul_with_conj(
				tmp.rb_mut(),
				BlockStructure::Rectangular,
				Accum::Replace,
				essentials_top.transpose(),
				BlockStructure::UnitTriangularUpper,
				Conj::Yes.compose(conj_lhs),
				top.rb(),
				BlockStructure::Rectangular,
				Conj::No,
				one(),
				inner_parallelism,
			);

			matmul_with_conj(
				tmp.rb_mut(),
				Accum::Add,
				essentials_bot.transpose(),
				Conj::Yes.compose(conj_lhs),
				bot.rb(),
				Conj::No,
				one(),
				inner_parallelism,
			);

			// [T^-1|T^-*] × essentials* × tmp
			if forward {
				triangular_solve::solve_lower_triangular_in_place_with_conj(
					householder_factor.transpose(),
					Conj::Yes.compose(conj_lhs),
					tmp.rb_mut(),
					inner_parallelism,
				);
			} else {
				triangular_solve::solve_upper_triangular_in_place_with_conj(
					householder_factor,
					Conj::No.compose(conj_lhs),
					tmp.rb_mut(),
					inner_parallelism,
				);
			}

			// essentials × [T^-1|T^-*] × essentials* × tmp
			triangular::matmul_with_conj(
				top.rb_mut(),
				BlockStructure::Rectangular,
				Accum::Add,
				essentials_top,
				BlockStructure::UnitTriangularLower,
				Conj::No.compose(conj_lhs),
				tmp.rb(),
				BlockStructure::Rectangular,
				Conj::No,
				-one::<T>(),
				inner_parallelism,
			);
			matmul_with_conj(
				bot.rb_mut(),
				Accum::Add,
				essentials_bot,
				Conj::No.compose(conj_lhs),
				tmp.rb(),
				Conj::No,
				-one::<T>(),
				inner_parallelism,
			);
		};

		if n_tasks <= 1 {
			func((tmp.as_dyn_cols_mut(), matrix.as_dyn_cols_mut()));
			return;
		} else {
			#[cfg(feature = "rayon")]
			{
				use rayon::prelude::*;
				tmp.rb_mut()
					.par_col_partition_mut(n_tasks)
					.zip_eq(matrix.rb_mut().par_col_partition_mut(n_tasks))
					.for_each(func);
			}
		}
	}
}

/// computes the product of the matrix, multiplied by the given block householder transformation,
/// and stores the result in `matrix`
#[track_caller]
pub fn apply_block_householder_on_the_right_in_place_with_conj<T: ComplexField>(
	householder_basis: MatRef<'_, T>,
	householder_factor: MatRef<'_, T>,
	conj_rhs: Conj,
	matrix: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	apply_block_householder_transpose_on_the_left_in_place_with_conj(
		householder_basis,
		householder_factor,
		conj_rhs,
		matrix.transpose_mut(),
		par,
		stack,
	)
}

/// computes the product of the matrix, multiplied by the transpose of the given block householder
/// transformation, and stores the result in `matrix`
#[track_caller]
pub fn apply_block_householder_transpose_on_the_right_in_place_with_conj<T: ComplexField>(
	householder_basis: MatRef<'_, T>,
	householder_factor: MatRef<'_, T>,
	conj_rhs: Conj,
	matrix: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	apply_block_householder_on_the_left_in_place_with_conj(householder_basis, householder_factor, conj_rhs, matrix.transpose_mut(), par, stack)
}

/// computes the product of the given block householder transformation, multiplied by `matrix`, and
/// stores the result in `matrix`
#[track_caller]
pub fn apply_block_householder_on_the_left_in_place_with_conj<T: ComplexField>(
	householder_basis: MatRef<'_, T>,
	householder_factor: MatRef<'_, T>,
	conj_lhs: Conj,
	matrix: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	make_guard!(M);
	make_guard!(N);
	make_guard!(K);
	let M = householder_basis.nrows().bind(M);
	let N = householder_basis.ncols().bind(N);
	let K = matrix.ncols().bind(K);

	apply_block_householder_on_the_left_in_place_generic(
		householder_basis.as_shape(M, N).as_dyn_stride(),
		householder_factor.as_shape(N, N).as_dyn_stride(),
		conj_lhs,
		matrix.as_shape_mut(M, K).as_dyn_stride_mut(),
		false,
		par,
		stack,
	)
}

/// computes the product of the transpose of the given block householder transformation, multiplied
/// by `matrix`, and stores the result in `matrix`
#[track_caller]
pub fn apply_block_householder_transpose_on_the_left_in_place_with_conj<T: ComplexField>(
	householder_basis: MatRef<'_, T>,
	householder_factor: MatRef<'_, T>,
	conj_lhs: Conj,
	matrix: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	with_dim!(M, householder_basis.nrows());
	with_dim!(N, householder_basis.ncols());
	with_dim!(K, matrix.ncols());

	apply_block_householder_on_the_left_in_place_generic(
		householder_basis.as_shape(M, N).as_dyn_stride(),
		householder_factor.as_shape(N, N).as_dyn_stride(),
		conj_lhs.compose(Conj::Yes),
		matrix.as_shape_mut(M, K).as_dyn_stride_mut(),
		true,
		par,
		stack,
	)
}

/// computes the product of a sequence of block householder transformations given by
/// `householder_basis` and `householder_factor`, multiplied by `matrix`, and stores the result in
/// `matrix`
#[track_caller]
pub fn apply_block_householder_sequence_on_the_left_in_place_with_conj<T: ComplexField>(
	householder_basis: MatRef<'_, T>,
	householder_factor: MatRef<'_, T>,
	conj_lhs: Conj,
	matrix: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	let mut matrix = matrix;
	let mut stack = stack;
	let m = householder_basis.nrows();
	let n = householder_basis.ncols();

	assert!(all(householder_factor.nrows() > 0, householder_factor.ncols() == Ord::min(m, n),));

	let size = householder_factor.ncols();

	let mut j = size;

	let mut blocksize = size % householder_factor.nrows();
	if blocksize == 0 {
		blocksize = householder_factor.nrows();
	}

	while j > 0 {
		let j_prev = j - blocksize;
		blocksize = householder_factor.nrows();

		let essentials = householder_basis.get(j_prev.., j_prev..j);
		let householder = householder_factor.get(.., j_prev..j).subrows(0, j - j_prev);
		let matrix = matrix.rb_mut().get_mut(j_prev.., ..);

		apply_block_householder_on_the_left_in_place_with_conj(essentials, householder, conj_lhs, matrix, par, stack.rb_mut());

		j = j_prev;
	}
}

/// computes the product of the transpose of a sequence block householder transformations given by
/// `householder_basis` and `householder_factor`, multiplied by `matrix`, and stores the result in
/// `matrix`
#[track_caller]
pub fn apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj<T: ComplexField>(
	householder_basis: MatRef<'_, T>,
	householder_factor: MatRef<'_, T>,
	conj_lhs: Conj,
	matrix: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	let mut matrix = matrix;
	let mut stack = stack;

	let blocksize = householder_factor.nrows();

	let m = householder_basis.nrows();
	let n = householder_basis.ncols();

	assert!(all(householder_factor.nrows() > 0, householder_factor.ncols() == Ord::min(m, n),));

	let size = householder_factor.ncols();

	let mut j = 0;
	while j < size {
		let blocksize = Ord::min(blocksize, size - j);

		let essentials = householder_basis.get(j.., j..j + blocksize);
		let householder = householder_factor.get(.., j..j + blocksize).subrows(0, blocksize);

		let matrix = matrix.rb_mut().get_mut(j.., ..);

		apply_block_householder_transpose_on_the_left_in_place_with_conj(essentials, householder, conj_lhs, matrix, par, stack.rb_mut());

		j += blocksize;
	}
}

/// computes the product of `matrix`, multiplied by a sequence of block householder transformations
/// given by `householder_basis` and `householder_factor`, and stores the result in `matrix`
#[track_caller]
pub fn apply_block_householder_sequence_on_the_right_in_place_with_conj<T: ComplexField>(
	householder_basis: MatRef<'_, T>,
	householder_factor: MatRef<'_, T>,
	conj_rhs: Conj,
	matrix: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
		householder_basis,
		householder_factor,
		conj_rhs,
		matrix.transpose_mut(),
		par,
		stack,
	)
}

/// computes the product of `matrix`, multiplied by the transpose of a sequence of block householder
/// transformations given by `householder_basis` and `householder_factor`, and stores the result in
/// `matrix`
#[track_caller]
pub fn apply_block_householder_sequence_transpose_on_the_right_in_place_with_conj<T: ComplexField>(
	householder_basis: MatRef<'_, T>,
	householder_factor: MatRef<'_, T>,
	conj_rhs: Conj,
	matrix: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	apply_block_householder_sequence_on_the_left_in_place_with_conj(
		householder_basis,
		householder_factor,
		conj_rhs,
		matrix.transpose_mut(),
		par,
		stack,
	)
}
