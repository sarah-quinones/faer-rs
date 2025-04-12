use super::*;
use crate::linalg::temp_mat_uninit;
use crate::linalg::zip::Diag;
use crate::mat::{AsMatMut, MatMut, MatRef};
use crate::utils::thread::join_raw;
use crate::{assert, debug_assert, unzip, zip};

#[repr(u8)]
#[derive(Copy, Clone, Debug)]
pub(crate) enum DiagonalKind {
	Zero,
	Unit,
	Generic,
}

#[inline]
fn pointer_offset<T>(ptr: *const T) -> usize {
	if try_const! {core::mem::size_of::<T>().is_power_of_two() &&core::mem::size_of::<T>() <= 64 } {
		ptr.align_offset(64).wrapping_neg() % 16
	} else {
		0
	}
}

#[faer_macros::math]
fn copy_lower<'N, T: ComplexField>(dst: MatMut<'_, T, Dim<'N>, Dim<'N>>, src: MatRef<'_, T, Dim<'N>, Dim<'N>>, src_diag: DiagonalKind) {
	let N = dst.nrows();
	let mut dst = dst;
	match src_diag {
		DiagonalKind::Zero => {
			dst.copy_from_strict_triangular_lower(src);
			for j in N.indices() {
				let zero = zero();
				dst[(j, j)] = zero;
			}
		},
		DiagonalKind::Unit => {
			dst.copy_from_strict_triangular_lower(src);
			for j in N.indices() {
				let one = one();
				dst[(j, j)] = one;
			}
		},
		DiagonalKind::Generic => dst.copy_from_triangular_lower(src),
	}

	zip!(dst).for_each_triangular_upper(Diag::Skip, |unzip!(dst)| *dst = zero());
}

#[faer_macros::math]
fn accum_lower<'N, T: ComplexField>(dst: MatMut<'_, T, Dim<'N>, Dim<'N>>, src: MatRef<'_, T, Dim<'N>, Dim<'N>>, skip_diag: bool, beta: Accum) {
	let N = dst.nrows();
	debug_assert!(N == dst.nrows());
	debug_assert!(N == dst.ncols());
	debug_assert!(N == src.nrows());
	debug_assert!(N == src.ncols());

	match beta {
		Accum::Add => {
			zip!(dst, src).for_each_triangular_lower(if skip_diag { Diag::Skip } else { Diag::Include }, |unzip!(dst, src)| *dst = *dst + *src);
		},
		Accum::Replace => {
			zip!(dst, src).for_each_triangular_lower(if skip_diag { Diag::Skip } else { Diag::Include }, |unzip!(dst, src)| *dst = copy(*src));
		},
	}
}

#[faer_macros::math]
fn copy_upper<'N, T: ComplexField>(dst: MatMut<'_, T, Dim<'N>, Dim<'N>>, src: MatRef<'_, T, Dim<'N>, Dim<'N>>, src_diag: DiagonalKind) {
	copy_lower(dst.transpose_mut(), src.transpose(), src_diag)
}

#[repr(align(64))]
struct Storage<T>([T; 32 * 16]);

macro_rules! stack_mat_16x16 {
	($name: ident, $n: expr, $offset: expr, $rs: expr, $cs: expr,  $T: ty $(,)?) => {
		let mut __tmp = core::mem::MaybeUninit::<Storage<$T>>::uninit();
		let __stack = MemStack::new_any(core::slice::from_mut(&mut __tmp));
		let mut $name = unsafe { temp_mat_uninit(32, $n, __stack) }.0;
		let mut $name = $name.as_mat_mut().subrows_mut($offset, $n);
		if $cs.unsigned_abs() == 1 {
			$name = $name.transpose_mut();
			if $cs == 1 {
				$name = $name.transpose_mut().reverse_cols_mut();
			}
		} else if $rs == -1 {
			$name = $name.reverse_rows_mut();
		}
	};
}

#[faer_macros::math]
fn mat_x_lower_impl_unchecked<'M, 'N, T: ComplexField>(
	dst: MatMut<'_, T, Dim<'M>, Dim<'N>>,
	beta: Accum,
	lhs: MatRef<'_, T, Dim<'M>, Dim<'N>>,
	rhs: MatRef<'_, T, Dim<'N>, Dim<'N>>,
	rhs_diag: DiagonalKind,
	alpha: &T,
	conj_lhs: Conj,
	conj_rhs: Conj,
	par: Par,
) {
	let N = rhs.nrows();
	let M = lhs.nrows();
	let n = N.unbound();
	let m = M.unbound();
	debug_assert!(M == lhs.nrows());
	debug_assert!(N == lhs.ncols());
	debug_assert!(N == rhs.nrows());
	debug_assert!(N == rhs.ncols());
	debug_assert!(M == dst.nrows());
	debug_assert!(N == dst.ncols());

	let join_parallelism = if n * n * m < 128usize * 128usize * 64usize { Par::Seq } else { par };

	if n <= 16 {
		let op = {
			#[inline(never)]
			|| {
				stack_mat_16x16!(temp_rhs, N, pointer_offset(rhs.as_ptr()), rhs.row_stride(), rhs.col_stride(), T);

				copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);

				let mut dst = dst;
				super::matmul_with_conj(dst.rb_mut(), beta, lhs, conj_lhs, temp_rhs.rb(), conj_rhs, alpha.clone(), par);
			}
		};
		op();
	} else {
		// split rhs into 3 sections
		// split lhs and dst into 2 sections

		make_guard!(HEAD);
		make_guard!(TAIL);
		let bs = N.partition(N.checked_idx_inc(N.unbound() / 2), HEAD, TAIL);

		let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_with(bs, bs);
		let (lhs_left, lhs_right) = lhs.split_cols_with(bs);
		let (mut dst_left, mut dst_right) = dst.split_cols_with_mut(bs);

		{
			join_raw(
				|par| mat_x_lower_impl_unchecked(dst_left.rb_mut(), beta, lhs_left, rhs_top_left, rhs_diag, alpha, conj_lhs, conj_rhs, par),
				|par| {
					mat_x_lower_impl_unchecked(
						dst_right.rb_mut(),
						beta,
						lhs_right,
						rhs_bot_right,
						rhs_diag,
						alpha,
						conj_lhs,
						conj_rhs,
						par,
					)
				},
				join_parallelism,
			)
		};

		super::matmul_with_conj(dst_left, Accum::Add, lhs_right, conj_lhs, rhs_bot_left, conj_rhs, alpha.clone(), par);
	}
}

#[faer_macros::math]
fn lower_x_lower_into_lower_impl_unchecked<'N, T: ComplexField>(
	dst: MatMut<'_, T, Dim<'N>, Dim<'N>>,
	beta: Accum,
	skip_diag: bool,
	lhs: MatRef<'_, T, Dim<'N>, Dim<'N>>,
	lhs_diag: DiagonalKind,
	rhs: MatRef<'_, T, Dim<'N>, Dim<'N>>,
	rhs_diag: DiagonalKind,
	alpha: &T,
	conj_lhs: Conj,
	conj_rhs: Conj,
	par: Par,
) {
	let N = dst.nrows();
	let n = N.unbound();
	debug_assert!(N == lhs.nrows());
	debug_assert!(N == lhs.ncols());
	debug_assert!(N == rhs.nrows());
	debug_assert!(N == rhs.ncols());
	debug_assert!(N == dst.nrows());
	debug_assert!(N == dst.ncols());

	if n <= 16 {
		let op = {
			#[inline(never)]
			|| {
				stack_mat_16x16!(temp_dst, N, pointer_offset(dst.as_ptr()), dst.row_stride(), dst.col_stride(), T);
				stack_mat_16x16!(temp_lhs, N, pointer_offset(lhs.as_ptr()), lhs.row_stride(), lhs.col_stride(), T);
				stack_mat_16x16!(temp_rhs, N, pointer_offset(rhs.as_ptr()), rhs.row_stride(), rhs.col_stride(), T);

				copy_lower(temp_lhs.rb_mut(), lhs, lhs_diag);
				copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);

				super::matmul_with_conj(
					temp_dst.rb_mut(),
					Accum::Replace,
					temp_lhs.rb(),
					conj_lhs,
					temp_rhs.rb(),
					conj_rhs,
					alpha.clone(),
					par,
				);
				accum_lower(dst, temp_dst.rb(), skip_diag, beta);
			}
		};
		op();
	} else {
		make_guard!(HEAD);
		make_guard!(TAIL);
		let bs = N.partition(N.checked_idx_inc(N.unbound() / 2), HEAD, TAIL);

		let (dst_top_left, _, mut dst_bot_left, dst_bot_right) = dst.split_with_mut(bs, bs);
		let (lhs_top_left, _, lhs_bot_left, lhs_bot_right) = lhs.split_with(bs, bs);
		let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_with(bs, bs);

		// lhs_top_left  × rhs_top_left  => dst_top_left  | low × low => low |   X
		// lhs_bot_left  × rhs_top_left  => dst_bot_left  | mat × low => mat | 1/2
		// lhs_bot_right × rhs_bot_left  => dst_bot_left  | low × mat => mat | 1/2
		// lhs_bot_right × rhs_bot_right => dst_bot_right | low × low => low |   X

		lower_x_lower_into_lower_impl_unchecked(
			dst_top_left,
			beta,
			skip_diag,
			lhs_top_left,
			lhs_diag,
			rhs_top_left,
			rhs_diag,
			alpha,
			conj_lhs,
			conj_rhs,
			par,
		);
		mat_x_lower_impl_unchecked(
			dst_bot_left.rb_mut(),
			beta,
			lhs_bot_left,
			rhs_top_left,
			rhs_diag,
			alpha,
			conj_lhs,
			conj_rhs,
			par,
		);
		mat_x_lower_impl_unchecked(
			dst_bot_left.reverse_rows_and_cols_mut().transpose_mut(),
			Accum::Add,
			rhs_bot_left.reverse_rows_and_cols().transpose(),
			lhs_bot_right.reverse_rows_and_cols().transpose(),
			lhs_diag,
			alpha,
			conj_rhs,
			conj_lhs,
			par,
		);
		lower_x_lower_into_lower_impl_unchecked(
			dst_bot_right,
			beta,
			skip_diag,
			lhs_bot_right,
			lhs_diag,
			rhs_bot_right,
			rhs_diag,
			alpha,
			conj_lhs,
			conj_rhs,
			par,
		)
	}
}

#[math]
fn upper_x_lower_impl_unchecked<'N, T: ComplexField>(
	dst: MatMut<'_, T, Dim<'N>, Dim<'N>>,
	beta: Accum,
	lhs: MatRef<'_, T, Dim<'N>, Dim<'N>>,
	lhs_diag: DiagonalKind,
	rhs: MatRef<'_, T, Dim<'N>, Dim<'N>>,
	rhs_diag: DiagonalKind,
	alpha: &T,
	conj_lhs: Conj,
	conj_rhs: Conj,
	par: Par,
) {
	let N = dst.nrows();
	let n = N.unbound();
	debug_assert!(N == lhs.nrows());
	debug_assert!(N == lhs.ncols());
	debug_assert!(N == rhs.nrows());
	debug_assert!(N == rhs.ncols());
	debug_assert!(N == dst.nrows());
	debug_assert!(N == dst.ncols());

	if n <= 16 {
		let op = {
			#[inline(never)]
			|| {
				stack_mat_16x16!(temp_lhs, N, pointer_offset(lhs.as_ptr()), lhs.row_stride(), lhs.col_stride(), T);
				stack_mat_16x16!(temp_rhs, N, pointer_offset(rhs.as_ptr()), rhs.row_stride(), rhs.col_stride(), T);

				copy_upper(temp_lhs.rb_mut(), lhs, lhs_diag);
				copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);

				super::matmul_with_conj(dst, beta, temp_lhs.rb(), conj_lhs, temp_rhs.rb(), conj_rhs, alpha.clone(), par);
			}
		};
		op();
	} else {
		make_guard!(HEAD);
		make_guard!(TAIL);
		let bs = N.partition(N.checked_idx_inc(N.unbound() / 2), HEAD, TAIL);

		let (mut dst_top_left, dst_top_right, dst_bot_left, dst_bot_right) = dst.split_with_mut(bs, bs);
		let (lhs_top_left, lhs_top_right, _, lhs_bot_right) = lhs.split_with(bs, bs);
		let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_with(bs, bs);

		// lhs_top_right × rhs_bot_left  => dst_top_left  | mat × mat => mat |   1
		// lhs_top_left  × rhs_top_left  => dst_top_left  | upp × low => mat |   X
		//
		// lhs_top_right × rhs_bot_right => dst_top_right | mat × low => mat | 1/2
		// lhs_bot_right × rhs_bot_left  => dst_bot_left  | upp × mat => mat | 1/2
		// lhs_bot_right × rhs_bot_right => dst_bot_right | upp × low => mat |   X

		join_raw(
			|par| {
				super::matmul_with_conj(
					dst_top_left.rb_mut(),
					beta,
					lhs_top_right,
					conj_lhs,
					rhs_bot_left,
					conj_rhs,
					alpha.clone(),
					par,
				);
				upper_x_lower_impl_unchecked(
					dst_top_left,
					Accum::Add,
					lhs_top_left,
					lhs_diag,
					rhs_top_left,
					rhs_diag,
					alpha,
					conj_lhs,
					conj_rhs,
					par,
				)
			},
			|par| {
				join_raw(
					|par| {
						mat_x_lower_impl_unchecked(
							dst_top_right,
							beta,
							lhs_top_right,
							rhs_bot_right,
							rhs_diag,
							alpha,
							conj_lhs,
							conj_rhs,
							par,
						)
					},
					|par| {
						mat_x_lower_impl_unchecked(
							dst_bot_left.transpose_mut(),
							beta,
							rhs_bot_left.transpose(),
							lhs_bot_right.transpose(),
							lhs_diag,
							alpha,
							conj_rhs,
							conj_lhs,
							par,
						)
					},
					par,
				);

				upper_x_lower_impl_unchecked(
					dst_bot_right,
					beta,
					lhs_bot_right,
					lhs_diag,
					rhs_bot_right,
					rhs_diag,
					alpha,
					conj_lhs,
					conj_rhs,
					par,
				)
			},
			par,
		);
	}
}

#[math]
fn upper_x_lower_into_lower_impl_unchecked<'N, T: ComplexField>(
	dst: MatMut<'_, T, Dim<'N>, Dim<'N>>,
	beta: Accum,
	skip_diag: bool,
	lhs: MatRef<'_, T, Dim<'N>, Dim<'N>>,
	lhs_diag: DiagonalKind,
	rhs: MatRef<'_, T, Dim<'N>, Dim<'N>>,
	rhs_diag: DiagonalKind,
	alpha: &T,
	conj_lhs: Conj,
	conj_rhs: Conj,
	par: Par,
) {
	let N = dst.nrows();
	let n = N.unbound();
	debug_assert!(N == lhs.nrows());
	debug_assert!(N == lhs.ncols());
	debug_assert!(N == rhs.nrows());
	debug_assert!(N == rhs.ncols());
	debug_assert!(N == dst.nrows());
	debug_assert!(N == dst.ncols());

	if n <= 16 {
		let op = {
			#[inline(never)]
			|| {
				stack_mat_16x16!(temp_dst, N, pointer_offset(dst.as_ptr()), dst.row_stride(), dst.col_stride(), T);
				stack_mat_16x16!(temp_lhs, N, pointer_offset(lhs.as_ptr()), lhs.row_stride(), lhs.col_stride(), T);
				stack_mat_16x16!(temp_rhs, N, pointer_offset(rhs.as_ptr()), rhs.row_stride(), rhs.col_stride(), T);

				copy_upper(temp_lhs.rb_mut(), lhs, lhs_diag);
				copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);

				super::matmul_with_conj(
					temp_dst.rb_mut(),
					Accum::Replace,
					temp_lhs.rb(),
					conj_lhs,
					temp_rhs.rb(),
					conj_rhs,
					alpha.clone(),
					par,
				);

				accum_lower(dst, temp_dst.rb(), skip_diag, beta);
			}
		};
		op();
	} else {
		make_guard!(HEAD);
		make_guard!(TAIL);
		let bs = N.partition(N.checked_idx_inc(N.unbound() / 2), HEAD, TAIL);

		let (mut dst_top_left, _, dst_bot_left, dst_bot_right) = dst.split_with_mut(bs, bs);
		let (lhs_top_left, lhs_top_right, _, lhs_bot_right) = lhs.split_with(bs, bs);
		let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_with(bs, bs);

		// lhs_top_left  × rhs_top_left  => dst_top_left  | upp × low => low |   X
		// lhs_top_right × rhs_bot_left  => dst_top_left  | mat × mat => low | 1/2
		//
		// lhs_bot_right × rhs_bot_left  => dst_bot_left  | upp × mat => mat | 1/2
		// lhs_bot_right × rhs_bot_right => dst_bot_right | upp × low => low |   X

		join_raw(
			|par| {
				mat_x_mat_into_lower_impl_unchecked(
					dst_top_left.rb_mut(),
					beta,
					skip_diag,
					lhs_top_right,
					rhs_bot_left,
					alpha,
					conj_lhs,
					conj_rhs,
					par,
				);
				upper_x_lower_into_lower_impl_unchecked(
					dst_top_left,
					Accum::Add,
					skip_diag,
					lhs_top_left,
					lhs_diag,
					rhs_top_left,
					rhs_diag,
					alpha,
					conj_lhs,
					conj_rhs,
					par,
				)
			},
			|par| {
				mat_x_lower_impl_unchecked(
					dst_bot_left.transpose_mut(),
					beta,
					rhs_bot_left.transpose(),
					lhs_bot_right.transpose(),
					lhs_diag,
					alpha,
					conj_rhs,
					conj_lhs,
					par,
				);
				upper_x_lower_into_lower_impl_unchecked(
					dst_bot_right,
					beta,
					skip_diag,
					lhs_bot_right,
					lhs_diag,
					rhs_bot_right,
					rhs_diag,
					alpha,
					conj_lhs,
					conj_rhs,
					par,
				)
			},
			par,
		);
	}
}

#[math]
fn mat_x_mat_into_lower_impl_unchecked<'N, 'K, T: ComplexField>(
	dst: MatMut<'_, T, Dim<'N>, Dim<'N>>,
	beta: Accum,
	skip_diag: bool,
	lhs: MatRef<'_, T, Dim<'N>, Dim<'K>>,
	rhs: MatRef<'_, T, Dim<'K>, Dim<'N>>,
	alpha: &T,
	conj_lhs: Conj,
	conj_rhs: Conj,
	par: Par,
) {
	#[cfg(all(target_arch = "x86_64", feature = "std"))]
	if const { T::IS_NATIVE_F64 || T::IS_NATIVE_F32 || T::IS_NATIVE_C64 || T::IS_NATIVE_C32 } {
		use private_gemm_x86::*;

		let feat = if std::arch::is_x86_feature_detected!("avx512f") {
			Some(InstrSet::Avx512)
		} else if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
			Some(InstrSet::Avx256)
		} else {
			None
		};

		if *dst.nrows() > 0 && *dst.ncols() > 0 && *lhs.ncols() > 0 {
			if let Some(feat) = feat {
				unsafe {
					let (dst, lhs) = if skip_diag {
						(dst.as_dyn_mut().get_mut(1.., ..), lhs.as_dyn().get(1.., ..))
					} else {
						(dst.as_dyn_mut(), lhs.as_dyn())
					};

					private_gemm_x86::gemm(
						const {
							if T::IS_NATIVE_F64 {
								DType::F64
							} else if T::IS_NATIVE_F32 {
								DType::F32
							} else if T::IS_NATIVE_C64 {
								DType::C64
							} else {
								DType::C32
							}
						},
						const { IType::U32 },
						feat,
						dst.nrows(),
						dst.ncols(),
						lhs.ncols(),
						dst.as_ptr_mut() as _,
						dst.row_stride(),
						dst.col_stride(),
						core::ptr::null(),
						core::ptr::null(),
						DstKind::Lower,
						match beta {
							crate::Accum::Replace => Accum::Replace,
							crate::Accum::Add => Accum::Add,
						},
						lhs.as_ptr() as _,
						lhs.row_stride(),
						lhs.col_stride(),
						conj_lhs == Conj::Yes,
						core::ptr::null(),
						0,
						rhs.as_ptr() as _,
						rhs.row_stride(),
						rhs.col_stride(),
						conj_rhs == Conj::Yes,
						alpha as *const T as *const (),
						par.degree(),
					);

					return;
				}
			}
		}
	}

	let N = dst.nrows();
	let K = lhs.ncols();
	let n = N.unbound();
	let k = K.unbound();
	debug_assert!(dst.nrows() == dst.ncols());
	debug_assert!(dst.nrows() == lhs.nrows());
	debug_assert!(dst.ncols() == rhs.ncols());
	debug_assert!(lhs.ncols() == rhs.nrows());

	let par = if n * n * k < 128usize * 128usize * 128usize { Par::Seq } else { par };

	if n <= 16 {
		let op = {
			#[inline(never)]
			|| {
				stack_mat_16x16!(temp_dst, N, pointer_offset(dst.as_ptr()), dst.row_stride(), dst.col_stride(), T);

				super::matmul_with_conj(temp_dst.rb_mut(), Accum::Replace, lhs, conj_lhs, rhs, conj_rhs, alpha.clone(), par);
				accum_lower(dst, temp_dst.rb(), skip_diag, beta);
			}
		};
		op();
	} else {
		make_guard!(HEAD);
		make_guard!(TAIL);
		let bs = N.partition(N.checked_idx_inc(N.unbound() / 2), HEAD, TAIL);

		let (dst_top_left, _, dst_bot_left, dst_bot_right) = dst.split_with_mut(bs, bs);
		let (lhs_top, lhs_bot) = lhs.split_rows_with(bs);
		let (rhs_left, rhs_right) = rhs.split_cols_with(bs);

		join_raw(
			|par| super::matmul_with_conj(dst_bot_left, beta, lhs_bot, conj_lhs, rhs_left, conj_rhs, alpha.clone(), par),
			|par| {
				join_raw(
					|par| mat_x_mat_into_lower_impl_unchecked(dst_top_left, beta, skip_diag, lhs_top, rhs_left, alpha, conj_lhs, conj_rhs, par),
					|par| mat_x_mat_into_lower_impl_unchecked(dst_bot_right, beta, skip_diag, lhs_bot, rhs_right, alpha, conj_lhs, conj_rhs, par),
					par,
				)
			},
			par,
		);
	}
}

#[math]
fn mat_x_lower_into_lower_impl_unchecked<'N, T: ComplexField>(
	dst: MatMut<'_, T, Dim<'N>, Dim<'N>>,
	beta: Accum,
	skip_diag: bool,
	lhs: MatRef<'_, T, Dim<'N>, Dim<'N>>,
	rhs: MatRef<'_, T, Dim<'N>, Dim<'N>>,
	rhs_diag: DiagonalKind,
	alpha: &T,
	conj_lhs: Conj,
	conj_rhs: Conj,
	par: Par,
) {
	let N = dst.nrows();
	let n = N.unbound();
	debug_assert!(N == dst.nrows());
	debug_assert!(N == dst.ncols());
	debug_assert!(N == lhs.nrows());
	debug_assert!(N == lhs.ncols());
	debug_assert!(N == rhs.nrows());
	debug_assert!(N == rhs.ncols());

	if n <= 16 {
		let op = {
			#[inline(never)]
			|| {
				stack_mat_16x16!(temp_dst, N, pointer_offset(dst.as_ptr()), dst.row_stride(), dst.col_stride(), T);
				stack_mat_16x16!(temp_rhs, N, pointer_offset(rhs.as_ptr()), rhs.row_stride(), rhs.col_stride(), T);

				copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);
				super::matmul_with_conj(
					temp_dst.rb_mut(),
					Accum::Replace,
					lhs,
					conj_lhs,
					temp_rhs.rb(),
					conj_rhs,
					alpha.clone(),
					par,
				);
				accum_lower(dst, temp_dst.rb(), skip_diag, beta);
			}
		};
		op();
	} else {
		make_guard!(HEAD);
		make_guard!(TAIL);
		let bs = N.partition(N.checked_idx_inc(N.unbound() / 2), HEAD, TAIL);

		let (mut dst_top_left, _, mut dst_bot_left, dst_bot_right) = dst.split_with_mut(bs, bs);
		let (lhs_top_left, lhs_top_right, lhs_bot_left, lhs_bot_right) = lhs.split_with(bs, bs);
		let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_with(bs, bs);

		// lhs_bot_right × rhs_bot_left  => dst_bot_left  | mat × mat => mat |   1
		// lhs_bot_right × rhs_bot_right => dst_bot_right | mat × low => low |   X
		//
		// lhs_top_left  × rhs_top_left  => dst_top_left  | mat × low => low |   X
		// lhs_top_right × rhs_bot_left  => dst_top_left  | mat × mat => low | 1/2
		// lhs_bot_left  × rhs_top_left  => dst_bot_left  | mat × low => mat | 1/2

		super::matmul_with_conj(
			dst_bot_left.rb_mut(),
			beta,
			lhs_bot_right,
			conj_lhs,
			rhs_bot_left,
			conj_rhs,
			alpha.clone(),
			par,
		);
		mat_x_lower_into_lower_impl_unchecked(
			dst_bot_right,
			beta,
			skip_diag,
			lhs_bot_right,
			rhs_bot_right,
			rhs_diag,
			alpha,
			conj_lhs,
			conj_rhs,
			par,
		);

		mat_x_lower_into_lower_impl_unchecked(
			dst_top_left.rb_mut(),
			beta,
			skip_diag,
			lhs_top_left,
			rhs_top_left,
			rhs_diag,
			alpha,
			conj_lhs,
			conj_rhs,
			par,
		);
		mat_x_mat_into_lower_impl_unchecked(
			dst_top_left,
			Accum::Add,
			skip_diag,
			lhs_top_right,
			rhs_bot_left,
			alpha,
			conj_lhs,
			conj_rhs,
			par,
		);
		mat_x_lower_impl_unchecked(
			dst_bot_left,
			Accum::Add,
			lhs_bot_left,
			rhs_top_left,
			rhs_diag,
			alpha,
			conj_lhs,
			conj_rhs,
			par,
		);
	}
}

/// describes the parts of the matrix that must be accessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockStructure {
	/// the full matrix is accessed.
	Rectangular,
	/// the lower triangular half (including the diagonal) is accessed.
	TriangularLower,
	/// the lower triangular half (excluding the diagonal) is accessed.
	StrictTriangularLower,
	/// the lower triangular half (excluding the diagonal, which is assumed to be equal to
	/// `1.0`) is accessed.
	UnitTriangularLower,
	/// the upper triangular half (including the diagonal) is accessed.
	TriangularUpper,
	/// the upper triangular half (excluding the diagonal) is accessed.
	StrictTriangularUpper,
	/// the upper triangular half (excluding the diagonal, which is assumed to be equal to
	/// `1.0`) is accessed.
	UnitTriangularUpper,
}

impl BlockStructure {
	/// checks if `self` is full.
	#[inline]
	pub fn is_dense(self) -> bool {
		matches!(self, BlockStructure::Rectangular)
	}

	/// checks if `self` is triangular lower (either inclusive or exclusive).
	#[inline]
	pub fn is_lower(self) -> bool {
		use BlockStructure::*;
		matches!(self, TriangularLower | StrictTriangularLower | UnitTriangularLower)
	}

	/// checks if `self` is triangular upper (either inclusive or exclusive).
	#[inline]
	pub fn is_upper(self) -> bool {
		use BlockStructure::*;
		matches!(self, TriangularUpper | StrictTriangularUpper | UnitTriangularUpper)
	}

	/// returns the block structure corresponding to the transposed matrix.
	#[inline]
	pub fn transpose(self) -> Self {
		use BlockStructure::*;
		match self {
			Rectangular => Rectangular,
			TriangularLower => TriangularUpper,
			StrictTriangularLower => StrictTriangularUpper,
			UnitTriangularLower => UnitTriangularUpper,
			TriangularUpper => TriangularLower,
			StrictTriangularUpper => StrictTriangularLower,
			UnitTriangularUpper => UnitTriangularLower,
		}
	}

	#[inline]
	pub(crate) fn diag_kind(self) -> DiagonalKind {
		use BlockStructure::*;
		match self {
			Rectangular | TriangularLower | TriangularUpper => DiagonalKind::Generic,
			StrictTriangularLower | StrictTriangularUpper => DiagonalKind::Zero,
			UnitTriangularLower | UnitTriangularUpper => DiagonalKind::Unit,
		}
	}
}

#[track_caller]
fn precondition<M: Shape, N: Shape, K: Shape>(
	dst_nrows: M,
	dst_ncols: N,
	dst_structure: BlockStructure,
	lhs_nrows: M,
	lhs_ncols: K,
	lhs_structure: BlockStructure,
	rhs_nrows: K,
	rhs_ncols: N,
	rhs_structure: BlockStructure,
) {
	assert!(all(dst_nrows == lhs_nrows, dst_ncols == rhs_ncols, lhs_ncols == rhs_nrows,));

	let dst_nrows = dst_nrows.unbound();
	let dst_ncols = dst_ncols.unbound();
	let lhs_nrows = lhs_nrows.unbound();
	let lhs_ncols = lhs_ncols.unbound();
	let rhs_nrows = rhs_nrows.unbound();
	let rhs_ncols = rhs_ncols.unbound();

	if !dst_structure.is_dense() {
		assert!(dst_nrows == dst_ncols);
	}
	if !lhs_structure.is_dense() {
		assert!(lhs_nrows == lhs_ncols);
	}
	if !rhs_structure.is_dense() {
		assert!(rhs_nrows == rhs_ncols);
	}
}

/// computes the matrix product `[beta * acc] + alpha * lhs * rhs` (implicitly conjugating the
/// operands if needed) and stores the result in `acc`
///
/// performs the operation:
/// - `acc = alpha * lhs * rhs` if `beta` is `accum::replace` (in this case, the preexisting
/// values in `acc` are not read)
/// - `acc = acc + alpha * lhs * rhs` if `beta` is `accum::add`
///
/// the left hand side and right hand side may be interpreted as triangular depending on the
/// given corresponding matrix structure.
///
/// for the destination matrix, the result is:
/// - fully computed if the structure is rectangular,
/// - only the triangular half (including the diagonal) is computed if the structure is
/// triangular
/// - only the strict triangular half (excluding the diagonal) is computed if the structure is
/// strictly triangular or unit triangular
///
/// # panics
///
/// panics if the matrix dimensions are not compatible for matrix multiplication.
/// i.e.  
///  - `acc.nrows() == lhs.nrows()`
///  - `acc.ncols() == rhs.ncols()`
///  - `lhs.ncols() == rhs.nrows()`
///
/// additionally, matrices that are marked as triangular must be square, i.e., they must have
/// the same number of rows and columns.
///
/// # example
///
/// ```
/// use faer::linalg::matmul::triangular::{BlockStructure, matmul_with_conj};
/// use faer::{Accum, Conj, Mat, Par, mat, unzip, zip};
///
/// let lhs = mat![[0.0, 2.0], [1.0, 3.0]];
/// let rhs = mat![[4.0, 6.0], [5.0, 7.0]];
///
/// let mut acc = Mat::<f64>::zeros(2, 2);
/// let target = mat![
/// 	[
/// 		2.5 * (lhs[(0, 0)] * rhs[(0, 0)] + lhs[(0, 1)] * rhs[(1, 0)]),
/// 		0.0,
/// 	],
/// 	[
/// 		2.5 * (lhs[(1, 0)] * rhs[(0, 0)] + lhs[(1, 1)] * rhs[(1, 0)]),
/// 		2.5 * (lhs[(1, 0)] * rhs[(0, 1)] + lhs[(1, 1)] * rhs[(1, 1)]),
/// 	],
/// ];
///
/// matmul_with_conj(
/// 	&mut acc,
/// 	BlockStructure::TriangularLower,
/// 	Accum::Replace,
/// 	&lhs,
/// 	BlockStructure::Rectangular,
/// 	Conj::No,
/// 	&rhs,
/// 	BlockStructure::Rectangular,
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
	dst_structure: BlockStructure,
	beta: Accum,
	lhs: impl AsMatRef<T = T, Rows = M, Cols = K>,
	lhs_structure: BlockStructure,
	conj_lhs: Conj,
	rhs: impl AsMatRef<T = T, Rows = K, Cols = N>,
	rhs_structure: BlockStructure,
	conj_rhs: Conj,
	alpha: T,
	par: Par,
) {
	let mut dst = dst;
	let dst = dst.as_mat_mut();
	let lhs = lhs.as_mat_ref();
	let rhs = rhs.as_mat_ref();

	precondition(
		dst.nrows(),
		dst.ncols(),
		dst_structure,
		lhs.nrows(),
		lhs.ncols(),
		lhs_structure,
		rhs.nrows(),
		rhs.ncols(),
		rhs_structure,
	);

	make_guard!(M);
	make_guard!(N);
	make_guard!(K);
	let M = dst.nrows().bind(M);
	let N = dst.ncols().bind(N);
	let K = lhs.ncols().bind(K);

	matmul_imp(
		dst.as_dyn_stride_mut().as_shape_mut(M, N),
		dst_structure,
		beta,
		lhs.as_dyn_stride().canonical().as_shape(M, K),
		lhs_structure,
		conj_lhs,
		rhs.as_dyn_stride().canonical().as_shape(K, N),
		rhs_structure,
		conj_rhs,
		&alpha,
		par,
	);
}

/// computes the matrix product `[beta * acc] + alpha * lhs * rhs` (implicitly conjugating the
/// operands if needed) and stores the result in `acc`
///
/// performs the operation:
/// - `acc = alpha * lhs * rhs` if `beta` is `accum::replace` (in this case, the preexisting
/// values in `acc` are not read)
/// - `acc = acc + alpha * lhs * rhs` if `beta` is `accum::add`
///
/// the left hand side and right hand side may be interpreted as triangular depending on the
/// given corresponding matrix structure.
///
/// for the destination matrix, the result is:
/// - fully computed if the structure is rectangular,
/// - only the triangular half (including the diagonal) is computed if the structure is
/// triangular
/// - only the strict triangular half (excluding the diagonal) is computed if the structure is
/// strictly triangular or unit triangular
///
/// # panics
///
/// panics if the matrix dimensions are not compatible for matrix multiplication.
/// i.e.  
///  - `acc.nrows() == lhs.nrows()`
///  - `acc.ncols() == rhs.ncols()`
///  - `lhs.ncols() == rhs.nrows()`
///
/// additionally, matrices that are marked as triangular must be square, i.e., they must have
/// the same number of rows and columns.
///
/// # example
///
/// ```
/// use faer::linalg::matmul::triangular::{BlockStructure, matmul};
/// use faer::{Accum, Conj, Mat, Par, mat, unzip, zip};
///
/// let lhs = mat![[0.0, 2.0], [1.0, 3.0]];
/// let rhs = mat![[4.0, 6.0], [5.0, 7.0]];
///
/// let mut acc = Mat::<f64>::zeros(2, 2);
/// let target = mat![
/// 	[
/// 		2.5 * (lhs[(0, 0)] * rhs[(0, 0)] + lhs[(0, 1)] * rhs[(1, 0)]),
/// 		0.0,
/// 	],
/// 	[
/// 		2.5 * (lhs[(1, 0)] * rhs[(0, 0)] + lhs[(1, 1)] * rhs[(1, 0)]),
/// 		2.5 * (lhs[(1, 0)] * rhs[(0, 1)] + lhs[(1, 1)] * rhs[(1, 1)]),
/// 	],
/// ];
///
/// matmul(
/// 	&mut acc,
/// 	BlockStructure::TriangularLower,
/// 	Accum::Replace,
/// 	&lhs,
/// 	BlockStructure::Rectangular,
/// 	&rhs,
/// 	BlockStructure::Rectangular,
/// 	2.5,
/// 	Par::Seq,
/// );
///
/// zip!(&acc, &target).for_each(|unzip!(acc, target)| assert!((acc - target).abs() < 1e-10));
/// ```
#[track_caller]
#[inline]
pub fn matmul<T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, M: Shape, N: Shape, K: Shape>(
	dst: impl AsMatMut<T = T, Rows = M, Cols = N>,
	dst_structure: BlockStructure,
	beta: Accum,
	lhs: impl AsMatRef<T = LhsT, Rows = M, Cols = K>,
	lhs_structure: BlockStructure,
	rhs: impl AsMatRef<T = RhsT, Rows = K, Cols = N>,
	rhs_structure: BlockStructure,
	alpha: T,
	par: Par,
) {
	let mut dst = dst;
	let dst = dst.as_mat_mut();
	let lhs = lhs.as_mat_ref();
	let rhs = rhs.as_mat_ref();

	precondition(
		dst.nrows(),
		dst.ncols(),
		dst_structure,
		lhs.nrows(),
		lhs.ncols(),
		lhs_structure,
		rhs.nrows(),
		rhs.ncols(),
		rhs_structure,
	);

	make_guard!(M);
	make_guard!(N);
	make_guard!(K);
	let M = dst.nrows().bind(M);
	let N = dst.ncols().bind(N);
	let K = lhs.ncols().bind(K);

	matmul_imp(
		dst.as_dyn_stride_mut().as_shape_mut(M, N),
		dst_structure,
		beta,
		lhs.as_dyn_stride().canonical().as_shape(M, K),
		lhs_structure,
		try_const! { Conj::get::<LhsT>() },
		rhs.as_dyn_stride().canonical().as_shape(K, N),
		rhs_structure,
		try_const! { Conj::get::<RhsT>() },
		alpha.by_ref(),
		par,
	);
}

#[math]
fn matmul_imp<'M, 'N, 'K, T: ComplexField>(
	dst: MatMut<'_, T, Dim<'M>, Dim<'N>>,
	dst_structure: BlockStructure,
	beta: Accum,
	lhs: MatRef<'_, T, Dim<'M>, Dim<'K>>,
	lhs_structure: BlockStructure,
	conj_lhs: Conj,
	rhs: MatRef<'_, T, Dim<'K>, Dim<'N>>,
	rhs_structure: BlockStructure,
	conj_rhs: Conj,
	alpha: &T,
	par: Par,
) {
	let mut acc = dst.as_dyn_mut();
	let mut lhs = lhs.as_dyn();
	let mut rhs = rhs.as_dyn();

	let mut acc_structure = dst_structure;
	let mut lhs_structure = lhs_structure;
	let mut rhs_structure = rhs_structure;

	let mut conj_lhs = conj_lhs;
	let mut conj_rhs = conj_rhs;

	// if either the lhs or the rhs is triangular
	if rhs_structure.is_lower() {
		// do nothing
		false
	} else if rhs_structure.is_upper() {
		// invert acc, lhs and rhs
		acc = acc.reverse_rows_and_cols_mut();
		lhs = lhs.reverse_rows_and_cols();
		rhs = rhs.reverse_rows_and_cols();
		acc_structure = acc_structure.transpose();
		lhs_structure = lhs_structure.transpose();
		rhs_structure = rhs_structure.transpose();
		false
	} else if lhs_structure.is_lower() {
		// invert and transpose
		acc = acc.reverse_rows_and_cols_mut().transpose_mut();
		(lhs, rhs) = (rhs.reverse_rows_and_cols().transpose(), lhs.reverse_rows_and_cols().transpose());
		(conj_lhs, conj_rhs) = (conj_rhs, conj_lhs);
		(lhs_structure, rhs_structure) = (rhs_structure, lhs_structure);
		true
	} else if lhs_structure.is_upper() {
		// transpose
		acc_structure = acc_structure.transpose();
		acc = acc.transpose_mut();
		(lhs, rhs) = (rhs.transpose(), lhs.transpose());
		(conj_lhs, conj_rhs) = (conj_rhs, conj_lhs);
		(lhs_structure, rhs_structure) = (rhs_structure.transpose(), lhs_structure.transpose());
		true
	} else {
		// do nothing
		false
	};

	make_guard!(M);
	make_guard!(N);
	make_guard!(K);
	let M = acc.nrows().bind(M);
	let N = acc.ncols().bind(N);
	let K = lhs.ncols().bind(K);

	let clear_upper = |acc: MatMut<'_, T>, skip_diag: bool| match &beta {
		Accum::Add => {},

		Accum::Replace => zip!(acc).for_each_triangular_upper(if skip_diag { Diag::Skip } else { Diag::Include }, |unzip!(acc)| *acc = zero()),
	};

	let skip_diag = matches!(
		acc_structure,
		BlockStructure::StrictTriangularLower
			| BlockStructure::StrictTriangularUpper
			| BlockStructure::UnitTriangularLower
			| BlockStructure::UnitTriangularUpper
	);
	let lhs_diag = lhs_structure.diag_kind();
	let rhs_diag = rhs_structure.diag_kind();

	if acc_structure.is_dense() {
		if lhs_structure.is_dense() && rhs_structure.is_dense() {
			super::matmul_with_conj(acc, beta, lhs, conj_lhs, rhs, conj_rhs, alpha.clone(), par);
		} else {
			debug_assert!(rhs_structure.is_lower());

			if lhs_structure.is_dense() {
				mat_x_lower_impl_unchecked(
					acc.as_shape_mut(M, N),
					beta,
					lhs.as_shape(M, N),
					rhs.as_shape(N, N),
					rhs_diag,
					alpha,
					conj_lhs,
					conj_rhs,
					par,
				)
			} else if lhs_structure.is_lower() {
				clear_upper(acc.rb_mut(), true);
				lower_x_lower_into_lower_impl_unchecked(
					acc.as_shape_mut(N, N),
					beta,
					false,
					lhs.as_shape(N, N),
					lhs_diag,
					rhs.as_shape(N, N),
					rhs_diag,
					alpha,
					conj_lhs,
					conj_rhs,
					par,
				);
			} else {
				debug_assert!(lhs_structure.is_upper());
				upper_x_lower_impl_unchecked(
					acc.as_shape_mut(N, N),
					beta,
					lhs.as_shape(N, N),
					lhs_diag,
					rhs.as_shape(N, N),
					rhs_diag,
					alpha,
					conj_lhs,
					conj_rhs,
					par,
				)
			}
		}
	} else if acc_structure.is_lower() {
		if lhs_structure.is_dense() && rhs_structure.is_dense() {
			mat_x_mat_into_lower_impl_unchecked(
				acc.as_shape_mut(N, N),
				beta,
				skip_diag,
				lhs.as_shape(N, K),
				rhs.as_shape(K, N),
				alpha,
				conj_lhs,
				conj_rhs,
				par,
			)
		} else {
			debug_assert!(rhs_structure.is_lower());
			if lhs_structure.is_dense() {
				mat_x_lower_into_lower_impl_unchecked(
					acc.as_shape_mut(N, N),
					beta,
					skip_diag,
					lhs.as_shape(N, N),
					rhs.as_shape(N, N),
					rhs_diag,
					alpha,
					conj_lhs,
					conj_rhs,
					par,
				);
			} else if lhs_structure.is_lower() {
				lower_x_lower_into_lower_impl_unchecked(
					acc.as_shape_mut(N, N),
					beta,
					skip_diag,
					lhs.as_shape(N, N),
					lhs_diag,
					rhs.as_shape(N, N),
					rhs_diag,
					alpha,
					conj_lhs,
					conj_rhs,
					par,
				)
			} else {
				upper_x_lower_into_lower_impl_unchecked(
					acc.as_shape_mut(N, N),
					beta,
					skip_diag,
					lhs.as_shape(N, N),
					lhs_diag,
					rhs.as_shape(N, N),
					rhs_diag,
					alpha,
					conj_lhs,
					conj_rhs,
					par,
				)
			}
		}
	} else if lhs_structure.is_dense() && rhs_structure.is_dense() {
		mat_x_mat_into_lower_impl_unchecked(
			acc.as_shape_mut(N, N).transpose_mut(),
			beta,
			skip_diag,
			rhs.transpose().as_shape(N, K),
			lhs.transpose().as_shape(K, N),
			alpha,
			conj_rhs,
			conj_lhs,
			par,
		)
	} else {
		debug_assert!(rhs_structure.is_lower());
		if lhs_structure.is_dense() {
			// lower part of lhs does not contribute to result
			upper_x_lower_into_lower_impl_unchecked(
				acc.as_shape_mut(N, N).transpose_mut(),
				beta,
				skip_diag,
				rhs.transpose().as_shape(N, N),
				rhs_diag,
				lhs.transpose().as_shape(N, N),
				lhs_diag,
				alpha,
				conj_rhs,
				conj_lhs,
				par,
			)
		} else if lhs_structure.is_lower() {
			if !skip_diag {
				match beta {
					Accum::Add => {
						for j in 0..N.unbound() {
							acc[(j, j)] = acc[(j, j)] + *alpha * lhs[(j, j)] * rhs[(j, j)];
						}
					},
					Accum::Replace => {
						for j in 0..N.unbound() {
							acc[(j, j)] = *alpha * lhs[(j, j)] * rhs[(j, j)];
						}
					},
				}
			}
			clear_upper(acc.rb_mut(), true);
		} else {
			debug_assert!(lhs_structure.is_upper());
			upper_x_lower_into_lower_impl_unchecked(
				acc.as_shape_mut(N, N).transpose_mut(),
				beta,
				skip_diag,
				rhs.transpose().as_shape(N, N),
				rhs_diag,
				lhs.transpose().as_shape(N, N),
				lhs_diag,
				alpha,
				conj_rhs,
				conj_lhs,
				par,
			)
		}
	}
}
