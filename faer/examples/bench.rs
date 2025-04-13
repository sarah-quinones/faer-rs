#![allow(non_snake_case, non_camel_case_types, unused_imports, dead_code)]

use std::collections::HashMap;
use std::ffi::*;
use std::ptr::*;

use aligned_vec::avec;
use diol::prelude::*;
use diol::result::BenchResult;
use dyn_stack::{MemBuffer, MemStack};
use equator::assert;
use reborrow::*;
use toml::{Table, Value};

use ::faer::diag::Diag;
use ::faer::linalg::cholesky::lblt::factor::LbltParams;
use ::faer::prelude::*;
use ::faer::stats::prelude::*;
use ::faer::{Auto, linalg};
use ::faer_traits::math_utils::*;

use ::nalgebra as na;

#[cfg(any(openblas, mkl, blis))]
use lapack_sys as la;
#[cfg(any(openblas, mkl, blis))]
extern crate lapack_src;
#[cfg(any(openblas, mkl, blis))]
extern crate openmp_sys;

#[cfg(openblas)]
extern crate openblas_src;

#[cfg(mkl)]
extern crate intel_mkl_src;

#[cfg(blis)]
extern crate blis_src;

#[cfg(any(openblas, mkl, blis))]
unsafe extern "C" {
	#[cfg(openblas)]
	fn openblas_set_num_threads(num: i32) -> c_void;
	#[cfg(openblas)]
	fn goto_set_num_threads(num: i32) -> c_void;

	#[cfg(mkl)]
	fn MKL_Set_Num_Threads(num: i32) -> c_void;

	#[cfg(blis)]
	fn bli_thread_set_num_threads(num: i64) -> c_void;

	fn omp_set_num_threads(num: i32) -> c_void;

	fn sgetc2_(n: *const c_int, A: *mut f32, lda: *const c_int, ipiv: *mut c_int, jpiv: *mut c_int, info: *mut c_int) -> c_void;
	fn dgetc2_(n: *const c_int, A: *mut f64, lda: *const c_int, ipiv: *mut c_int, jpiv: *mut c_int, info: *mut c_int) -> c_void;
	fn cgetc2_(n: *const c_int, A: *mut c32, lda: *const c_int, ipiv: *mut c_int, jpiv: *mut c_int, info: *mut c_int) -> c_void;
	fn zgetc2_(n: *const c_int, A: *mut c64, lda: *const c_int, ipiv: *mut c_int, jpiv: *mut c_int, info: *mut c_int) -> c_void;
}

fn lapack_set_num_threads(parallel: Par) {
	let _ = parallel;
	#[cfg(any(openblas, mkl, blis))]
	match parallel {
		Par::Seq => unsafe {
			#[cfg(openblas)]
			openblas_set_num_threads(1);
			#[cfg(openblas)]
			goto_set_num_threads(1);

			#[cfg(mkl)]
			MKL_Set_Num_Threads(1);

			#[cfg(blis)]
			bli_thread_set_num_threads(1);

			omp_set_num_threads(1);
		},
		Par::Rayon(nthreads) => unsafe {
			let nthreads = nthreads.get();

			#[cfg(openblas)]
			openblas_set_num_threads(nthreads as _);
			#[cfg(openblas)]
			goto_set_num_threads(nthreads as _);

			#[cfg(mkl)]
			MKL_Set_Num_Threads(nthreads as _);

			#[cfg(blis)]
			bli_thread_set_num_threads(nthreads as _);

			omp_set_num_threads(nthreads as _);
		},
	};
}

trait Scalar: faer_traits::ComplexField + na::ComplexField {
	const IS_NATIVE: bool = Self::IS_NATIVE_F32 || Self::IS_NATIVE_C32 || Self::IS_NATIVE_F64 || Self::IS_NATIVE_C64;

	fn random(rng: &mut dyn RngCore, nrows: usize, ncols: usize) -> Mat<Self>;
}

trait Lib {
	const LAPACK: bool = false;
	const FAER: bool = false;
	const NALGEBRA: bool = false;
}

trait Thread {
	const SEQ: bool = false;
	const PAR: bool = false;
}

#[cfg(openblas)]
struct openblas;
#[cfg(openblas)]
type lapack = openblas;

#[cfg(mkl)]
struct mkl;
#[cfg(mkl)]
type lapack = mkl;

#[cfg(blis)]
struct blis;
#[cfg(blis)]
type lapack = blis;

#[cfg(not(any(openblas, mkl, blis)))]
struct lapack;

struct faer;
struct nalgebra;

struct seq;
struct par;

impl Thread for seq {
	const SEQ: bool = true;
}
impl Thread for par {
	const PAR: bool = true;
}

impl Lib for faer {
	const FAER: bool = true;
}
impl Lib for lapack {
	const LAPACK: bool = true;
}
impl Lib for nalgebra {
	const NALGEBRA: bool = true;
}

impl Scalar for f64 {
	fn random(rng: &mut dyn RngCore, nrows: usize, ncols: usize) -> Mat<Self> {
		CwiseMatDistribution {
			nrows,
			ncols,
			dist: StandardNormal,
		}
		.rand(rng)
	}
}
impl Scalar for f32 {
	fn random(rng: &mut dyn RngCore, nrows: usize, ncols: usize) -> Mat<Self> {
		CwiseMatDistribution {
			nrows,
			ncols,
			dist: StandardNormal,
		}
		.rand(rng)
	}
}
impl Scalar for c32 {
	fn random(rng: &mut dyn RngCore, nrows: usize, ncols: usize) -> Mat<Self> {
		CwiseMatDistribution {
			nrows,
			ncols,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		}
		.rand(rng)
	}
}
impl Scalar for c64 {
	fn random(rng: &mut dyn RngCore, nrows: usize, ncols: usize) -> Mat<Self> {
		CwiseMatDistribution {
			nrows,
			ncols,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		}
		.rand(rng)
	}
}

fn llt<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && Lib::NALGEBRA) || (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis)))) {
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A * A.adjoint() + Scale(from_f64::<T>(m as f64)) * Mat::<T>::identity(n, n);
	let mut L = Mat::zeros(n, n);
	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::cholesky::llt::factor::cholesky_in_place_scratch::<T>(n, parallel, params));
	let stack = MemStack::new(stack);

	if !Lib::LAPACK || T::IS_NATIVE {
		bencher.bench(|| {
			if Lib::FAER {
				L.copy_from_triangular_lower(&A);
				linalg::cholesky::llt::factor::cholesky_in_place(L.rb_mut(), Default::default(), parallel, stack, params).unwrap();
			} else if Lib::LAPACK {
				L.copy_from_triangular_lower(&A);
				#[cfg(any(openblas, mkl, blis))]
				unsafe {
					if T::IS_NATIVE_F32 {
						la::spotrf_(
							&(b'L' as i8),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_F64 {
						la::dpotrf_(
							&(b'L' as i8),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C32 {
						la::cpotrf_(
							&(b'L' as i8),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C64 {
						la::zpotrf_(
							&(b'L' as _),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					}
				}
			} else if Lib::NALGEBRA {
				unsafe {
					na::DMatrixView::from_data(na::ViewStorage::from_raw_parts(
						A.as_ptr(),
						(na::Dyn(n), na::Dyn(n)),
						(na::Const::<1>, na::Dyn(A.col_stride() as usize)),
					))
					.clone_owned()
					.cholesky()
					.unwrap();
				};
			}
		})
	}
}

fn ldlt<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && Lib::NALGEBRA) || (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis)))) {
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A * A.adjoint() + Scale(from_f64::<T>(m as f64)) * Mat::<T>::identity(n, n);
	let mut L = Mat::zeros(n, n);
	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::cholesky::ldlt::factor::cholesky_in_place_scratch::<T>(n, parallel, params));
	let stack = MemStack::new(stack);

	if Lib::FAER {
		bencher.bench(|| {
			L.copy_from_triangular_lower(&A);
			linalg::cholesky::ldlt::factor::cholesky_in_place(L.rb_mut(), Default::default(), parallel, stack, params).unwrap();
		})
	}
}

fn lblt<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && Lib::NALGEBRA) || (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis)))) {
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A + A.adjoint();
	let mut L = Mat::zeros(n, n);
	let mut subdiag = Diag::zeros(n);
	let fwd = &mut *avec![0usize; n];
	let bwd = &mut *avec![0usize; n];

	let params = LbltParams {
		pivoting: linalg::cholesky::lblt::factor::PivotingStrategy::Partial,
		..Auto::<T>::auto()
	}
	.into();

	#[cfg(any(openblas, mkl, blis))]
	let lwork = unsafe {
		if T::IS_NATIVE_F32 {
			let mut lwork = core::mem::zeroed();
			la::ssytrf_(
				&(b'L' as _),
				(&n) as *const _ as *const _,
				L.as_ptr_mut() as _,
				(&L.col_stride()) as *const _ as *const _,
				fwd.as_mut_ptr() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork as usize
		} else if T::IS_NATIVE_F64 {
			let mut lwork = core::mem::zeroed();
			la::dsytrf_(
				&(b'L' as _),
				(&n) as *const _ as *const _,
				L.as_ptr_mut() as _,
				(&L.col_stride()) as *const _ as *const _,
				fwd.as_mut_ptr() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork as usize
		} else if T::IS_NATIVE_C32 {
			let mut lwork = core::mem::zeroed();
			la::csytrf_(
				&(b'L' as _),
				(&n) as *const _ as *const _,
				L.as_ptr_mut() as _,
				(&L.col_stride()) as *const _ as *const _,
				fwd.as_mut_ptr() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork.re as usize
		} else if T::IS_NATIVE_C64 {
			let mut lwork = core::mem::zeroed();
			la::zsytrf_(
				&(b'L' as _),
				(&n) as *const _ as *const _,
				L.as_ptr_mut() as _,
				(&L.col_stride()) as *const _ as *const _,
				fwd.as_mut_ptr() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork.re as usize
		} else {
			0
		}
	};

	#[cfg(any(openblas, mkl, blis))]
	let work = &mut *avec![zero::<T>(); lwork];

	let stack = &mut MemBuffer::new(linalg::cholesky::lblt::factor::cholesky_in_place_scratch::<usize, T>(n, parallel, params));
	let stack = MemStack::new(stack);

	if Lib::FAER || (Lib::LAPACK && T::IS_NATIVE) {
		bencher.bench(|| {
			L.copy_from_triangular_lower(&A);
			if Lib::FAER {
				linalg::cholesky::lblt::factor::cholesky_in_place(L.rb_mut(), subdiag.rb_mut(), fwd, bwd, parallel, stack, params);
			} else if Lib::LAPACK {
				fwd.fill(0);

				#[cfg(any(openblas, mkl, blis))]
				unsafe {
					if T::IS_NATIVE_F32 {
						la::ssytrf_(
							&(b'L' as _),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_F64 {
						la::dsytrf_(
							&(b'L' as _),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C32 {
						la::csytrf_(
							&(b'L' as _),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C64 {
						la::zsytrf_(
							&(b'L' as _),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					}
				}
			}
		})
	}
}

fn lblt_diag<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && Lib::NALGEBRA) || (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis)))) {
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A + A.adjoint();
	let mut L = Mat::zeros(n, n);
	let mut subdiag = Diag::zeros(n);
	let fwd = &mut *avec![0usize; n];
	let bwd = &mut *avec![0usize; n];

	let params = LbltParams {
		pivoting: linalg::cholesky::lblt::factor::PivotingStrategy::PartialDiag,
		..Auto::<T>::auto()
	}
	.into();

	let stack = &mut MemBuffer::new(linalg::cholesky::lblt::factor::cholesky_in_place_scratch::<usize, T>(n, parallel, params));
	let stack = MemStack::new(stack);
	if Lib::FAER {
		bencher.bench(|| {
			L.copy_from_triangular_lower(&A);
			linalg::cholesky::lblt::factor::cholesky_in_place(L.rb_mut(), subdiag.rb_mut(), fwd, bwd, parallel, stack, params);
		})
	}
}

fn lblt_rook<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && Lib::NALGEBRA) || (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis)))) {
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A + A.adjoint();
	let mut L = Mat::zeros(n, n);
	let mut subdiag = Diag::zeros(n);
	let fwd = &mut *avec![0usize; n];
	let bwd = &mut *avec![0usize; n];

	let params = LbltParams {
		pivoting: linalg::cholesky::lblt::factor::PivotingStrategy::Rook,
		..Auto::<T>::auto()
	}
	.into();

	#[cfg(any(openblas, mkl, blis))]
	let lwork = unsafe {
		if T::IS_NATIVE_F32 {
			let mut lwork = core::mem::zeroed();
			la::ssytrf_(
				&(b'L' as _),
				(&n) as *const _ as *const _,
				L.as_ptr_mut() as _,
				(&L.col_stride()) as *const _ as *const _,
				fwd.as_mut_ptr() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork as usize
		} else if T::IS_NATIVE_F64 {
			let mut lwork = core::mem::zeroed();
			la::dsytrf_(
				&(b'L' as _),
				(&n) as *const _ as *const _,
				L.as_ptr_mut() as _,
				(&L.col_stride()) as *const _ as *const _,
				fwd.as_mut_ptr() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork as usize
		} else if T::IS_NATIVE_C32 {
			let mut lwork = core::mem::zeroed();
			la::csytrf_(
				&(b'L' as _),
				(&n) as *const _ as *const _,
				L.as_ptr_mut() as _,
				(&L.col_stride()) as *const _ as *const _,
				fwd.as_mut_ptr() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork.re as usize
		} else if T::IS_NATIVE_C64 {
			let mut lwork = core::mem::zeroed();
			la::zsytrf_(
				&(b'L' as _),
				(&n) as *const _ as *const _,
				L.as_ptr_mut() as _,
				(&L.col_stride()) as *const _ as *const _,
				fwd.as_mut_ptr() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork.re as usize
		} else {
			0
		}
	};

	#[cfg(any(openblas, mkl, blis))]
	let work = &mut *avec![zero::<T>(); lwork];

	let stack = &mut MemBuffer::new(linalg::cholesky::lblt::factor::cholesky_in_place_scratch::<usize, T>(n, parallel, params));
	let stack = MemStack::new(stack);
	if Lib::FAER || (Lib::LAPACK && T::IS_NATIVE) {
		bencher.bench(|| {
			L.copy_from_triangular_lower(&A);
			if Lib::FAER {
				linalg::cholesky::lblt::factor::cholesky_in_place(L.rb_mut(), subdiag.rb_mut(), fwd, bwd, parallel, stack, params);
			} else if Lib::LAPACK {
				fwd.fill(0);

				#[cfg(any(openblas, mkl, blis))]
				unsafe {
					if T::IS_NATIVE_F32 {
						la::ssytrf_rook_(
							&(b'L' as _),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_F64 {
						la::dsytrf_rook_(
							&(b'L' as _),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C32 {
						la::csytrf_rook_(
							&(b'L' as _),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C64 {
						la::zsytrf_rook_(
							&(b'L' as _),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					}
				}
			}
		})
	}
}

fn lblt_rook_diag<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && Lib::NALGEBRA) || (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis)))) {
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A + A.adjoint();
	let mut L = Mat::zeros(n, n);
	let mut subdiag = Diag::zeros(n);
	let fwd = &mut *avec![0usize; n];
	let bwd = &mut *avec![0usize; n];

	let params = LbltParams {
		pivoting: linalg::cholesky::lblt::factor::PivotingStrategy::Rook,
		..Auto::<T>::auto()
	}
	.into();

	let stack = &mut MemBuffer::new(linalg::cholesky::lblt::factor::cholesky_in_place_scratch::<usize, T>(n, parallel, params));
	let stack = MemStack::new(stack);

	if Lib::FAER {
		bencher.bench(|| {
			L.copy_from_triangular_lower(&A);
			linalg::cholesky::lblt::factor::cholesky_in_place(L.rb_mut(), subdiag.rb_mut(), fwd, bwd, parallel, stack, params);
		})
	}
}
fn lblt_full<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && Lib::NALGEBRA) || (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis)))) {
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A + A.adjoint();
	let mut L = Mat::zeros(n, n);
	let mut subdiag = Diag::zeros(n);
	let fwd = &mut *avec![0usize; n];
	let bwd = &mut *avec![0usize; n];

	let params = LbltParams {
		pivoting: linalg::cholesky::lblt::factor::PivotingStrategy::Full,
		..Auto::<T>::auto()
	}
	.into();

	let stack = &mut MemBuffer::new(linalg::cholesky::lblt::factor::cholesky_in_place_scratch::<usize, T>(n, parallel, params));
	let stack = MemStack::new(stack);

	if Lib::FAER {
		bencher.bench(|| {
			L.copy_from_triangular_lower(&A);
			linalg::cholesky::lblt::factor::cholesky_in_place(L.rb_mut(), subdiag.rb_mut(), fwd, bwd, parallel, stack, params);
		})
	}
}

fn qr<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && Lib::NALGEBRA) || (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis))))
		// parallel mkl sometimes segfaults here ¯\_(ツ)_/¯
		|| (Lib::LAPACK && Thd::PAR && cfg!(mkl))
	{
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	let blocksize = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(m, n);

	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);

	let mut Q = Mat::zeros(blocksize, Ord::min(m, n));
	let mut QR = Mat::zeros(m, n);

	#[cfg(any(openblas, mkl, blis))]
	let lwork = unsafe {
		if T::IS_NATIVE_F32 {
			let mut lwork = core::mem::zeroed();
			la::sgeqrf_(
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				QR.as_ptr_mut() as _,
				(&QR.col_stride()) as *const _ as *const _,
				Q.as_ptr_mut() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork as usize
		} else if T::IS_NATIVE_F64 {
			let mut lwork = core::mem::zeroed();
			la::dgeqrf_(
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				QR.as_ptr_mut() as _,
				(&QR.col_stride()) as *const _ as *const _,
				Q.as_ptr_mut() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork as usize
		} else if T::IS_NATIVE_C32 {
			let mut lwork = core::mem::zeroed();
			la::cgeqrf_(
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				QR.as_ptr_mut() as _,
				(&QR.col_stride()) as *const _ as *const _,
				Q.as_ptr_mut() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork.re as usize
		} else if T::IS_NATIVE_C64 {
			let mut lwork = core::mem::zeroed();
			la::zgeqrf_(
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				QR.as_ptr_mut() as _,
				(&QR.col_stride()) as *const _ as *const _,
				Q.as_ptr_mut() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork.re as usize
		} else {
			0
		}
	};

	#[cfg(any(openblas, mkl, blis))]
	let work = &mut *avec![zero::<T>(); lwork];

	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::qr::no_pivoting::factor::qr_in_place_scratch::<T>(
		m, n, blocksize, parallel, params,
	));
	let stack = MemStack::new(stack);

	if !Lib::LAPACK || T::IS_NATIVE {
		bencher.bench(|| {
			if Lib::FAER {
				QR.copy_from(&A);
				linalg::qr::no_pivoting::factor::qr_in_place(QR.rb_mut(), Q.rb_mut(), parallel, stack, params);
			} else if Lib::LAPACK {
				QR.copy_from(&A);
				#[cfg(any(openblas, mkl, blis))]
				unsafe {
					if T::IS_NATIVE_F32 {
						la::sgeqrf_(
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							QR.as_ptr_mut() as _,
							(&QR.col_stride()) as *const _ as *const _,
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_F64 {
						la::dgeqrf_(
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							QR.as_ptr_mut() as _,
							(&QR.col_stride()) as *const _ as *const _,
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C32 {
						la::cgeqrf_(
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							QR.as_ptr_mut() as _,
							(&QR.col_stride()) as *const _ as *const _,
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C64 {
						la::zgeqrf_(
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							QR.as_ptr_mut() as _,
							(&QR.col_stride()) as *const _ as *const _,
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					}
				}
			} else if Lib::NALGEBRA {
				unsafe {
					na::DMatrixView::from_data(na::ViewStorage::from_raw_parts(
						A.as_ptr(),
						(na::Dyn(n), na::Dyn(n)),
						(na::Const::<1>, na::Dyn(A.col_stride() as usize)),
					))
					.clone_owned()
					.qr();
				};
			}
		});
	}
}

fn col_piv_qr<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && Lib::NALGEBRA) || (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis)))) {
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	let blocksize = linalg::qr::col_pivoting::factor::recommended_blocksize::<T>(m, n);

	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);

	let mut Q = Mat::zeros(blocksize, Ord::min(m, n));
	let mut QR = Mat::zeros(m, n);

	let col_fwd = &mut *avec![0usize; n];
	let col_bwd = &mut *avec![0usize; n];

	#[cfg(any(openblas, mkl, blis))]
	let rwork = &mut *avec![zero::<T>(); 2 * n];

	#[cfg(any(openblas, mkl, blis))]
	let lwork = unsafe {
		if T::IS_NATIVE_F32 {
			let mut lwork = core::mem::zeroed();
			la::sgeqp3_(
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				QR.as_ptr_mut() as _,
				(&QR.col_stride()) as *const _ as *const _,
				col_fwd.as_mut_ptr() as _,
				Q.as_ptr_mut() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork as usize
		} else if T::IS_NATIVE_F64 {
			let mut lwork = core::mem::zeroed();
			la::dgeqp3_(
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				QR.as_ptr_mut() as _,
				(&QR.col_stride()) as *const _ as *const _,
				col_fwd.as_mut_ptr() as _,
				Q.as_ptr_mut() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork as usize
		} else if T::IS_NATIVE_C32 {
			let mut lwork = core::mem::zeroed();
			la::cgeqp3_(
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				QR.as_ptr_mut() as _,
				(&QR.col_stride()) as *const _ as *const _,
				col_fwd.as_mut_ptr() as _,
				Q.as_ptr_mut() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				rwork.as_mut_ptr() as _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork.re as usize
		} else if T::IS_NATIVE_C64 {
			let mut lwork = core::mem::zeroed();
			la::zgeqp3_(
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				QR.as_ptr_mut() as _,
				(&QR.col_stride()) as *const _ as *const _,
				col_fwd.as_mut_ptr() as _,
				Q.as_ptr_mut() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				rwork.as_mut_ptr() as _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork.re as usize
		} else {
			0
		}
	};

	#[cfg(any(openblas, mkl, blis))]
	let work = &mut *avec![zero::<T>(); lwork];

	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::qr::col_pivoting::factor::qr_in_place_scratch::<usize, T>(
		m, n, blocksize, parallel, params,
	));
	let stack = MemStack::new(stack);

	if !Lib::LAPACK || T::IS_NATIVE {
		bencher.bench(|| {
			if Lib::FAER {
				QR.copy_from(&A);
				linalg::qr::col_pivoting::factor::qr_in_place(QR.rb_mut(), Q.rb_mut(), col_fwd, col_bwd, parallel, stack, params);
			} else if Lib::LAPACK {
				QR.copy_from(&A);
				col_fwd.fill(0);
				#[cfg(any(openblas, mkl, blis))]
				unsafe {
					if T::IS_NATIVE_F32 {
						la::sgeqp3_(
							(&mut { m } as *mut _) as *mut _,
							(&mut { n } as *mut _) as *mut _,
							QR.as_ptr_mut() as _,
							(&mut { QR.col_stride() } as *mut _) as *mut _,
							col_fwd.as_mut_ptr() as _,
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_F64 {
						la::dgeqp3_(
							(&mut { m } as *mut _) as *mut _,
							(&mut { n } as *mut _) as *mut _,
							QR.as_ptr_mut() as _,
							(&mut { QR.col_stride() } as *mut _) as *mut _,
							col_fwd.as_mut_ptr() as _,
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C32 {
						la::cgeqp3_(
							(&mut { m } as *mut _) as *mut _,
							(&mut { n } as *mut _) as *mut _,
							QR.as_ptr_mut() as _,
							(&mut { QR.col_stride() } as *mut _) as *mut _,
							col_fwd.as_mut_ptr() as _,
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							rwork.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C64 {
						la::zgeqp3_(
							(&mut { m } as *mut _) as *mut _,
							(&mut { n } as *mut _) as *mut _,
							QR.as_ptr_mut() as _,
							(&mut { QR.col_stride() } as *mut _) as *mut _,
							col_fwd.as_mut_ptr() as _,
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							rwork.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					}
				}
			} else if Lib::NALGEBRA {
				unsafe {
					na::DMatrixView::from_data(na::ViewStorage::from_raw_parts(
						A.as_ptr(),
						(na::Dyn(n), na::Dyn(n)),
						(na::Const::<1>, na::Dyn(A.col_stride() as usize)),
					))
					.clone_owned()
					.col_piv_qr();
				};
			}
		});
	}
}

fn partial_piv_lu<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && Lib::NALGEBRA) || (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis)))) {
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let mut LU = Mat::zeros(n, n);
	let row_fwd = &mut *avec![0usize; n];
	let row_bwd = &mut *avec![0usize; n];

	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, T>(
		n, n, parallel, params,
	));
	let stack = MemStack::new(stack);

	if !Lib::LAPACK || T::IS_NATIVE {
		bencher.bench(|| {
			if Lib::FAER {
				LU.copy_from(&A);
				linalg::lu::partial_pivoting::factor::lu_in_place(LU.rb_mut(), row_fwd, row_bwd, parallel, stack, params);
			} else if Lib::LAPACK {
				LU.copy_from(&A);
				row_fwd.fill(0);

				#[cfg(any(openblas, mkl, blis))]
				unsafe {
					if T::IS_NATIVE_F32 {
						la::sgetrf_(
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							LU.as_ptr_mut() as _,
							(&LU.col_stride()) as *const _ as *const _,
							row_fwd.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_F64 {
						la::dgetrf_(
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							LU.as_ptr_mut() as _,
							(&LU.col_stride()) as *const _ as *const _,
							row_fwd.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C32 {
						la::cgetrf_(
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							LU.as_ptr_mut() as _,
							(&LU.col_stride()) as *const _ as *const _,
							row_fwd.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C64 {
						la::zgetrf_(
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							LU.as_ptr_mut() as _,
							(&LU.col_stride()) as *const _ as *const _,
							row_fwd.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					}
				}
			} else if Lib::NALGEBRA {
				unsafe {
					na::DMatrixView::from_data(na::ViewStorage::from_raw_parts(
						A.as_ptr(),
						(na::Dyn(n), na::Dyn(n)),
						(na::Const::<1>, na::Dyn(A.col_stride() as usize)),
					))
					.clone_owned()
					.lu();
				};
			}
		})
	}
}

fn full_piv_lu<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && Lib::NALGEBRA) || (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis)))) {
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let mut LU = Mat::zeros(m, n);
	let row_fwd = &mut *avec![0usize; m];
	let row_bwd = &mut *avec![0usize; m];
	let col_fwd = &mut *avec![0usize; n];
	let col_bwd = &mut *avec![0usize; n];

	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::lu::full_pivoting::factor::lu_in_place_scratch::<usize, T>(m, n, parallel, params));
	let stack = MemStack::new(stack);

	if !Lib::LAPACK || T::IS_NATIVE {
		bencher.bench(|| {
			if Lib::FAER {
				LU.copy_from(&A);
				linalg::lu::full_pivoting::factor::lu_in_place(LU.rb_mut(), row_fwd, row_bwd, col_fwd, col_bwd, parallel, stack, params);
			} else if Lib::LAPACK {
				LU.copy_from(&A);
				row_fwd.fill(0);
				col_fwd.fill(0);

				#[cfg(any(openblas, mkl, blis))]
				unsafe {
					if T::IS_NATIVE_F32 {
						sgetc2_(
							(&n) as *const _ as *const _,
							LU.as_ptr_mut() as _,
							(&LU.col_stride()) as *const _ as *const _,
							row_fwd.as_mut_ptr() as _,
							col_fwd.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_F64 {
						dgetc2_(
							(&n) as *const _ as *const _,
							LU.as_ptr_mut() as _,
							(&LU.col_stride()) as *const _ as *const _,
							row_fwd.as_mut_ptr() as _,
							col_fwd.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C32 {
						cgetc2_(
							(&n) as *const _ as *const _,
							LU.as_ptr_mut() as _,
							(&LU.col_stride()) as *const _ as *const _,
							row_fwd.as_mut_ptr() as _,
							col_fwd.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C64 {
						zgetc2_(
							(&n) as *const _ as *const _,
							LU.as_ptr_mut() as _,
							(&LU.col_stride()) as *const _ as *const _,
							row_fwd.as_mut_ptr() as _,
							col_fwd.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					}
				}
			} else if Lib::NALGEBRA {
				unsafe {
					na::DMatrixView::from_data(na::ViewStorage::from_raw_parts(
						A.as_ptr(),
						(na::Dyn(n), na::Dyn(n)),
						(na::Const::<1>, na::Dyn(A.col_stride() as usize)),
					))
					.clone_owned()
					.full_piv_lu();
				};
			}
		})
	}
}

fn svd<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && Lib::NALGEBRA) || (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis)))) {
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let mut U = Mat::zeros(m, m);
	let mut V = Mat::zeros(n, n);
	let mut S = Diag::zeros(Ord::min(m, n));
	#[cfg(any(openblas, mkl, blis))]
	let mut clone = A.cloned();

	#[cfg(any(openblas, mkl, blis))]
	let rwork = &mut *avec![zero::<T>(); m * n * 10];
	#[cfg(any(openblas, mkl, blis))]
	let iwork = &mut *avec![0usize; Ord::min(m, n) * 8];

	#[cfg(any(openblas, mkl, blis))]
	let lwork = unsafe {
		if T::IS_NATIVE_F32 {
			let mut work = core::mem::zeroed();
			la::sgesdd_(
				&(b'A' as _),
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				(&U.col_stride()) as *const _ as *const _,
				V.as_ptr_mut() as _,
				(&V.col_stride()) as *const _ as *const _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				iwork.as_mut_ptr() as _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			work as usize
		} else if T::IS_NATIVE_F64 {
			let mut work = core::mem::zeroed();
			la::dgesdd_(
				&(b'A' as _),
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				(&U.col_stride()) as *const _ as *const _,
				V.as_ptr_mut() as _,
				(&V.col_stride()) as *const _ as *const _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				iwork.as_mut_ptr() as _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			work as usize
		} else if T::IS_NATIVE_C32 {
			let mut work = core::mem::zeroed();
			la::cgesdd_(
				&(b'A' as _),
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				(&U.col_stride()) as *const _ as *const _,
				V.as_ptr_mut() as _,
				(&V.col_stride()) as *const _ as *const _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				rwork.as_mut_ptr() as _,
				iwork.as_mut_ptr() as _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			work.re as usize
		} else if T::IS_NATIVE_C64 {
			let mut work = core::mem::zeroed();
			la::zgesdd_(
				&(b'A' as _),
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				(&U.col_stride()) as *const _ as *const _,
				V.as_ptr_mut() as _,
				(&V.col_stride()) as *const _ as *const _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				rwork.as_mut_ptr() as _,
				iwork.as_mut_ptr() as _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			work.re as usize
		} else {
			0
		}
	};
	#[cfg(any(openblas, mkl, blis))]
	let work = &mut *avec![zero::<T>(); lwork];

	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::svd::svd_scratch::<T>(
		m,
		n,
		linalg::svd::ComputeSvdVectors::Full,
		linalg::svd::ComputeSvdVectors::Full,
		parallel,
		params,
	));
	let stack = MemStack::new(stack);

	if !Lib::LAPACK || T::IS_NATIVE {
		bencher.bench(|| {
			if Lib::FAER {
				linalg::svd::svd(A.rb(), S.rb_mut(), Some(U.rb_mut()), Some(V.rb_mut()), parallel, stack, params).unwrap();
			} else if Lib::LAPACK {
				#[cfg(any(openblas, mkl, blis))]
				unsafe {
					clone.copy_from(&A);
					if T::IS_NATIVE_F32 {
						la::sgesdd_(
							&(b'A' as _),
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							(&U.col_stride()) as *const _ as *const _,
							V.as_ptr_mut() as _,
							(&V.col_stride()) as *const _ as *const _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							iwork.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_F64 {
						la::dgesdd_(
							&(b'A' as _),
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							(&U.col_stride()) as *const _ as *const _,
							V.as_ptr_mut() as _,
							(&V.col_stride()) as *const _ as *const _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							iwork.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C32 {
						la::cgesdd_(
							&(b'A' as _),
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							(&U.col_stride()) as *const _ as *const _,
							V.as_ptr_mut() as _,
							(&V.col_stride()) as *const _ as *const _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							rwork.as_mut_ptr() as _,
							iwork.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C64 {
						la::zgesdd_(
							&(b'A' as _),
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							(&U.col_stride()) as *const _ as *const _,
							V.as_ptr_mut() as _,
							(&V.col_stride()) as *const _ as *const _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							rwork.as_mut_ptr() as _,
							iwork.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					}
				};
			} else if Lib::NALGEBRA {
				unsafe {
					na::DMatrixView::from_data(na::ViewStorage::from_raw_parts(
						A.as_ptr(),
						(na::Dyn(n), na::Dyn(n)),
						(na::Const::<1>, na::Dyn(A.col_stride() as usize)),
					))
					.clone_owned()
					.svd(true, true);
				};
			}
		});
	}
}

fn self_adjoint_evd<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && Lib::NALGEBRA) || (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis))))
		// parallel mkl sometimes segfaults here ¯\_(ツ)_/¯
		|| (Lib::LAPACK && Thd::PAR && cfg!(mkl))
	{
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	assert!(m == n);

	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A + A.adjoint();

	let mut U = Mat::zeros(m, m);
	let mut S = Diag::zeros(Ord::min(m, n));
	#[cfg(any(openblas, mkl, blis))]
	let mut clone = A.cloned();

	#[cfg(any(openblas, mkl, blis))]
	let (lwork, lrwork, liwork) = unsafe {
		clone.copy_from(&A);
		if T::IS_NATIVE_F32 {
			let mut work = core::mem::zeroed();
			let mut iwork = 0usize;
			la::ssyevd_(
				&(b'V' as _),
				&(b'L' as _),
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				(&mut iwork) as *mut _ as *mut _,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			(work as usize, 0, iwork as usize)
		} else if T::IS_NATIVE_F64 {
			let mut work = core::mem::zeroed();
			let mut iwork = 0usize;
			la::dsyevd_(
				&(b'V' as _),
				&(b'L' as _),
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				(&mut iwork) as *mut _ as *mut _,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			(work as usize, 0, iwork as usize)
		} else if T::IS_NATIVE_C32 {
			let mut work = core::mem::zeroed();
			let mut rwork = core::mem::zeroed();
			let mut iwork = 0usize;
			la::cheevd_(
				&(b'V' as _),
				&(b'L' as _),
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				&mut rwork,
				(&-1isize) as *const _ as *const _,
				(&mut iwork) as *mut _ as *mut _,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			(work.re as usize, 2 * rwork as usize, iwork as usize)
		} else if T::IS_NATIVE_C64 {
			let mut work = core::mem::zeroed();
			let mut rwork = core::mem::zeroed();
			let mut iwork = 0usize;
			la::zheevd_(
				&(b'V' as _),
				&(b'L' as _),
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				&mut rwork,
				(&-1isize) as *const _ as *const _,
				(&mut iwork) as *mut _ as *mut _,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			(work.re as usize, rwork as usize, iwork as usize)
		} else {
			(0, 0, 0)
		}
	};
	#[cfg(any(openblas, mkl, blis))]
	let work = &mut *avec![zero::<T>(); lwork];
	#[cfg(any(openblas, mkl, blis))]
	let rwork = &mut *avec![zero::<T>(); lrwork];
	#[cfg(any(openblas, mkl, blis))]
	let iwork = &mut *avec![0usize; liwork];

	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::evd::self_adjoint_evd_scratch::<T>(
		m,
		linalg::evd::ComputeEigenvectors::Yes,
		parallel,
		params,
	));
	let stack = MemStack::new(stack);

	if !Lib::LAPACK || T::IS_NATIVE {
		bencher.bench(|| {
			if Lib::FAER {
				linalg::evd::self_adjoint_evd(A.rb(), S.rb_mut(), Some(U.rb_mut()), parallel, stack, params).unwrap();
			} else if Lib::LAPACK {
				#[cfg(any(openblas, mkl, blis))]
				unsafe {
					clone.copy_from(&A);
					if T::IS_NATIVE_F32 {
						la::ssyevd_(
							&(b'V' as _),
							&(b'L' as _),
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							iwork.as_mut_ptr() as _,
							(&liwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_F64 {
						la::dsyevd_(
							&(b'V' as _),
							&(b'L' as _),
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							iwork.as_mut_ptr() as _,
							(&liwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C32 {
						la::cheevd_(
							&(b'V' as _),
							&(b'L' as _),
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							rwork.as_mut_ptr() as _,
							(&lrwork) as *const _ as *const _,
							iwork.as_mut_ptr() as _,
							(&liwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C64 {
						la::zheevd_(
							&(b'V' as _),
							&(b'L' as _),
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							rwork.as_mut_ptr() as _,
							(&lrwork) as *const _ as *const _,
							iwork.as_mut_ptr() as _,
							(&liwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					}
				}
			} else if Lib::NALGEBRA {
				unsafe {
					na::DMatrixView::from_data(na::ViewStorage::from_raw_parts(
						A.as_ptr(),
						(na::Dyn(n), na::Dyn(n)),
						(na::Const::<1>, na::Dyn(A.col_stride() as usize)),
					))
					.clone_owned()
					.symmetric_eigen();
				};
			}
		});
	}
}

fn evd<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && Lib::NALGEBRA) || (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis)))) {
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let mut U = Mat::<T>::zeros(m, m);
	let mut S = Diag::<T>::zeros(Ord::min(m, n));
	let mut S_im = Diag::<T>::zeros(Ord::min(m, n));
	#[cfg(any(openblas, mkl, blis))]
	let mut clone = A.cloned();

	#[cfg(any(openblas, mkl, blis))]
	let rwork = &mut *avec![zero::<T>(); 2 * n];

	#[cfg(any(openblas, mkl, blis))]
	let lwork = unsafe {
		if T::IS_NATIVE_F32 {
			let mut work = core::mem::zeroed();
			la::sgeev_(
				&(b'V' as _),
				&(b'N' as _),
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				S_im.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				(&U.col_stride()) as *const _ as *const _,
				null_mut(),
				(&1usize) as *const _ as *const _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			work as usize
		} else if T::IS_NATIVE_F64 {
			let mut work = core::mem::zeroed();
			la::dgeev_(
				&(b'V' as _),
				&(b'N' as _),
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				S_im.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				(&U.col_stride()) as *const _ as *const _,
				null_mut(),
				(&1usize) as *const _ as *const _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			work as usize
		} else if T::IS_NATIVE_C32 {
			let mut work = core::mem::zeroed();
			la::cgeev_(
				&(b'V' as _),
				&(b'N' as _),
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				(&U.col_stride()) as *const _ as *const _,
				null_mut(),
				(&1usize) as *const _ as *const _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				rwork.as_mut_ptr() as _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			work.re as usize
		} else if T::IS_NATIVE_C64 {
			let mut work = core::mem::zeroed();
			la::zgeev_(
				&(b'V' as _),
				&(b'N' as _),
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				(&U.col_stride()) as *const _ as *const _,
				null_mut(),
				(&1usize) as *const _ as *const _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				rwork.as_mut_ptr() as _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			work.re as usize
		} else {
			0
		}
	};

	#[cfg(any(openblas, mkl, blis))]
	let work = &mut *avec![zero::<T>(); lwork];

	let params = <linalg::evd::EvdParams as Auto<T>>::auto();
	let stack = &mut MemBuffer::new(linalg::evd::evd_scratch::<T>(
		n,
		linalg::evd::ComputeEigenvectors::Yes,
		linalg::evd::ComputeEigenvectors::Yes,
		parallel,
		params.into(),
	));
	let stack = MemStack::new(stack);

	if Lib::FAER || (Lib::LAPACK && T::IS_NATIVE) {
		bencher.bench(|| {
			if Lib::FAER {
				use core::mem::transmute;
				// SAFETY: dont worry im a pro
				unsafe {
					if T::IS_REAL {
						linalg::evd::evd_real::<T::Real>(
							transmute(A.rb()),
							transmute(S.rb_mut()),
							transmute(S_im.rb_mut()),
							Some(transmute(U.rb_mut())),
							None,
							parallel,
							stack,
							params.into(),
						)
						.unwrap();
					} else {
						linalg::evd::evd_cplx::<T::Real>(
							transmute(A.rb()),
							transmute(S.rb_mut()),
							Some(transmute(U.rb_mut())),
							None,
							parallel,
							stack,
							params.into(),
						)
						.unwrap();
					}
				}
			} else if Lib::LAPACK {
				#[cfg(any(openblas, mkl, blis))]
				unsafe {
					clone.copy_from(&A);
					if T::IS_NATIVE_F32 {
						la::sgeev_(
							&(b'V' as _),
							&(b'N' as _),
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							S_im.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							(&U.col_stride()) as *const _ as *const _,
							null_mut(),
							(&1usize) as *const _ as *const _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_F64 {
						la::dgeev_(
							&(b'V' as _),
							&(b'N' as _),
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							S_im.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							(&U.col_stride()) as *const _ as *const _,
							null_mut(),
							(&1usize) as *const _ as *const _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C32 {
						la::cgeev_(
							&(b'V' as _),
							&(b'N' as _),
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							(&U.col_stride()) as *const _ as *const _,
							null_mut(),
							(&1usize) as *const _ as *const _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							rwork.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C64 {
						la::zgeev_(
							&(b'V' as _),
							&(b'N' as _),
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							(&U.col_stride()) as *const _ as *const _,
							null_mut(),
							(&1usize) as *const _ as *const _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							rwork.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					};
				}
			}
		});
	}
}

fn main() -> eyre::Result<()> {
	let config = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/bench.toml"))?
		.parse::<Table>()
		.unwrap();

	let timings_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../target/timings.json");
	let mut timings = serde_json::de::from_str::<BenchResult>(&*std::fs::read_to_string(timings_path).unwrap_or(String::new()))
		.unwrap_or(BenchResult { groups: HashMap::new() });

	let mut parallel = vec![];

	if config["par"]["seq"].as_bool().unwrap() {
		parallel.push(Par::Seq);
	}
	if config["par"]["rayon"].as_bool().unwrap() {
		parallel.push(Par::rayon(0));
	}

	let shapes = |name: &str| {
		config[name]["shapes"]
			.as_array()
			.unwrap()
			.iter()
			.map(|i| PlotArg(i.as_integer().unwrap() as usize))
			.collect::<Vec<_>>()
	};
	let bench_config = Config::from_args()?;

	macro_rules! register {
		($T: ty) => {{
			type T = $T;

			for &parallel in &parallel {
				let bench = Bench::new(&bench_config);

				macro_rules! register_one {
					($name: ident, $config: expr) => {
						match parallel {
							Par::Seq => bench.register_many(
								std::stringify!($name),
								{
									let list = diol::variadics::Nil;
									#[cfg(any(openblas, mkl, blis))]
									let list = diol::variadics::Cons {
										head: $name::<T, self::lapack, self::seq>,
										tail: list,
									};
									#[cfg(nalgebra)]
									let list = diol::variadics::Cons {
										head: $name::<T, self::nalgebra, self::seq>,
										tail: list,
									};
									#[cfg(faer)]
									let list = diol::variadics::Cons {
										head: $name::<T, self::faer, self::seq>,
										tail: list,
									};

									list
								},
								shapes($config),
							),

							Par::Rayon(_) => bench.register_many(
								std::stringify!($name),
								{
									let list = diol::variadics::Nil;
									#[cfg(any(openblas, mkl, blis))]
									let list = diol::variadics::Cons {
										head: $name::<T, self::lapack, self::par>,
										tail: list,
									};
									#[cfg(nalgebra)]
									let list = diol::variadics::Cons {
										head: $name::<T, self::nalgebra, self::par>,
										tail: list,
									};
									#[cfg(faer)]
									let list = diol::variadics::Cons {
										head: $name::<T, self::faer, self::par>,
										tail: list,
									};

									list
								},
								shapes($config),
							),
						}
					};
				}

				register_one!(llt, "block_decomp");
				register_one!(ldlt, "block_decomp");
				register_one!(lblt, "block_decomp");
				register_one!(lblt_diag, "block_decomp");
				register_one!(lblt_rook, "block_decomp");
				register_one!(lblt_rook_diag, "block_decomp");
				register_one!(lblt_full, "decomp");

				register_one!(partial_piv_lu, "block_decomp");
				register_one!(full_piv_lu, "decomp");

				register_one!(qr, "block_decomp");
				register_one!(col_piv_qr, "decomp");

				register_one!(svd, "svd");
				register_one!(self_adjoint_evd, "svd");
				register_one!(evd, "evd");

				let result = bench.run()?;

				fn get_data(f: &str) -> (&str, &str, &str, &str) {
					let open = f.find('<').unwrap();
					let close = f.rfind('>').unwrap();
					let name = &f[..open];
					let args = &f[open + 1..close];
					let mut args = args.split(',');
					let ty = args.next().unwrap().trim();
					let backend = args.next().unwrap().trim();
					let thd = args.next().unwrap().trim();
					(name, ty, backend, thd)
				}

				timings = timings.combine(&result);
			}
		}};
	}

	spindle::with_lock(rayon::current_num_threads(), || -> eyre::Result<()> {
		register!(f32);
		register!(c32);
		register!(f64);
		register!(c64);
		Ok(())
	})?;

	std::fs::write(timings_path, serde_json::to_string(&timings).unwrap())?;

	Ok(())
}
