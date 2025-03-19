#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use std::ffi::c_void;
use std::ptr::null_mut;

use ::faer::diag::Diag;
use ::faer::linalg::cholesky::bunch_kaufman::factor::BunchKaufmanParams;
use ::faer::prelude::*;
use ::faer::stats::prelude::*;
use ::faer::{Auto, linalg};
use faer_traits::math_utils::*;

use {::lapack_sys as la, ::nalgebra as na};

use diol::prelude::*;
use dyn_stack::{MemBuffer, MemStack};
use equator::assert;
use reborrow::*;
use toml::Table;

extern crate openblas_src;
extern crate openmp_sys;

unsafe extern "C" {
	fn openblas_set_num_threads(num: i32) -> c_void;
	fn goto_set_num_threads(num: i32) -> c_void;
	fn omp_set_num_threads(num: i32) -> c_void;
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

struct faer;
struct lapack;
struct nalgebra;

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

fn llt<T: Scalar, Lib: self::Lib>(bencher: Bencher, (m, n, par): (usize, usize, Par)) {
	match par {
		Par::Seq => unsafe {
			openblas_set_num_threads(1);
			goto_set_num_threads(1);
			omp_set_num_threads(1);
		},
		Par::Rayon(nthreads) => unsafe {
			openblas_set_num_threads(nthreads.get() as _);
			goto_set_num_threads(nthreads.get() as _);
			omp_set_num_threads(nthreads.get() as _);
		},
	};
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A * A.adjoint() + Scale(from_f64::<T>(m as f64)) * Mat::<T>::identity(n, n);
	let mut L = Mat::zeros(n, n);
	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::cholesky::llt::factor::cholesky_in_place_scratch::<T>(n, par, params));
	let stack = MemStack::new(stack);

	if !Lib::LAPACK || T::IS_NATIVE {
		bencher.bench(|| {
			if Lib::FAER {
				L.copy_from_triangular_lower(&A);
				linalg::cholesky::llt::factor::cholesky_in_place(L.rb_mut(), Default::default(), par, stack, params).unwrap();
			} else if Lib::LAPACK {
				L.copy_from_triangular_lower(&A);
				unsafe {
					if T::IS_NATIVE_F32 {
						la::spotrf_(&(b'L' as i8), &(n as _), L.as_ptr_mut() as _, &(L.col_stride() as _), &mut 0);
					} else if T::IS_NATIVE_F64 {
						la::dpotrf_(&(b'L' as i8), &(n as _), L.as_ptr_mut() as _, &(L.col_stride() as _), &mut 0);
					} else if T::IS_NATIVE_C32 {
						la::cpotrf_(&(b'L' as i8), &(n as _), L.as_ptr_mut() as _, &(L.col_stride() as _), &mut 0);
					} else if T::IS_NATIVE_C64 {
						la::zpotrf_(&(b'L' as _), &(n as _), L.as_ptr_mut() as _, &(L.col_stride() as _), &mut 0);
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

fn ldlt<T: Scalar, Lib: self::Lib>(bencher: Bencher, (m, n, par): (usize, usize, Par)) {
	match par {
		Par::Seq => unsafe {
			openblas_set_num_threads(1);
			goto_set_num_threads(1);
			omp_set_num_threads(1);
		},
		Par::Rayon(nthreads) => unsafe {
			openblas_set_num_threads(nthreads.get() as _);
			goto_set_num_threads(nthreads.get() as _);
			omp_set_num_threads(nthreads.get() as _);
		},
	};
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A * A.adjoint() + Scale(from_f64::<T>(m as f64)) * Mat::<T>::identity(n, n);
	let mut L = Mat::zeros(n, n);
	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::cholesky::ldlt::factor::cholesky_in_place_scratch::<T>(n, par, params));
	let stack = MemStack::new(stack);

	if Lib::FAER {
		bencher.bench(|| {
			L.copy_from_triangular_lower(&A);
			linalg::cholesky::ldlt::factor::cholesky_in_place(L.rb_mut(), Default::default(), par, stack, params).unwrap();
		})
	}
}

fn lblt<T: Scalar, Lib: self::Lib>(bencher: Bencher, (m, n, par): (usize, usize, Par)) {
	match par {
		Par::Seq => unsafe {
			openblas_set_num_threads(1);
			goto_set_num_threads(1);
			omp_set_num_threads(1);
		},
		Par::Rayon(nthreads) => unsafe {
			openblas_set_num_threads(nthreads.get() as _);
			goto_set_num_threads(nthreads.get() as _);
			omp_set_num_threads(nthreads.get() as _);
		},
	};
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A + A.adjoint();
	let mut L = Mat::zeros(n, n);
	let mut subdiag = Diag::zeros(n);
	let fwd = &mut *vec![0usize; n];
	let bwd = &mut *vec![0usize; n];

	let params = BunchKaufmanParams {
		pivoting: linalg::cholesky::bunch_kaufman::factor::PivotingStrategy::Partial,
		..Auto::<T>::auto()
	}
	.into();

	let lwork = unsafe {
		if T::IS_NATIVE_F32 {
			let mut lwork = 0.0;
			la::ssytrf_(
				&(b'L' as _),
				&(n as _),
				L.as_ptr_mut() as _,
				&(L.col_stride() as _),
				fwd.as_mut_ptr() as _,
				&mut lwork,
				&-1,
				&mut 0,
			);
			lwork as usize
		} else if T::IS_NATIVE_F64 {
			let mut lwork = 0.0;
			la::dsytrf_(
				&(b'L' as _),
				&(n as _),
				L.as_ptr_mut() as _,
				&(L.col_stride() as _),
				fwd.as_mut_ptr() as _,
				&mut lwork,
				&-1,
				&mut 0,
			);
			lwork as usize
		} else if T::IS_NATIVE_C32 {
			let mut lwork = core::mem::zeroed();
			la::csytrf_(
				&(b'L' as _),
				&(n as _),
				L.as_ptr_mut() as _,
				&(L.col_stride() as _),
				fwd.as_mut_ptr() as _,
				&mut lwork,
				&-1,
				&mut 0,
			);
			lwork.re as usize
		} else if T::IS_NATIVE_C64 {
			let mut lwork = core::mem::zeroed();
			la::zsytrf_(
				&(b'L' as _),
				&(n as _),
				L.as_ptr_mut() as _,
				&(L.col_stride() as _),
				fwd.as_mut_ptr() as _,
				&mut lwork,
				&-1,
				&mut 0,
			);
			lwork.re as usize
		} else {
			0
		}
	};

	let work = &mut *vec![zero::<T>(); lwork];

	let stack = &mut MemBuffer::new(linalg::cholesky::bunch_kaufman::factor::cholesky_in_place_scratch::<usize, T>(
		n, par, params,
	));
	let stack = MemStack::new(stack);

	if Lib::FAER || (Lib::LAPACK && T::IS_NATIVE) {
		bencher.bench(|| {
			L.copy_from_triangular_lower(&A);
			if Lib::FAER {
				linalg::cholesky::bunch_kaufman::factor::cholesky_in_place(
					L.rb_mut(),
					subdiag.rb_mut(),
					Default::default(),
					fwd,
					bwd,
					par,
					stack,
					params,
				);
			} else if Lib::LAPACK {
				unsafe {
					fwd.fill(0);

					if T::IS_NATIVE_F32 {
						la::ssytrf_(
							&(b'L' as _),
							&(n as _),
							L.as_ptr_mut() as _,
							&(L.col_stride() as _),
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							&(lwork as _),
							&mut 0,
						);
					} else if T::IS_NATIVE_F64 {
						la::dsytrf_(
							&(b'L' as _),
							&(n as _),
							L.as_ptr_mut() as _,
							&(L.col_stride() as _),
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							&(lwork as _),
							&mut 0,
						);
					} else if T::IS_NATIVE_C32 {
						la::csytrf_(
							&(b'L' as _),
							&(n as _),
							L.as_ptr_mut() as _,
							&(L.col_stride() as _),
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							&(lwork as _),
							&mut 0,
						);
					} else if T::IS_NATIVE_C64 {
						la::zsytrf_(
							&(b'L' as _),
							&(n as _),
							L.as_ptr_mut() as _,
							&(L.col_stride() as _),
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							&(lwork as _),
							&mut 0,
						);
					}
				}
			}
		})
	}
}

fn lblt_diag<T: Scalar, Lib: self::Lib>(bencher: Bencher, (m, n, par): (usize, usize, Par)) {
	match par {
		Par::Seq => unsafe {
			openblas_set_num_threads(1);
			goto_set_num_threads(1);
			omp_set_num_threads(1);
		},
		Par::Rayon(nthreads) => unsafe {
			openblas_set_num_threads(nthreads.get() as _);
			goto_set_num_threads(nthreads.get() as _);
			omp_set_num_threads(nthreads.get() as _);
		},
	};
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A + A.adjoint();
	let mut L = Mat::zeros(n, n);
	let mut subdiag = Diag::zeros(n);
	let fwd = &mut *vec![0usize; n];
	let bwd = &mut *vec![0usize; n];

	let params = BunchKaufmanParams {
		pivoting: linalg::cholesky::bunch_kaufman::factor::PivotingStrategy::PartialDiag,
		..Auto::<T>::auto()
	}
	.into();

	let stack = &mut MemBuffer::new(linalg::cholesky::bunch_kaufman::factor::cholesky_in_place_scratch::<usize, T>(
		n, par, params,
	));
	let stack = MemStack::new(stack);
	if Lib::FAER {
		bencher.bench(|| {
			L.copy_from_triangular_lower(&A);
			linalg::cholesky::bunch_kaufman::factor::cholesky_in_place(
				L.rb_mut(),
				subdiag.rb_mut(),
				Default::default(),
				fwd,
				bwd,
				par,
				stack,
				params,
			);
		})
	}
}

fn lblt_rook<T: Scalar, Lib: self::Lib>(bencher: Bencher, (m, n, par): (usize, usize, Par)) {
	match par {
		Par::Seq => unsafe {
			openblas_set_num_threads(1);
			goto_set_num_threads(1);
			omp_set_num_threads(1);
		},
		Par::Rayon(nthreads) => unsafe {
			openblas_set_num_threads(nthreads.get() as _);
			goto_set_num_threads(nthreads.get() as _);
			omp_set_num_threads(nthreads.get() as _);
		},
	};
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A + A.adjoint();
	let mut L = Mat::zeros(n, n);
	let mut subdiag = Diag::zeros(n);
	let fwd = &mut *vec![0usize; n];
	let bwd = &mut *vec![0usize; n];

	let params = BunchKaufmanParams {
		pivoting: linalg::cholesky::bunch_kaufman::factor::PivotingStrategy::Rook,
		..Auto::<T>::auto()
	}
	.into();

	let lwork = unsafe {
		if T::IS_NATIVE_F32 {
			let mut lwork = 0.0;
			la::ssytrf_(
				&(b'L' as _),
				&(n as _),
				L.as_ptr_mut() as _,
				&(L.col_stride() as _),
				fwd.as_mut_ptr() as _,
				&mut lwork,
				&-1,
				&mut 0,
			);
			lwork as usize
		} else if T::IS_NATIVE_F64 {
			let mut lwork = 0.0;
			la::dsytrf_(
				&(b'L' as _),
				&(n as _),
				L.as_ptr_mut() as _,
				&(L.col_stride() as _),
				fwd.as_mut_ptr() as _,
				&mut lwork,
				&-1,
				&mut 0,
			);
			lwork as usize
		} else if T::IS_NATIVE_C32 {
			let mut lwork = core::mem::zeroed();
			la::csytrf_(
				&(b'L' as _),
				&(n as _),
				L.as_ptr_mut() as _,
				&(L.col_stride() as _),
				fwd.as_mut_ptr() as _,
				&mut lwork,
				&-1,
				&mut 0,
			);
			lwork.re as usize
		} else if T::IS_NATIVE_C64 {
			let mut lwork = core::mem::zeroed();
			la::zsytrf_(
				&(b'L' as _),
				&(n as _),
				L.as_ptr_mut() as _,
				&(L.col_stride() as _),
				fwd.as_mut_ptr() as _,
				&mut lwork,
				&-1,
				&mut 0,
			);
			lwork.re as usize
		} else {
			0
		}
	};

	let work = &mut *vec![zero::<T>(); lwork];

	let stack = &mut MemBuffer::new(linalg::cholesky::bunch_kaufman::factor::cholesky_in_place_scratch::<usize, T>(
		n, par, params,
	));
	let stack = MemStack::new(stack);
	if Lib::FAER || (Lib::LAPACK && T::IS_NATIVE) {
		bencher.bench(|| {
			L.copy_from_triangular_lower(&A);
			if Lib::FAER {
				linalg::cholesky::bunch_kaufman::factor::cholesky_in_place(
					L.rb_mut(),
					subdiag.rb_mut(),
					Default::default(),
					fwd,
					bwd,
					par,
					stack,
					params,
				);
			} else if Lib::LAPACK {
				unsafe {
					fwd.fill(0);

					if T::IS_NATIVE_F32 {
						la::ssytrf_rook_(
							&(b'L' as _),
							&(n as _),
							L.as_ptr_mut() as _,
							&(L.col_stride() as _),
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							&(lwork as _),
							&mut 0,
						);
					} else if T::IS_NATIVE_F64 {
						la::dsytrf_rook_(
							&(b'L' as _),
							&(n as _),
							L.as_ptr_mut() as _,
							&(L.col_stride() as _),
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							&(lwork as _),
							&mut 0,
						);
					} else if T::IS_NATIVE_C32 {
						la::csytrf_rook_(
							&(b'L' as _),
							&(n as _),
							L.as_ptr_mut() as _,
							&(L.col_stride() as _),
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							&(lwork as _),
							&mut 0,
						);
					} else if T::IS_NATIVE_C64 {
						la::zsytrf_rook_(
							&(b'L' as _),
							&(n as _),
							L.as_ptr_mut() as _,
							&(L.col_stride() as _),
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							&(lwork as _),
							&mut 0,
						);
					}
				}
			}
		})
	}
}

fn lblt_rook_diag<T: Scalar, Lib: self::Lib>(bencher: Bencher, (m, n, par): (usize, usize, Par)) {
	match par {
		Par::Seq => unsafe {
			openblas_set_num_threads(1);
			goto_set_num_threads(1);
			omp_set_num_threads(1);
		},
		Par::Rayon(nthreads) => unsafe {
			openblas_set_num_threads(nthreads.get() as _);
			goto_set_num_threads(nthreads.get() as _);
			omp_set_num_threads(nthreads.get() as _);
		},
	};
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A + A.adjoint();
	let mut L = Mat::zeros(n, n);
	let mut subdiag = Diag::zeros(n);
	let fwd = &mut *vec![0usize; n];
	let bwd = &mut *vec![0usize; n];

	let params = BunchKaufmanParams {
		pivoting: linalg::cholesky::bunch_kaufman::factor::PivotingStrategy::Rook,
		..Auto::<T>::auto()
	}
	.into();

	let stack = &mut MemBuffer::new(linalg::cholesky::bunch_kaufman::factor::cholesky_in_place_scratch::<usize, T>(
		n, par, params,
	));
	let stack = MemStack::new(stack);

	if Lib::FAER {
		bencher.bench(|| {
			L.copy_from_triangular_lower(&A);
			linalg::cholesky::bunch_kaufman::factor::cholesky_in_place(
				L.rb_mut(),
				subdiag.rb_mut(),
				Default::default(),
				fwd,
				bwd,
				par,
				stack,
				params,
			);
		})
	}
}
fn lblt_full<T: Scalar, Lib: self::Lib>(bencher: Bencher, (m, n, par): (usize, usize, Par)) {
	match par {
		Par::Seq => unsafe {
			openblas_set_num_threads(1);
			goto_set_num_threads(1);
			omp_set_num_threads(1);
		},
		Par::Rayon(nthreads) => unsafe {
			openblas_set_num_threads(nthreads.get() as _);
			goto_set_num_threads(nthreads.get() as _);
			omp_set_num_threads(nthreads.get() as _);
		},
	};
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A + A.adjoint();
	let mut L = Mat::zeros(n, n);
	let mut subdiag = Diag::zeros(n);
	let fwd = &mut *vec![0usize; n];
	let bwd = &mut *vec![0usize; n];

	let params = BunchKaufmanParams {
		pivoting: linalg::cholesky::bunch_kaufman::factor::PivotingStrategy::Full,
		..Auto::<T>::auto()
	}
	.into();

	let stack = &mut MemBuffer::new(linalg::cholesky::bunch_kaufman::factor::cholesky_in_place_scratch::<usize, T>(
		n, par, params,
	));
	let stack = MemStack::new(stack);

	if Lib::FAER {
		bencher.bench(|| {
			L.copy_from_triangular_lower(&A);
			linalg::cholesky::bunch_kaufman::factor::cholesky_in_place(
				L.rb_mut(),
				subdiag.rb_mut(),
				Default::default(),
				fwd,
				bwd,
				par,
				stack,
				params,
			);
		})
	}
}

fn qr<T: Scalar, Lib: self::Lib>(bencher: Bencher, (m, n, par): (usize, usize, Par)) {
	match par {
		Par::Seq => unsafe {
			openblas_set_num_threads(1);
			goto_set_num_threads(1);
			omp_set_num_threads(1);
		},
		Par::Rayon(nthreads) => unsafe {
			openblas_set_num_threads(nthreads.get() as _);
			goto_set_num_threads(nthreads.get() as _);
			omp_set_num_threads(nthreads.get() as _);
		},
	};
	let blocksize = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(m, n);

	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);

	let mut Q = Mat::zeros(blocksize, Ord::min(m, n));
	let mut QR = Mat::zeros(m, n);

	let lwork = unsafe {
		if T::IS_NATIVE_F32 {
			let mut lwork = 0.0;
			la::sgeqrf_(
				&(m as _),
				&(n as _),
				QR.as_ptr_mut() as _,
				&(QR.col_stride() as _),
				Q.as_ptr_mut() as _,
				&mut lwork,
				&-1,
				&mut 0,
			);
			lwork as usize
		} else if T::IS_NATIVE_F64 {
			let mut lwork = 0.0;
			la::dgeqrf_(
				&(m as _),
				&(n as _),
				QR.as_ptr_mut() as _,
				&(QR.col_stride() as _),
				Q.as_ptr_mut() as _,
				&mut lwork,
				&-1,
				&mut 0,
			);
			lwork as usize
		} else if T::IS_NATIVE_C32 {
			let mut lwork = core::mem::zeroed();
			la::cgeqrf_(
				&(m as _),
				&(n as _),
				QR.as_ptr_mut() as _,
				&(QR.col_stride() as _),
				Q.as_ptr_mut() as _,
				&mut lwork,
				&-1,
				&mut 0,
			);
			lwork.re as usize
		} else if T::IS_NATIVE_C64 {
			let mut lwork = core::mem::zeroed();
			la::zgeqrf_(
				&(m as _),
				&(n as _),
				QR.as_ptr_mut() as _,
				&(QR.col_stride() as _),
				Q.as_ptr_mut() as _,
				&mut lwork,
				&-1,
				&mut 0,
			);
			lwork.re as usize
		} else {
			0
		}
	};

	let work = &mut *vec![zero::<T>(); lwork];

	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::qr::no_pivoting::factor::qr_in_place_scratch::<T>(m, n, blocksize, par, params));
	let stack = MemStack::new(stack);

	if !Lib::LAPACK || T::IS_NATIVE {
		bencher.bench(|| {
			if Lib::FAER {
				QR.copy_from(&A);
				linalg::qr::no_pivoting::factor::qr_in_place(QR.rb_mut(), Q.rb_mut(), par, stack, params);
			} else if Lib::LAPACK {
				QR.copy_from(&A);
				unsafe {
					if T::IS_NATIVE_F32 {
						la::sgeqrf_(
							&(m as _),
							&(n as _),
							QR.as_ptr_mut() as _,
							&(QR.col_stride() as _),
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							&(lwork as _),
							&mut 0,
						);
					} else if T::IS_NATIVE_F64 {
						la::dgeqrf_(
							&(m as _),
							&(n as _),
							QR.as_ptr_mut() as _,
							&(QR.col_stride() as _),
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							&(lwork as _),
							&mut 0,
						);
					} else if T::IS_NATIVE_C32 {
						la::cgeqrf_(
							&(m as _),
							&(n as _),
							QR.as_ptr_mut() as _,
							&(QR.col_stride() as _),
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							&(lwork as _),
							&mut 0,
						);
					} else if T::IS_NATIVE_C64 {
						la::zgeqrf_(
							&(m as _),
							&(n as _),
							QR.as_ptr_mut() as _,
							&(QR.col_stride() as _),
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							&(lwork as _),
							&mut 0,
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

fn col_piv_qr<T: Scalar, Lib: self::Lib>(bencher: Bencher, (m, n, par): (usize, usize, Par)) {
	match par {
		Par::Seq => unsafe {
			openblas_set_num_threads(1);
			goto_set_num_threads(1);
			omp_set_num_threads(1);
		},
		Par::Rayon(nthreads) => unsafe {
			openblas_set_num_threads(nthreads.get() as _);
			goto_set_num_threads(nthreads.get() as _);
			omp_set_num_threads(nthreads.get() as _);
		},
	};
	let blocksize = linalg::qr::col_pivoting::factor::recommended_blocksize::<T>(m, n);

	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);

	let mut Q = Mat::zeros(blocksize, Ord::min(m, n));
	let mut QR = Mat::zeros(m, n);

	let col_fwd = &mut *vec![0usize; n];
	let col_bwd = &mut *vec![0usize; n];

	let work = &mut *vec![zero::<T>(); 3 * n];
	let rwork = &mut *vec![zero::<T>(); 2 * n];

	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::qr::col_pivoting::factor::qr_in_place_scratch::<usize, T>(
		m, n, blocksize, par, params,
	));
	let stack = MemStack::new(stack);

	if !Lib::LAPACK || T::IS_NATIVE {
		bencher.bench(|| {
			if Lib::FAER {
				QR.copy_from(&A);
				linalg::qr::col_pivoting::factor::qr_in_place(QR.rb_mut(), Q.rb_mut(), col_fwd, col_bwd, par, stack, params);
			} else if Lib::LAPACK {
				QR.copy_from(&A);
				col_fwd.fill(0);
				unsafe {
					if T::IS_NATIVE_F32 {
						la::sgeqpf_(
							&mut (m as _),
							&mut (n as _),
							QR.as_ptr_mut() as _,
							&mut (QR.col_stride() as _),
							col_fwd.as_mut_ptr() as _,
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							&mut 0,
						);
					} else if T::IS_NATIVE_F64 {
						la::dgeqpf_(
							&mut (m as _),
							&mut (n as _),
							QR.as_ptr_mut() as _,
							&mut (QR.col_stride() as _),
							col_fwd.as_mut_ptr() as _,
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							&mut 0,
						);
					} else if T::IS_NATIVE_C32 {
						la::cgeqpf_(
							&mut (m as _),
							&mut (n as _),
							QR.as_ptr_mut() as _,
							&mut (QR.col_stride() as _),
							col_fwd.as_mut_ptr() as _,
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							rwork.as_mut_ptr() as _,
							&mut 0,
						);
					} else if T::IS_NATIVE_C64 {
						la::zgeqpf_(
							&mut (m as _),
							&mut (n as _),
							QR.as_ptr_mut() as _,
							&mut (QR.col_stride() as _),
							col_fwd.as_mut_ptr() as _,
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							rwork.as_mut_ptr() as _,
							&mut 0,
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

fn partial_piv_lu<T: Scalar, Lib: self::Lib>(bencher: Bencher, (m, n, par): (usize, usize, Par)) {
	match par {
		Par::Seq => unsafe {
			openblas_set_num_threads(1);
			goto_set_num_threads(1);
			omp_set_num_threads(1);
		},
		Par::Rayon(nthreads) => unsafe {
			openblas_set_num_threads(nthreads.get() as _);
			goto_set_num_threads(nthreads.get() as _);
			omp_set_num_threads(nthreads.get() as _);
		},
	};
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let mut LU = Mat::zeros(n, n);
	let row_fwd = &mut *vec![0usize; n];
	let row_bwd = &mut *vec![0usize; n];

	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, T>(n, n, par, params));
	let stack = MemStack::new(stack);

	if !Lib::LAPACK || T::IS_NATIVE {
		bencher.bench(|| {
			if Lib::FAER {
				LU.copy_from(&A);
				linalg::lu::partial_pivoting::factor::lu_in_place(LU.rb_mut(), row_fwd, row_bwd, par, stack, params);
			} else if Lib::LAPACK {
				LU.copy_from(&A);
				row_fwd.fill(0);

				unsafe {
					if T::IS_NATIVE_F32 {
						la::sgetrf_(
							&(m as _),
							&(n as _),
							LU.as_ptr_mut() as _,
							&(LU.col_stride() as _),
							row_fwd.as_mut_ptr() as _,
							&mut 0,
						);
					} else if T::IS_NATIVE_F64 {
						la::dgetrf_(
							&(m as _),
							&(n as _),
							LU.as_ptr_mut() as _,
							&(LU.col_stride() as _),
							row_fwd.as_mut_ptr() as _,
							&mut 0,
						);
					} else if T::IS_NATIVE_C32 {
						la::cgetrf_(
							&(m as _),
							&(n as _),
							LU.as_ptr_mut() as _,
							&(LU.col_stride() as _),
							row_fwd.as_mut_ptr() as _,
							&mut 0,
						);
					} else if T::IS_NATIVE_C64 {
						la::zgetrf_(
							&(m as _),
							&(n as _),
							LU.as_ptr_mut() as _,
							&(LU.col_stride() as _),
							row_fwd.as_mut_ptr() as _,
							&mut 0,
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

fn full_piv_lu<T: Scalar, Lib: self::Lib>(bencher: Bencher, (m, n, par): (usize, usize, Par)) {
	match par {
		Par::Seq => unsafe {
			openblas_set_num_threads(1);
			goto_set_num_threads(1);
			omp_set_num_threads(1);
		},
		Par::Rayon(nthreads) => unsafe {
			openblas_set_num_threads(nthreads.get() as _);
			goto_set_num_threads(nthreads.get() as _);
			omp_set_num_threads(nthreads.get() as _);
		},
	};
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let mut LU = Mat::zeros(m, n);
	let row_fwd = &mut *vec![0usize; m];
	let row_bwd = &mut *vec![0usize; m];
	let col_fwd = &mut *vec![0usize; n];
	let col_bwd = &mut *vec![0usize; n];

	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::lu::full_pivoting::factor::lu_in_place_scratch::<usize, T>(m, n, par, params));
	let stack = MemStack::new(stack);

	if !Lib::LAPACK {
		bencher.bench(|| {
			if Lib::FAER {
				LU.copy_from(&A);
				linalg::lu::full_pivoting::factor::lu_in_place(LU.rb_mut(), row_fwd, row_bwd, col_fwd, col_bwd, par, stack, params);
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

fn svd<T: Scalar, Lib: self::Lib>(bencher: Bencher, (m, n, par): (usize, usize, Par)) {
	match par {
		Par::Seq => unsafe {
			openblas_set_num_threads(1);
			goto_set_num_threads(1);
			omp_set_num_threads(1);
		},
		Par::Rayon(nthreads) => unsafe {
			openblas_set_num_threads(nthreads.get() as _);
			goto_set_num_threads(nthreads.get() as _);
			omp_set_num_threads(nthreads.get() as _);
		},
	};
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let mut U = Mat::zeros(m, m);
	let mut V = Mat::zeros(n, n);
	let mut S = Diag::zeros(Ord::min(m, n));
	let mut clone = A.cloned();

	let rwork = &mut *vec![zero::<T>(); m * n * 10];
	let iwork = &mut *vec![0; Ord::min(m, n) * 8];

	let lwork = unsafe {
		if T::IS_NATIVE_F32 {
			let mut work = core::mem::zeroed();
			la::sgesdd_(
				&(b'A' as _),
				&(m as _),
				&(n as _),
				clone.as_ptr_mut() as _,
				&(clone.col_stride() as _),
				S.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				&(U.col_stride() as _),
				V.as_ptr_mut() as _,
				&(V.col_stride() as _),
				&mut work,
				&-1,
				iwork.as_mut_ptr(),
				&mut 0,
			);
			work as usize
		} else if T::IS_NATIVE_F64 {
			let mut work = core::mem::zeroed();
			la::dgesdd_(
				&(b'A' as _),
				&(m as _),
				&(n as _),
				clone.as_ptr_mut() as _,
				&(clone.col_stride() as _),
				S.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				&(U.col_stride() as _),
				V.as_ptr_mut() as _,
				&(V.col_stride() as _),
				&mut work,
				&-1,
				iwork.as_mut_ptr(),
				&mut 0,
			);
			work as usize
		} else if T::IS_NATIVE_C32 {
			let mut work = core::mem::zeroed();
			la::cgesdd_(
				&(b'A' as _),
				&(m as _),
				&(n as _),
				clone.as_ptr_mut() as _,
				&(clone.col_stride() as _),
				S.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				&(U.col_stride() as _),
				V.as_ptr_mut() as _,
				&(V.col_stride() as _),
				&mut work,
				&-1,
				rwork.as_mut_ptr() as _,
				iwork.as_mut_ptr(),
				&mut 0,
			);
			work.re as usize
		} else if T::IS_NATIVE_C64 {
			let mut work = core::mem::zeroed();
			la::zgesdd_(
				&(b'A' as _),
				&(m as _),
				&(n as _),
				clone.as_ptr_mut() as _,
				&(clone.col_stride() as _),
				S.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				&(U.col_stride() as _),
				V.as_ptr_mut() as _,
				&(V.col_stride() as _),
				&mut work,
				&-1,
				rwork.as_mut_ptr() as _,
				iwork.as_mut_ptr(),
				&mut 0,
			);
			work.re as usize
		} else {
			0
		}
	};
	let work = &mut *vec![zero::<T>(); lwork];

	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::svd::svd_scratch::<T>(
		m,
		n,
		linalg::svd::ComputeSvdVectors::Full,
		linalg::svd::ComputeSvdVectors::Full,
		par,
		params,
	));
	let stack = MemStack::new(stack);

	if !Lib::LAPACK || T::IS_NATIVE {
		bencher.bench(|| {
			if Lib::FAER {
				linalg::svd::svd(A.rb(), S.rb_mut(), Some(U.rb_mut()), Some(V.rb_mut()), par, stack, params).unwrap();
			} else if Lib::LAPACK {
				unsafe {
					if T::IS_NATIVE_F32 {
						la::sgesdd_(
							&(b'A' as _),
							&(m as _),
							&(n as _),
							clone.as_ptr_mut() as _,
							&(clone.col_stride() as _),
							S.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							&(U.col_stride() as _),
							V.as_ptr_mut() as _,
							&(V.col_stride() as _),
							work.as_mut_ptr() as _,
							&(lwork as _),
							iwork.as_mut_ptr(),
							&mut 0,
						);
					} else if T::IS_NATIVE_F64 {
						la::dgesdd_(
							&(b'A' as _),
							&(m as _),
							&(n as _),
							clone.as_ptr_mut() as _,
							&(clone.col_stride() as _),
							S.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							&(U.col_stride() as _),
							V.as_ptr_mut() as _,
							&(V.col_stride() as _),
							work.as_mut_ptr() as _,
							&(lwork as _),
							iwork.as_mut_ptr(),
							&mut 0,
						);
					} else if T::IS_NATIVE_C32 {
						la::cgesdd_(
							&(b'A' as _),
							&(m as _),
							&(n as _),
							clone.as_ptr_mut() as _,
							&(clone.col_stride() as _),
							S.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							&(U.col_stride() as _),
							V.as_ptr_mut() as _,
							&(V.col_stride() as _),
							work.as_mut_ptr() as _,
							&(lwork as _),
							rwork.as_mut_ptr() as _,
							iwork.as_mut_ptr(),
							&mut 0,
						);
					} else if T::IS_NATIVE_C64 {
						la::zgesdd_(
							&(b'A' as _),
							&(m as _),
							&(n as _),
							clone.as_ptr_mut() as _,
							&(clone.col_stride() as _),
							S.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							&(U.col_stride() as _),
							V.as_ptr_mut() as _,
							&(V.col_stride() as _),
							work.as_mut_ptr() as _,
							&(lwork as _),
							rwork.as_mut_ptr() as _,
							iwork.as_mut_ptr(),
							&mut 0,
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

fn self_adjoint_evd<T: Scalar, Lib: self::Lib>(bencher: Bencher, (m, n, par): (usize, usize, Par)) {
	match par {
		Par::Seq => unsafe {
			openblas_set_num_threads(1);
			goto_set_num_threads(1);
			omp_set_num_threads(1);
		},
		Par::Rayon(nthreads) => unsafe {
			openblas_set_num_threads(nthreads.get() as _);
			goto_set_num_threads(nthreads.get() as _);
			omp_set_num_threads(nthreads.get() as _);
		},
	};
	assert!(m == n);

	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A + A.adjoint();

	let mut U = Mat::zeros(m, m);
	let mut S = Diag::zeros(Ord::min(m, n));
	let mut clone = A.cloned();

	let rwork = &mut *vec![zero::<T>(); 3 * n];

	let lwork = unsafe {
		if T::IS_NATIVE_F32 {
			let mut work = core::mem::zeroed();
			la::ssyev_(
				&(b'V' as _),
				&(b'L' as _),
				&(n as _),
				clone.as_ptr_mut() as _,
				&(clone.col_stride() as _),
				S.column_vector_mut().as_ptr_mut() as _,
				&mut work,
				&-1,
				&mut 0,
			);
			work as usize
		} else if T::IS_NATIVE_F64 {
			let mut work = core::mem::zeroed();
			la::dsyev_(
				&(b'V' as _),
				&(b'L' as _),
				&(n as _),
				clone.as_ptr_mut() as _,
				&(clone.col_stride() as _),
				S.column_vector_mut().as_ptr_mut() as _,
				&mut work,
				&-1,
				&mut 0,
			);
			work as usize
		} else if T::IS_NATIVE_C32 {
			let mut work = core::mem::zeroed();
			la::cheev_(
				&(b'V' as _),
				&(b'L' as _),
				&(n as _),
				clone.as_ptr_mut() as _,
				&(clone.col_stride() as _),
				S.column_vector_mut().as_ptr_mut() as _,
				&mut work,
				&-1,
				rwork.as_mut_ptr() as _,
				&mut 0,
			);
			work.re as usize
		} else if T::IS_NATIVE_C64 {
			let mut work = core::mem::zeroed();
			la::zheev_(
				&(b'V' as _),
				&(b'L' as _),
				&(n as _),
				clone.as_ptr_mut() as _,
				&(clone.col_stride() as _),
				S.column_vector_mut().as_ptr_mut() as _,
				&mut work,
				&-1,
				rwork.as_mut_ptr() as _,
				&mut 0,
			);
			work.re as usize
		} else {
			0
		}
	};

	let work = &mut *vec![zero::<T>(); lwork];

	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::evd::self_adjoint_evd_scratch::<T>(
		m,
		linalg::evd::ComputeEigenvectors::Yes,
		par,
		params,
	));
	let stack = MemStack::new(stack);

	if !Lib::LAPACK || T::IS_NATIVE {
		bencher.bench(|| {
			if Lib::FAER {
				linalg::evd::self_adjoint_evd(A.rb(), S.rb_mut(), Some(U.rb_mut()), par, stack, params).unwrap();
			} else if Lib::LAPACK {
				unsafe {
					if T::IS_NATIVE_F32 {
						la::ssyev_(
							&(b'V' as _),
							&(b'L' as _),
							&(n as _),
							clone.as_ptr_mut() as _,
							&(clone.col_stride() as _),
							S.column_vector_mut().as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							&(lwork as _),
							&mut 0,
						);
					} else if T::IS_NATIVE_F64 {
						la::dsyev_(
							&(b'V' as _),
							&(b'L' as _),
							&(n as _),
							clone.as_ptr_mut() as _,
							&(clone.col_stride() as _),
							S.column_vector_mut().as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							&(lwork as _),
							&mut 0,
						);
					} else if T::IS_NATIVE_C32 {
						la::cheev_(
							&(b'V' as _),
							&(b'L' as _),
							&(n as _),
							clone.as_ptr_mut() as _,
							&(clone.col_stride() as _),
							S.column_vector_mut().as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							&(lwork as _),
							rwork.as_mut_ptr() as _,
							&mut 0,
						);
					} else if T::IS_NATIVE_C64 {
						la::zheev_(
							&(b'V' as _),
							&(b'L' as _),
							&(n as _),
							clone.as_ptr_mut() as _,
							&(clone.col_stride() as _),
							S.column_vector_mut().as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							&(lwork as _),
							rwork.as_mut_ptr() as _,
							&mut 0,
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

fn evd<T: Scalar, Lib: self::Lib>(bencher: Bencher, (m, n, par): (usize, usize, Par)) {
	match par {
		Par::Seq => unsafe {
			openblas_set_num_threads(1);
			goto_set_num_threads(1);
			omp_set_num_threads(1);
		},
		Par::Rayon(nthreads) => unsafe {
			openblas_set_num_threads(nthreads.get() as _);
			goto_set_num_threads(nthreads.get() as _);
			omp_set_num_threads(nthreads.get() as _);
		},
	};
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let mut U = Mat::<T>::zeros(m, m);
	let mut S = Diag::<T>::zeros(Ord::min(m, n));
	let mut S_im = Diag::<T>::zeros(Ord::min(m, n));
	let mut clone = A.cloned();

	let rwork = &mut *vec![zero::<T>(); 2 * n];

	let lwork = unsafe {
		if T::IS_NATIVE_F32 {
			let mut work = core::mem::zeroed();
			la::sgeev_(
				&(b'V' as _),
				&(b'N' as _),
				&(n as _),
				clone.as_ptr_mut() as _,
				&(clone.col_stride() as _),
				S.column_vector_mut().as_ptr_mut() as _,
				S_im.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				&(U.col_stride() as _),
				null_mut(),
				&1,
				&mut work,
				&-1,
				&mut 0,
			);
			work as usize
		} else if T::IS_NATIVE_F64 {
			let mut work = core::mem::zeroed();
			la::dgeev_(
				&(b'V' as _),
				&(b'N' as _),
				&(n as _),
				clone.as_ptr_mut() as _,
				&(clone.col_stride() as _),
				S.column_vector_mut().as_ptr_mut() as _,
				S_im.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				&(U.col_stride() as _),
				null_mut(),
				&1,
				&mut work,
				&-1,
				&mut 0,
			);
			work as usize
		} else if T::IS_NATIVE_C32 {
			let mut work = core::mem::zeroed();
			la::cgeev_(
				&(b'V' as _),
				&(b'N' as _),
				&(n as _),
				clone.as_ptr_mut() as _,
				&(clone.col_stride() as _),
				S.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				&(U.col_stride() as _),
				null_mut(),
				&1,
				&mut work,
				&-1,
				rwork.as_mut_ptr() as _,
				&mut 0,
			);
			work.re as usize
		} else if T::IS_NATIVE_C64 {
			let mut work = core::mem::zeroed();
			la::zgeev_(
				&(b'V' as _),
				&(b'N' as _),
				&(n as _),
				clone.as_ptr_mut() as _,
				&(clone.col_stride() as _),
				S.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				&(U.col_stride() as _),
				null_mut(),
				&1,
				&mut work,
				&-1,
				rwork.as_mut_ptr() as _,
				&mut 0,
			);
			work.re as usize
		} else {
			0
		}
	};

	let work = &mut *vec![zero::<T>(); lwork];

	let params = <linalg::evd::EvdParams as Auto<T>>::auto();
	let stack = &mut MemBuffer::new(linalg::evd::evd_scratch::<T>(
		n,
		linalg::evd::ComputeEigenvectors::Yes,
		linalg::evd::ComputeEigenvectors::Yes,
		par,
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
							par,
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
							par,
							stack,
							params.into(),
						)
						.unwrap();
					}
				}
			} else if Lib::LAPACK {
				clone.copy_from(&A);
				unsafe {
					if T::IS_NATIVE_F32 {
						la::sgeev_(
							&(b'V' as _),
							&(b'N' as _),
							&(n as _),
							clone.as_ptr_mut() as _,
							&(clone.col_stride() as _),
							S.column_vector_mut().as_ptr_mut() as _,
							S_im.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							&(U.col_stride() as _),
							null_mut(),
							&1,
							work.as_mut_ptr() as _,
							&(lwork as _),
							&mut 0,
						);
					} else if T::IS_NATIVE_F64 {
						la::dgeev_(
							&(b'V' as _),
							&(b'N' as _),
							&(n as _),
							clone.as_ptr_mut() as _,
							&(clone.col_stride() as _),
							S.column_vector_mut().as_ptr_mut() as _,
							S_im.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							&(U.col_stride() as _),
							null_mut(),
							&1,
							work.as_mut_ptr() as _,
							&(lwork as _),
							&mut 0,
						);
					} else if T::IS_NATIVE_C32 {
						la::cgeev_(
							&(b'V' as _),
							&(b'N' as _),
							&(n as _),
							clone.as_ptr_mut() as _,
							&(clone.col_stride() as _),
							S.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							&(U.col_stride() as _),
							null_mut(),
							&1,
							work.as_mut_ptr() as _,
							&(lwork as _),
							rwork.as_mut_ptr() as _,
							&mut 0,
						);
					} else if T::IS_NATIVE_C64 {
						la::zgeev_(
							&(b'V' as _),
							&(b'N' as _),
							&(n as _),
							clone.as_ptr_mut() as _,
							&(clone.col_stride() as _),
							S.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							&(U.col_stride() as _),
							null_mut(),
							&1,
							work.as_mut_ptr() as _,
							&(lwork as _),
							rwork.as_mut_ptr() as _,
							&mut 0,
						);
					};
				}
			}
		});
	}
}

fn main() -> std::io::Result<()> {
	let config = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/bench.toml"))?
		.parse::<Table>()
		.unwrap();

	let mut par = vec![];

	if config["par"]["seq"].as_bool().unwrap() {
		par.push(Par::Seq);
	}
	if config["par"]["rayon"].as_bool().unwrap() {
		par.push(Par::rayon(0));
	}

	let mut bench = Bench::new(BenchConfig::from_args()?);

	let shapes = |name: &str, par: Par| {
		config[name]["shapes"]
			.as_array()
			.unwrap()
			.iter()
			.map(|i| i.as_array().unwrap())
			.map(|v| (v[0].as_integer().unwrap() as usize, v[1].as_integer().unwrap() as usize, par))
			.collect::<Vec<_>>()
	};

	macro_rules! register {
		($T: ty) => {{
			type T = $T;

			for &par in &par {
				macro_rules! register_one {
					($name: ident, $config: expr) => {
						bench.register_many(
							list![
								($name::<T, self::faer>),
								($name::<T, self::lapack>),
								($name::<T, self::nalgebra>)
							],
							shapes($config, par),
						);
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
				register_one!(full_piv_lu, "block_decomp");

				register_one!(qr, "block_decomp");
				register_one!(col_piv_qr, "decomp");

				register_one!(svd, "svd");
				register_one!(self_adjoint_evd, "evd");
				register_one!(evd, "evd");
			}
		}};
	}

	register!(f32);
	register!(c32);
	register!(f64);
	register!(c64);

	bench.run()?;

	Ok(())
}
