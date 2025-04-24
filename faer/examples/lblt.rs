#![allow(non_snake_case)]

use diol::prelude::*;
use dyn_stack::{MemBuffer, MemStack};
use faer::diag::Diag;
use faer::linalg::cholesky::lblt;
use faer::prelude::*;
use faer::reborrow::ReborrowMut;
use faer::stats::prelude::*;

#[cfg(any(openblas, mkl, blis))]
use lapack_sys as la;

#[cfg(openblas)]
extern crate openblas_src;

#[cfg(mkl)]
extern crate intel_mkl_src;

#[cfg(blis)]
extern crate blis_src;

#[cfg(any(openblas, mkl, blis))]
fn lapack(bencher: Bencher, PlotArg(n): PlotArg) {
	use aligned_vec::avec;

	let rng = &mut StdRng::seed_from_u64(0);

	let A = CwiseMatDistribution {
		nrows: n,
		ncols: n,
		dist: StandardNormal,
	}
	.rand::<Mat<f64>>(rng);

	let A = &A + A.adjoint();
	let mut lblt = A.clone();
	let perm = &mut *vec![0usize; n];
	let perm_inv = &mut *vec![0usize; n];

	let lwork = unsafe {
		let mut lwork = core::mem::zeroed();
		la::dsytrf_(
			&(b'L' as _),
			(&n) as *const _ as *const _,
			lblt.as_ptr_mut() as _,
			(&lblt.col_stride()) as *const _ as *const _,
			perm.as_mut_ptr() as _,
			&mut lwork,
			(&-1isize) as *const _ as *const _,
			(&mut 0usize) as *mut _ as *mut _,
		);
		lwork as usize
	};

	let work = &mut *avec![0.0f64; lwork];

	bencher.bench(|| unsafe {
		lblt.copy_from_triangular_lower(&A);
		perm.fill(0);
		la::dsytrf_(
			&(b'L' as _),
			(&n) as *const _ as *const _,
			lblt.as_ptr_mut() as _,
			(&lblt.col_stride()) as *const _ as *const _,
			perm.as_mut_ptr() as _,
			work.as_mut_ptr() as _,
			(&lwork) as *const _ as *const _,
			(&mut 0usize) as *mut _ as *mut _,
		);
	});
}

fn faer(bencher: Bencher, PlotArg(n): PlotArg) {
	let rng = &mut StdRng::seed_from_u64(0);

	let A = CwiseMatDistribution {
		nrows: n,
		ncols: n,
		dist: StandardNormal,
	}
	.rand::<Mat<f64>>(rng);

	let A = &A + A.adjoint();
	let mut lblt = A.clone();
	let mut subdiag = Diag::zeros(n);
	let perm = &mut *vec![0usize; n];
	let perm_inv = &mut *vec![0usize; n];

	let par = Par::Seq;
	let params = default();

	let scratch = lblt::factor::cholesky_in_place_scratch::<usize, f64>(n, par, params);
	let mut mem = MemBuffer::new(scratch);
	let stack = MemStack::new(&mut mem);

	bencher.bench(|| {
		lblt.copy_from_triangular_lower(&A);
		lblt::factor::cholesky_in_place(lblt.rb_mut(), subdiag.rb_mut(), perm, perm_inv, par, stack, params);
	});
}

fn main() -> eyre::Result<()> {
	let bench = Bench::from_args()?;

	bench.register_many(
		"lblt",
		{
			let f = list![faer];

			#[cfg(any(openblas, mkl, blis))]
			let f = diol::variadics::Cons { head: lapack, tail: f };

			f
		},
		[64, 96, 128, 192, 256, 1024, 2048, 4096, 8192].map(PlotArg),
	);

	bench.run()?;

	Ok(())
}
