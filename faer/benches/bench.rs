#![allow(non_snake_case)]

use diol::prelude::*;
use faer::prelude::*;
use faer::stats::prelude::*;
use reborrow::*;

fn bench_new(bencher: Bencher, n: usize) {
	let rng = &mut StdRng::seed_from_u64(0);

	let A = CwiseMatDistribution {
		nrows: n,
		ncols: n,
		dist: StandardNormal,
	}
	.rand::<Mat<f64>>(rng);

	let ref A = &A * &A.adjoint();

	let mut LD = A.clone();
	let mut LD = LD.as_mut();

	bencher.bench(|| {
		LD.copy_from(A);

		_ = faer::linalg::cholesky::ldlt::factor::cholesky_in_place(
			LD.rb_mut(),
			Default::default(),
			Par::Seq,
			dyn_stack::DynStack::new(&mut []),
			Default::default(),
		);
	});
}

fn bench_copy(bencher: Bencher, n: usize) {
	let rng = &mut StdRng::seed_from_u64(0);

	let a = CwiseMatDistribution {
		nrows: n,
		ncols: n,
		dist: StandardNormal,
	}
	.rand::<Mat<f64>>(rng);
	let a = &a * &a.transpose();

	let mut l = a.clone();
	let mut l = l.try_as_col_major_mut().unwrap();

	bencher.bench(|| {
		l.copy_from(a.as_ref());
	});
}
fn main() -> std::io::Result<()> {
	let mut bench = Bench::new(BenchConfig::from_args()?);

	bench.register_many(list![bench_new, bench_copy], [4, 8, 12, 16, 32, 64, 128, 256]);
	bench.run()?;

	Ok(())
}
