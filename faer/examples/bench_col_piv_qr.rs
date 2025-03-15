#![allow(non_snake_case)]

use diol::prelude::*;
use faer::prelude::*;
use faer::stats::prelude::*;
use reborrow::*;

use faer::linalg::qr::col_pivoting::factor;

fn bench_old(bencher: Bencher, (m, n): (usize, usize)) {
	let blocksize = 1;
	let par = Par::Seq;

	let rng = &mut StdRng::seed_from_u64(0);
	let A = CwiseMatDistribution {
		nrows: m,
		ncols: n,
		dist: StandardNormal,
	}
	.rand::<Mat<f64>>(rng);

	let mut Q = Mat::zeros(blocksize, Ord::min(m, n));
	let mut QR = Mat::zeros(m, n);

	let col_perm = &mut *vec![0usize; n];
	let col_perm_inv = &mut *vec![0usize; n];
	let params = Default::default();
	let stack = &mut dyn_stack::MemBuffer::new(factor::qr_in_place_scratch::<usize, f64>(m, n, blocksize, par, params));
	let stack = dyn_stack::MemStack::new(stack);

	bencher.bench(|| {
		QR.copy_from(&A);
		factor::qr_in_place(QR.rb_mut(), Q.rb_mut(), col_perm, col_perm_inv, par, stack, params);
	});
}

fn main() -> std::io::Result<()> {
	let mut bench = Bench::new(BenchConfig::from_args()?);

	bench.register_many(list![bench_old], [128, 256, 1024, 512, 2048, 3072, 4096].map(|n| (n, n)));

	bench.run()?;

	Ok(())
}
