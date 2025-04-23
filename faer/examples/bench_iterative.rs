#![allow(non_snake_case)]

use diol::prelude::*;
use dyn_stack::{MemBuffer, MemStack};
use faer::prelude::*;
use faer::stats::prelude::*;

fn eigen(bencher: Bencher, PlotArg(n): PlotArg) {
	let rng = &mut StdRng::seed_from_u64(0);

	let A = CwiseMatDistribution {
		nrows: n,
		ncols: n,
		dist: StandardNormal,
	}
	.rand::<Mat<f64>>(rng);

	let v0 = CwiseColDistribution {
		nrows: n,
		dist: StandardNormal,
	}
	.rand::<Col<f64>>(rng);

	let A = A.as_ref();
	let v0 = v0.as_ref();

	let n_eigval = 5;

	let mut V = Mat::zeros(n, n_eigval);
	let mut w = vec![c64::ZERO; n_eigval];

	let mut mem = MemBuffer::new(faer::matrix_free::eigen::partial_eigen_scratch(&A, n_eigval, Par::Seq, default()));

	bencher.bench(|| {
		faer::matrix_free::eigen::partial_eigen(
			V.rb_mut(),
			&mut w,
			&A,
			v0,
			f64::EPSILON * 128.0,
			Par::Seq,
			MemStack::new(&mut mem),
			default(),
		)
	});
}

fn main() -> eyre::Result<()> {
	let bench = Bench::from_args()?;

	bench.register_many("eigensolver", list![eigen], [512, 1024, 2048].map(PlotArg));

	bench.run()?;

	Ok(())
}
