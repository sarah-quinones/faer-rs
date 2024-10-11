#![allow(non_snake_case)]

use diol::prelude::*;
use faer::{prelude::*, stats::prelude::*, with_dim, Mat, Row};
use reborrow::*;

fn bench_new(bencher: Bencher, n: usize) {
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
    let mut d = Row::<f64>::zeros_with_ctx(&default(), n);
    let mut d = d.as_mut();

    bencher.bench(|| {
        l.copy_from_with_ctx(&default(), a.as_ref());

        with_dim!(N, n);
        let mut full_l = l.rb_mut().as_shape_mut(N, N);
        let mut d = d.rb_mut().as_col_shape_mut(N);
        _ = faer::linalg::cholesky::ldlt::factor::simd_cholesky(
            &default(),
            full_l.as_mut(),
            d.as_mut(),
            false,
            false,
            &0.0,
            &0.0,
            None,
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
        l.copy_from_with_ctx(&default(), a.as_ref());
    });
}
fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);

    bench.register_many(
        list![bench_new, bench_copy],
        [4, 8, 12, 16, 32, 64, 128, 256],
    );
    bench.run()?;

    Ok(())
}
