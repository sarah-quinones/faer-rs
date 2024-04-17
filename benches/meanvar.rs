use diol::prelude::*;
use faer::{prelude::*, ComplexField};

fn args() -> Vec<List![usize, usize]> {
    (5..12).map(|i| 1 << i).map(|n| list![n, n]).collect()
}

fn col_mean_propagate<E: ComplexField>(bencher: Bencher, unlist![m, n]: List![usize, usize]) {
    let a = Mat::from_fn(m, n, |_, _| E::faer_from_f64(1.0_f64));
    let mut out = Col::zeros(m);

    bencher.bench(|| {
        faer::stats::col_mean(
            out.as_mut(),
            a.as_ref(),
            faer::stats::NanHandling::Propagate,
        );
    })
}

fn col_mean_ignore<E: ComplexField>(bencher: Bencher, unlist![m, n]: List![usize, usize]) {
    let a = Mat::from_fn(m, n, |_, _| E::faer_from_f64(1.0_f64));
    let mut out = Col::zeros(m);

    bencher.bench(|| {
        faer::stats::col_mean(out.as_mut(), a.as_ref(), faer::stats::NanHandling::Ignore);
    })
}

fn row_mean_propagate<E: ComplexField>(bencher: Bencher, unlist![m, n]: List![usize, usize]) {
    let a = Mat::from_fn(m, n, |_, _| E::faer_from_f64(1.0_f64));
    let mut out = Row::zeros(n);

    bencher.bench(|| {
        faer::stats::row_mean(
            out.as_mut(),
            a.as_ref(),
            faer::stats::NanHandling::Propagate,
        );
    })
}

fn row_mean_ignore<E: ComplexField>(bencher: Bencher, unlist![m, n]: List![usize, usize]) {
    let a = Mat::from_fn(m, n, |_, _| E::faer_from_f64(1.0_f64));
    let mut out = Row::zeros(n);

    bencher.bench(|| {
        faer::stats::row_mean(out.as_mut(), a.as_ref(), faer::stats::NanHandling::Ignore);
    })
}

fn col_varm_propagate<E: ComplexField>(bencher: Bencher, unlist![m, n]: List![usize, usize]) {
    let a = Mat::from_fn(m, n, |_, _| E::faer_from_f64(1.0_f64));
    let mean = Col::from_fn(m, |_| E::faer_from_f64(1.0_f64));
    let mut out = Col::zeros(m);

    bencher.bench(|| {
        faer::stats::col_varm(
            out.as_mut(),
            a.as_ref(),
            mean.as_ref(),
            faer::stats::NanHandling::Propagate,
        );
    })
}

fn row_varm_propagate<E: ComplexField>(bencher: Bencher, unlist![m, n]: List![usize, usize]) {
    let a = Mat::from_fn(m, n, |_, _| E::faer_from_f64(1.0_f64));
    let mut out = Row::zeros(n);
    let mean = Row::from_fn(n, |_| E::faer_from_f64(1.0_f64));

    bencher.bench(|| {
        faer::stats::row_varm(
            out.as_mut(),
            a.as_ref(),
            mean.as_ref(),
            faer::stats::NanHandling::Propagate,
        );
    })
}

fn col_varm_ignore<E: ComplexField>(bencher: Bencher, unlist![m, n]: List![usize, usize]) {
    let a = Mat::from_fn(m, n, |_, _| E::faer_from_f64(1.0_f64));
    let mean = Col::from_fn(m, |_| E::faer_from_f64(1.0_f64));
    let mut out = Col::zeros(m);

    bencher.bench(|| {
        faer::stats::col_varm(
            out.as_mut(),
            a.as_ref(),
            mean.as_ref(),
            faer::stats::NanHandling::Ignore,
        );
    })
}

fn row_varm_ignore<E: ComplexField>(bencher: Bencher, unlist![m, n]: List![usize, usize]) {
    let a = Mat::from_fn(m, n, |_, _| E::faer_from_f64(1.0_f64));
    let mut out = Row::zeros(n);
    let mean = Row::from_fn(n, |_| E::faer_from_f64(1.0_f64));

    bencher.bench(|| {
        faer::stats::row_varm(
            out.as_mut(),
            a.as_ref(),
            mean.as_ref(),
            faer::stats::NanHandling::Ignore,
        );
    })
}

fn run_benches<E: ComplexField>() {
    let mut bench = Bench::new(BenchConfig {
        split: Split::ByGroup,
        ..BenchConfig::default()
    });
    bench.register(col_mean_propagate::<E>, args());
    bench.register(col_mean_ignore::<E>, args());
    bench.register(row_mean_propagate::<E>, args());
    bench.register(row_mean_ignore::<E>, args());
    bench.register(col_varm_propagate::<E>, args());
    bench.register(col_varm_ignore::<E>, args());
    bench.register(row_varm_propagate::<E>, args());
    bench.register(row_varm_ignore::<E>, args());
    bench.run();
}

fn main() {
    run_benches::<f32>();
    run_benches::<f64>();
    run_benches::<c32>();
    run_benches::<c64>();
}
