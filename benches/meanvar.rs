use faer::{prelude::*, ComplexField};

fn args() -> Vec<[usize; 2]> {
    vec![[4096; 2]]
}

#[divan::bench(types = [f32, c32, f64, c64], args = args())]
fn col_mean_propagate<E: ComplexField>(bencher: divan::Bencher, [m, n]: [usize; 2]) {
    let a = Mat::from_fn(m, n, |_, _| E::faer_from_f64(1.0_f64));
    let mut out = Col::zeros(m);

    bencher.bench_local(|| {
        faer::stats::col_mean(
            out.as_mut(),
            a.as_ref(),
            faer::stats::NanHandling::Propagate,
        );
    })
}

#[divan::bench(types = [f32, c32, f64, c64], args = args())]
fn col_mean_ignore<E: ComplexField>(bencher: divan::Bencher, [m, n]: [usize; 2]) {
    let a = Mat::from_fn(m, n, |_, _| E::faer_from_f64(1.0_f64));
    let mut out = Col::zeros(m);

    bencher.bench_local(|| {
        faer::stats::col_mean(out.as_mut(), a.as_ref(), faer::stats::NanHandling::Ignore);
    })
}

#[divan::bench(types = [f32, c32, f64, c64], args = args())]
fn row_mean_propagate<E: ComplexField>(bencher: divan::Bencher, [m, n]: [usize; 2]) {
    let a = Mat::from_fn(m, n, |_, _| E::faer_from_f64(1.0_f64));
    let mut out = Row::zeros(n);

    bencher.bench_local(|| {
        faer::stats::row_mean(
            out.as_mut(),
            a.as_ref(),
            faer::stats::NanHandling::Propagate,
        );
    })
}

#[divan::bench(types = [f32, c32, f64, c64], args = args())]
fn row_mean_ignore<E: ComplexField>(bencher: divan::Bencher, [m, n]: [usize; 2]) {
    let a = Mat::from_fn(m, n, |_, _| E::faer_from_f64(1.0_f64));
    let mut out = Row::zeros(n);

    bencher.bench_local(|| {
        faer::stats::row_mean(out.as_mut(), a.as_ref(), faer::stats::NanHandling::Ignore);
    })
}

#[divan::bench(types = [f32, c32, f64, c64], args = args())]
fn col_varm_propagate<E: ComplexField>(bencher: divan::Bencher, [m, n]: [usize; 2]) {
    let a = Mat::from_fn(m, n, |_, _| E::faer_from_f64(1.0_f64));
    let mean = Col::from_fn(m, |_| E::faer_from_f64(1.0_f64));
    let mut out = Col::zeros(m);

    bencher.bench_local(|| {
        faer::stats::col_varm(
            out.as_mut(),
            a.as_ref(),
            mean.as_ref(),
            faer::stats::NanHandling::Propagate,
        );
    })
}

#[divan::bench(types = [f32, c32, f64, c64], args = args())]
fn row_varm_propagate<E: ComplexField>(bencher: divan::Bencher, [m, n]: [usize; 2]) {
    let a = Mat::from_fn(m, n, |_, _| E::faer_from_f64(1.0_f64));
    let mut out = Row::zeros(n);
    let mean = Row::from_fn(n, |_| E::faer_from_f64(1.0_f64));

    bencher.bench_local(|| {
        faer::stats::row_varm(
            out.as_mut(),
            a.as_ref(),
            mean.as_ref(),
            faer::stats::NanHandling::Propagate,
        );
    })
}

#[divan::bench(types = [f32, c32, f64, c64], args = args())]
fn col_varm_ignore<E: ComplexField>(bencher: divan::Bencher, [m, n]: [usize; 2]) {
    let a = Mat::from_fn(m, n, |_, _| E::faer_from_f64(1.0_f64));
    let mean = Col::from_fn(m, |_| E::faer_from_f64(1.0_f64));
    let mut out = Col::zeros(m);

    bencher.bench_local(|| {
        faer::stats::col_varm(
            out.as_mut(),
            a.as_ref(),
            mean.as_ref(),
            faer::stats::NanHandling::Ignore,
        );
    })
}

#[divan::bench(types = [f32, c32, f64, c64], args = args())]
fn row_varm_ignore<E: ComplexField>(bencher: divan::Bencher, [m, n]: [usize; 2]) {
    let a = Mat::from_fn(m, n, |_, _| E::faer_from_f64(1.0_f64));
    let mut out = Row::zeros(n);
    let mean = Row::from_fn(n, |_| E::faer_from_f64(1.0_f64));

    bencher.bench_local(|| {
        faer::stats::row_varm(
            out.as_mut(),
            a.as_ref(),
            mean.as_ref(),
            faer::stats::NanHandling::Ignore,
        );
    })
}

fn main() {
    divan::main()
}
