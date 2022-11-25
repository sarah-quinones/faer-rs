use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;

use dyn_stack::*;
use faer_core::Parallelism;
use rand::random;

use faer_core::Mat;

pub fn qr(c: &mut Criterion) {
    use faer_qr::*;
    for (m, n) in [
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (10000, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
    ] {
        c.bench_function(&format!("faer-st-qr-{m}x{n}"), |b| {
            let mut mat = Mat::with_dims(|_, _| random::<f64>(), m, n);
            let mut householder = Mat::with_dims(|_, _| random::<f64>(), n, 1);

            let mut mem = GlobalMemBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                no_pivoting::compute::qr_in_place_blocked(
                    mat.as_mut(),
                    householder.as_mut().col(0),
                    32,
                    Parallelism::None,
                    stack.rb_mut(),
                );
            })
        });

        c.bench_function(&format!("faer-mt-qr-{m}x{n}"), |b| {
            let mut mat = Mat::with_dims(|_, _| random::<f64>(), m, n);
            let mut householder = Mat::with_dims(|_, _| random::<f64>(), n, 1);

            let mut mem = GlobalMemBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                no_pivoting::compute::qr_in_place_blocked(
                    mat.as_mut(),
                    householder.as_mut().col(0),
                    32,
                    Parallelism::Rayon(0),
                    stack.rb_mut(),
                );
            })
        });

        c.bench_function(&format!("faer-st-colqr-{m}x{n}"), |b| {
            let mut mat = Mat::with_dims(|_, _| random::<f64>(), m, n);
            let mut householder = Mat::with_dims(|_, _| random::<f64>(), n, n);
            let mut transpositions = vec![0; n];

            b.iter(|| {
                col_pivoting::compute::qr_in_place(
                    mat.as_mut(),
                    householder.as_mut().diagonal(),
                    &mut transpositions,
                    Parallelism::None,
                );
            })
        });

        c.bench_function(&format!("faer-mt-colqr-{m}x{n}"), |b| {
            let mut mat = Mat::with_dims(|_, _| random::<f64>(), m, n);
            let mut householder = Mat::with_dims(|_, _| random::<f64>(), n, n);
            let mut transpositions = vec![0; n];

            b.iter(|| {
                col_pivoting::compute::qr_in_place(
                    mat.as_mut(),
                    householder.as_mut().diagonal(),
                    &mut transpositions,
                    Parallelism::Rayon(rayon::current_num_threads()),
                );
            })
        });
    }

    let _c = c;
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(1))
        .sample_size(10);
    targets = qr
);
criterion_main!(benches);
