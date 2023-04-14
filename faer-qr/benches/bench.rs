use criterion::{criterion_group, criterion_main, Criterion};
use dyn_stack::*;
use faer_core::{c64, Mat, Parallelism};
use faer_qr::no_pivoting::compute::recommended_blocksize;
use rand::random;
use std::time::Duration;

pub fn qr(c: &mut Criterion) {
    use faer_qr::*;

    for (m, n) in [
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (10000, 128),
        (10000, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
    ] {
        c.bench_function(&format!("faer-st-qr-{m}x{n}"), |b| {
            let mut mat = Mat::with_dims(m, n, |_, _| random::<f64>());
            let blocksize = no_pivoting::compute::recommended_blocksize::<f64>(m, n);
            let mut householder = Mat::with_dims(blocksize, n, |_, _| random::<f64>());

            let mut mem = GlobalMemBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                no_pivoting::compute::qr_in_place(
                    mat.as_mut(),
                    householder.as_mut(),
                    Parallelism::None,
                    stack.rb_mut(),
                    Default::default(),
                );
            })
        });

        c.bench_function(&format!("faer-mt-qr-{m}x{n}"), |b| {
            let mut mat = Mat::with_dims(m, n, |_, _| random::<f64>());
            let blocksize = no_pivoting::compute::recommended_blocksize::<f64>(m, n);
            let mut householder = Mat::with_dims(blocksize, n, |_, _| random::<f64>());

            let mut mem = GlobalMemBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                no_pivoting::compute::qr_in_place(
                    mat.as_mut(),
                    householder.as_mut(),
                    Parallelism::Rayon(0),
                    stack.rb_mut(),
                    Default::default(),
                );
            })
        });

        c.bench_function(&format!("faer-st-colqr-{m}x{n}"), |b| {
            let mat = Mat::with_dims(m, n, |_, _| random::<f64>());
            let mut copy = mat.clone();
            let blocksize = recommended_blocksize::<f64>(m, n);
            let mut householder = Mat::with_dims(blocksize, n, |_, _| random::<f64>());
            let mut perm = vec![0; n];
            let mut perm_inv = vec![0; n];

            b.iter(|| {
                faer_core::zipped!(copy.as_mut(), mat.as_ref())
                    .for_each(|mut dst, src| dst.write(src.read()));
                col_pivoting::compute::qr_in_place(
                    copy.as_mut(),
                    householder.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    Parallelism::None,
                    DynStack::new(&mut []),
                    Default::default(),
                );
            })
        });

        c.bench_function(&format!("faer-mt-colqr-{m}x{n}"), |b| {
            let mut mat = Mat::with_dims(m, n, |_, _| random::<f64>());
            let blocksize = recommended_blocksize::<f64>(m, n);
            let mut householder = Mat::with_dims(blocksize, n, |_, _| random::<f64>());
            let mut perm = vec![0; n];
            let mut perm_inv = vec![0; n];

            b.iter(|| {
                col_pivoting::compute::qr_in_place(
                    mat.as_mut(),
                    householder.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    Parallelism::Rayon(0),
                    DynStack::new(&mut []),
                    Default::default(),
                );
            })
        });

        c.bench_function(&format!("faer-st-cplx-colqr-{m}x{n}"), |b| {
            let mut mat = Mat::with_dims(m, n, |_, _| c64::new(random(), random()));
            let blocksize = recommended_blocksize::<c64>(m, n);
            let mut householder = Mat::with_dims(blocksize, n, |_, _| c64::new(random(), random()));
            let mut perm = vec![0; n];
            let mut perm_inv = vec![0; n];

            b.iter(|| {
                col_pivoting::compute::qr_in_place(
                    mat.as_mut(),
                    householder.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    Parallelism::None,
                    DynStack::new(&mut []),
                    Default::default(),
                );
            })
        });

        c.bench_function(&format!("faer-mt-cplx-colqr-{m}x{n}"), |b| {
            let mut mat = Mat::with_dims(m, n, |_, _| c64::new(random(), random()));
            let blocksize = recommended_blocksize::<c64>(m, n);
            let mut householder = Mat::with_dims(blocksize, n, |_, _| c64::new(random(), random()));
            let mut perm = vec![0; n];
            let mut perm_inv = vec![0; n];

            b.iter(|| {
                col_pivoting::compute::qr_in_place(
                    mat.as_mut(),
                    householder.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    Parallelism::Rayon(0),
                    DynStack::new(&mut []),
                    Default::default(),
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
