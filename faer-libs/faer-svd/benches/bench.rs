use criterion::{criterion_group, criterion_main, Criterion};
use faer_svd::{
    bidiag::bidiagonalize_in_place, bidiag_real_svd::compute_bidiag_real_svd, compute_svd,
    SvdParams,
};
use std::time::Duration;

use dyn_stack::*;
use rand::random;

use faer_core::{Mat, Parallelism};

pub fn bidiag(c: &mut Criterion) {
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
    ] {
        let mat = Mat::from_fn(m, n, |_, _| random::<f64>());

        {
            let mut copy = mat.clone();
            let mut householder_left = Mat::from_fn(n, 1, |_, _| random::<f64>());
            let mut householder_right = Mat::from_fn(n, 1, |_, _| random::<f64>());

            let mut mem = GlobalPodBuffer::new(
                faer_svd::bidiag::bidiagonalize_in_place_req::<f64>(m, n, Parallelism::None)
                    .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);
            c.bench_function(&format!("faer-st-bidiag-{m}x{n}"), |b| {
                b.iter(|| {
                    copy.as_mut().copy_from(mat.as_ref());
                    bidiagonalize_in_place(
                        copy.as_mut(),
                        householder_left.as_mut().col_mut(0).as_2d_mut(),
                        householder_right.as_mut().col_mut(0).as_2d_mut(),
                        Parallelism::None,
                        stack.rb_mut(),
                    )
                })
            });
        }

        {
            let mut copy = mat.clone();
            let mut householder_left = Mat::from_fn(n, 1, |_, _| random::<f64>());
            let mut householder_right = Mat::from_fn(n, 1, |_, _| random::<f64>());

            let mut mem = GlobalPodBuffer::new(
                faer_svd::bidiag::bidiagonalize_in_place_req::<f64>(m, n, Parallelism::Rayon(0))
                    .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);
            c.bench_function(&format!("faer-st-bidiag-{m}x{n}"), |b| {
                b.iter(|| {
                    copy.as_mut().copy_from(mat.as_ref());
                    bidiagonalize_in_place(
                        copy.as_mut(),
                        householder_left.as_mut().col_mut(0).as_2d_mut(),
                        householder_right.as_mut().col_mut(0).as_2d_mut(),
                        Parallelism::Rayon(0),
                        stack.rb_mut(),
                    )
                })
            });
        }
    }
}

fn bidiag_svd(c: &mut Criterion) {
    for n in [32, 64, 128, 256, 1024, 4096] {
        let diag = (0..n).map(|_| rand::random::<f64>()).collect::<Vec<_>>();
        let subdiag = (0..n).map(|_| rand::random::<f64>()).collect::<Vec<_>>();

        {
            let mut diag_copy = diag.clone();
            let mut subdiag_copy = subdiag.clone();

            let mut u = Mat::zeros(n + 1, n + 1);
            let mut v = Mat::zeros(n, n);

            let mut mem = GlobalPodBuffer::new(
                faer_svd::bidiag_real_svd::bidiag_real_svd_req::<f64>(
                    n,
                    4,
                    true,
                    true,
                    Parallelism::None,
                )
                .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);

            c.bench_function(&format!("faer-st-bidiag-svd-{n}"), |bencher| {
                bencher.iter(|| {
                    diag_copy.clone_from_slice(&diag);
                    subdiag_copy.clone_from_slice(&subdiag);
                    let mut diag = (0..n).map(|_| rand::random::<f64>()).collect::<Vec<_>>();
                    let mut subdiag = (0..n).map(|_| rand::random::<f64>()).collect::<Vec<_>>();
                    compute_bidiag_real_svd(
                        &mut diag,
                        &mut subdiag,
                        Some(u.as_mut()),
                        Some(v.as_mut()),
                        4,
                        f64::EPSILON,
                        f64::MIN_POSITIVE,
                        Parallelism::None,
                        stack.rb_mut(),
                    );
                });
            });
        }
        let mut diag_copy = diag.clone();
        let mut subdiag_copy = subdiag.clone();

        let mut u = Mat::zeros(n + 1, n + 1);
        let mut v = Mat::zeros(n, n);

        let mut mem = GlobalPodBuffer::new(
            faer_svd::bidiag_real_svd::bidiag_real_svd_req::<f64>(
                n,
                4,
                true,
                true,
                Parallelism::Rayon(0),
            )
            .unwrap(),
        );
        let mut stack = PodStack::new(&mut mem);
        {
            c.bench_function(&format!("faer-mt-bidiag-svd-{n}"), |bencher| {
                bencher.iter(|| {
                    diag_copy.clone_from_slice(&diag);
                    subdiag_copy.clone_from_slice(&subdiag);
                    compute_bidiag_real_svd(
                        &mut diag_copy,
                        &mut subdiag_copy,
                        Some(u.as_mut()),
                        Some(v.as_mut()),
                        4,
                        f64::EPSILON,
                        f64::MIN_POSITIVE,
                        Parallelism::Rayon(0),
                        stack.rb_mut(),
                    );
                });
            });
        }
    }
}

fn real_svd(c: &mut Criterion) {
    for (m, n) in [
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (32, 4096),
        (1024, 1024),
        (10000, 128),
        (10000, 1024),
        (2048, 2048),
        (4096, 4096),
    ] {
        let mat = Mat::from_fn(m, n, |_, _| rand::random::<f64>());
        let mat = mat.as_ref();

        let size = m.min(n);
        let mut s = Mat::zeros(size, 1);
        let mut u = Mat::zeros(m, size);
        let mut v = Mat::zeros(n, size);

        {
            let mut mem = GlobalPodBuffer::new(
                faer_svd::compute_svd_req::<f64>(
                    m,
                    n,
                    faer_svd::ComputeVectors::Full,
                    faer_svd::ComputeVectors::Full,
                    Parallelism::None,
                    SvdParams::default(),
                )
                .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);
            c.bench_function(&format!("faer-st-svd-f64-{m}x{n}"), |bencher| {
                bencher.iter(|| {
                    compute_svd(
                        mat,
                        s.as_mut().col_mut(0).as_2d_mut(),
                        Some(u.as_mut()),
                        Some(v.as_mut()),
                        Parallelism::None,
                        stack.rb_mut(),
                        SvdParams::default(),
                    );
                });
            });
        }
        {
            let mut mem = GlobalPodBuffer::new(
                faer_svd::compute_svd_req::<f64>(
                    m,
                    n,
                    faer_svd::ComputeVectors::Full,
                    faer_svd::ComputeVectors::Full,
                    Parallelism::Rayon(0),
                    SvdParams::default(),
                )
                .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);
            c.bench_function(&format!("faer-mt-svd-f64-{m}x{n}"), |bencher| {
                bencher.iter(|| {
                    compute_svd(
                        mat,
                        s.as_mut().col_mut(0).as_2d_mut(),
                        Some(u.as_mut()),
                        Some(v.as_mut()),
                        Parallelism::Rayon(0),
                        stack.rb_mut(),
                        SvdParams::default(),
                    );
                });
            });
        }
    }

    for (m, n) in [
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (32, 4096),
    ] {
        let mut mat = nalgebra::DMatrix::<f64>::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                mat[(i, j)] = random();
            }
        }

        c.bench_function(&format!("nalgebra-svd-f64-{m}x{n}"), |bencher| {
            bencher.iter(|| mat.clone().svd(true, true))
        });
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5))
        .sample_size(10);
    targets = bidiag, bidiag_svd, real_svd,
);
criterion_main!(benches);
