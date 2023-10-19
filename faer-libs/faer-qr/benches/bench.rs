use criterion::{criterion_group, criterion_main, Criterion};
use dyn_stack::*;
use faer_core::{c64, Mat, Parallelism};
use faer_qr::no_pivoting::compute::recommended_blocksize;
use rand::random;

pub fn qr(c: &mut Criterion) {
    use faer_qr::*;

    for (m, n) in [
        (6, 6),
        (8, 8),
        (10, 10),
        (12, 12),
        (24, 24),
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
        let mat = nalgebra::DMatrix::<f64>::from_fn(m, n, |_, _| random::<f64>());
        {
            c.bench_function(&format!("nalg-st-qr-{m}x{n}"), |b| {
                b.iter(|| {
                    mat.clone().qr();
                })
            });
            c.bench_function(&format!("nalg-st-colqr-{m}x{n}"), |b| {
                b.iter(|| {
                    mat.clone().col_piv_qr();
                })
            });
        }

        let mat = Mat::from_fn(m, n, |_, _| random::<f64>());

        {
            let mut copy = mat.clone();
            let blocksize = no_pivoting::compute::recommended_blocksize::<f64>(m, n);
            let mut householder = Mat::from_fn(blocksize, n, |_, _| random::<f64>());

            let mut mem = GlobalPodBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = PodStack::new(&mut mem);
            c.bench_function(&format!("faer-st-qr-{m}x{n}"), |b| {
                b.iter(|| {
                    faer_core::zipped!(copy.as_mut(), mat.as_ref())
                        .for_each(|mut dst, src| dst.write(src.read()));
                    no_pivoting::compute::qr_in_place(
                        copy.as_mut(),
                        householder.as_mut(),
                        Parallelism::None,
                        stack.rb_mut(),
                        Default::default(),
                    );
                })
            });
        }
        {
            let mut copy = mat.clone();
            let blocksize = no_pivoting::compute::recommended_blocksize::<f64>(m, n);
            let mut householder = Mat::from_fn(blocksize, n, |_, _| random::<f64>());

            let mut mem = GlobalPodBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = PodStack::new(&mut mem);
            c.bench_function(&format!("faer-mt-qr-{m}x{n}"), |b| {
                b.iter(|| {
                    faer_core::zipped!(copy.as_mut(), mat.as_ref())
                        .for_each(|mut dst, src| dst.write(src.read()));
                    no_pivoting::compute::qr_in_place(
                        copy.as_mut(),
                        householder.as_mut(),
                        Parallelism::Rayon(0),
                        stack.rb_mut(),
                        Default::default(),
                    );
                })
            });
        }

        {
            let mut copy = mat.clone();
            let blocksize = recommended_blocksize::<f64>(m, n);
            let mut householder = Mat::from_fn(blocksize, n, |_, _| random::<f64>());
            let mut perm = vec![0usize; n];
            let mut perm_inv = vec![0; n];
            c.bench_function(&format!("faer-st-colqr-{m}x{n}"), |b| {
                b.iter(|| {
                    faer_core::zipped!(copy.as_mut(), mat.as_ref())
                        .for_each(|mut dst, src| dst.write(src.read()));
                    col_pivoting::compute::qr_in_place(
                        copy.as_mut(),
                        householder.as_mut(),
                        &mut perm,
                        &mut perm_inv,
                        Parallelism::None,
                        PodStack::new(&mut []),
                        Default::default(),
                    );
                })
            });
        }

        {
            let mut copy = mat.clone();
            let blocksize = recommended_blocksize::<f64>(m, n);
            let mut householder = Mat::from_fn(blocksize, n, |_, _| random::<f64>());
            let mut perm = vec![0usize; n];
            let mut perm_inv = vec![0; n];
            c.bench_function(&format!("faer-mt-colqr-{m}x{n}"), |b| {
                b.iter(|| {
                    faer_core::zipped!(copy.as_mut(), mat.as_ref())
                        .for_each(|mut dst, src| dst.write(src.read()));
                    col_pivoting::compute::qr_in_place(
                        copy.as_mut(),
                        householder.as_mut(),
                        &mut perm,
                        &mut perm_inv,
                        Parallelism::Rayon(0),
                        PodStack::new(&mut []),
                        Default::default(),
                    );
                })
            });
        }

        let mat = Mat::from_fn(m, n, |_, _| c64::new(random(), random()));
        {
            let mut copy = mat.clone();
            let blocksize = recommended_blocksize::<c64>(m, n);
            let mut householder = Mat::from_fn(blocksize, n, |_, _| c64::new(random(), random()));
            let mut perm = vec![0usize; n];
            let mut perm_inv = vec![0; n];
            c.bench_function(&format!("faer-st-cplx-colqr-{m}x{n}"), |b| {
                b.iter(|| {
                    faer_core::zipped!(copy.as_mut(), mat.as_ref())
                        .for_each(|mut dst, src| dst.write(src.read()));
                    col_pivoting::compute::qr_in_place(
                        copy.as_mut(),
                        householder.as_mut(),
                        &mut perm,
                        &mut perm_inv,
                        Parallelism::None,
                        PodStack::new(&mut []),
                        Default::default(),
                    );
                })
            });
        }

        {
            let mut copy = mat.clone();
            let blocksize = recommended_blocksize::<c64>(m, n);
            let mut householder = Mat::from_fn(blocksize, n, |_, _| c64::new(random(), random()));
            let mut perm = vec![0usize; n];
            let mut perm_inv = vec![0; n];
            c.bench_function(&format!("faer-mt-cplx-colqr-{m}x{n}"), |b| {
                b.iter(|| {
                    faer_core::zipped!(copy.as_mut(), mat.as_ref())
                        .for_each(|mut dst, src| dst.write(src.read()));
                    col_pivoting::compute::qr_in_place(
                        copy.as_mut(),
                        householder.as_mut(),
                        &mut perm,
                        &mut perm_inv,
                        Parallelism::Rayon(0),
                        PodStack::new(&mut []),
                        Default::default(),
                    );
                })
            });
        }
    }

    let _c = c;
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = qr
);
criterion_main!(benches);
