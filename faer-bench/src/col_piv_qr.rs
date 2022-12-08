use super::timeit;
use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut};
use faer_core::{Mat, Parallelism};
use faer_qr::no_pivoting::compute::recommended_blocksize;
use rand::random;
use std::time::Duration;

pub fn ndarray(sizes: &[usize]) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|_| 0.0)
        .map(Duration::from_secs_f64)
        .collect()
}

pub fn nalgebra(sizes: &[usize]) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = nalgebra::DMatrix::<f64>::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    c[(i, j)] = random();
                }
            }

            let time = timeit(|| {
                nalgebra::linalg::ColPivQR::new(c.clone());
            });

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}

pub fn faer(sizes: &[usize], parallelism: Parallelism) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = Mat::<f64>::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    c[(i, j)] = random();
                }
            }
            let mut qr = Mat::<f64>::zeros(n, n);
            let blocksize = recommended_blocksize::<f64>(n, n);
            let mut householder = Mat::<f64>::zeros(blocksize, n);
            let mut perm = vec![0; n];
            let mut perm_inv = vec![0; n];

            let mut mem = GlobalMemBuffer::new(
                faer_qr::col_pivoting::compute::qr_in_place_req::<f64>(
                    n,
                    n,
                    blocksize,
                    parallelism,
                )
                .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            let mut block = || {
                qr.as_mut()
                    .cwise()
                    .zip(c.as_ref())
                    .for_each(|dst, src| *dst = *src);
                faer_qr::col_pivoting::compute::qr_in_place(
                    qr.as_mut(),
                    householder.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    parallelism,
                    stack.rb_mut(),
                    Default::default(),
                );
            };

            let time = timeit(|| block());

            let _ = c;

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}
