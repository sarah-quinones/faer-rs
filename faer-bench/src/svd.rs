use super::timeit;
use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut};
use faer_core::{Mat, Parallelism};
use ndarray_linalg::SVD;
use rand::random;
use std::time::Duration;

pub fn ndarray(sizes: &[usize]) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = ndarray::Array::<f64, _>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    c[(i, j)] = random();
                }
            }

            let time = timeit(|| {
                c.svd(true, true).unwrap();
            });

            time
        })
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
                c.clone().svd(true, true);
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
            let mut s = Mat::<f64>::zeros(n, n);
            let mut u = Mat::<f64>::zeros(n, n);
            let mut v = Mat::<f64>::zeros(n, n);

            let mut mem = GlobalMemBuffer::new(
                faer_svd::compute_svd_req::<f64>(
                    n,
                    n,
                    faer_svd::ComputeVectors::Full,
                    faer_svd::ComputeVectors::Full,
                    parallelism,
                    faer_svd::SvdParams::default(),
                )
                .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            let time = timeit(|| {
                faer_svd::compute_svd(
                    c.as_ref(),
                    s.as_mut().diagonal(),
                    Some(u.as_mut()),
                    Some(v.as_mut()),
                    f64::EPSILON,
                    f64::MIN_POSITIVE,
                    parallelism,
                    stack.rb_mut(),
                    faer_svd::SvdParams::default(),
                );
            });

            let _ = c;

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}
