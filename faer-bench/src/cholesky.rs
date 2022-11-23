use super::timeit;
use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut};
use faer_core::{Mat, Parallelism};
use ndarray_linalg::Cholesky;
use std::time::Duration;

pub fn ndarray(sizes: &[usize]) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = ndarray::Array::<f64, _>::zeros((n, n));
            for i in 0..n {
                c[(i, i)] = 1.0;
            }

            let time = timeit(|| {
                c.cholesky(ndarray_linalg::UPLO::Lower).unwrap();
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
                c[(i, i)] = 1.0;
            }

            let time = timeit(|| {
                nalgebra::linalg::Cholesky::new(c.clone());
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
                c[(i, i)] = 1.0;
            }
            let mut lu = Mat::<f64>::zeros(n, n);

            let mut mem = GlobalMemBuffer::new(
                faer_cholesky::llt::compute::raw_cholesky_in_place_req::<f64>(n, parallelism)
                    .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            let time = timeit(|| {
                lu.as_mut()
                    .cwise()
                    .zip(c.as_ref())
                    .for_each(|dst, src| *dst = *src);
                faer_cholesky::llt::compute::raw_cholesky_in_place(
                    lu.as_mut(),
                    parallelism,
                    stack.rb_mut(),
                )
                .unwrap();
            });

            let _ = c;

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}
