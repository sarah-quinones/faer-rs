use faer_core::{Conj, Mat, Parallelism};
use std::time::Duration;
use timeit::*;

pub fn ndarray(sizes: &[usize]) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = ndarray::Array::<f64, _>::zeros((n, n));
            let a = ndarray::Array::<f64, _>::zeros((n, n));
            let b = ndarray::Array::<f64, _>::zeros((n, n));

            let time = timeit_loops! {100, {
                c = a.dot(&b);
            }};

            let _ = c;

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
            let a = nalgebra::DMatrix::<f64>::zeros(n, n);
            let b = nalgebra::DMatrix::<f64>::zeros(n, n);

            let time = timeit_loops! {100, {
                a.mul_to(&b, &mut c);
            }};

            let _ = c;

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
            let a = Mat::<f64>::zeros(n, n);
            let b = Mat::<f64>::zeros(n, n);

            let time = timeit_loops! {100, {
                faer_core::mul::matmul(
                    c.as_mut(),
                    Conj::No,
                    a.as_ref(),
                    Conj::No,
                    b.as_ref(),
                    Conj::No,
                    None,
                    1.0,
                    parallelism,
                );
            }};

            let _ = c;

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}
