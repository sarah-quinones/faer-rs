use super::timeit;
use faer_core::{Conj, Mat, Parallelism};
use ndarray_linalg::*;
use std::time::Duration;

pub fn ndarray(sizes: &[usize]) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = ndarray::Array::<f64, _>::zeros((n, n));
            let a = ndarray::Array::<f64, _>::zeros((n, n));

            let time = timeit(|| {
                a.solve_triangular_inplace(UPLO::Lower, Diag::Unit, &mut c)
                    .unwrap();
            });

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

            let time = timeit(|| {
                a.solve_lower_triangular_with_diag_mut(&mut c, 1.0);
            });

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

            let time = timeit(|| {
                faer_core::solve::solve_unit_lower_triangular_in_place(
                    a.as_ref(),
                    Conj::No,
                    c.as_mut(),
                    Conj::No,
                    parallelism,
                );
            });

            let _ = c;

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}
