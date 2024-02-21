use super::timeit;
use faer::{Mat, Parallelism};
use ndarray_linalg::*;
use std::time::Duration;

pub fn ndarray<T: Lapack>(sizes: &[usize]) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = ndarray::Array::<T, _>::zeros((n, n));
            let a = ndarray::Array::<T, _>::zeros((n, n));

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

pub fn nalgebra<T: nalgebra::ComplexField>(sizes: &[usize]) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = nalgebra::DMatrix::<T>::zeros(n, n);
            let a = nalgebra::DMatrix::<T>::zeros(n, n);

            let time = timeit(|| {
                a.solve_lower_triangular_with_diag_mut(&mut c, T::one());
            });

            let _ = c;

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}

pub fn faer<T: faer::ComplexField>(sizes: &[usize], parallelism: Parallelism) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = Mat::<T>::zeros(n, n);
            let a = Mat::<T>::zeros(n, n);

            let time = timeit(|| {
                faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                    a.as_ref(),
                    c.as_mut(),
                    parallelism,
                );
            });

            let _ = c;

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}
