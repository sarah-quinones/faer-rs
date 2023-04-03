use super::timeit;
use faer_core::{Conj, Mat, Parallelism};
use num_traits::Zero;
use std::time::Duration;

pub fn ndarray<T: Zero + ndarray::LinalgScalar>(sizes: &[usize]) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = ndarray::Array::<T, _>::zeros((n, n));
            let a = ndarray::Array::<T, _>::zeros((n, n));
            let b = ndarray::Array::<T, _>::zeros((n, n));

            let time = timeit(|| {
                c = a.dot(&b);
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
            let b = nalgebra::DMatrix::<T>::zeros(n, n);

            let time = timeit(|| {
                a.mul_to(&b, &mut c);
            });

            let _ = c;

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}

pub fn faer<T: faer_core::ComplexField>(
    sizes: &[usize],
    parallelism: Parallelism,
) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = Mat::<T>::zeros(n, n);
            let a = Mat::<T>::zeros(n, n);
            let b = Mat::<T>::zeros(n, n);

            let time = timeit(|| {
                faer_core::mul::matmul(
                    c.as_mut(),
                    Conj::No,
                    a.as_ref(),
                    Conj::No,
                    b.as_ref(),
                    Conj::No,
                    None,
                    T::one(),
                    parallelism,
                );
            });

            let _ = c;

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}
