use super::timeit;
use faer::{Mat, Parallelism};
use std::time::Duration;

pub fn ndarray<T: ndarray_linalg::Lapack>(sizes: &[usize]) -> Vec<Duration> {
    super::trsm::ndarray::<T>(sizes)
}

pub fn nalgebra<T: nalgebra::ComplexField>(sizes: &[usize]) -> Vec<Duration> {
    super::trsm::nalgebra::<T>(sizes)
}

pub fn faer<T: faer::ComplexField>(sizes: &[usize], parallelism: Parallelism) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = Mat::<T>::zeros(n, n);
            let a = Mat::<T>::zeros(n, n);

            let time = timeit(|| {
                faer::linalg::triangular_inverse::invert_unit_lower_triangular(
                    c.as_mut(),
                    a.as_ref(),
                    parallelism,
                );
            });

            let _ = c;

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}
