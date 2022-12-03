use super::timeit;
use faer_core::{Conj, Mat, Parallelism};
use std::time::Duration;

pub fn ndarray(sizes: &[usize]) -> Vec<Duration> {
    super::trsm::ndarray(sizes)
}

pub fn nalgebra(sizes: &[usize]) -> Vec<Duration> {
    super::trsm::nalgebra(sizes)
}

pub fn faer(sizes: &[usize], parallelism: Parallelism) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = Mat::<f64>::zeros(n, n);
            let a = Mat::<f64>::zeros(n, n);

            let time = timeit(|| {
                faer_core::inverse::invert_unit_lower_triangular(
                    c.as_mut(),
                    a.as_ref(),
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
