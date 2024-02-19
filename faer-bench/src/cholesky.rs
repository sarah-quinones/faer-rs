use super::timeit;
use dyn_stack::{GlobalPodBuffer, PodStack, ReborrowMut};
use faer_core::{unzipped, zipped, Mat, Parallelism};
use ndarray_linalg::Cholesky;
use std::time::Duration;

pub fn ndarray<T: ndarray_linalg::Lapack>(sizes: &[usize]) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = ndarray::Array::<T, _>::zeros((n, n));
            for i in 0..n {
                c[(i, i)] = T::one();
            }

            let time = timeit(|| {
                c.cholesky(ndarray_linalg::UPLO::Lower).unwrap();
            });

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
            for i in 0..n {
                c[(i, i)] = T::one();
            }

            let time = timeit(|| {
                nalgebra::linalg::Cholesky::new(c.clone());
            });

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
            for i in 0..n {
                c.write(i, i, T::faer_one());
            }
            let mut chol = Mat::<T>::zeros(n, n);

            let mut mem = GlobalPodBuffer::new(
                faer_cholesky::llt::compute::cholesky_in_place_req::<T>(
                    n,
                    parallelism,
                    Default::default(),
                )
                .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);

            let time = timeit(|| {
                zipped!(chol.as_mut(), c.as_ref())
                    .for_each(|unzipped!(mut dst, src)| dst.write(src.read()));
                faer_cholesky::llt::compute::cholesky_in_place(
                    chol.as_mut(),
                    Default::default(),
                    parallelism,
                    stack.rb_mut(),
                    Default::default(),
                )
                .unwrap();
            });

            let _ = c;

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}
