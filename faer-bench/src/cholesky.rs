use super::timeit;
use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut};
use faer_core::{Mat, Parallelism};
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

pub fn faer<T: faer_core::ComplexField>(sizes: &[usize], parallelism: Parallelism) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = Mat::<T>::zeros(n, n);
            for i in 0..n {
                c[(i, i)] = T::one();
            }
            let mut chol = Mat::<T>::zeros(n, n);

            let mut mem = GlobalMemBuffer::new(
                faer_cholesky::llt::compute::cholesky_in_place_req::<T>(
                    n,
                    parallelism,
                    Default::default(),
                )
                .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            let time = timeit(|| {
                chol.as_mut()
                    .cwise()
                    .zip(c.as_ref())
                    .for_each(|dst, src| *dst = src.clone());
                faer_cholesky::llt::compute::cholesky_in_place(
                    chol.as_mut(),
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
