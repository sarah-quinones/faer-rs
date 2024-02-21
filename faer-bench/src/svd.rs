use super::timeit;
use crate::random;
use dyn_stack::{GlobalPodBuffer, PodStack, ReborrowMut};
use faer::{linalg::svd as faer_svd, prelude::*, Parallelism};
use ndarray_linalg::{JobSvd, SVDDC};
use std::time::Duration;

pub fn ndarray<T: ndarray_linalg::Lapack>(sizes: &[usize]) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = ndarray::Array::<T, _>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    c[(i, j)] = random();
                }
            }

            let time = timeit(|| {
                c.svddc(JobSvd::All).unwrap();
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

pub fn faer<T: faer::ComplexField>(sizes: &[usize], parallelism: Parallelism) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = Mat::<T>::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    c.write(i, j, random());
                }
            }
            let mut s = Mat::<T>::zeros(n, n);
            let mut u = Mat::<T>::zeros(n, n);
            let mut v = Mat::<T>::zeros(n, n);

            let mut mem = GlobalPodBuffer::new(
                faer_svd::compute_svd_req::<T>(
                    n,
                    n,
                    faer_svd::ComputeVectors::Full,
                    faer_svd::ComputeVectors::Full,
                    parallelism,
                    faer_svd::SvdParams::default(),
                )
                .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);

            let time = timeit(|| {
                faer_svd::compute_svd(
                    c.as_ref(),
                    s.as_mut()
                        .submatrix_mut(0, 0, n, n)
                        .diagonal_mut()
                        .column_vector_mut()
                        .as_2d_mut(),
                    Some(u.as_mut()),
                    Some(v.as_mut()),
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
