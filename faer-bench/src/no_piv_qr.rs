use super::timeit;
use crate::random;
use dyn_stack::{PodStack, GlobalPodBuffer, ReborrowMut};
use faer_core::{Mat, Parallelism};
use ndarray_linalg::QR;
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
                c.qr().unwrap();
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
                nalgebra::linalg::QR::new(c.clone());
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
                for j in 0..n {
                    c.write(i, j, random());
                }
            }
            let mut qr = Mat::<T>::zeros(n, n);
            let blocksize = faer_qr::no_pivoting::compute::recommended_blocksize::<T>(n, n);
            let mut householder = Mat::<T>::zeros(blocksize, n);

            let mut mem = GlobalPodBuffer::new(
                faer_qr::no_pivoting::compute::qr_in_place_req::<T>(
                    n,
                    n,
                    blocksize,
                    parallelism,
                    Default::default(),
                )
                .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);

            let mut block = || {
                qr.as_mut()
                    .cwise()
                    .zip(c.as_ref())
                    .for_each(|mut dst, src| dst.write(src.read()));
                faer_qr::no_pivoting::compute::qr_in_place(
                    qr.as_mut(),
                    householder.as_mut(),
                    parallelism,
                    stack.rb_mut(),
                    Default::default(),
                );
            };

            let time = timeit(|| block());

            let _ = c;

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}
