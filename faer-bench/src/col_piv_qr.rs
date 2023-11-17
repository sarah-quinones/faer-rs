use super::timeit;
use crate::random;
use dyn_stack::{GlobalPodBuffer, PodStack, ReborrowMut};
use faer_core::{unzipped, zipped, Mat, Parallelism};
use faer_qr::no_pivoting::compute::recommended_blocksize;
use std::time::Duration;

pub fn ndarray<T: ndarray_linalg::Lapack>(sizes: &[usize]) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|_| 0.0)
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
                nalgebra::linalg::ColPivQR::new(c.clone());
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
            let blocksize = recommended_blocksize::<T>(n, n);
            let mut householder = Mat::<T>::zeros(blocksize, n);
            let mut perm = vec![0u32; n];
            let mut perm_inv = vec![0u32; n];

            let mut mem = GlobalPodBuffer::new(
                faer_qr::col_pivoting::compute::qr_in_place_req::<u32, T>(
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
                zipped!(qr.as_mut(), c.as_ref())
                    .for_each(|unzipped!(mut dst, src)| dst.write(src.read()));
                faer_qr::col_pivoting::compute::qr_in_place(
                    qr.as_mut(),
                    householder.as_mut(),
                    &mut perm,
                    &mut perm_inv,
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
