use super::timeit;
use crate::random;
use dyn_stack::{GlobalPodBuffer, PodStack, ReborrowMut, StackReq};
use faer::{linalg::lu as faer_lu, unzipped, zipped, Mat, Parallelism};
use ndarray_linalg::Inverse;
use reborrow::*;
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
                c.inv().unwrap();
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
                c.clone().try_inverse().unwrap();
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
            let mut lu = Mat::<T>::zeros(n, n);
            let mut row_fwd = vec![0u32; n];
            let mut row_inv = vec![0u32; n];

            let mut mem = GlobalPodBuffer::new(StackReq::any_of([
                faer_lu::partial_pivoting::compute::lu_in_place_req::<u32, T>(
                    n,
                    n,
                    parallelism,
                    Default::default(),
                )
                .unwrap(),
                faer_lu::partial_pivoting::inverse::invert_req::<u32, T>(n, n, parallelism)
                    .unwrap(),
            ]));
            let mut stack = PodStack::new(&mut mem);

            let mut block = || {
                zipped!(lu.as_mut(), c.as_ref())
                    .for_each(|unzipped!(mut dst, src)| dst.write(src.read()));
                let (_, row_perm) = faer_lu::partial_pivoting::compute::lu_in_place(
                    lu.as_mut(),
                    &mut row_fwd,
                    &mut row_inv,
                    parallelism,
                    stack.rb_mut(),
                    Default::default(),
                );
                faer_lu::partial_pivoting::inverse::invert_in_place(
                    lu.as_mut(),
                    row_perm.rb(),
                    parallelism,
                    stack.rb_mut(),
                );
            };

            let time = timeit(|| block());

            let _ = c;

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}
