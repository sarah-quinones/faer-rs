use super::timeit;
use crate::random;
use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut};
use faer_core::{Mat, Parallelism};
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
                nalgebra::linalg::FullPivLU::new(c.clone());
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
                    c[(i, j)] = random();
                }
            }
            let mut lu = Mat::<T>::zeros(n, n);
            let mut row_fwd = vec![0; n];
            let mut row_inv = vec![0; n];
            let mut col_fwd = vec![0; n];
            let mut col_inv = vec![0; n];

            let mut mem = GlobalMemBuffer::new(
                faer_lu::full_pivoting::compute::lu_in_place_req::<T>(
                    n,
                    n,
                    parallelism,
                    Default::default(),
                )
                .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            let mut block = || {
                lu.as_mut()
                    .cwise()
                    .zip(c.as_ref())
                    .for_each(|dst, src| *dst = src.clone());
                faer_lu::full_pivoting::compute::lu_in_place(
                    lu.as_mut(),
                    &mut row_fwd,
                    &mut row_inv,
                    &mut col_fwd,
                    &mut col_inv,
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
