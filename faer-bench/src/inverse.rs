use super::timeit;
use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut, StackReq};
use faer_core::{Mat, Parallelism};
use ndarray_linalg::Inverse;
use rand::random;
use reborrow::*;
use std::time::Duration;

pub fn ndarray(sizes: &[usize]) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = ndarray::Array::<f64, _>::zeros((n, n));
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

pub fn nalgebra(sizes: &[usize]) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = nalgebra::DMatrix::<f64>::zeros(n, n);
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

pub fn faer(sizes: &[usize], parallelism: Parallelism) -> Vec<Duration> {
    sizes
        .iter()
        .copied()
        .map(|n| {
            let mut c = Mat::<f64>::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    c[(i, j)] = random();
                }
            }
            let mut lu = Mat::<f64>::zeros(n, n);
            let mut row_fwd = vec![0; n];
            let mut row_inv = vec![0; n];

            let mut mem = GlobalMemBuffer::new(StackReq::any_of([
                faer_lu::partial_pivoting::compute::lu_in_place_req::<f64>(
                    n,
                    n,
                    parallelism,
                    Default::default(),
                )
                .unwrap(),
                faer_lu::partial_pivoting::inverse::invert_to_req::<f64>(n, n, parallelism)
                    .unwrap(),
            ]));
            let mut stack = DynStack::new(&mut mem);

            let mut block = || {
                lu.as_mut()
                    .cwise()
                    .zip(c.as_ref())
                    .for_each(|dst, src| *dst = *src);
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
