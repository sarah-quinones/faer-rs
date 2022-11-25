use super::timeit;
use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut, StackReq};
use faer_core::{Mat, Parallelism};
use ndarray_linalg::QR;
use rand::random;
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
                c.qr().unwrap();
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
                nalgebra::linalg::QR::new(c.clone());
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
            let mut qr = Mat::<f64>::zeros(n, n);
            let mut householder = Mat::<f64>::zeros(n, 1);

            let mut mem = GlobalMemBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = DynStack::new(&mut mem);

            let mut block = || {
                qr.as_mut()
                    .cwise()
                    .zip(c.as_ref())
                    .for_each(|dst, src| *dst = *src);
                faer_qr::no_pivoting::compute::qr_in_place_blocked(
                    qr.as_mut(),
                    householder.as_mut().col(0),
                    128,
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
