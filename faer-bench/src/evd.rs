use super::timeit;
use crate::random;
use dyn_stack::{PodStack, GlobalPodBuffer, ReborrowMut};
use faer_core::{Mat, Parallelism};
use ndarray_linalg::Eig;
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
                c.eig().unwrap();
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
        .map(|_| 0.0)
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
            let mut s = Mat::<T>::zeros(n, n);
            let mut s_im = Mat::<T>::zeros(n, n);
            let mut u = Mat::<T>::zeros(n, n);

            let mut mem = GlobalPodBuffer::new(
                faer_evd::compute_evd_req::<T>(
                    n,
                    faer_evd::ComputeVectors::Yes,
                    parallelism,
                    Default::default(),
                )
                .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);

            let time = timeit(|| {
                if coe::is_same::<T, T::Real>() {
                    faer_evd::compute_evd_real::<T::Real>(
                        coe::coerce(c.as_ref()),
                        coe::coerce(s.as_mut().diagonal()),
                        coe::coerce(s_im.as_mut().diagonal()),
                        Some(coe::coerce(u.as_mut())),
                        parallelism,
                        stack.rb_mut(),
                        Default::default(),
                    );
                } else {
                    faer_evd::compute_evd_complex(
                        c.as_ref(),
                        s.as_mut().diagonal(),
                        Some(u.as_mut()),
                        parallelism,
                        stack.rb_mut(),
                        Default::default(),
                    );
                }
            });

            let _ = c;

            time
        })
        .map(Duration::from_secs_f64)
        .collect()
}
