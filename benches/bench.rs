use criterion::{criterion_group, criterion_main, Criterion};
use dyn_stack::{DynStack, GlobalMemBuffer};
use reborrow::*;

use faer::Mat;
use nalgebra::DMatrix;

pub fn cholesky(c: &mut Criterion) {
    use faer::backend::cholesky::*;
    for n in [64, 128, 256, 512, 1024] {
        c.bench_function(&format!("faer-st-cholesky-left-{n}"), |b| {
            let mut mat = Mat::new();

            mat.resize_with(|i, j| if i == j { 1.0 } else { 0.0 }, n, n);
            let mut mem =
                GlobalMemBuffer::new(cholesky_in_place_left_looking_req::<f64>(n, 128, 1).unwrap());
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                cholesky_in_place_left_looking(mat.as_mut(), 128, 1, stack.rb_mut());
            })
        });

        c.bench_function(&format!("faer-mt-cholesky-left-{n}"), |b| {
            let mut mat = Mat::new();

            mat.resize_with(|i, j| if i == j { 1.0 } else { 0.0 }, n, n);
            let mut mem = GlobalMemBuffer::new(
                cholesky_in_place_left_looking_req::<f64>(n, 128, rayon::current_num_threads())
                    .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                cholesky_in_place_left_looking(
                    mat.as_mut(),
                    128,
                    rayon::current_num_threads(),
                    stack.rb_mut(),
                );
            })
        });

        c.bench_function(&format!("faer-st-cholesky-right-{n}"), |b| {
            let mut mat = Mat::new();

            mat.resize_with(|i, j| if i == j { 1.0 } else { 0.0 }, n, n);
            let mut mem =
                GlobalMemBuffer::new(cholesky_in_place_right_looking_req::<f64>(n, 1).unwrap());
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                cholesky_in_place_right_looking(mat.as_mut(), 1, stack.rb_mut());
            })
        });

        c.bench_function(&format!("faer-mt-cholesky-right-{n}"), |b| {
            let mut mat = Mat::new();

            mat.resize_with(|i, j| if i == j { 1.0 } else { 0.0 }, n, n);
            let mut mem = GlobalMemBuffer::new(
                cholesky_in_place_right_looking_req::<f64>(n, rayon::current_num_threads())
                    .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                cholesky_in_place_right_looking(
                    mat.as_mut(),
                    rayon::current_num_threads(),
                    stack.rb_mut(),
                );
            })
        });

        c.bench_function(&format!("nalg-st-cholesky-{n}"), |b| {
            let mut mat = DMatrix::<f64>::zeros(n, n);
            for i in 0..n {
                mat[(i, i)] = 1.0;
            }

            b.iter(|| {
                let _ = mat.clone().cholesky();
            })
        });
    }
}

criterion_group!(benches, cholesky);
criterion_main!(benches);
