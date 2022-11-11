use criterion::{criterion_group, criterion_main, Criterion};
use dyn_stack::{DynStack, GlobalMemBuffer};
use faer_core::{c64, ComplexField};
use reborrow::*;

use faer_core::{Mat, Parallelism};
use nalgebra::DMatrix;

pub fn cholesky(c: &mut Criterion) {
    use faer_cholesky::llt;

    for n in [6, 12, 64, 128, 256, 512, 1024] {
        // c.bench_function(&format!("faer-st-ldlt-{n}"), |b| {
        //     let mut mat = Mat::new();

        //     mat.resize_with(|i, j| if i == j { 1.0 } else { 0.0 }, n, n);
        //     let mut mem = GlobalMemBuffer::new(
        //         ldlt::compute::raw_cholesky_in_place_req::<f64>(n, 1).unwrap(),
        //     );
        //     let mut stack = DynStack::new(&mut mem);

        //     b.iter(|| {
        //         ldlt::compute::raw_cholesky_in_place(mat.as_mut(), 1, stack.rb_mut());
        //     })
        // });

        // c.bench_function(&format!("faer-mt-ldlt-{n}"), |b| {
        //     let mut mat = Mat::new();

        //     mat.resize_with(|i, j| if i == j { 1.0 } else { 0.0 }, n, n);
        //     let mut mem = GlobalMemBuffer::new(
        //         ldlt::compute::raw_cholesky_in_place_req::<f64>(n, rayon::current_num_threads())
        //             .unwrap(),
        //     );
        //     let mut stack = DynStack::new(&mut mem);

        //     b.iter(|| {
        //         ldlt::compute::raw_cholesky_in_place(
        //             mat.as_mut(),
        //             rayon::current_num_threads(),
        //             stack.rb_mut(),
        //         );
        //     })
        // });

        c.bench_function(&format!("faer-st-llt-{n}"), |b| {
            let mut mat = Mat::new();

            mat.resize_with(|i, j| if i == j { 1.0 } else { 0.0 }, n, n);
            let mut mem =
                GlobalMemBuffer::new(llt::compute::raw_cholesky_in_place_req::<f64>(n, 1).unwrap());
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                llt::compute::raw_cholesky_in_place(
                    mat.as_mut(),
                    Parallelism::None,
                    stack.rb_mut(),
                )
                .unwrap();
            })
        });

        c.bench_function(&format!("faer-mt-llt-{n}"), |b| {
            let mut mat = Mat::new();

            mat.resize_with(|i, j| if i == j { 1.0 } else { 0.0 }, n, n);
            let mut mem = GlobalMemBuffer::new(
                llt::compute::raw_cholesky_in_place_req::<f64>(n, rayon::current_num_threads())
                    .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                llt::compute::raw_cholesky_in_place(
                    mat.as_mut(),
                    Parallelism::Rayon,
                    stack.rb_mut(),
                )
                .unwrap();
            })
        });

        c.bench_function(&format!("faer-mt-cplx-llt-{n}"), |b| {
            let mut mat =
                Mat::with_dims(|i, j| if i == j { c64::one() } else { c64::zero() }, n, n);

            let mut mem = GlobalMemBuffer::new(
                llt::compute::raw_cholesky_in_place_req::<c64>(n, rayon::current_num_threads())
                    .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                llt::compute::raw_cholesky_in_place(
                    mat.as_mut(),
                    Parallelism::Rayon,
                    stack.rb_mut(),
                )
                .unwrap();
            })
        });

        c.bench_function(&format!("nalg-st-llt-{n}"), |b| {
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
