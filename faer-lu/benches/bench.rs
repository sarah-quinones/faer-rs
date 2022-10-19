use criterion::{criterion_group, criterion_main, Criterion};
use dyn_stack::{DynStack, GlobalMemBuffer, StackReq};
use rand::random;
use reborrow::*;

use faer_core::Mat;

pub fn lu(c: &mut Criterion) {
    use faer_lu::{full_pivoting, partial_pivoting};

    for n in [64, 128, 256, 512, 1024, 4096] {
        c.bench_function(&format!("faer-st-plu-{n}"), |b| {
            let mut mat = Mat::with_dims(|_, _| random::<f64>(), n, n);
            let mut perm = vec![0; n];
            let mut perm_inv = vec![0; n];

            let mut mem = GlobalMemBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                partial_pivoting::compute::lu_in_place(
                    mat.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    1,
                    stack.rb_mut(),
                );
            })
        });
        c.bench_function(&format!("faer-mt-plu-{n}"), |b| {
            let mut mat = Mat::with_dims(|_, _| random::<f64>(), n, n);
            let mut perm = vec![0; n];
            let mut perm_inv = vec![0; n];

            let mut mem = GlobalMemBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                partial_pivoting::compute::lu_in_place(
                    mat.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    rayon::current_num_threads(),
                    stack.rb_mut(),
                );
            })
        });

        c.bench_function(&format!("faer-st-flu-{n}"), |b| {
            let mut mat = Mat::with_dims(|_, _| random::<f64>(), n, n);
            let mut row_perm = vec![0; n];
            let mut row_perm_inv = vec![0; n];
            let mut col_perm = vec![0; n];
            let mut col_perm_inv = vec![0; n];

            let mut mem = GlobalMemBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                full_pivoting::compute::lu_in_place(
                    mat.as_mut(),
                    &mut row_perm,
                    &mut row_perm_inv,
                    &mut col_perm,
                    &mut col_perm_inv,
                    1,
                    stack.rb_mut(),
                );
            })
        });
        c.bench_function(&format!("faer-mt-flu-{n}"), |b| {
            let mut mat = Mat::with_dims(|_, _| random::<f64>(), n, n);
            let mut row_perm = vec![0; n];
            let mut row_perm_inv = vec![0; n];
            let mut col_perm = vec![0; n];
            let mut col_perm_inv = vec![0; n];

            let mut mem = GlobalMemBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                full_pivoting::compute::lu_in_place(
                    mat.as_mut(),
                    &mut row_perm,
                    &mut row_perm_inv,
                    &mut col_perm,
                    &mut col_perm_inv,
                    rayon::current_num_threads(),
                    stack.rb_mut(),
                );
            })
        });
    }
}

criterion_group!(benches, lu);
criterion_main!(benches);
