use criterion::{criterion_group, criterion_main, Criterion};
use dyn_stack::{DynStack, GlobalMemBuffer};
use faer_core::{zipped, Mat, Parallelism};
use faer_lu::{
    full_pivoting::compute::FullPivLuComputeParams,
    partial_pivoting::compute::PartialPivLuComputeParams,
};
use rand::random;
use reborrow::*;
use std::time::Duration;

pub fn lu(c: &mut Criterion) {
    use faer_lu::{full_pivoting, partial_pivoting};

    for n in [4, 6, 8, 12, 32, 64, 128, 256, 512, 1023, 1024, 4096] {
        let partial_params = PartialPivLuComputeParams::default();
        let full_params = FullPivLuComputeParams::default();
        c.bench_function(&format!("faer-st-plu-{n}"), |b| {
            let mut mat = Mat::with_dims(n, n, |_, _| random::<f64>());
            let mut perm = vec![0; n];
            let mut perm_inv = vec![0; n];

            let mut mem = GlobalMemBuffer::new(
                partial_pivoting::compute::lu_in_place_req::<f64>(
                    n,
                    n,
                    Parallelism::None,
                    partial_params,
                )
                .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                partial_pivoting::compute::lu_in_place(
                    mat.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    Parallelism::None,
                    stack.rb_mut(),
                    partial_params,
                );
            })
        });
        c.bench_function(&format!("faer-mt-plu-{n}"), |b| {
            let mat_orig = Mat::with_dims(n, n, |_, _| random::<f64>());
            let mut mat = mat_orig.clone();
            let mut perm = vec![0; n];
            let mut perm_inv = vec![0; n];

            let mut mem = GlobalMemBuffer::new(
                partial_pivoting::compute::lu_in_place_req::<f64>(
                    n,
                    n,
                    Parallelism::Rayon(rayon::current_num_threads()),
                    partial_params,
                )
                .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                zipped!(mat.as_mut(), mat_orig.as_ref())
                    .for_each(|mut dst, src| dst.write(src.read()));
                partial_pivoting::compute::lu_in_place(
                    mat.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    Parallelism::Rayon(rayon::current_num_threads()),
                    stack.rb_mut(),
                    partial_params,
                );
            })
        });

        c.bench_function(&format!("faer-r-mt-plu-{n}"), |b| {
            let mut mat = Mat::with_dims(n, n, |_, _| random::<f64>());
            let mut perm = vec![0; n];
            let mut perm_inv = vec![0; n];

            let mut mem = GlobalMemBuffer::new(
                partial_pivoting::compute::lu_in_place_req::<f64>(
                    n,
                    n,
                    Parallelism::Rayon(rayon::current_num_threads()),
                    partial_params,
                )
                .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                partial_pivoting::compute::lu_in_place(
                    mat.as_mut().transpose(),
                    &mut perm,
                    &mut perm_inv,
                    Parallelism::Rayon(rayon::current_num_threads()),
                    stack.rb_mut(),
                    partial_params,
                );
            })
        });

        c.bench_function(&format!("faer-st-flu-{n}"), |b| {
            let mut mat = Mat::with_dims(n, n, |_, _| random::<f64>());
            let mut row_perm = vec![0; n];
            let mut row_perm_inv = vec![0; n];
            let mut col_perm = vec![0; n];
            let mut col_perm_inv = vec![0; n];

            let mut mem = GlobalMemBuffer::new(
                full_pivoting::compute::lu_in_place_req::<f64>(
                    n,
                    n,
                    Parallelism::None,
                    full_params,
                )
                .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                full_pivoting::compute::lu_in_place(
                    mat.as_mut(),
                    &mut row_perm,
                    &mut row_perm_inv,
                    &mut col_perm,
                    &mut col_perm_inv,
                    Parallelism::None,
                    stack.rb_mut(),
                    full_params,
                );
            })
        });

        c.bench_function(&format!("faer-mt-flu-{n}"), |b| {
            let mut mat = Mat::with_dims(n, n, |_, _| random::<f64>());
            let mut row_perm = vec![0; n];
            let mut row_perm_inv = vec![0; n];
            let mut col_perm = vec![0; n];
            let mut col_perm_inv = vec![0; n];

            let mut mem = GlobalMemBuffer::new(
                full_pivoting::compute::lu_in_place_req::<f64>(
                    n,
                    n,
                    Parallelism::None,
                    full_params,
                )
                .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                full_pivoting::compute::lu_in_place(
                    mat.as_mut(),
                    &mut row_perm,
                    &mut row_perm_inv,
                    &mut col_perm,
                    &mut col_perm_inv,
                    Parallelism::Rayon(rayon::current_num_threads()),
                    stack.rb_mut(),
                    full_params,
                );
            })
        });

        c.bench_function(&format!("faer-r-mt-flu-{n}"), |b| {
            let mut mat = Mat::with_dims(n, n, |_, _| random::<f64>());
            let mut row_perm = vec![0; n];
            let mut row_perm_inv = vec![0; n];
            let mut col_perm = vec![0; n];
            let mut col_perm_inv = vec![0; n];

            let mut mem = GlobalMemBuffer::new(
                full_pivoting::compute::lu_in_place_req::<f64>(
                    n,
                    n,
                    Parallelism::None,
                    full_params,
                )
                .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                full_pivoting::compute::lu_in_place(
                    mat.as_mut().transpose(),
                    &mut row_perm,
                    &mut row_perm_inv,
                    &mut col_perm,
                    &mut col_perm_inv,
                    Parallelism::Rayon(rayon::current_num_threads()),
                    stack.rb_mut(),
                    full_params,
                );
            })
        });

        c.bench_function(&format!("faer-mt-flu-f32-{n}"), |b| {
            let mut mat = Mat::with_dims(n, n, |_, _| random::<f32>());
            let mut row_perm = vec![0; n];
            let mut row_perm_inv = vec![0; n];
            let mut col_perm = vec![0; n];
            let mut col_perm_inv = vec![0; n];

            let mut mem = GlobalMemBuffer::new(
                full_pivoting::compute::lu_in_place_req::<f32>(
                    n,
                    n,
                    Parallelism::None,
                    full_params,
                )
                .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                full_pivoting::compute::lu_in_place(
                    mat.as_mut(),
                    &mut row_perm,
                    &mut row_perm_inv,
                    &mut col_perm,
                    &mut col_perm_inv,
                    Parallelism::Rayon(rayon::current_num_threads()),
                    stack.rb_mut(),
                    full_params,
                );
            })
        });
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(1))
        .sample_size(10);
    targets = lu
);
criterion_main!(benches);
