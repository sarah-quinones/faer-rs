use criterion::{criterion_group, criterion_main, Criterion};
use dyn_stack::{GlobalPodBuffer, PodStack};
use faer_core::{Mat, Parallelism};
use faer_lu::{
    full_pivoting::compute::FullPivLuComputeParams,
    partial_pivoting::compute::PartialPivLuComputeParams,
};
use rand::random;
use reborrow::*;

use faer_lu::{full_pivoting, partial_pivoting};

pub fn lu(c: &mut Criterion) {
    for n in [4, 6, 8, 12, 32, 64, 128, 256, 512, 1023, 1024, 2048, 4096] {
        let partial_params = PartialPivLuComputeParams::default();
        let full_params = FullPivLuComputeParams::default();

        let mat = nalgebra::DMatrix::<f64>::from_fn(n, n, |_, _| random::<f64>());
        {
            c.bench_function(&format!("nalg-st-plu-{n}"), |b| {
                b.iter(|| {
                    mat.clone().lu();
                })
            });
            c.bench_function(&format!("nalg-st-flu-{n}"), |b| {
                b.iter(|| {
                    mat.clone().full_piv_lu();
                })
            });
        }

        let mat = Mat::from_fn(n, n, |_, _| random::<f64>());
        {
            let mut perm = vec![0; n];
            let mut perm_inv = vec![0; n];
            let mut copy = mat.clone();

            let mut mem = GlobalPodBuffer::new(
                partial_pivoting::compute::lu_in_place_req::<f64>(
                    n,
                    n,
                    Parallelism::None,
                    partial_params,
                )
                .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);
            c.bench_function(&format!("faer-st-plu-{n}"), |b| {
                b.iter(|| {
                    copy.as_mut().clone_from(mat.as_ref());
                    partial_pivoting::compute::lu_in_place(
                        copy.as_mut(),
                        &mut perm,
                        &mut perm_inv,
                        Parallelism::None,
                        stack.rb_mut(),
                        partial_params,
                    );
                })
            });
        }
        {
            let mut copy = mat.clone();
            let mut perm = vec![0; n];
            let mut perm_inv = vec![0; n];

            let mut mem = GlobalPodBuffer::new(
                partial_pivoting::compute::lu_in_place_req::<f64>(
                    n,
                    n,
                    Parallelism::Rayon(rayon::current_num_threads()),
                    partial_params,
                )
                .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);
            c.bench_function(&format!("faer-mt-plu-{n}"), |b| {
                b.iter(|| {
                    copy.as_mut().clone_from(mat.as_ref());
                    partial_pivoting::compute::lu_in_place(
                        copy.as_mut(),
                        &mut perm,
                        &mut perm_inv,
                        Parallelism::Rayon(0),
                        stack.rb_mut(),
                        partial_params,
                    );
                })
            });
        }

        {
            let mut copy = mat.clone();
            let mut row_perm = vec![0; n];
            let mut row_perm_inv = vec![0; n];
            let mut col_perm = vec![0; n];
            let mut col_perm_inv = vec![0; n];

            let mut mem = GlobalPodBuffer::new(
                full_pivoting::compute::lu_in_place_req::<f64>(
                    n,
                    n,
                    Parallelism::None,
                    full_params,
                )
                .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);

            c.bench_function(&format!("faer-st-flu-{n}"), |b| {
                b.iter(|| {
                    copy.as_mut().clone_from(mat.as_ref());
                    full_pivoting::compute::lu_in_place(
                        copy.as_mut(),
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
        }

        {
            let mut copy = mat.clone();
            let mut row_perm = vec![0; n];
            let mut row_perm_inv = vec![0; n];
            let mut col_perm = vec![0; n];
            let mut col_perm_inv = vec![0; n];

            let mut mem = GlobalPodBuffer::new(
                full_pivoting::compute::lu_in_place_req::<f64>(
                    n,
                    n,
                    Parallelism::None,
                    full_params,
                )
                .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);
            c.bench_function(&format!("faer-mt-flu-{n}"), |b| {
                b.iter(|| {
                    copy.as_mut().clone_from(mat.as_ref());
                    full_pivoting::compute::lu_in_place(
                        copy.as_mut(),
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
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = lu
);
criterion_main!(benches);
