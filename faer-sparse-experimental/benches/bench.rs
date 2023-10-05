#![allow(non_snake_case)]

use assert2::assert;
use core::iter::zip;
use criterion::*;
use dyn_stack::{GlobalPodBuffer, PodStack, StackReq};
use faer_core::Parallelism;
use faer_sparse_experimental::{
    cholesky::*,
    ghost::{self, Array},
    Index, SparseColMatRef,
};
use matrix_market_rs::MtxData;

fn bench_supernodal(criterion: &mut Criterion) {
    type I = i64;
    let truncate = I::truncate;

    let files = ["nd3k"];

    for file in files {
        let Ok(data) = MtxData::<f64>::from_file("./".to_string() + file + ".mtx") else {
            continue;
        };
        let MtxData::Sparse([nrows, _], coo_indices, coo_values, _) = data else {
            panic!()
        };

        let n = nrows;
        let mut col_counts = vec![0i64; n];
        for &[j, i] in &coo_indices {
            assert!(i <= j);
            col_counts[j] += 1;
        }
        let mut col_ptr = vec![0i64; n + 1];
        for i in 0..n {
            col_ptr[i + 1] = col_ptr[i] + col_counts[i];
        }
        let mut row_ind = vec![0i64; col_ptr[n] as usize];
        let mut values = vec![0.0; col_ptr[n] as usize];

        col_counts.copy_from_slice(&col_ptr[..n]);

        for (&[j, i], &val) in zip(&coo_indices, &coo_values) {
            row_ind[col_counts[j] as usize] = i as i64;
            values[col_counts[j] as usize] = val;
            col_counts[j] += 1;
        }

        let A = SparseColMatRef::new_checked(n, n, &col_ptr, None, &row_ind, &*values);
        let zero = truncate(0);
        let mut etree = vec![zero; n];
        let mut col_count = vec![zero; n];
        ghost::with_size(n, |N| {
            let A = ghost::SparseColMatRef::new(A, N, N);
            let etree = prefactorize_symbolic(
                Array::from_mut(&mut etree, N),
                Array::from_mut(&mut col_count, N),
                A.symbolic(),
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(5 * n))),
            );

            let symbolic = factorize_supernodal_symbolic(
                A.symbolic(),
                etree,
                Array::from_ref(&col_count, N),
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
                Default::default(),
            )
            .unwrap();

            let mut A_lower_col_ptr = col_ptr.to_vec();
            let mut A_lower_values = values.to_vec();
            let mut A_lower_row_ind = row_ind.to_vec();

            let A_lower = transpose(
                &mut A_lower_col_ptr,
                &mut A_lower_row_ind,
                &mut A_lower_values,
                A,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
            );

            let n_supernodes = symbolic.supernode_etree__.len();
            let mut values = vec![0.0f64; symbolic.col_ptrs_for_values__[n_supernodes].zx()];

            let parallelism = Parallelism::None;
            let mut mem = GlobalPodBuffer::new(
                factorize_supernodal_numeric_ldlt_req::<I, f64>(&symbolic, parallelism).unwrap(),
            );

            criterion.bench_function(&format!("supernodal-st-{file}"), |bench| {
                bench.iter(|| {
                    factorize_supernodal_numeric(
                        *A_lower,
                        &mut values,
                        &symbolic,
                        parallelism,
                        PodStack::new(&mut mem),
                    );
                });
            });

            let parallelism = Parallelism::Rayon(0);
            let mut mem = GlobalPodBuffer::new(
                factorize_supernodal_numeric_ldlt_req::<I, f64>(&symbolic, parallelism).unwrap(),
            );
            criterion.bench_function(&format!("supernodal-mt-{file}"), |bench| {
                bench.iter(|| {
                    factorize_supernodal_numeric(
                        *A_lower,
                        &mut values,
                        &symbolic,
                        parallelism,
                        PodStack::new(&mut mem),
                    );
                });
            });
        });
    }
}

criterion_group!(benches, bench_supernodal);
criterion_main!(benches);
