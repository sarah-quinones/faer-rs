#![allow(non_snake_case)]

use core::iter::zip;
use criterion::*;
use dyn_stack::{GlobalPodBuffer, PodStack};
use faer_core::Parallelism;
use faer_sparse_experimental::{
    cholesky::*, Index, Side, SliceGroup, SliceGroupMut, SparseColMatRef, SymbolicSparseColMatRef,
};
use matrix_market_rs::MtxData;
use reborrow::*;

fn load_mtx(data: MtxData<f64>) -> (usize, Vec<i64>, Vec<i64>, Vec<f64>) {
    let MtxData::Sparse([nrows, _], coo_indices, coo_values, _) = data else {
        panic!()
    };

    let n = nrows;
    let mut col_counts = vec![0i64; n];
    let mut col_ptr = vec![0i64; n + 1];

    for &[_, j] in &coo_indices {
        col_counts[j] += 1;
    }

    for i in 0..n {
        col_ptr[i + 1] = col_ptr[i] + col_counts[i];
    }
    let nnz = col_ptr[n] as usize;

    let mut row_ind = vec![0i64; nnz];
    let mut values = vec![0.0; nnz];

    col_counts.copy_from_slice(&col_ptr[..n]);

    for (&[i, j], &val) in zip(&coo_indices, &coo_values) {
        row_ind[col_counts[j] as usize] = i as i64;
        values[col_counts[j] as usize] = val;
        col_counts[j] += 1;
    }

    (n, col_ptr, row_ind, values)
}

fn bench_ldlt(criterion: &mut Criterion) {
    type I = i64;
    let I = I::truncate;

    let files = [
        ("stiffness", Side::Upper),
        ("nd3k", Side::Lower),
        ("nd12k", Side::Lower),
    ];

    for (file, side) in files {
        let Ok(data) = MtxData::<f64>::from_file("./".to_string() + file + ".mtx") else {
            continue;
        };
        let (n, col_ptr, row_ind, values) = load_mtx(data);
        let col_ptr = col_ptr
            .into_iter()
            .map(|x| I(x as usize))
            .collect::<Vec<_>>();
        let row_ind = row_ind
            .into_iter()
            .map(|x| I(x as usize))
            .collect::<Vec<_>>();

        let A = SparseColMatRef::<_, f64>::new(
            SymbolicSparseColMatRef::new_checked(n, n, &col_ptr, None, &row_ind),
            SliceGroup::new(&*values),
        );

        let symbolic = factorize_symbolic(A.symbolic(), side, Default::default()).unwrap();
        let parallelism = Parallelism::None;
        let mut mem = GlobalPodBuffer::new(
            symbolic
                .factorize_numeric_ldlt_req::<f64>(side, parallelism)
                .unwrap(),
        );
        let mut L_values = vec![0.0f64; symbolic.len_values()];
        let mut L_values = SliceGroupMut::new(&mut *L_values);

        criterion.bench_function(&format!("ldlt-st-{file}"), |bench| {
            bench.iter(|| {
                symbolic.factorize_numeric_ldlt(
                    L_values.rb_mut(),
                    A,
                    side,
                    parallelism,
                    PodStack::new(&mut mem),
                );
            });
        });

        let parallelism = Parallelism::Rayon(0);
        let mut mem = GlobalPodBuffer::new(
            symbolic
                .factorize_numeric_ldlt_req::<f64>(side, parallelism)
                .unwrap(),
        );
        criterion.bench_function(&format!("ldlt-mt-{file}"), |bench| {
            bench.iter(|| {
                symbolic.factorize_numeric_ldlt(
                    L_values.rb_mut(),
                    A,
                    side,
                    parallelism,
                    PodStack::new(&mut mem),
                );
            });
        });
    }
}

criterion_group!(benches, bench_ldlt);
criterion_main!(benches);
