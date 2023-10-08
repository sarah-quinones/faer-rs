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
use std::{ffi::OsStr, time::Instant};

fn load_mtx<I: Index>(data: MtxData<f64>) -> (usize, Vec<I>, Vec<I>, Vec<f64>) {
    let I = I::truncate;

    let MtxData::Sparse([nrows, _], coo_indices, coo_values, _) = data else {
        panic!()
    };

    let n = nrows;
    let mut col_counts = vec![I(0); n];
    let mut col_ptr = vec![I(0); n + 1];

    for &[_, j] in &coo_indices {
        col_counts[j] += I(1);
    }

    for i in 0..n {
        col_ptr[i + 1] = col_ptr[i] + col_counts[i];
    }
    let nnz = col_ptr[n].zx();

    let mut row_ind = vec![I(0); nnz];
    let mut values = vec![0.0; nnz];

    col_counts.copy_from_slice(&col_ptr[..n]);

    for (&[i, j], &val) in zip(&coo_indices, &coo_values) {
        row_ind[col_counts[j].zx()] = I(i);
        values[col_counts[j].zx()] = val;
        col_counts[j] += I(1);
    }

    (n, col_ptr, row_ind, values)
}

fn bench_ldlt(criterion: &mut Criterion) {
    type I = i64;
    let mut files = Vec::new();

    for file in std::fs::read_dir(".").unwrap() {
        let file = file.unwrap();
        if file.path().extension() == Some(OsStr::new("mtx")) {
            let name = file
                .path()
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned();
            files.push((name.strip_suffix(".mtx").unwrap().to_string(), Side::Upper))
        }
    }

    files.sort_by(|(f0, _), (f1, _)| str::cmp(f0, f1));

    for (file, side) in files {
        let Ok(data) = MtxData::<f64>::from_file("./".to_string() + &*file + ".mtx") else {
            continue;
        };
        let (n, col_ptr, row_ind, values) = load_mtx::<I>(data);
        let A = SparseColMatRef::<_, f64>::new(
            SymbolicSparseColMatRef::new_checked(n, n, &col_ptr, None, &row_ind),
            SliceGroup::new(&*values),
        );

        let symbolic = factorize_symbolic(A.symbolic(), side, Default::default()).unwrap();
        let symbolic_type = match symbolic.raw() {
            SymbolicCholeskyRaw::Simplicial(_) => "simplicial",
            SymbolicCholeskyRaw::Supernodal(_) => "supernodal",
        };
        println!("picked {symbolic_type} method for {file}");
        let simplicial = {
            let symbolic = factorize_symbolic(
                A.symbolic(),
                side,
                CholeskySymbolicParams {
                    supernodal_flop_ratio_threshold: f64::INFINITY,
                    ..Default::default()
                },
            )
            .unwrap();
            let parallelism = Parallelism::None;
            let mut mem = GlobalPodBuffer::new(
                symbolic
                    .factorize_numeric_ldlt_req::<f64>(side, parallelism)
                    .unwrap(),
            );
            let mut L_values = vec![0.0f64; symbolic.len_values()];
            let mut L_values = SliceGroupMut::new(&mut *L_values);

            let start = Instant::now();
            let mut n = 0u32;
            criterion.bench_function(&format!("simplicial-st-{file}"), |bench| {
                bench.iter(|| {
                    n += 1;
                    symbolic.factorize_numeric_ldlt(
                        L_values.rb_mut(),
                        A,
                        side,
                        parallelism,
                        PodStack::new(&mut mem),
                    );
                });
            });
            start.elapsed() / n.max(1)
        };
        let supernodal = {
            let symbolic = factorize_symbolic(
                A.symbolic(),
                side,
                CholeskySymbolicParams {
                    supernodal_flop_ratio_threshold: 0.0,
                    ..Default::default()
                },
            )
            .unwrap();
            let parallelism = Parallelism::None;
            let mut mem = GlobalPodBuffer::new(
                symbolic
                    .factorize_numeric_ldlt_req::<f64>(side, parallelism)
                    .unwrap(),
            );
            let mut L_values = vec![0.0f64; symbolic.len_values()];
            let mut L_values = SliceGroupMut::new(&mut *L_values);

            let start = Instant::now();
            let mut n = 0u32;
            criterion.bench_function(&format!("supernodal-st-{file}"), |bench| {
                bench.iter(|| {
                    n += 1;
                    symbolic.factorize_numeric_ldlt(
                        L_values.rb_mut(),
                        A,
                        side,
                        parallelism,
                        PodStack::new(&mut mem),
                    );
                });
            });
            start.elapsed() / n.max(1)
        };
        let _supernodal_threaded = {
            let symbolic = factorize_symbolic(
                A.symbolic(),
                side,
                CholeskySymbolicParams {
                    supernodal_flop_ratio_threshold: 0.0,
                    ..Default::default()
                },
            )
            .unwrap();
            let parallelism = Parallelism::Rayon(0);
            let mut mem = GlobalPodBuffer::new(
                symbolic
                    .factorize_numeric_ldlt_req::<f64>(side, parallelism)
                    .unwrap(),
            );
            let mut L_values = vec![0.0f64; symbolic.len_values()];
            let mut L_values = SliceGroupMut::new(&mut *L_values);

            let start = Instant::now();
            let mut n = 0u32;
            criterion.bench_function(&format!("supernodal-mt-{file}"), |bench| {
                bench.iter(|| {
                    n += 1;
                    symbolic.factorize_numeric_ldlt(
                        L_values.rb_mut(),
                        A,
                        side,
                        parallelism,
                        PodStack::new(&mut mem),
                    );
                });
            });
            start.elapsed() / n.max(1)
        };
        if (symbolic_type == "supernodal") == (supernodal <= simplicial) {
            println!("GOOD");
        } else {
            println!("BAD");
        }
        println!();
    }
}

criterion_group!(benches, bench_ldlt);
criterion_main!(benches);
