#![allow(non_snake_case)]

use core::iter::zip;
use dyn_stack::{GlobalPodBuffer, PodStack, StackReq};
use faer_core::{
    permutation::PermutationRef,
    sparse::{SparseColMatRef, SymbolicSparseColMatRef},
};
use faer_entity::Symbolic;
use faer_sparse::{
    adjoint,
    qr::{
        col_etree, column_counts_aat, postorder,
        supernodal::{factorize_supernodal_numeric_qr, factorize_supernodal_symbolic_qr},
    },
    Index,
};
use matrix_market_rs::MtxData;
use reborrow::*;
use regex::Regex;
use std::{
    env::args,
    ffi::OsStr,
    time::{Duration, Instant},
};

fn load_mtx<I: Index>(data: MtxData<f64>) -> (usize, usize, Vec<I>, Vec<I>, Vec<f64>) {
    let I = I::truncate;

    let MtxData::Sparse([nrows, ncols], coo_indices, coo_values, _) = data else {
        panic!()
    };

    let m = nrows;
    let n = ncols;
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

    (m, n, col_ptr, row_ind, values)
}

fn time(mut f: impl FnMut()) -> Duration {
    let now = Instant::now();
    f();
    now.elapsed()
}

fn timeit(mut f: impl FnMut(), time_limit: Duration) -> Duration {
    let mut n_iters: u32 = 1;
    loop {
        let t = time(|| {
            for _ in 0..n_iters {
                f();
            }
        });

        if t >= time_limit || n_iters > 1_000_000_000 {
            return t / n_iters;
        }

        n_iters = 2 * Ord::max((time_limit.as_secs_f64() / t.as_secs_f64()) as u32, n_iters);
    }
}

fn main() {
    let regexes = args()
        .skip(1)
        .filter(|x| !x.trim().starts_with('-'))
        .map(|s| Regex::new(&s).unwrap())
        .collect::<Vec<_>>();

    let matches = |s: &str| regexes.is_empty() || regexes.iter().any(|regex| regex.is_match(s));
    let time_limit = Duration::from_secs_f64(1.0);

    type I = usize;
    let I = I::truncate;

    let mut files = Vec::new();

    for file in std::fs::read_dir("./bench_data/qr").unwrap() {
        let file = file.unwrap();
        if file.path().extension() == Some(OsStr::new("mtx")) {
            let name = file
                .path()
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned();
            files.push(name.strip_suffix(".mtx").unwrap().to_string())
        }
    }
    files.sort();

    let mut mem = GlobalPodBuffer::new(StackReq::new::<u8>(1024 * 1024 * 1024));

    for file in files {
        if !matches(&file) {
            continue;
        }
        let path = "./bench_data/qr/".to_string() + &*file + ".mtx";
        let Ok(data) = MtxData::<f64>::from_file(path) else {
            continue;
        };

        let (m, n, col_ptr, row_ind, values) = load_mtx::<I>(data);
        let nnz = row_ind.len();

        let A = SparseColMatRef::<'_, I, f64>::new(
            SymbolicSparseColMatRef::new_checked(m, n, &col_ptr, None, &row_ind),
            &values,
        );

        let zero = I(0);
        let mut new_col_ptrs = vec![zero; m + 1];
        let mut new_row_ind = vec![zero; nnz];
        let mut new_values = vec![0.0; nnz];

        let AT = adjoint::<I, f64>(
            &mut new_col_ptrs,
            &mut new_row_ind,
            &mut new_values,
            A,
            PodStack::new(&mut mem),
        )
        .into_const();

        let mut p = vec![0usize; n].into_boxed_slice();
        let mut p_inv = vec![0usize; n].into_boxed_slice();

        faer_sparse::colamd::order(
            &mut p,
            &mut p_inv,
            *A,
            Default::default(),
            PodStack::new(&mut mem),
        )
        .unwrap();

        let p = PermutationRef::<'_, I, Symbolic>::new_checked(&p, &p_inv);

        let mut etree = vec![zero; n];
        let mut post = vec![zero; n];
        let mut col_counts = vec![zero; n];
        let mut min_row = vec![zero; m];

        let etree = col_etree(*A, Some(p), &mut etree, PodStack::new(&mut mem));
        postorder(&mut post, etree, PodStack::new(&mut mem));

        column_counts_aat(
            &mut col_counts,
            &mut min_row,
            *AT,
            Some(p),
            etree,
            &post,
            PodStack::new(&mut mem),
        );

        let min_col = min_row;

        let symbolic = factorize_supernodal_symbolic_qr::<I>(
            *A,
            Some(p),
            min_col,
            etree,
            &col_counts,
            PodStack::new(&mut mem),
            Default::default(),
        )
        .unwrap();

        let householder_nnz = symbolic.householder().len_householder_row_indices();
        let mut row_indices_in_panel = vec![zero; householder_nnz];

        dbg!(&file);
        dbg!(m, n, A.compute_nnz());
        let mut L_values = vec![0.0; symbolic.r_adjoint().len_values()];
        let mut householder_values = vec![0.0; symbolic.householder().len_householder_values()];
        let mut tau_values = vec![0.0; symbolic.householder().len_tau_values()];

        let mut tau_blocksize = vec![I(0); n];
        let mut householder_nrows = vec![I(0); n];
        let mut householder_ncols = vec![I(0); n];

        let multithread = timeit(
            || {
                factorize_supernodal_numeric_qr::<I, f64>(
                    &mut row_indices_in_panel,
                    &mut tau_blocksize,
                    &mut householder_nrows,
                    &mut householder_ncols,
                    &mut L_values,
                    &mut householder_values,
                    &mut tau_values,
                    AT,
                    Some(p.cast()),
                    &symbolic,
                    faer_core::Parallelism::Rayon(0),
                    PodStack::new(&mut mem),
                );
            },
            time_limit,
        );
        dbg!(multithread);
        let single_thread = timeit(
            || {
                factorize_supernodal_numeric_qr::<I, f64>(
                    &mut row_indices_in_panel,
                    &mut tau_blocksize,
                    &mut householder_nrows,
                    &mut householder_ncols,
                    &mut L_values,
                    &mut householder_values,
                    &mut tau_values,
                    AT,
                    Some(p.cast()),
                    &symbolic,
                    faer_core::Parallelism::None,
                    PodStack::new(&mut mem),
                );
            },
            time_limit,
        );
        dbg!(single_thread);
    }
}
