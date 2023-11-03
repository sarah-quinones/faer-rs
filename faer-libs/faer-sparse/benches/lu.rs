#![allow(non_snake_case)]

use core::iter::zip;
use dyn_stack::{GlobalPodBuffer, PodStack, StackReq};
use faer_core::{
    permutation::PermutationRef,
    sparse::{SparseColMatRef, SymbolicSparseColMatRef},
};
use faer_entity::*;
use faer_sparse::{lu::supernodal::*, qr::col_etree, Index};
use matrix_market_rs::MtxData;
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

    for &[i, j] in &coo_indices {
        col_counts[j] += I(1);
        if i != j {
            col_counts[i] += I(1);
        }
    }

    for i in 0..n {
        col_ptr[i + 1] = col_ptr[i] + col_counts[i];
    }
    let nnz = col_ptr[n].zx();

    let mut row_ind = vec![I(0); nnz];
    let mut values = vec![0.0; nnz];

    col_counts.copy_from_slice(&col_ptr[..n]);

    for (&[i, j], &val) in zip(&coo_indices, &coo_values) {
        if i == j {
            values[col_counts[j].zx()] = 2.0 * val;
        } else {
            values[col_counts[i].zx()] = val;
            values[col_counts[j].zx()] = val;
        }

        row_ind[col_counts[j].zx()] = I(i);
        col_counts[j] += I(1);

        if i != j {
            row_ind[col_counts[i].zx()] = I(j);
            col_counts[i] += I(1);
        }
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

// fn permute_cols<'out, I: Index>(
//     new_col_ptrs: &'out mut [I],
//     new_row_ind: &'out mut [I],
//     new_val: &'out mut [f64],
//     A: SparseColMatRef<'_, I, f64>,
//     col_perm: PermutationRef<'_, I, f64>,
// ) -> SparseColMatRef<'out, I, f64> { let I = I::truncate;

//     let m = A.nrows();
//     let n = A.ncols();
//     new_col_ptrs[0] = I(0);
//     for j in 0..n {
//         let pj = col_perm.into_arrays().0[j].zx();
//         let row_ind = A.row_indices_of_col_raw(pj);
//         new_col_ptrs[j + 1] = new_col_ptrs[j] + I(row_ind.len());
//         new_row_ind[new_col_ptrs[j].zx()..new_col_ptrs[j + 1].zx()].copy_from_slice(row_ind);
//         new_val[new_col_ptrs[j].zx()..new_col_ptrs[j + 1].zx()]
//             .copy_from_slice(A.values_of_col(pj));
//     }

//     SparseColMatRef::new(
//         SymbolicSparseColMatRef::new_checked(m, n, new_col_ptrs, None, new_row_ind),
//         &*new_val,
//     )
// }

fn main() {
    let regexes = args()
        .skip(1)
        .filter(|x| !x.trim().starts_with('-'))
        .map(|s| Regex::new(&s).unwrap())
        .collect::<Vec<_>>();

    let matches = |s: &str| regexes.is_empty() || regexes.iter().any(|regex| regex.is_match(s));
    let time_limit = Duration::from_secs_f64(1.0);

    type I = usize;

    let mut files = Vec::new();

    for file in std::fs::read_dir("./bench_data").unwrap() {
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

    for file in files {
        if !matches(&file) {
            continue;
        }
        let path = "./bench_data/".to_string() + &*file + ".mtx";
        let Ok(data) = MtxData::<f64>::from_file(path) else {
            continue;
        };

        let (m, n, col_ptr, row_ind, values) = load_mtx::<I>(data);
        let nnz = values.len();
        dbg!(&file, m, n, nnz);

        let A = SparseColMatRef::<'_, I, f64>::new(
            SymbolicSparseColMatRef::new_checked(m, n, &col_ptr, None, &row_ind),
            &values,
        );

        let mut p = vec![0usize; n].into_boxed_slice();
        let mut p_inv = vec![0usize; n].into_boxed_slice();
        {
            let mut mem = GlobalPodBuffer::new(StackReq::new::<u8>(4 * 1024 * 1024 * 1024));

            faer_sparse::colamd::order(
                &mut p,
                &mut p_inv,
                *A,
                Default::default(),
                PodStack::new(&mut mem),
            )
            .unwrap();
        }
        let fill_col_perm = PermutationRef::<'_, I, Symbolic>::new_checked(&p, &p_inv);

        let mut etree = vec![0usize; n];
        let etree = {
            let mut mem = GlobalPodBuffer::new(StackReq::new::<u8>(4 * 1024 * 1024 * 1024));
            let nnz = A.compute_nnz();
            let mut new_col_ptrs = vec![0usize; m + 1];
            let mut new_row_ind = vec![0usize; nnz];
            let mut new_values = vec![0.0; nnz];
            let AT = faer_sparse::adjoint::<usize, f64>(
                &mut new_col_ptrs,
                &mut new_row_ind,
                &mut new_values,
                A,
                PodStack::new(&mut mem),
            );

            let mut post = vec![0usize; n];
            let mut col_counts = vec![0usize; n];
            let mut min_col = vec![0usize; m];

            let etree = col_etree(*A, Some(fill_col_perm), &mut etree, PodStack::new(&mut mem));
            faer_sparse::qr::postorder(&mut post, etree, PodStack::new(&mut mem));
            faer_sparse::qr::column_counts_aat(
                &mut col_counts,
                &mut min_col,
                *AT,
                Some(fill_col_perm),
                etree,
                &post,
                PodStack::new(&mut mem),
            );
            etree
        };

        let mut row_perm = vec![0; n];
        let mut row_perm_inv = vec![0; n];
        let mut col_perm = vec![0; n];
        let mut col_perm_inv = vec![0; n];

        let mut lu = SupernodalLuT2::<usize, f64>::new();
        let mut work = vec![];

        let mut mem = GlobalPodBuffer::new(StackReq::new::<u8>(4 * 1024 * 1024 * 1024));
        let mut op = |parallelism| {
            let _ = factorize_supernodal_numeric_dynamic_lu::<I, f64>(
                &mut row_perm,
                &mut row_perm_inv,
                &mut col_perm,
                &mut col_perm_inv,
                &mut lu,
                &mut work,
                A,
                fill_col_perm.cast(),
                etree,
                parallelism,
                PodStack::new(&mut mem),
                Default::default(),
            )
            .unwrap();
        };

        // warm up
        let warmup = time(|| op(faer_core::Parallelism::None));
        dbg!(warmup);
        {
            let single_thread = timeit(|| op(faer_core::Parallelism::None), time_limit);
            dbg!(single_thread);
        }

        {
            let multithread = timeit(|| op(faer_core::Parallelism::Rayon(0)), time_limit);
            dbg!(multithread);
        }
    }
}
