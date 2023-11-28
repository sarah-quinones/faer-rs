#![allow(non_snake_case)]

use core::iter::zip;
use dyn_stack::{GlobalPodBuffer, PodStack};
use faer_core::{
    sparse::{SparseColMatRef, SymbolicSparseColMatRef},
    Parallelism, Side,
};
use faer_sparse::{cholesky::*, Index};
use matrix_market_rs::MtxData;
use reborrow::*;
use regex::Regex;
use std::{
    env::args,
    ffi::OsStr,
    time::{Duration, Instant},
};

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

        n_iters = 10 * (time_limit.as_secs_f64() / t.as_secs_f64()) as u32;
    }
}

fn main() {
    let regexes = args()
        .skip(1)
        .filter(|x| !x.trim().starts_with('-'))
        .map(|s| Regex::new(&s).unwrap())
        .collect::<Vec<_>>();

    let matches = |s: &str| regexes.is_empty() || regexes.iter().any(|regex| regex.is_match(s));

    let methods = [
        ("simplicial   ", f64::INFINITY, Parallelism::None),
        ("supernodal   ", 0.0, Parallelism::None),
        ("supernodal-bk", 0.0, Parallelism::None),
    ];

    let time_limit = Duration::from_secs_f64(0.1);

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
            files.push((name.strip_suffix(".mtx").unwrap().to_string(), Side::Upper))
        }
    }

    files.sort_by(|(f0, _), (f1, _)| str::cmp(f0, f1));

    for (file, side) in files {
        if !matches(&file) {
            continue;
        }
        if file.starts_with("chain") {
            continue;
        }
        let path = "./bench_data/".to_string() + &*file + ".mtx";
        let Ok(data) = MtxData::<f64>::from_file(path) else {
            continue;
        };

        let (n, col_ptr, row_ind, values) = load_mtx::<I>(data);
        let A = SparseColMatRef::<_, f64>::new(
            SymbolicSparseColMatRef::new_checked(n, n, &col_ptr, None, &row_ind),
            &*values,
        );

        let mut auto = "";
        time(|| {
            let symbolic_cholesky =
                &factorize_symbolic_cholesky(A.symbolic(), side, Default::default()).unwrap();
            auto = match symbolic_cholesky.raw() {
                SymbolicCholeskyRaw::Simplicial(_) => "simplicial",
                SymbolicCholeskyRaw::Supernodal(_) => "supernodal",
            };
            println!("picked {auto} method for {file}");
        });

        let times = methods.map(|(method, supernodal_flop_ratio_threshold, parallelism)| {
            let symbolic = factorize_symbolic_cholesky(
                A.symbolic(),
                side,
                CholeskySymbolicParams {
                    supernodal_flop_ratio_threshold,
                    ..Default::default()
                },
            )
            .unwrap();

            let mut mem = GlobalPodBuffer::new(
                symbolic
                    .factorize_numeric_ldlt_req::<f64>(false, parallelism)
                    .unwrap(),
            );
            let mut L_values = vec![0.0f64; symbolic.len_values()];
            let mut subdiag = vec![0.0f64; n];
            let mut fwd = vec![0; n];
            let mut inv = vec![0; n];
            let mut L_values = &mut *L_values;

            let f = || {
                if method == "supernodal-bk" {
                    symbolic.factorize_numeric_intranode_bunch_kaufman(
                        L_values.rb_mut(),
                        &mut *subdiag,
                        &mut fwd,
                        &mut inv,
                        A,
                        side,
                        Default::default(),
                        parallelism,
                        PodStack::new(&mut mem),
                    );
                } else {
                    symbolic.factorize_numeric_ldlt(
                        L_values.rb_mut(),
                        A,
                        side,
                        Default::default(),
                        parallelism,
                        PodStack::new(&mut mem),
                    );
                }
            };

            let time = timeit(f, time_limit);
            println!("{method}: {time:>35?}");
            (method, time)
        });
        let best = times[..2].iter().min_by_key(|(_, time)| time).unwrap();
        let worst = times[..2].iter().max_by_key(|(_, time)| time).unwrap();

        if best.0.trim_end() == auto {
            println!("good: {}", worst.1.as_secs_f64() / best.1.as_secs_f64());
        } else {
            println!("bad: {}", best.1.as_secs_f64() / worst.1.as_secs_f64());
        }
        println!();
    }
}
