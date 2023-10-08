#![allow(non_snake_case)]

use core::iter::zip;
use criterion::*;
use dyn_stack::{GlobalPodBuffer, PodStack, StackReq};
use faer_core::Parallelism;
use faer_sparse_experimental::{
    amd::order_maybe_unsorted_req,
    cholesky::*,
    ghost::{self, Array},
    ghost_permute_symmetric, Index, PermutationRef, SliceGroup, SliceGroupMut, SparseColMatRef,
    SymbolicSparseColMatRef,
};
use matrix_market_rs::MtxData;

#[derive(Copy, Clone, Debug)]
enum Side {
    Upper,
    Lower,
}

fn load_mtx(data: MtxData<f64>, side: Side) -> (usize, Vec<i64>, Vec<i64>, Vec<f64>) {
    let MtxData::Sparse([nrows, _], coo_indices, coo_values, _) = data else {
        panic!()
    };

    let n = nrows;
    let mut col_counts = vec![0i64; n];
    let mut col_ptr = vec![0i64; n + 1];

    match side {
        Side::Upper => {
            for &[i, j] in &coo_indices {
                if i <= j {
                    col_counts[j] += 1;
                }
            }
        }
        Side::Lower => {
            for &[j, i] in &coo_indices {
                if i <= j {
                    col_counts[j] += 1;
                }
            }
        }
    }

    for i in 0..n {
        col_ptr[i + 1] = col_ptr[i] + col_counts[i];
    }
    let nnz = col_ptr[n] as usize;

    let mut row_ind = vec![0i64; nnz];
    let mut values = vec![0.0; nnz];

    col_counts.copy_from_slice(&col_ptr[..n]);

    match side {
        Side::Upper => {
            for (&[i, j], &val) in zip(&coo_indices, &coo_values) {
                if i <= j {
                    row_ind[col_counts[j] as usize] = i as i64;
                    values[col_counts[j] as usize] = val;
                    col_counts[j] += 1;
                }
            }
        }
        Side::Lower => {
            for (&[j, i], &val) in zip(&coo_indices, &coo_values) {
                if i <= j {
                    row_ind[col_counts[j] as usize] = i as i64;
                    values[col_counts[j] as usize] = val;
                    col_counts[j] += 1;
                }
            }
        }
    }

    (n, col_ptr, row_ind, values)
}

fn bench_supernodal(criterion: &mut Criterion) {
    type I = i32;
    let I = I::truncate;
    let zero = I(0);

    let files = [
        ("stiffness", Side::Upper),
        ("nd3k", Side::Lower),
        ("nd12k", Side::Lower),
    ];

    for (file, side) in files {
        let Ok(data) = MtxData::<f64>::from_file("./".to_string() + file + ".mtx") else {
            continue;
        };
        let (n, col_ptr, row_ind, values) = load_mtx(data, side);
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

        let (perm_ref, perm_inv_ref, _) = amd::order(
            I(n),
            A.col_ptrs(),
            A.row_indices(),
            &amd::Control::default(),
        )
        .unwrap();

        let mut perm = vec![zero; n];
        let mut perm_inv = vec![zero; n];

        faer_sparse_experimental::amd::order_maybe_unsorted(
            &mut perm,
            &mut perm_inv,
            A.symbolic(),
            faer_sparse_experimental::amd::Control::default(),
            PodStack::new(&mut GlobalPodBuffer::new(
                order_maybe_unsorted_req::<I>(n, values.len()).unwrap(),
            )),
        )
        .unwrap();

        assert2::assert!(perm == perm_ref);
        assert2::assert!(perm_inv == perm_inv_ref);

        let mut etree = vec![zero; n];
        let mut col_count = vec![zero; n];
        ghost::with_size(n, |N| {
            let A = ghost::SparseColMatRef::new(A, N, N);
            let perm = PermutationRef::new_checked(&perm, &perm_inv);
            let perm = ghost::PermutationRef::new(perm, N);

            let mut A_colptr = col_ptr.clone();
            let mut A_rowind = row_ind.clone();
            let mut A_values = values.clone();
            let A = ghost_permute_symmetric(
                SliceGroupMut::new(&mut *A_values),
                &mut A_colptr,
                &mut A_rowind,
                A,
                perm,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(n))),
            );

            let etree = ghost_prefactorize_symbolic(
                Array::from_mut(&mut etree, N),
                Array::from_mut(&mut col_count, N),
                A.symbolic(),
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(5 * n))),
            );

            let symbolic = ghost_factorize_supernodal_symbolic(
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

            let A_lower = ghost_transpose(
                &mut A_lower_col_ptr,
                &mut A_lower_row_ind,
                SliceGroupMut::new(&mut *A_lower_values),
                A,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
            );

            let n_supernodes = symbolic.n_supernodes();
            let mut values = vec![0.0f64; symbolic.col_ptrs_for_values()[n_supernodes].zx()];

            let parallelism = Parallelism::None;
            let mut mem = GlobalPodBuffer::new(
                factorize_supernodal_numeric_ldlt_req::<I, f64>(&symbolic, parallelism).unwrap(),
            );

            criterion.bench_function(&format!("supernodal-st-{file}"), |bench| {
                bench.iter(|| {
                    factorize_supernodal_numeric(
                        *A_lower,
                        SliceGroupMut::new(&mut *values),
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
                        SliceGroupMut::new(&mut *values),
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
