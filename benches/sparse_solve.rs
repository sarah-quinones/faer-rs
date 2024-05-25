#![allow(non_snake_case)]
use diol::prelude::*;
use matrix_market_rs::MtxData;
use reborrow::ReborrowMut;

fn solve_simplicial(bencher: Bencher, (): ()) {
    let Ok(MtxData::Sparse([nrows, ncols], indices, data, _)) =
        MtxData::<f64>::from_file("./test_data/example_L.mtx")
    else {
        panic!()
    };
    let L = faer::sparse::SparseColMat::try_new_from_triplets(
        nrows,
        ncols,
        &indices
            .iter()
            .zip(data.iter())
            .map(|(&[row, col], &val)| (row, col, val))
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let mut x = vec![0.0; nrows];
    let mut x = faer::col::from_slice_mut(&mut x);
    bencher.bench(|| L.sp_solve_unit_lower_triangular_in_place(x.rb_mut()))
}

fn scale_simplicial(bencher: Bencher, (): ()) {
    let Ok(MtxData::Sparse([nrows, ncols], indices, data, _)) =
        MtxData::<f64>::from_file("./test_data/example_L.mtx")
    else {
        panic!()
    };
    let L = faer::sparse::SparseColMat::try_new_from_triplets(
        nrows,
        ncols,
        &indices
            .iter()
            .zip(data.iter())
            .map(|(&[row, col], &val)| (row, col, val))
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let mut x = vec![0.0; nrows];
    let mut x = faer::col::from_slice_mut(&mut x);
    bencher.bench(|| {
        for j in 0..nrows {
            let d_inv = L.values_of_col(j)[0].recip();
            x.write(j, x.read(j) * d_inv);
        }
    })
}

fn tsolve_simplicial(bencher: Bencher, (): ()) {
    let Ok(MtxData::Sparse([nrows, ncols], indices, data, _)) =
        MtxData::<f64>::from_file("./test_data/example_L.mtx")
    else {
        panic!()
    };
    let L = faer::sparse::SparseColMat::try_new_from_triplets(
        nrows,
        ncols,
        &indices
            .iter()
            .zip(data.iter())
            .map(|(&[row, col], &val)| (row, col, val))
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let mut x = vec![0.0; nrows];
    let mut x = faer::col::from_slice_mut(&mut x);
    bencher.bench(|| {
        L.transpose()
            .sp_solve_unit_upper_triangular_in_place(x.rb_mut())
    })
}

fn tsolve_scale_simplicial(bencher: Bencher, (): ()) {
    let Ok(MtxData::Sparse([nrows, ncols], indices, data, _)) =
        MtxData::<f64>::from_file("./test_data/example_L.mtx")
    else {
        panic!()
    };
    let L = faer::sparse::SparseColMat::try_new_from_triplets(
        nrows,
        ncols,
        &indices
            .iter()
            .zip(data.iter())
            .map(|(&[row, col], &val)| (row, col, val))
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let mut x = vec![0.0; nrows];
    let mut x = faer::col::from_slice_mut(&mut x);
    bencher.bench(|| {
        faer::sparse::linalg::triangular_solve::ldlt_scale_solve_unit_lower_triangular_transpose_in_place(L.as_ref(), faer::Conj::No, x.rb_mut().as_2d_mut(), faer::Parallelism::None);
    })
}

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register(solve_simplicial, [()]);
    bench.register(scale_simplicial, [()]);
    bench.register(tsolve_simplicial, [()]);
    bench.register(tsolve_scale_simplicial, [()]);
    bench.run()?;
    Ok(())
}
