// PERF: optimize matmul
// - parallelization
// - simd(?)

use super::*;
use crate::{
    assert,
    mat::{As2D, As2DMut},
};
use core::cell::UnsafeCell;

/// Info about the matrix multiplication operation to help split the workload between multiple
/// threads.
#[derive(Clone, Debug)]
pub struct SparseMatmulInfo {
    flops_prefix_sum: alloc::vec::Vec<f64>,
}

/// Performs a symbolic matrix multiplication of a sparse matrix `lhs` by a sparse matrix `rhs`,
/// and returns the result.
///
/// # Note
/// Allows unsorted matrices, and produces a sorted output.
#[track_caller]
pub fn sparse_sparse_matmul_symbolic<I: Index>(
    lhs: SymbolicSparseColMatRef<'_, I>,
    rhs: SymbolicSparseColMatRef<'_, I>,
) -> Result<(SymbolicSparseColMat<I>, SparseMatmulInfo), FaerError> {
    assert!(lhs.ncols() == rhs.nrows());

    let m = lhs.nrows();
    let n = rhs.ncols();
    let mut col_ptrs = try_zeroed::<I>(n + 1)?;
    let mut row_ind = alloc::vec::Vec::new();

    let mut work = try_collect((0..m).into_iter().map(|_| I::truncate(usize::MAX)))?;
    let mut info = SparseMatmulInfo {
        flops_prefix_sum: try_zeroed::<f64>(n + 1)?,
    };

    for j in 0..n {
        let mut count = 0usize;
        let mut flops = 0.0f64;
        for k in rhs.row_indices_of_col(j) {
            for i in lhs.row_indices_of_col(k) {
                if work[i] != I::truncate(j) {
                    row_ind.try_reserve(1).map_err(|_| FaerError::OutOfMemory)?;
                    row_ind.push(I::truncate(i));
                    work[i] = I::truncate(j);
                    count += 1;
                }
            }
            flops += lhs.row_indices_of_col_raw(k).len() as f64;
        }
        info.flops_prefix_sum[j + 1] = info.flops_prefix_sum[j] + flops;

        col_ptrs[j + 1] = col_ptrs[j] + I::truncate(count);
        if col_ptrs[j + 1] > I::from_signed(I::Signed::MAX) {
            return Err(FaerError::IndexOverflow);
        }
        row_ind[col_ptrs[j].zx()..col_ptrs[j + 1].zx()].sort_unstable();
    }
    row_ind.shrink_to_fit();

    unsafe {
        Ok((
            SymbolicSparseColMat::new_unchecked(m, n, col_ptrs, None, row_ind),
            info,
        ))
    }
}

/// Computes the size and alignment of the workspace required to perform the numeric matrix
/// multiplication into `dst`.
pub fn sparse_sparse_matmul_numeric_req<I: Index, E: ComplexField>(
    dst: SymbolicSparseColMatRef<'_, I>,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    crate::linalg::temp_mat_req::<E>(
        dst.nrows(),
        crate::utils::thread::parallelism_degree(parallelism),
    )
}

/// Performs a numeric matrix multiplication of a sparse matrix `lhs` by a sparse matrix `rhs`
/// multiplied by `k`, and stores the result in `dst`.
///
/// # Note
/// `lhs` and `rhs` are allowed to be unsorted matrices.
#[track_caller]
pub fn sparse_sparse_matmul_numeric<
    I: Index,
    E: ComplexField,
    LhsE: Conjugate<Canonical = E>,
    RhsE: Conjugate<Canonical = E>,
>(
    dst: SparseColMatMut<'_, I, E>,
    lhs: SparseColMatRef<'_, I, LhsE>,
    rhs: SparseColMatRef<'_, I, RhsE>,
    k: E,
    info: &SparseMatmulInfo,
    parallelism: Parallelism,
    stack: &mut PodStack,
) {
    let (c_symbolic, c_values) = dst.parts_mut();
    {
        let c_values = SliceGroup::<'_, E>::new(E::faer_rb(E::faer_as_ref(&c_values)));
        assert!(all(
            lhs.nrows() == c_symbolic.nrows(),
            rhs.ncols() == c_symbolic.ncols(),
            lhs.ncols() == rhs.nrows(),
            c_values.len() == c_symbolic.row_indices().len(),
        ));
    }
    let m = lhs.nrows();
    let n = rhs.ncols();
    let total_flop_count = info.flops_prefix_sum[n];
    let par = if total_flop_count >= 512.0 * 512.0 {
        crate::utils::thread::parallelism_degree(parallelism)
    } else {
        1
    };
    let (work, _) = crate::linalg::temp_mat_zeroed::<E>(m, par, stack);
    let beta = k;

    struct SyncWrapper<T>(T);
    unsafe impl<T> Sync for SyncWrapper<T> {}

    let c_values = SyncWrapper(E::faer_map(c_values, |slice| {
        let len = slice.len();
        unsafe {
            core::slice::from_raw_parts(slice.as_mut_ptr() as *const UnsafeCell<E::Unit>, len)
        }
    }));

    crate::utils::thread::for_each_raw(
        par,
        |tid| {
            let mut work = crate::utils::slice::SliceGroupMut::<'_, E>::new(
                unsafe { work.rb().col(tid).const_cast() }
                    .try_as_slice_mut()
                    .unwrap(),
            );
            let col_start = info
                .flops_prefix_sum
                .partition_point(|&x| x < total_flop_count * (tid as f64 / par as f64));
            let col_end = col_start
                + info.flops_prefix_sum[col_start..]
                    .partition_point(|&x| x < total_flop_count * ((tid + 1) as f64 / par as f64));
            let c_values = &{ &c_values }.0;

            for j in col_start..col_end {
                for (k, b_k) in zip(
                    rhs.row_indices_of_col(j),
                    SliceGroup::<'_, RhsE>::new(rhs.values_of_col(j)).into_ref_iter(),
                ) {
                    let b_k = b_k.read().canonicalize().faer_mul(beta);
                    for (i, a_i) in zip(
                        lhs.row_indices_of_col(k),
                        SliceGroup::<'_, LhsE>::new(lhs.values_of_col(k)).into_ref_iter(),
                    ) {
                        let a_i = a_i.read().canonicalize();
                        work.write(i, work.read(i).faer_add(a_i.faer_mul(b_k)));
                    }
                }

                let range = c_symbolic.col_range(j);
                let start = range.start;
                let end = range.end;
                let c_values = SliceGroupMut::<'_, E>::new(E::faer_map(
                    E::faer_as_ref(c_values),
                    |c_values| {
                        let slice = &c_values[start..end];
                        unsafe {
                            core::slice::from_raw_parts_mut(
                                slice.as_ptr() as *mut E::Unit,
                                end - start,
                            )
                        }
                    },
                ));

                let mut i_prev = usize::MAX;
                for (i, mut c_i) in zip(c_symbolic.row_indices_of_col(j), c_values.into_mut_iter())
                {
                    if i != i_prev {
                        c_i.write(work.read(i));
                    } else {
                        c_i.write(E::faer_zero());
                    }
                    work.write(i, E::faer_zero());
                    i_prev = i;
                }
            }
        },
        parallelism,
    );
}

/// Multiplies a sparse matrix `lhs` by a sparse matrix `rhs`, multiplied by `k`, and returns
/// the result.
///
/// # Note
/// Allows unsorted matrices, and produces a sorted output.
#[track_caller]
pub fn sparse_sparse_matmul<
    I: Index,
    E: ComplexField,
    LhsE: Conjugate<Canonical = E>,
    RhsE: Conjugate<Canonical = E>,
>(
    lhs: SparseColMatRef<'_, I, LhsE>,
    rhs: SparseColMatRef<'_, I, RhsE>,
    k: E,
    parallelism: Parallelism,
) -> Result<SparseColMat<I, E>, FaerError> {
    assert!(lhs.ncols() == rhs.nrows());

    let (symbolic, info) = sparse_sparse_matmul_symbolic(lhs.symbolic(), rhs.symbolic())?;
    let mut values = VecGroup::<E>::new();
    values
        .try_reserve_exact(symbolic.row_indices().len())
        .map_err(|_| FaerError::OutOfMemory)?;
    values.resize(
        symbolic.row_indices().len(),
        E::faer_zero().faer_into_units(),
    );

    sparse_sparse_matmul_numeric(
        SparseColMatMut::new(symbolic.as_ref(), values.as_slice_mut().into_inner()),
        lhs,
        rhs,
        k,
        &info,
        parallelism,
        PodStack::new(
            &mut GlobalPodBuffer::try_new(
                sparse_sparse_matmul_numeric_req::<I, E>(symbolic.as_ref(), parallelism)
                    .map_err(|_| FaerError::OutOfMemory)?,
            )
            .map_err(|_| FaerError::OutOfMemory)?,
        ),
    );

    Ok(SparseColMat::<I, E>::new(symbolic, values.into_inner()))
}

/// Multiplies a sparse matrix `lhs` by a dense matrix `rhs`, and stores the result in
/// `acc`. See [`faer::linalg::matmul::matmul`](crate::linalg::matmul::matmul) for more details.
///
/// # Note
/// Allows unsorted matrices.
#[track_caller]
pub fn sparse_dense_matmul<
    I: Index,
    E: ComplexField,
    LhsE: Conjugate<Canonical = E>,
    RhsE: Conjugate<Canonical = E>,
>(
    acc: impl As2DMut<E>,
    lhs: SparseColMatRef<'_, I, LhsE>,
    rhs: impl As2D<RhsE>,
    alpha: Option<E>,
    beta: E,
    parallelism: Parallelism,
) {
    let mut acc = acc;
    let acc = acc.as_2d_mut();
    let rhs = rhs.as_2d_ref();

    #[track_caller]
    fn implementation<
        I: Index,
        E: ComplexField,
        LhsE: Conjugate<Canonical = E>,
        RhsE: Conjugate<Canonical = E>,
    >(
        acc: MatMut<'_, E>,
        lhs: SparseColMatRef<'_, I, LhsE>,
        rhs: MatRef<'_, RhsE>,
        alpha: Option<E>,
        beta: E,
        parallelism: Parallelism,
    ) {
        assert!(all(
            acc.nrows() == lhs.nrows(),
            acc.ncols() == rhs.ncols(),
            lhs.ncols() == rhs.nrows(),
        ));

        let _ = parallelism;
        let m = acc.nrows();
        let n = acc.ncols();
        let k = lhs.ncols();

        let mut acc = acc;

        match alpha {
            Some(alpha) => {
                if alpha != E::faer_one() {
                    zipped!(__rw, acc.rb_mut())
                        .for_each(|unzipped!(mut dst)| dst.write(dst.read().faer_mul(alpha)))
                }
            }
            None => acc.fill_zero(),
        }

        with_dim!(m, m);
        with_dim!(n, n);
        with_dim!(k, k);

        let mut acc = acc.as_shape_mut(m, n);
        let lhs = lhs.as_shape(m, k);
        let rhs = rhs.as_shape(k, n);

        for j in n.indices() {
            for depth in k.indices() {
                let rhs_kj = rhs.read(depth, j).canonicalize().faer_mul(beta);
                for (i, lhs_ik) in zip(
                    lhs.row_indices_of_col(depth),
                    SliceGroup::<'_, LhsE>::new(lhs.values_of_col(depth)).into_ref_iter(),
                ) {
                    acc.write(
                        i,
                        j,
                        acc.read(i, j)
                            .faer_add(lhs_ik.read().canonicalize().faer_mul(rhs_kj)),
                    );
                }
            }
        }
    }

    implementation(
        { acc }.as_2d_mut(),
        lhs,
        rhs.as_2d_ref(),
        alpha,
        beta,
        parallelism,
    )
}

/// Multiplies a dense matrix `lhs` by a sparse matrix `rhs`, and stores the result in
/// `acc`. See [`faer::linalg::matmul::matmul`](crate::linalg::matmul::matmul) for more details.
///
/// # Note
/// Allows unsorted matrices.
#[track_caller]
pub fn dense_sparse_matmul<
    I: Index,
    E: ComplexField,
    LhsE: Conjugate<Canonical = E>,
    RhsE: Conjugate<Canonical = E>,
>(
    acc: impl As2DMut<E>,
    lhs: impl As2D<LhsE>,
    rhs: SparseColMatRef<'_, I, RhsE>,
    alpha: Option<E>,
    beta: E,
    parallelism: Parallelism,
) {
    #[track_caller]
    fn implementation<
        I: Index,
        E: ComplexField,
        LhsE: Conjugate<Canonical = E>,
        RhsE: Conjugate<Canonical = E>,
    >(
        acc: MatMut<'_, E>,
        lhs: MatRef<'_, LhsE>,
        rhs: SparseColMatRef<'_, I, RhsE>,
        alpha: Option<E>,
        beta: E,
        parallelism: Parallelism,
    ) {
        assert!(all(
            acc.nrows() == lhs.nrows(),
            acc.ncols() == rhs.ncols(),
            lhs.ncols() == rhs.nrows(),
        ));

        let _ = parallelism;
        let m = acc.nrows();
        let n = acc.ncols();
        let k = lhs.ncols();

        let mut acc = acc;

        match alpha {
            Some(alpha) => {
                if alpha != E::faer_one() {
                    zipped!(__rw, acc.rb_mut())
                        .for_each(|unzipped!(mut dst)| dst.write(dst.read().faer_mul(alpha)))
                }
            }
            None => acc.fill_zero(),
        }

        with_dim!(m, m);
        with_dim!(n, n);
        with_dim!(k, k);
        let mut acc = acc.as_shape_mut(m, n);
        let lhs = lhs.as_shape(m, k);
        let rhs = rhs.as_shape(k, n);

        for i in m.indices() {
            for j in n.indices() {
                let mut acc_ij = E::faer_zero();
                for (depth, rhs_kj) in zip(
                    rhs.row_indices_of_col(j),
                    SliceGroup::<'_, RhsE>::new(rhs.values_of_col(j)).into_ref_iter(),
                ) {
                    let lhs_ik = lhs.read(i, depth);
                    acc_ij = acc_ij
                        .faer_add(lhs_ik.canonicalize().faer_mul(rhs_kj.read().canonicalize()));
                }

                acc.write(i, j, acc.read(i, j).faer_add(beta.faer_mul(acc_ij)));
            }
        }
    }
    implementation(
        { acc }.as_2d_mut(),
        lhs.as_2d_ref(),
        rhs,
        alpha,
        beta,
        parallelism,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use equator::assert;

    #[test]
    fn test_sp_matmul() {
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(
            5,
            4,
            &[
                (0, 0, 1.0),
                (1, 0, 2.0),
                (3, 0, 3.0),
                //
                (1, 1, 5.0),
                (4, 1, 6.0),
                //
                (0, 2, 7.0),
                (2, 2, 8.0),
                //
                (0, 3, 9.0),
                (2, 3, 10.0),
                (3, 3, 11.0),
                (4, 3, 12.0),
            ],
        )
        .unwrap();

        let b = SparseColMat::<usize, f64>::try_new_from_triplets(
            4,
            6,
            &[
                (0, 0, 1.0),
                (1, 0, 2.0),
                (3, 0, 3.0),
                //
                (1, 1, 5.0),
                (3, 1, 6.0),
                //
                (1, 2, 7.0),
                (3, 2, 8.0),
                //
                (1, 3, 9.0),
                (3, 3, 10.0),
                //
                (1, 4, 11.0),
                (3, 4, 12.0),
                //
                (1, 5, 13.0),
                (3, 5, 14.0),
            ],
        )
        .unwrap();

        let c = sparse_sparse_matmul(a.as_ref(), b.as_ref(), 2.0, Parallelism::Rayon(12)).unwrap();

        assert!(c.to_dense() == crate::scale(2.00) * a.to_dense() * b.to_dense());
    }
}
