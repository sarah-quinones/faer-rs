use super::*;
use crate::assert;

/// Returns the resulting matrix obtained by applying `f` to the elements from `lhs` and `rhs`,
/// skipping entries that are unavailable in both of `lhs` and `rhs`.
///
/// # Panics
/// Panics if `lhs` and `rhs` don't have matching dimensions.  
#[track_caller]
pub fn binary_op<I: Index, E: Entity, LhsE: Entity, RhsE: Entity>(
    lhs: SparseColMatRef<'_, I, LhsE>,
    rhs: SparseColMatRef<'_, I, RhsE>,
    f: impl FnMut(LhsE, RhsE) -> E,
) -> Result<SparseColMat<I, E>, FaerError> {
    assert!(lhs.nrows() == rhs.nrows());
    assert!(lhs.ncols() == rhs.ncols());
    let mut f = f;
    let m = lhs.nrows();
    let n = lhs.ncols();

    let mut col_ptrs = try_zeroed::<I>(n + 1)?;

    let mut nnz = 0usize;
    for j in 0..n {
        let lhs = lhs.row_indices_of_col_raw(j);
        let rhs = rhs.row_indices_of_col_raw(j);

        let mut lhs_pos = 0usize;
        let mut rhs_pos = 0usize;
        while lhs_pos < lhs.len() && rhs_pos < rhs.len() {
            let lhs = lhs[lhs_pos];
            let rhs = rhs[rhs_pos];

            lhs_pos += (lhs <= rhs) as usize;
            rhs_pos += (rhs <= lhs) as usize;
            nnz += 1;
        }
        nnz += lhs.len() - lhs_pos;
        nnz += rhs.len() - rhs_pos;
        col_ptrs[j + 1] = I::truncate(nnz);
    }

    if nnz > I::Signed::MAX.zx() {
        return Err(FaerError::IndexOverflow);
    }

    let mut row_indices = try_zeroed(nnz)?;
    let mut values = VecGroup::<E>::new();
    values
        .try_reserve_exact(nnz)
        .map_err(|_| FaerError::OutOfMemory)?;
    values.resize(nnz, unsafe { core::mem::zeroed() });

    let mut nnz = 0usize;
    for j in 0..n {
        let mut values = values.as_slice_mut();
        let lhs_values = SliceGroup::<LhsE>::new(lhs.values_of_col(j));
        let rhs_values = SliceGroup::<RhsE>::new(rhs.values_of_col(j));
        let lhs = lhs.row_indices_of_col_raw(j);
        let rhs = rhs.row_indices_of_col_raw(j);

        let mut lhs_pos = 0usize;
        let mut rhs_pos = 0usize;
        while lhs_pos < lhs.len() && rhs_pos < rhs.len() {
            let lhs = lhs[lhs_pos];
            let rhs = rhs[rhs_pos];

            match lhs.cmp(&rhs) {
                core::cmp::Ordering::Less => {
                    row_indices[nnz] = lhs;
                    values.write(
                        nnz,
                        f(lhs_values.read(lhs_pos), unsafe { core::mem::zeroed() }),
                    );
                }
                core::cmp::Ordering::Equal => {
                    row_indices[nnz] = lhs;
                    values.write(nnz, f(lhs_values.read(lhs_pos), rhs_values.read(rhs_pos)));
                }
                core::cmp::Ordering::Greater => {
                    row_indices[nnz] = rhs;
                    values.write(
                        nnz,
                        f(unsafe { core::mem::zeroed() }, rhs_values.read(rhs_pos)),
                    );
                }
            }

            lhs_pos += (lhs <= rhs) as usize;
            rhs_pos += (rhs <= lhs) as usize;
            nnz += 1;
        }
        row_indices[nnz..nnz + lhs.len() - lhs_pos].copy_from_slice(&lhs[lhs_pos..]);
        for (mut dst, src) in values
            .rb_mut()
            .subslice(nnz..nnz + lhs.len() - lhs_pos)
            .into_mut_iter()
            .zip(lhs_values.subslice(lhs_pos..lhs.len()).into_ref_iter())
        {
            dst.write(f(src.read(), unsafe { core::mem::zeroed() }));
        }
        nnz += lhs.len() - lhs_pos;

        row_indices[nnz..nnz + rhs.len() - rhs_pos].copy_from_slice(&rhs[rhs_pos..]);
        for (mut dst, src) in values
            .rb_mut()
            .subslice(nnz..nnz + rhs.len() - rhs_pos)
            .into_mut_iter()
            .zip(rhs_values.subslice(rhs_pos..rhs.len()).into_ref_iter())
        {
            dst.write(f(unsafe { core::mem::zeroed() }, src.read()));
        }
        nnz += rhs.len() - rhs_pos;
    }

    Ok(SparseColMat::<I, E>::new(
        SymbolicSparseColMat::<I>::new_checked(m, n, col_ptrs, None, row_indices),
        values.into_inner(),
    ))
}

/// Returns the resulting matrix obtained by applying `f` to the elements from `dst` and `src`
/// skipping entries that are unavailable in both of them.  
/// The sparsity patter of `dst` is unchanged.
///
/// # Panics
/// Panics if `src` and `dst` don't have matching dimensions.  
/// Panics if `src` contains an index that's unavailable in `dst`.  
#[track_caller]
pub fn binary_op_assign_into<I: Index, E: Entity, SrcE: Entity>(
    dst: SparseColMatMut<'_, I, E>,
    src: SparseColMatRef<'_, I, SrcE>,
    f: impl FnMut(E, SrcE) -> E,
) {
    {
        assert!(dst.nrows() == src.nrows());
        assert!(dst.ncols() == src.ncols());

        let n = dst.ncols();
        let mut dst = dst;
        let mut f = f;
        unsafe {
            assert!(f(core::mem::zeroed(), core::mem::zeroed()) == core::mem::zeroed());
        }

        for j in 0..n {
            let (dst, dst_val) = dst.rb_mut().into_parts_mut();

            let mut dst_val = SliceGroupMut::<E>::new(dst_val).subslice(dst.col_range(j));
            let src_val = SliceGroup::<SrcE>::new(src.values_of_col(j));

            let dst = dst.row_indices_of_col_raw(j);
            let src = src.row_indices_of_col_raw(j);

            let mut dst_pos = 0usize;
            let mut src_pos = 0usize;

            while src_pos < src.len() {
                let src = src[src_pos];

                if dst[dst_pos] < src {
                    dst_val.write(
                        dst_pos,
                        f(dst_val.read(dst_pos), unsafe { core::mem::zeroed() }),
                    );
                    dst_pos += 1;
                    continue;
                }

                assert!(dst[dst_pos] == src);

                dst_val.write(dst_pos, f(dst_val.read(dst_pos), src_val.read(src_pos)));

                src_pos += 1;
                dst_pos += 1;
            }
            while dst_pos < dst.len() {
                dst_val.write(
                    dst_pos,
                    f(dst_val.read(dst_pos), unsafe { core::mem::zeroed() }),
                );
                dst_pos += 1;
            }
        }
    }
}

/// Returns the resulting matrix obtained by applying `f` to the elements from `dst`, `lhs` and
/// `rhs`, skipping entries that are unavailable in all of `dst`, `lhs` and `rhs`.  
/// The sparsity patter of `dst` is unchanged.
///
/// # Panics
/// Panics if `lhs`, `rhs` and `dst` don't have matching dimensions.  
/// Panics if `lhs` or `rhs` contains an index that's unavailable in `dst`.  
#[track_caller]
pub fn ternary_op_assign_into<I: Index, E: Entity, LhsE: Entity, RhsE: Entity>(
    dst: SparseColMatMut<'_, I, E>,
    lhs: SparseColMatRef<'_, I, LhsE>,
    rhs: SparseColMatRef<'_, I, RhsE>,
    f: impl FnMut(E, LhsE, RhsE) -> E,
) {
    {
        assert!(dst.nrows() == lhs.nrows());
        assert!(dst.ncols() == lhs.ncols());
        assert!(dst.nrows() == rhs.nrows());
        assert!(dst.ncols() == rhs.ncols());

        let n = dst.ncols();
        let mut dst = dst;
        let mut f = f;
        unsafe {
            assert!(
                f(
                    core::mem::zeroed(),
                    core::mem::zeroed(),
                    core::mem::zeroed()
                ) == core::mem::zeroed()
            );
        }

        for j in 0..n {
            let (dst, dst_val) = dst.rb_mut().into_parts_mut();

            let mut dst_val = SliceGroupMut::<E>::new(dst_val).subslice(dst.col_range(j));
            let lhs_val = SliceGroup::<LhsE>::new(lhs.values_of_col(j));
            let rhs_val = SliceGroup::<RhsE>::new(rhs.values_of_col(j));

            let dst = dst.row_indices_of_col_raw(j);
            let rhs = rhs.row_indices_of_col_raw(j);
            let lhs = lhs.row_indices_of_col_raw(j);

            let mut dst_pos = 0usize;
            let mut lhs_pos = 0usize;
            let mut rhs_pos = 0usize;

            while lhs_pos < lhs.len() && rhs_pos < rhs.len() {
                let lhs = lhs[lhs_pos];
                let rhs = rhs[rhs_pos];

                if dst[dst_pos] < Ord::min(lhs, rhs) {
                    dst_val.write(
                        dst_pos,
                        f(
                            dst_val.read(dst_pos),
                            unsafe { core::mem::zeroed() },
                            unsafe { core::mem::zeroed() },
                        ),
                    );
                    dst_pos += 1;
                    continue;
                }

                assert!(dst[dst_pos] == Ord::min(lhs, rhs));

                match lhs.cmp(&rhs) {
                    core::cmp::Ordering::Less => {
                        dst_val.write(
                            dst_pos,
                            f(dst_val.read(dst_pos), lhs_val.read(lhs_pos), unsafe {
                                core::mem::zeroed()
                            }),
                        );
                    }
                    core::cmp::Ordering::Equal => {
                        dst_val.write(
                            dst_pos,
                            f(
                                dst_val.read(dst_pos),
                                lhs_val.read(lhs_pos),
                                rhs_val.read(rhs_pos),
                            ),
                        );
                    }
                    core::cmp::Ordering::Greater => {
                        dst_val.write(
                            dst_pos,
                            f(
                                dst_val.read(dst_pos),
                                unsafe { core::mem::zeroed() },
                                rhs_val.read(rhs_pos),
                            ),
                        );
                    }
                }

                lhs_pos += (lhs <= rhs) as usize;
                rhs_pos += (rhs <= lhs) as usize;
                dst_pos += 1;
            }
            while lhs_pos < lhs.len() {
                let lhs = lhs[lhs_pos];
                if dst[dst_pos] < lhs {
                    dst_val.write(
                        dst_pos,
                        f(
                            dst_val.read(dst_pos),
                            unsafe { core::mem::zeroed() },
                            unsafe { core::mem::zeroed() },
                        ),
                    );
                    dst_pos += 1;
                    continue;
                }
                dst_val.write(
                    dst_pos,
                    f(dst_val.read(dst_pos), lhs_val.read(lhs_pos), unsafe {
                        core::mem::zeroed()
                    }),
                );
                lhs_pos += 1;
                dst_pos += 1;
            }
            while rhs_pos < rhs.len() {
                let rhs = rhs[rhs_pos];
                if dst[dst_pos] < rhs {
                    dst_val.write(
                        dst_pos,
                        f(
                            dst_val.read(dst_pos),
                            unsafe { core::mem::zeroed() },
                            unsafe { core::mem::zeroed() },
                        ),
                    );
                    dst_pos += 1;
                    continue;
                }
                dst_val.write(
                    dst_pos,
                    f(
                        dst_val.read(dst_pos),
                        unsafe { core::mem::zeroed() },
                        rhs_val.read(rhs_pos),
                    ),
                );
                rhs_pos += 1;
                dst_pos += 1;
            }
            while rhs_pos < rhs.len() {
                let rhs = rhs[rhs_pos];
                dst_pos += dst[dst_pos..].binary_search(&rhs).unwrap();
                dst_val.write(
                    dst_pos,
                    f(
                        dst_val.read(dst_pos),
                        unsafe { core::mem::zeroed() },
                        rhs_val.read(rhs_pos),
                    ),
                );
                rhs_pos += 1;
            }
        }
    }
}

/// Returns the sparsity pattern containing the union of those of `lhs` and `rhs`.
///
/// # Panics
/// Panics if `lhs` and `rhs` don't have matching dimensions.  
#[track_caller]
#[inline]
pub fn union_symbolic<I: Index>(
    lhs: SymbolicSparseColMatRef<'_, I>,
    rhs: SymbolicSparseColMatRef<'_, I>,
) -> Result<SymbolicSparseColMat<I>, FaerError> {
    Ok(binary_op(
        SparseColMatRef::<I, Symbolic>::new(lhs, Symbolic::materialize(lhs.compute_nnz())),
        SparseColMatRef::<I, Symbolic>::new(rhs, Symbolic::materialize(rhs.compute_nnz())),
        #[inline(always)]
        |_, _| Symbolic,
    )?
    .into_parts()
    .0)
}

/// Returns the sum of `lhs` and `rhs`.
///
/// # Panics
/// Panics if `lhs` and `rhs` don't have matching dimensions.  
#[track_caller]
#[inline]
pub fn add<
    I: Index,
    E: ComplexField,
    LhsE: Conjugate<Canonical = E>,
    RhsE: Conjugate<Canonical = E>,
>(
    lhs: SparseColMatRef<'_, I, LhsE>,
    rhs: SparseColMatRef<'_, I, RhsE>,
) -> Result<SparseColMat<I, E>, FaerError> {
    binary_op(lhs, rhs, |lhs, rhs| {
        lhs.canonicalize().faer_add(rhs.canonicalize())
    })
}

/// Returns the difference of `lhs` and `rhs`.
///
/// # Panics
/// Panics if `lhs` and `rhs` don't have matching dimensions.  
#[track_caller]
#[inline]
pub fn sub<
    I: Index,
    LhsE: Conjugate<Canonical = E>,
    RhsE: Conjugate<Canonical = E>,
    E: ComplexField,
>(
    lhs: SparseColMatRef<'_, I, LhsE>,
    rhs: SparseColMatRef<'_, I, RhsE>,
) -> Result<SparseColMat<I, E>, FaerError> {
    binary_op(lhs, rhs, |lhs, rhs| {
        lhs.canonicalize().faer_sub(rhs.canonicalize())
    })
}

/// Computes the sum of `dst` and `src` and stores the result in `dst` without changing its
/// symbolic structure.
///
/// # Panics
/// Panics if `dst` and `rhs` don't have matching dimensions.  
/// Panics if `rhs` contains an index that's unavailable in `dst`.  
pub fn add_assign<I: Index, E: ComplexField, RhsE: Conjugate<Canonical = E>>(
    dst: SparseColMatMut<'_, I, E>,
    rhs: SparseColMatRef<'_, I, RhsE>,
) {
    binary_op_assign_into(dst, rhs, |dst, rhs| dst.faer_add(rhs.canonicalize()))
}

/// Computes the difference of `dst` and `src` and stores the result in `dst` without changing
/// its symbolic structure.
///
/// # Panics
/// Panics if `dst` and `rhs` don't have matching dimensions.  
/// Panics if `rhs` contains an index that's unavailable in `dst`.  
pub fn sub_assign<I: Index, E: ComplexField, RhsE: Conjugate<Canonical = E>>(
    dst: SparseColMatMut<'_, I, E>,
    rhs: SparseColMatRef<'_, I, RhsE>,
) {
    binary_op_assign_into(dst, rhs, |dst, rhs| dst.faer_sub(rhs.canonicalize()))
}

/// Computes the sum of `lhs` and `rhs`, storing the result in `dst` without changing its
/// symbolic structure.
///
/// # Panics
/// Panics if `dst`, `lhs` and `rhs` don't have matching dimensions.  
/// Panics if `lhs` or `rhs` contains an index that's unavailable in `dst`.  
#[track_caller]
#[inline]
pub fn add_into<
    I: Index,
    E: ComplexField,
    LhsE: Conjugate<Canonical = E>,
    RhsE: Conjugate<Canonical = E>,
>(
    dst: SparseColMatMut<'_, I, E>,
    lhs: SparseColMatRef<'_, I, LhsE>,
    rhs: SparseColMatRef<'_, I, RhsE>,
) {
    ternary_op_assign_into(dst, lhs, rhs, |_, lhs, rhs| {
        lhs.canonicalize().faer_add(rhs.canonicalize())
    })
}

/// Computes the difference of `lhs` and `rhs`, storing the result in `dst` without changing its
/// symbolic structure.
///
/// # Panics
/// Panics if `dst`, `lhs` and `rhs` don't have matching dimensions.  
/// Panics if `lhs` or `rhs` contains an index that's unavailable in `dst`.  
#[track_caller]
#[inline]
pub fn sub_into<
    I: Index,
    E: ComplexField,
    LhsE: Conjugate<Canonical = E>,
    RhsE: Conjugate<Canonical = E>,
>(
    dst: SparseColMatMut<'_, I, E>,
    lhs: SparseColMatRef<'_, I, LhsE>,
    rhs: SparseColMatRef<'_, I, RhsE>,
) {
    ternary_op_assign_into(dst, lhs, rhs, |_, lhs, rhs| {
        lhs.canonicalize().faer_sub(rhs.canonicalize())
    })
}
