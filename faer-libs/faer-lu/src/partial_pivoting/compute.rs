#[cfg(feature = "std")]
use assert2::{assert, debug_assert};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_core::{
    mul::matmul,
    permutation::{Index, PermutationMut, SignedIndex},
    solve::solve_unit_lower_triangular_in_place,
    temp_mat_req, zipped, ComplexField, Entity, MatMut, Parallelism, SimdCtx,
};
use faer_entity::*;
use reborrow::*;

#[inline(always)]
fn swap_two_elems<E: ComplexField>(mut m: MatMut<'_, E>, i: usize, j: usize) {
    debug_assert!(m.ncols() == 1);
    debug_assert!(i < m.nrows());
    debug_assert!(j < m.nrows());
    unsafe {
        let a = m.read_unchecked(i, 0);
        let b = m.read_unchecked(j, 0);
        m.write_unchecked(i, 0, b);
        m.write_unchecked(j, 0, a);
    }
}

#[inline(always)]
fn swap_two_elems_contiguous<E: ComplexField>(m: MatMut<'_, E>, i: usize, j: usize) {
    debug_assert!(m.ncols() == 1);
    debug_assert!(m.row_stride() == 1);
    debug_assert!(i < m.nrows());
    debug_assert!(j < m.nrows());
    unsafe {
        let ptr = m.as_ptr();
        let ptr_a = E::faer_map(
            E::faer_copy(&ptr),
            #[inline(always)]
            |ptr| ptr.add(i),
        );
        let ptr_b = E::faer_map(
            E::faer_copy(&ptr),
            #[inline(always)]
            |ptr| ptr.add(j),
        );

        E::faer_map(
            E::faer_zip(ptr_a, ptr_b),
            #[inline(always)]
            |(a, b)| core::ptr::swap(a, b),
        );
    }
}

#[allow(clippy::extra_unused_type_parameters)]
fn lu_unblocked_req<E: Entity>(_m: usize, _n: usize) -> Result<StackReq, SizeOverflow> {
    Ok(StackReq::default())
}

#[inline(never)]
fn lu_in_place_unblocked<E: ComplexField, I: Index>(
    mut matrix: MatMut<'_, E>,
    col_start: usize,
    n: usize,
    transpositions: &mut [I],
    stack: PodStack<'_>,
) -> usize {
    let _ = &stack;
    let m = matrix.nrows();
    let ncols = matrix.ncols();
    debug_assert!(m >= n);

    let truncate = <I::Signed as SignedIndex>::truncate;

    if n == 0 {
        return 0;
    }

    let mut n_transpositions = 0;

    let arch = E::Simd::default();

    for (k, t) in transpositions.iter_mut().enumerate() {
        let imax;
        {
            let col = k + col_start;
            let col = matrix.rb().col(col).subrows(k, m - k);
            let m = col.nrows();

            let mut imax0 = 0;
            let mut imax1 = 0;
            let mut max0 = E::Real::faer_zero();
            let mut max1 = E::Real::faer_zero();

            for i in 0..m / 2 {
                let i = 2 * i;

                let abs0 = unsafe { col.read_unchecked(i, 0) }.faer_score();
                let abs1 = unsafe { col.read_unchecked(i + 1, 0) }.faer_score();

                if abs0 > max0 {
                    imax0 = i;
                    max0 = abs0;
                }
                if abs1 > max1 {
                    imax1 = i + 1;
                    max1 = abs1;
                }
            }

            if m % 2 != 0 {
                let i = m - 1;
                let abs0 = unsafe { col.read_unchecked(i, 0) }.faer_score();
                if abs0 > max0 {
                    imax0 = i;
                    max0 = abs0;
                }
            }

            if max0 > max1 {
                imax = imax0 + k;
            } else {
                imax = imax1 + k;
            }
        }

        *t = I::from_signed(truncate(imax - k));

        if imax != k {
            n_transpositions += 1;
        }

        if k != imax {
            for j in 0..ncols {
                unsafe {
                    let mk = matrix.read_unchecked(k, j);
                    let mi = matrix.read_unchecked(imax, j);
                    matrix.write_unchecked(k, j, mi);
                    matrix.write_unchecked(imax, j, mk);
                }
            }
        }

        let [_, _, _, middle_right] = matrix.rb_mut().split_at(0, col_start);
        let [_, _, middle, _] = middle_right.split_at(0, n);
        update(arch, middle, k);
    }

    n_transpositions
}

struct Update<'a, E: ComplexField> {
    matrix: MatMut<'a, E>,
    j: usize,
}

impl<E: ComplexField> pulp::WithSimd for Update<'_, E> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let Self { mut matrix, j } = self;

        debug_assert_eq!(matrix.row_stride(), 1);

        let m = matrix.nrows();
        let inv = matrix.read(j, j).faer_inv();
        for i in j + 1..m {
            unsafe {
                matrix.write_unchecked(i, j, matrix.read_unchecked(i, j).faer_mul(inv));
            }
        }
        let [_, top_right, bottom_left, bottom_right] = matrix.rb_mut().split_at(j + 1, j + 1);
        let lhs = bottom_left.rb().col(j);
        let rhs = top_right.rb().row(j);
        let mut mat = bottom_right;

        let m = mat.nrows();
        let lhs = E::faer_map(
            lhs.as_ptr(),
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts(ptr, m) },
        );

        let lane_count =
            core::mem::size_of::<SimdUnitFor<E, S>>() / core::mem::size_of::<UnitFor<E>>();

        let prefix = m % lane_count;

        let (lhs_head, lhs_tail) = E::faer_unzip(E::faer_map(
            lhs,
            #[inline(always)]
            |slice| slice.split_at(prefix),
        ));
        let lhs_tail = faer_core::simd::slice_as_simd::<E, S>(lhs_tail).0;

        for k in 0..mat.ncols() {
            let acc = E::faer_map(
                mat.rb_mut().ptr_at(0, k),
                #[inline(always)]
                |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, m) },
            );
            let (acc_head, acc_tail) = E::faer_unzip(E::faer_map(
                acc,
                #[inline(always)]
                |slice| slice.split_at_mut(prefix),
            ));
            let acc_tail = faer_core::simd::slice_as_mut_simd::<E, S>(acc_tail).0;

            let rhs = E::faer_simd_splat(simd, unsafe { rhs.read_unchecked(0, k).faer_neg() });

            let mut acc_head_ =
                E::faer_partial_load_last(simd, E::faer_rb(E::faer_as_ref(&acc_head)));
            acc_head_ = E::faer_simd_mul_adde(
                simd,
                E::faer_copy(&rhs),
                E::faer_partial_load_last(simd, E::faer_copy(&lhs_head)),
                acc_head_,
            );
            E::faer_partial_store_last(simd, acc_head, acc_head_);

            for (acc, lhs) in
                E::faer_into_iter(acc_tail).zip(E::faer_into_iter(E::faer_copy(&lhs_tail)))
            {
                let mut acc_ = E::faer_deref(E::faer_rb(E::faer_as_ref(&acc)));
                let lhs = E::faer_deref(lhs);
                acc_ = E::faer_simd_mul_adde(simd, E::faer_copy(&rhs), lhs, acc_);
                E::faer_map(
                    E::faer_zip(acc, acc_),
                    #[inline(always)]
                    |(acc, acc_)| *acc = acc_,
                );
            }
        }
    }
}

fn update<E: ComplexField>(arch: E::Simd, mut matrix: MatMut<E>, j: usize) {
    if matrix.row_stride() == 1 {
        arch.dispatch(Update { matrix, j });
    } else {
        let m = matrix.nrows();
        let inv = matrix.read(j, j).faer_inv();
        for i in j + 1..m {
            matrix.write(i, j, matrix.read(i, j).faer_mul(inv));
        }
        let [_, top_right, bottom_left, bottom_right] = matrix.rb_mut().split_at(j + 1, j + 1);
        let lhs = bottom_left.rb().col(j);
        let rhs = top_right.rb().row(j);
        let mut mat = bottom_right;

        for k in 0..mat.ncols() {
            let col = mat.rb_mut().col(k);
            let rhs = rhs.read(0, k);
            zipped!(col, lhs)
                .for_each(|mut x, lhs| x.write(x.read().faer_sub(lhs.read().faer_mul(rhs))));
        }
    }
}

#[allow(clippy::extra_unused_type_parameters)]
fn recursion_threshold<E: Entity>(_m: usize) -> usize {
    16
}

#[inline]
#[allow(clippy::extra_unused_type_parameters)]
// we want remainder to be a multiple of register size
fn blocksize<E: Entity>(n: usize) -> usize {
    let base_rem = n / 2;
    n - if n >= 32 {
        (base_rem + 15) / 16 * 16
    } else if n >= 16 {
        (base_rem + 7) / 8 * 8
    } else if n >= 8 {
        (base_rem + 3) / 4 * 4
    } else {
        base_rem
    }
}

fn lu_recursive_req<I: Index, E: Entity>(
    m: usize,
    n: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    if n <= recursion_threshold::<E>(m) {
        return lu_unblocked_req::<E>(m, n);
    }

    let bs = blocksize::<E>(n);
    let _ = parallelism;

    StackReq::try_any_of([
        lu_recursive_req::<I, E>(m, bs, parallelism)?,
        StackReq::try_all_of([
            StackReq::try_new::<I>(m - bs)?,
            lu_recursive_req::<I, E>(m - bs, n - bs, parallelism)?,
        ])?,
        temp_mat_req::<E>(m, 1)?,
    ])
}

fn lu_in_place_impl<I: Index, E: ComplexField>(
    mut matrix: MatMut<'_, E>,
    col_start: usize,
    n: usize,
    transpositions: &mut [I],
    parallelism: Parallelism,
    mut stack: PodStack<'_>,
) -> usize {
    let m = matrix.nrows();
    let full_n = matrix.ncols();

    debug_assert!(m >= n);

    if n <= recursion_threshold::<E>(m) {
        return lu_in_place_unblocked(matrix, col_start, n, transpositions, stack);
    }

    // recursing is fine-ish since we halve the blocksize at each recursion step
    let bs = blocksize::<E>(n);

    let mut n_transpositions = 0;

    n_transpositions += lu_in_place_impl(
        matrix.rb_mut().submatrix(0, col_start, m, n),
        0,
        bs,
        &mut transpositions[..bs],
        parallelism,
        stack.rb_mut(),
    );

    let [mat_top_left, mut mat_top_right, mat_bot_left, mut mat_bot_right] = matrix
        .rb_mut()
        .submatrix(0, col_start, m, n)
        .split_at(bs, bs);

    solve_unit_lower_triangular_in_place(mat_top_left.rb(), mat_top_right.rb_mut(), parallelism);
    matmul(
        mat_bot_right.rb_mut(),
        mat_bot_left.rb(),
        mat_top_right.rb(),
        Some(E::faer_one()),
        E::faer_one().faer_neg(),
        parallelism,
    );

    n_transpositions += lu_in_place_impl(
        matrix.rb_mut().submatrix(bs, col_start, m - bs, n),
        bs,
        n - bs,
        &mut transpositions[bs..],
        parallelism,
        stack.rb_mut(),
    );

    let parallelism = if m * (full_n - n) > 128 * 128 {
        parallelism
    } else {
        Parallelism::None
    };

    if matrix.row_stride() == 1 {
        faer_core::for_each_raw(
            col_start + (full_n - (col_start + n)),
            |j| {
                let j = if j >= col_start { col_start + n + j } else { j };
                let mut col = unsafe { matrix.rb().col(j).const_cast() };
                for (i, &t) in transpositions[..bs].iter().enumerate() {
                    swap_two_elems_contiguous(
                        col.rb_mut(),
                        i,
                        t.to_signed().zx() + i.to_signed().zx(),
                    );
                }
                let [_, mut col] = col.split_at_row(bs);
                for (i, &t) in transpositions[bs..].iter().enumerate() {
                    swap_two_elems_contiguous(
                        col.rb_mut(),
                        i,
                        t.to_signed().zx() + i.to_signed().zx(),
                    );
                }
            },
            parallelism,
        );
    } else {
        faer_core::for_each_raw(
            col_start + (full_n - (col_start + n)),
            |j| {
                let j = if j >= col_start { col_start + n + j } else { j };
                let mut col = unsafe { matrix.rb().col(j).const_cast() };
                for (i, &t) in transpositions[..bs].iter().enumerate() {
                    swap_two_elems(col.rb_mut(), i, t.to_signed().zx() + i.to_signed().zx());
                }
                let [_, mut col] = col.split_at_row(bs);
                for (i, &t) in transpositions[bs..].iter().enumerate() {
                    swap_two_elems(col.rb_mut(), i, t.to_signed().zx() + i.to_signed().zx());
                }
            },
            parallelism,
        );
    }

    n_transpositions
}

#[derive(Default, Copy, Clone)]
#[non_exhaustive]
pub struct PartialPivLuComputeParams {}

/// Computes the size and alignment of required workspace for performing an LU
/// decomposition with partial pivoting.
pub fn lu_in_place_req<I: Index, E: Entity>(
    m: usize,
    n: usize,
    parallelism: Parallelism,
    params: PartialPivLuComputeParams,
) -> Result<StackReq, SizeOverflow> {
    let _ = &params;

    let size = Ord::min(n, m);
    StackReq::try_all_of([
        StackReq::try_new::<I>(size)?,
        lu_recursive_req::<I, E>(m, size, parallelism)?,
    ])
}

/// Computes the LU decomposition of the given matrix with partial pivoting, replacing the matrix
/// with its factors in place.
///
/// The decomposition is such that:
/// $$PA = LU,$$
/// where $P$ is a permutation matrix, $L$ is a unit lower triangular matrix, and $U$ is an upper
/// triangular matrix.
///
/// $L$ is stored in the strictly lower triangular half of `matrix`, with an implicit unit
/// diagonal, $U$ is stored in the upper triangular half of `matrix`, and the permutation
/// representing $P$, as well as its inverse, are stored in `perm` and `perm_inv` respectively.
///
/// After the function returns, `perm` contains the order of the rows after pivoting, i.e. the
/// result is the same as computing the non-pivoted LU decomposition of the matrix `matrix[perm,
/// :]`. `perm_inv` contains its inverse permutation.
///
/// # Output
///
/// - The number of transpositions that constitute the permutation,
/// - a structure representing the permutation $P$.
///
/// # Panics
///
/// - Panics if the length of the permutation slices is not equal to the number of rows of the
/// matrix.
/// - Panics if the provided memory in `stack` is insufficient (see [`lu_in_place_req`]).
pub fn lu_in_place<'out, I: Index, E: ComplexField>(
    matrix: MatMut<'_, E>,
    perm: &'out mut [I],
    perm_inv: &'out mut [I],
    parallelism: Parallelism,
    stack: PodStack<'_>,
    params: PartialPivLuComputeParams,
) -> (usize, PermutationMut<'out, I, E>) {
    let _ = &params;
    let truncate = <I::Signed as SignedIndex>::truncate;

    assert!(perm.len() == matrix.nrows());
    assert!(perm_inv.len() == matrix.nrows());

    #[cfg(feature = "perf-warn")]
    if (matrix.col_stride().unsigned_abs() == 1 || matrix.row_stride().unsigned_abs() != 1)
        && faer_core::__perf_warn!(LU_WARN)
    {
        log::warn!(target: "faer_perf", "LU with partial pivoting prefers column-major or row-major matrix. Found matrix with generic strides.");
    }

    let mut matrix = matrix;
    let mut stack = stack;
    let m = matrix.nrows();
    let n = matrix.ncols();
    let size = Ord::min(n, m);

    for (i, p) in perm.iter_mut().enumerate() {
        *p = I::from_signed(truncate(i));
    }

    let (transpositions, mut stack) = stack
        .rb_mut()
        .make_with(size, |_| I::from_signed(truncate(0)));
    let n_transpositions = lu_in_place_impl(
        matrix.rb_mut(),
        0,
        size,
        transpositions,
        parallelism,
        stack.rb_mut(),
    );

    for (idx, t) in transpositions.iter().enumerate() {
        perm.swap(idx, idx + t.to_signed().zx());
    }

    let [_, _, left, right] = matrix.split_at(0, size);

    if m < n {
        solve_unit_lower_triangular_in_place(left.rb(), right, parallelism);
    }

    for (i, &p) in perm.iter().enumerate() {
        perm_inv[p.to_signed().zx()] = I::from_signed(truncate(i));
    }

    (n_transpositions, unsafe {
        PermutationMut::new_unchecked(perm, perm_inv)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partial_pivoting::reconstruct;
    use assert2::assert;
    use assert_approx_eq::assert_approx_eq;
    use dyn_stack::GlobalPodBuffer;
    use faer_core::{permutation::PermutationRef, Mat, MatRef};
    use rand::random;

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::PodStack::new(&mut ::dyn_stack::GlobalPodBuffer::new($req.unwrap()))
        };
    }

    fn reconstruct_matrix<I: Index, E: ComplexField>(
        lu_factors: MatRef<'_, E>,
        row_perm: PermutationRef<'_, I, E>,
    ) -> Mat<E> {
        let m = lu_factors.nrows();
        let n = lu_factors.ncols();
        let mut dst = Mat::zeros(m, n);
        reconstruct::reconstruct(
            dst.as_mut(),
            lu_factors,
            row_perm,
            Parallelism::Rayon(0),
            make_stack!(reconstruct::reconstruct_req::<I, E>(
                m,
                n,
                Parallelism::Rayon(0)
            )),
        );
        dst
    }

    #[test]
    fn compute_lu() {
        for (m, n) in [
            (10, 10),
            (4, 4),
            (2, 4),
            (2, 20),
            (2, 2),
            (20, 20),
            (4, 2),
            (20, 2),
            (40, 20),
            (20, 40),
            (40, 60),
            (60, 40),
            (200, 100),
            (100, 200),
            (200, 200),
        ] {
            let mut mat = Mat::from_fn(m, n, |_, _| random::<f64>());
            let mat_orig = mat.clone();
            let mut perm = vec![0usize; m];
            let mut perm_inv = vec![0; m];

            let mut mem = GlobalPodBuffer::new(
                lu_in_place_req::<usize, f64>(m, n, Parallelism::Rayon(8), Default::default())
                    .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);

            let (_, row_perm) = lu_in_place(
                mat.as_mut(),
                &mut perm,
                &mut perm_inv,
                Parallelism::Rayon(8),
                stack.rb_mut(),
                Default::default(),
            );
            let reconstructed = reconstruct_matrix(mat.as_ref(), row_perm.rb());

            for i in 0..m {
                for j in 0..n {
                    assert_approx_eq!(mat_orig.read(i, j), reconstructed.read(i, j));
                }
            }
        }
    }

    #[test]
    fn compute_lu_non_contiguous() {
        for (m, n) in [
            (10, 10),
            (4, 4),
            (2, 4),
            (2, 20),
            (2, 2),
            (20, 20),
            (4, 2),
            (20, 2),
            (40, 20),
            (20, 40),
            (40, 60),
            (60, 40),
            (200, 100),
            (100, 200),
            (200, 200),
        ] {
            let mut mat = Mat::from_fn(m, n, |_, _| random::<f64>());
            let mut mat = mat.as_mut().reverse_rows();
            let mat_orig = mat.to_owned();
            let mut perm = vec![0usize; m];
            let mut perm_inv = vec![0; m];

            let mut mem = GlobalPodBuffer::new(
                lu_in_place_req::<usize, f64>(m, n, Parallelism::Rayon(8), Default::default())
                    .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);

            let (_, row_perm) = lu_in_place(
                mat.rb_mut(),
                &mut perm,
                &mut perm_inv,
                Parallelism::Rayon(8),
                stack.rb_mut(),
                Default::default(),
            );
            let reconstructed = reconstruct_matrix(mat.rb(), row_perm.rb());

            for i in 0..m {
                for j in 0..n {
                    assert_approx_eq!(mat_orig.read(i, j), reconstructed.read(i, j));
                }
            }
        }
    }

    #[test]
    fn compute_lu_row_major() {
        for (m, n) in [
            (3, 3),
            (2, 2),
            (4, 2),
            (2, 4),
            (4, 4),
            (10, 10),
            (2, 20),
            (20, 20),
            (20, 2),
            (40, 20),
            (20, 40),
            (40, 60),
            (60, 40),
            (200, 100),
            (100, 200),
            (200, 200),
        ] {
            let mut mat = Mat::from_fn(n, m, |_, _| random::<f64>());
            let mut mat = mat.as_mut().transpose();
            let mat_orig = mat.to_owned();
            let mut perm = vec![0usize; m];
            let mut perm_inv = vec![0; m];

            let mut mem = GlobalPodBuffer::new(
                lu_in_place_req::<usize, f64>(m, n, Parallelism::Rayon(8), Default::default())
                    .unwrap(),
            );
            let mut stack = PodStack::new(&mut mem);

            let (_, row_perm) = lu_in_place(
                mat.rb_mut(),
                &mut perm,
                &mut perm_inv,
                Parallelism::Rayon(8),
                stack.rb_mut(),
                Default::default(),
            );
            let reconstructed = reconstruct_matrix(mat.rb(), row_perm.rb());

            for i in 0..m {
                for j in 0..n {
                    assert_approx_eq!(mat_orig.read(i, j), reconstructed.read(i, j));
                }
            }
        }
    }
}
