use crate::{
    assert, debug_assert,
    linalg::{matmul::matmul, triangular_solve::solve_unit_lower_triangular_in_place},
    perm::PermRef,
    unzipped,
    utils::{simd::*, slice::*},
    zipped, ColMut, Index, MatMut, Parallelism, SignedIndex,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_entity::*;
use reborrow::*;

#[inline(always)]
fn swap_two_elems<E: ComplexField>(mut m: ColMut<'_, E>, i: usize, j: usize) {
    debug_assert!(i < m.nrows());
    debug_assert!(j < m.nrows());
    unsafe {
        let a = m.read_unchecked(i);
        let b = m.read_unchecked(j);
        m.write_unchecked(i, b);
        m.write_unchecked(j, a);
    }
}

#[inline(always)]
fn swap_two_elems_contiguous<E: ComplexField>(m: ColMut<'_, E>, i: usize, j: usize) {
    debug_assert!(m.row_stride() == 1);
    debug_assert!(i < m.nrows());
    debug_assert!(j < m.nrows());
    unsafe {
        let ptr = m.as_ptr_mut();
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

#[inline(never)]
fn lu_in_place_unblocked<E: ComplexField, I: Index>(
    mut matrix: MatMut<'_, E>,
    col_start: usize,
    n: usize,
    transpositions: &mut [I],
) -> usize {
    let m = matrix.nrows();
    let ncols = matrix.ncols();
    assert!(m >= n);

    let truncate = <I::Signed as SignedIndex>::truncate;

    if n == 0 {
        return 0;
    }

    let mut n_transpositions = 0;

    let arch = E::Simd::default();

    for (k, t) in transpositions.iter_mut().enumerate() {
        let mut imax = 0;
        {
            let col = k + col_start;
            let col = matrix.rb().col(col).subrows(k, m - k);
            let m = col.nrows();

            let mut max = E::Real::faer_zero();

            for i in 0..m {
                let abs = unsafe { col.read_unchecked(i) }.faer_score();
                if abs > max {
                    imax = i;
                    max = abs;
                }
            }

            imax += k;
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

        let (_, _, _, middle_right) = matrix.rb_mut().split_at_mut(0, col_start);
        let (_, _, middle, _) = middle_right.split_at_mut(0, n);
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
        let (_, top_right, bottom_left, bottom_right) = matrix.rb_mut().split_at_mut(j + 1, j + 1);
        let lhs = bottom_left.rb().col(j);
        let rhs = top_right.rb().row(j);
        let mut mat = bottom_right;

        let lhs = SliceGroup::<'_, E>::new(lhs.try_get_contiguous_col());

        let simd = SimdFor::<E, S>::new(simd);
        let offset = simd.align_offset(lhs);
        let (lhs_head, lhs_body, lhs_tail) = simd.as_aligned_simd(lhs, offset);

        for k in 0..mat.ncols() {
            let acc = SliceGroupMut::<'_, E>::new(mat.rb_mut().try_get_contiguous_col_mut(k));
            let rhs = simd.splat(rhs.read(k).faer_neg());
            let (acc_head, acc_body, acc_tail) = simd.as_aligned_simd_mut(acc, offset);

            #[inline(always)]
            fn process<E: ComplexField, S: pulp::Simd>(
                simd: SimdFor<E, S>,
                mut acc: impl Write<Output = SimdGroupFor<E, S>>,
                lhs: impl Read<Output = SimdGroupFor<E, S>>,
                rhs: SimdGroupFor<E, S>,
            ) {
                let zero = simd.splat(E::faer_zero());
                acc.write(simd.mul_add_e(rhs, lhs.read_or(zero), acc.read_or(zero)));
            }

            process(simd, acc_head, lhs_head, rhs);
            for (acc, lhs) in acc_body.into_mut_iter().zip(lhs_body.into_ref_iter()) {
                process(simd, acc, lhs, rhs);
            }
            process(simd, acc_tail, lhs_tail, rhs);
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
        let (_, top_right, bottom_left, bottom_right) = matrix.rb_mut().split_at_mut(j + 1, j + 1);
        let lhs = bottom_left.rb().col(j);
        let rhs = top_right.rb().row(j);
        let mut mat = bottom_right;

        for k in 0..mat.ncols() {
            let col = mat.rb_mut().col_mut(k);
            let rhs = rhs.read(k);
            zipped!(col, lhs).for_each(|unzipped!(mut x, lhs)| {
                x.write(x.read().faer_sub(lhs.read().faer_mul(rhs)))
            });
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

#[doc(hidden)]
pub fn lu_in_place_impl<I: Index, E: ComplexField>(
    mut matrix: MatMut<'_, E>,
    col_start: usize,
    n: usize,
    transpositions: &mut [I],
    parallelism: Parallelism,
) -> usize {
    let m = matrix.nrows();
    let full_n = matrix.ncols();

    debug_assert!(m >= n);

    if n <= recursion_threshold::<E>(m) {
        return lu_in_place_unblocked(matrix, col_start, n, transpositions);
    }

    // recursing is fine-ish since we halve the blocksize at each recursion step
    let bs = blocksize::<E>(n);

    let mut n_transpositions = 0;

    n_transpositions += lu_in_place_impl(
        matrix.rb_mut().submatrix_mut(0, col_start, m, n),
        0,
        bs,
        &mut transpositions[..bs],
        parallelism,
    );

    let (mat_top_left, mut mat_top_right, mat_bot_left, mut mat_bot_right) = matrix
        .rb_mut()
        .submatrix_mut(0, col_start, m, n)
        .split_at_mut(bs, bs);

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
        matrix.rb_mut().submatrix_mut(bs, col_start, m - bs, n),
        bs,
        n - bs,
        &mut transpositions[bs..],
        parallelism,
    );

    let parallelism = if m * (full_n - n) > 128 * 128 {
        parallelism
    } else {
        Parallelism::None
    };

    if matrix.row_stride() == 1 {
        crate::utils::thread::for_each_raw(
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
                let (_, mut col) = col.split_at_mut(bs);
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
        crate::utils::thread::for_each_raw(
            col_start + (full_n - (col_start + n)),
            |j| {
                let j = if j >= col_start { col_start + n + j } else { j };
                let mut col = unsafe { matrix.rb().col(j).const_cast() };
                for (i, &t) in transpositions[..bs].iter().enumerate() {
                    swap_two_elems(col.rb_mut(), i, t.to_signed().zx() + i.to_signed().zx());
                }
                let (_, mut col) = col.split_at_mut(bs);
                for (i, &t) in transpositions[bs..].iter().enumerate() {
                    swap_two_elems(col.rb_mut(), i, t.to_signed().zx() + i.to_signed().zx());
                }
            },
            parallelism,
        );
    }

    n_transpositions
}

/// LUfactorization tuning parameters.
#[derive(Default, Copy, Clone)]
#[non_exhaustive]
pub struct PartialPivLuComputeParams {}

/// Information about the resulting LU factorization.
#[derive(Copy, Clone, Debug)]
pub struct PartialPivLuInfo {
    /// Number of transpositions that were performed, can be used to compute the determinant of
    /// $P$.
    pub transposition_count: usize,
}

/// Computes the size and alignment of required workspace for performing an LU
/// decomposition with partial pivoting.
pub fn lu_in_place_req<I: Index, E: Entity>(
    m: usize,
    n: usize,
    parallelism: Parallelism,
    params: PartialPivLuComputeParams,
) -> Result<StackReq, SizeOverflow> {
    let _ = &params;
    let _ = &parallelism;

    let size = Ord::min(n, m);
    StackReq::try_new::<I>(size)
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
) -> (PartialPivLuInfo, PermRef<'out, I>) {
    let _ = &params;
    let truncate = <I::Signed as SignedIndex>::truncate;

    assert!(perm.len() == matrix.nrows());
    assert!(perm_inv.len() == matrix.nrows());

    #[cfg(feature = "perf-warn")]
    if (matrix.col_stride().unsigned_abs() == 1 || matrix.row_stride().unsigned_abs() != 1)
        && crate::__perf_warn!(LU_WARN)
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

    let (transpositions, _) = stack
        .rb_mut()
        .make_with(size, |_| I::from_signed(truncate(0)));
    let n_transpositions = lu_in_place_impl(matrix.rb_mut(), 0, size, transpositions, parallelism);

    for (idx, t) in transpositions.iter().enumerate() {
        perm.swap(idx, idx + t.to_signed().zx());
    }

    let (_, _, left, right) = matrix.split_at_mut(0, size);

    if m < n {
        solve_unit_lower_triangular_in_place(left.rb(), right, parallelism);
    }

    for (i, &p) in perm.iter().enumerate() {
        perm_inv[p.to_signed().zx()] = I::from_signed(truncate(i));
    }

    (
        PartialPivLuInfo {
            transposition_count: n_transpositions,
        },
        unsafe { PermRef::new_unchecked(perm, perm_inv) },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, linalg::lu::partial_pivoting::reconstruct, Mat, MatRef};
    use assert_approx_eq::assert_approx_eq;
    use dyn_stack::GlobalPodBuffer;
    use rand::random;

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::PodStack::new(&mut ::dyn_stack::GlobalPodBuffer::new($req.unwrap()))
        };
    }

    fn reconstruct_matrix<I: Index, E: ComplexField>(
        lu_factors: MatRef<'_, E>,
        row_perm: PermRef<'_, I>,
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
            let mut mat = mat.as_mut().reverse_rows_mut();
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
            let mut mat = mat.as_mut().transpose_mut();
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
