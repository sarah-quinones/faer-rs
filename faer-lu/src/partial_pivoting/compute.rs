use assert2::{assert, debug_assert};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_core::{
    mul::matmul,
    permutation::{swap_rows, PermutationMut},
    solve::solve_unit_lower_triangular_in_place,
    temp_mat_req, zipped, ComplexField, Entity, MatMut, Parallelism,
};
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
fn swap_two_elems_contiguous<E: ComplexField>(mut m: MatMut<'_, E>, i: usize, j: usize) {
    debug_assert!(m.ncols() == 1);
    debug_assert!(m.row_stride() == 1);
    debug_assert!(i < m.nrows());
    debug_assert!(j < m.nrows());
    unsafe {
        let ptr = m.rb_mut().as_ptr();

        let ptr_a = E::map(
            E::copy(&ptr),
            #[inline(always)]
            |ptr| ptr.add(i),
        );
        let ptr_b = E::map(
            E::copy(&ptr),
            #[inline(always)]
            |ptr| ptr.add(j),
        );

        let a = E::map(
            E::copy(&ptr_a),
            #[inline(always)]
            |ptr| (*ptr),
        );
        let b = E::map(
            E::copy(&ptr_b),
            #[inline(always)]
            |ptr| (*ptr),
        );

        E::map(
            E::zip(ptr_b, a),
            #[inline(always)]
            |(ptr, val)| *ptr = val,
        );
        E::map(
            E::zip(ptr_a, b),
            #[inline(always)]
            |(ptr, val)| *ptr = val,
        );
    }
}

#[allow(clippy::extra_unused_type_parameters)]
fn lu_unblocked_req<E: Entity>(_m: usize, _n: usize) -> Result<StackReq, SizeOverflow> {
    Ok(StackReq::default())
}

#[inline(never)]
fn lu_in_place_unblocked<E: ComplexField>(
    mut matrix: MatMut<'_, E>,
    col_start: usize,
    n: usize,
    perm: &mut [usize],
    transpositions: &mut [usize],
    stack: PodStack<'_>,
) -> usize {
    let _ = &stack;
    let m = matrix.nrows();
    let ncols = matrix.ncols();
    debug_assert!(m >= n);
    debug_assert!(perm.len() == m);

    if n == 0 {
        return 0;
    }

    let mut n_transpositions = 0;

    let arch = pulp::Arch::new();

    for (k, t) in transpositions.iter_mut().enumerate() {
        let imax;
        {
            let col = k + col_start;
            let col = matrix.rb().col(col).subrows(k, m - k);
            let m = col.nrows();

            let mut imax0 = 0;
            let mut imax1 = 0;
            let mut max0 = E::Real::zero();
            let mut max1 = E::Real::zero();

            for i in 0..m / 2 {
                let i = 2 * i;

                let abs0 = unsafe { col.read_unchecked(i, 0) }.score();
                let abs1 = unsafe { col.read_unchecked(i + 1, 0) }.score();

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
                let abs0 = unsafe { col.read_unchecked(i, 0) }.score();
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

        *t = imax - k;

        if imax != k {
            n_transpositions += 1;
            perm.swap(k, imax);
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
        let inv = matrix.read(j, j).inv();
        for i in j + 1..m {
            unsafe {
                matrix.write_unchecked(i, j, matrix.read_unchecked(i, j).mul(inv));
            }
        }
        let [_, top_right, bottom_left, bottom_right] = matrix.rb_mut().split_at(j + 1, j + 1);
        let lhs = bottom_left.rb().col(j);
        let rhs = top_right.rb().row(j);
        let mut mat = bottom_right;

        let m = mat.nrows();
        let lhs = E::map(
            lhs.as_ptr(),
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts(ptr, m) },
        );

        let lane_count = core::mem::size_of::<E::SimdUnit<S>>() / core::mem::size_of::<E::Unit>();

        let prefix = m % lane_count;

        let (lhs_head, lhs_tail) = E::unzip(E::map(
            lhs,
            #[inline(always)]
            |slice| slice.split_at(prefix),
        ));
        let lhs_tail = faer_core::simd::slice_as_simd::<E, S>(lhs_tail).0;

        for k in 0..mat.ncols() {
            let acc = E::map(
                mat.rb_mut().ptr_at(0, k),
                #[inline(always)]
                |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, m) },
            );
            let (acc_head, acc_tail) = E::unzip(E::map(
                acc,
                #[inline(always)]
                |slice| slice.split_at_mut(prefix),
            ));
            let acc_tail = faer_core::simd::slice_as_mut_simd::<E, S>(acc_tail).0;

            let rhs = E::simd_splat(simd, unsafe { rhs.read_unchecked(0, k).neg() });

            let mut acc_head_ = E::partial_load_last(simd, E::rb(E::as_ref(&acc_head)));
            acc_head_ = E::simd_mul_adde(
                simd,
                E::copy(&rhs),
                E::partial_load_last(simd, E::copy(&lhs_head)),
                acc_head_,
            );
            E::partial_store_last(simd, acc_head, acc_head_);

            for (acc, lhs) in E::into_iter(acc_tail).zip(E::into_iter(E::copy(&lhs_tail))) {
                let mut acc_ = E::deref(E::rb(E::as_ref(&acc)));
                let lhs = E::deref(lhs);
                acc_ = E::simd_mul_adde(simd, E::copy(&rhs), lhs, acc_);
                E::map(
                    E::zip(acc, acc_),
                    #[inline(always)]
                    |(acc, acc_)| *acc = acc_,
                );
            }
        }
    }
}

fn update<E: ComplexField>(arch: pulp::Arch, mut matrix: MatMut<E>, j: usize) {
    if E::HAS_SIMD && matrix.row_stride() == 1 {
        arch.dispatch(Update { matrix, j });
    } else {
        let m = matrix.nrows();
        let inv = matrix.read(j, j).inv();
        for i in j + 1..m {
            matrix.write(i, j, matrix.read(i, j).mul(inv));
        }
        let [_, top_right, bottom_left, bottom_right] = matrix.rb_mut().split_at(j + 1, j + 1);
        let lhs = bottom_left.rb().col(j);
        let rhs = top_right.rb().row(j);
        let mut mat = bottom_right;

        for k in 0..mat.ncols() {
            let col = mat.rb_mut().col(k);
            let rhs = rhs.read(0, k);
            zipped!(col, lhs).for_each(|mut x, lhs| x.write(x.read().sub(lhs.read().mul(rhs))));
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

fn lu_recursive_req<E: Entity>(
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
        lu_recursive_req::<E>(m, bs, parallelism)?,
        StackReq::try_all_of([
            StackReq::try_new::<usize>(m - bs)?,
            lu_recursive_req::<E>(m - bs, n - bs, parallelism)?,
        ])?,
        temp_mat_req::<E>(m, 1)?,
    ])
}

fn lu_in_place_impl<E: ComplexField>(
    mut matrix: MatMut<'_, E>,
    col_start: usize,
    n: usize,
    perm: &mut [usize],
    transpositions: &mut [usize],
    parallelism: Parallelism,
    mut stack: PodStack<'_>,
) -> usize {
    let m = matrix.nrows();
    let full_n = matrix.ncols();

    debug_assert!(m >= n);
    debug_assert!(perm.len() == m);

    if n <= recursion_threshold::<E>(m) {
        return lu_in_place_unblocked(matrix, col_start, n, perm, transpositions, stack);
    }

    let bs = blocksize::<E>(n);

    let mut n_transpositions = 0;

    n_transpositions += lu_in_place_impl(
        matrix.rb_mut().submatrix(0, col_start, m, n),
        0,
        bs,
        perm,
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
        Some(E::one()),
        E::one().neg(),
        parallelism,
    );

    {
        let (mut tmp_perm, mut stack) = stack.rb_mut().make_with(m - bs, |i| i);
        let tmp_perm = &mut *tmp_perm;
        n_transpositions += lu_in_place_impl(
            matrix.rb_mut().submatrix(bs, col_start, m - bs, n),
            bs,
            n - bs,
            tmp_perm,
            &mut transpositions[bs..],
            parallelism,
            stack.rb_mut(),
        );

        for tmp in tmp_perm.iter_mut() {
            *tmp = perm[bs + *tmp];
        }
        perm[bs..].copy_from_slice(tmp_perm);
    }

    let parallelism = if m * (col_start + (full_n - (col_start + n))) > 128 * 128 {
        parallelism
    } else {
        Parallelism::None
    };
    if matrix.col_stride().abs() < matrix.row_stride().abs() {
        for (i, &t) in transpositions[..bs].iter().enumerate() {
            swap_rows(matrix.rb_mut().submatrix(0, 0, m, col_start), i, t + i);
        }
        for (i, &t) in transpositions[bs..].iter().enumerate() {
            swap_rows(
                matrix.rb_mut().submatrix(bs, 0, m - bs, col_start),
                i,
                t + i,
            );
        }
        for (i, &t) in transpositions[..bs].iter().enumerate() {
            swap_rows(
                matrix
                    .rb_mut()
                    .submatrix(0, col_start + n, m, full_n - col_start - n),
                i,
                t + i,
            );
        }
        for (i, &t) in transpositions[bs..].iter().enumerate() {
            swap_rows(
                matrix
                    .rb_mut()
                    .submatrix(bs, col_start + n, m - bs, full_n - col_start - n),
                i,
                t + i,
            );
        }
    } else if matrix.row_stride() == 1 {
        faer_core::for_each_raw(
            col_start + (full_n - (col_start + n)),
            |j| {
                let j = if j >= col_start { col_start + n + j } else { j };
                let mut col = unsafe { matrix.rb().col(j).const_cast() };
                for (i, &t) in transpositions[..bs].iter().enumerate() {
                    swap_two_elems_contiguous(col.rb_mut(), i, t + i);
                }
                let [_, mut col] = col.split_at_row(bs);
                for (i, &t) in transpositions[bs..].iter().enumerate() {
                    swap_two_elems_contiguous(col.rb_mut(), i, t + i);
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
                    swap_two_elems(col.rb_mut(), i, t + i);
                }
                let [_, mut col] = col.split_at_row(bs);
                for (i, &t) in transpositions[bs..].iter().enumerate() {
                    swap_two_elems(col.rb_mut(), i, t + i);
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
pub fn lu_in_place_req<E: Entity>(
    m: usize,
    n: usize,
    parallelism: Parallelism,
    params: PartialPivLuComputeParams,
) -> Result<StackReq, SizeOverflow> {
    let _ = &params;

    let size = Ord::min(n, m);
    StackReq::try_all_of([
        StackReq::try_new::<usize>(size)?,
        lu_recursive_req::<E>(m, size, parallelism)?,
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
pub fn lu_in_place<'out, E: ComplexField>(
    matrix: MatMut<'_, E>,
    perm: &'out mut [usize],
    perm_inv: &'out mut [usize],
    parallelism: Parallelism,
    stack: PodStack<'_>,
    params: PartialPivLuComputeParams,
) -> (usize, PermutationMut<'out>) {
    let _ = &params;

    assert!(perm.len() == matrix.nrows());
    assert!(perm_inv.len() == matrix.nrows());
    let mut matrix = matrix;
    let mut stack = stack;
    let m = matrix.nrows();
    let n = matrix.ncols();
    let size = Ord::min(n, m);

    for (i, p) in perm.iter_mut().enumerate() {
        *p = i;
    }

    let n_transpositions = {
        let (mut transpositions, mut stack) = stack.rb_mut().make_with(size, |_| 0);

        lu_in_place_impl(
            matrix.rb_mut(),
            0,
            size,
            perm,
            &mut transpositions,
            parallelism,
            stack.rb_mut(),
        )
    };

    let [_, _, left, right] = matrix.split_at(0, size);

    if m < n {
        solve_unit_lower_triangular_in_place(left.rb(), right, parallelism);
    }

    for i in 0..m {
        perm_inv[perm[i]] = i;
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

    fn reconstruct_matrix<E: ComplexField>(
        lu_factors: MatRef<'_, E>,
        row_perm: PermutationRef<'_>,
    ) -> Mat<E> {
        let m = lu_factors.nrows();
        let n = lu_factors.ncols();
        let mut dst = Mat::zeros(m, n);
        reconstruct::reconstruct(
            dst.as_mut(),
            lu_factors,
            row_perm,
            Parallelism::Rayon(0),
            make_stack!(reconstruct::reconstruct_req::<E>(
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
            let mut perm = vec![0; m];
            let mut perm_inv = vec![0; m];

            let mut mem = GlobalPodBuffer::new(
                lu_in_place_req::<f64>(m, n, Parallelism::Rayon(8), Default::default()).unwrap(),
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
            let mut perm = vec![0; m];
            let mut perm_inv = vec![0; m];

            let mut mem = GlobalPodBuffer::new(
                lu_in_place_req::<f64>(m, n, Parallelism::Rayon(8), Default::default()).unwrap(),
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
            let mut mat = Mat::from_fn(n, m, |_, _| random::<f64>());
            let mut mat = mat.as_mut().transpose();
            let mat_orig = mat.to_owned();
            let mut perm = vec![0; m];
            let mut perm_inv = vec![0; m];

            let mut mem = GlobalPodBuffer::new(
                lu_in_place_req::<f64>(m, n, Parallelism::Rayon(8), Default::default()).unwrap(),
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
