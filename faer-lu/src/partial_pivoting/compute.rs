use assert2::{assert as fancy_assert, debug_assert as fancy_debug_assert};
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{
    mul::matmul,
    permutation::{swap_rows, PermutationMut},
    solve::solve_unit_lower_triangular_in_place,
    temp_mat_req, temp_mat_uninit,
    zip::ColUninit,
    ColMut, ComplexField, Conj, MatMut, Parallelism,
};
use reborrow::*;

#[inline]
fn swap_two_elems<T>(m: ColMut<'_, T>, i: usize, j: usize) {
    swap_rows(m.as_2d(), i, j);
}

fn lu_unblocked_req<T: 'static>(_m: usize, _n: usize) -> Result<StackReq, SizeOverflow> {
    Ok(StackReq::default())
}

#[inline(never)]
fn lu_in_place_unblocked<T: ComplexField>(
    mut matrix: MatMut<'_, T>,
    col_start: usize,
    n: usize,
    perm: &mut [usize],
    transpositions: &mut [usize],
    mut stack: DynStack<'_>,
) -> usize {
    let m = matrix.nrows();
    fancy_debug_assert!(m >= n);
    fancy_debug_assert!(perm.len() == m);

    if n == 0 {
        return 0;
    }

    let mut n_transpositions = 0;

    for (j, t) in transpositions.iter_mut().enumerate() {
        let mut max = T::Real::zero();
        let mut imax = j;

        for i in j..m {
            let abs = (*matrix.rb().get(i, j + col_start)).score();
            if abs > max {
                imax = i;
                max = abs;
            }
        }

        *t = imax - j;

        if imax != j {
            n_transpositions += 1;
            perm.swap(j, imax);
        }

        swap_rows(matrix.rb_mut(), j, imax);

        let (_, _, _, middle_right) = matrix.rb_mut().split_at(0, col_start);
        let (_, _, middle, _) = middle_right.split_at(0, n);
        update(middle, j, stack.rb_mut());
    }

    n_transpositions
}

fn update<T: ComplexField>(mut matrix: MatMut<T>, j: usize, _stack: DynStack<'_>) {
    let m = matrix.nrows();
    let inv = matrix.rb().get(j, j).inv();
    for i in j + 1..m {
        let elem = matrix.rb_mut().get(i, j);
        *elem = *elem * inv;
    }
    let (_, top_right, bottom_left, bottom_right) = matrix.rb_mut().split_at(j + 1, j + 1);
    matmul(
        bottom_right,
        Conj::No,
        bottom_left.rb().col(j).as_2d(),
        Conj::No,
        top_right.rb().row(j).as_2d(),
        Conj::No,
        Some(T::one()),
        -T::one(),
        Parallelism::None,
    )
}

fn recursion_threshold<T: 'static>(_m: usize) -> usize {
    16
}

#[inline]
// we want remainder to be a multiple of register size
fn blocksize<T: 'static>(n: usize) -> usize {
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

fn lu_recursive_req<T: 'static>(
    m: usize,
    n: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    if n <= recursion_threshold::<T>(m) {
        return lu_unblocked_req::<T>(m, n);
    }

    let bs = blocksize::<T>(n);

    StackReq::try_any_of([
        lu_recursive_req::<T>(m, bs, parallelism)?,
        StackReq::try_all_of([
            StackReq::try_new::<usize>(m - bs)?,
            lu_recursive_req::<T>(m - bs, n - bs, parallelism)?,
        ])?,
        temp_mat_req::<T>(m, 1)?,
    ])
}

fn lu_in_place_impl<T: ComplexField>(
    mut matrix: MatMut<'_, T>,
    col_start: usize,
    n: usize,
    perm: &mut [usize],
    transpositions: &mut [usize],
    parallelism: Parallelism,
    mut stack: DynStack<'_>,
) -> usize {
    let m = matrix.nrows();
    let full_n = matrix.ncols();

    fancy_debug_assert!(m >= n);
    fancy_debug_assert!(perm.len() == m);

    if n <= recursion_threshold::<T>(m) {
        return lu_in_place_unblocked(matrix, col_start, n, perm, transpositions, stack);
    }

    let bs = blocksize::<T>(n);

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

    let (mat_top_left, mut mat_top_right, mat_bot_left, mut mat_bot_right) = matrix
        .rb_mut()
        .submatrix(0, col_start, m, n)
        .split_at(bs, bs);

    solve_unit_lower_triangular_in_place(
        mat_top_left.rb(),
        Conj::No,
        mat_top_right.rb_mut(),
        Conj::No,
        parallelism,
    );
    matmul(
        mat_bot_right.rb_mut(),
        Conj::No,
        mat_bot_left.rb(),
        Conj::No,
        mat_top_right.rb(),
        Conj::No,
        Some(T::one()),
        -T::one(),
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

    if n_transpositions >= m - m / 2 {
        // use permutations
        temp_mat_uninit! {
            let (tmp_col, _) = unsafe {
                temp_mat_uninit::<T>(m, 1, stack.rb_mut())
            };
        }

        let mut tmp_col = tmp_col.col(0);
        let mut func = |j| {
            let mut col = matrix.rb_mut().col(j);
            ColUninit(tmp_col.rb_mut())
                .cwise()
                .zip(col.rb())
                .for_each(|a, b| unsafe { *a = b.clone() });

            for i in 0..m {
                *col.rb_mut().get(i) = tmp_col.rb().get(perm[i]).clone();
            }
        };

        for j in 0..col_start {
            func(j);
        }
        for j in col_start + n..full_n {
            func(j);
        }
    } else {
        // use transpositions
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
        } else {
            for j in 0..col_start {
                let mut col = matrix.rb_mut().col(j);
                for (i, &t) in transpositions[..bs].iter().enumerate() {
                    swap_two_elems(col.rb_mut(), i, t + i);
                }
                let mut col = col.split_at(bs).1;
                for (i, &t) in transpositions[bs..].iter().enumerate() {
                    swap_two_elems(col.rb_mut(), i, t + i);
                }
            }

            for j in col_start + n..full_n {
                let mut col = matrix.rb_mut().col(j);
                for (i, &t) in transpositions[..bs].iter().enumerate() {
                    swap_two_elems(col.rb_mut(), i, t + i);
                }
                let mut col = col.split_at(bs).1;
                for (i, &t) in transpositions[bs..].iter().enumerate() {
                    swap_two_elems(col.rb_mut(), i, t + i);
                }
            }
        }
    }

    n_transpositions
}

#[derive(Default, Copy, Clone)]
#[non_exhaustive]
pub struct PartialPivLuComputeParams {}

/// Computes the size and alignment of required workspace for performing an LU
/// decomposition with partial pivoting.
pub fn lu_in_place_req<T: 'static>(
    m: usize,
    n: usize,
    parallelism: Parallelism,
    params: PartialPivLuComputeParams,
) -> Result<StackReq, SizeOverflow> {
    let _ = &params;
    StackReq::try_any_of([
        StackReq::try_new::<usize>(n.min(m))?,
        lu_recursive_req::<T>(m, n.min(m), parallelism)?,
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
/// # Output
///
/// - The number of transpositions that constitute the permutation,
/// - a structure representing the permutation $P$.
///
/// # Panics
///
/// Panics if the length of the permutation slices is not equal to the number of rows of the
/// matrix, or if the provided memory in `stack` is insufficient.
pub fn lu_in_place<'out, T: ComplexField>(
    matrix: MatMut<'_, T>,
    perm: &'out mut [usize],
    perm_inv: &'out mut [usize],
    parallelism: Parallelism,
    stack: DynStack<'_>,
    params: PartialPivLuComputeParams,
) -> (usize, PermutationMut<'out>) {
    let _ = &params;

    fancy_assert!(perm.len() == matrix.nrows());
    fancy_assert!(perm_inv.len() == matrix.nrows());
    let mut matrix = matrix;
    let mut stack = stack;
    let m = matrix.nrows();
    let n = matrix.ncols();

    for i in 0..m {
        perm[i] = i;
    }

    let n_transpositions = {
        let (mut transpositions, mut stack) = stack.rb_mut().make_with(n.min(m), |_| 0);

        lu_in_place_impl(
            matrix.rb_mut(),
            0,
            n.min(m),
            perm,
            &mut transpositions,
            parallelism,
            stack.rb_mut(),
        )
    };

    let (_, _, left, right) = matrix.split_at(0, n.min(m));

    if m < n {
        solve_unit_lower_triangular_in_place(left.rb(), Conj::No, right, Conj::No, parallelism);
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
    use assert_approx_eq::assert_approx_eq;
    use dyn_stack::GlobalMemBuffer;
    use faer_core::{permutation::PermutationRef, Mat, MatRef};
    use rand::random;

    use crate::partial_pivoting::reconstruct;

    use super::*;

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::DynStack::new(&mut ::dyn_stack::GlobalMemBuffer::new($req))
        };
    }

    fn reconstruct_matrix<T: ComplexField>(
        lu_factors: MatRef<'_, T>,
        row_perm: PermutationRef<'_>,
    ) -> Mat<T> {
        let m = lu_factors.nrows();
        let n = lu_factors.ncols();
        let mut dst = Mat::zeros(m, n);
        reconstruct::reconstruct_to(
            dst.as_mut(),
            lu_factors,
            row_perm,
            Parallelism::Rayon(0),
            make_stack!(reconstruct::reconstruct_to_req::<T>(m, n, Parallelism::Rayon(0)).unwrap()),
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
            let mut mat = Mat::with_dims(|_i, _j| random::<f64>(), m, n);
            let mat_orig = mat.clone();
            let mut perm = vec![0; m];
            let mut perm_inv = vec![0; m];

            let mut mem = GlobalMemBuffer::new(
                lu_in_place_req::<f64>(m, n, Parallelism::Rayon(8), Default::default()).unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

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
                    assert_approx_eq!(mat_orig[(i, j)], reconstructed[(i, j)]);
                }
            }
        }
    }
}
