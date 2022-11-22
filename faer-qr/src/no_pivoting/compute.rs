use assert2::debug_assert as fancy_debug_assert;
use dyn_stack::DynStack;
use faer_core::{
    householder::{
        apply_block_househodler_on_the_left, apply_househodler_on_the_left,
        make_householder_in_place_unchecked,
    },
    mul::{matmul, triangular},
    temp_mat_uninit, ComplexField, Conj, MatMut, MatRef, Parallelism,
};
use reborrow::*;

unsafe fn qr_in_place_unblocked<T: ComplexField>(
    mut matrix: MatMut<'_, T>,
    mut householder_factor: MatMut<'_, T>,
    mut stack: DynStack<'_>,
) {
    let m = matrix.nrows();
    let n = matrix.ncols();
    let size = n.min(m);

    fancy_debug_assert!(householder_factor.nrows() == size);
    fancy_debug_assert!(householder_factor.ncols() == size);

    for k in 0..size {
        let mat_rem = matrix.rb_mut().submatrix_unchecked(k, k, m - k, n - k);
        let (_, _, first_col, last_cols) = mat_rem.split_at_unchecked(0, 1);
        let (mut first_col_head, mut first_col_tail) =
            first_col.col_unchecked(0).split_at_unchecked(1);

        let mut tail_squared_norm = T::Real::zero();
        for &elem in first_col_tail.rb() {
            tail_squared_norm = tail_squared_norm + (elem * elem.conj()).real();
        }

        let (tau, beta) = make_householder_in_place_unchecked(
            first_col_tail.rb_mut(),
            *first_col_head.rb().get_unchecked(0),
            tail_squared_norm,
        );

        *householder_factor.rb_mut().ptr_in_bounds_at_unchecked(k, k) = tau;
        *first_col_head.rb_mut().get_unchecked(0) = beta;

        if last_cols.ncols() > 0 {
            apply_househodler_on_the_left(
                last_cols,
                first_col_tail.rb(),
                *householder_factor.rb().get_unchecked(k, k),
                stack.rb_mut(),
            );
        }
    }

    make_householder_factor_unblocked(householder_factor, matrix.rb(), stack);
}

unsafe fn make_householder_factor_unblocked<T: ComplexField>(
    mut householder_factor: MatMut<'_, T>,
    matrix: MatRef<'_, T>,
    mut stack: DynStack<'_>,
) {
    let m = matrix.nrows();
    let n = matrix.ncols();
    let size = m.min(n);

    fancy_debug_assert!(householder_factor.nrows() == size);
    fancy_debug_assert!(householder_factor.ncols() == size);

    for i in (0..size).rev() {
        let rs = m - i - 1;
        let rt = size - i - 1;

        if rt > 0 {
            let factor = -*householder_factor.rb().get_unchecked(i, i);

            let mut tail_row = householder_factor
                .rb_mut()
                .row_unchecked(i)
                .split_at_unchecked(size - rt)
                .1;

            use faer_core::mul::triangular::BlockStructure::*;

            let mut dst = tail_row.rb_mut().as_2d();
            let lhs = matrix
                .col(i)
                .split_at_unchecked(m - rs)
                .1
                .transpose()
                .as_2d();

            let rhs = matrix.submatrix_unchecked(m - rs, n - rt, rs, rt);
            triangular::matmul(
                dst.rb_mut(),
                Rectangular,
                Conj::No,
                lhs.split_at_unchecked(0, rt).2,
                Rectangular,
                Conj::Yes,
                rhs.split_at_unchecked(rt, 0).1,
                UnitTriangularLower,
                Conj::No,
                None,
                factor,
                Parallelism::None,
            );
            matmul(
                dst.rb_mut(),
                Conj::No,
                lhs.split_at_unchecked(0, rt).3,
                Conj::Yes,
                rhs.split_at_unchecked(rt, 0).3,
                Conj::No,
                Some(T::one()),
                factor,
                Parallelism::None,
            );

            temp_mat_uninit! {
                let (mut tmp, _) = unsafe { temp_mat_uninit::<T>(rt, 1, stack.rb_mut()) };
            }

            triangular::matmul(
                tmp.rb_mut().transpose(),
                Rectangular,
                Conj::No,
                householder_factor
                    .rb()
                    .submatrix_unchecked(i, size - rt, 1, rt),
                Rectangular,
                Conj::No,
                householder_factor
                    .rb()
                    .submatrix_unchecked(size - rt, size - rt, rt, rt),
                TriangularUpper,
                Conj::No,
                None,
                T::one(),
                Parallelism::None,
            );
            householder_factor
                .rb_mut()
                .submatrix_unchecked(i, size - rt, 1, rt)
                .row_unchecked(0)
                .cwise()
                .zip_unchecked(tmp.transpose().row_unchecked(0))
                .for_each(|a, b| *a = *b);
        }
    }
}

unsafe fn make_householder_top_right<T: ComplexField>(
    t01: MatMut<'_, T>,
    t00: MatRef<'_, T>,
    t11: MatRef<'_, T>,
    basis0: MatRef<'_, T>,
    basis1: MatRef<'_, T>,
    parallelism: Parallelism,
    mut stack: DynStack<'_>,
) {
    let m = basis0.nrows();
    let n = basis0.ncols() + basis1.ncols();
    let bs = basis0.ncols();

    temp_mat_uninit! {
        let (mut tmp0, stack) = unsafe { temp_mat_uninit::<T>(m - bs, bs, stack.rb_mut()) };
        let (mut tmp1, _) = unsafe { temp_mat_uninit::<T>(m - bs, n - bs, stack) };
    }

    let v0 = basis0.submatrix_unchecked(bs, 0, m - bs, bs);
    let v1 = basis1;

    let (_, v1_top, _, v1_bot) = v1.split_at_unchecked(n - bs, 0);
    let (_, tmp1_top, _, tmp1_bot) = tmp1.rb_mut().split_at_unchecked(n - bs, 0);

    use triangular::BlockStructure::*;

    faer_core::join_raw(
        |_| {
            triangular::matmul(
                tmp0.rb_mut(),
                Rectangular,
                Conj::No,
                v0,
                Rectangular,
                Conj::No,
                t00.transpose(),
                TriangularLower,
                Conj::Yes,
                None,
                T::one(),
                parallelism,
            )
        },
        |_| {
            faer_core::join_raw(
                |_| {
                    triangular::matmul(
                        tmp1_top,
                        Rectangular,
                        Conj::No,
                        v1_top,
                        UnitTriangularLower,
                        Conj::No,
                        t11,
                        TriangularUpper,
                        Conj::No,
                        None,
                        T::one(),
                        parallelism,
                    )
                },
                |_| {
                    triangular::matmul(
                        tmp1_bot,
                        Rectangular,
                        Conj::No,
                        v1_bot,
                        Rectangular,
                        Conj::No,
                        t11,
                        TriangularUpper,
                        Conj::No,
                        None,
                        T::one(),
                        parallelism,
                    )
                },
                parallelism,
            )
        },
        parallelism,
    );

    matmul(
        t01,
        Conj::No,
        tmp0.rb().transpose(),
        Conj::Yes,
        tmp1.rb(),
        Conj::No,
        None,
        -T::one(),
        parallelism,
    );
}

fn recursion_threshold<T: 'static>(m: usize) -> usize {
    let _m = m;
    16
}

fn block_size<T: 'static>(n: usize) -> usize {
    n / 2
}

pub unsafe fn qr_in_place_recursive<T: ComplexField>(
    mut matrix: MatMut<'_, T>,
    mut householder_factor: MatMut<'_, T>,
    parallelism: Parallelism,
    recursion_level: usize,
    mut stack: DynStack<'_>,
) {
    let m = matrix.nrows();
    let n = matrix.ncols();

    fancy_debug_assert!(m >= n);

    if n < recursion_threshold::<T>(m) {
        return qr_in_place_unblocked(matrix, householder_factor, stack);
    }

    let bs = block_size::<T>(n);
    let (_, _, mut left, mut right) = matrix.rb_mut().split_at_unchecked(0, bs);

    let (mut t00, t01, _, mut t11) = householder_factor.rb_mut().split_at_unchecked(bs, bs);

    qr_in_place_recursive(
        left.rb_mut(),
        t00.rb_mut(),
        parallelism,
        recursion_level + 1,
        stack.rb_mut(),
    );

    apply_block_househodler_on_the_left(
        right.rb_mut(),
        left.rb().submatrix_unchecked(0, 0, m, bs.min(m)),
        t00.rb(),
        false,
        parallelism,
        stack.rb_mut(),
    );

    let (_, _, _, mut bot_right) = right.rb_mut().split_at_unchecked(bs, 0);
    qr_in_place_recursive(
        bot_right.rb_mut(),
        t11.rb_mut(),
        parallelism,
        recursion_level,
        stack.rb_mut(),
    );

    make_householder_top_right(
        t01,
        t00.rb(),
        t11.rb(),
        left.rb(),
        bot_right.rb(),
        parallelism,
        stack,
    );
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use assert_approx_eq::assert_approx_eq;
    use dyn_stack::{GlobalMemBuffer, StackReq};
    use faer_core::{c64, mul::matmul, zip::Diag, Mat, MatRef};

    use super::*;

    macro_rules! placeholder_stack {
        () => {
            DynStack::new(&mut GlobalMemBuffer::new(StackReq::new::<T>(1024 * 1024)))
        };
    }

    use rand::prelude::*;
    thread_local! {
        static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(0));
    }

    type T = c64;

    fn random_value() -> T {
        RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            let rng = &mut *rng;
            T::new(rng.gen(), rng.gen())
        })
    }

    fn reconstruct_factors(
        qr_factors: MatRef<'_, T>,
        householder: MatRef<'_, T>,
    ) -> (Mat<T>, Mat<T>) {
        let m = qr_factors.nrows();
        let n = qr_factors.ncols();
        let size = m.min(n);

        let mut q = Mat::zeros(m, m);
        let mut q2 = Mat::zeros(m, m);
        let mut r = Mat::zeros(m, n);

        r.as_mut()
            .cwise()
            .zip(qr_factors)
            .for_each_triangular_upper(Diag::Include, |a, b| *a = *b);

        q.as_mut().diagonal().cwise().for_each(|a| *a = T::one());
        q2.as_mut().diagonal().cwise().for_each(|a| *a = T::one());

        let (basis_top, _, basis_bot, _) = qr_factors.split_at(size, size);
        let mut tmp = Mat::zeros(m, size);

        let (_, mut tmp_top, _, mut tmp_bot) = tmp.as_mut().split_at(size, 0);

        use triangular::BlockStructure::*;
        triangular::matmul(
            tmp_top.rb_mut(),
            Rectangular,
            Conj::No,
            basis_top,
            UnitTriangularLower,
            Conj::No,
            householder,
            TriangularUpper,
            Conj::No,
            None,
            T::one(),
            Parallelism::Rayon(8),
        );
        triangular::matmul(
            tmp_bot.rb_mut(),
            Rectangular,
            Conj::No,
            basis_bot,
            Rectangular,
            Conj::No,
            householder,
            TriangularUpper,
            Conj::No,
            None,
            T::one(),
            Parallelism::Rayon(8),
        );

        let (q_top_left, q_top_right, q_bot_left, q_bot_right) = q.as_mut().split_at(size, size);
        triangular::matmul(
            q_top_left,
            Rectangular,
            Conj::No,
            tmp_top.rb(),
            Rectangular,
            Conj::No,
            basis_top.transpose(),
            UnitTriangularUpper,
            Conj::Yes,
            Some(T::one()),
            -T::one(),
            Parallelism::Rayon(8),
        );
        triangular::matmul(
            q_bot_left,
            Rectangular,
            Conj::No,
            tmp_bot.rb(),
            Rectangular,
            Conj::No,
            basis_top.transpose(),
            UnitTriangularUpper,
            Conj::Yes,
            Some(T::one()),
            -T::one(),
            Parallelism::Rayon(8),
        );
        matmul(
            q_top_right,
            Conj::No,
            tmp_top.rb(),
            Conj::No,
            basis_bot.transpose(),
            Conj::Yes,
            Some(T::one()),
            -T::one(),
            Parallelism::Rayon(8),
        );
        matmul(
            q_bot_right,
            Conj::No,
            tmp_bot.rb(),
            Conj::No,
            basis_bot.transpose(),
            Conj::Yes,
            Some(T::one()),
            -T::one(),
            Parallelism::Rayon(8),
        );

        for k in (0..size).rev() {
            let tau = ComplexField::conj(householder[(k, k)]);
            let essential = qr_factors.col(k).split_at(k + 1).1;
            unsafe {
                apply_househodler_on_the_left(
                    q2.as_mut().submatrix(k, k, m - k, m - k),
                    essential,
                    tau,
                    placeholder_stack!(),
                );
            }
        }

        (q, r)
    }

    #[test]
    fn test_unblocked() {
        for (m, n) in [(2, 2), (2, 4), (4, 2), (4, 4)] {
            let mut mat = Mat::with_dims(|_, _| random_value(), m, n);
            let mat_orig = mat.clone();
            let size = m.min(n);
            let mut householder = Mat::zeros(size, size);

            dbg!(&mat_orig);
            unsafe {
                qr_in_place_unblocked(mat.as_mut(), householder.as_mut(), placeholder_stack!())
            }

            let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref());
            let mut qhq = Mat::zeros(m, m);
            let mut reconstructed = Mat::zeros(m, n);

            matmul(
                reconstructed.as_mut(),
                Conj::No,
                q.as_ref(),
                Conj::No,
                r.as_ref(),
                Conj::No,
                None,
                T::one(),
                Parallelism::Rayon(8),
            );
            matmul(
                qhq.as_mut(),
                Conj::No,
                q.as_ref().transpose(),
                Conj::Yes,
                q.as_ref(),
                Conj::No,
                None,
                T::one(),
                Parallelism::Rayon(8),
            );

            for i in 0..m {
                for j in 0..m {
                    assert_approx_eq!(qhq[(i, j)], if i == j { T::one() } else { T::zero() });
                }
            }
            for i in 0..m {
                for j in 0..n {
                    assert_approx_eq!(reconstructed[(i, j)], mat_orig[(i, j)]);
                }
            }
        }
    }

    #[test]
    fn test_recursive() {
        for (m, n) in [(2, 2), (4, 2), (4, 4), (33, 33), (64, 64), (65, 65)] {
            let mut mat = Mat::with_dims(|_, _| random_value(), m, n);
            let mat_orig = mat.clone();
            let size = m.min(n);
            let mut householder = Mat::zeros(size, size);
            unsafe {
                qr_in_place_recursive(
                    mat.as_mut(),
                    householder.as_mut(),
                    Parallelism::Rayon(8),
                    0,
                    placeholder_stack!(),
                )
            }

            let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref());
            let mut qhq = Mat::zeros(m, m);
            let mut reconstructed = Mat::zeros(m, n);

            matmul(
                reconstructed.as_mut(),
                Conj::No,
                q.as_ref(),
                Conj::No,
                r.as_ref(),
                Conj::No,
                None,
                T::one(),
                Parallelism::Rayon(8),
            );
            matmul(
                qhq.as_mut(),
                Conj::No,
                q.as_ref().transpose(),
                Conj::Yes,
                q.as_ref(),
                Conj::No,
                None,
                T::one(),
                Parallelism::Rayon(8),
            );

            for i in 0..m {
                for j in 0..m {
                    assert_approx_eq!(qhq[(i, j)], if i == j { T::one() } else { T::zero() });
                }
            }
            for i in 0..m {
                for j in 0..n {
                    assert_approx_eq!(reconstructed[(i, j)], mat_orig[(i, j)]);
                }
            }
        }
    }
}
