use assert2::{assert as fancy_assert, debug_assert as fancy_debug_assert};
use dyn_stack::DynStack;
use faer_core::{
    householder::{apply_block_househodler_on_the_left, make_householder_in_place_unchecked},
    mul::triangular,
    temp_mat_uninit, ColMut, ComplexField, Conj, MatMut, MatRef, Parallelism,
};
use reborrow::*;

use crate::col_pivoting::compute::dot;

fn qr_in_place_unblocked<T: ComplexField>(
    matrix: MatMut<'_, T>,
    householder_factor: ColMut<'_, T>,
    _stack: DynStack<'_>,
) {
    struct QrInPlaceUnblocked<'a, T> {
        matrix: MatMut<'a, T>,
        householder_factor: ColMut<'a, T>,
    }

    impl<'a, T: ComplexField> pulp::WithSimd for QrInPlaceUnblocked<'a, T> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self {
                mut matrix,
                mut householder_factor,
            } = self;
            let m = matrix.nrows();
            let n = matrix.ncols();
            let size = n.min(m);

            fancy_assert!(householder_factor.nrows() == size);

            for k in 0..size {
                let mat_rem = matrix.rb_mut().submatrix(k, k, m - k, n - k);
                let (_, _, first_col, last_cols) = mat_rem.split_at(0, 1);
                let (mut first_col_head, mut first_col_tail) = first_col.col(0).split_at(1);

                let mut tail_squared_norm = T::Real::zero();
                for &elem in first_col_tail.rb() {
                    tail_squared_norm = tail_squared_norm + (elem * elem.conj()).real();
                }

                let (tau, beta) = unsafe {
                    let (tau, beta) = make_householder_in_place_unchecked(
                        first_col_tail.rb_mut(),
                        *first_col_head.rb().get_unchecked(0),
                        tail_squared_norm,
                    );

                    *householder_factor.rb_mut().ptr_in_bounds_at(k) = tau;
                    (tau, beta)
                };

                *first_col_head.rb_mut().get(0) = beta;

                for col in last_cols.into_col_iter() {
                    let (col_head, col_tail) = col.split_at(1);
                    let col_head = col_head.get(0);

                    let dot = *col_head + dot(simd, first_col_tail.rb(), col_tail.rb());
                    let k = -tau * dot;
                    *col_head = *col_head + k;
                    col_tail.cwise().zip(first_col_tail.rb()).for_each(|a, b| {
                        *a = *a + k * *b;
                    });
                }
            }
        }
    }

    pulp::Arch::new().dispatch(QrInPlaceUnblocked {
        matrix,
        householder_factor,
    });
}

pub fn qr_in_place_blocked<T: ComplexField>(
    mut matrix: MatMut<'_, T>,
    mut householder_factor: ColMut<'_, T>,
    blocksize: usize,
    parallelism: Parallelism,
    mut stack: DynStack<'_>,
) {
    let m = matrix.nrows();
    let n = matrix.ncols();
    let size = n.min(m);

    if m < blocksize * 2 {
        return qr_in_place_unblocked(matrix, householder_factor, stack);
    }

    fancy_assert!(householder_factor.nrows() == size);

    let mut k = 0;
    loop {
        if k == size {
            break;
        }

        let bs = blocksize.min(size - k);
        let mut matrix = matrix.rb_mut().submatrix(k, k, m - k, n - k);
        let (_, _, mut block, remaining_cols) = matrix.rb_mut().split_at(0, bs);
        let mut householder = householder_factor.rb_mut().split_at(k).1.split_at(bs).0;
        qr_in_place_unblocked(block.rb_mut(), householder.rb_mut(), stack.rb_mut());

        if k + bs < n {
            unsafe {
                temp_mat_uninit! {
                    let (t, mut stack) = unsafe { temp_mat_uninit::<T>(bs, bs, stack.rb_mut()) };
                }
                let mut t = t.transpose();
                for i in 0..bs {
                    *t.rb_mut().ptr_in_bounds_at_unchecked(i, i) = householder[i];
                }
                make_householder_factor_unblocked(t.rb_mut(), block.rb(), stack.rb_mut());

                match parallelism {
                    Parallelism::None => {
                        apply_block_househodler_on_the_left(
                            remaining_cols,
                            block.rb(),
                            t.rb(),
                            false,
                            parallelism,
                            stack.rb_mut(),
                        );
                    }
                    Parallelism::Rayon(mut n_threads) => {
                        if n_threads == 0 {
                            n_threads = rayon::current_num_threads();
                        }
                        use rayon::prelude::*;
                        let remaining_col_count = n - k - bs;

                        let cols_per_thread = remaining_col_count / n_threads;
                        let rem = remaining_col_count % n_threads;

                        let remaining_cols = remaining_cols.rb();
                        (0..n_threads).into_par_iter().for_each_init(
                            || {
                                dyn_stack::GlobalMemBuffer::new(
                                    faer_core::temp_mat_req::<T>(bs, cols_per_thread + 1)
                                        .unwrap()
                                        .and(
                                            faer_core::temp_mat_req::<T>(bs, cols_per_thread + 1)
                                                .unwrap(),
                                        ),
                                )
                            },
                            |mem, tid| {
                                let tid_to_col_start = |tid| {
                                    if tid < rem {
                                        tid * (cols_per_thread + 1)
                                    } else {
                                        rem * (cols_per_thread + 1) + (tid - rem) * cols_per_thread
                                    }
                                };
                                let col_start = tid_to_col_start(tid);
                                let col_end = tid_to_col_start(tid + 1);

                                let stack = DynStack::new(mem);
                                let rem_blk = remaining_cols.submatrix(
                                    0,
                                    col_start,
                                    m - k,
                                    col_end - col_start,
                                );
                                let rem_blk = rem_blk.const_cast();
                                apply_block_househodler_on_the_left(
                                    rem_blk,
                                    block.rb(),
                                    t.rb(),
                                    false,
                                    parallelism,
                                    stack,
                                );
                            },
                        );
                    }
                }
            }
        }
        k += bs;
    }
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
    fancy_assert!(householder_factor.col_stride() == 1);

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

            let lhs = lhs.split_at_unchecked(0, rt).3;
            let rhs = rhs.split_at_unchecked(rt, 0).3;

            struct Gevm<'a, T> {
                dst: MatMut<'a, T>,
                lhs: MatRef<'a, T>,
                rhs: MatRef<'a, T>,
                beta: T,
            }
            impl<'a, T: ComplexField> pulp::WithSimd for Gevm<'a, T> {
                type Output = ();

                #[inline(always)]
                fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                    let Self {
                        dst,
                        lhs,
                        rhs,
                        beta,
                    } = self;
                    let mut dst = dst.row(0);
                    let lhs = lhs.row(0).transpose();
                    fancy_assert!(rhs.row_stride() == 1);

                    for (i, rhs) in rhs.into_col_iter().enumerate() {
                        let dst = &mut dst[i];
                        *dst = *dst + beta * dot(simd, lhs, rhs);
                    }
                }
            }

            pulp::Arch::new().dispatch(Gevm {
                dst: dst.rb_mut(),
                lhs,
                rhs,
                beta: factor,
            });

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

#[cfg(test)]
mod tests {
    use faer_core::householder::apply_househodler_on_the_left;
    use std::cell::RefCell;

    use assert_approx_eq::assert_approx_eq;
    use dyn_stack::{GlobalMemBuffer, StackReq};
    use faer_core::{c64, mul::matmul, zip::Diag, ColRef, Mat, MatRef};

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
        householder: ColRef<'_, T>,
    ) -> (Mat<T>, Mat<T>) {
        let m = qr_factors.nrows();
        let n = qr_factors.ncols();
        let size = m.min(n);

        let mut q = Mat::zeros(m, m);
        let mut r = Mat::zeros(m, n);

        r.as_mut()
            .cwise()
            .zip(qr_factors)
            .for_each_triangular_upper(Diag::Include, |a, b| *a = *b);

        q.as_mut().diagonal().cwise().for_each(|a| *a = T::one());

        for k in (0..size).rev() {
            let tau = ComplexField::conj(householder[k]);
            let essential = qr_factors.col(k).split_at(k + 1).1;
            unsafe {
                apply_househodler_on_the_left(
                    q.as_mut().submatrix(k, k, m - k, m - k),
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
            let mut householder = Mat::zeros(size, 1);

            qr_in_place_unblocked(
                mat.as_mut(),
                householder.as_mut().col(0),
                placeholder_stack!(),
            );

            let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref().col(0));
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
    fn test_blocked() {
        for (m, n) in [
            (2, 2),
            (2, 4),
            (4, 2),
            (4, 4),
            (64, 64),
            (63, 63),
            (1024, 1024),
        ] {
            let mut mat = Mat::with_dims(|_, _| random_value(), m, n);
            let mat_orig = mat.clone();
            let size = m.min(n);
            let mut householder = Mat::zeros(size, 1);

            qr_in_place_blocked(
                mat.as_mut(),
                householder.as_mut().col(0),
                16,
                Parallelism::Rayon(0),
                placeholder_stack!(),
            );

            let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref().col(0));
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
