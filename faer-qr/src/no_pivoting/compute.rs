use assert2::assert as fancy_assert;
use dyn_stack::DynStack;
use faer_core::{
    householder::{
        apply_block_householder_on_the_left, make_householder_factor_unblocked,
        make_householder_in_place,
    },
    mul::dot,
    temp_mat_uninit, ColMut, ComplexField, MatMut, Parallelism,
};
use reborrow::*;

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

                let (tau, beta) = make_householder_in_place(
                    first_col_tail.rb_mut(),
                    *first_col_head.rb().get(0),
                    tail_squared_norm,
                );
                unsafe { *householder_factor.rb_mut().ptr_in_bounds_at(k) = tau };

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

#[inline]
fn default_blocksize(m: usize, n: usize) -> usize {
    let prod = m * n;

    if prod > 8192 * 8192 {
        128
    } else if prod > 2048 * 2048 {
        64
    } else if prod > 1024 * 1024 {
        32
    } else if prod > 512 * 512 {
        16
    } else {
        8
    }
}

fn default_disable_parallelism(m: usize, n: usize) -> bool {
    let prod = m * n;
    prod < 192 * 256
}

fn default_disable_blocking(m: usize, n: usize) -> bool {
    let prod = m * n;
    prod < 48 * 48
}

#[derive(Default, Copy, Clone)]
#[non_exhaustive]
pub struct QrComputeParams {
    pub max_blocksize: usize,
    pub blocksize: Option<fn(nrows: usize, ncols: usize) -> usize>,
    pub disable_blocking: Option<fn(nrows: usize, ncols: usize) -> bool>,
    pub disable_parallelism: Option<fn(nrows: usize, ncols: usize) -> bool>,
}

impl QrComputeParams {
    fn normalize(
        self,
    ) -> (
        usize,
        fn(usize, usize) -> usize,
        fn(usize, usize) -> bool,
        fn(usize, usize) -> bool,
    ) {
        (
            if self.max_blocksize == 0 {
                128
            } else {
                self.max_blocksize
            },
            self.blocksize.unwrap_or(default_blocksize),
            self.disable_blocking.unwrap_or(default_disable_blocking),
            self.disable_parallelism
                .unwrap_or(default_disable_parallelism),
        )
    }
}

pub fn qr_in_place<T: ComplexField>(
    mut matrix: MatMut<'_, T>,
    mut householder_factor: ColMut<'_, T>,
    parallelism: Parallelism,
    mut stack: DynStack<'_>,
    params: QrComputeParams,
) {
    let (max_blocksize, blocksize_fn, disable_blocking_fn, disable_parallelism_fn) =
        params.normalize();

    if max_blocksize == 1 {
        return qr_in_place_unblocked(matrix, householder_factor, stack);
    }

    let m = matrix.nrows();
    let n = matrix.ncols();
    let size = n.min(m);

    fancy_assert!(householder_factor.nrows() == size);

    let mut k = 0;
    loop {
        if k == size {
            break;
        }

        let extra_parallelism = if disable_parallelism_fn(m - k, n - k) {
            Parallelism::None
        } else {
            parallelism
        };

        let blocksize = blocksize_fn(m - k, n - k).min(max_blocksize);
        let bs = blocksize.min(size - k).max(1);
        let mut matrix = matrix.rb_mut().submatrix(k, k, m - k, n - k);

        if disable_blocking_fn(m - k, n - k) {
            return qr_in_place_unblocked(matrix, householder_factor.subrows(k, size - k), stack);
        }

        let (_, _, mut block, remaining_cols) = matrix.rb_mut().split_at(0, bs);
        let mut householder = householder_factor.rb_mut().split_at(k).1.split_at(bs).0;
        if blocksize <= 8 {
            qr_in_place_unblocked(block.rb_mut(), householder.rb_mut(), stack.rb_mut());
        } else {
            qr_in_place(
                block.rb_mut(),
                householder.rb_mut(),
                parallelism,
                stack.rb_mut(),
                QrComputeParams {
                    max_blocksize: 8,
                    ..params
                },
            );
        }

        if k + bs < n {
            temp_mat_uninit! {
                let (t, mut stack) = unsafe { temp_mat_uninit::<T>(bs, bs, stack.rb_mut()) };
            }
            let mut t = t.transpose();
            for i in 0..bs {
                unsafe { *t.rb_mut().ptr_in_bounds_at(i, i) = householder[i] };
            }
            make_householder_factor_unblocked(t.rb_mut(), block.rb(), stack.rb_mut());

            match extra_parallelism {
                Parallelism::None => {
                    apply_block_householder_on_the_left(
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

                    remaining_cols.into_par_col_chunks(n_threads).for_each_init(
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
                        |mem, (_col_start, rem_blk)| {
                            let stack = DynStack::new(mem);
                            apply_block_householder_on_the_left(
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
        k += bs;
    }
}

#[cfg(test)]
mod tests {
    use faer_core::{householder::apply_householder_sequence_on_the_left, Conj};
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

        let mut q = Mat::zeros(m, m);
        let mut r = Mat::zeros(m, n);

        r.as_mut()
            .cwise()
            .zip(qr_factors)
            .for_each_triangular_upper(Diag::Include, |a, b| *a = *b);

        q.as_mut().diagonal().cwise().for_each(|a| *a = T::one());

        apply_householder_sequence_on_the_left(
            q.as_mut(),
            qr_factors,
            householder,
            false,
            placeholder_stack!(),
        );

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

            qr_in_place(
                mat.as_mut(),
                householder.as_mut().col(0),
                Parallelism::Rayon(0),
                placeholder_stack!(),
                Default::default(),
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
