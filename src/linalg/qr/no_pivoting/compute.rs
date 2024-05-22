use crate::{
    assert,
    linalg::{
        householder::{
            self, apply_block_householder_transpose_on_the_left_in_place_with_conj,
            upgrade_householder_factor,
        },
        matmul::inner_prod::{self, inner_prod_with_conj_arch},
        temp_mat_req,
    },
    unzipped,
    utils::{
        simd::{SimdFor, YesConj},
        slice::*,
    },
    zipped, Conj, MatMut, MatRef, Parallelism,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_entity::*;
use reborrow::*;

fn qr_in_place_unblocked<E: ComplexField>(
    mut matrix: MatMut<'_, E>,
    mut householder_factor: MatMut<'_, E>,
    _stack: PodStack<'_>,
) {
    let m = matrix.nrows();
    let n = matrix.ncols();
    let size = Ord::min(m, n);

    assert!(householder_factor.nrows() == size);

    let arch = E::Simd::default();
    let row_stride = matrix.row_stride();

    for k in 0..size {
        let mat_rem = matrix.rb_mut().submatrix_mut(k, k, m - k, n - k);
        let (_, _, first_col, mut last_cols) = mat_rem.split_at_mut(0, 1);
        let (mut first_col_head, mut first_col_tail) = first_col.col_mut(0).split_at_mut(1);

        let tail_norm = first_col_tail.norm_l2();

        let (tau, beta) = householder::make_householder_in_place(
            Some(first_col_tail.rb_mut()),
            first_col_head.read(0),
            tail_norm,
        );
        householder_factor.write(k, 0, tau);
        let tau_inv = tau.faer_inv();

        first_col_head.write(0, beta);

        if row_stride == 1 {
            struct TrailingColsUpdate<'a, E: ComplexField> {
                tau_inv: E,
                first_col_tail: MatRef<'a, E>,
                last_cols: MatMut<'a, E>,
            }

            impl<E: ComplexField> pulp::WithSimd for TrailingColsUpdate<'_, E> {
                type Output = ();

                #[inline(always)]
                fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                    let Self {
                        tau_inv,
                        first_col_tail,
                        mut last_cols,
                    } = self;
                    debug_assert_eq!(first_col_tail.row_stride(), 1);
                    debug_assert_eq!(last_cols.row_stride(), 1);

                    let n = last_cols.ncols();

                    if last_cols.nrows() == 0 {
                        return;
                    }

                    let first = SliceGroup::<'_, E>::new(first_col_tail.try_get_contiguous_col(0));
                    let simd = SimdFor::<E, S>::new(simd);
                    let offset = simd.align_offset(first.rb());

                    for idx in 0..n {
                        let (mut col0, col) = SliceGroupMut::<'_, E>::new(
                            last_cols.rb_mut().try_get_contiguous_col_mut(idx),
                        )
                        .split_at(1);
                        let col0_ = col0.read(0);

                        let dot = col0_.faer_add(inner_prod::with_simd_and_offset(
                            simd,
                            YesConj,
                            first,
                            col.rb(),
                            offset,
                        ));

                        let k = (dot.faer_mul(tau_inv)).faer_neg();
                        col0.write(0, col0_.faer_add(k));

                        let (col_head, col_body, col_tail) = simd.as_aligned_simd_mut(col, offset);
                        let (first_head, first_body, first_tail) =
                            simd.as_aligned_simd(first, offset);
                        let k = simd.splat(k);

                        #[inline(always)]
                        fn process<E: ComplexField, S: pulp::Simd>(
                            simd: SimdFor<E, S>,
                            mut a: impl pulp::Write<Output = SimdGroupFor<E, S>>,
                            b: impl pulp::Read<Output = SimdGroupFor<E, S>>,
                            k: SimdGroupFor<E, S>,
                        ) {
                            let zero = simd.splat(E::faer_zero());
                            a.write(simd.mul_add_e(k, b.read_or(zero), a.read_or(zero)));
                        }

                        process(simd, col_head, first_head, k);
                        for (col, first) in col_body.into_mut_iter().zip(first_body.into_ref_iter())
                        {
                            process(simd, col, first, k);
                        }
                        process(simd, col_tail, first_tail, k);
                    }
                }
            }

            arch.dispatch(TrailingColsUpdate {
                tau_inv,
                first_col_tail: first_col_tail.rb().as_2d(),
                last_cols,
            });
        } else {
            for idx in 0..last_cols.ncols() {
                let col = last_cols.rb_mut().col_mut(idx);
                let (mut col_head, col_tail) = col.split_at_mut(1);
                let col_head_ = col_head.read(0);

                let dot = col_head_.faer_add(inner_prod_with_conj_arch(
                    arch,
                    first_col_tail.rb(),
                    Conj::Yes,
                    col_tail.rb(),
                    Conj::No,
                ));
                let k = (dot.faer_mul(tau_inv)).faer_neg();
                col_head.write(0, col_head_.faer_add(k));
                zipped!(col_tail.as_2d_mut(), first_col_tail.rb().as_2d()).for_each(
                    |unzipped!(mut a, b)| a.write(a.read().faer_add(k.faer_mul(b.read()))),
                );
            }
        }
    }
}

/// The recommended block size to use for a QR decomposition of a matrix with the given shape.
#[inline]
pub fn recommended_blocksize<E: Entity>(nrows: usize, ncols: usize) -> usize {
    let prod = nrows * ncols;
    let size = nrows.min(ncols);

    (if prod > 8192 * 8192 {
        256
    } else if prod > 2048 * 2048 {
        128
    } else if prod > 1024 * 1024 {
        64
    } else if prod > 512 * 512 {
        48
    } else if prod > 128 * 128 {
        32
    } else if prod > 32 * 32 {
        16
    } else if prod > 16 * 16 {
        8
    } else {
        1
    })
    .min(size)
    .max(1)
}

fn default_disable_parallelism(m: usize, n: usize) -> bool {
    let prod = m * n;
    prod < 192 * 256
}

fn default_disable_blocking(m: usize, n: usize) -> bool {
    let prod = m * n;
    prod < 48 * 48
}

/// QR factorization tuning parameters.
#[derive(Default, Copy, Clone)]
#[non_exhaustive]
pub struct QrComputeParams {
    /// At which size blocking algorithms should be disabled. `None` to automatically determine
    /// this threshold.
    pub disable_blocking: Option<fn(nrows: usize, ncols: usize) -> bool>,
    /// At which size the parallelism should be disabled. `None` to automatically determine this
    /// threshold.
    pub disable_parallelism: Option<fn(nrows: usize, ncols: usize) -> bool>,
}

impl QrComputeParams {
    fn normalize(self) -> (fn(usize, usize) -> bool, fn(usize, usize) -> bool) {
        (
            self.disable_blocking.unwrap_or(default_disable_blocking),
            self.disable_parallelism
                .unwrap_or(default_disable_parallelism),
        )
    }
}

fn qr_in_place_blocked<E: ComplexField>(
    matrix: MatMut<'_, E>,
    householder_factor: MatMut<'_, E>,
    blocksize: usize,
    parallelism: Parallelism,
    stack: PodStack<'_>,
    params: QrComputeParams,
) {
    let mut matrix = matrix;
    let mut householder_factor = householder_factor;
    let mut stack = stack;
    let mut parallelism = parallelism;

    let m = matrix.nrows();
    let n = matrix.ncols();
    let size = Ord::min(m, n);

    let (disable_blocking, disable_parallelism) = params.normalize();
    let householder_is_full_matrix = householder_factor.nrows() == householder_factor.ncols();

    if !householder_is_full_matrix {
        if blocksize == 1 || disable_blocking(m, n) {
            qr_in_place_unblocked(
                matrix.rb_mut(),
                householder_factor.rb_mut().transpose_mut(),
                stack,
            );

            let mut j = 0;
            while j < size {
                let bs = Ord::min(blocksize, size - j);
                let mut householder_factor =
                    householder_factor.rb_mut().submatrix_mut(0, j, bs, bs);
                let matrix = matrix.rb().submatrix(j, j, m - j, bs);

                upgrade_householder_factor(
                    householder_factor.rb_mut(),
                    matrix.rb(),
                    blocksize,
                    1,
                    parallelism,
                );
                j += bs;
            }
            return;
        }
    }
    if blocksize == 1 && householder_is_full_matrix {
        qr_in_place_unblocked(
            matrix.rb_mut(),
            householder_factor
                .rb_mut()
                .diagonal_mut()
                .column_vector_mut()
                .as_2d_mut(),
            stack,
        );
        return;
    }

    let mut j = 0;
    while j < size {
        let bs = Ord::min(blocksize, size - j);
        let mut householder_factor = if householder_is_full_matrix {
            householder_factor.rb_mut().submatrix_mut(j, j, bs, bs)
        } else {
            householder_factor.rb_mut().submatrix_mut(0, j, bs, bs)
        };
        let mut matrix = matrix.rb_mut().submatrix_mut(j, j, m - j, n - j);
        let m = m - j;
        let n = n - j;

        let (mut current_block, mut trailing_cols) = matrix.rb_mut().split_at_col_mut(bs);

        let prev_blocksize = if disable_blocking(m, n) || blocksize <= 4 || blocksize % 2 != 0 {
            1
        } else {
            blocksize / 2
        };

        if parallelism != Parallelism::None && disable_parallelism(m, n) {
            parallelism = Parallelism::None
        }

        qr_in_place_blocked(
            current_block.rb_mut(),
            householder_factor.rb_mut(),
            prev_blocksize,
            parallelism,
            stack.rb_mut(),
            params,
        );

        upgrade_householder_factor(
            householder_factor.rb_mut(),
            current_block.rb(),
            blocksize,
            prev_blocksize,
            parallelism,
        );

        if trailing_cols.ncols() > 0 {
            apply_block_householder_transpose_on_the_left_in_place_with_conj(
                current_block.rb(),
                householder_factor.rb(),
                Conj::Yes,
                trailing_cols.rb_mut(),
                parallelism,
                stack.rb_mut(),
            );
        }

        j += bs;
    }
}

/// Computes the QR decomposition of a rectangular matrix $A$, into a unitary matrix $Q$,
/// represented as a block Householder sequence, and an upper trapezoidal matrix $R$, such that
/// $$A = QR.$$
///
/// The Householder bases of $Q$ are stored in the strictly lower trapezoidal part of `matrix` with
/// an implicit unit diagonal, and its upper triangular Householder factors are stored in
/// `householder_factor`, blockwise in chunks of `blocksize×blocksize`.
///
/// The block size is chosen as the number of rows of `householder_factor`.
///
/// # Panics
///
/// - Panics if the number of columns of the householder factor is not equal to the minimum of the
/// number of rows and the number of columns of the input matrix.
/// - Panics if the block size is zero.
/// - Panics if the provided memory in `stack` is insufficient (see [`qr_in_place_req`]).
#[track_caller]
pub fn qr_in_place<E: ComplexField>(
    matrix: MatMut<'_, E>,
    householder_factor: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
    params: QrComputeParams,
) {
    let blocksize = householder_factor.nrows();
    let size = Ord::min(matrix.nrows(), matrix.ncols());
    assert!(all(
        blocksize > 0,
        householder_factor.nrows() == blocksize,
        householder_factor.ncols() == size,
    ));

    #[cfg(feature = "perf-warn")]
    if matrix.row_stride().unsigned_abs() != 1 && crate::__perf_warn!(QR_WARN) {
        if matrix.col_stride().unsigned_abs() == 1 {
            log::warn!(target: "faer_perf", "QR prefers column-major matrix. Found row-major matrix.");
        } else {
            log::warn!(target: "faer_perf", "QR prefers column-major matrix. Found matrix with generic strides.");
        }
    }

    if blocksize == 1 {
        qr_in_place_unblocked(matrix, householder_factor.transpose_mut(), stack);
    } else {
        qr_in_place_blocked(
            matrix,
            householder_factor,
            blocksize,
            parallelism,
            stack,
            params,
        );
    }
}

/// Computes the size and alignment of required workspace for performing a QR
/// decomposition with no pivoting.
#[inline]
pub fn qr_in_place_req<E: Entity>(
    nrows: usize,
    ncols: usize,
    blocksize: usize,
    parallelism: Parallelism,
    params: QrComputeParams,
) -> Result<StackReq, SizeOverflow> {
    let _ = parallelism;
    let _ = nrows;
    let _ = &params;
    temp_mat_req::<E>(blocksize, ncols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assert,
        complex_native::c64,
        linalg::{
            householder::{
                apply_block_householder_sequence_on_the_left_in_place_req,
                apply_block_householder_sequence_on_the_left_in_place_with_conj,
            },
            matmul::matmul,
            zip::Diag,
        },
        Mat,
    };
    use assert_approx_eq::assert_approx_eq;
    use std::cell::RefCell;

    macro_rules! make_stack {
        ($req: expr $(,)?) => {
            ::dyn_stack::PodStack::new(&mut ::dyn_stack::GlobalPodBuffer::new($req.unwrap()))
        };
    }

    use rand::prelude::*;
    thread_local! {
        static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(0));
    }

    type E = c64;

    fn random_value() -> E {
        RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            let rng = &mut *rng;
            E {
                re: rng.gen(),
                im: rng.gen(),
            }
        })
    }

    fn reconstruct_factors(
        qr_factors: MatRef<'_, E>,
        householder: MatRef<'_, E>,
    ) -> (Mat<E>, Mat<E>) {
        let m = qr_factors.nrows();
        let n = qr_factors.ncols();

        let mut q = Mat::<E>::zeros(m, m);
        let mut r = Mat::<E>::zeros(m, n);

        zipped!(r.as_mut(), qr_factors)
            .for_each_triangular_upper(Diag::Include, |unzipped!(mut a, b)| a.write(b.read()));

        zipped!(q.as_mut().diagonal_mut().column_vector_mut().as_2d_mut())
            .for_each(|unzipped!(mut a)| a.write(E::faer_one()));

        apply_block_householder_sequence_on_the_left_in_place_with_conj(
            qr_factors,
            householder,
            Conj::No,
            q.as_mut(),
            Parallelism::Rayon(0),
            make_stack!(
                apply_block_householder_sequence_on_the_left_in_place_req::<E>(
                    m,
                    householder.nrows(),
                    m,
                )
            ),
        );

        (q, r)
    }

    #[test]
    fn test_unblocked() {
        for (m, n) in [(7, 7), (2, 2), (2, 4), (4, 2), (4, 4)] {
            let mut mat = Mat::from_fn(m, n, |_, _| random_value());
            let mat_orig = mat.clone();
            let size = m.min(n);
            let mut householder = Mat::zeros(1, size);

            qr_in_place_unblocked(
                mat.as_mut(),
                householder.as_mut().row_mut(0).transpose_mut().as_2d_mut(),
                make_stack!(StackReq::try_new::<()>(0)),
            );

            let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref());
            let mut qhq = Mat::zeros(m, m);
            let mut reconstructed = Mat::zeros(m, n);

            matmul(
                reconstructed.as_mut(),
                q.as_ref(),
                r.as_ref(),
                None,
                E::faer_one(),
                Parallelism::Rayon(8),
            );
            matmul(
                qhq.as_mut(),
                q.as_ref().adjoint(),
                q.as_ref(),
                None,
                E::faer_one(),
                Parallelism::Rayon(8),
            );

            for i in 0..m {
                for j in 0..m {
                    assert_approx_eq!(
                        qhq.read(i, j),
                        if i == j {
                            E::faer_one()
                        } else {
                            E::faer_zero()
                        }
                    );
                }
            }
            for i in 0..m {
                for j in 0..n {
                    assert_approx_eq!(reconstructed.read(i, j), mat_orig.read(i, j));
                }
            }
        }
    }

    #[test]
    fn test_blocked() {
        for parallelism in [Parallelism::None, Parallelism::Rayon(0)] {
            for (m, n) in [
                (7, 5),
                (5, 7),
                (7, 7),
                (16, 4),
                (2, 3),
                (2, 2),
                (2, 4),
                (4, 2),
                (4, 4),
                (64, 64),
                (1024, 1024),
            ] {
                let mat_orig = Mat::from_fn(m, n, |_, _| random_value());
                let mut mat = mat_orig.clone();
                let size = m.min(n);
                let blocksize = size.min(512);
                let mut householder = Mat::zeros(blocksize, size);

                qr_in_place(
                    mat.as_mut(),
                    householder.as_mut(),
                    parallelism,
                    make_stack!(qr_in_place_req::<E>(
                        m,
                        n,
                        blocksize,
                        parallelism,
                        Default::default(),
                    )),
                    Default::default(),
                );

                let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref());
                let mut qhq = Mat::zeros(m, m);
                let mut reconstructed = Mat::zeros(m, n);

                matmul(
                    reconstructed.as_mut(),
                    q.as_ref(),
                    r.as_ref(),
                    None,
                    E::faer_one(),
                    Parallelism::Rayon(8),
                );
                matmul(
                    qhq.as_mut(),
                    q.as_ref().adjoint(),
                    q.as_ref(),
                    None,
                    E::faer_one(),
                    Parallelism::Rayon(8),
                );

                for i in 0..m {
                    for j in 0..m {
                        assert_approx_eq!(
                            qhq.read(i, j),
                            if i == j {
                                E::faer_one()
                            } else {
                                E::faer_zero()
                            }
                        );
                    }
                }
                for i in 0..m {
                    for j in 0..n {
                        assert_approx_eq!(reconstructed.read(i, j), mat_orig.read(i, j));
                    }
                }
            }
        }
    }

    #[test]
    fn test_zero() {
        for parallelism in [Parallelism::None, Parallelism::Rayon(0)] {
            for (m, n) in [(2, 3), (2, 2), (2, 4), (4, 2), (4, 4), (64, 64)] {
                let mat_orig = Mat::<E>::zeros(m, n);

                let mut mat = mat_orig.clone();
                let size = m.min(n);
                let blocksize = 8;
                let mut householder = Mat::zeros(blocksize, size);

                qr_in_place(
                    mat.as_mut(),
                    householder.as_mut(),
                    parallelism,
                    make_stack!(qr_in_place_req::<E>(
                        m,
                        n,
                        blocksize,
                        parallelism,
                        Default::default(),
                    )),
                    Default::default(),
                );

                let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref());
                let mut qhq = Mat::zeros(m, m);
                let mut reconstructed = Mat::zeros(m, n);

                matmul(
                    reconstructed.as_mut(),
                    q.as_ref(),
                    r.as_ref(),
                    None,
                    E::faer_one(),
                    Parallelism::Rayon(8),
                );
                matmul(
                    qhq.as_mut(),
                    q.as_ref().adjoint(),
                    q.as_ref(),
                    None,
                    E::faer_one(),
                    Parallelism::Rayon(8),
                );

                for i in 0..m {
                    for j in 0..m {
                        assert_approx_eq!(
                            qhq.read(i, j),
                            if i == j {
                                E::faer_one()
                            } else {
                                E::faer_zero()
                            }
                        );
                    }
                }
                for i in 0..m {
                    for j in 0..n {
                        assert_approx_eq!(reconstructed.read(i, j), mat_orig.read(i, j));
                    }
                }
            }
        }
    }
}
