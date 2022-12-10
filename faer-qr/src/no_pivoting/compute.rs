use assert2::assert as fancy_assert;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{
    householder::{
        apply_block_householder_transpose_on_the_left_in_place, make_householder_in_place,
        upgrade_householder_factor,
    },
    mul::dot,
    temp_mat_req, ColMut, ComplexField, Conj, MatMut, Parallelism,
};
use num_traits::Zero;
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
                let tau_inv = tau.inv();

                *first_col_head.rb_mut().get(0) = beta;

                for col in last_cols.into_col_iter() {
                    let (col_head, col_tail) = col.split_at(1);
                    let col_head = col_head.get(0);

                    let dot = *col_head + dot(simd, first_col_tail.rb(), col_tail.rb());
                    let k = -dot * tau_inv;
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

/// The recommended block size to use for a QR decomposition of a matrix with the given shape.
#[inline]
pub fn recommended_blocksize<T: ComplexField>(nrows: usize, ncols: usize) -> usize {
    let prod = nrows * ncols;
    let size = nrows.min(ncols);

    (if prod > 8192 * 8192 {
        256
    } else if prod > 2048 * 2048 {
        128
    } else if prod > 1024 * 1024 {
        64
    } else if prod > 512 * 512 {
        32
    } else {
        16
    })
    .min(size)
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

fn qr_in_place_blocked<T: ComplexField>(
    matrix: MatMut<'_, T>,
    householder_factor: MatMut<'_, T>,
    blocksize: usize,
    parallelism: Parallelism,
    stack: DynStack<'_>,
    params: QrComputeParams,
) {
    if blocksize == 1 {
        return qr_in_place_unblocked(matrix, householder_factor.diagonal(), stack);
    }

    let mut matrix = matrix;
    let mut householder_factor = householder_factor;
    let mut stack = stack;
    let mut parallelism = parallelism;
    let m = matrix.nrows();
    let n = matrix.ncols();
    let size = m.min(n);

    let (disable_blocking, disable_parallelism) = params.normalize();

    let householder_is_full_matrix = householder_factor.nrows() == householder_factor.ncols();

    let mut j = 0;
    while j < size {
        let bs = blocksize.min(size - j);
        let mut householder_factor = if householder_is_full_matrix {
            householder_factor.rb_mut().submatrix(j, j, bs, bs)
        } else {
            householder_factor.rb_mut().submatrix(0, j, bs, bs)
        };
        let mut matrix = matrix.rb_mut().submatrix(j, j, m - j, n - j);
        let m = m - j;
        let n = n - j;

        let (mut current_block, mut trailing_cols) = matrix.rb_mut().split_at_col(bs);

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

        apply_block_householder_transpose_on_the_left_in_place(
            current_block.rb(),
            householder_factor.rb(),
            Conj::Yes,
            trailing_cols.rb_mut(),
            Conj::No,
            parallelism,
            stack.rb_mut(),
        );

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
/// The block size is chosed as the number of rows of `householder_factor`.
///
/// # Panics
///
/// - Panics if the number of columns of the householder factor is not equal to the minimum of the
/// number of rows and the number of columns of the input matrix.
/// - Panics if the block size is zero.
/// - Panics if the provided memory in `stack` is insufficient.
#[track_caller]
pub fn qr_in_place<T: ComplexField>(
    matrix: MatMut<'_, T>,
    householder_factor: MatMut<'_, T>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
    params: QrComputeParams,
) {
    let blocksize = householder_factor.nrows();
    let size = matrix.nrows().min(matrix.ncols());
    fancy_assert!(blocksize > 0);
    fancy_assert!((householder_factor.nrows(), householder_factor.ncols()) == (blocksize, size));
    qr_in_place_blocked(
        matrix,
        householder_factor,
        blocksize,
        parallelism,
        stack,
        params,
    );
}

/// Computes the size and alignment of required workspace for performing a QR
/// decomposition with no pivoting.
#[inline]
pub fn qr_in_place_req<T: 'static>(
    nrows: usize,
    ncols: usize,
    blocksize: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = parallelism;
    let _ = nrows;
    temp_mat_req::<T>(blocksize, ncols)
}

#[cfg(test)]
mod tests {
    use faer_core::{
        householder::apply_block_householder_sequence_on_the_left_in_place, Conj, Parallelism,
    };
    use num_traits::One;
    use std::cell::RefCell;

    use assert_approx_eq::assert_approx_eq;
    use faer_core::{c64, mul::matmul, zip::Diag, Mat, MatRef};

    use super::*;

    macro_rules! placeholder_stack {
        () => {
            ::dyn_stack::DynStack::new(&mut ::dyn_stack::GlobalMemBuffer::new(
                ::dyn_stack::StackReq::new::<T>(1024 * 1024),
            ))
        };
    }

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::DynStack::new(&mut ::dyn_stack::GlobalMemBuffer::new($req))
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

        let mut q = Mat::zeros(m, m);
        let mut r = Mat::zeros(m, n);

        r.as_mut()
            .cwise()
            .zip(qr_factors)
            .for_each_triangular_upper(Diag::Include, |a, b| *a = *b);

        q.as_mut().diagonal().cwise().for_each(|a| *a = T::one());

        apply_block_householder_sequence_on_the_left_in_place(
            qr_factors,
            householder,
            Conj::No,
            q.as_mut(),
            Conj::No,
            Parallelism::Rayon(0),
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
            let mut householder = Mat::zeros(1, size);

            qr_in_place_unblocked(
                mat.as_mut(),
                householder.as_mut().row(0).transpose(),
                placeholder_stack!(),
            );

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
    fn test_blocked() {
        for parallelism in [Parallelism::None, Parallelism::Rayon(0)] {
            for (m, n) in [(2, 3), (2, 2), (2, 4), (4, 2), (4, 4), (64, 64)] {
                let mat_orig = Mat::with_dims(|_, _| random_value(), m, n);
                let mut mat = mat_orig.clone();
                let size = m.min(n);
                let blocksize = 8;
                let mut householder = Mat::zeros(blocksize, size);

                qr_in_place(
                    mat.as_mut(),
                    householder.as_mut(),
                    parallelism,
                    make_stack!(qr_in_place_req::<T>(m, n, blocksize, parallelism).unwrap()),
                    Default::default(),
                );

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
}
