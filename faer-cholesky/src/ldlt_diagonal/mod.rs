//! The Cholesky decomposition with diagonal $D$ of a hermitian matrix $A$ is such that:
//! $$A = LDL^*,$$
//! where $D$ is a diagonal matrix, and $L$ is a unit lower triangular matrix.
//!
//! The Cholesky decomposition with diagonal may have poor numerical stability properties when used
//! with non positive definite matrices. In the general case, it is recommended to first permute the
//! matrix using [`crate::compute_cholesky_permutation`] and
//! [`permute_rows_and_cols_symmetric`](faer_core::permutation::permute_rows_and_cols_symmetric_lower).

pub mod compute;
pub mod solve;
pub mod update;

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use dyn_stack::{DynStack, GlobalMemBuffer};
    use faer_core::{c64, mat, Conj};
    use rand::random;

    use super::*;
    use compute::*;
    use faer_core::{mul, mul::triangular::BlockStructure, ComplexField, Mat, MatRef, Parallelism};
    use solve::*;
    use update::*;

    type T = c64;

    fn reconstruct_matrix(cholesky_factors: MatRef<'_, T>) -> Mat<T> {
        let n = cholesky_factors.nrows();

        let mut lxd = Mat::zeros(n, n);
        for j in 0..n {
            let dj = cholesky_factors[(j, j)];
            lxd[(j, j)] = dj;
            for i in j + 1..n {
                lxd[(i, j)] = cholesky_factors[(i, j)] * dj;
            }
        }

        let mut a_reconstructed = Mat::zeros(n, n);

        mul::triangular::matmul(
            a_reconstructed.as_mut(),
            BlockStructure::Rectangular,
            Conj::No,
            lxd.as_ref(),
            BlockStructure::TriangularLower,
            Conj::No,
            cholesky_factors.transpose(),
            BlockStructure::UnitTriangularUpper,
            Conj::Yes,
            None,
            T::one(),
            Parallelism::Rayon(8),
        );

        a_reconstructed
    }

    fn random_positive_definite(n: usize) -> Mat<T> {
        let a = Mat::with_dims(|_, _| T::new(random(), random()), n, n);
        let mut ata = Mat::zeros(n, n);

        mul::matmul(
            ata.as_mut(),
            Conj::No,
            a.as_ref().transpose(),
            Conj::Yes,
            a.as_ref(),
            Conj::No,
            None,
            T::one(),
            Parallelism::Rayon(8),
        );

        ata
    }

    #[test]
    fn test_roundtrip() {
        for n in (0..32).chain((2..32).map(|i| i * 16)) {
            let mut a = random_positive_definite(n);
            let a_orig = a.clone();
            raw_cholesky_in_place(
                a.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut []),
                Default::default(),
            );
            let a_reconstructed = reconstruct_matrix(a.as_ref());

            for j in 0..n {
                for i in j..n {
                    assert_approx_eq!(a_reconstructed[(i, j)], a_orig[(i, j)]);
                }
            }
        }
    }

    #[test]
    fn test_solve() {
        let n = 511;
        let k = 5;
        let mut a = random_positive_definite(n);
        let mut rhs = Mat::with_dims(|_, _| T::new(random(), random()), n, k);
        let a_orig = a.clone();
        let rhs_orig = rhs.clone();
        raw_cholesky_in_place(
            a.as_mut(),
            Parallelism::Rayon(8),
            DynStack::new(&mut []),
            Default::default(),
        );
        solve_in_place(
            a.as_ref(),
            Conj::No,
            rhs.as_mut(),
            Conj::No,
            Parallelism::Rayon(8),
            DynStack::new(&mut []),
        );

        let mut result = Mat::zeros(n, k);
        use mul::triangular::BlockStructure::*;
        mul::triangular::matmul(
            result.as_mut(),
            Rectangular,
            Conj::No,
            a_orig.as_ref(),
            TriangularLower,
            Conj::No,
            rhs.as_ref(),
            Rectangular,
            Conj::No,
            None,
            T::one(),
            Parallelism::Rayon(8),
        );

        mul::triangular::matmul(
            result.as_mut(),
            Rectangular,
            Conj::No,
            a_orig.as_ref().transpose(),
            StrictTriangularUpper,
            Conj::Yes,
            rhs.as_ref(),
            Rectangular,
            Conj::No,
            Some(T::one()),
            T::one(),
            Parallelism::Rayon(8),
        );

        for j in 0..k {
            for i in 0..n {
                assert_approx_eq!(result[(i, j)], rhs_orig[(i, j)], 1e-3);
            }
        }
    }

    #[test]
    fn test_update() {
        use mul::triangular::BlockStructure::*;
        for k in [0, 1, 2, 3, 4, 5] {
            let n = 511;
            let mut a = random_positive_definite(n);
            let mut a_updated = a.clone();
            let mut w = Mat::with_dims(|_, _| T::new(random(), random()), n, k);
            let mut alpha = Mat::with_dims(|_, _| T::from_real(random()), k, 1);
            let alpha = alpha.as_mut().col(0);

            let mut w_alpha = Mat::zeros(n, k);
            for j in 0..k {
                for i in 0..n {
                    w_alpha[(i, j)] = alpha[j] * w[(i, j)];
                }
            }

            mul::triangular::matmul(
                a_updated.as_mut(),
                TriangularLower,
                Conj::No,
                w_alpha.as_ref(),
                Rectangular,
                Conj::No,
                w.as_ref().transpose(),
                Rectangular,
                Conj::Yes,
                Some(T::one()),
                T::one(),
                Parallelism::Rayon(8),
            );

            raw_cholesky_in_place(
                a.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut []),
                Default::default(),
            );
            rank_r_update_clobber(a.as_mut(), w.as_mut(), alpha);

            let a_reconstructed = reconstruct_matrix(a.as_ref());

            for j in 0..n {
                for i in j..n {
                    assert_approx_eq!(a_reconstructed[(i, j)], a_updated[(i, j)], 1e-4);
                }
            }
        }
    }

    #[test]
    fn test_delete() {
        let a_orig = random_positive_definite(16);

        {
            let mut a = a_orig.clone();
            let n = a.nrows();
            let r = 2;

            raw_cholesky_in_place(
                a.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut []),
                Default::default(),
            );

            delete_rows_and_cols_clobber(
                a.as_mut(),
                &mut [1, 3],
                DynStack::new(&mut GlobalMemBuffer::new(
                    delete_rows_and_cols_clobber_req::<T>(n, r).unwrap(),
                )),
            );

            let a_reconstructed = reconstruct_matrix(a.as_ref().submatrix(0, 0, n - r, n - r));
            assert_approx_eq!(a_reconstructed[(0, 0)], a_orig[(0, 0)]);
            assert_approx_eq!(a_reconstructed[(1, 0)], a_orig[(2, 0)]);
            assert_approx_eq!(a_reconstructed[(1, 1)], a_orig[(2, 2)]);
        }

        {
            let mut a = a_orig.clone();
            let n = a.nrows();
            let r = 2;

            raw_cholesky_in_place(
                a.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut []),
                Default::default(),
            );

            delete_rows_and_cols_clobber(
                a.as_mut(),
                &mut [0, 2],
                DynStack::new(&mut GlobalMemBuffer::new(
                    delete_rows_and_cols_clobber_req::<T>(n, r).unwrap(),
                )),
            );

            let a_reconstructed = reconstruct_matrix(a.as_ref().submatrix(0, 0, n - r, n - r));
            assert_approx_eq!(a_reconstructed[(0, 0)], a_orig[(1, 1)]);
            assert_approx_eq!(a_reconstructed[(1, 0)], a_orig[(3, 1)]);
            assert_approx_eq!(a_reconstructed[(1, 1)], a_orig[(3, 3)]);
        }

        {
            let mut a = a_orig.clone();
            let n = a.nrows();
            let r = 3;

            raw_cholesky_in_place(
                a.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut []),
                Default::default(),
            );

            delete_rows_and_cols_clobber(
                a.as_mut(),
                &mut [0, 2, 3],
                DynStack::new(&mut GlobalMemBuffer::new(
                    delete_rows_and_cols_clobber_req::<T>(n, r).unwrap(),
                )),
            );

            let a_reconstructed = reconstruct_matrix(a.as_ref().submatrix(0, 0, n - r, n - r));
            assert_approx_eq!(a_reconstructed[(0, 0)], a_orig[(1, 1)]);
        }
    }

    #[test]
    fn test_insert() {
        let a_orig = random_positive_definite(4);

        {
            let mut a = a_orig.clone();
            let mut w = Mat::with_dims(|_, _| T::new(random(), random()), 6, 2);

            w[(2, 0)].im = 0.0;
            w[(3, 1)].im = 0.0;
            w[(2, 1)] = ComplexField::conj(w[(3, 0)]);

            let a_new = {
                let w = |i, j| w[(i, j)];
                let wc = |i, j| ComplexField::conj(w(i, j));
                let a = |i, j| a[(i, j)];
                mat![
                    [a(0, 0), a(0, 1), w(0, 0), w(0, 1), a(0, 2), a(0, 3)],
                    [a(1, 0), a(1, 1), w(1, 0), w(1, 1), a(1, 2), a(1, 3)],
                    [wc(0, 0), wc(1, 0), w(2, 0), w(2, 1), wc(4, 0), wc(5, 0)],
                    [wc(0, 1), wc(1, 1), w(3, 0), w(3, 1), wc(4, 1), wc(5, 1)],
                    [a(2, 0), a(2, 1), w(4, 0), w(4, 1), a(2, 2), a(2, 3)],
                    [a(3, 0), a(3, 1), w(5, 0), w(5, 1), a(3, 2), a(3, 3)],
                ]
            };

            let n = a.nrows();
            let r = w.ncols();
            let position = 2;

            raw_cholesky_in_place(
                a.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut []),
                Default::default(),
            );

            a.resize_with(|_, _| T::zero(), n + r, n + r);
            insert_rows_and_cols_clobber(
                a.as_mut(),
                position,
                w.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut GlobalMemBuffer::new(
                    insert_rows_and_cols_clobber_req::<f64>(r, Parallelism::Rayon(8)).unwrap(),
                )),
            );

            let a_reconstructed = reconstruct_matrix(a.as_ref());

            for j in 0..n + r {
                for i in 0..n + r {
                    assert_approx_eq!(a_reconstructed[(i, j)], a_new[(i, j)]);
                }
            }
        }

        {
            let mut a = a_orig.clone();
            let mut w = Mat::with_dims(|_, _| T::new(random(), random()), 6, 2);

            w[(0, 0)].im = 0.0;
            w[(1, 1)].im = 0.0;
            w[(0, 1)] = ComplexField::conj(w[(1, 0)]);

            let a_new = {
                let w = |i, j| w[(i, j)];
                let wc = |i, j| ComplexField::conj(w(i, j));
                let a = |i, j| a[(i, j)];
                mat![
                    [w(0, 0), w(0, 1), wc(2, 0), wc(3, 0), wc(4, 0), wc(5, 0)],
                    [w(1, 0), w(1, 1), wc(2, 1), wc(3, 1), wc(4, 1), wc(5, 1)],
                    [w(2, 0), w(2, 1), a(0, 0), a(0, 1), a(0, 2), a(0, 3)],
                    [w(3, 0), w(3, 1), a(1, 0), a(1, 1), a(1, 2), a(1, 3)],
                    [w(4, 0), w(4, 1), a(2, 0), a(2, 1), a(2, 2), a(2, 3)],
                    [w(5, 0), w(5, 1), a(3, 0), a(3, 1), a(3, 2), a(3, 3)],
                ]
            };

            let n = a.nrows();
            let r = w.ncols();
            let position = 0;

            raw_cholesky_in_place(
                a.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut []),
                Default::default(),
            );

            a.resize_with(|_, _| T::zero(), n + r, n + r);
            insert_rows_and_cols_clobber(
                a.as_mut(),
                position,
                w.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut GlobalMemBuffer::new(
                    insert_rows_and_cols_clobber_req::<c64>(r, Parallelism::Rayon(8)).unwrap(),
                )),
            );

            let a_reconstructed = reconstruct_matrix(a.as_ref());

            for j in 0..n + r {
                for i in 0..n + r {
                    assert_approx_eq!(a_reconstructed[(i, j)], a_new[(i, j)]);
                }
            }
        }

        {
            let mut a = a_orig.clone();
            let mut w = Mat::with_dims(|_, _| T::new(random(), random()), 6, 2);

            w[(4, 0)].im = 0.0;
            w[(5, 1)].im = 0.0;
            w[(4, 1)] = ComplexField::conj(w[(5, 0)]);

            let a_new = {
                let w = |i, j| w[(i, j)];
                let wc = |i, j| ComplexField::conj(w(i, j));
                let a = |i, j| a[(i, j)];
                mat![
                    [a(0, 0), a(0, 1), a(0, 2), a(0, 3), w(0, 0), w(0, 1)],
                    [a(1, 0), a(1, 1), a(1, 2), a(1, 3), w(1, 0), w(1, 1)],
                    [a(2, 0), a(2, 1), a(2, 2), a(2, 3), w(2, 0), w(2, 1)],
                    [a(3, 0), a(3, 1), a(3, 2), a(3, 3), w(3, 0), w(3, 1)],
                    [wc(0, 0), wc(1, 0), wc(2, 0), wc(3, 0), w(4, 0), w(4, 1)],
                    [wc(0, 1), wc(1, 1), wc(2, 1), wc(3, 1), w(5, 0), w(5, 1)],
                ]
            };

            let n = a.nrows();
            let r = w.ncols();
            let position = 4;

            raw_cholesky_in_place(
                a.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut []),
                Default::default(),
            );

            a.resize_with(|_, _| T::zero(), n + r, n + r);
            insert_rows_and_cols_clobber(
                a.as_mut(),
                position,
                w.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut GlobalMemBuffer::new(
                    insert_rows_and_cols_clobber_req::<c64>(r, Parallelism::Rayon(8)).unwrap(),
                )),
            );

            let a_reconstructed = reconstruct_matrix(a.as_ref());

            for j in 0..n + r {
                for i in 0..n + r {
                    assert_approx_eq!(a_reconstructed[(i, j)], a_new[(i, j)]);
                }
            }
        }
    }
}
