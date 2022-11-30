//! The Cholesky decomposition of a hermitian positive definite matrix $A$ is such that:
//! $$A = LL^*,$$
//! where $L$ is a lower triangular matrix.

pub mod compute;
pub mod inverse;
pub mod reconstruct;
pub mod solve;
pub mod update;

#[derive(Debug, Clone, Copy)]
pub struct CholeskyError;

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use dyn_stack::{DynStack, GlobalMemBuffer};
    use rand::random;

    use super::{compute::*, inverse::*, reconstruct::*, solve::*, update::*};
    use faer_core::{c64, mul, ComplexField, Conj, Mat, MatRef, Parallelism};

    type T = c64;

    fn reconstruct_matrix(cholesky_factor: MatRef<'_, T>) -> Mat<T> {
        let n = cholesky_factor.nrows();

        let mut a_reconstructed = Mat::zeros(n, n);
        reconstruct_lower_to(
            a_reconstructed.as_mut(),
            cholesky_factor,
            Parallelism::Rayon(0),
            DynStack::new(&mut GlobalMemBuffer::new(
                reconstruct_lower_to_req::<T>(n).unwrap(),
            )),
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
            cholesky_in_place(
                a.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut []),
                Default::default(),
            )
            .unwrap();
            let mut a_reconstructed = reconstruct_matrix(a.as_ref());
            let mut inv = Mat::zeros(n, n);
            invert_lower_to(
                inv.as_mut(),
                a.as_ref(),
                Parallelism::Rayon(0),
                DynStack::new(&mut GlobalMemBuffer::new(
                    invert_lower_to_req::<T>(n, Parallelism::Rayon(0)).unwrap(),
                )),
            );

            for j in 0..n {
                for i in 0..j {
                    a_reconstructed[(i, j)] = a_reconstructed[(j, i)].conj();
                    inv[(i, j)] = inv[(j, i)].conj();
                }
            }

            let mut prod = Mat::zeros(n, n);
            mul::matmul(
                prod.as_mut(),
                Conj::No,
                a_reconstructed.as_ref(),
                Conj::No,
                inv.as_ref(),
                Conj::No,
                None,
                T::one(),
                Parallelism::Rayon(0),
            );

            for j in 0..n {
                for i in j..n {
                    assert_approx_eq!(a_reconstructed[(i, j)], a_orig[(i, j)]);
                }
            }

            for j in 0..n {
                for i in 0..n {
                    let target = if i == j { T::one() } else { T::zero() };
                    assert_approx_eq!(prod[(i, j)], target);
                }
            }
        }
    }

    #[test]
    fn test_solve() {
        for n in 0..20 {
            let k = 5;
            let mut a = random_positive_definite(n);
            let mut rhs = Mat::with_dims(|_, _| T::new(random(), random()), n, k);
            let a_orig = a.clone();
            let rhs_orig = rhs.clone();

            cholesky_in_place(
                a.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut []),
                Default::default(),
            )
            .unwrap();
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
    }

    #[test]
    fn test_update() {
        use mul::triangular::BlockStructure::*;
        for k in [0, 1, 2, 3, 4, 5] {
            let n = 4;
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

            cholesky_in_place(
                a.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut []),
                Default::default(),
            )
            .unwrap();

            rank_r_update_clobber(a.as_mut(), w.as_mut(), alpha).unwrap();

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
        let a_orig = random_positive_definite(4);

        {
            let mut a = a_orig.clone();
            let n = a.nrows();
            let r = 2;

            cholesky_in_place(
                a.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut []),
                Default::default(),
            )
            .unwrap();

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

            cholesky_in_place(
                a.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut []),
                Default::default(),
            )
            .unwrap();

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

            cholesky_in_place(
                a.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut []),
                Default::default(),
            )
            .unwrap();

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
        let n = 4;
        let r = 2;

        let a_orig = random_positive_definite(n + r);

        for position in [0, 2, 4] {
            let mut a = a_orig.clone();
            let mut w = Mat::zeros(n + r, r);

            for j in 0..r {
                for i in 0..n + r {
                    w[(i, j)] = a[(i, j + position)];
                }
            }

            cholesky_in_place(
                a.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut []),
                Default::default(),
            )
            .unwrap();

            delete_rows_and_cols_clobber(
                a.as_mut(),
                &mut [position, position + 1],
                DynStack::new(&mut GlobalMemBuffer::new(
                    delete_rows_and_cols_clobber_req::<f64>(n + r, r).unwrap(),
                )),
            );

            insert_rows_and_cols_clobber(
                a.as_mut(),
                position,
                w.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut GlobalMemBuffer::new(
                    insert_rows_and_cols_clobber_req::<T>(r, Parallelism::Rayon(8)).unwrap(),
                )),
            )
            .unwrap();

            let a_reconstructed = reconstruct_matrix(a.as_ref());
            for j in 0..n + r {
                for i in j..n + r {
                    assert_approx_eq!(a_reconstructed[(i, j)], a_orig[(i, j)], 1e-4);
                }
            }
        }
    }
}
