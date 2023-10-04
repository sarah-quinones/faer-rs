//! The Cholesky decomposition of a hermitian positive definite matrix $A$ is such that:
//! $$A = LL^H,$$
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

    use super::{compute::*, inverse::*, reconstruct::*, solve::*, update::*};
    use faer_core::{c64, mul, ComplexField, Conj, Mat, MatRef, Parallelism};

    type E = c64;

    fn reconstruct_matrix(cholesky_factor: MatRef<'_, E>) -> Mat<E> {
        let n = cholesky_factor.nrows();

        let mut a_reconstructed = Mat::zeros(n, n);
        reconstruct_lower(
            a_reconstructed.as_mut(),
            cholesky_factor,
            Parallelism::Rayon(0),
            DynStack::new(&mut GlobalMemBuffer::new(
                reconstruct_lower_req::<E>(n).unwrap(),
            )),
        );

        a_reconstructed
    }

    fn random() -> E {
        E {
            re: rand::random(),
            im: rand::random(),
        }
    }

    fn random_positive_definite(n: usize) -> Mat<E> {
        let a = Mat::from_fn(n, n, |_, _| random());
        let mut ata = Mat::zeros(n, n);

        mul::matmul(
            ata.as_mut(),
            a.as_ref().adjoint(),
            a.as_ref(),
            None,
            E::one(),
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
            invert_lower(
                inv.as_mut(),
                a.as_ref(),
                Parallelism::Rayon(0),
                DynStack::new(&mut GlobalMemBuffer::new(
                    invert_lower_req::<E>(n, Parallelism::Rayon(0)).unwrap(),
                )),
            );

            for j in 0..n {
                for i in 0..j {
                    a_reconstructed.write(i, j, a_reconstructed.read(j, i).conj());
                    inv.write(i, j, inv.read(j, i).conj());
                }
            }

            let mut prod = Mat::zeros(n, n);
            mul::matmul(
                prod.as_mut(),
                a_reconstructed.as_ref(),
                inv.as_ref(),
                None,
                E::one(),
                Parallelism::Rayon(0),
            );

            for j in 0..n {
                for i in j..n {
                    assert_approx_eq!(a_reconstructed.read(i, j), a_orig.read(i, j));
                }
            }

            for j in 0..n {
                for i in 0..n {
                    let target = if i == j { E::one() } else { E::zero() };
                    assert_approx_eq!(prod.read(i, j), target);
                }
            }
        }
    }

    #[test]
    fn test_solve() {
        for n in 0..20 {
            let k = 5;
            let mut a = random_positive_definite(n);
            let mut rhs = Mat::from_fn(n, k, |_, _| random());
            let a_orig = a.clone();
            let rhs_orig = rhs.clone();

            cholesky_in_place(
                a.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut []),
                Default::default(),
            )
            .unwrap();
            solve_in_place_with_conj(
                a.as_ref(),
                Conj::No,
                rhs.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut []),
            );

            let mut result = Mat::zeros(n, k);
            use mul::triangular::BlockStructure::*;
            mul::triangular::matmul(
                result.as_mut(),
                Rectangular,
                a_orig.as_ref(),
                TriangularLower,
                rhs.as_ref(),
                Rectangular,
                None,
                E::one(),
                Parallelism::Rayon(8),
            );

            mul::triangular::matmul(
                result.as_mut(),
                Rectangular,
                a_orig.as_ref().adjoint(),
                StrictTriangularUpper,
                rhs.as_ref(),
                Rectangular,
                Some(E::one()),
                E::one(),
                Parallelism::Rayon(8),
            );

            for j in 0..k {
                for i in 0..n {
                    assert_approx_eq!(result.read(i, j), rhs_orig.read(i, j), 1e-3);
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
            let mut w = Mat::from_fn(n, k, |_, _| random());
            let mut alpha = Mat::from_fn(k, 1, |_, _| E::from_real(rand::random()));
            let alpha = alpha.as_mut().col(0);

            let mut w_alpha = Mat::zeros(n, k);
            for j in 0..k {
                for i in 0..n {
                    w_alpha.write(i, j, alpha.read(j, 0).mul(w.read(i, j)));
                }
            }

            mul::triangular::matmul(
                a_updated.as_mut(),
                TriangularLower,
                w_alpha.as_ref(),
                Rectangular,
                w.as_ref().adjoint(),
                Rectangular,
                Some(E::one()),
                E::one(),
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
                    assert_approx_eq!(a_reconstructed.read(i, j), a_updated.read(i, j), 1e-4);
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
                Parallelism::Rayon(8),
                DynStack::new(&mut GlobalMemBuffer::new(
                    delete_rows_and_cols_clobber_req::<E>(n, r, Parallelism::Rayon(8)).unwrap(),
                )),
            );

            let a_reconstructed = reconstruct_matrix(a.as_ref().submatrix(0, 0, n - r, n - r));
            assert_approx_eq!(a_reconstructed.read(0, 0), a_orig.read(0, 0));
            assert_approx_eq!(a_reconstructed.read(1, 0), a_orig.read(2, 0));
            assert_approx_eq!(a_reconstructed.read(1, 1), a_orig.read(2, 2));
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
                Parallelism::Rayon(8),
                DynStack::new(&mut GlobalMemBuffer::new(
                    delete_rows_and_cols_clobber_req::<E>(n, r, Parallelism::Rayon(8)).unwrap(),
                )),
            );

            let a_reconstructed = reconstruct_matrix(a.as_ref().submatrix(0, 0, n - r, n - r));
            assert_approx_eq!(a_reconstructed.read(0, 0), a_orig.read(1, 1));
            assert_approx_eq!(a_reconstructed.read(1, 0), a_orig.read(3, 1));
            assert_approx_eq!(a_reconstructed.read(1, 1), a_orig.read(3, 3));
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
                Parallelism::Rayon(8),
                DynStack::new(&mut GlobalMemBuffer::new(
                    delete_rows_and_cols_clobber_req::<E>(n, r, Parallelism::Rayon(8)).unwrap(),
                )),
            );

            let a_reconstructed = reconstruct_matrix(a.as_ref().submatrix(0, 0, n - r, n - r));
            assert_approx_eq!(a_reconstructed.read(0, 0), a_orig.read(1, 1));
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
                    w.write(i, j, a.read(i, j + position));
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
                Parallelism::Rayon(8),
                DynStack::new(&mut GlobalMemBuffer::new(
                    delete_rows_and_cols_clobber_req::<f64>(n + r, r, Parallelism::Rayon(8))
                        .unwrap(),
                )),
            );

            insert_rows_and_cols_clobber(
                a.as_mut(),
                position,
                w.as_mut(),
                Parallelism::Rayon(8),
                DynStack::new(&mut GlobalMemBuffer::new(
                    insert_rows_and_cols_clobber_req::<E>(r, Parallelism::Rayon(8)).unwrap(),
                )),
            )
            .unwrap();

            let a_reconstructed = reconstruct_matrix(a.as_ref());
            for j in 0..n + r {
                for i in j..n + r {
                    assert_approx_eq!(a_reconstructed.read(i, j), a_orig.read(i, j), 1e-4);
                }
            }
        }
    }
}
