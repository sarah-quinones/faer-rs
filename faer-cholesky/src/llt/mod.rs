pub mod compute;
pub mod solve;
// pub mod update;

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use dyn_stack::{DynStack, GlobalMemBuffer};
    use num_complex::ComplexFloat;
    use rand::random;

    use super::compute::*;
    use super::solve::*;
    // use super::update::*;
    use faer_core::mul::triangular::BlockStructure;
    use faer_core::{c64, mul, ComplexField, Mat, MatRef, Parallelism};

    type T = c64;

    fn reconstruct_matrix(cholesky_factor: MatRef<'_, T>) -> Mat<T> {
        let n = cholesky_factor.nrows();

        let mut a_reconstructed = Mat::zeros(n, n);

        mul::triangular::matmul(
            a_reconstructed.as_mut(),
            BlockStructure::Rectangular,
            cholesky_factor,
            BlockStructure::TriangularLower,
            cholesky_factor.transpose(),
            BlockStructure::TriangularUpper,
            None,
            T::one(),
            false,
            false,
            true,
            Parallelism::Rayon,
        );

        a_reconstructed
    }

    fn random_positive_definite(n: usize) -> Mat<T> {
        let a = Mat::with_dims(|_, _| T::new(random(), random()), n, n);
        let mut ata = Mat::zeros(n, n);

        mul::matmul(
            ata.as_mut(),
            a.as_ref().transpose(),
            a.as_ref(),
            None,
            T::one(),
            false,
            true,
            false,
            Parallelism::Rayon,
        );

        ata
    }

    #[test]
    fn test_roundtrip() {
        for n in 0..512 {
            let mut a = random_positive_definite(n);
            let a_orig = a.clone();
            raw_cholesky_in_place(a.as_mut(), Parallelism::Rayon, DynStack::new(&mut [])).unwrap();
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
        for n in 0..20 {
            let k = 5;
            let mut a = random_positive_definite(n);
            let mut rhs = Mat::with_dims(|_, _| T::new(random(), random()), n, k);
            let a_orig = a.clone();
            let rhs_orig = rhs.clone();

            raw_cholesky_in_place(a.as_mut(), Parallelism::Rayon, DynStack::new(&mut [])).unwrap();
            solve_in_place(a.as_ref(), rhs.as_mut(), false, false, Parallelism::Rayon);

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
                T::one(),
                false,
                false,
                false,
                Parallelism::Rayon,
            );

            mul::triangular::matmul(
                result.as_mut(),
                Rectangular,
                a_orig.as_ref().transpose(),
                StrictTriangularUpper,
                rhs.as_ref(),
                Rectangular,
                Some(T::one()),
                T::one(),
                false,
                true,
                false,
                Parallelism::Rayon,
            );

            for j in 0..k {
                for i in 0..n {
                    assert_approx_eq!(result[(i, j)], rhs_orig[(i, j)], 1e-3);
                }
            }
        }
    }

    // #[test]
    // fn test_update() {
    //     use mul::triangular::BlockStructure::*;
    //     let random = |_, _| random::<f64>();

    //     for k in [0, 1, 2, 3, 4, 5] {
    //         let n = 4;
    //         let mut a = random_positive_definite(n);
    //         let mut a_updated = a.clone();
    //         let mut w = Mat::with_dims(random, n, k);
    //         let mut alpha = Mat::with_dims(random, k, 1);
    //         let alpha = alpha.as_mut().col(0);

    //         let mut w_alpha = Mat::zeros(n, k);
    //         for j in 0..k {
    //             for i in 0..n {
    //                 w_alpha[(i, j)] = alpha[j] * w[(i, j)];
    //             }
    //         }

    //         mul::triangular::matmul(
    //             a_updated.as_mut(),
    //             TriangularLower,
    //             w_alpha.as_ref(),
    //             Rectangular,
    //             w.as_ref().transpose(),
    //             Rectangular,
    //             Some(1.0),
    //             1.0,
    //             false,
    //             false,
    //             false,
    //             Parallelism::Rayon,
    //         );

    //         raw_cholesky_in_place(
    //             a.as_mut(),
    //             12,
    //             DynStack::new(&mut GlobalMemBuffer::new(
    //                 raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
    //             )),
    //         )
    //         .unwrap();

    //         rank_r_update_clobber(a.as_mut(), w.as_mut(), alpha).unwrap();

    //         let a_reconstructed = reconstruct_matrix(a.as_ref());

    //         for j in 0..n {
    //             for i in j..n {
    //                 assert_approx_eq!(a_reconstructed[(i, j)], a_updated[(i, j)], 1e-4);
    //             }
    //         }
    //     }
    // }

    // #[test]
    // fn test_delete() {
    //     let a_orig = random_positive_definite(4);

    //     {
    //         let mut a = a_orig.clone();
    //         let n = a.nrows();
    //         let r = 2;

    //         raw_cholesky_in_place(
    //             a.as_mut(),
    //             12,
    //             DynStack::new(&mut GlobalMemBuffer::new(
    //                 raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
    //             )),
    //         )
    //         .unwrap();

    //         delete_rows_and_cols_clobber(
    //             a.as_mut(),
    //             &mut [1, 3],
    //             DynStack::new(&mut GlobalMemBuffer::new(
    //                 delete_rows_and_cols_clobber_req::<f64>(n, r).unwrap(),
    //             )),
    //         );

    //         let a_reconstructed = reconstruct_matrix(a.as_ref().submatrix(0, 0, n - r, n - r));
    //         assert_approx_eq!(a_reconstructed[(0, 0)], a_orig[(0, 0)]);
    //         assert_approx_eq!(a_reconstructed[(1, 0)], a_orig[(2, 0)]);
    //         assert_approx_eq!(a_reconstructed[(1, 1)], a_orig[(2, 2)]);
    //     }

    //     {
    //         let mut a = a_orig.clone();
    //         let n = a.nrows();
    //         let r = 2;

    //         raw_cholesky_in_place(
    //             a.as_mut(),
    //             12,
    //             DynStack::new(&mut GlobalMemBuffer::new(
    //                 raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
    //             )),
    //         )
    //         .unwrap();

    //         delete_rows_and_cols_clobber(
    //             a.as_mut(),
    //             &mut [0, 2],
    //             DynStack::new(&mut GlobalMemBuffer::new(
    //                 delete_rows_and_cols_clobber_req::<f64>(n, r).unwrap(),
    //             )),
    //         );

    //         let a_reconstructed = reconstruct_matrix(a.as_ref().submatrix(0, 0, n - r, n - r));
    //         assert_approx_eq!(a_reconstructed[(0, 0)], a_orig[(1, 1)]);
    //         assert_approx_eq!(a_reconstructed[(1, 0)], a_orig[(3, 1)]);
    //         assert_approx_eq!(a_reconstructed[(1, 1)], a_orig[(3, 3)]);
    //     }

    //     {
    //         let mut a = a_orig.clone();
    //         let n = a.nrows();
    //         let r = 3;

    //         raw_cholesky_in_place(
    //             a.as_mut(),
    //             12,
    //             DynStack::new(&mut GlobalMemBuffer::new(
    //                 raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
    //             )),
    //         )
    //         .unwrap();

    //         delete_rows_and_cols_clobber(
    //             a.as_mut(),
    //             &mut [0, 2, 3],
    //             DynStack::new(&mut GlobalMemBuffer::new(
    //                 delete_rows_and_cols_clobber_req::<f64>(n, r).unwrap(),
    //             )),
    //         );

    //         let a_reconstructed = reconstruct_matrix(a.as_ref().submatrix(0, 0, n - r, n - r));
    //         assert_approx_eq!(a_reconstructed[(0, 0)], a_orig[(1, 1)]);
    //     }
    // }

    // #[test]
    // fn test_insert() {
    //     let n = 4;
    //     let r = 2;

    //     let a_orig = random_positive_definite(n + r);

    //     for position in [0, 2, 4] {
    //         let mut a = a_orig.clone();
    //         let mut w = Mat::zeros(n + r, r);

    //         for j in 0..r {
    //             for i in 0..n + r {
    //                 w[(i, j)] = a[(i, j + position)];
    //             }
    //         }

    //         raw_cholesky_in_place(
    //             a.as_mut(),
    //             12,
    //             DynStack::new(&mut GlobalMemBuffer::new(
    //                 raw_cholesky_in_place_req::<f64>(n + r, 12).unwrap(),
    //             )),
    //         )
    //         .unwrap();

    //         delete_rows_and_cols_clobber(
    //             a.as_mut(),
    //             &mut [position, position + 1],
    //             DynStack::new(&mut GlobalMemBuffer::new(
    //                 delete_rows_and_cols_clobber_req::<f64>(n + r, r).unwrap(),
    //             )),
    //         );

    //         insert_rows_and_cols_clobber(
    //             a.as_mut(),
    //             position,
    //             w.as_mut(),
    //             12,
    //             DynStack::new(&mut GlobalMemBuffer::new(
    //                 insert_rows_and_cols_clobber_req::<f64>(n + r, position, r, 12).unwrap(),
    //             )),
    //         )
    //         .unwrap();

    //         let a_reconstructed = reconstruct_matrix(a.as_ref());
    //         for j in 0..n + r {
    //             for i in j..n + r {
    //                 assert_approx_eq!(a_reconstructed[(i, j)], a_orig[(i, j)], 1e-4);
    //             }
    //         }
    //     }
    // }
}
