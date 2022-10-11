pub mod compute;
pub mod solve;
pub mod update;

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use dyn_stack::{DynStack, GlobalMemBuffer};
    use rand::random;

    use super::*;
    use compute::*;
    use faer_core::mul::triangular::BlockStructure;
    use faer_core::{mat, mul, Mat, MatRef};
    use solve::*;
    use update::*;

    fn reconstruct_matrix(cholesky_factors: MatRef<'_, f64>) -> Mat<f64> {
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

        use mul::triangular::BlockStructure::*;
        mul::triangular::matmul(
            a_reconstructed.as_mut(),
            BlockStructure::Rectangular,
            lxd.as_ref(),
            BlockStructure::TriangularLower,
            cholesky_factors.transpose(),
            BlockStructure::UnitTriangularUpper,
            None,
            &1.0,
            12,
            DynStack::new(&mut dyn_stack::GlobalMemBuffer::new(
                mul::triangular::matmul_req::<f64>(
                    Rectangular,
                    TriangularLower,
                    TriangularUpper,
                    n,
                    n,
                    n,
                    12,
                )
                .unwrap(),
            )),
        );

        a_reconstructed
    }

    #[test]
    fn test_roundtrip() {
        for n in 0..512 {
            let mut a = Mat::with_dims(|_, _| random::<f64>(), n, n);
            let a_orig = a.clone();

            raw_cholesky_in_place(
                a.as_mut(),
                12,
                DynStack::new(&mut GlobalMemBuffer::new(
                    raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
                )),
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
        let mut a = Mat::with_dims(|_, _| random::<f64>(), n, n);
        let mut rhs = Mat::with_dims(|_, _| random::<f64>(), n, k);
        let a_orig = a.clone();
        let rhs_orig = rhs.clone();

        raw_cholesky_in_place(
            a.as_mut(),
            12,
            DynStack::new(&mut GlobalMemBuffer::new(
                raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
            )),
        );

        solve_in_place(
            a.as_ref(),
            rhs.as_mut(),
            12,
            DynStack::new(&mut GlobalMemBuffer::new(
                solve_in_place_req::<f64>(n, k, 12).unwrap(),
            )),
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
            &1.0,
            12,
            DynStack::new(&mut GlobalMemBuffer::new(
                mul::triangular::matmul_req::<f64>(
                    Rectangular,
                    TriangularLower,
                    Rectangular,
                    n,
                    k,
                    n,
                    12,
                )
                .unwrap(),
            )),
        );

        mul::triangular::matmul(
            result.as_mut(),
            Rectangular,
            a_orig.as_ref().transpose(),
            StrictTriangularUpper,
            rhs.as_ref(),
            Rectangular,
            Some(&1.0),
            &1.0,
            12,
            DynStack::new(&mut GlobalMemBuffer::new(
                mul::triangular::matmul_req::<f64>(
                    Rectangular,
                    StrictTriangularUpper,
                    Rectangular,
                    n,
                    k,
                    n,
                    12,
                )
                .unwrap(),
            )),
        );

        for j in 0..k {
            for i in 0..n {
                assert_approx_eq!(result[(i, j)], rhs_orig[(i, j)]);
            }
        }
    }

    #[test]
    fn test_update() {
        use mul::triangular::BlockStructure::*;
        let random = |_, _| random::<f64>();

        for k in [0, 1, 2, 3, 4, 5] {
            let n = 511;
            let mut a = Mat::with_dims(random, n, n);
            let mut a_updated = a.clone();
            let mut w = Mat::with_dims(random, n, k);
            let mut alpha = Mat::with_dims(random, k, 1);
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
                w_alpha.as_ref(),
                Rectangular,
                w.as_ref().transpose(),
                Rectangular,
                Some(&1.0),
                &1.0,
                12,
                DynStack::new(&mut GlobalMemBuffer::new(
                    mul::triangular::matmul_req::<f64>(
                        TriangularLower,
                        Rectangular,
                        Rectangular,
                        n,
                        n,
                        k,
                        12,
                    )
                    .unwrap(),
                )),
            );

            raw_cholesky_in_place(
                a.as_mut(),
                12,
                DynStack::new(&mut GlobalMemBuffer::new(
                    raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
                )),
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
        let a_orig = mat![
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 5.0, 0.0, 0.0],
            [3.0, 6.0, 8.0, 0.0],
            [4.0, 7.0, 9.0, 10.0],
        ];

        {
            let mut a = a_orig.clone();
            let n = a.nrows();
            let r = 2;

            raw_cholesky_in_place(
                a.as_mut(),
                12,
                DynStack::new(&mut GlobalMemBuffer::new(
                    raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
                )),
            );

            delete_rows_and_cols_clobber(
                a.as_mut(),
                &mut [1, 3],
                DynStack::new(&mut GlobalMemBuffer::new(
                    delete_rows_and_cols_clobber_req::<f64>(n, r).unwrap(),
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
                12,
                DynStack::new(&mut GlobalMemBuffer::new(
                    raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
                )),
            );

            delete_rows_and_cols_clobber(
                a.as_mut(),
                &mut [0, 2],
                DynStack::new(&mut GlobalMemBuffer::new(
                    delete_rows_and_cols_clobber_req::<f64>(n, r).unwrap(),
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
                12,
                DynStack::new(&mut GlobalMemBuffer::new(
                    raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
                )),
            );

            delete_rows_and_cols_clobber(
                a.as_mut(),
                &mut [0, 2, 3],
                DynStack::new(&mut GlobalMemBuffer::new(
                    delete_rows_and_cols_clobber_req::<f64>(n, r).unwrap(),
                )),
            );

            let a_reconstructed = reconstruct_matrix(a.as_ref().submatrix(0, 0, n - r, n - r));
            assert_approx_eq!(a_reconstructed[(0, 0)], a_orig[(1, 1)]);
        }
    }

    #[test]
    fn test_insert() {
        let a_orig = mat![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 5.0, 6.0, 7.0],
            [3.0, 6.0, 8.0, 9.0],
            [4.0, 7.0, 9.0, 10.0],
        ];

        {
            let mut a = a_orig.clone();

            let mut w = mat![
                [11.0, 17.0],
                [12.0, 18.0],
                [13.0, 14.0],
                [14.0, 20.0],
                [15.0, 21.0],
                [16.0, 22.0],
            ];

            let a_new = mat![
                [1.0, 2.0, 11.0, 17.0, 3.0, 4.0],
                [2.0, 5.0, 12.0, 18.0, 6.0, 7.0],
                [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 14.0, 20.0, 21.0, 22.0],
                [3.0, 6.0, 15.0, 21.0, 8.0, 9.0],
                [4.0, 7.0, 16.0, 22.0, 9.0, 10.0],
            ];

            let n = a.nrows();
            let r = w.ncols();
            let position = 2;

            raw_cholesky_in_place(
                a.as_mut(),
                12,
                DynStack::new(&mut GlobalMemBuffer::new(
                    raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
                )),
            );

            a.resize_with(|_, _| 0.0, n + r, n + r);
            insert_rows_and_cols_clobber(
                a.as_mut(),
                position,
                w.as_mut(),
                12,
                DynStack::new(&mut GlobalMemBuffer::new(
                    insert_rows_and_cols_clobber_req::<f64>(n, position, r, 12).unwrap(),
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

            let mut w = mat![
                [11.0, 12.0],
                [12.0, 18.0],
                [13.0, 19.0],
                [14.0, 20.0],
                [15.0, 21.0],
                [16.0, 22.0],
            ];

            let a_new = mat![
                [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                [12.0, 18.0, 19.0, 20.0, 21.0, 22.0],
                [13.0, 19.0, 1.0, 2.0, 3.0, 4.0],
                [14.0, 20.0, 2.0, 5.0, 6.0, 7.0],
                [15.0, 21.0, 3.0, 6.0, 8.0, 9.0],
                [16.0, 22.0, 4.0, 7.0, 9.0, 10.0],
            ];

            let n = a.nrows();
            let r = w.ncols();
            let position = 0;

            raw_cholesky_in_place(
                a.as_mut(),
                12,
                DynStack::new(&mut GlobalMemBuffer::new(
                    raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
                )),
            );

            a.resize_with(|_, _| 0.0, n + r, n + r);
            insert_rows_and_cols_clobber(
                a.as_mut(),
                position,
                w.as_mut(),
                12,
                DynStack::new(&mut GlobalMemBuffer::new(
                    insert_rows_and_cols_clobber_req::<f64>(n, position, r, 12).unwrap(),
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

            let mut w = mat![
                [11.0, 17.0],
                [12.0, 18.0],
                [13.0, 19.0],
                [14.0, 20.0],
                [15.0, 16.0],
                [16.0, 22.0],
            ];

            let a_new = mat![
                [1.0, 2.0, 3.0, 4.0, 11.0, 17.0],
                [2.0, 5.0, 6.0, 7.0, 12.0, 18.0],
                [3.0, 6.0, 8.0, 9.0, 13.0, 19.0],
                [4.0, 7.0, 9.0, 10.0, 14.0, 20.0],
                [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0, 16.0, 22.0],
            ];

            let n = a.nrows();
            let r = w.ncols();
            let position = 4;

            raw_cholesky_in_place(
                a.as_mut(),
                12,
                DynStack::new(&mut GlobalMemBuffer::new(
                    raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
                )),
            );

            a.resize_with(|_, _| 0.0, n + r, n + r);
            insert_rows_and_cols_clobber(
                a.as_mut(),
                position,
                w.as_mut(),
                12,
                DynStack::new(&mut GlobalMemBuffer::new(
                    insert_rows_and_cols_clobber_req::<f64>(n, position, r, 12).unwrap(),
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
