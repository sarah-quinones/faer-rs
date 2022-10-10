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
    use faer_core::{mul, Mat, MatRef};

    fn reconstruct_matrix(cholesky_factor: MatRef<'_, f64>) -> Mat<f64> {
        let n = cholesky_factor.nrows();

        let mut a_reconstructed = Mat::zeros(n, n);

        use mul::triangular::BlockStructure::*;
        mul::triangular::matmul(
            a_reconstructed.as_mut(),
            BlockStructure::Rectangular,
            cholesky_factor,
            BlockStructure::TriangularLower,
            cholesky_factor.transpose(),
            BlockStructure::TriangularUpper,
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

    fn random_positive_definite(n: usize) -> Mat<f64> {
        let a = Mat::with_dims(|_, _| random::<f64>(), n, n);
        let mut ata = Mat::zeros(n, n);

        mul::matmul(
            ata.as_mut(),
            a.as_ref().transpose(),
            a.as_ref(),
            None,
            &1.0,
            12,
            DynStack::new(&mut GlobalMemBuffer::new(
                mul::matmul_req::<f64>(n, n, n, 12).unwrap(),
            )),
        );

        ata
    }

    #[test]
    fn test_roundtrip() {
        for n in 0..512 {
            let mut a = random_positive_definite(n);
            let a_orig = a.clone();

            raw_cholesky_in_place(
                a.as_mut(),
                12,
                DynStack::new(&mut GlobalMemBuffer::new(
                    raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
                )),
            )
            .unwrap();

            let a_reconstructed = reconstruct_matrix(a.as_ref());

            for j in 0..n {
                for i in j..n {
                    assert_approx_eq!(a_reconstructed[(i, j)], a_orig[(i, j)]);
                }
            }
        }
    }
}
