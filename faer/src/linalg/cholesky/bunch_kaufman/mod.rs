pub mod factor;
pub mod solve;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, c64, internal_prelude::*, stats::prelude::*, Col, Mat};
    use dyn_stack::GlobalMemBuffer;
    use factor::BunchKaufmanParams;
    use num_complex::ComplexFloat;

    #[test]
    fn test_real() {
        let rng = &mut StdRng::seed_from_u64(0);

        for n in [3, 6, 19, 100, 421] {
            let a = CwiseMatDistribution {
                nrows: n,
                ncols: n,
                dist: StandardNormal,
            }
            .rand::<Mat<f64>>(rng);

            let a = &a + a.adjoint();
            let rhs = CwiseMatDistribution {
                nrows: n,
                ncols: 2,
                dist: StandardNormal,
            }
            .rand::<Mat<f64>>(rng);

            let mut ldl = a.clone();
            let mut subdiag = Col::<f64>::zeros(n);

            let mut perm = vec![0usize; n];
            let mut perm_inv = vec![0; n];

            let params = auto!(f64);
            let mut mem = GlobalMemBuffer::new(
                factor::cholesky_in_place_scratch::<usize, f64>(n, Par::Seq, params).unwrap(),
            );
            let (_, perm) = factor::cholesky_in_place(
                ldl.as_mut(),
                subdiag.as_mut(),
                Default::default(),
                &mut perm,
                &mut perm_inv,
                Par::Seq,
                DynStack::new(&mut mem),
                params,
            );

            let mut mem = GlobalMemBuffer::new(
                solve::solve_in_place_scratch::<usize, f64>(n, rhs.ncols(), Par::Seq).unwrap(),
            );
            let mut x = rhs.clone();
            solve::solve_in_place_with_conj(
                ldl.as_ref(),
                subdiag.as_ref(),
                Conj::No,
                perm.rb(),
                x.as_mut(),
                Par::Seq,
                DynStack::new(&mut mem),
            );

            let err = &a * &x - &rhs;
            let mut max = 0.0;
            zipped!(err.as_ref()).for_each(|unzipped!(err)| {
                let err = err.abs();
                if err > max {
                    max = err
                }
            });
            assert!(max < 1e-9);
        }
    }

    #[test]
    fn test_cplx() {
        let rng = &mut StdRng::seed_from_u64(0);

        for n in [2, 3, 6, 19, 100, 421] {
            let distribution = ComplexDistribution::new(StandardNormal, StandardNormal);
            let a = CwiseMatDistribution {
                nrows: n,
                ncols: n,
                dist: distribution,
            }
            .rand::<Mat<c64>>(rng);

            let a = &a + a.adjoint();
            let rhs = CwiseMatDistribution {
                nrows: n,
                ncols: 2,
                dist: distribution,
            }
            .rand::<Mat<c64>>(rng);

            let mut ldl = a.clone();
            let mut subdiag = Col::<c64>::zeros(n);

            let mut perm = vec![0usize; n];
            let mut perm_inv = vec![0; n];

            let params = BunchKaufmanParams {
                pivoting: factor::PivotingStrategy::Diagonal,
                blocksize: 32,
                ..auto!(c32)
            };
            let mut mem = GlobalMemBuffer::new(
                factor::cholesky_in_place_scratch::<usize, c64>(n, Par::Seq, params).unwrap(),
            );
            let (_, perm) = factor::cholesky_in_place(
                ldl.as_mut(),
                subdiag.as_mut(),
                Default::default(),
                &mut perm,
                &mut perm_inv,
                Par::Seq,
                DynStack::new(&mut mem),
                params,
            );

            let mut x = rhs.clone();
            let mut mem = GlobalMemBuffer::new(
                solve::solve_in_place_scratch::<usize, c64>(n, rhs.ncols(), Par::Seq).unwrap(),
            );
            solve::solve_in_place_with_conj(
                ldl.as_ref(),
                subdiag.as_ref(),
                Conj::Yes,
                perm.rb(),
                x.as_mut(),
                Par::Seq,
                DynStack::new(&mut mem),
            );

            let err = a.conjugate() * &x - &rhs;
            let mut max = 0.0;
            zipped!(err.as_ref()).for_each(|unzipped!(err)| {
                let err = err.abs();
                if err > max {
                    max = err
                }
            });
            for i in 0..n {
                assert!(ldl[(i, i)].im == 0.0);
            }
            assert!(max < 1e-9);
        }
    }
}
