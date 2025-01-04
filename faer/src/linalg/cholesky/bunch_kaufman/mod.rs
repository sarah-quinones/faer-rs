pub mod factor;
pub mod solve;

pub mod inverse;
pub mod reconstruct;

#[cfg(test)]
mod tests {
	use super::*;
	use crate::internal_prelude::*;
	use crate::stats::prelude::*;
	use crate::{assert, c64};
	use dyn_stack::MemBuffer;
	use factor::BunchKaufmanParams;
	use num_complex::ComplexFloat;
	use std::vec;

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
			let mut subdiag = Diag::<f64>::zeros(n);

			let mut perm = vec![0usize; n];
			let mut perm_inv = vec![0; n];

			let params = Default::default();
			let mut mem = MemBuffer::new(factor::cholesky_in_place_scratch::<usize, f64>(n, Par::Seq, params));
			let (_, perm) = factor::cholesky_in_place(
				ldl.as_mut(),
				subdiag.as_mut(),
				Default::default(),
				&mut perm,
				&mut perm_inv,
				Par::Seq,
				MemStack::new(&mut mem),
				params,
			);

			let mut mem = MemBuffer::new(solve::solve_in_place_scratch::<usize, f64>(n, rhs.ncols(), Par::Seq));
			let mut x = rhs.clone();
			solve::solve_in_place_with_conj(
				ldl.as_ref(),
				ldl.diagonal(),
				subdiag.as_ref(),
				Conj::No,
				perm.rb(),
				x.as_mut(),
				Par::Seq,
				MemStack::new(&mut mem),
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
			let mut subdiag = Diag::<c64>::zeros(n);

			let mut perm = vec![0usize; n];
			let mut perm_inv = vec![0; n];

			let params = BunchKaufmanParams {
				pivoting: factor::PivotingStrategy::Diagonal,
				blocksize: 4,
				..auto!(c64)
			};
			let mut mem = MemBuffer::new(factor::cholesky_in_place_scratch::<usize, c64>(n, Par::Seq, params.into()));
			let (_, perm) = factor::cholesky_in_place(
				ldl.as_mut(),
				subdiag.as_mut(),
				Default::default(),
				&mut perm,
				&mut perm_inv,
				Par::Seq,
				MemStack::new(&mut mem),
				params.into(),
			);

			let mut x = rhs.clone();
			let mut mem = MemBuffer::new(solve::solve_in_place_scratch::<usize, c64>(n, rhs.ncols(), Par::Seq));
			solve::solve_in_place_with_conj(
				ldl.as_ref(),
				ldl.diagonal(),
				subdiag.as_ref(),
				Conj::Yes,
				perm.rb(),
				x.as_mut(),
				Par::Seq,
				MemStack::new(&mut mem),
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
