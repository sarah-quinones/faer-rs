#![allow(missing_docs)]

use crate::internal_prelude::*;
use rand::distributions::Distribution;

mod meanvar;
pub use meanvar::{NanHandling, col_mean, col_varm, row_mean, row_varm};

pub mod prelude {
	pub use num_complex::ComplexDistribution;
	pub use rand::prelude::*;
	pub use rand_distr::{Standard, StandardNormal};

	pub use super::{CwiseColDistribution, CwiseMatDistribution, CwiseRowDistribution, DistributionExt, UnitaryMat};
}

pub trait DistributionExt {
	fn rand<T>(&self, rng: &mut (impl ?Sized + rand::Rng)) -> T
	where
		Self: Distribution<T>,
	{
		self.sample(rng)
	}
}
impl<T: ?Sized> DistributionExt for T {}

#[derive(Copy, Clone, Debug)]
pub struct CwiseMatDistribution<Rows: Shape, Cols: Shape, D> {
	pub nrows: Rows,
	pub ncols: Cols,
	pub dist: D,
}

#[derive(Copy, Clone, Debug)]
pub struct CwiseColDistribution<Rows: Shape, D> {
	pub nrows: Rows,
	pub dist: D,
}

#[derive(Copy, Clone, Debug)]
pub struct CwiseRowDistribution<Cols: Shape, D> {
	pub ncols: Cols,
	pub dist: D,
}

#[derive(Copy, Clone, Debug)]
pub struct UnitaryMat<Dim: Shape, D> {
	pub dim: Dim,
	pub standard_normal: D,
}

impl<T, Rows: Shape, Cols: Shape, D: Distribution<T>> Distribution<Mat<T, Rows, Cols>> for CwiseMatDistribution<Rows, Cols, D> {
	#[inline]
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Mat<T, Rows, Cols> {
		Mat::from_fn(self.nrows, self.ncols, |_, _| self.dist.sample(rng))
	}
}

impl<T, Rows: Shape, D: Distribution<T>> Distribution<Col<T, Rows>> for CwiseColDistribution<Rows, D> {
	#[inline]
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Col<T, Rows> {
		Col::from_fn(self.nrows, |_| self.dist.sample(rng))
	}
}

impl<T, Cols: Shape, D: Distribution<T>> Distribution<Row<T, Cols>> for CwiseRowDistribution<Cols, D> {
	#[inline]
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Row<T, Cols> {
		Row::from_fn(self.ncols, |_| self.dist.sample(rng))
	}
}

impl<T: ComplexField, D: Distribution<T>> Distribution<Mat<T>> for UnitaryMat<usize, D> {
	#[math]
	fn sample<R: rand::prelude::Rng + ?Sized>(&self, rng: &mut R) -> Mat<T> {
		let qr = CwiseMatDistribution {
			nrows: self.dim,
			ncols: self.dim,
			dist: &self.standard_normal,
		}
		.sample(rng)
		.qr();

		let r_diag = qr.R().diagonal().column_vector();
		let mut q = qr.compute_Q();

		for j in 0..self.dim {
			let r = r_diag.read(j);
			let r = if r == zero() { one() } else { mul_real(r, recip(abs(r))) };

			z!(q.as_mut().col_mut(j)).for_each(|uz!(q)| {
				*q = *q * r;
			});
		}

		q
	}
}
