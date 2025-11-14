use crate::internal_prelude::*;
use core::ops::Mul;
use faer_traits::Real;
extern crate alloc;
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ApproxEq<T> {
	pub abs_tol: T,
	pub rel_tol: T,
}
pub struct CwiseMat<Cmp>(pub Cmp);
impl<T: RealField> ApproxEq<T> {
	#[inline]
	pub fn eps() -> Self {
		Self {
			abs_tol: eps::<T>() * from_f64::<T>(128.0),
			rel_tol: eps::<T>() * from_f64::<T>(128.0),
		}
	}
}
impl<T: RealField> Mul<T> for ApproxEq<T> {
	type Output = ApproxEq<T>;

	#[inline]
	fn mul(self, rhs: Real<T>) -> Self::Output {
		ApproxEq {
			abs_tol: &self.abs_tol * &rhs,
			rel_tol: &self.rel_tol * &rhs,
		}
	}
}
#[derive(Copy, Clone, Debug)]
pub struct ApproxEqError;
#[derive(Clone, Debug)]
pub enum CwiseMatError<Rows: Shape, Cols: Shape, Error> {
	DimMismatch,
	Elements(alloc::vec::Vec<(crate::Idx<Rows>, crate::Idx<Cols>, Error)>),
}
#[derive(Clone, Debug)]
pub enum CwiseColError<Rows: Shape, Error> {
	DimMismatch,
	Elements(alloc::vec::Vec<(crate::Idx<Rows>, Error)>),
}
#[derive(Clone, Debug)]
pub enum CwiseRowError<Cols: Shape, Error> {
	DimMismatch,
	Elements(alloc::vec::Vec<(crate::Idx<Cols>, Error)>),
}
impl<R: RealField, T: ComplexField<Real = R>> equator::Cmp<T, T>
	for ApproxEq<R>
{
	fn test(&self, lhs: &T, rhs: &T) -> bool {
		let Self { abs_tol, rel_tol } = self;
		let diff = (lhs - rhs).abs();
		let max = lhs.abs().fmax(rhs.abs());
		(max == zero() && diff <= *abs_tol)
			|| (diff <= *abs_tol || diff <= rel_tol * max)
	}
}
impl<
	T: ComplexField,
	Rows: Shape,
	Cols: Shape,
	L: AsMatRef<T = T, Rows = Rows, Cols = Cols>,
	R: AsMatRef<T = T, Rows = Rows, Cols = Cols>,
	Cmp: equator::Cmp<T, T>,
> equator::Cmp<L, R> for CwiseMat<Cmp>
{
	fn test(&self, lhs: &L, rhs: &R) -> bool {
		let lhs = lhs.as_mat_ref();
		let rhs = rhs.as_mat_ref();
		if lhs.nrows() != rhs.nrows() || lhs.ncols() != rhs.ncols() {
			return false;
		}
		for j in 0..lhs.ncols().unbound() {
			let j = lhs.ncols().checked_idx(j);
			for i in 0..lhs.nrows().unbound() {
				let i = lhs.nrows().checked_idx(i);
				if !self.0.test(&lhs[(i, j)], &rhs[(i, j)]) {
					return false;
				}
			}
		}
		true
	}
}
