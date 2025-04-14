use crate::internal_prelude::*;
use crate::{Scale, assert, col, diag, get_global_parallelism, mat, perm, row};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

extern crate alloc;

macro_rules! impl_perm {
	($lhs: ty, $rhs: ty, $out: ty) => {
		impl<I: Index, T: ComplexField, TT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> Mul<$rhs> for $lhs {
			type Output = $out;

			#[track_caller]
			fn mul(self, other: $rhs) -> Self::Output {
				self.as_ref().mul(other.as_ref())
			}
		}
	};
}
macro_rules! impl_1d_perm {
	($lhs: ty, $rhs: ty, $out: ty) => {
		impl<I: Index, T: ComplexField, TT: Conjugate<Canonical = T>, Len: Shape> Mul<$rhs> for $lhs {
			type Output = $out;

			#[track_caller]
			fn mul(self, other: $rhs) -> Self::Output {
				self.as_ref().mul(other.as_ref())
			}
		}
	};
}

macro_rules! impl_perm_perm {
	($lhs: ty, $rhs: ty, $out: ty) => {
		impl<I: Index> Mul<$rhs> for $lhs {
			type Output = $out;

			#[track_caller]
			fn mul(self, other: $rhs) -> Self::Output {
				self.as_ref().mul(other.as_ref())
			}
		}
	};
}

macro_rules! impl_scalar_mul {
	($lhs: ty, $rhs: ty, $out: ty) => {
		impl<T: ComplexField, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> Mul<$rhs> for $lhs {
			type Output = $out;

			#[track_caller]
			fn mul(self, other: $rhs) -> Self::Output {
				self.mul(other.as_ref())
			}
		}
		impl<T: ComplexField, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> Mul<$rhs> for &$lhs {
			type Output = $out;

			#[track_caller]
			fn mul(self, other: $rhs) -> Self::Output {
				self.clone().mul(other.as_ref())
			}
		}
	};
}

macro_rules! impl_mul_scalar {
	($lhs: ty, $rhs: ty, $out: ty) => {
		impl<T: ComplexField, LhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> Mul<$rhs> for $lhs {
			type Output = $out;

			#[track_caller]
			fn mul(self, other: $rhs) -> Self::Output {
				self.as_ref().mul(other)
			}
		}
		impl<T: ComplexField, LhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> Mul<&$rhs> for $lhs {
			type Output = $out;

			#[track_caller]
			fn mul(self, other: &$rhs) -> Self::Output {
				self.as_ref().mul(other.clone())
			}
		}
	};
}

macro_rules! impl_div_scalar {
	($lhs: ty, $rhs: ty, $out: ty) => {
		impl<T: ComplexField, LhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> Div<$rhs> for $lhs {
			type Output = $out;

			#[track_caller]
			fn div(self, other: $rhs) -> Self::Output {
				self.as_ref().mul(Scale(recip(&other.0)))
			}
		}
		impl<T: ComplexField, LhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> Div<&$rhs> for $lhs {
			type Output = $out;

			#[track_caller]
			fn div(self, other: &$rhs) -> Self::Output {
				self.as_ref().mul(Scale(recip(&other.0)))
			}
		}
	};
}

macro_rules! impl_mul_primitive {
	($rhs: ty, $out: ty) => {
		impl<T: ComplexField, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> Mul<$rhs> for f64 {
			type Output = $out;

			#[track_caller]
			fn mul(self, other: $rhs) -> Self::Output {
				Scale(T::from_f64_impl(self)).mul(other)
			}
		}

		impl<T: ComplexField, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> Mul<f64> for $rhs {
			type Output = $out;

			#[track_caller]
			fn mul(self, other: f64) -> Self::Output {
				self.mul(Scale(T::from_f64_impl(other)))
			}
		}
		impl<T: ComplexField, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> Div<f64> for $rhs {
			type Output = $out;

			#[track_caller]
			fn div(self, other: f64) -> Self::Output {
				self.mul(Scale(T::from_f64_impl(f64::recip(other))))
			}
		}
	};
}

macro_rules! impl_mul_assign_primitive {
	($lhs: ty) => {
		impl<LhsT: ComplexField, Rows: Shape, Cols: Shape> MulAssign<f64> for $lhs {
			#[track_caller]
			fn mul_assign(&mut self, other: f64) {
				self.mul_assign(Scale(LhsT::from_f64_impl(other)))
			}
		}
		impl<LhsT: ComplexField, Rows: Shape, Cols: Shape> DivAssign<f64> for $lhs {
			#[track_caller]
			fn div_assign(&mut self, other: f64) {
				self.mul_assign(Scale(LhsT::from_f64_impl(f64::recip(other))))
			}
		}
	};
}

macro_rules! impl_mul_assign_scalar {
	($lhs: ty, $rhs: ty) => {
		impl<LhsT: ComplexField, Rows: Shape, Cols: Shape> MulAssign<$rhs> for $lhs {
			#[track_caller]
			fn mul_assign(&mut self, other: $rhs) {
				self.as_mut().mul_assign(other)
			}
		}

		impl<LhsT: ComplexField, Rows: Shape, Cols: Shape> MulAssign<&$rhs> for $lhs {
			#[track_caller]
			fn mul_assign(&mut self, other: &$rhs) {
				self.as_mut().mul_assign(other.clone())
			}
		}
	};
}

macro_rules! impl_div_assign_scalar {
	($lhs: ty, $rhs: ty) => {
		impl<LhsT: ComplexField, Rows: Shape, Cols: Shape> DivAssign<$rhs> for $lhs {
			#[track_caller]
			fn div_assign(&mut self, other: $rhs) {
				self.as_mut().mul_assign(Scale(recip(&other.0)))
			}
		}
		impl<LhsT: ComplexField, Rows: Shape, Cols: Shape> DivAssign<&$rhs> for $lhs {
			#[track_caller]
			fn div_assign(&mut self, other: &$rhs) {
				self.as_mut().mul_assign(Scale(recip(&other.0)))
			}
		}
	};
}

macro_rules! impl_1d_scalar_mul {
	($lhs: ty, $rhs: ty, $out: ty) => {
		impl<T: ComplexField, RhsT: Conjugate<Canonical = T>, Len: Shape> Mul<$rhs> for $lhs {
			type Output = $out;

			#[track_caller]
			fn mul(self, other: $rhs) -> Self::Output {
				self.mul(other.as_ref())
			}
		}
		impl<T: ComplexField, RhsT: Conjugate<Canonical = T>, Len: Shape> Mul<$rhs> for &$lhs {
			type Output = $out;

			#[track_caller]
			fn mul(self, other: $rhs) -> Self::Output {
				self.clone().mul(other.as_ref())
			}
		}
	};
}

macro_rules! impl_1d_mul_scalar {
	($lhs: ty, $rhs: ty, $out: ty) => {
		impl<T: ComplexField, LhsT: Conjugate<Canonical = T>, Len: Shape> Mul<$rhs> for $lhs {
			type Output = $out;

			#[track_caller]
			fn mul(self, other: $rhs) -> Self::Output {
				self.as_ref().mul(other)
			}
		}
		impl<T: ComplexField, LhsT: Conjugate<Canonical = T>, Len: Shape> Mul<&$rhs> for $lhs {
			type Output = $out;

			#[track_caller]
			fn mul(self, other: &$rhs) -> Self::Output {
				self.as_ref().mul(other.clone())
			}
		}
	};
}

macro_rules! impl_1d_div_scalar {
	($lhs: ty, $rhs: ty, $out: ty) => {
		impl<T: ComplexField, LhsT: Conjugate<Canonical = T>, Len: Shape> Div<$rhs> for $lhs {
			type Output = $out;

			#[track_caller]
			fn div(self, other: $rhs) -> Self::Output {
				self.as_ref().mul(Scale(recip(&other.0)))
			}
		}
		impl<T: ComplexField, LhsT: Conjugate<Canonical = T>, Len: Shape> Div<&$rhs> for $lhs {
			type Output = $out;

			#[track_caller]
			fn div(self, other: &$rhs) -> Self::Output {
				self.as_ref().mul(Scale(recip(&other.0)))
			}
		}
	};
}

macro_rules! impl_1d_mul_primitive {
	($rhs: ty, $out: ty) => {
		impl<T: ComplexField, RhsT: Conjugate<Canonical = T>, Len: Shape> Mul<$rhs> for f64 {
			type Output = $out;

			#[track_caller]
			fn mul(self, other: $rhs) -> Self::Output {
				Scale(T::from_f64_impl(self)).mul(other)
			}
		}

		impl<T: ComplexField, RhsT: Conjugate<Canonical = T>, Len: Shape> Mul<f64> for $rhs {
			type Output = $out;

			#[track_caller]
			fn mul(self, other: f64) -> Self::Output {
				self.mul(Scale(T::from_f64_impl(other)))
			}
		}
		impl<T: ComplexField, RhsT: Conjugate<Canonical = T>, Len: Shape> Div<f64> for $rhs {
			type Output = $out;

			#[track_caller]
			fn div(self, other: f64) -> Self::Output {
				self.mul(Scale(T::from_f64_impl(f64::recip(other))))
			}
		}
	};
}

macro_rules! impl_1d_mul_assign_primitive {
	($lhs: ty) => {
		impl<LhsT: ComplexField, Len: Shape> MulAssign<f64> for $lhs {
			#[track_caller]
			fn mul_assign(&mut self, other: f64) {
				self.mul_assign(Scale(LhsT::from_f64_impl(other)))
			}
		}
		impl<LhsT: ComplexField, Len: Shape> DivAssign<f64> for $lhs {
			#[track_caller]
			fn div_assign(&mut self, other: f64) {
				self.mul_assign(Scale(LhsT::from_f64_impl(f64::recip(other))))
			}
		}
	};
}

macro_rules! impl_1d_mul_assign_scalar {
	($lhs: ty, $rhs: ty) => {
		impl<LhsT: ComplexField, Len: Shape> MulAssign<$rhs> for $lhs {
			#[track_caller]
			fn mul_assign(&mut self, other: $rhs) {
				self.as_mut().mul_assign(other)
			}
		}
		impl<LhsT: ComplexField, Len: Shape> MulAssign<&$rhs> for $lhs {
			#[track_caller]
			fn mul_assign(&mut self, other: &$rhs) {
				self.as_mut().mul_assign(other.clone())
			}
		}
	};
}

macro_rules! impl_1d_div_assign_scalar {
	($lhs: ty, $rhs: ty) => {
		impl<LhsT: ComplexField, Len: Shape> DivAssign<$rhs> for $lhs {
			#[track_caller]
			fn div_assign(&mut self, other: $rhs) {
				self.as_mut().mul_assign(Scale(recip(&other.0)))
			}
		}
		impl<LhsT: ComplexField, Len: Shape> DivAssign<&$rhs> for $lhs {
			#[track_caller]
			fn div_assign(&mut self, other: &$rhs) {
				self.as_mut().mul_assign(Scale(recip(&other.0)))
			}
		}
	};
}

macro_rules! impl_binop {
	({$(
		impl<$($ty_param: ident  $(: $bound: tt)?),* $(,)?>
		$trait: ident<$rhs: ty> for $lhs: ty {
			type Output = $out: ty;

			fn $name: ident($self: ident, $rhs_: ident : _$(,)?) $block: block
		}
	)*}) => {$(
		impl<$($ty_param  $(: $bound)?, )*>
		$trait<&$rhs> for &$lhs {
			type Output = $out;

			#[track_caller]
			fn $name ($self, $rhs_: &$rhs) -> Self::Output $block
		}

		impl<$($ty_param  $(: $bound)?, )*>
		$trait<$rhs> for $lhs {
			type Output = $out;

			#[track_caller]
			fn $name ($self, $rhs_: $rhs) -> Self::Output {
				$trait::$name(&$self, &$rhs_)
			}
		}
		impl<$($ty_param  $(: $bound)?, )*>
		$trait<$rhs> for &$lhs {
			type Output = $out;

			#[track_caller]
			fn $name ($self, $rhs_: $rhs) -> Self::Output {
				$trait::$name($self, &$rhs_)
			}
		}
		impl<$($ty_param  $(: $bound)?, )*>
		$trait<&$rhs> for $lhs {
			type Output = $out;

			#[track_caller]
			fn $name ($self, $rhs_: &$rhs) -> Self::Output {
				$trait::$name(&$self, $rhs_)
			}
		}
	)*};
}

macro_rules! impl_op {
	({$(
		impl<$($ty_param: ident  $(: $bound: tt)?),* $(,)?>
		$trait: ident for $lhs: ty {
			type Output = $out: ty;

			fn $name: ident($self: ident$(,)?) $block: block
		}
	)*}) => {$(
		impl<$($ty_param  $(: $bound)?, )*>
		$trait for &$lhs {
			type Output = $out;

			#[track_caller]
			fn $name ($self) -> Self::Output $block
		}

		impl<$($ty_param  $(: $bound)?, )*>
		$trait for $lhs {
			type Output = $out;

			#[track_caller]
			fn $name ($self) -> Self::Output {
				$trait::$name(&$self)
			}
		}
	)*};
}

macro_rules! impl_op_assign {
	({$(
		impl<$($ty_param: ident  $(: $bound: tt)?),* $(,)?>
		$trait: ident<$rhs: ty> for $lhs: ty {
			fn $name: ident(&mut $self: ident, $rhs_: ident : _$(,)?) $block: block
		}
	)*}) => {$(
		impl<$($ty_param  $(: $bound)?, )*>
		$trait<&$rhs> for $lhs {
			#[track_caller]
			fn $name (&mut $self, $rhs_: &$rhs)  $block
		}

		impl<$($ty_param  $(: $bound)?, )*>
		$trait<$rhs> for $lhs {
			#[track_caller]
			fn $name (&mut $self, $rhs_: $rhs)  {
				$trait::$name($self, &$rhs_)
			}
		}
	)*};
}

impl<
	LT: PartialEq<RT>,
	LRows: Shape,
	LCols: Shape,
	LRStride: Stride,
	LCStride: Stride,
	RT,
	RRows: Shape,
	RCols: Shape,
	RRStride: Stride,
	RCStride: Stride,
	L: for<'a> Reborrow<'a, Target = mat::Ref<'a, LT, LRows, LCols, LRStride, LCStride>>,
	R: for<'a> Reborrow<'a, Target = mat::Ref<'a, RT, RRows, RCols, RRStride, RCStride>>,
> PartialEq<mat::generic::Mat<R>> for mat::generic::Mat<L>
{
	fn eq(&self, other: &mat::generic::Mat<R>) -> bool {
		fn imp<LT: PartialEq<RT>, RT>(l: MatRef<'_, LT>, r: MatRef<'_, RT>) -> bool {
			if l.nrows() != r.nrows() {
				return false;
			}

			with_dim!(M, l.nrows());
			with_dim!(N, l.ncols());

			let l = l.as_shape(M, N);
			let r = r.as_shape(M, N);

			for j in N.indices() {
				for i in M.indices() {
					if l[(i, j)] != r[(i, j)] {
						return false;
					}
				}
			}
			true
		}
		imp(self.rb().as_dyn().as_dyn_stride(), other.rb().as_dyn().as_dyn_stride())
	}
}

impl<
	LT: PartialEq<RT>,
	LRows: Shape,
	LRStride: Stride,
	RT,
	RRows: Shape,
	RRStride: Stride,
	L: for<'a> Reborrow<'a, Target = col::Ref<'a, LT, LRows, LRStride>>,
	R: for<'a> Reborrow<'a, Target = col::Ref<'a, RT, RRows, RRStride>>,
> PartialEq<col::generic::Col<R>> for col::generic::Col<L>
{
	fn eq(&self, other: &col::generic::Col<R>) -> bool {
		fn imp<LT: PartialEq<RT>, RT>(l: ColRef<'_, LT>, r: ColRef<'_, RT>) -> bool {
			if l.nrows() != r.nrows() {
				return false;
			}

			with_dim!(N, l.nrows());

			let l = l.as_row_shape(N);
			let r = r.as_row_shape(N);

			for i in N.indices() {
				if l[i] != r[i] {
					return false;
				}
			}
			true
		}
		imp(self.rb().as_dyn_rows().as_dyn_stride(), other.rb().as_dyn_rows().as_dyn_stride())
	}
}

impl<
	LT: PartialEq<RT>,
	LCols: Shape,
	LCStride: Stride,
	RT,
	RCols: Shape,
	RCStride: Stride,
	L: for<'a> Reborrow<'a, Target = row::Ref<'a, LT, LCols, LCStride>>,
	R: for<'a> Reborrow<'a, Target = row::Ref<'a, RT, RCols, RCStride>>,
> PartialEq<row::generic::Row<R>> for row::generic::Row<L>
{
	fn eq(&self, other: &row::generic::Row<R>) -> bool {
		self.rb().transpose() == other.rb().transpose()
	}
}

impl<
	LT: PartialEq<RT>,
	LDim: Shape,
	LStride: Stride,
	RT,
	RDim: Shape,
	RStride: Stride,
	L: for<'a> Reborrow<'a, Target = diag::Ref<'a, LT, LDim, LStride>>,
	R: for<'a> Reborrow<'a, Target = diag::Ref<'a, RT, RDim, RStride>>,
> PartialEq<diag::generic::Diag<R>> for diag::generic::Diag<L>
{
	fn eq(&self, other: &diag::generic::Diag<R>) -> bool {
		self.rb().column_vector() == other.rb().column_vector()
	}
}

impl<
	LI: Index,
	LN: Shape,
	RI: Index,
	RN: Shape,
	L: for<'a> Reborrow<'a, Target = perm::Ref<'a, LI, LN>>,
	R: for<'a> Reborrow<'a, Target = perm::Ref<'a, RI, RN>>,
> PartialEq<perm::generic::Perm<R>> for perm::generic::Perm<L>
where
	LN::Idx<LI>: PartialEq<RN::Idx<RI>>,
{
	#[inline]
	fn eq(&self, other: &perm::generic::Perm<R>) -> bool {
		self.rb().arrays().0 == other.rb().arrays().0
	}
}

impl_op_assign!({
	impl<
		T: ComplexField,
		Rows: Shape,
		Cols: Shape,
		LRStride: Stride,
		LCStride: Stride,
		RT: (Conjugate<Canonical = T>),
		RRStride: Stride,
		RCStride: Stride,
		L: (for<'a> ReborrowMut<'a, Target = mat::Mut<'a, T, Rows, Cols, LRStride, LCStride>>),
		R: (for<'a> Reborrow<'a, Target = mat::Ref<'a, RT, Rows, Cols, RRStride, RCStride>>),
	> SubAssign<mat::generic::Mat<R>> for mat::generic::Mat<L>
	{
		fn sub_assign(&mut self, rhs: _) {
			fn imp<T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: MatMut<'_, T>, rhs: MatRef<'_, RT>) {
				zip!(lhs, rhs).for_each(sub_assign_fn::<T, RT>())
			}

			let lhs = self.rb_mut();
			imp(lhs.as_dyn_mut().as_dyn_stride_mut(), rhs.rb().as_dyn().as_dyn_stride());
		}
	}

	impl<
		T: ComplexField,
		Rows: Shape,
		LRStride: Stride,
		RT: (Conjugate<Canonical = T>),
		RRStride: Stride,
		L: (for<'a> ReborrowMut<'a, Target = col::Mut<'a, T, Rows, LRStride>>),
		R: (for<'a> Reborrow<'a, Target = col::Ref<'a, RT, Rows, RRStride>>),
	> SubAssign<col::generic::Col<R>> for col::generic::Col<L>
	{
		fn sub_assign(&mut self, rhs: _) {
			fn imp<T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: ColMut<'_, T>, rhs: ColRef<'_, RT>) {
				zip!(lhs, rhs).for_each(sub_assign_fn::<T, RT>())
			}

			let lhs = self.rb_mut();
			imp(lhs.as_dyn_rows_mut().as_dyn_stride_mut(), rhs.rb().as_dyn_rows().as_dyn_stride());
		}
	}

	impl<
		T: ComplexField,
		Cols: Shape,
		LCStride: Stride,
		RT: (Conjugate<Canonical = T>),
		RCStride: Stride,
		L: (for<'a> ReborrowMut<'a, Target = row::Mut<'a, T, Cols, LCStride>>),
		R: (for<'a> Reborrow<'a, Target = row::Ref<'a, RT, Cols, RCStride>>),
	> SubAssign<row::generic::Row<R>> for row::generic::Row<L>
	{
		fn sub_assign(&mut self, rhs: _) {
			fn imp<T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: RowMut<'_, T>, rhs: RowRef<'_, RT>) {
				SubAssign::sub_assign(&mut lhs.transpose_mut(), &rhs.transpose())
			}

			let lhs = self.rb_mut();
			imp(lhs.as_dyn_cols_mut().as_dyn_stride_mut(), rhs.rb().as_dyn_cols().as_dyn_stride());
		}
	}

	impl<
		T: ComplexField,
		Dim: Shape,
		LStride: Stride,
		RT: (Conjugate<Canonical = T>),
		RStride: Stride,
		L: (for<'a> ReborrowMut<'a, Target = diag::Mut<'a, T, Dim, LStride>>),
		R: (for<'a> Reborrow<'a, Target = diag::Ref<'a, RT, Dim, RStride>>),
	> SubAssign<diag::generic::Diag<R>> for diag::generic::Diag<L>
	{
		fn sub_assign(&mut self, rhs: _) {
			fn imp<T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: ColMut<'_, T>, rhs: ColRef<'_, RT>) {
				SubAssign::sub_assign(&mut { lhs }, &rhs)
			}

			let lhs = self.rb_mut();
			imp(
				lhs.column_vector_mut().as_dyn_rows_mut().as_dyn_stride_mut(),
				rhs.rb().column_vector().as_dyn_rows().as_dyn_stride(),
			);
		}
	}
});

impl_op_assign!({
	impl<
		T: ComplexField,
		Rows: Shape,
		Cols: Shape,
		LRStride: Stride,
		LCStride: Stride,
		RT: (Conjugate<Canonical = T>),
		RRStride: Stride,
		RCStride: Stride,
		L: (for<'a> ReborrowMut<'a, Target = mat::Mut<'a, T, Rows, Cols, LRStride, LCStride>>),
		R: (for<'a> Reborrow<'a, Target = mat::Ref<'a, RT, Rows, Cols, RRStride, RCStride>>),
	> AddAssign<mat::generic::Mat<R>> for mat::generic::Mat<L>
	{
		fn add_assign(&mut self, rhs: _) {
			fn imp<T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: MatMut<'_, T>, rhs: MatRef<'_, RT>) {
				zip!(lhs, rhs).for_each(add_assign_fn::<T, RT>())
			}

			let lhs = self.rb_mut();
			imp(lhs.as_dyn_mut().as_dyn_stride_mut(), rhs.rb().as_dyn().as_dyn_stride());
		}
	}

	impl<
		T: ComplexField,
		Rows: Shape,
		LRStride: Stride,
		RT: (Conjugate<Canonical = T>),
		RRStride: Stride,
		L: (for<'a> ReborrowMut<'a, Target = col::Mut<'a, T, Rows, LRStride>>),
		R: (for<'a> Reborrow<'a, Target = col::Ref<'a, RT, Rows, RRStride>>),
	> AddAssign<col::generic::Col<R>> for col::generic::Col<L>
	{
		fn add_assign(&mut self, rhs: _) {
			fn imp<T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: ColMut<'_, T>, rhs: ColRef<'_, RT>) {
				zip!(lhs, rhs).for_each(add_assign_fn::<T, RT>())
			}

			let lhs = self.rb_mut();
			imp(lhs.as_dyn_rows_mut().as_dyn_stride_mut(), rhs.rb().as_dyn_rows().as_dyn_stride());
		}
	}

	impl<
		T: ComplexField,
		Cols: Shape,
		LCStride: Stride,
		RT: (Conjugate<Canonical = T>),
		RCStride: Stride,
		L: (for<'a> ReborrowMut<'a, Target = row::Mut<'a, T, Cols, LCStride>>),
		R: (for<'a> Reborrow<'a, Target = row::Ref<'a, RT, Cols, RCStride>>),
	> AddAssign<row::generic::Row<R>> for row::generic::Row<L>
	{
		fn add_assign(&mut self, rhs: _) {
			fn imp<T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: RowMut<'_, T>, rhs: RowRef<'_, RT>) {
				AddAssign::add_assign(&mut lhs.transpose_mut(), &rhs.transpose())
			}

			let lhs = self.rb_mut();
			imp(lhs.as_dyn_cols_mut().as_dyn_stride_mut(), rhs.rb().as_dyn_cols().as_dyn_stride());
		}
	}

	impl<
		T: ComplexField,
		Dim: Shape,
		LStride: Stride,
		RT: (Conjugate<Canonical = T>),
		RStride: Stride,
		L: (for<'a> ReborrowMut<'a, Target = diag::Mut<'a, T, Dim, LStride>>),
		R: (for<'a> Reborrow<'a, Target = diag::Ref<'a, RT, Dim, RStride>>),
	> AddAssign<diag::generic::Diag<R>> for diag::generic::Diag<L>
	{
		fn add_assign(&mut self, rhs: _) {
			fn imp<T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: ColMut<'_, T>, rhs: ColRef<'_, RT>) {
				AddAssign::add_assign(&mut { lhs }, &rhs)
			}

			let lhs = self.rb_mut();
			imp(
				lhs.column_vector_mut().as_dyn_rows_mut().as_dyn_stride_mut(),
				rhs.rb().column_vector().as_dyn_rows().as_dyn_stride(),
			);
		}
	}
});

impl_binop!({
	impl<
		T: ComplexField,
		Rows: Shape,
		Cols: Shape,
		LT: (Conjugate<Canonical = T>),
		LRStride: Stride,
		LCStride: Stride,
		RT: (Conjugate<Canonical = T>),
		RRStride: Stride,
		RCStride: Stride,
		L: (for<'a> Reborrow<'a, Target = mat::Ref<'a, LT, Rows, Cols, LRStride, LCStride>>),
		R: (for<'a> Reborrow<'a, Target = mat::Ref<'a, RT, Rows, Cols, RRStride, RCStride>>),
	> Add<mat::generic::Mat<R>> for mat::generic::Mat<L>
	{
		type Output = Mat<T, Rows, Cols>;

		fn add(self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(lhs: MatRef<'_, LT>, rhs: MatRef<'_, RT>) -> Mat<T> {
				assert!(all(lhs.nrows() == rhs.nrows(), lhs.ncols() == rhs.ncols()));
				zip!(lhs, rhs).map(add_fn::<LT, RT>())
			}
			let lhs = self.rb();
			imp(lhs.as_dyn().as_dyn_stride(), rhs.rb().as_dyn().as_dyn_stride()).into_shape(lhs.nrows(), lhs.ncols())
		}
	}

	impl<
		T: ComplexField,
		Rows: Shape,
		LT: (Conjugate<Canonical = T>),
		LRStride: Stride,
		RT: (Conjugate<Canonical = T>),
		RRStride: Stride,
		L: (for<'a> Reborrow<'a, Target = col::Ref<'a, LT, Rows, LRStride>>),
		R: (for<'a> Reborrow<'a, Target = col::Ref<'a, RT, Rows, RRStride>>),
	> Add<col::generic::Col<R>> for col::generic::Col<L>
	{
		type Output = Col<T, Rows>;

		fn add(self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(lhs: ColRef<'_, LT>, rhs: ColRef<'_, RT>) -> Col<T> {
				assert!(all(lhs.nrows() == rhs.nrows(), lhs.ncols() == rhs.ncols()));
				zip!(lhs, rhs).map(add_fn::<LT, RT>())
			}
			let lhs = self.rb();
			imp(lhs.as_dyn_rows().as_dyn_stride(), rhs.rb().as_dyn_rows().as_dyn_stride()).into_row_shape(lhs.nrows())
		}
	}

	impl<
		T: ComplexField,
		Cols: Shape,
		LT: (Conjugate<Canonical = T>),
		LCStride: Stride,
		RT: (Conjugate<Canonical = T>),
		RCStride: Stride,
		L: (for<'a> Reborrow<'a, Target = row::Ref<'a, LT, Cols, LCStride>>),
		R: (for<'a> Reborrow<'a, Target = row::Ref<'a, RT, Cols, RCStride>>),
	> Add<row::generic::Row<R>> for row::generic::Row<L>
	{
		type Output = Row<T, Cols>;

		fn add(self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(lhs: RowRef<'_, LT>, rhs: RowRef<'_, RT>) -> Row<T> {
				assert!(all(lhs.nrows() == rhs.nrows(), lhs.ncols() == rhs.ncols()));
				(lhs.transpose() + rhs.transpose()).into_transpose()
			}
			let lhs = self.rb();
			imp(lhs.as_dyn_cols().as_dyn_stride(), rhs.rb().as_dyn_cols().as_dyn_stride()).into_col_shape(lhs.ncols())
		}
	}

	impl<
		T: ComplexField,
		Dim: Shape,
		LT: (Conjugate<Canonical = T>),
		LStride: Stride,
		RT: (Conjugate<Canonical = T>),
		RStride: Stride,
		L: (for<'a> Reborrow<'a, Target = diag::Ref<'a, LT, Dim, LStride>>),
		R: (for<'a> Reborrow<'a, Target = diag::Ref<'a, RT, Dim, RStride>>),
	> Add<diag::generic::Diag<R>> for diag::generic::Diag<L>
	{
		type Output = Diag<T, Dim>;

		fn add(self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(lhs: ColRef<'_, LT>, rhs: ColRef<'_, RT>) -> Col<T> {
				assert!(all(lhs.nrows() == rhs.nrows()));
				lhs + rhs
			}
			let lhs = self.rb();
			imp(
				lhs.column_vector().as_dyn_rows().as_dyn_stride(),
				rhs.rb().column_vector().as_dyn_rows().as_dyn_stride(),
			)
			.into_row_shape(lhs.dim())
			.into_diagonal()
		}
	}
});

impl_binop!({
	impl<
		T: ComplexField,
		Rows: Shape,
		Cols: Shape,
		LT: (Conjugate<Canonical = T>),
		LRStride: Stride,
		LCStride: Stride,
		RT: (Conjugate<Canonical = T>),
		RRStride: Stride,
		RCStride: Stride,
		L: (for<'a> Reborrow<'a, Target = mat::Ref<'a, LT, Rows, Cols, LRStride, LCStride>>),
		R: (for<'a> Reborrow<'a, Target = mat::Ref<'a, RT, Rows, Cols, RRStride, RCStride>>),
	> Sub<mat::generic::Mat<R>> for mat::generic::Mat<L>
	{
		type Output = Mat<T, Rows, Cols>;

		fn sub(self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(lhs: MatRef<'_, LT>, rhs: MatRef<'_, RT>) -> Mat<T> {
				assert!(all(lhs.nrows() == rhs.nrows(), lhs.ncols() == rhs.ncols()));
				zip!(lhs, rhs).map(sub_fn::<LT, RT>())
			}
			let lhs = self.rb();
			imp(lhs.as_dyn().as_dyn_stride(), rhs.rb().as_dyn().as_dyn_stride()).into_shape(lhs.nrows(), lhs.ncols())
		}
	}

	impl<
		T: ComplexField,
		Rows: Shape,
		LT: (Conjugate<Canonical = T>),
		LRStride: Stride,
		RT: (Conjugate<Canonical = T>),
		RRStride: Stride,
		L: (for<'a> Reborrow<'a, Target = col::Ref<'a, LT, Rows, LRStride>>),
		R: (for<'a> Reborrow<'a, Target = col::Ref<'a, RT, Rows, RRStride>>),
	> Sub<col::generic::Col<R>> for col::generic::Col<L>
	{
		type Output = Col<T, Rows>;

		fn sub(self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(lhs: ColRef<'_, LT>, rhs: ColRef<'_, RT>) -> Col<T> {
				assert!(all(lhs.nrows() == rhs.nrows(), lhs.ncols() == rhs.ncols()));
				zip!(lhs, rhs).map(sub_fn::<LT, RT>())
			}
			let lhs = self.rb();
			imp(lhs.as_dyn_rows().as_dyn_stride(), rhs.rb().as_dyn_rows().as_dyn_stride()).into_row_shape(lhs.nrows())
		}
	}

	impl<
		T: ComplexField,
		Cols: Shape,
		LT: (Conjugate<Canonical = T>),
		LCStride: Stride,
		RT: (Conjugate<Canonical = T>),
		RCStride: Stride,
		L: (for<'a> Reborrow<'a, Target = row::Ref<'a, LT, Cols, LCStride>>),
		R: (for<'a> Reborrow<'a, Target = row::Ref<'a, RT, Cols, RCStride>>),
	> Sub<row::generic::Row<R>> for row::generic::Row<L>
	{
		type Output = Row<T, Cols>;

		fn sub(self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(lhs: RowRef<'_, LT>, rhs: RowRef<'_, RT>) -> Row<T> {
				assert!(all(lhs.nrows() == rhs.nrows(), lhs.ncols() == rhs.ncols()));
				(lhs.transpose() + rhs.transpose()).into_transpose()
			}
			let lhs = self.rb();
			imp(lhs.as_dyn_cols().as_dyn_stride(), rhs.rb().as_dyn_cols().as_dyn_stride()).into_col_shape(lhs.ncols())
		}
	}

	impl<
		T: ComplexField,
		Dim: Shape,
		LT: (Conjugate<Canonical = T>),
		LStride: Stride,
		RT: (Conjugate<Canonical = T>),
		RStride: Stride,
		L: (for<'a> Reborrow<'a, Target = diag::Ref<'a, LT, Dim, LStride>>),
		R: (for<'a> Reborrow<'a, Target = diag::Ref<'a, RT, Dim, RStride>>),
	> Sub<diag::generic::Diag<R>> for diag::generic::Diag<L>
	{
		type Output = Diag<T, Dim>;

		fn sub(self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(lhs: ColRef<'_, LT>, rhs: ColRef<'_, RT>) -> Col<T> {
				assert!(all(lhs.nrows() == rhs.nrows()));
				lhs + rhs
			}
			let lhs = self.rb();
			imp(
				lhs.column_vector().as_dyn_rows().as_dyn_stride(),
				rhs.rb().column_vector().as_dyn_rows().as_dyn_stride(),
			)
			.into_row_shape(lhs.dim())
			.into_diagonal()
		}
	}
});

impl_op!({
	impl<
		T: Conjugate,
		Rows: Shape,
		Cols: Shape,
		RStride: Stride,
		CStride: Stride,
		Inner: (for<'a> Reborrow<'a, Target = mat::Ref<'a, T, Rows, Cols, RStride, CStride>>),
	> Neg for mat::generic::Mat<Inner>
	{
		type Output = Mat<T::Canonical, Rows, Cols>;

		fn neg(self) {
			#[track_caller]
			fn imp<T: Conjugate>(A: MatRef<'_, T>) -> Mat<T::Canonical> {
				zip!(A).map(neg_fn::<T>())
			}
			let A = self.rb();
			imp(A.as_dyn().as_dyn_stride()).into_shape(A.nrows(), A.ncols())
		}
	}

	impl<T: Conjugate, Rows: Shape, RStride: Stride, Inner: (for<'a> Reborrow<'a, Target = col::Ref<'a, T, Rows, RStride>>)> Neg
		for col::generic::Col<Inner>
	{
		type Output = Col<T::Canonical, Rows>;

		fn neg(self) {
			#[track_caller]
			fn imp<T: Conjugate>(A: ColRef<'_, T>) -> Col<T::Canonical> {
				zip!(A).map(neg_fn::<T>())
			}
			let A = self.rb();
			imp(A.as_dyn_rows().as_dyn_stride()).into_row_shape(A.nrows())
		}
	}

	impl<T: Conjugate, Cols: Shape, CStride: Stride, Inner: (for<'a> Reborrow<'a, Target = row::Ref<'a, T, Cols, CStride>>)> Neg
		for row::generic::Row<Inner>
	{
		type Output = Row<T::Canonical, Cols>;

		fn neg(self) {
			#[track_caller]
			fn imp<T: Conjugate>(A: RowRef<'_, T>) -> Row<T::Canonical> {
				(-A.transpose()).into_transpose()
			}
			let A = self.rb();
			imp(A.as_dyn_cols().as_dyn_stride()).into_col_shape(A.ncols())
		}
	}

	impl<T: Conjugate, Dim: Shape, Stride: (crate::Stride), Inner: (for<'a> Reborrow<'a, Target = diag::Ref<'a, T, Dim, Stride>>)> Neg
		for diag::generic::Diag<Inner>
	{
		type Output = Diag<T::Canonical, Dim>;

		fn neg(self) {
			#[track_caller]
			fn imp<T: Conjugate>(A: ColRef<'_, T>) -> Col<T::Canonical> {
				-A
			}
			let A = self.rb().column_vector();
			imp(A.as_dyn_rows().as_dyn_stride()).into_row_shape(A.nrows()).into_diagonal()
		}
	}
});

#[inline]
#[math]
fn add_fn<LhsT: Conjugate<Canonical: ComplexField>, RhsT: Conjugate<Canonical = LhsT::Canonical>>()
-> impl FnMut(linalg::zip::Zip<&LhsT, linalg::zip::Last<&RhsT>>) -> LhsT::Canonical {
	#[inline(always)]
	move |unzip!(a, b)| Conj::apply(a) + Conj::apply(b)
}

#[inline]
#[math]
fn sub_fn<LhsT: Conjugate<Canonical: ComplexField>, RhsT: Conjugate<Canonical = LhsT::Canonical>>()
-> impl FnMut(linalg::zip::Zip<&LhsT, linalg::zip::Last<&RhsT>>) -> LhsT::Canonical {
	#[inline(always)]
	move |unzip!(a, b)| Conj::apply(a) - Conj::apply(b)
}

#[inline]
#[math]
fn mul_fn<LhsT: Conjugate<Canonical: ComplexField>, RhsT: Conjugate<Canonical = LhsT::Canonical>>()
-> impl FnMut(linalg::zip::Zip<&LhsT, linalg::zip::Last<&RhsT>>) -> LhsT::Canonical {
	#[inline(always)]
	move |unzip!(a, b)| Conj::apply(a) * Conj::apply(b)
}

#[inline]
#[math]
fn neg_fn<LhsT: Conjugate<Canonical: ComplexField>>() -> impl FnMut(linalg::zip::Last<&LhsT>) -> LhsT::Canonical {
	#[inline(always)]
	move |unzip!(a)| -Conj::apply(a)
}

#[inline]
#[math]
fn add_assign_fn<LhsT: ComplexField, RhsT: Conjugate<Canonical = LhsT>>() -> impl FnMut(linalg::zip::Zip<&mut LhsT, linalg::zip::Last<&RhsT>>) {
	#[inline(always)]
	move |unzip!(a, b)| *a = Conj::apply(a) + Conj::apply(b)
}

#[inline]
#[math]
fn sub_assign_fn<LhsT: ComplexField, RhsT: Conjugate<Canonical = LhsT>>() -> impl FnMut(linalg::zip::Zip<&mut LhsT, linalg::zip::Last<&RhsT>>) {
	#[inline(always)]
	move |unzip!(a, b)| *a = Conj::apply(a) - Conj::apply(b)
}

mod matmul {
	use super::*;
	use crate::assert;

	impl_binop!({
		impl<
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			Depth: Shape,
			LT: (Conjugate<Canonical = T>),
			LRStride: Stride,
			LCStride: Stride,
			RT: (Conjugate<Canonical = T>),
			RRStride: Stride,
			RCStride: Stride,
			L: (for<'a> Reborrow<'a, Target = mat::Ref<'a, LT, Rows, Depth, LRStride, LCStride>>),
			R: (for<'a> Reborrow<'a, Target = mat::Ref<'a, RT, Depth, Cols, RRStride, RCStride>>),
		> Mul<mat::generic::Mat<R>> for mat::generic::Mat<L>
		{
			type Output = Mat<T, Rows, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: MatRef<'_, LT>,
					rhs: MatRef<'_, RT>,
				) -> Mat<T> {
					assert!(lhs.ncols() == rhs.nrows());
					let mut out = Mat::zeros(lhs.nrows(), rhs.ncols());
					crate::linalg::matmul::matmul(out.as_mut(), Accum::Replace, lhs, rhs, T::one_impl(), get_global_parallelism());
					out
				}
				let lhs = self.rb();
				let rhs = rhs.rb();
				imp(lhs.as_dyn().as_dyn_stride(), rhs.as_dyn().as_dyn_stride()).into_shape(lhs.nrows(), rhs.ncols())
			}
		}

		impl<
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			LT: (Conjugate<Canonical = T>),
			LRStride: Stride,
			LCStride: Stride,
			RT: (Conjugate<Canonical = T>),
			RRStride: Stride,
			L: (for<'a> Reborrow<'a, Target = mat::Ref<'a, LT, Rows, Cols, LRStride, LCStride>>),
			R: (for<'a> Reborrow<'a, Target = col::Ref<'a, RT, Cols, RRStride>>),
		> Mul<col::generic::Col<R>> for mat::generic::Mat<L>
		{
			type Output = Col<T, Rows>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: MatRef<'_, LT>,
					rhs: ColRef<'_, RT>,
				) -> Col<T> {
					assert!(lhs.ncols() == rhs.nrows());
					let mut out = Col::zeros(lhs.nrows());
					crate::linalg::matmul::matmul(out.as_mut(), Accum::Replace, lhs, rhs, T::one_impl(), get_global_parallelism());
					out
				}
				let lhs = self.rb();
				let rhs = rhs.rb();
				imp(lhs.as_dyn().as_dyn_stride(), rhs.as_dyn_rows().as_dyn_stride()).into_row_shape(lhs.nrows())
			}
		}

		impl<
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			LT: (Conjugate<Canonical = T>),
			LCStride: Stride,
			RT: (Conjugate<Canonical = T>),
			RRStride: Stride,
			RCStride: Stride,
			L: (for<'a> Reborrow<'a, Target = row::Ref<'a, LT, Rows, LCStride>>),
			R: (for<'a> Reborrow<'a, Target = mat::Ref<'a, RT, Rows, Cols, RRStride, RCStride>>),
		> Mul<mat::generic::Mat<R>> for row::generic::Row<L>
		{
			type Output = Row<T, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: RowRef<'_, LT>,
					rhs: MatRef<'_, RT>,
				) -> Row<T> {
					assert!(lhs.ncols() == rhs.nrows());
					let mut out = Row::zeros(rhs.ncols());
					crate::linalg::matmul::matmul(out.as_mut(), Accum::Replace, lhs, rhs, T::one_impl(), get_global_parallelism());
					out
				}
				let lhs = self.rb();
				let rhs = rhs.rb();
				imp(lhs.as_dyn_cols().as_dyn_stride(), rhs.as_dyn().as_dyn_stride()).into_col_shape(rhs.ncols())
			}
		}

		impl<
			T: ComplexField,
			Dim: Shape,
			LT: (Conjugate<Canonical = T>),
			LCStride: Stride,
			RT: (Conjugate<Canonical = T>),
			RRStride: Stride,
			L: (for<'a> Reborrow<'a, Target = row::Ref<'a, LT, Dim, LCStride>>),
			R: (for<'a> Reborrow<'a, Target = col::Ref<'a, RT, Dim, RRStride>>),
		> Mul<col::generic::Col<R>> for row::generic::Row<L>
		{
			type Output = T;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(lhs: RowRef<'_, LT>, rhs: ColRef<'_, RT>) -> T {
					assert!(lhs.ncols() == rhs.nrows());
					let mut out = [[zero::<T>()]];
					crate::linalg::matmul::matmul(
						MatMut::from_row_major_array_mut(&mut out),
						Accum::Replace,
						lhs.as_mat(),
						rhs.as_mat(),
						T::one_impl(),
						get_global_parallelism(),
					);
					let [[out]] = out;
					out
				}
				let lhs = self.rb();
				let rhs = rhs.rb();
				imp(lhs.as_dyn_cols().as_dyn_stride(), rhs.as_dyn_rows().as_dyn_stride())
			}
		}

		impl<
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			LT: (Conjugate<Canonical = T>),
			LRStride: Stride,
			RT: (Conjugate<Canonical = T>),
			RCStride: Stride,
			L: (for<'a> Reborrow<'a, Target = col::Ref<'a, LT, Rows, LRStride>>),
			R: (for<'a> Reborrow<'a, Target = row::Ref<'a, RT, Cols, RCStride>>),
		> Mul<row::generic::Row<R>> for col::generic::Col<L>
		{
			type Output = Mat<T, Rows, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: ColRef<'_, LT>,
					rhs: RowRef<'_, RT>,
				) -> Mat<T> {
					assert!(lhs.ncols() == rhs.nrows());
					let mut out = Mat::zeros(lhs.nrows(), rhs.ncols());
					crate::linalg::matmul::matmul(out.rb_mut(), Accum::Replace, lhs, rhs, T::one_impl(), get_global_parallelism());
					out
				}
				let lhs = self.rb();
				let rhs = rhs.rb();
				imp(lhs.as_dyn_rows().as_dyn_stride(), rhs.as_dyn_cols().as_dyn_stride()).into_shape(lhs.nrows(), rhs.ncols())
			}
		}

		impl<
			T: ComplexField,
			Dim: Shape,
			Cols: Shape,
			LT: (Conjugate<Canonical = T>),
			LStride: Stride,
			RT: (Conjugate<Canonical = T>),
			RRStride: Stride,
			RCStride: Stride,
			L: (for<'a> Reborrow<'a, Target = diag::Ref<'a, LT, Dim, LStride>>),
			R: (for<'a> Reborrow<'a, Target = mat::Ref<'a, RT, Dim, Cols, RRStride, RCStride>>),
		> Mul<mat::generic::Mat<R>> for diag::generic::Diag<L>
		{
			type Output = Mat<T, Dim, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				#[math]
				fn imp<T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: ColRef<'_, LT>,
					rhs: MatRef<'_, RT>,
				) -> Mat<T> {
					let lhs_dim = lhs.nrows();
					let rhs_nrows = rhs.nrows();
					assert!(lhs_dim == rhs_nrows);

					Mat::from_fn(rhs.nrows(), rhs.ncols(), |i, j| Conj::apply(lhs.at(i)) * Conj::apply(rhs.at(i, j)))
				}

				let lhs = self.rb();
				let rhs = rhs.rb();
				imp(lhs.column_vector().as_dyn_rows().as_dyn_stride(), rhs.as_dyn().as_dyn_stride()).into_shape(rhs.nrows(), rhs.ncols())
			}
		}

		impl<
			T: ComplexField,
			Dim: Shape,
			LT: (Conjugate<Canonical = T>),
			LStride: Stride,
			RT: (Conjugate<Canonical = T>),
			RRStride: Stride,
			L: (for<'a> Reborrow<'a, Target = diag::Ref<'a, LT, Dim, LStride>>),
			R: (for<'a> Reborrow<'a, Target = col::Ref<'a, RT, Dim, RRStride>>),
		> Mul<col::generic::Col<R>> for diag::generic::Diag<L>
		{
			type Output = Col<T, Dim>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: ColRef<'_, LT>,
					rhs: ColRef<'_, RT>,
				) -> Col<T> {
					let lhs_dim = lhs.nrows();
					let rhs_nrows = rhs.nrows();
					assert!(lhs_dim == rhs_nrows);

					zip!(lhs, rhs).map(mul_fn::<LT, RT>())
				}

				let lhs = self.rb();
				let rhs = rhs.rb();
				imp(lhs.column_vector().as_dyn_rows().as_dyn_stride(), rhs.as_dyn_rows().as_dyn_stride()).into_row_shape(rhs.nrows())
			}
		}

		impl<
			T: ComplexField,
			Dim: Shape,
			Rows: Shape,
			LT: (Conjugate<Canonical = T>),
			LRStride: Stride,
			LCStride: Stride,
			RT: (Conjugate<Canonical = T>),
			RStride: Stride,
			L: (for<'a> Reborrow<'a, Target = mat::Ref<'a, LT, Rows, Dim, LRStride, LCStride>>),
			R: (for<'a> Reborrow<'a, Target = diag::Ref<'a, RT, Dim, RStride>>),
		> Mul<diag::generic::Diag<R>> for mat::generic::Mat<L>
		{
			type Output = Mat<T, Rows, Dim>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: MatRef<'_, LT>,
					rhs: ColRef<'_, RT>,
				) -> Mat<T> {
					let lhs_ncols = lhs.ncols();
					let rhs_dim = rhs.nrows();
					assert!(lhs_ncols == rhs_dim);

					let mut f = mul_fn::<LT, RT>();
					Mat::from_fn(lhs.nrows(), lhs.ncols(), |i, j| {
						f(linalg::zip::Zip(lhs.at(i, j), linalg::zip::Last(rhs.at(j))))
					})
				}

				let lhs = self.rb();
				let rhs = rhs.rb();
				imp(lhs.as_dyn().as_dyn_stride(), rhs.column_vector().as_dyn_rows().as_dyn_stride()).into_shape(lhs.nrows(), lhs.ncols())
			}
		}

		impl<
			T: ComplexField,
			Dim: Shape,
			LT: (Conjugate<Canonical = T>),
			LCStride: Stride,
			RT: (Conjugate<Canonical = T>),
			RStride: Stride,
			L: (for<'a> Reborrow<'a, Target = row::Ref<'a, LT, Dim, LCStride>>),
			R: (for<'a> Reborrow<'a, Target = diag::Ref<'a, RT, Dim, RStride>>),
		> Mul<diag::generic::Diag<R>> for row::generic::Row<L>
		{
			type Output = Row<T, Dim>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: RowRef<'_, LT>,
					rhs: ColRef<'_, RT>,
				) -> Row<T> {
					let lhs_ncols = lhs.ncols();
					let rhs_dim = rhs.nrows();
					assert!(lhs_ncols == rhs_dim);

					(rhs.as_diagonal() * lhs.transpose()).into_transpose()
				}

				let lhs = self.rb();
				let rhs = rhs.rb();
				imp(lhs.as_dyn_cols().as_dyn_stride(), rhs.column_vector().as_dyn_rows().as_dyn_stride()).into_col_shape(lhs.ncols())
			}
		}

		impl<
			T: ComplexField,
			Dim: Shape,
			LT: (Conjugate<Canonical = T>),
			LStride: Stride,
			RT: (Conjugate<Canonical = T>),
			RStride: Stride,
			L: (for<'a> Reborrow<'a, Target = diag::Ref<'a, LT, Dim, LStride>>),
			R: (for<'a> Reborrow<'a, Target = diag::Ref<'a, RT, Dim, RStride>>),
		> Mul<diag::generic::Diag<R>> for diag::generic::Diag<L>
		{
			type Output = Diag<T, Dim>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: ColRef<'_, LT>,
					rhs: ColRef<'_, RT>,
				) -> Col<T> {
					let lhs_dim = lhs.nrows();
					let rhs_dim = rhs.nrows();
					assert!(lhs_dim == rhs_dim);

					lhs.as_diagonal() * rhs
				}

				let lhs = self.rb().column_vector();
				let rhs = rhs.rb().column_vector();
				imp(lhs.as_dyn_rows().as_dyn_stride(), rhs.as_dyn_rows().as_dyn_stride())
					.into_row_shape(lhs.nrows())
					.into_diagonal()
			}
		}
	});
}

impl<I: Index> Mul<PermRef<'_, I>> for PermRef<'_, I> {
	type Output = Perm<I>;

	#[track_caller]
	fn mul(self, rhs: PermRef<'_, I>) -> Self::Output {
		let lhs = self;
		assert!(lhs.len() == rhs.len());
		let truncate = <I::Signed as SignedIndex>::truncate;
		let mut fwd = alloc::vec![I::from_signed(truncate(0)); lhs.len()].into_boxed_slice();
		let mut inv = alloc::vec![I::from_signed(truncate(0)); lhs.len()].into_boxed_slice();

		for (fwd, rhs) in fwd.iter_mut().zip(rhs.arrays().0) {
			*fwd = lhs.arrays().0[rhs.to_signed().zx()];
		}
		for (i, fwd) in fwd.iter().enumerate() {
			inv[fwd.to_signed().zx()] = I::from_signed(I::Signed::truncate(i));
		}

		Perm::new_checked(fwd, inv, lhs.len())
	}
}

// impl_perm_perm!(PermRef<'_, I>, PermRef<'_, I>, Perm<I>);
impl_perm_perm!(PermRef<'_, I>, Perm<I>, Perm<I>);
impl_perm_perm!(PermRef<'_, I>, &PermRef<'_, I>, Perm<I>);
impl_perm_perm!(PermRef<'_, I>, &Perm<I>, Perm<I>);
impl_perm_perm!(&PermRef<'_, I>, PermRef<'_, I>, Perm<I>);
impl_perm_perm!(&PermRef<'_, I>, Perm<I>, Perm<I>);
impl_perm_perm!(&PermRef<'_, I>, &PermRef<'_, I>, Perm<I>);
impl_perm_perm!(&PermRef<'_, I>, &Perm<I>, Perm<I>);

impl_perm_perm!(Perm<I>, PermRef<'_, I>, Perm<I>);
impl_perm_perm!(Perm<I>, Perm<I>, Perm<I>);
impl_perm_perm!(Perm<I>, &PermRef<'_, I>, Perm<I>);
impl_perm_perm!(Perm<I>, &Perm<I>, Perm<I>);
impl_perm_perm!(&Perm<I>, PermRef<'_, I>, Perm<I>);
impl_perm_perm!(&Perm<I>, Perm<I>, Perm<I>);
impl_perm_perm!(&Perm<I>, &PermRef<'_, I>, Perm<I>);
impl_perm_perm!(&Perm<I>, &Perm<I>, Perm<I>);

impl<I: Index, T: ComplexField, TT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> Mul<MatRef<'_, TT, Rows, Cols>> for PermRef<'_, I, Rows> {
	type Output = Mat<T, Rows, Cols>;

	#[track_caller]
	#[math]
	fn mul(self, rhs: MatRef<'_, TT, Rows, Cols>) -> Self::Output {
		let lhs = self;

		assert!(lhs.len() == rhs.nrows());
		let mut out = Mat::zeros(rhs.nrows(), rhs.ncols());

		fn imp<'ROWS, 'COLS, I: Index, T: ComplexField, TT: Conjugate<Canonical = T>>(
			mut out: MatMut<'_, T, Dim<'ROWS>, Dim<'COLS>>,
			lhs: PermRef<'_, I, Dim<'ROWS>>,
			rhs: MatRef<'_, TT, Dim<'ROWS>, Dim<'COLS>>,
		) {
			let fwd = lhs.bound_arrays().0;

			for j in rhs.ncols().indices() {
				for i in rhs.nrows().indices() {
					let fwd = fwd[i];
					let rhs = rhs.at(fwd.zx(), j);

					*out.as_mut().at_mut(i, j) = Conj::apply(rhs);
				}
			}
		}

		with_dim!(M, out.nrows().unbound());
		with_dim!(N, out.ncols().unbound());
		imp(out.as_mut().as_shape_mut(M, N), lhs.as_shape(M), rhs.as_shape(M, N));

		out
	}
}

// impl_perm!(PermRef<'_, I>, MatRef<'_,  TT>, Mat< T>);
impl_perm!(PermRef<'_, I, Rows>, MatMut<'_,  TT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_perm!(PermRef<'_, I, Rows>, Mat< TT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_perm!(PermRef<'_, I, Rows>, &MatRef<'_,  TT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_perm!(PermRef<'_, I, Rows>, &MatMut<'_,  TT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_perm!(PermRef<'_, I, Rows>, &Mat< TT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&PermRef<'_, I, Rows>, MatRef<'_,  TT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&PermRef<'_, I, Rows>, MatMut<'_,  TT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&PermRef<'_, I, Rows>, Mat< TT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&PermRef<'_, I, Rows>, &MatRef<'_,  TT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&PermRef<'_, I, Rows>, &MatMut<'_,  TT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&PermRef<'_, I, Rows>, &Mat< TT, Rows, Cols>, Mat< T, Rows, Cols>);

impl_perm!(Perm<I, Rows>, MatRef<'_,  TT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_perm!(Perm<I, Rows>, MatMut<'_,  TT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_perm!(Perm<I, Rows>, Mat< TT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_perm!(Perm<I, Rows>, &MatRef<'_,  TT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_perm!(Perm<I, Rows>, &MatMut<'_,  TT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_perm!(Perm<I, Rows>, &Mat< TT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&Perm<I, Rows>, MatRef<'_,  TT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&Perm<I, Rows>, MatMut<'_,  TT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&Perm<I, Rows>, Mat< TT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&Perm<I, Rows>, &MatRef<'_,  TT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&Perm<I, Rows>, &MatMut<'_,  TT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&Perm<I, Rows>, &Mat< TT, Rows, Cols>, Mat< T, Rows, Cols>);

impl<I: Index, T: ComplexField, TT: Conjugate<Canonical = T>, Len: Shape> Mul<ColRef<'_, TT, Len>> for PermRef<'_, I, Len> {
	type Output = Col<T, Len>;

	#[track_caller]
	#[math]
	fn mul(self, rhs: ColRef<'_, TT, Len>) -> Self::Output {
		let lhs = self;

		assert!(lhs.len() == rhs.nrows());
		let mut out = Col::zeros(rhs.nrows());

		fn imp<'ROWS, I: Index, T: ComplexField, TT: Conjugate<Canonical = T>>(
			mut out: ColMut<'_, T, Dim<'ROWS>>,
			lhs: PermRef<'_, I, Dim<'ROWS>>,
			rhs: ColRef<'_, TT, Dim<'ROWS>>,
		) {
			let fwd = lhs.bound_arrays().0;

			for i in rhs.nrows().indices() {
				let fwd = fwd[i];
				let rhs = rhs.at(fwd.zx());

				*out.as_mut().at_mut(i) = Conj::apply(rhs);
			}
		}

		with_dim!(M, out.nrows().unbound());
		imp(out.as_mut().as_row_shape_mut(M), lhs.as_shape(M), rhs.as_row_shape(M));

		out
	}
}

// impl_perm!(PermRef<'_, I>, ColRef<'_,  TT>, Col< T>);
impl_1d_perm!(PermRef<'_, I, Len>, ColMut<'_,  TT, Len>, Col< T, Len>);
impl_1d_perm!(PermRef<'_, I, Len>, Col< TT, Len>, Col< T, Len>);
impl_1d_perm!(PermRef<'_, I, Len>, &ColRef<'_,  TT, Len>, Col< T, Len>);
impl_1d_perm!(PermRef<'_, I, Len>, &ColMut<'_,  TT, Len>, Col< T, Len>);
impl_1d_perm!(PermRef<'_, I, Len>, &Col< TT, Len>, Col< T, Len>);
impl_1d_perm!(&PermRef<'_, I, Len>, ColRef<'_,  TT, Len>, Col< T, Len>);
impl_1d_perm!(&PermRef<'_, I, Len>, ColMut<'_,  TT, Len>, Col< T, Len>);
impl_1d_perm!(&PermRef<'_, I, Len>, Col< TT, Len>, Col< T, Len>);
impl_1d_perm!(&PermRef<'_, I, Len>, &ColRef<'_,  TT, Len>, Col< T, Len>);
impl_1d_perm!(&PermRef<'_, I, Len>, &ColMut<'_,  TT, Len>, Col< T, Len>);
impl_1d_perm!(&PermRef<'_, I, Len>, &Col< TT, Len>, Col< T, Len>);

impl_1d_perm!(Perm<I, Len>, ColRef<'_,  TT, Len>, Col< T, Len>);
impl_1d_perm!(Perm<I, Len>, ColMut<'_,  TT, Len>, Col< T, Len>);
impl_1d_perm!(Perm<I, Len>, Col< TT, Len>, Col< T, Len>);
impl_1d_perm!(Perm<I, Len>, &ColRef<'_,  TT, Len>, Col< T, Len>);
impl_1d_perm!(Perm<I, Len>, &ColMut<'_,  TT, Len>, Col< T, Len>);
impl_1d_perm!(Perm<I, Len>, &Col< TT, Len>, Col< T, Len>);
impl_1d_perm!(&Perm<I, Len>, ColRef<'_,  TT, Len>, Col< T, Len>);
impl_1d_perm!(&Perm<I, Len>, ColMut<'_,  TT, Len>, Col< T, Len>);
impl_1d_perm!(&Perm<I, Len>, Col< TT, Len>, Col< T, Len>);
impl_1d_perm!(&Perm<I, Len>, &ColRef<'_,  TT, Len>, Col< T, Len>);
impl_1d_perm!(&Perm<I, Len>, &ColMut<'_,  TT, Len>, Col< T, Len>);
impl_1d_perm!(&Perm<I, Len>, &Col< TT, Len>, Col< T, Len>);

impl<I: Index, T: ComplexField, TT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> Mul<PermRef<'_, I, Cols>> for MatRef<'_, TT, Rows, Cols> {
	type Output = Mat<T, Rows, Cols>;

	#[track_caller]
	#[math]
	fn mul(self, rhs: PermRef<'_, I, Cols>) -> Self::Output {
		let lhs = self;

		assert!(lhs.ncols() == rhs.len());
		let mut out = Mat::zeros(lhs.nrows(), lhs.ncols());

		fn imp<'ROWS, 'COLS, I: Index, T: ComplexField, TT: Conjugate<Canonical = T>>(
			mut out: MatMut<'_, T, Dim<'ROWS>, Dim<'COLS>>,
			lhs: MatRef<'_, TT, Dim<'ROWS>, Dim<'COLS>>,
			rhs: PermRef<'_, I, Dim<'COLS>>,
		) {
			let inv = rhs.bound_arrays().1;

			for j in lhs.ncols().indices() {
				let inv = inv[j];
				for i in lhs.nrows().indices() {
					let lhs = lhs.at(i, inv.zx());

					*out.as_mut().at_mut(i, j) = Conj::apply(lhs);
				}
			}
		}

		with_dim!(M, out.nrows().unbound());
		with_dim!(N, out.ncols().unbound());
		imp(out.as_shape_mut(M, N), lhs.as_shape(M, N), rhs.as_shape(N));

		out
	}
}

// impl_perm!(MatRef<'_,  TT>, PermRef<'_, I>, Mat< T>);
impl_perm!(MatRef<'_,  TT, Rows, Cols>, Perm<I, Cols>, Mat< T, Rows, Cols>);
impl_perm!(MatRef<'_,  TT, Rows, Cols>, &PermRef<'_, I, Cols>, Mat< T, Rows, Cols>);
impl_perm!(MatRef<'_,  TT, Rows, Cols>, &Perm<I, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&MatRef<'_,  TT, Rows, Cols>, PermRef<'_, I, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&MatRef<'_,  TT, Rows, Cols>, Perm<I, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&MatRef<'_,  TT, Rows, Cols>, &PermRef<'_, I, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&MatRef<'_,  TT, Rows, Cols>, &Perm<I, Cols>, Mat< T, Rows, Cols>);

impl_perm!(MatMut<'_,  TT, Rows, Cols>, PermRef<'_, I, Cols>, Mat< T, Rows, Cols>);
impl_perm!(MatMut<'_,  TT, Rows, Cols>, Perm<I, Cols>, Mat< T, Rows, Cols>);
impl_perm!(MatMut<'_,  TT, Rows, Cols>, &PermRef<'_, I, Cols>, Mat< T, Rows, Cols>);
impl_perm!(MatMut<'_,  TT, Rows, Cols>, &Perm<I, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&MatMut<'_,  TT, Rows, Cols>, PermRef<'_, I, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&MatMut<'_,  TT, Rows, Cols>, Perm<I, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&MatMut<'_,  TT, Rows, Cols>, &PermRef<'_, I, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&MatMut<'_,  TT, Rows, Cols>, &Perm<I, Cols>, Mat< T, Rows, Cols>);

impl_perm!(Mat< TT, Rows, Cols>, PermRef<'_, I, Cols>, Mat< T, Rows, Cols>);
impl_perm!(Mat< TT, Rows, Cols>, Perm<I, Cols>, Mat< T, Rows, Cols>);
impl_perm!(Mat< TT, Rows, Cols>, &PermRef<'_, I, Cols>, Mat< T, Rows, Cols>);
impl_perm!(Mat< TT, Rows, Cols>, &Perm<I, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&Mat< TT, Rows, Cols>, PermRef<'_, I, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&Mat< TT, Rows, Cols>, Perm<I, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&Mat< TT, Rows, Cols>, &PermRef<'_, I, Cols>, Mat< T, Rows, Cols>);
impl_perm!(&Mat< TT, Rows, Cols>, &Perm<I, Cols>, Mat< T, Rows, Cols>);

impl<I: Index, T: ComplexField, TT: Conjugate<Canonical = T>, Len: Shape> Mul<PermRef<'_, I, Len>> for RowRef<'_, TT, Len> {
	type Output = Row<T, Len>;

	#[track_caller]
	#[math]
	fn mul(self, rhs: PermRef<'_, I, Len>) -> Self::Output {
		let lhs = self;

		assert!(lhs.ncols() == rhs.len());
		let mut out = Row::zeros(lhs.ncols());

		fn imp<'COLS, I: Index, T: ComplexField, TT: Conjugate<Canonical = T>>(
			mut out: RowMut<'_, T, Dim<'COLS>>,
			lhs: RowRef<'_, TT, Dim<'COLS>>,
			rhs: PermRef<'_, I, Dim<'COLS>>,
		) {
			let inv = rhs.bound_arrays().1;

			for j in lhs.ncols().indices() {
				let inv = inv[j];
				let lhs = lhs.at(inv.zx());

				*out.as_mut().at_mut(j) = Conj::apply(lhs)
			}
		}

		with_dim!(N, out.ncols().unbound());
		imp(out.as_col_shape_mut(N), lhs.as_col_shape(N), rhs.as_shape(N));
		out
	}
}

// impl_perm!(RowRef<'_,  TT>, PermRef<'_, I>, Row< T>);
impl_1d_perm!(RowRef<'_,  TT, Len>, Perm<I, Len>, Row< T, Len>);
impl_1d_perm!(RowRef<'_,  TT, Len>, &PermRef<'_, I, Len>, Row< T, Len>);
impl_1d_perm!(RowRef<'_,  TT, Len>, &Perm<I, Len>, Row< T, Len>);
impl_1d_perm!(&RowRef<'_,  TT, Len>, PermRef<'_, I, Len>, Row< T, Len>);
impl_1d_perm!(&RowRef<'_,  TT, Len>, Perm<I, Len>, Row< T, Len>);
impl_1d_perm!(&RowRef<'_,  TT, Len>, &PermRef<'_, I, Len>, Row< T, Len>);
impl_1d_perm!(&RowRef<'_,  TT, Len>, &Perm<I, Len>, Row< T, Len>);

impl_1d_perm!(RowMut<'_,  TT, Len>, PermRef<'_, I, Len>, Row< T, Len>);
impl_1d_perm!(RowMut<'_,  TT, Len>, Perm<I, Len>, Row< T, Len>);
impl_1d_perm!(RowMut<'_,  TT, Len>, &PermRef<'_, I, Len>, Row< T, Len>);
impl_1d_perm!(RowMut<'_,  TT, Len>, &Perm<I, Len>, Row< T, Len>);
impl_1d_perm!(&RowMut<'_,  TT, Len>, PermRef<'_, I, Len>, Row< T, Len>);
impl_1d_perm!(&RowMut<'_,  TT, Len>, Perm<I, Len>, Row< T, Len>);
impl_1d_perm!(&RowMut<'_,  TT, Len>, &PermRef<'_, I, Len>, Row< T, Len>);
impl_1d_perm!(&RowMut<'_,  TT, Len>, &Perm<I, Len>, Row< T, Len>);

impl_1d_perm!(Row< TT, Len>, PermRef<'_, I, Len>, Row< T, Len>);
impl_1d_perm!(Row< TT, Len>, Perm<I, Len>, Row< T, Len>);
impl_1d_perm!(Row< TT, Len>, &PermRef<'_, I, Len>, Row< T, Len>);
impl_1d_perm!(Row< TT, Len>, &Perm<I, Len>, Row< T, Len>);
impl_1d_perm!(&Row< TT, Len>, PermRef<'_, I, Len>, Row< T, Len>);
impl_1d_perm!(&Row< TT, Len>, Perm<I, Len>, Row< T, Len>);
impl_1d_perm!(&Row< TT, Len>, &PermRef<'_, I, Len>, Row< T, Len>);
impl_1d_perm!(&Row< TT, Len>, &Perm<I, Len>, Row< T, Len>);

impl<T: ComplexField, LhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> Mul<Scale<T>> for MatRef<'_, LhsT, Rows, Cols> {
	type Output = Mat<T, Rows, Cols>;

	#[math]
	fn mul(self, rhs: Scale<T>) -> Self::Output {
		let rhs = &rhs.0;
		let lhs = self;
		zip!(lhs).map(|unzip!(x)| Conj::apply(x) * rhs)
	}
}

impl<T: ComplexField, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> Mul<MatRef<'_, RhsT, Rows, Cols>> for Scale<T> {
	type Output = Mat<T, Rows, Cols>;

	#[math]
	fn mul(self, rhs: MatRef<'_, RhsT, Rows, Cols>) -> Self::Output {
		let lhs = &self.0;
		zip!(rhs).map(|unzip!(x)| *lhs * Conj::apply(x))
	}
}

impl<T: ComplexField, LhsT: Conjugate<Canonical = T>, Len: Shape> Mul<Scale<T>> for ColRef<'_, LhsT, Len> {
	type Output = Col<T, Len>;

	#[math]
	fn mul(self, rhs: Scale<T>) -> Self::Output {
		let rhs = &rhs.0;
		let lhs = self;
		zip!(lhs).map(|unzip!(x)| Conj::apply(x) * *rhs)
	}
}
impl<T: ComplexField, RhsT: Conjugate<Canonical = T>, Len: Shape> Mul<ColRef<'_, RhsT, Len>> for Scale<T> {
	type Output = Col<T, Len>;

	#[math]
	fn mul(self, rhs: ColRef<'_, RhsT, Len>) -> Self::Output {
		let lhs = &self.0;
		zip!(rhs).map(|unzip!(x)| *lhs * Conj::apply(x))
	}
}

impl<T: ComplexField, LhsT: Conjugate<Canonical = T>, Len: Shape> Mul<Scale<T>> for RowRef<'_, LhsT, Len> {
	type Output = Row<T, Len>;

	#[math]
	fn mul(self, rhs: Scale<T>) -> Self::Output {
		let rhs = &rhs.0;
		let lhs = self;
		zip!(lhs).map(|unzip!(x)| Conj::apply(x) * *rhs)
	}
}
impl<T: ComplexField, RhsT: Conjugate<Canonical = T>, Len: Shape> Mul<RowRef<'_, RhsT, Len>> for Scale<T> {
	type Output = Row<T, Len>;

	#[math]
	fn mul(self, rhs: RowRef<'_, RhsT, Len>) -> Self::Output {
		let lhs = &self.0;
		zip!(rhs).map(|unzip!(x)| *lhs * Conj::apply(x))
	}
}

impl<T: ComplexField, LhsT: Conjugate<Canonical = T>, Len: Shape> Mul<Scale<T>> for DiagRef<'_, LhsT, Len> {
	type Output = Diag<T, Len>;

	fn mul(self, rhs: Scale<T>) -> Self::Output {
		(self.column_vector() * rhs).into_diagonal()
	}
}
impl<T: ComplexField, RhsT: Conjugate<Canonical = T>, Len: Shape> Mul<DiagRef<'_, RhsT, Len>> for Scale<T> {
	type Output = Diag<T, Len>;

	fn mul(self, rhs: DiagRef<'_, RhsT, Len>) -> Self::Output {
		(self * rhs.column_vector()).into_diagonal()
	}
}

impl_mul_scalar!(MatMut<'_,  LhsT, Rows, Cols>, Scale<T>, Mat< T, Rows, Cols>);
impl_mul_scalar!(Mat< LhsT, Rows, Cols>, Scale<T>, Mat< T, Rows, Cols>);
impl_mul_scalar!(&MatRef<'_,  LhsT, Rows, Cols>, Scale<T>, Mat< T, Rows, Cols>);
impl_mul_scalar!(&MatMut<'_,  LhsT, Rows, Cols>, Scale<T>, Mat< T, Rows, Cols>);
impl_mul_scalar!(&Mat< LhsT, Rows, Cols>, Scale<T>, Mat< T, Rows, Cols>);

impl_div_scalar!(MatRef<'_,  LhsT, Rows, Cols>, Scale<T>, Mat< T, Rows, Cols>);
impl_div_scalar!(MatMut<'_,  LhsT, Rows, Cols>, Scale<T>, Mat< T, Rows, Cols>);
impl_div_scalar!(Mat< LhsT, Rows, Cols>, Scale<T>, Mat< T, Rows, Cols>);
impl_div_scalar!(&MatRef<'_,  LhsT, Rows, Cols>, Scale<T>, Mat< T, Rows, Cols>);
impl_div_scalar!(&MatMut<'_,  LhsT, Rows, Cols>, Scale<T>, Mat< T, Rows, Cols>);
impl_div_scalar!(&Mat< LhsT, Rows, Cols>, Scale<T>, Mat< T, Rows, Cols>);

impl_scalar_mul!(Scale<T>, MatMut<'_,  RhsT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_scalar_mul!(Scale<T>, Mat< RhsT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_scalar_mul!(Scale<T>, &MatRef<'_,  RhsT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_scalar_mul!(Scale<T>, &MatMut<'_,  RhsT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_scalar_mul!(Scale<T>, &Mat< RhsT, Rows, Cols>, Mat< T, Rows, Cols>);

impl_mul_primitive!(MatRef<'_,  RhsT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_mul_primitive!(MatMut<'_,  RhsT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_mul_primitive!(Mat< RhsT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_mul_primitive!(&MatRef<'_,  RhsT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_mul_primitive!(&MatMut<'_,  RhsT, Rows, Cols>, Mat< T, Rows, Cols>);
impl_mul_primitive!(&Mat< RhsT, Rows, Cols>, Mat< T, Rows, Cols>);

// impl_mul_scalar!(ColRef<'_,  LhsT>, Scale<T>, Col< T>);
impl_1d_mul_scalar!(ColMut<'_,  LhsT, Len>, Scale<T>, Col< T, Len>);
impl_1d_mul_scalar!(Col< LhsT, Len>, Scale<T>, Col< T, Len>);
impl_1d_mul_scalar!(&ColRef<'_,  LhsT, Len>, Scale<T>, Col< T, Len>);
impl_1d_mul_scalar!(&ColMut<'_,  LhsT, Len>, Scale<T>, Col< T, Len>);
impl_1d_mul_scalar!(&Col< LhsT, Len>, Scale<T>, Col< T, Len>);

impl_1d_div_scalar!(ColRef<'_,  LhsT, Len>, Scale<T>, Col< T, Len>);
impl_1d_div_scalar!(ColMut<'_,  LhsT, Len>, Scale<T>, Col< T, Len>);
impl_1d_div_scalar!(Col< LhsT, Len>, Scale<T>, Col< T, Len>);
impl_1d_div_scalar!(&ColRef<'_,  LhsT, Len>, Scale<T>, Col< T, Len>);
impl_1d_div_scalar!(&ColMut<'_,  LhsT, Len>, Scale<T>, Col< T, Len>);
impl_1d_div_scalar!(&Col< LhsT, Len>, Scale<T>, Col< T, Len>);

impl_1d_scalar_mul!(Scale<T>, ColMut<'_,  RhsT, Len>, Col< T, Len>);
impl_1d_scalar_mul!(Scale<T>, Col< RhsT, Len>, Col< T, Len>);
impl_1d_scalar_mul!(Scale<T>, &ColRef<'_,  RhsT, Len>, Col< T, Len>);
impl_1d_scalar_mul!(Scale<T>, &ColMut<'_,  RhsT, Len>, Col< T, Len>);
impl_1d_scalar_mul!(Scale<T>, &Col< RhsT, Len>, Col< T, Len>);

impl_1d_mul_primitive!(ColRef<'_,  RhsT, Len>, Col< T, Len>);
impl_1d_mul_primitive!(ColMut<'_,  RhsT, Len>, Col< T, Len>);
impl_1d_mul_primitive!(Col< RhsT, Len>, Col< T, Len>);
impl_1d_mul_primitive!(&ColRef<'_,  RhsT, Len>, Col< T, Len>);
impl_1d_mul_primitive!(&ColMut<'_,  RhsT, Len>, Col< T, Len>);
impl_1d_mul_primitive!(&Col< RhsT, Len>, Col< T, Len>);

impl_1d_mul_scalar!(RowMut<'_,  LhsT, Len>, Scale<T>, Row< T, Len>);
impl_1d_mul_scalar!(Row< LhsT, Len>, Scale<T>, Row< T, Len>);
impl_1d_mul_scalar!(&RowRef<'_,  LhsT, Len>, Scale<T>, Row< T, Len>);
impl_1d_mul_scalar!(&RowMut<'_,  LhsT, Len>, Scale<T>, Row< T, Len>);
impl_1d_mul_scalar!(&Row< LhsT, Len>, Scale<T>, Row< T, Len>);

impl_1d_div_scalar!(RowRef<'_,  LhsT, Len>, Scale<T>, Row< T, Len>);
impl_1d_div_scalar!(RowMut<'_,  LhsT, Len>, Scale<T>, Row< T, Len>);
impl_1d_div_scalar!(Row< LhsT, Len>, Scale<T>, Row< T, Len>);
impl_1d_div_scalar!(&RowRef<'_,  LhsT, Len>, Scale<T>, Row< T, Len>);
impl_1d_div_scalar!(&RowMut<'_,  LhsT, Len>, Scale<T>, Row< T, Len>);
impl_1d_div_scalar!(&Row< LhsT, Len>, Scale<T>, Row< T, Len>);

impl_1d_scalar_mul!(Scale<T>, RowMut<'_,  RhsT, Len>, Row< T, Len>);
impl_1d_scalar_mul!(Scale<T>, Row< RhsT, Len>, Row< T, Len>);
impl_1d_scalar_mul!(Scale<T>, &RowRef<'_,  RhsT, Len>, Row< T, Len>);
impl_1d_scalar_mul!(Scale<T>, &RowMut<'_,  RhsT, Len>, Row< T, Len>);
impl_1d_scalar_mul!(Scale<T>, &Row< RhsT, Len>, Row< T, Len>);

impl_1d_mul_primitive!(RowRef<'_,  RhsT, Len>, Row< T, Len>);
impl_1d_mul_primitive!(RowMut<'_,  RhsT, Len>, Row< T, Len>);
impl_1d_mul_primitive!(Row< RhsT, Len>, Row< T, Len>);
impl_1d_mul_primitive!(&RowRef<'_,  RhsT, Len>, Row< T, Len>);
impl_1d_mul_primitive!(&RowMut<'_,  RhsT, Len>, Row< T, Len>);
impl_1d_mul_primitive!(&Row< RhsT, Len>, Row< T, Len>);

impl_1d_mul_scalar!(DiagMut<'_,  LhsT, Len>, Scale<T>, Diag< T, Len>);
impl_1d_mul_scalar!(Diag< LhsT, Len>, Scale<T>, Diag< T, Len>);
impl_1d_mul_scalar!(&DiagRef<'_,  LhsT, Len>, Scale<T>, Diag< T, Len>);
impl_1d_mul_scalar!(&DiagMut<'_,  LhsT, Len>, Scale<T>, Diag< T, Len>);
impl_1d_mul_scalar!(&Diag< LhsT, Len>, Scale<T>, Diag< T, Len>);

impl_1d_div_scalar!(DiagRef<'_,  LhsT, Len>, Scale<T>, Diag< T, Len>);
impl_1d_div_scalar!(DiagMut<'_,  LhsT, Len>, Scale<T>, Diag< T, Len>);
impl_1d_div_scalar!(Diag< LhsT, Len>, Scale<T>, Diag< T, Len>);
impl_1d_div_scalar!(&DiagRef<'_,  LhsT, Len>, Scale<T>, Diag< T, Len>);
impl_1d_div_scalar!(&DiagMut<'_,  LhsT, Len>, Scale<T>, Diag< T, Len>);
impl_1d_div_scalar!(&Diag< LhsT, Len>, Scale<T>, Diag< T, Len>);

impl_1d_scalar_mul!(Scale<T>, DiagMut<'_,  RhsT, Len>, Diag< T, Len>);
impl_1d_scalar_mul!(Scale<T>, Diag< RhsT, Len>, Diag< T, Len>);
impl_1d_scalar_mul!(Scale<T>, &DiagRef<'_,  RhsT, Len>, Diag< T, Len>);
impl_1d_scalar_mul!(Scale<T>, &DiagMut<'_,  RhsT, Len>, Diag< T, Len>);
impl_1d_scalar_mul!(Scale<T>, &Diag< RhsT, Len>, Diag< T, Len>);

impl_1d_mul_primitive!(DiagRef<'_,  RhsT, Len>, Diag< T, Len>);
impl_1d_mul_primitive!(DiagMut<'_,  RhsT, Len>, Diag< T, Len>);
impl_1d_mul_primitive!(Diag< RhsT, Len>, Diag< T, Len>);
impl_1d_mul_primitive!(&DiagRef<'_,  RhsT, Len>, Diag< T, Len>);
impl_1d_mul_primitive!(&DiagMut<'_,  RhsT, Len>, Diag< T, Len>);
impl_1d_mul_primitive!(&Diag< RhsT, Len>, Diag< T, Len>);

impl<LhsT: ComplexField, Rows: Shape, Cols: Shape> MulAssign<Scale<LhsT>> for MatMut<'_, LhsT, Rows, Cols> {
	#[math]
	fn mul_assign(&mut self, rhs: Scale<LhsT>) {
		let rhs = &rhs.0;

		zip!(self.rb_mut()).for_each(|unzip!(x)| *x = *x * *rhs)
	}
}
impl<LhsT: ComplexField, Len: Shape> MulAssign<Scale<LhsT>> for ColMut<'_, LhsT, Len> {
	#[math]
	fn mul_assign(&mut self, rhs: Scale<LhsT>) {
		let rhs = &rhs.0;

		zip!(self.rb_mut()).for_each(|unzip!(x)| *x = *x * *rhs)
	}
}
impl<LhsT: ComplexField, Len: Shape> MulAssign<Scale<LhsT>> for RowMut<'_, LhsT, Len> {
	#[math]
	fn mul_assign(&mut self, rhs: Scale<LhsT>) {
		let rhs = &rhs.0;

		zip!(self.rb_mut()).for_each(|unzip!(x)| *x = *x * *rhs)
	}
}
impl<LhsT: ComplexField, Len: Shape> MulAssign<Scale<LhsT>> for DiagMut<'_, LhsT, Len> {
	fn mul_assign(&mut self, rhs: Scale<LhsT>) {
		let mut this = self.rb_mut().column_vector_mut();
		this *= rhs;
	}
}

impl_mul_assign_scalar!(Mat< LhsT, Rows, Cols>, Scale<LhsT>);
impl_1d_mul_assign_scalar!(Col< LhsT, Len>, Scale<LhsT>);
impl_1d_mul_assign_scalar!(Row< LhsT, Len>, Scale<LhsT>);
impl_1d_mul_assign_scalar!(Diag< LhsT, Len>, Scale<LhsT>);

impl_div_assign_scalar!(MatMut<'_, LhsT, Rows, Cols>, Scale<LhsT>);
impl_div_assign_scalar!(Mat< LhsT, Rows, Cols>, Scale<LhsT>);
impl_1d_div_assign_scalar!(ColMut<'_, LhsT, Len>, Scale<LhsT>);
impl_1d_div_assign_scalar!(Col< LhsT, Len>, Scale<LhsT>);
impl_1d_div_assign_scalar!(RowMut<'_, LhsT, Len>, Scale<LhsT>);
impl_1d_div_assign_scalar!(Row< LhsT, Len>, Scale<LhsT>);
impl_1d_div_assign_scalar!(DiagMut<'_, LhsT, Len>, Scale<LhsT>);
impl_1d_div_assign_scalar!(Diag< LhsT, Len>, Scale<LhsT>);

impl_mul_assign_primitive!(MatMut<'_, LhsT, Rows, Cols>);
impl_mul_assign_primitive!(Mat< LhsT, Rows, Cols>);
impl_1d_mul_assign_primitive!(ColMut<'_, LhsT, Len>);
impl_1d_mul_assign_primitive!(Col< LhsT, Len>);
impl_1d_mul_assign_primitive!(RowMut<'_, LhsT, Len>);
impl_1d_mul_assign_primitive!(Row< LhsT, Len>);
impl_1d_mul_assign_primitive!(DiagMut<'_, LhsT, Len>);
impl_1d_mul_assign_primitive!(Diag< LhsT, Len>);

#[cfg(feature = "sparse")]
mod sparse {
	use super::*;
	use crate::internal_prelude_sp::*;

	macro_rules! impl_scalar_mul_sparse {
		($lhs: ty, $rhs: ty, $out: ty) => {
			impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Mul<$rhs> for $lhs {
				type Output = $out;

				#[track_caller]
				fn mul(self, other: $rhs) -> Self::Output {
					self.mul(other.rb())
				}
			}
			impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Mul<$rhs> for &$lhs {
				type Output = $out;

				#[track_caller]
				fn mul(self, other: $rhs) -> Self::Output {
					self.clone().mul(other.rb())
				}
			}
		};
	}

	impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Mul<SparseRowMatRef<'_, I, T, Rows, Cols>> for Scale<T::Canonical> {
		type Output = SparseRowMat<I, T::Canonical, Rows, Cols>;

		#[track_caller]
		fn mul(self, rhs: SparseRowMatRef<'_, I, T, Rows, Cols>) -> Self::Output {
			self.mul(rhs.transpose()).into_transpose()
		}
	}

	impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Mul<SparseColMatRef<'_, I, T, Rows, Cols>> for Scale<T::Canonical> {
		type Output = SparseColMat<I, T::Canonical, Rows, Cols>;

		#[track_caller]
		fn mul(self, rhs: SparseColMatRef<'_, I, T, Rows, Cols>) -> Self::Output {
			#[math]
			fn imp<'ROWS, 'COLS, I: Index, T: Conjugate>(
				lhs: Scale<T::Canonical>,
				rhs: SparseColMatRef<'_, I, T, Dim<'ROWS>, Dim<'COLS>>,
			) -> SparseColMat<I, T::Canonical, Dim<'ROWS>, Dim<'COLS>> {
				let symbolic = rhs.symbolic().to_owned().unwrap();
				let mut val = alloc::vec::Vec::new();
				val.resize(symbolic.row_idx().len(), zero());
				let lhs = lhs.0;

				for j in rhs.ncols().indices() {
					for (val, rhs) in iter::zip(&mut val[symbolic.col_range(j)], rhs.val_of_col(j)) {
						*val = lhs * Conj::apply(rhs);
					}
				}

				SparseColMat::new(symbolic, val)
			}

			with_dim!(ROWS, rhs.nrows().unbound());
			with_dim!(COLS, rhs.ncols().unbound());

			imp(self, rhs.as_shape(ROWS, COLS)).into_shape(rhs.nrows(), rhs.ncols())
		}
	}

	macro_rules! impl_mul_scalar_sparse {
		($lhs: ty, $rhs: ty, $out: ty) => {
			impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Mul<$rhs> for $lhs {
				type Output = $out;

				#[track_caller]
				fn mul(self, other: $rhs) -> Self::Output {
					self.rb().mul(other)
				}
			}

			impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Div<$rhs> for $lhs {
				type Output = $out;

				#[track_caller]
				fn div(self, other: $rhs) -> Self::Output {
					self.rb().div(other)
				}
			}
			impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Mul<&$rhs> for $lhs {
				type Output = $out;

				#[track_caller]
				fn mul(self, other: &$rhs) -> Self::Output {
					self.rb().mul(other.clone())
				}
			}

			impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Div<&$rhs> for $lhs {
				type Output = $out;

				#[track_caller]
				fn div(self, other: &$rhs) -> Self::Output {
					self.rb().div(other.clone())
				}
			}
		};
	}

	macro_rules! impl_mul_assign_scalar_sparse {
		($lhs: ty, $rhs: ty) => {
			impl<I: Index, T: ComplexField, Rows: Shape, Cols: Shape> MulAssign<$rhs> for $lhs {
				#[track_caller]
				fn mul_assign(&mut self, other: $rhs) {
					(*self).rb_mut().mul_assign(other)
				}
			}

			impl<I: Index, T: ComplexField, Rows: Shape, Cols: Shape> DivAssign<$rhs> for $lhs {
				#[track_caller]
				fn div_assign(&mut self, other: $rhs) {
					(*self).rb_mut().div_assign(other)
				}
			}
			impl<I: Index, T: ComplexField, Rows: Shape, Cols: Shape> MulAssign<&$rhs> for $lhs {
				#[track_caller]
				fn mul_assign(&mut self, other: &$rhs) {
					(*self).rb_mut().mul_assign(other.clone())
				}
			}

			impl<I: Index, T: ComplexField, Rows: Shape, Cols: Shape> DivAssign<&$rhs> for $lhs {
				#[track_caller]
				fn div_assign(&mut self, other: &$rhs) {
					(*self).rb_mut().div_assign(other.clone())
				}
			}
		};
	}

	impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Mul<Scale<T::Canonical>> for SparseRowMatRef<'_, I, T, Rows, Cols> {
		type Output = SparseRowMat<I, T::Canonical, Rows, Cols>;

		#[track_caller]
		fn mul(self, rhs: Scale<T::Canonical>) -> Self::Output {
			self.transpose().mul(rhs).into_transpose()
		}
	}

	impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Div<Scale<T::Canonical>> for SparseRowMatRef<'_, I, T, Rows, Cols> {
		type Output = SparseRowMat<I, T::Canonical, Rows, Cols>;

		#[track_caller]
		fn div(self, rhs: Scale<T::Canonical>) -> Self::Output {
			self.transpose().div(rhs).into_transpose()
		}
	}

	impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Mul<Scale<T::Canonical>> for SparseColMatRef<'_, I, T, Rows, Cols> {
		type Output = SparseColMat<I, T::Canonical, Rows, Cols>;

		#[track_caller]
		fn mul(self, rhs: Scale<T::Canonical>) -> Self::Output {
			#[math]
			fn imp<'ROWS, 'COLS, I: Index, T: Conjugate>(
				lhs: SparseColMatRef<'_, I, T, Dim<'ROWS>, Dim<'COLS>>,
				rhs: Scale<T::Canonical>,
			) -> SparseColMat<I, T::Canonical, Dim<'ROWS>, Dim<'COLS>> {
				let symbolic = lhs.symbolic().to_owned().unwrap();
				let mut val = alloc::vec::Vec::new();
				val.resize(symbolic.row_idx().len(), zero());
				let rhs = rhs.0;

				for j in lhs.ncols().indices() {
					for (val, lhs) in iter::zip(&mut val[symbolic.col_range(j)], lhs.val_of_col(j)) {
						*val = Conj::apply(lhs) * rhs;
					}
				}

				SparseColMat::new(symbolic, val)
			}

			with_dim!(ROWS, self.nrows().unbound());
			with_dim!(COLS, self.ncols().unbound());

			imp(self.as_shape(ROWS, COLS), rhs).into_shape(self.nrows(), self.ncols())
		}
	}

	impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Div<Scale<T::Canonical>> for SparseColMatRef<'_, I, T, Rows, Cols> {
		type Output = SparseColMat<I, T::Canonical, Rows, Cols>;

		#[track_caller]
		#[math]
		fn div(self, rhs: Scale<T::Canonical>) -> Self::Output {
			self.mul(Scale(recip(rhs.0)))
		}
	}

	impl<I: Index, T: ComplexField, Rows: Shape, Cols: Shape> MulAssign<Scale<T>> for SparseColMatMut<'_, I, T, Rows, Cols> {
		#[track_caller]
		fn mul_assign(&mut self, rhs: Scale<T>) {
			#[math]
			fn imp<'ROWS, 'COLS, I: Index, T: ComplexField>(mut lhs: SparseColMatMut<'_, I, T, Dim<'ROWS>, Dim<'COLS>>, rhs: Scale<T>) {
				let rhs = rhs.0;

				for j in lhs.ncols().indices() {
					let v = lhs.rb_mut().val_of_col_mut(j);

					for val in v {
						*val = *val * rhs;
					}
				}
			}

			let (nrows, ncols) = self.shape();
			with_dim!(ROWS, nrows.unbound());
			with_dim!(COLS, ncols.unbound());

			imp(self.rb_mut().as_shape_mut(ROWS, COLS), rhs);
		}
	}

	impl<I: Index, T: ComplexField, Rows: Shape, Cols: Shape> DivAssign<Scale<T>> for SparseColMatMut<'_, I, T, Rows, Cols> {
		#[track_caller]
		#[math]
		fn div_assign(&mut self, rhs: Scale<T>) {
			self.mul_assign(Scale(recip(rhs.0)))
		}
	}

	macro_rules! impl_mul_primitive_sparse {
		($rhs: ty, $out: ty) => {
			impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Mul<$rhs> for f64 {
				type Output = $out;

				#[track_caller]
				fn mul(self, other: $rhs) -> Self::Output {
					Scale(T::Canonical::from_f64_impl(self)).mul(other.rb())
				}
			}

			impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Mul<f64> for $rhs {
				type Output = $out;

				#[track_caller]
				fn mul(self, other: f64) -> Self::Output {
					self.rb().mul(Scale(T::Canonical::from_f64_impl(other)))
				}
			}
			impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Div<f64> for $rhs {
				type Output = $out;

				#[track_caller]
				fn div(self, other: f64) -> Self::Output {
					self.rb().mul(Scale(T::Canonical::from_f64_impl(f64::recip(other))))
				}
			}
		};
	}

	macro_rules! impl_neg_sparse {
		($mat: ty, $out: ty) => {
			impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Neg for $mat {
				type Output = $out;

				fn neg(self) -> Self::Output {
					self.rb().neg()
				}
			}
		};
	}

	impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Neg for SparseRowMatRef<'_, I, T, Rows, Cols> {
		type Output = SparseRowMat<I, T::Canonical, Rows, Cols>;

		#[math]
		fn neg(self) -> Self::Output {
			self.transpose().neg().into_transpose()
		}
	}

	impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Neg for SparseColMatRef<'_, I, T, Rows, Cols> {
		type Output = SparseColMat<I, T::Canonical, Rows, Cols>;

		#[math]
		fn neg(self) -> Self::Output {
			fn imp<'ROWS, 'COLS, I: Index, T: Conjugate>(
				mat: SparseColMatRef<'_, I, T, Dim<'ROWS>, Dim<'COLS>>,
			) -> SparseColMat<I, T::Canonical, Dim<'ROWS>, Dim<'COLS>> {
				let symbolic = mat.symbolic().to_owned().unwrap();
				let mut val = alloc::vec::Vec::new();
				val.resize(symbolic.row_idx().len(), zero());

				for j in mat.ncols().indices() {
					for (val, lhs) in iter::zip(&mut val[symbolic.col_range(j)], mat.val_of_col(j)) {
						*val = -Conj::apply(lhs);
					}
				}

				SparseColMat::new(symbolic, val)
			}
			with_dim!(ROWS, self.nrows().unbound());
			with_dim!(COLS, self.ncols().unbound());
			imp(self.as_shape(ROWS, COLS)).into_shape(self.nrows(), self.ncols())
		}
	}

	macro_rules! impl_add_sub_sparse {
		($lhs: ty, $rhs: ty, $out: ty) => {
			impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> Add<$rhs>
				for $lhs
			{
				type Output = $out;

				#[track_caller]
				fn add(self, rhs: $rhs) -> Self::Output {
					self.rb().add(rhs.rb())
				}
			}

			impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> Sub<$rhs>
				for $lhs
			{
				type Output = $out;

				#[track_caller]
				fn sub(self, rhs: $rhs) -> Self::Output {
					self.rb().sub(rhs.rb())
				}
			}
		};
	}

	#[math]
	fn add_fn<T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>>(lhs: Option<&LhsT>, rhs: Option<&RhsT>) -> T {
		lhs.map(Conj::apply).unwrap_or_else(zero::<T>) + rhs.map(Conj::apply).unwrap_or_else(zero::<T>)
	}

	#[math]
	fn sub_fn<T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>>(lhs: Option<&LhsT>, rhs: Option<&RhsT>) -> T {
		lhs.map(Conj::apply).unwrap_or_else(zero::<T>) - rhs.map(Conj::apply).unwrap_or_else(zero::<T>)
	}

	impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape>
		Add<SparseColMatRef<'_, I, RhsT, Rows, Cols>> for SparseColMatRef<'_, I, LhsT, Rows, Cols>
	{
		type Output = SparseColMat<I, T, Rows, Cols>;

		#[track_caller]
		fn add(self, rhs: SparseColMatRef<'_, I, RhsT, Rows, Cols>) -> Self::Output {
			let (nrows, ncols) = self.shape();
			crate::sparse::ops::binary_op(self.as_dyn(), rhs.as_dyn(), add_fn::<T, LhsT, RhsT>)
				.unwrap()
				.into_shape(nrows, ncols)
		}
	}

	impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape>
		Sub<SparseColMatRef<'_, I, RhsT, Rows, Cols>> for SparseColMatRef<'_, I, LhsT, Rows, Cols>
	{
		type Output = SparseColMat<I, T, Rows, Cols>;

		#[track_caller]
		fn sub(self, rhs: SparseColMatRef<'_, I, RhsT, Rows, Cols>) -> Self::Output {
			let (nrows, ncols) = self.shape();
			crate::sparse::ops::binary_op(self.as_dyn(), rhs.as_dyn(), sub_fn)
				.unwrap()
				.into_shape(nrows, ncols)
		}
	}

	impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape>
		Add<SparseRowMatRef<'_, I, RhsT, Rows, Cols>> for SparseRowMatRef<'_, I, LhsT, Rows, Cols>
	{
		type Output = SparseRowMat<I, T, Rows, Cols>;

		#[track_caller]
		fn add(self, rhs: SparseRowMatRef<'_, I, RhsT, Rows, Cols>) -> Self::Output {
			self.transpose().add(rhs.transpose()).into_transpose()
		}
	}

	impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape>
		Sub<SparseRowMatRef<'_, I, RhsT, Rows, Cols>> for SparseRowMatRef<'_, I, LhsT, Rows, Cols>
	{
		type Output = SparseRowMat<I, T, Rows, Cols>;

		#[track_caller]
		fn sub(self, rhs: SparseRowMatRef<'_, I, RhsT, Rows, Cols>) -> Self::Output {
			self.transpose().sub(rhs.transpose()).into_transpose()
		}
	}

	macro_rules! impl_add_sub_assign_sparse {
		($lhs: ty, $rhs: ty) => {
			impl<I: Index, T: ComplexField, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> AddAssign<$rhs> for $lhs {
				#[track_caller]
				fn add_assign(&mut self, rhs: $rhs) {
					self.rb_mut().add_assign(rhs.rb())
				}
			}

			impl<I: Index, T: ComplexField, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> SubAssign<$rhs> for $lhs {
				#[track_caller]
				fn sub_assign(&mut self, rhs: $rhs) {
					self.rb_mut().sub_assign(rhs.rb())
				}
			}
		};
	}

	#[math]
	fn add_assign_fn<T: ComplexField, RhsT: Conjugate<Canonical = T>>(dst: &mut T, rhs: Option<&RhsT>) {
		if let Some(rhs) = rhs {
			*dst = *dst + Conj::apply(rhs);
		}
	}

	#[math]
	fn sub_assign_fn<T: ComplexField, RhsT: Conjugate<Canonical = T>>(dst: &mut T, rhs: Option<&RhsT>) {
		if let Some(rhs) = rhs {
			*dst = *dst - Conj::apply(rhs);
		}
	}

	impl<I: Index, T: ComplexField, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> AddAssign<SparseColMatRef<'_, I, RhsT, Rows, Cols>>
		for SparseColMatMut<'_, I, T, Rows, Cols>
	{
		#[track_caller]
		fn add_assign(&mut self, rhs: SparseColMatRef<'_, I, RhsT, Rows, Cols>) {
			crate::sparse::ops::binary_op_assign_into(self.rb_mut().as_dyn_mut(), rhs.as_dyn(), add_assign_fn::<T, RhsT>);
		}
	}
	impl<I: Index, T: ComplexField, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> SubAssign<SparseColMatRef<'_, I, RhsT, Rows, Cols>>
		for SparseColMatMut<'_, I, T, Rows, Cols>
	{
		#[track_caller]
		fn sub_assign(&mut self, rhs: SparseColMatRef<'_, I, RhsT, Rows, Cols>) {
			crate::sparse::ops::binary_op_assign_into(self.rb_mut().as_dyn_mut(), rhs.as_dyn(), sub_assign_fn::<T, RhsT>);
		}
	}

	impl<I: Index, T: ComplexField, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> AddAssign<SparseRowMatRef<'_, I, RhsT, Rows, Cols>>
		for SparseRowMatMut<'_, I, T, Rows, Cols>
	{
		#[track_caller]
		fn add_assign(&mut self, rhs: SparseRowMatRef<'_, I, RhsT, Rows, Cols>) {
			self.rb_mut().transpose_mut().add_assign(rhs.transpose())
		}
	}
	impl<I: Index, T: ComplexField, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> SubAssign<SparseRowMatRef<'_, I, RhsT, Rows, Cols>>
		for SparseRowMatMut<'_, I, T, Rows, Cols>
	{
		#[track_caller]
		fn sub_assign(&mut self, rhs: SparseRowMatRef<'_, I, RhsT, Rows, Cols>) {
			self.rb_mut().transpose_mut().sub_assign(rhs.transpose())
		}
	}

	macro_rules! impl_matmul_sparse {
		($lhs: ty, $rhs: ty, $out: ty) => {
			impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape, Depth: Shape>
				Mul<$rhs> for $lhs
			{
				type Output = $out;

				#[track_caller]
				fn mul(self, rhs: $rhs) -> Self::Output {
					self.rb().mul(rhs.rb())
				}
			}
		};
	}

	macro_rules! impl_matmul_1_sparse {
		($lhs: ty, $rhs: ty, $out: ty) => {
			impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape> Mul<$rhs>
				for $lhs
			{
				type Output = $out;

				#[track_caller]
				fn mul(self, rhs: $rhs) -> Self::Output {
					self.rb().mul(rhs.rb())
				}
			}
		};
	}

	macro_rules! impl_matmul_perm_sparse {
		($lhs: ty, $rhs: ty, $out: ty) => {
			impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Mul<$rhs> for $lhs {
				type Output = $out;

				#[track_caller]
				fn mul(self, rhs: $rhs) -> Self::Output {
					self.rb().mul(rhs.rb())
				}
			}
		};
	}

	impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape, Depth: Shape>
		Mul<SparseColMatRef<'_, I, RhsT, Depth, Cols>> for SparseColMatRef<'_, I, LhsT, Rows, Depth>
	{
		type Output = SparseColMat<I, T, Rows, Cols>;

		#[track_caller]
		fn mul(self, rhs: SparseColMatRef<'_, I, RhsT, Depth, Cols>) -> Self::Output {
			let nrows = self.nrows();
			let ncols = rhs.ncols();
			linalg_sp::matmul::sparse_sparse_matmul(self.as_dyn(), rhs.as_dyn(), T::one_impl(), crate::get_global_parallelism())
				.unwrap()
				.into_shape(nrows, ncols)
		}
	}

	impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape, Depth: Shape>
		Mul<MatRef<'_, RhsT, Depth, Cols>> for SparseColMatRef<'_, I, LhsT, Rows, Depth>
	{
		type Output = Mat<T, Rows, Cols>;

		#[track_caller]
		fn mul(self, rhs: MatRef<'_, RhsT, Depth, Cols>) -> Self::Output {
			let nrows = self.nrows();
			let ncols = rhs.ncols();
			let mut out = Mat::zeros(nrows, ncols);
			linalg_sp::matmul::sparse_dense_matmul(
				out.as_dyn_mut(),
				Accum::Add,
				self.as_dyn(),
				rhs.as_dyn(),
				T::one_impl(),
				crate::get_global_parallelism(),
			);
			out
		}
	}

	impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape, Depth: Shape>
		Mul<SparseColMatRef<'_, I, RhsT, Depth, Cols>> for MatRef<'_, LhsT, Rows, Depth>
	{
		type Output = Mat<T, Rows, Cols>;

		#[track_caller]
		fn mul(self, rhs: SparseColMatRef<'_, I, RhsT, Depth, Cols>) -> Self::Output {
			let nrows = self.nrows();
			let ncols = rhs.ncols();
			let mut out = Mat::zeros(nrows, ncols);
			linalg_sp::matmul::dense_sparse_matmul(
				out.as_dyn_mut(),
				Accum::Add,
				self.as_dyn(),
				rhs.as_dyn(),
				T::one_impl(),
				crate::get_global_parallelism(),
			);
			out
		}
	}

	impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape>
		Mul<ColRef<'_, RhsT, Cols>> for SparseColMatRef<'_, I, LhsT, Rows, Cols>
	{
		type Output = Col<T, Rows>;

		#[track_caller]
		fn mul(self, rhs: ColRef<'_, RhsT, Cols>) -> Self::Output {
			let (nrows, _) = self.shape();
			let mut out = Col::zeros(nrows);
			linalg_sp::matmul::sparse_dense_matmul(
				out.as_mat_mut().as_dyn_mut(),
				Accum::Add,
				self.as_dyn(),
				rhs.as_mat().as_dyn(),
				T::one_impl(),
				crate::get_global_parallelism(),
			);

			out
		}
	}

	impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape>
		Mul<SparseColMatRef<'_, I, RhsT, Rows, Cols>> for RowRef<'_, LhsT, Rows>
	{
		type Output = Row<T, Cols>;

		#[track_caller]
		fn mul(self, rhs: SparseColMatRef<'_, I, RhsT, Rows, Cols>) -> Self::Output {
			let (_, ncols) = rhs.shape();
			let mut out = Row::zeros(ncols);
			linalg_sp::matmul::dense_sparse_matmul(
				out.as_mat_mut().as_dyn_mut(),
				Accum::Add,
				self.as_mat().as_dyn(),
				rhs.as_dyn(),
				T::one_impl(),
				crate::get_global_parallelism(),
			);
			out
		}
	}

	impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape>
		Mul<DiagRef<'_, RhsT, Cols>> for SparseColMatRef<'_, I, LhsT, Rows, Cols>
	{
		type Output = SparseColMat<I, T, Rows, Cols>;

		#[track_caller]
		fn mul(self, rhs: DiagRef<'_, RhsT, Cols>) -> Self::Output {
			#[math]
			fn imp<'ROWS, 'COLS, I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>>(
				lhs: SparseColMatRef<'_, I, LhsT, Dim<'ROWS>, Dim<'COLS>>,
				rhs: DiagRef<'_, RhsT, Dim<'COLS>>,
			) -> SparseColMat<I, T, Dim<'ROWS>, Dim<'COLS>> {
				let symbolic = lhs.symbolic().to_owned().unwrap();
				let mut out = alloc::vec::Vec::new();
				out.resize(symbolic.row_idx().len(), T::zero_impl());

				for j in lhs.ncols().indices() {
					let rhs = Conj::apply(&rhs[j]);
					for (out, lhs) in iter::zip(&mut out[symbolic.col_range(j)], lhs.val_of_col(j)) {
						*out = Conj::apply(lhs) * rhs;
					}
				}

				SparseColMat::new(symbolic, out)
			}
			let lhs = self;
			with_dim!(M, lhs.nrows().unbound());
			with_dim!(N, lhs.ncols().unbound());
			imp(lhs.as_shape(M, N), rhs.as_shape(N)).into_shape(lhs.nrows(), lhs.ncols())
		}
	}

	impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape>
		Mul<SparseColMatRef<'_, I, RhsT, Rows, Cols>> for DiagRef<'_, LhsT, Rows>
	{
		type Output = SparseColMat<I, T, Rows, Cols>;

		#[track_caller]
		fn mul(self, rhs: SparseColMatRef<'_, I, RhsT, Rows, Cols>) -> Self::Output {
			#[math]
			fn imp<'ROWS, 'COLS, I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>>(
				lhs: DiagRef<'_, RhsT, Dim<'ROWS>>,
				rhs: SparseColMatRef<'_, I, LhsT, Dim<'ROWS>, Dim<'COLS>>,
			) -> SparseColMat<I, T, Dim<'ROWS>, Dim<'COLS>> {
				let symbolic = rhs.symbolic().to_owned().unwrap();
				let mut out = alloc::vec::Vec::new();
				out.resize(symbolic.row_idx().len(), T::zero_impl());

				for j in rhs.ncols().indices() {
					for ((i, out), rhs) in iter::zip(iter::zip(symbolic.row_idx_of_col(j), &mut out[symbolic.col_range(j)]), rhs.val_of_col(j)) {
						*out = Conj::apply(&lhs[i]) * Conj::apply(rhs);
					}
				}

				SparseColMat::new(symbolic, out)
			}
			let lhs = self;
			with_dim!(M, rhs.nrows().unbound());
			with_dim!(N, rhs.ncols().unbound());
			imp(lhs.as_shape(M), rhs.as_shape(M, N)).into_shape(rhs.nrows(), rhs.ncols())
		}
	}

	impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Mul<PermRef<'_, I, Cols>> for SparseColMatRef<'_, I, T, Rows, Cols> {
		type Output = SparseColMat<I, T::Canonical, Rows, Cols>;

		#[track_caller]
		fn mul(self, rhs: PermRef<'_, I, Cols>) -> Self::Output {
			#[math]
			fn imp<'ROWS, 'COLS, I: Index, T: Conjugate>(
				lhs: SparseColMatRef<'_, I, T, Dim<'ROWS>, Dim<'COLS>>,
				rhs: PermRef<'_, I, Dim<'COLS>>,
			) -> SparseColMat<I, T::Canonical, Dim<'ROWS>, Dim<'COLS>> {
				let symbolic = lhs.symbolic();

				let mut out_col_ptr = alloc::vec::Vec::new();
				let mut out_row_idx = alloc::vec::Vec::new();
				let mut out = alloc::vec::Vec::new();

				out_col_ptr.resize(symbolic.col_ptr().len(), I::truncate(0));
				out_row_idx.resize(symbolic.col_ptr().len(), I::truncate(0));
				out.resize(symbolic.row_idx().len(), T::Canonical::zero_impl());

				let inv = rhs.bound_arrays().1;

				let mut pos = 0usize;
				for j in lhs.ncols().indices() {
					let inv = inv[j].zx();
					let row_idx = lhs.as_dyn().row_idx_of_col_raw(*inv);
					let len = row_idx.len();
					out_row_idx[pos..][..len].copy_from_slice(row_idx);

					for (out, lhs) in iter::zip(&mut out[pos..][..len], lhs.val_of_col(inv)) {
						*out = Conj::apply(lhs);
					}

					pos += row_idx.len();
				}

				out_row_idx.truncate(pos);
				out.truncate(pos);

				SparseColMat::new(
					unsafe { SymbolicSparseColMat::new_unchecked(symbolic.nrows(), symbolic.ncols(), out_col_ptr, None, out_row_idx) },
					out,
				)
			}
			let lhs = self;
			with_dim!(M, lhs.nrows().unbound());
			with_dim!(N, lhs.ncols().unbound());
			imp(lhs.as_shape(M, N), rhs.as_shape(N)).into_shape(lhs.nrows(), lhs.ncols())
		}
	}

	impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Mul<SparseColMatRef<'_, I, T, Rows, Cols>> for PermRef<'_, I, Rows> {
		type Output = SparseColMat<I, T::Canonical, Rows, Cols>;

		#[track_caller]
		fn mul(self, rhs: SparseColMatRef<'_, I, T, Rows, Cols>) -> Self::Output {
			#[math]
			fn imp<'ROWS, 'COLS, I: Index, T: Conjugate>(
				lhs: PermRef<'_, I, Dim<'ROWS>>,
				rhs: SparseColMatRef<'_, I, T, Dim<'ROWS>, Dim<'COLS>>,
			) -> SparseColMat<I, T::Canonical, Dim<'ROWS>, Dim<'COLS>> {
				let symbolic = rhs.symbolic();

				let mut out_col_ptr = alloc::vec::Vec::new();
				let mut out_row_idx = alloc::vec::Vec::new();
				let mut out = alloc::vec::Vec::new();

				out_col_ptr.resize(symbolic.col_ptr().len(), I::truncate(0));
				out_row_idx.resize(symbolic.col_ptr().len(), I::truncate(0));
				out.resize(symbolic.row_idx().len(), T::Canonical::zero_impl());

				let inv = lhs.bound_arrays().0;

				let mut pos = 0usize;
				for j in rhs.ncols().indices() {
					let row_idx = rhs.as_dyn().row_idx_of_col_raw(*j);
					let len = row_idx.len();

					for ((out_i, out_v), (rhs_i, rhs_v)) in iter::zip(
						iter::zip(&mut out_row_idx[pos..][..len], &mut out[pos..][..len]),
						iter::zip(rhs.row_idx_of_col(j), rhs.val_of_col(j)),
					) {
						*out_i = *inv[rhs_i];
						*out_v = Conj::apply(rhs_v);
					}

					pos += row_idx.len();
				}

				out_row_idx.truncate(pos);
				out.truncate(pos);

				SparseColMat::new(
					unsafe { SymbolicSparseColMat::new_unchecked(symbolic.nrows(), symbolic.ncols(), out_col_ptr, None, out_row_idx) },
					out,
				)
			}
			let lhs = self;
			with_dim!(M, rhs.nrows().unbound());
			with_dim!(N, rhs.ncols().unbound());
			imp(lhs.as_shape(M), rhs.as_shape(M, N)).into_shape(rhs.nrows(), rhs.ncols())
		}
	}

	impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape, Depth: Shape>
		Mul<SparseRowMatRef<'_, I, RhsT, Depth, Cols>> for SparseRowMatRef<'_, I, LhsT, Rows, Depth>
	{
		type Output = SparseRowMat<I, T, Rows, Cols>;

		#[track_caller]
		fn mul(self, rhs: SparseRowMatRef<'_, I, RhsT, Depth, Cols>) -> Self::Output {
			rhs.transpose().mul(self.transpose()).into_transpose()
		}
	}

	impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape, Depth: Shape>
		Mul<MatRef<'_, RhsT, Depth, Cols>> for SparseRowMatRef<'_, I, LhsT, Rows, Depth>
	{
		type Output = Mat<T, Rows, Cols>;

		#[track_caller]
		fn mul(self, rhs: MatRef<'_, RhsT, Depth, Cols>) -> Self::Output {
			rhs.transpose().mul(self.transpose()).transpose().to_owned()
		}
	}

	impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape, Depth: Shape>
		Mul<SparseRowMatRef<'_, I, RhsT, Depth, Cols>> for MatRef<'_, LhsT, Rows, Depth>
	{
		type Output = Mat<T, Rows, Cols>;

		#[track_caller]
		fn mul(self, rhs: SparseRowMatRef<'_, I, RhsT, Depth, Cols>) -> Self::Output {
			rhs.transpose().mul(self.transpose()).transpose().to_owned()
		}
	}

	impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape>
		Mul<ColRef<'_, RhsT, Cols>> for SparseRowMatRef<'_, I, LhsT, Rows, Cols>
	{
		type Output = Col<T, Rows>;

		#[track_caller]
		fn mul(self, rhs: ColRef<'_, RhsT, Cols>) -> Self::Output {
			rhs.transpose().mul(self.transpose()).transpose().to_owned()
		}
	}

	impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape>
		Mul<SparseRowMatRef<'_, I, RhsT, Rows, Cols>> for RowRef<'_, LhsT, Rows>
	{
		type Output = Row<T, Cols>;

		#[track_caller]
		fn mul(self, rhs: SparseRowMatRef<'_, I, RhsT, Rows, Cols>) -> Self::Output {
			rhs.transpose().mul(self.transpose()).transpose().to_owned()
		}
	}

	impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape>
		Mul<DiagRef<'_, RhsT, Cols>> for SparseRowMatRef<'_, I, LhsT, Rows, Cols>
	{
		type Output = SparseRowMat<I, T, Rows, Cols>;

		#[track_caller]
		fn mul(self, rhs: DiagRef<'_, RhsT, Cols>) -> Self::Output {
			rhs.mul(self.transpose()).into_transpose()
		}
	}

	impl<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>, Rows: Shape, Cols: Shape>
		Mul<SparseRowMatRef<'_, I, RhsT, Rows, Cols>> for DiagRef<'_, LhsT, Rows>
	{
		type Output = SparseRowMat<I, T, Rows, Cols>;

		#[track_caller]
		fn mul(self, rhs: SparseRowMatRef<'_, I, RhsT, Rows, Cols>) -> Self::Output {
			rhs.transpose().mul(self).into_transpose()
		}
	}

	impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Mul<PermRef<'_, I, Cols>> for SparseRowMatRef<'_, I, T, Rows, Cols> {
		type Output = SparseRowMat<I, T::Canonical, Rows, Cols>;

		#[track_caller]
		fn mul(self, rhs: PermRef<'_, I, Cols>) -> Self::Output {
			rhs.inverse().mul(self.transpose()).into_transpose()
		}
	}

	impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape> Mul<SparseRowMatRef<'_, I, T, Rows, Cols>> for PermRef<'_, I, Rows> {
		type Output = SparseRowMat<I, T::Canonical, Rows, Cols>;

		#[track_caller]
		fn mul(self, rhs: SparseRowMatRef<'_, I, T, Rows, Cols>) -> Self::Output {
			rhs.transpose().mul(self.inverse()).into_transpose()
		}
	}

	impl_scalar_mul_sparse!(Scale<T::Canonical>, SparseColMatMut<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_scalar_mul_sparse!(Scale<T::Canonical>, SparseColMat<I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_scalar_mul_sparse!(Scale<T::Canonical>, &SparseColMatRef<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_scalar_mul_sparse!(Scale<T::Canonical>, &SparseColMatMut<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_scalar_mul_sparse!(Scale<T::Canonical>, &SparseColMat<I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_scalar_mul_sparse!(Scale<T::Canonical>, SparseRowMatMut<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_scalar_mul_sparse!(Scale<T::Canonical>, SparseRowMat<I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_scalar_mul_sparse!(Scale<T::Canonical>, &SparseRowMatRef<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_scalar_mul_sparse!(Scale<T::Canonical>, &SparseRowMatMut<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_scalar_mul_sparse!(Scale<T::Canonical>, &SparseRowMat<I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);

	impl_mul_scalar_sparse!(SparseColMatMut<'_, I, T, Rows, Cols>, Scale<T::Canonical>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_mul_scalar_sparse!(SparseColMat<I, T, Rows, Cols>, Scale<T::Canonical>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_mul_scalar_sparse!(&SparseColMatRef<'_, I, T, Rows, Cols>, Scale<T::Canonical>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_mul_scalar_sparse!(&SparseColMatMut<'_, I, T, Rows, Cols>, Scale<T::Canonical>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_mul_scalar_sparse!(&SparseColMat<I, T, Rows, Cols>, Scale<T::Canonical>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_mul_scalar_sparse!(SparseRowMatMut<'_, I, T, Rows, Cols>, Scale<T::Canonical>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_mul_scalar_sparse!(SparseRowMat<I, T, Rows, Cols>, Scale<T::Canonical>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_mul_scalar_sparse!(&SparseRowMatRef<'_, I, T, Rows, Cols>, Scale<T::Canonical>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_mul_scalar_sparse!(&SparseRowMatMut<'_, I, T, Rows, Cols>, Scale<T::Canonical>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_mul_scalar_sparse!(&SparseRowMat<I, T, Rows, Cols>, Scale<T::Canonical>, SparseRowMat<I, T::Canonical, Rows, Cols>);

	impl_mul_assign_scalar_sparse!(SparseColMat<I, T, Rows, Cols>, Scale<T::Canonical>);

	impl_mul_primitive_sparse!(SparseColMatRef<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_mul_primitive_sparse!(SparseColMatMut<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_mul_primitive_sparse!(SparseColMat<I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_mul_primitive_sparse!(&SparseColMatRef<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_mul_primitive_sparse!(&SparseColMatMut<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_mul_primitive_sparse!(&SparseColMat<I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_mul_primitive_sparse!(SparseRowMatRef<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_mul_primitive_sparse!(SparseRowMatMut<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_mul_primitive_sparse!(SparseRowMat<I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_mul_primitive_sparse!(&SparseRowMatRef<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_mul_primitive_sparse!(&SparseRowMatMut<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_mul_primitive_sparse!(&SparseRowMat<I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);

	impl_neg_sparse!(SparseColMatMut<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_neg_sparse!(SparseColMat<I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_neg_sparse!(&SparseColMatRef<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_neg_sparse!(&SparseColMatMut<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_neg_sparse!(&SparseColMat<I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_neg_sparse!(SparseRowMatMut<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_neg_sparse!(SparseRowMat<I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_neg_sparse!(&SparseRowMatRef<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_neg_sparse!(&SparseRowMatMut<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_neg_sparse!(&SparseRowMat<I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);

	impl_add_sub_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Cols>, SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Cols>, SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Cols>, &SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Cols>, &SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Cols>, &SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Cols>, SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Cols>, SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Cols>, SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Cols>, &SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Cols>, &SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Cols>, &SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Cols>, SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Cols>, SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Cols>, SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Cols>, &SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Cols>, &SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Cols>, &SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Cols>, SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Cols>, SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Cols>, SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Cols>, &SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Cols>, &SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Cols>, &SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseColMat<I, LhsT, Rows, Cols>, SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseColMat<I, LhsT, Rows, Cols>, SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseColMat<I, LhsT, Rows, Cols>, SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseColMat<I, LhsT, Rows, Cols>, &SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseColMat<I, LhsT, Rows, Cols>, &SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseColMat<I, LhsT, Rows, Cols>, &SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseColMat<I, LhsT, Rows, Cols>, SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseColMat<I, LhsT, Rows, Cols>, SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseColMat<I, LhsT, Rows, Cols>, SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseColMat<I, LhsT, Rows, Cols>, &SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseColMat<I, LhsT, Rows, Cols>, &SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseColMat<I, LhsT, Rows, Cols>, &SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);

	impl_add_sub_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Cols>, SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Cols>, SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Cols>, &SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Cols>, &SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Cols>, &SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Cols>, SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Cols>, SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Cols>, SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Cols>, &SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Cols>, &SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Cols>, &SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Cols>, SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Cols>, SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Cols>, SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Cols>, &SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Cols>, &SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Cols>, &SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Cols>, SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Cols>, SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Cols>, SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Cols>, &SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Cols>, &SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Cols>, &SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseRowMat<I, LhsT, Rows, Cols>, SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseRowMat<I, LhsT, Rows, Cols>, SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseRowMat<I, LhsT, Rows, Cols>, SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseRowMat<I, LhsT, Rows, Cols>, &SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseRowMat<I, LhsT, Rows, Cols>, &SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(SparseRowMat<I, LhsT, Rows, Cols>, &SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseRowMat<I, LhsT, Rows, Cols>, SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseRowMat<I, LhsT, Rows, Cols>, SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseRowMat<I, LhsT, Rows, Cols>, SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseRowMat<I, LhsT, Rows, Cols>, &SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseRowMat<I, LhsT, Rows, Cols>, &SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_add_sub_sparse!(&SparseRowMat<I, LhsT, Rows, Cols>, &SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);

	impl_add_sub_assign_sparse!(SparseColMatMut<'_, I, T, Rows, Cols>, SparseColMatMut<'_, I, RhsT, Rows, Cols>);
	impl_add_sub_assign_sparse!(SparseColMatMut<'_, I, T, Rows, Cols>, SparseColMat<I, RhsT, Rows, Cols>);
	impl_add_sub_assign_sparse!(SparseColMat<I, T, Rows, Cols>, SparseColMatRef<'_, I, RhsT, Rows, Cols>);
	impl_add_sub_assign_sparse!(SparseColMat<I, T, Rows, Cols>, SparseColMatMut<'_, I, RhsT, Rows, Cols>);
	impl_add_sub_assign_sparse!(SparseColMat<I, T, Rows, Cols>, SparseColMat<I, RhsT, Rows, Cols>);
	impl_add_sub_assign_sparse!(SparseRowMatMut<'_, I, T, Rows, Cols>, SparseRowMatMut<'_, I, RhsT, Rows, Cols>);
	impl_add_sub_assign_sparse!(SparseRowMatMut<'_, I, T, Rows, Cols>, SparseRowMat<I, RhsT, Rows, Cols>);
	impl_add_sub_assign_sparse!(SparseRowMat<I, T, Rows, Cols>, SparseRowMatRef<'_, I, RhsT, Rows, Cols>);
	impl_add_sub_assign_sparse!(SparseRowMat<I, T, Rows, Cols>, SparseRowMatMut<'_, I, RhsT, Rows, Cols>);
	impl_add_sub_assign_sparse!(SparseRowMat<I, T, Rows, Cols>, SparseRowMat<I, RhsT, Rows, Cols>);

	// matmul
	impl_matmul_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Depth>, SparseColMatMut<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Depth>, SparseColMat<I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Depth>, &SparseColMatRef<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Depth>, &SparseColMatMut<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Depth>, &SparseColMat<I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Depth>, SparseColMatRef<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Depth>, SparseColMatMut<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Depth>, SparseColMat<I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Depth>, &SparseColMatRef<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Depth>, &SparseColMatMut<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Depth>, &SparseColMat<I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMat<I, LhsT, Rows, Depth>, SparseColMatRef<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMat<I, LhsT, Rows, Depth>, SparseColMatMut<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMat<I, LhsT, Rows, Depth>, SparseColMat<I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMat<I, LhsT, Rows, Depth>, &SparseColMatRef<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMat<I, LhsT, Rows, Depth>, &SparseColMatMut<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMat<I, LhsT, Rows, Depth>, &SparseColMat<I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Depth>, SparseColMatRef<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Depth>, SparseColMatMut<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Depth>, SparseColMat<I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Depth>, &SparseColMatRef<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Depth>, &SparseColMatMut<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Depth>, &SparseColMat<I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Depth>, SparseColMatRef<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Depth>, SparseColMatMut<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Depth>, SparseColMat<I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Depth>, &SparseColMatRef<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Depth>, &SparseColMatMut<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Depth>, &SparseColMat<I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMat<I, LhsT, Rows, Depth>, SparseColMatRef<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMat<I, LhsT, Rows, Depth>, SparseColMatMut<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMat<I, LhsT, Rows, Depth>, SparseColMat<I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMat<I, LhsT, Rows, Depth>, &SparseColMatRef<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMat<I, LhsT, Rows, Depth>, &SparseColMatMut<'_, I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMat<I, LhsT, Rows, Depth>, &SparseColMat<I, RhsT, Depth, Cols>, SparseColMat<I, T, Rows, Cols>);

	impl_matmul_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Depth>, MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Depth>, Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Depth>, &MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Depth>, &MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Depth>, &Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Depth>, MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Depth>, MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Depth>, Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Depth>, &MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Depth>, &MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Depth>, &Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMat<I, LhsT, Rows, Depth>, MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMat<I, LhsT, Rows, Depth>, MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMat<I, LhsT, Rows, Depth>, Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMat<I, LhsT, Rows, Depth>, &MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMat<I, LhsT, Rows, Depth>, &MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseColMat<I, LhsT, Rows, Depth>, &Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Depth>, MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Depth>, MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Depth>, Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Depth>, &MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Depth>, &MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Depth>, &Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Depth>, MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Depth>, MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Depth>, Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Depth>, &MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Depth>, &MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Depth>, &Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMat<I, LhsT, Rows, Depth>, MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMat<I, LhsT, Rows, Depth>, MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMat<I, LhsT, Rows, Depth>, Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMat<I, LhsT, Rows, Depth>, &MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMat<I, LhsT, Rows, Depth>, &MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseColMat<I, LhsT, Rows, Depth>, &Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);

	impl_matmul_sparse!(MatRef<'_, LhsT, Rows, Depth>, SparseColMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(MatRef<'_, LhsT, Rows, Depth>, SparseColMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(MatRef<'_, LhsT, Rows, Depth>, &SparseColMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(MatRef<'_, LhsT, Rows, Depth>, &SparseColMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(MatRef<'_, LhsT, Rows, Depth>, &SparseColMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(MatMut<'_, LhsT, Rows, Depth>, SparseColMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(MatMut<'_, LhsT, Rows, Depth>, SparseColMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(MatMut<'_, LhsT, Rows, Depth>, SparseColMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(MatMut<'_, LhsT, Rows, Depth>, &SparseColMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(MatMut<'_, LhsT, Rows, Depth>, &SparseColMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(MatMut<'_, LhsT, Rows, Depth>, &SparseColMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(Mat<LhsT, Rows, Depth>, SparseColMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(Mat<LhsT, Rows, Depth>, SparseColMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(Mat<LhsT, Rows, Depth>, SparseColMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(Mat<LhsT, Rows, Depth>, &SparseColMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(Mat<LhsT, Rows, Depth>, &SparseColMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(Mat<LhsT, Rows, Depth>, &SparseColMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatRef<'_, LhsT, Rows, Depth>, SparseColMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatRef<'_, LhsT, Rows, Depth>, SparseColMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatRef<'_, LhsT, Rows, Depth>, SparseColMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatRef<'_, LhsT, Rows, Depth>, &SparseColMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatRef<'_, LhsT, Rows, Depth>, &SparseColMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatRef<'_, LhsT, Rows, Depth>, &SparseColMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatMut<'_, LhsT, Rows, Depth>, SparseColMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatMut<'_, LhsT, Rows, Depth>, SparseColMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatMut<'_, LhsT, Rows, Depth>, SparseColMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatMut<'_, LhsT, Rows, Depth>, &SparseColMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatMut<'_, LhsT, Rows, Depth>, &SparseColMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatMut<'_, LhsT, Rows, Depth>, &SparseColMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&Mat<LhsT, Rows, Depth>, SparseColMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&Mat<LhsT, Rows, Depth>, SparseColMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&Mat<LhsT, Rows, Depth>, SparseColMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&Mat<LhsT, Rows, Depth>, &SparseColMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&Mat<LhsT, Rows, Depth>, &SparseColMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&Mat<LhsT, Rows, Depth>, &SparseColMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);

	impl_matmul_1_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Cols>, ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Cols>, Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Cols>, &ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Cols>, &ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Cols>, &Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Cols>, ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Cols>, ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Cols>, Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Cols>, &ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Cols>, &ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Cols>, &Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseColMat<I, LhsT, Rows, Cols>, ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseColMat<I, LhsT, Rows, Cols>, ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseColMat<I, LhsT, Rows, Cols>, Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseColMat<I, LhsT, Rows, Cols>, &ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseColMat<I, LhsT, Rows, Cols>, &ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseColMat<I, LhsT, Rows, Cols>, &Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Cols>, ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Cols>, ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Cols>, Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Cols>, &ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Cols>, &ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Cols>, &Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Cols>, ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Cols>, ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Cols>, Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Cols>, &ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Cols>, &ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Cols>, &Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseColMat<I, LhsT, Rows, Cols>, ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseColMat<I, LhsT, Rows, Cols>, ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseColMat<I, LhsT, Rows, Cols>, Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseColMat<I, LhsT, Rows, Cols>, &ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseColMat<I, LhsT, Rows, Cols>, &ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseColMat<I, LhsT, Rows, Cols>, &Col<RhsT, Cols>, Col<T, Rows>);

	impl_matmul_1_sparse!(RowRef<'_, LhsT, Rows>, SparseColMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(RowRef<'_, LhsT, Rows>, SparseColMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(RowRef<'_, LhsT, Rows>, &SparseColMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(RowRef<'_, LhsT, Rows>, &SparseColMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(RowRef<'_, LhsT, Rows>, &SparseColMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(RowMut<'_, LhsT, Rows>, SparseColMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(RowMut<'_, LhsT, Rows>, SparseColMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(RowMut<'_, LhsT, Rows>, SparseColMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(RowMut<'_, LhsT, Rows>, &SparseColMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(RowMut<'_, LhsT, Rows>, &SparseColMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(RowMut<'_, LhsT, Rows>, &SparseColMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(Row<LhsT, Rows>, SparseColMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(Row<LhsT, Rows>, SparseColMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(Row<LhsT, Rows>, SparseColMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(Row<LhsT, Rows>, &SparseColMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(Row<LhsT, Rows>, &SparseColMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(Row<LhsT, Rows>, &SparseColMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowRef<'_, LhsT, Rows>, SparseColMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowRef<'_, LhsT, Rows>, SparseColMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowRef<'_, LhsT, Rows>, SparseColMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowRef<'_, LhsT, Rows>, &SparseColMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowRef<'_, LhsT, Rows>, &SparseColMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowRef<'_, LhsT, Rows>, &SparseColMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowMut<'_, LhsT, Rows>, SparseColMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowMut<'_, LhsT, Rows>, SparseColMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowMut<'_, LhsT, Rows>, SparseColMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowMut<'_, LhsT, Rows>, &SparseColMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowMut<'_, LhsT, Rows>, &SparseColMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowMut<'_, LhsT, Rows>, &SparseColMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&Row<LhsT, Rows>, SparseColMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&Row<LhsT, Rows>, SparseColMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&Row<LhsT, Rows>, SparseColMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&Row<LhsT, Rows>, &SparseColMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&Row<LhsT, Rows>, &SparseColMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&Row<LhsT, Rows>, &SparseColMat<I, RhsT, Rows, Cols>, Row<T, Cols>);

	impl_matmul_1_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Cols>, DiagMut<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Cols>, Diag<RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Cols>, &DiagRef<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Cols>, &DiagMut<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseColMatRef<'_, I, LhsT, Rows, Cols>, &Diag<RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Cols>, DiagRef<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Cols>, DiagMut<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Cols>, Diag<RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Cols>, &DiagRef<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Cols>, &DiagMut<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseColMatMut<'_, I, LhsT, Rows, Cols>, &Diag<RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseColMat<I, LhsT, Rows, Cols>, DiagRef<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseColMat<I, LhsT, Rows, Cols>, DiagMut<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseColMat<I, LhsT, Rows, Cols>, Diag<RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseColMat<I, LhsT, Rows, Cols>, &DiagRef<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseColMat<I, LhsT, Rows, Cols>, &DiagMut<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseColMat<I, LhsT, Rows, Cols>, &Diag<RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Cols>, DiagRef<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Cols>, DiagMut<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Cols>, Diag<RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Cols>, &DiagRef<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Cols>, &DiagMut<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseColMatRef<'_, I, LhsT, Rows, Cols>, &Diag<RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Cols>, DiagRef<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Cols>, DiagMut<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Cols>, Diag<RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Cols>, &DiagRef<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Cols>, &DiagMut<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseColMatMut<'_, I, LhsT, Rows, Cols>, &Diag<RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseColMat<I, LhsT, Rows, Cols>, DiagRef<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseColMat<I, LhsT, Rows, Cols>, DiagMut<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseColMat<I, LhsT, Rows, Cols>, Diag<RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseColMat<I, LhsT, Rows, Cols>, &DiagRef<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseColMat<I, LhsT, Rows, Cols>, &DiagMut<'_, RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseColMat<I, LhsT, Rows, Cols>, &Diag<RhsT, Cols>, SparseColMat<I, T, Rows, Cols>);

	impl_matmul_1_sparse!(DiagRef<'_, LhsT, Rows>, SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(DiagRef<'_, LhsT, Rows>, SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(DiagRef<'_, LhsT, Rows>, &SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(DiagRef<'_, LhsT, Rows>, &SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(DiagRef<'_, LhsT, Rows>, &SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(DiagMut<'_, LhsT, Rows>, SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(DiagMut<'_, LhsT, Rows>, SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(DiagMut<'_, LhsT, Rows>, SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(DiagMut<'_, LhsT, Rows>, &SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(DiagMut<'_, LhsT, Rows>, &SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(DiagMut<'_, LhsT, Rows>, &SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(Diag<LhsT, Rows>, SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(Diag<LhsT, Rows>, SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(Diag<LhsT, Rows>, SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(Diag<LhsT, Rows>, &SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(Diag<LhsT, Rows>, &SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(Diag<LhsT, Rows>, &SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagRef<'_, LhsT, Rows>, SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagRef<'_, LhsT, Rows>, SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagRef<'_, LhsT, Rows>, SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagRef<'_, LhsT, Rows>, &SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagRef<'_, LhsT, Rows>, &SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagRef<'_, LhsT, Rows>, &SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagMut<'_, LhsT, Rows>, SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagMut<'_, LhsT, Rows>, SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagMut<'_, LhsT, Rows>, SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagMut<'_, LhsT, Rows>, &SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagMut<'_, LhsT, Rows>, &SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagMut<'_, LhsT, Rows>, &SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&Diag<LhsT, Rows>, SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&Diag<LhsT, Rows>, SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&Diag<LhsT, Rows>, SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&Diag<LhsT, Rows>, &SparseColMatRef<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&Diag<LhsT, Rows>, &SparseColMatMut<'_, I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&Diag<LhsT, Rows>, &SparseColMat<I, RhsT, Rows, Cols>, SparseColMat<I, T, Rows, Cols>);

	impl_matmul_perm_sparse!(PermRef<'_, I, Rows>, SparseColMatMut<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(PermRef<'_, I, Rows>, SparseColMat<I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(PermRef<'_, I, Rows>, &SparseColMatRef<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(PermRef<'_, I, Rows>, &SparseColMatMut<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(PermRef<'_, I, Rows>, &SparseColMat<I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(Perm<I, Rows>, SparseColMatRef<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(Perm<I, Rows>, SparseColMatMut<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(Perm<I, Rows>, SparseColMat<I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(Perm<I, Rows>, &SparseColMatRef<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(Perm<I, Rows>, &SparseColMatMut<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(Perm<I, Rows>, &SparseColMat<I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&PermRef<'_, I, Rows>, SparseColMatRef<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&PermRef<'_, I, Rows>, SparseColMatMut<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&PermRef<'_, I, Rows>, SparseColMat<I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&PermRef<'_, I, Rows>, &SparseColMatRef<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&PermRef<'_, I, Rows>, &SparseColMatMut<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&PermRef<'_, I, Rows>, &SparseColMat<I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&Perm<I, Rows>, SparseColMatRef<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&Perm<I, Rows>, SparseColMatMut<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&Perm<I, Rows>, SparseColMat<I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&Perm<I, Rows>, &SparseColMatRef<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&Perm<I, Rows>, &SparseColMatMut<'_, I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&Perm<I, Rows>, &SparseColMat<I, T, Rows, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);

	impl_matmul_perm_sparse!(SparseColMatMut<'_, I, T, Rows, Cols>, PermRef<'_, I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(SparseColMat<I, T, Rows, Cols>, PermRef<'_, I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseColMatRef<'_, I, T, Rows, Cols>, PermRef<'_, I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseColMatMut<'_, I, T, Rows, Cols>, PermRef<'_, I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseColMat<I, T, Rows, Cols>, PermRef<'_, I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(SparseColMatRef<'_, I, T, Rows, Cols>, Perm<I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(SparseColMatMut<'_, I, T, Rows, Cols>, Perm<I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(SparseColMat<I, T, Rows, Cols>, Perm<I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseColMatRef<'_, I, T, Rows, Cols>, Perm<I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseColMatMut<'_, I, T, Rows, Cols>, Perm<I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseColMat<I, T, Rows, Cols>, Perm<I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(SparseColMatRef<'_, I, T, Rows, Cols>, &PermRef<'_, I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(SparseColMatMut<'_, I, T, Rows, Cols>, &PermRef<'_, I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(SparseColMat<I, T, Rows, Cols>, &PermRef<'_, I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseColMatRef<'_, I, T, Rows, Cols>, &PermRef<'_, I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseColMatMut<'_, I, T, Rows, Cols>, &PermRef<'_, I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseColMat<I, T, Rows, Cols>, &PermRef<'_, I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(SparseColMatRef<'_, I, T, Rows, Cols>, &Perm<I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(SparseColMatMut<'_, I, T, Rows, Cols>, &Perm<I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(SparseColMat<I, T, Rows, Cols>, &Perm<I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseColMatRef<'_, I, T, Rows, Cols>, &Perm<I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseColMatMut<'_, I, T, Rows, Cols>, &Perm<I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseColMat<I, T, Rows, Cols>, &Perm<I, Cols>, SparseColMat<I, T::Canonical, Rows, Cols>);

	impl_matmul_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Depth>, SparseRowMatMut<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Depth>, SparseRowMat<I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Depth>, &SparseRowMatRef<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Depth>, &SparseRowMatMut<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Depth>, &SparseRowMat<I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Depth>, SparseRowMatRef<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Depth>, SparseRowMatMut<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Depth>, SparseRowMat<I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Depth>, &SparseRowMatRef<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Depth>, &SparseRowMatMut<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Depth>, &SparseRowMat<I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMat<I, LhsT, Rows, Depth>, SparseRowMatRef<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMat<I, LhsT, Rows, Depth>, SparseRowMatMut<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMat<I, LhsT, Rows, Depth>, SparseRowMat<I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMat<I, LhsT, Rows, Depth>, &SparseRowMatRef<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMat<I, LhsT, Rows, Depth>, &SparseRowMatMut<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMat<I, LhsT, Rows, Depth>, &SparseRowMat<I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Depth>, SparseRowMatRef<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Depth>, SparseRowMatMut<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Depth>, SparseRowMat<I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Depth>, &SparseRowMatRef<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Depth>, &SparseRowMatMut<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Depth>, &SparseRowMat<I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Depth>, SparseRowMatRef<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Depth>, SparseRowMatMut<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Depth>, SparseRowMat<I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Depth>, &SparseRowMatRef<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Depth>, &SparseRowMatMut<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Depth>, &SparseRowMat<I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMat<I, LhsT, Rows, Depth>, SparseRowMatRef<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMat<I, LhsT, Rows, Depth>, SparseRowMatMut<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMat<I, LhsT, Rows, Depth>, SparseRowMat<I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMat<I, LhsT, Rows, Depth>, &SparseRowMatRef<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMat<I, LhsT, Rows, Depth>, &SparseRowMatMut<'_, I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMat<I, LhsT, Rows, Depth>, &SparseRowMat<I, RhsT, Depth, Cols>, SparseRowMat<I, T, Rows, Cols>);

	impl_matmul_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Depth>, MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Depth>, Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Depth>, &MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Depth>, &MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Depth>, &Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Depth>, MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Depth>, MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Depth>, Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Depth>, &MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Depth>, &MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Depth>, &Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMat<I, LhsT, Rows, Depth>, MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMat<I, LhsT, Rows, Depth>, MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMat<I, LhsT, Rows, Depth>, Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMat<I, LhsT, Rows, Depth>, &MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMat<I, LhsT, Rows, Depth>, &MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(SparseRowMat<I, LhsT, Rows, Depth>, &Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Depth>, MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Depth>, MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Depth>, Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Depth>, &MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Depth>, &MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Depth>, &Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Depth>, MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Depth>, MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Depth>, Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Depth>, &MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Depth>, &MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Depth>, &Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMat<I, LhsT, Rows, Depth>, MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMat<I, LhsT, Rows, Depth>, MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMat<I, LhsT, Rows, Depth>, Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMat<I, LhsT, Rows, Depth>, &MatRef<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMat<I, LhsT, Rows, Depth>, &MatMut<'_, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&SparseRowMat<I, LhsT, Rows, Depth>, &Mat<RhsT, Depth, Cols>, Mat<T, Rows, Cols>);

	impl_matmul_sparse!(MatRef<'_, LhsT, Rows, Depth>, SparseRowMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(MatRef<'_, LhsT, Rows, Depth>, SparseRowMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(MatRef<'_, LhsT, Rows, Depth>, &SparseRowMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(MatRef<'_, LhsT, Rows, Depth>, &SparseRowMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(MatRef<'_, LhsT, Rows, Depth>, &SparseRowMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(MatMut<'_, LhsT, Rows, Depth>, SparseRowMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(MatMut<'_, LhsT, Rows, Depth>, SparseRowMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(MatMut<'_, LhsT, Rows, Depth>, SparseRowMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(MatMut<'_, LhsT, Rows, Depth>, &SparseRowMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(MatMut<'_, LhsT, Rows, Depth>, &SparseRowMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(MatMut<'_, LhsT, Rows, Depth>, &SparseRowMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(Mat<LhsT, Rows, Depth>, SparseRowMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(Mat<LhsT, Rows, Depth>, SparseRowMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(Mat<LhsT, Rows, Depth>, SparseRowMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(Mat<LhsT, Rows, Depth>, &SparseRowMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(Mat<LhsT, Rows, Depth>, &SparseRowMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(Mat<LhsT, Rows, Depth>, &SparseRowMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatRef<'_, LhsT, Rows, Depth>, SparseRowMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatRef<'_, LhsT, Rows, Depth>, SparseRowMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatRef<'_, LhsT, Rows, Depth>, SparseRowMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatRef<'_, LhsT, Rows, Depth>, &SparseRowMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatRef<'_, LhsT, Rows, Depth>, &SparseRowMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatRef<'_, LhsT, Rows, Depth>, &SparseRowMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatMut<'_, LhsT, Rows, Depth>, SparseRowMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatMut<'_, LhsT, Rows, Depth>, SparseRowMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatMut<'_, LhsT, Rows, Depth>, SparseRowMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatMut<'_, LhsT, Rows, Depth>, &SparseRowMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatMut<'_, LhsT, Rows, Depth>, &SparseRowMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&MatMut<'_, LhsT, Rows, Depth>, &SparseRowMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&Mat<LhsT, Rows, Depth>, SparseRowMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&Mat<LhsT, Rows, Depth>, SparseRowMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&Mat<LhsT, Rows, Depth>, SparseRowMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&Mat<LhsT, Rows, Depth>, &SparseRowMatRef<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&Mat<LhsT, Rows, Depth>, &SparseRowMatMut<'_, I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);
	impl_matmul_sparse!(&Mat<LhsT, Rows, Depth>, &SparseRowMat<I, RhsT, Depth, Cols>, Mat<T, Rows, Cols>);

	impl_matmul_1_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Cols>, ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Cols>, Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Cols>, &ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Cols>, &ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Cols>, &Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Cols>, ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Cols>, ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Cols>, Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Cols>, &ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Cols>, &ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Cols>, &Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseRowMat<I, LhsT, Rows, Cols>, ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseRowMat<I, LhsT, Rows, Cols>, ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseRowMat<I, LhsT, Rows, Cols>, Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseRowMat<I, LhsT, Rows, Cols>, &ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseRowMat<I, LhsT, Rows, Cols>, &ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(SparseRowMat<I, LhsT, Rows, Cols>, &Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Cols>, ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Cols>, ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Cols>, Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Cols>, &ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Cols>, &ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Cols>, &Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Cols>, ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Cols>, ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Cols>, Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Cols>, &ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Cols>, &ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Cols>, &Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseRowMat<I, LhsT, Rows, Cols>, ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseRowMat<I, LhsT, Rows, Cols>, ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseRowMat<I, LhsT, Rows, Cols>, Col<RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseRowMat<I, LhsT, Rows, Cols>, &ColRef<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseRowMat<I, LhsT, Rows, Cols>, &ColMut<'_, RhsT, Cols>, Col<T, Rows>);
	impl_matmul_1_sparse!(&SparseRowMat<I, LhsT, Rows, Cols>, &Col<RhsT, Cols>, Col<T, Rows>);

	impl_matmul_1_sparse!(RowRef<'_, LhsT, Rows>, SparseRowMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(RowRef<'_, LhsT, Rows>, SparseRowMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(RowRef<'_, LhsT, Rows>, &SparseRowMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(RowRef<'_, LhsT, Rows>, &SparseRowMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(RowRef<'_, LhsT, Rows>, &SparseRowMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(RowMut<'_, LhsT, Rows>, SparseRowMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(RowMut<'_, LhsT, Rows>, SparseRowMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(RowMut<'_, LhsT, Rows>, SparseRowMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(RowMut<'_, LhsT, Rows>, &SparseRowMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(RowMut<'_, LhsT, Rows>, &SparseRowMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(RowMut<'_, LhsT, Rows>, &SparseRowMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(Row<LhsT, Rows>, SparseRowMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(Row<LhsT, Rows>, SparseRowMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(Row<LhsT, Rows>, SparseRowMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(Row<LhsT, Rows>, &SparseRowMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(Row<LhsT, Rows>, &SparseRowMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(Row<LhsT, Rows>, &SparseRowMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowRef<'_, LhsT, Rows>, SparseRowMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowRef<'_, LhsT, Rows>, SparseRowMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowRef<'_, LhsT, Rows>, SparseRowMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowRef<'_, LhsT, Rows>, &SparseRowMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowRef<'_, LhsT, Rows>, &SparseRowMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowRef<'_, LhsT, Rows>, &SparseRowMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowMut<'_, LhsT, Rows>, SparseRowMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowMut<'_, LhsT, Rows>, SparseRowMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowMut<'_, LhsT, Rows>, SparseRowMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowMut<'_, LhsT, Rows>, &SparseRowMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowMut<'_, LhsT, Rows>, &SparseRowMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&RowMut<'_, LhsT, Rows>, &SparseRowMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&Row<LhsT, Rows>, SparseRowMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&Row<LhsT, Rows>, SparseRowMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&Row<LhsT, Rows>, SparseRowMat<I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&Row<LhsT, Rows>, &SparseRowMatRef<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&Row<LhsT, Rows>, &SparseRowMatMut<'_, I, RhsT, Rows, Cols>, Row<T, Cols>);
	impl_matmul_1_sparse!(&Row<LhsT, Rows>, &SparseRowMat<I, RhsT, Rows, Cols>, Row<T, Cols>);

	impl_matmul_1_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Cols>, DiagMut<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Cols>, Diag<RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Cols>, &DiagRef<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Cols>, &DiagMut<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseRowMatRef<'_, I, LhsT, Rows, Cols>, &Diag<RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Cols>, DiagRef<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Cols>, DiagMut<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Cols>, Diag<RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Cols>, &DiagRef<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Cols>, &DiagMut<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseRowMatMut<'_, I, LhsT, Rows, Cols>, &Diag<RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseRowMat<I, LhsT, Rows, Cols>, DiagRef<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseRowMat<I, LhsT, Rows, Cols>, DiagMut<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseRowMat<I, LhsT, Rows, Cols>, Diag<RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseRowMat<I, LhsT, Rows, Cols>, &DiagRef<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseRowMat<I, LhsT, Rows, Cols>, &DiagMut<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(SparseRowMat<I, LhsT, Rows, Cols>, &Diag<RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Cols>, DiagRef<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Cols>, DiagMut<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Cols>, Diag<RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Cols>, &DiagRef<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Cols>, &DiagMut<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseRowMatRef<'_, I, LhsT, Rows, Cols>, &Diag<RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Cols>, DiagRef<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Cols>, DiagMut<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Cols>, Diag<RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Cols>, &DiagRef<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Cols>, &DiagMut<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseRowMatMut<'_, I, LhsT, Rows, Cols>, &Diag<RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseRowMat<I, LhsT, Rows, Cols>, DiagRef<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseRowMat<I, LhsT, Rows, Cols>, DiagMut<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseRowMat<I, LhsT, Rows, Cols>, Diag<RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseRowMat<I, LhsT, Rows, Cols>, &DiagRef<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseRowMat<I, LhsT, Rows, Cols>, &DiagMut<'_, RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&SparseRowMat<I, LhsT, Rows, Cols>, &Diag<RhsT, Cols>, SparseRowMat<I, T, Rows, Cols>);

	impl_matmul_1_sparse!(DiagRef<'_, LhsT, Rows>, SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(DiagRef<'_, LhsT, Rows>, SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(DiagRef<'_, LhsT, Rows>, &SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(DiagRef<'_, LhsT, Rows>, &SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(DiagRef<'_, LhsT, Rows>, &SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(DiagMut<'_, LhsT, Rows>, SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(DiagMut<'_, LhsT, Rows>, SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(DiagMut<'_, LhsT, Rows>, SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(DiagMut<'_, LhsT, Rows>, &SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(DiagMut<'_, LhsT, Rows>, &SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(DiagMut<'_, LhsT, Rows>, &SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(Diag<LhsT, Rows>, SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(Diag<LhsT, Rows>, SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(Diag<LhsT, Rows>, SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(Diag<LhsT, Rows>, &SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(Diag<LhsT, Rows>, &SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(Diag<LhsT, Rows>, &SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagRef<'_, LhsT, Rows>, SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagRef<'_, LhsT, Rows>, SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagRef<'_, LhsT, Rows>, SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagRef<'_, LhsT, Rows>, &SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagRef<'_, LhsT, Rows>, &SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagRef<'_, LhsT, Rows>, &SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagMut<'_, LhsT, Rows>, SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagMut<'_, LhsT, Rows>, SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagMut<'_, LhsT, Rows>, SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagMut<'_, LhsT, Rows>, &SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagMut<'_, LhsT, Rows>, &SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&DiagMut<'_, LhsT, Rows>, &SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&Diag<LhsT, Rows>, SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&Diag<LhsT, Rows>, SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&Diag<LhsT, Rows>, SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&Diag<LhsT, Rows>, &SparseRowMatRef<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&Diag<LhsT, Rows>, &SparseRowMatMut<'_, I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);
	impl_matmul_1_sparse!(&Diag<LhsT, Rows>, &SparseRowMat<I, RhsT, Rows, Cols>, SparseRowMat<I, T, Rows, Cols>);

	impl_matmul_perm_sparse!(PermRef<'_, I, Rows>, SparseRowMatMut<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(PermRef<'_, I, Rows>, SparseRowMat<I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(PermRef<'_, I, Rows>, &SparseRowMatRef<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(PermRef<'_, I, Rows>, &SparseRowMatMut<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(PermRef<'_, I, Rows>, &SparseRowMat<I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(Perm<I, Rows>, SparseRowMatRef<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(Perm<I, Rows>, SparseRowMatMut<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(Perm<I, Rows>, SparseRowMat<I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(Perm<I, Rows>, &SparseRowMatRef<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(Perm<I, Rows>, &SparseRowMatMut<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(Perm<I, Rows>, &SparseRowMat<I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&PermRef<'_, I, Rows>, SparseRowMatRef<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&PermRef<'_, I, Rows>, SparseRowMatMut<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&PermRef<'_, I, Rows>, SparseRowMat<I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&PermRef<'_, I, Rows>, &SparseRowMatRef<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&PermRef<'_, I, Rows>, &SparseRowMatMut<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&PermRef<'_, I, Rows>, &SparseRowMat<I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&Perm<I, Rows>, SparseRowMatRef<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&Perm<I, Rows>, SparseRowMatMut<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&Perm<I, Rows>, SparseRowMat<I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&Perm<I, Rows>, &SparseRowMatRef<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&Perm<I, Rows>, &SparseRowMatMut<'_, I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&Perm<I, Rows>, &SparseRowMat<I, T, Rows, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);

	impl_matmul_perm_sparse!(SparseRowMatMut<'_, I, T, Rows, Cols>, PermRef<'_, I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(SparseRowMat<I, T, Rows, Cols>, PermRef<'_, I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseRowMatRef<'_, I, T, Rows, Cols>, PermRef<'_, I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseRowMatMut<'_, I, T, Rows, Cols>, PermRef<'_, I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseRowMat<I, T, Rows, Cols>, PermRef<'_, I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(SparseRowMatRef<'_, I, T, Rows, Cols>, Perm<I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(SparseRowMatMut<'_, I, T, Rows, Cols>, Perm<I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(SparseRowMat<I, T, Rows, Cols>, Perm<I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseRowMatRef<'_, I, T, Rows, Cols>, Perm<I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseRowMatMut<'_, I, T, Rows, Cols>, Perm<I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseRowMat<I, T, Rows, Cols>, Perm<I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(SparseRowMatRef<'_, I, T, Rows, Cols>, &PermRef<'_, I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(SparseRowMatMut<'_, I, T, Rows, Cols>, &PermRef<'_, I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(SparseRowMat<I, T, Rows, Cols>, &PermRef<'_, I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseRowMatRef<'_, I, T, Rows, Cols>, &PermRef<'_, I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseRowMatMut<'_, I, T, Rows, Cols>, &PermRef<'_, I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseRowMat<I, T, Rows, Cols>, &PermRef<'_, I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(SparseRowMatRef<'_, I, T, Rows, Cols>, &Perm<I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(SparseRowMatMut<'_, I, T, Rows, Cols>, &Perm<I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(SparseRowMat<I, T, Rows, Cols>, &Perm<I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseRowMatRef<'_, I, T, Rows, Cols>, &Perm<I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseRowMatMut<'_, I, T, Rows, Cols>, &Perm<I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
	impl_matmul_perm_sparse!(&SparseRowMat<I, T, Rows, Cols>, &Perm<I, Cols>, SparseRowMat<I, T::Canonical, Rows, Cols>);
}

#[cfg(test)]
#[allow(non_snake_case)]
mod test {
	use crate::col::*;
	use crate::internal_prelude::*;
	use crate::mat::*;
	use crate::perm::*;
	use crate::row::*;
	use crate::{assert, mat};
	use assert_approx_eq::assert_approx_eq;

	fn matrices() -> (Mat<f64>, Mat<f64>) {
		let A = mat![[2.8, -3.3], [-1.7, 5.2], [4.6, -8.3],];

		let B = mat![[-7.9, 8.3], [4.7, -3.2], [3.8, -5.2],];
		(A, B)
	}

	#[test]
	#[should_panic]
	fn test_adding_matrices_of_different_sizes_should_panic() {
		let A = mat![[1.0, 2.0], [3.0, 4.0]];
		let B = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
		_ = A + B;
	}

	#[test]
	#[should_panic]
	fn test_subtracting_two_matrices_of_different_sizes_should_panic() {
		let A = mat![[1.0, 2.0], [3.0, 4.0]];
		let B = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
		_ = A - B;
	}

	#[test]
	fn test_add() {
		let (A, B) = matrices();

		let expected = mat![[-5.1, 5.0], [3.0, 2.0], [8.4, -13.5],];

		assert_matrix_approx_eq(A.as_ref() + B.as_ref(), &expected);
		assert_matrix_approx_eq(&A + &B, &expected);
		assert_matrix_approx_eq(A.as_ref() + &B, &expected);
		assert_matrix_approx_eq(&A + B.as_ref(), &expected);
		assert_matrix_approx_eq(A.as_ref() + B.clone(), &expected);
		assert_matrix_approx_eq(&A + B.clone(), &expected);
		assert_matrix_approx_eq(A.clone() + B.as_ref(), &expected);
		assert_matrix_approx_eq(A.clone() + &B, &expected);
		assert_matrix_approx_eq(A + B, &expected);
	}

	#[test]
	fn test_sub() {
		let (A, B) = matrices();

		let expected = mat![[10.7, -11.6], [-6.4, 8.4], [0.8, -3.1],];

		assert_matrix_approx_eq(A.as_ref() - B.as_ref(), &expected);
		assert_matrix_approx_eq(&A - &B, &expected);
		assert_matrix_approx_eq(A.as_ref() - &B, &expected);
		assert_matrix_approx_eq(&A - B.as_ref(), &expected);
		assert_matrix_approx_eq(A.as_ref() - B.clone(), &expected);
		assert_matrix_approx_eq(&A - B.clone(), &expected);
		assert_matrix_approx_eq(A.clone() - B.as_ref(), &expected);
		assert_matrix_approx_eq(A.clone() - &B, &expected);
		assert_matrix_approx_eq(A - B, &expected);
	}

	#[test]
	fn test_neg() {
		let (A, _) = matrices();

		let expected = mat![[-2.8, 3.3], [1.7, -5.2], [-4.6, 8.3],];

		assert!(-A == expected);
	}

	#[test]
	fn test_scalar_mul() {
		use crate::Scale as scale;

		let (A, _) = matrices();
		let k = 3.0;
		let expected = Mat::from_fn(A.nrows(), A.ncols(), |i, j| A.as_ref()[(i, j)] * k);

		{
			assert_matrix_approx_eq(A.as_ref() * scale(k), &expected);
			assert_matrix_approx_eq(&A * scale(k), &expected);
			assert_matrix_approx_eq(A.as_ref() * scale(k), &expected);
			assert_matrix_approx_eq(&A * scale(k), &expected);
			assert_matrix_approx_eq(A.as_ref() * scale(k), &expected);
			assert_matrix_approx_eq(&A * scale(k), &expected);
			assert_matrix_approx_eq(A.clone() * scale(k), &expected);
			assert_matrix_approx_eq(A.clone() * scale(k), &expected);
			assert_matrix_approx_eq(A * scale(k), &expected);
		}

		let (A, _) = matrices();
		{
			assert_matrix_approx_eq(scale(k) * A.as_ref(), &expected);
			assert_matrix_approx_eq(scale(k) * &A, &expected);
			assert_matrix_approx_eq(scale(k) * A.as_ref(), &expected);
			assert_matrix_approx_eq(scale(k) * &A, &expected);
			assert_matrix_approx_eq(scale(k) * A.as_ref(), &expected);
			assert_matrix_approx_eq(scale(k) * &A, &expected);
			assert_matrix_approx_eq(scale(k) * A.clone(), &expected);
			assert_matrix_approx_eq(scale(k) * A.clone(), &expected);
			assert_matrix_approx_eq(scale(k) * A, &expected);
		}
	}

	#[test]
	fn test_diag_mul() {
		let (A, _) = matrices();
		let diag_left = mat![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
		let diag_right = mat![[4.0, 0.0], [0.0, 5.0]];

		assert!(&diag_left * &A == diag_left.as_ref().diagonal() * &A);
		assert!(&A * &diag_right == &A * diag_right.as_ref().diagonal());
	}

	#[test]
	fn test_perm_mul() {
		let A = Mat::from_fn(6, 5, |i, j| (j + 5 * i) as f64);
		let pl = Perm::<usize>::new_checked(Box::new([5, 1, 4, 0, 2, 3]), Box::new([3, 1, 4, 5, 2, 0]), 6);
		let pr = Perm::<usize>::new_checked(Box::new([1, 4, 0, 2, 3]), Box::new([2, 0, 3, 4, 1]), 5);

		let perm_left = mat![
			[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
			[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
			[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
		];
		let perm_right = mat![
			[0.0, 1.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 1.0],
			[1.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 1.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 1.0, 0.0],
		];

		assert!(&pl * pl.as_ref().inverse() == PermRef::<'_, usize>::new_checked(&[0, 1, 2, 3, 4, 5], &[0, 1, 2, 3, 4, 5], 6));
		assert!(&perm_left * &A == &pl * &A);
		assert!(&A * &perm_right == &A * &pr);
	}

	#[test]
	fn test_matmul_col_row() {
		let A = Col::from_fn(6, |i| i as f64);
		let B = Row::from_fn(6, |j| (5 * j + 1) as f64);

		// outer product
		assert!(&A * &B == A.as_mat() * B.as_mat());
		// inner product
		assert!(&B * &A == (B.as_mat() * A.as_mat())[(0, 0)],);
	}

	fn assert_matrix_approx_eq(given: Mat<f64>, expected: &Mat<f64>) {
		assert_eq!(given.nrows(), expected.nrows());
		assert_eq!(given.ncols(), expected.ncols());
		for i in 0..given.nrows() {
			for j in 0..given.ncols() {
				assert_approx_eq!(given.as_ref()[(i, j)], expected.as_ref()[(i, j)]);
			}
		}
	}
}
