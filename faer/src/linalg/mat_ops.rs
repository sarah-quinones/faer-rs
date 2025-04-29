use crate::internal_prelude::*;
use crate::{Scale, assert, col, diag, get_global_parallelism, mat, perm, row};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

extern crate alloc;

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
				(lhs.transpose() - rhs.transpose()).into_transpose()
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

impl_binop!({
	impl<I: Index, N: Shape, L: (for<'a> Reborrow<'a, Target = perm::Ref<'a, I, N>>), R: (for<'a> Reborrow<'a, Target = perm::Ref<'a, I, N>>)>
		Mul<perm::generic::Perm<R>> for perm::generic::Perm<L>
	{
		type Output = Perm<I, N>;

		fn mul(self, rhs: _) {
			#[track_caller]
			fn imp<I: Index>(lhs: PermRef<'_, I>, rhs: PermRef<'_, I>) -> Perm<I> {
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

			let lhs = self.rb();
			let rhs = rhs.rb();
			let N = lhs.len();
			let n = N.unbound();
			imp(lhs.as_shape(n), rhs.as_shape(n)).into_shape(N)
		}
	}

	impl<
		I: Index,
		T: ComplexField,
		Rows: Shape,
		Cols: Shape,
		RT: (Conjugate<Canonical = T>),
		RRStride: Stride,
		RCStride: Stride,
		L: (for<'a> Reborrow<'a, Target = perm::Ref<'a, I, Rows>>),
		R: (for<'a> Reborrow<'a, Target = mat::Ref<'a, RT, Rows, Cols, RRStride, RCStride>>),
	> Mul<mat::generic::Mat<R>> for perm::generic::Perm<L>
	{
		type Output = Mat<T, Rows, Cols>;

		fn mul(self, rhs: _) {
			#[track_caller]
			fn imp<I: Index, T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: PermRef<'_, I>, rhs: MatRef<'_, RT>) -> Mat<T> {
				with_dim!(M, rhs.nrows());
				with_dim!(N, rhs.ncols());
				let rhs = rhs.as_shape(M, N);
				let lhs = lhs.as_shape(M);

				let mut out = Mat::zeros(rhs.nrows(), rhs.ncols());

				let fwd = lhs.bound_arrays().0;

				for j in rhs.ncols().indices() {
					for i in rhs.nrows().indices() {
						let fwd = fwd[i];
						let rhs = rhs.at(fwd.zx(), j);

						out[(i, j)] = Conj::apply(rhs);
					}
				}

				out.into_shape(*M, *N)
			}

			let lhs = self.rb();
			let rhs = rhs.rb();
			let m = rhs.nrows().unbound();
			imp(lhs.as_shape(m), rhs.as_dyn().as_dyn_stride()).into_shape(rhs.nrows(), rhs.ncols())
		}
	}

	impl<
		I: Index,
		T: ComplexField,
		Rows: Shape,
		RT: (Conjugate<Canonical = T>),
		RRStride: Stride,
		L: (for<'a> Reborrow<'a, Target = perm::Ref<'a, I, Rows>>),
		R: (for<'a> Reborrow<'a, Target = col::Ref<'a, RT, Rows, RRStride>>),
	> Mul<col::generic::Col<R>> for perm::generic::Perm<L>
	{
		type Output = Col<T, Rows>;

		fn mul(self, rhs: _) {
			#[track_caller]
			fn imp<I: Index, T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: PermRef<'_, I>, rhs: ColRef<'_, RT>) -> Col<T> {
				with_dim!(M, rhs.nrows());
				let rhs = rhs.as_row_shape(M);
				let lhs = lhs.as_shape(M);

				let mut out = Col::zeros(rhs.nrows());

				let fwd = lhs.bound_arrays().0;

				for i in rhs.nrows().indices() {
					let fwd = fwd[i];
					let rhs = rhs.at(fwd.zx());

					out[i] = Conj::apply(rhs);
				}

				out.into_row_shape(*M)
			}

			let lhs = self.rb();
			let rhs = rhs.rb();
			let m = rhs.nrows().unbound();
			imp(lhs.as_shape(m), rhs.as_dyn_rows().as_dyn_stride()).into_row_shape(rhs.nrows())
		}
	}

	impl<
		I: Index,
		T: ComplexField,
		Rows: Shape,
		Cols: Shape,
		LT: (Conjugate<Canonical = T>),
		LRStride: Stride,
		LCStride: Stride,
		L: (for<'a> Reborrow<'a, Target = mat::Ref<'a, LT, Rows, Cols, LRStride, LCStride>>),
		R: (for<'a> Reborrow<'a, Target = perm::Ref<'a, I, Cols>>),
	> Mul<perm::generic::Perm<R>> for mat::generic::Mat<L>
	{
		type Output = Mat<T, Rows, Cols>;

		fn mul(self, rhs: _) {
			#[track_caller]
			fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>>(lhs: MatRef<'_, LT>, rhs: PermRef<'_, I>) -> Mat<T> {
				with_dim!(M, lhs.nrows());
				with_dim!(N, lhs.ncols());
				let lhs = lhs.as_shape(M, N);
				let rhs = rhs.as_shape(N);

				let mut out = Mat::zeros(M, N);

				let inv = rhs.bound_arrays().1;

				for j in N.indices() {
					let inv = inv[j];
					for i in M.indices() {
						let lhs = lhs.at(i, inv.zx());

						out[(i, j)] = Conj::apply(lhs);
					}
				}

				out.into_shape(*M, *N)
			}

			let lhs = self.rb();
			let rhs = rhs.rb();
			let n = lhs.ncols().unbound();
			imp(lhs.as_dyn().as_dyn_stride(), rhs.as_shape(n)).into_shape(lhs.nrows(), lhs.ncols())
		}
	}

	impl<
		I: Index,
		T: ComplexField,
		Cols: Shape,
		LT: (Conjugate<Canonical = T>),
		LCStride: Stride,
		L: (for<'a> Reborrow<'a, Target = row::Ref<'a, LT, Cols, LCStride>>),
		R: (for<'a> Reborrow<'a, Target = perm::Ref<'a, I, Cols>>),
	> Mul<perm::generic::Perm<R>> for row::generic::Row<L>
	{
		type Output = Row<T, Cols>;

		fn mul(self, rhs: _) {
			#[track_caller]
			fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>>(lhs: RowRef<'_, LT>, rhs: PermRef<'_, I>) -> Row<T> {
				with_dim!(N, lhs.ncols());
				let lhs = lhs.as_col_shape(N);
				let rhs = rhs.as_shape(N);

				let mut out = Row::zeros(N);

				let inv = rhs.bound_arrays().1;

				for j in N.indices() {
					let inv = inv[j];
					let lhs = lhs.at(inv.zx());
					out[j] = Conj::apply(lhs);
				}

				out.into_col_shape(*N)
			}

			let lhs = self.rb();
			let rhs = rhs.rb();
			let n = lhs.ncols().unbound();
			imp(lhs.as_dyn_cols().as_dyn_stride(), rhs.as_shape(n)).into_col_shape(lhs.ncols())
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
		L: (for<'a> Reborrow<'a, Target = mat::Ref<'a, LT, Rows, Cols, LRStride, LCStride>>),
	> Mul<Scale<T>> for mat::generic::Mat<L>
	{
		type Output = Mat<T, Rows, Cols>;

		fn mul(self, rhs: _) {
			#[track_caller]
			#[math]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>>(lhs: MatRef<'_, LT>, rhs: &T) -> Mat<T> {
				zip!(lhs).map(|unzip!(x)| Conj::apply(x) * rhs)
			}
			let lhs = self.rb();
			let rhs = &rhs.0;

			imp(lhs.as_dyn().as_dyn_stride(), rhs).into_shape(lhs.nrows(), lhs.ncols())
		}
	}

	impl<
		T: ComplexField,
		Rows: Shape,
		Cols: Shape,
		LT: (Conjugate<Canonical = T>),
		LRStride: Stride,
		LCStride: Stride,
		L: (for<'a> Reborrow<'a, Target = mat::Ref<'a, LT, Rows, Cols, LRStride, LCStride>>),
	> Div<Scale<T>> for mat::generic::Mat<L>
	{
		type Output = Mat<T, Rows, Cols>;

		fn div(self, rhs: _) {
			#[track_caller]
			#[math]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>>(lhs: MatRef<'_, LT>, rhs: &T) -> Mat<T> {
				lhs * Scale(recip(*rhs))
			}
			let lhs = self.rb();
			let rhs = &rhs.0;

			imp(lhs.as_dyn().as_dyn_stride(), rhs).into_shape(lhs.nrows(), lhs.ncols())
		}
	}

	impl<
		T: ComplexField,
		Rows: Shape,
		Cols: Shape,
		RT: (Conjugate<Canonical = T>),
		RRStride: Stride,
		RCStride: Stride,
		R: (for<'a> Reborrow<'a, Target = mat::Ref<'a, RT, Rows, Cols, RRStride, RCStride>>),
	> Mul<mat::generic::Mat<R>> for Scale<T>
	{
		type Output = Mat<T, Rows, Cols>;

		fn mul(self, rhs: _) {
			#[track_caller]
			#[math]
			fn imp<T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: &T, rhs: MatRef<'_, RT>) -> Mat<T> {
				zip!(rhs).map(|unzip!(x)| *lhs * Conj::apply(x))
			}
			let lhs = &self.0;
			let rhs = rhs.rb();

			imp(lhs, rhs.as_dyn().as_dyn_stride()).into_shape(rhs.nrows(), rhs.ncols())
		}
	}
});

impl_binop!({
	impl<
		T: ComplexField,
		Rows: Shape,
		LT: (Conjugate<Canonical = T>),
		LRStride: Stride,
		L: (for<'a> Reborrow<'a, Target = col::Ref<'a, LT, Rows, LRStride>>),
	> Mul<Scale<T>> for col::generic::Col<L>
	{
		type Output = Col<T, Rows>;

		fn mul(self, rhs: _) {
			#[track_caller]
			#[math]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>>(lhs: ColRef<'_, LT>, rhs: &T) -> Col<T> {
				zip!(lhs).map(|unzip!(x)| Conj::apply(x) * rhs)
			}
			let lhs = self.rb();
			let rhs = &rhs.0;

			imp(lhs.as_dyn_rows().as_dyn_stride(), rhs).into_row_shape(lhs.nrows())
		}
	}

	impl<
		T: ComplexField,
		Rows: Shape,
		LT: (Conjugate<Canonical = T>),
		LRStride: Stride,
		L: (for<'a> Reborrow<'a, Target = col::Ref<'a, LT, Rows, LRStride>>),
	> Div<Scale<T>> for col::generic::Col<L>
	{
		type Output = Col<T, Rows>;

		fn div(self, rhs: _) {
			#[track_caller]
			#[math]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>>(lhs: ColRef<'_, LT>, rhs: &T) -> Col<T> {
				lhs * Scale(recip(*rhs))
			}
			let lhs = self.rb();
			let rhs = &rhs.0;

			imp(lhs.as_dyn_rows().as_dyn_stride(), rhs).into_row_shape(lhs.nrows())
		}
	}

	impl<
		T: ComplexField,
		Rows: Shape,
		RT: (Conjugate<Canonical = T>),
		RRStride: Stride,
		R: (for<'a> Reborrow<'a, Target = col::Ref<'a, RT, Rows, RRStride>>),
	> Mul<col::generic::Col<R>> for Scale<T>
	{
		type Output = Col<T, Rows>;

		fn mul(self, rhs: _) {
			#[track_caller]
			#[math]
			fn imp<T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: &T, rhs: ColRef<'_, RT>) -> Col<T> {
				zip!(rhs).map(|unzip!(x)| *lhs * Conj::apply(x))
			}
			let lhs = &self.0;
			let rhs = rhs.rb();

			imp(lhs, rhs.as_dyn_rows().as_dyn_stride()).into_row_shape(rhs.nrows())
		}
	}
});

impl_binop!({
	impl<
		T: ComplexField,
		Cols: Shape,
		LT: (Conjugate<Canonical = T>),
		LCStride: Stride,
		L: (for<'a> Reborrow<'a, Target = row::Ref<'a, LT, Cols, LCStride>>),
	> Mul<Scale<T>> for row::generic::Row<L>
	{
		type Output = Row<T, Cols>;

		fn mul(self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>>(lhs: RowRef<'_, LT>, rhs: &T) -> Row<T> {
				(lhs.transpose() * Scale::from_ref(rhs)).into_transpose()
			}
			let lhs = self.rb();
			let rhs = &rhs.0;

			imp(lhs.as_dyn_cols().as_dyn_stride(), rhs).into_col_shape(lhs.ncols())
		}
	}

	impl<
		T: ComplexField,
		Cols: Shape,
		LT: (Conjugate<Canonical = T>),
		LCStride: Stride,
		L: (for<'a> Reborrow<'a, Target = row::Ref<'a, LT, Cols, LCStride>>),
	> Div<Scale<T>> for row::generic::Row<L>
	{
		type Output = Row<T, Cols>;

		fn div(self, rhs: _) {
			#[track_caller]
			#[math]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>>(lhs: RowRef<'_, LT>, rhs: &T) -> Row<T> {
				lhs * Scale(recip(*rhs))
			}
			let lhs = self.rb();
			let rhs = &rhs.0;

			imp(lhs.as_dyn_cols().as_dyn_stride(), rhs).into_col_shape(lhs.ncols())
		}
	}

	impl<
		T: ComplexField,
		Cols: Shape,
		RT: (Conjugate<Canonical = T>),
		RCStride: Stride,
		R: (for<'a> Reborrow<'a, Target = row::Ref<'a, RT, Cols, RCStride>>),
	> Mul<row::generic::Row<R>> for Scale<T>
	{
		type Output = Row<T, Cols>;

		fn mul(self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: &T, rhs: RowRef<'_, RT>) -> Row<T> {
				(Scale::from_ref(lhs) * rhs.transpose()).into_transpose()
			}
			let lhs = &self.0;
			let rhs = rhs.rb();

			imp(lhs, rhs.as_dyn_cols().as_dyn_stride()).into_col_shape(rhs.ncols())
		}
	}
});

impl_binop!({
	impl<
		T: ComplexField,
		Dim: Shape,
		LT: (Conjugate<Canonical = T>),
		LStride: Stride,
		L: (for<'a> Reborrow<'a, Target = diag::Ref<'a, LT, Dim, LStride>>),
	> Mul<Scale<T>> for diag::generic::Diag<L>
	{
		type Output = Diag<T, Dim>;

		fn mul(self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>>(lhs: ColRef<'_, LT>, rhs: &T) -> Col<T> {
				lhs * Scale::from_ref(rhs)
			}
			let lhs = self.rb().column_vector();
			let rhs = &rhs.0;

			imp(lhs.as_dyn_rows().as_dyn_stride(), rhs).into_row_shape(lhs.nrows()).into_diagonal()
		}
	}

	impl<
		T: ComplexField,
		Dim: Shape,
		LT: (Conjugate<Canonical = T>),
		LStride: Stride,
		L: (for<'a> Reborrow<'a, Target = diag::Ref<'a, LT, Dim, LStride>>),
	> Div<Scale<T>> for diag::generic::Diag<L>
	{
		type Output = Diag<T, Dim>;

		fn div(self, rhs: _) {
			#[track_caller]
			#[math]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>>(lhs: ColRef<'_, LT>, rhs: &T) -> Col<T> {
				lhs * Scale(recip(*rhs))
			}
			let lhs = self.rb().column_vector();
			let rhs = &rhs.0;

			imp(lhs.as_dyn_rows().as_dyn_stride(), rhs).into_row_shape(lhs.nrows()).into_diagonal()
		}
	}

	impl<
		T: ComplexField,
		Dim: Shape,
		RT: (Conjugate<Canonical = T>),
		RStride: Stride,
		R: (for<'a> Reborrow<'a, Target = diag::Ref<'a, RT, Dim, RStride>>),
	> Mul<diag::generic::Diag<R>> for Scale<T>
	{
		type Output = Diag<T, Dim>;

		fn mul(self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: &T, rhs: ColRef<'_, RT>) -> Col<T> {
				Scale::from_ref(lhs) * rhs
			}
			let lhs = &self.0;
			let rhs = rhs.rb().column_vector();

			imp(lhs, rhs.as_dyn_rows().as_dyn_stride()).into_row_shape(rhs.nrows()).into_diagonal()
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
		L: (for<'a> Reborrow<'a, Target = mat::Ref<'a, LT, Rows, Cols, LRStride, LCStride>>),
	> Mul<f64> for mat::generic::Mat<L>
	{
		type Output = Mat<T, Rows, Cols>;

		fn mul(self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>>(lhs: MatRef<'_, LT>, rhs: &T) -> Mat<T> {
				lhs * Scale::from_ref(rhs)
			}
			let lhs = self.rb();
			let rhs = &from_f64::<T>(*rhs);

			imp(lhs.as_dyn().as_dyn_stride(), rhs).into_shape(lhs.nrows(), lhs.ncols())
		}
	}

	impl<
		T: ComplexField,
		Rows: Shape,
		Cols: Shape,
		LT: (Conjugate<Canonical = T>),
		LRStride: Stride,
		LCStride: Stride,
		L: (for<'a> Reborrow<'a, Target = mat::Ref<'a, LT, Rows, Cols, LRStride, LCStride>>),
	> Div<f64> for mat::generic::Mat<L>
	{
		type Output = Mat<T, Rows, Cols>;

		fn div(self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>>(lhs: MatRef<'_, LT>, rhs: &T) -> Mat<T> {
				lhs * Scale(recip(rhs))
			}
			let lhs = self.rb();
			let rhs = &from_f64::<T>(*rhs);

			imp(lhs.as_dyn().as_dyn_stride(), rhs).into_shape(lhs.nrows(), lhs.ncols())
		}
	}

	impl<
		T: ComplexField,
		Rows: Shape,
		Cols: Shape,
		RT: (Conjugate<Canonical = T>),
		RRStride: Stride,
		RCStride: Stride,
		R: (for<'a> Reborrow<'a, Target = mat::Ref<'a, RT, Rows, Cols, RRStride, RCStride>>),
	> Mul<mat::generic::Mat<R>> for f64
	{
		type Output = Mat<T, Rows, Cols>;

		fn mul(self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: &T, rhs: MatRef<'_, RT>) -> Mat<T> {
				Scale::from_ref(lhs) * rhs
			}
			let lhs = &from_f64::<T>(*self);
			let rhs = rhs.rb();

			imp(lhs, rhs.as_dyn().as_dyn_stride()).into_shape(rhs.nrows(), rhs.ncols())
		}
	}
});

impl_binop!({
	impl<
		T: ComplexField,
		Rows: Shape,
		LT: (Conjugate<Canonical = T>),
		LRStride: Stride,
		L: (for<'a> Reborrow<'a, Target = col::Ref<'a, LT, Rows, LRStride>>),
	> Mul<f64> for col::generic::Col<L>
	{
		type Output = Col<T, Rows>;

		fn mul(self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>>(lhs: ColRef<'_, LT>, rhs: &T) -> Col<T> {
				lhs * Scale::from_ref(rhs)
			}
			let lhs = self.rb();
			let rhs = &from_f64::<T>(*rhs);

			imp(lhs.as_dyn_rows().as_dyn_stride(), rhs).into_row_shape(lhs.nrows())
		}
	}

	impl<
		T: ComplexField,
		Rows: Shape,
		LT: (Conjugate<Canonical = T>),
		LRStride: Stride,
		L: (for<'a> Reborrow<'a, Target = col::Ref<'a, LT, Rows, LRStride>>),
	> Div<f64> for col::generic::Col<L>
	{
		type Output = Col<T, Rows>;

		fn div(self, rhs: _) {
			#[track_caller]
			#[math]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>>(lhs: ColRef<'_, LT>, rhs: &T) -> Col<T> {
				lhs * Scale(recip(*rhs))
			}
			let lhs = self.rb();
			let rhs = &from_f64::<T>(*rhs);

			imp(lhs.as_dyn_rows().as_dyn_stride(), rhs).into_row_shape(lhs.nrows())
		}
	}

	impl<
		T: ComplexField,
		Rows: Shape,
		RT: (Conjugate<Canonical = T>),
		RRStride: Stride,
		R: (for<'a> Reborrow<'a, Target = col::Ref<'a, RT, Rows, RRStride>>),
	> Mul<col::generic::Col<R>> for f64
	{
		type Output = Col<T, Rows>;

		fn mul(self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: &T, rhs: ColRef<'_, RT>) -> Col<T> {
				Scale::from_ref(lhs) * rhs
			}
			let lhs = &from_f64::<T>(*self);
			let rhs = rhs.rb();

			imp(lhs, rhs.as_dyn_rows().as_dyn_stride()).into_row_shape(rhs.nrows())
		}
	}
});

impl_binop!({
	impl<
		T: ComplexField,
		Cols: Shape,
		LT: (Conjugate<Canonical = T>),
		LCStride: Stride,
		L: (for<'a> Reborrow<'a, Target = row::Ref<'a, LT, Cols, LCStride>>),
	> Mul<f64> for row::generic::Row<L>
	{
		type Output = Row<T, Cols>;

		fn mul(self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>>(lhs: RowRef<'_, LT>, rhs: &T) -> Row<T> {
				(lhs.transpose() * Scale::from_ref(rhs)).into_transpose()
			}
			let lhs = self.rb();
			let rhs = &from_f64::<T>(*rhs);

			imp(lhs.as_dyn_cols().as_dyn_stride(), rhs).into_col_shape(lhs.ncols())
		}
	}

	impl<
		T: ComplexField,
		Cols: Shape,
		LT: (Conjugate<Canonical = T>),
		LCStride: Stride,
		L: (for<'a> Reborrow<'a, Target = row::Ref<'a, LT, Cols, LCStride>>),
	> Div<f64> for row::generic::Row<L>
	{
		type Output = Row<T, Cols>;

		fn div(self, rhs: _) {
			#[track_caller]
			#[math]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>>(lhs: RowRef<'_, LT>, rhs: &T) -> Row<T> {
				lhs * Scale(recip(*rhs))
			}
			let lhs = self.rb();
			let rhs = &from_f64::<T>(*rhs);

			imp(lhs.as_dyn_cols().as_dyn_stride(), rhs).into_col_shape(lhs.ncols())
		}
	}

	impl<
		T: ComplexField,
		Cols: Shape,
		RT: (Conjugate<Canonical = T>),
		RCStride: Stride,
		R: (for<'a> Reborrow<'a, Target = row::Ref<'a, RT, Cols, RCStride>>),
	> Mul<row::generic::Row<R>> for f64
	{
		type Output = Row<T, Cols>;

		fn mul(self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: &T, rhs: RowRef<'_, RT>) -> Row<T> {
				(Scale::from_ref(lhs) * rhs.transpose()).into_transpose()
			}
			let lhs = &from_f64::<T>(*self);
			let rhs = rhs.rb();

			imp(lhs, rhs.as_dyn_cols().as_dyn_stride()).into_col_shape(rhs.ncols())
		}
	}
});

impl_binop!({
	impl<
		T: ComplexField,
		Dim: Shape,
		LT: (Conjugate<Canonical = T>),
		LStride: Stride,
		L: (for<'a> Reborrow<'a, Target = diag::Ref<'a, LT, Dim, LStride>>),
	> Mul<f64> for diag::generic::Diag<L>
	{
		type Output = Diag<T, Dim>;

		fn mul(self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>>(lhs: ColRef<'_, LT>, rhs: &T) -> Col<T> {
				lhs * Scale::from_ref(rhs)
			}
			let lhs = self.rb().column_vector();
			let rhs = &from_f64::<T>(*rhs);

			imp(lhs.as_dyn_rows().as_dyn_stride(), rhs).into_row_shape(lhs.nrows()).into_diagonal()
		}
	}

	impl<
		T: ComplexField,
		Dim: Shape,
		LT: (Conjugate<Canonical = T>),
		LStride: Stride,
		L: (for<'a> Reborrow<'a, Target = diag::Ref<'a, LT, Dim, LStride>>),
	> Div<f64> for diag::generic::Diag<L>
	{
		type Output = Diag<T, Dim>;

		fn div(self, rhs: _) {
			#[track_caller]
			#[math]
			fn imp<T: ComplexField, LT: Conjugate<Canonical = T>>(lhs: ColRef<'_, LT>, rhs: &T) -> Col<T> {
				lhs * Scale(recip(*rhs))
			}
			let lhs = self.rb().column_vector();
			let rhs = &from_f64::<T>(*rhs);

			imp(lhs.as_dyn_rows().as_dyn_stride(), rhs).into_row_shape(lhs.nrows()).into_diagonal()
		}
	}

	impl<
		T: ComplexField,
		Dim: Shape,
		RT: (Conjugate<Canonical = T>),
		RStride: Stride,
		R: (for<'a> Reborrow<'a, Target = diag::Ref<'a, RT, Dim, RStride>>),
	> Mul<diag::generic::Diag<R>> for f64
	{
		type Output = Diag<T, Dim>;

		fn mul(self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: &T, rhs: ColRef<'_, RT>) -> Col<T> {
				Scale::from_ref(lhs) * rhs
			}
			let lhs = &from_f64::<T>(*self);
			let rhs = rhs.rb().column_vector();

			imp(lhs, rhs.as_dyn_rows().as_dyn_stride()).into_row_shape(rhs.nrows()).into_diagonal()
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
		L: (for<'a> ReborrowMut<'a, Target = mat::Mut<'a, T, Rows, Cols, LRStride, LCStride>>),
	> MulAssign<Scale<T>> for mat::generic::Mat<L>
	{
		fn mul_assign(&mut self, rhs: _) {
			#[track_caller]
			#[math]
			fn imp<T: ComplexField>(lhs: MatMut<'_, T>, rhs: &T) {
				zip!(lhs).for_each(|unzip!(x)| *x = *x * *rhs)
			}
			let lhs = self.rb_mut();
			let rhs = &rhs.0;

			imp(lhs.as_dyn_mut().as_dyn_stride_mut(), rhs)
		}
	}

	impl<
		T: ComplexField,
		Rows: Shape,
		Cols: Shape,
		LRStride: Stride,
		LCStride: Stride,
		L: (for<'a> ReborrowMut<'a, Target = mat::Mut<'a, T, Rows, Cols, LRStride, LCStride>>),
	> DivAssign<Scale<T>> for mat::generic::Mat<L>
	{
		fn div_assign(&mut self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField>(mut lhs: MatMut<'_, T>, rhs: &T) {
				lhs *= Scale(recip(rhs));
			}
			let lhs = self.rb_mut();
			let rhs = &rhs.0;

			imp(lhs.as_dyn_mut().as_dyn_stride_mut(), rhs)
		}
	}
});

impl_op_assign!({
	impl<T: ComplexField, Rows: Shape, LRStride: Stride, L: (for<'a> ReborrowMut<'a, Target = col::Mut<'a, T, Rows, LRStride>>)> MulAssign<Scale<T>>
		for col::generic::Col<L>
	{
		fn mul_assign(&mut self, rhs: _) {
			#[track_caller]
			#[math]
			fn imp<T: ComplexField>(lhs: ColMut<'_, T>, rhs: &T) {
				zip!(lhs).for_each(|unzip!(x)| *x = *x * *rhs)
			}
			let lhs = self.rb_mut();
			let rhs = &rhs.0;

			imp(lhs.as_dyn_rows_mut().as_dyn_stride_mut(), rhs)
		}
	}

	impl<T: ComplexField, Rows: Shape, LRStride: Stride, L: (for<'a> ReborrowMut<'a, Target = col::Mut<'a, T, Rows, LRStride>>)> DivAssign<Scale<T>>
		for col::generic::Col<L>
	{
		fn div_assign(&mut self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField>(mut lhs: ColMut<'_, T>, rhs: &T) {
				lhs *= Scale(recip(rhs));
			}
			let lhs = self.rb_mut();
			let rhs = &rhs.0;

			imp(lhs.as_dyn_rows_mut().as_dyn_stride_mut(), rhs)
		}
	}
});

impl_op_assign!({
	impl<T: ComplexField, Cols: Shape, LCStride: Stride, L: (for<'a> ReborrowMut<'a, Target = row::Mut<'a, T, Cols, LCStride>>)> MulAssign<Scale<T>>
		for row::generic::Row<L>
	{
		fn mul_assign(&mut self, rhs: _) {
			#[track_caller]
			#[math]
			fn imp<T: ComplexField>(lhs: RowMut<'_, T>, rhs: &T) {
				let mut lhs = lhs.transpose_mut();
				lhs *= Scale::from_ref(rhs);
			}
			let lhs = self.rb_mut();
			let rhs = &rhs.0;

			imp(lhs.as_dyn_cols_mut().as_dyn_stride_mut(), rhs)
		}
	}

	impl<T: ComplexField, Cols: Shape, LCStride: Stride, L: (for<'a> ReborrowMut<'a, Target = row::Mut<'a, T, Cols, LCStride>>)> DivAssign<Scale<T>>
		for row::generic::Row<L>
	{
		fn div_assign(&mut self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField>(mut lhs: RowMut<'_, T>, rhs: &T) {
				lhs *= Scale(recip(rhs));
			}
			let lhs = self.rb_mut();
			let rhs = &rhs.0;

			imp(lhs.as_dyn_cols_mut().as_dyn_stride_mut(), rhs)
		}
	}
});

impl_op_assign!({
	impl<T: ComplexField, Cols: Shape, LCStride: Stride, L: (for<'a> ReborrowMut<'a, Target = diag::Mut<'a, T, Cols, LCStride>>)> MulAssign<Scale<T>>
		for diag::generic::Diag<L>
	{
		fn mul_assign(&mut self, rhs: _) {
			#[track_caller]
			#[math]
			fn imp<T: ComplexField>(mut lhs: ColMut<'_, T>, rhs: &T) {
				lhs *= Scale::from_ref(rhs);
			}
			let lhs = self.rb_mut().column_vector_mut();
			let rhs = &rhs.0;

			imp(lhs.as_dyn_rows_mut().as_dyn_stride_mut(), rhs)
		}
	}

	impl<T: ComplexField, Cols: Shape, LCStride: Stride, L: (for<'a> ReborrowMut<'a, Target = diag::Mut<'a, T, Cols, LCStride>>)> DivAssign<Scale<T>>
		for diag::generic::Diag<L>
	{
		fn div_assign(&mut self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField>(mut lhs: ColMut<'_, T>, rhs: &T) {
				lhs *= Scale(recip(rhs));
			}
			let lhs = self.rb_mut().column_vector_mut();
			let rhs = &rhs.0;

			imp(lhs.as_dyn_rows_mut().as_dyn_stride_mut(), rhs)
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
		L: (for<'a> ReborrowMut<'a, Target = mat::Mut<'a, T, Rows, Cols, LRStride, LCStride>>),
	> MulAssign<f64> for mat::generic::Mat<L>
	{
		fn mul_assign(&mut self, rhs: _) {
			#[track_caller]
			#[math]
			fn imp<T: ComplexField>(lhs: MatMut<'_, T>, rhs: &T) {
				zip!(lhs).for_each(|unzip!(x)| *x = *x * *rhs)
			}
			let lhs = self.rb_mut();
			let rhs = &from_f64::<T>(*rhs);

			imp(lhs.as_dyn_mut().as_dyn_stride_mut(), rhs)
		}
	}

	impl<
		T: ComplexField,
		Rows: Shape,
		Cols: Shape,
		LRStride: Stride,
		LCStride: Stride,
		L: (for<'a> ReborrowMut<'a, Target = mat::Mut<'a, T, Rows, Cols, LRStride, LCStride>>),
	> DivAssign<f64> for mat::generic::Mat<L>
	{
		fn div_assign(&mut self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField>(mut lhs: MatMut<'_, T>, rhs: &T) {
				lhs *= Scale(recip(rhs));
			}
			let lhs = self.rb_mut();
			let rhs = &from_f64::<T>(*rhs);

			imp(lhs.as_dyn_mut().as_dyn_stride_mut(), rhs)
		}
	}
});

impl_op_assign!({
	impl<T: ComplexField, Rows: Shape, LRStride: Stride, L: (for<'a> ReborrowMut<'a, Target = col::Mut<'a, T, Rows, LRStride>>)> MulAssign<f64>
		for col::generic::Col<L>
	{
		fn mul_assign(&mut self, rhs: _) {
			#[track_caller]
			#[math]
			fn imp<T: ComplexField>(lhs: ColMut<'_, T>, rhs: &T) {
				zip!(lhs).for_each(|unzip!(x)| *x = *x * *rhs)
			}
			let lhs = self.rb_mut();
			let rhs = &from_f64::<T>(*rhs);

			imp(lhs.as_dyn_rows_mut().as_dyn_stride_mut(), rhs)
		}
	}

	impl<T: ComplexField, Rows: Shape, LRStride: Stride, L: (for<'a> ReborrowMut<'a, Target = col::Mut<'a, T, Rows, LRStride>>)> DivAssign<f64>
		for col::generic::Col<L>
	{
		fn div_assign(&mut self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField>(mut lhs: ColMut<'_, T>, rhs: &T) {
				lhs *= Scale(recip(rhs));
			}
			let lhs = self.rb_mut();
			let rhs = &from_f64::<T>(*rhs);

			imp(lhs.as_dyn_rows_mut().as_dyn_stride_mut(), rhs)
		}
	}
});

impl_op_assign!({
	impl<T: ComplexField, Cols: Shape, LCStride: Stride, L: (for<'a> ReborrowMut<'a, Target = row::Mut<'a, T, Cols, LCStride>>)> MulAssign<f64>
		for row::generic::Row<L>
	{
		fn mul_assign(&mut self, rhs: _) {
			#[track_caller]
			#[math]
			fn imp<T: ComplexField>(lhs: RowMut<'_, T>, rhs: &T) {
				let mut lhs = lhs.transpose_mut();
				lhs *= Scale::from_ref(rhs);
			}
			let lhs = self.rb_mut();
			let rhs = &from_f64::<T>(*rhs);

			imp(lhs.as_dyn_cols_mut().as_dyn_stride_mut(), rhs)
		}
	}

	impl<T: ComplexField, Cols: Shape, LCStride: Stride, L: (for<'a> ReborrowMut<'a, Target = row::Mut<'a, T, Cols, LCStride>>)> DivAssign<f64>
		for row::generic::Row<L>
	{
		fn div_assign(&mut self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField>(mut lhs: RowMut<'_, T>, rhs: &T) {
				lhs *= Scale(recip(rhs));
			}
			let lhs = self.rb_mut();
			let rhs = &from_f64::<T>(*rhs);

			imp(lhs.as_dyn_cols_mut().as_dyn_stride_mut(), rhs)
		}
	}
});

impl_op_assign!({
	impl<T: ComplexField, Cols: Shape, LCStride: Stride, L: (for<'a> ReborrowMut<'a, Target = diag::Mut<'a, T, Cols, LCStride>>)> MulAssign<f64>
		for diag::generic::Diag<L>
	{
		fn mul_assign(&mut self, rhs: _) {
			#[track_caller]
			#[math]
			fn imp<T: ComplexField>(mut lhs: ColMut<'_, T>, rhs: &T) {
				lhs *= Scale::from_ref(rhs);
			}
			let lhs = self.rb_mut().column_vector_mut();
			let rhs = &from_f64::<T>(*rhs);

			imp(lhs.as_dyn_rows_mut().as_dyn_stride_mut(), rhs)
		}
	}

	impl<T: ComplexField, Cols: Shape, LCStride: Stride, L: (for<'a> ReborrowMut<'a, Target = diag::Mut<'a, T, Cols, LCStride>>)> DivAssign<f64>
		for diag::generic::Diag<L>
	{
		fn div_assign(&mut self, rhs: _) {
			#[track_caller]
			fn imp<T: ComplexField>(mut lhs: ColMut<'_, T>, rhs: &T) {
				lhs *= Scale(recip(rhs));
			}
			let lhs = self.rb_mut().column_vector_mut();
			let rhs = &from_f64::<T>(*rhs);

			imp(lhs.as_dyn_rows_mut().as_dyn_stride_mut(), rhs)
		}
	}
});

#[cfg(feature = "sparse")]
mod sparse {
	use super::*;
	use crate::internal_prelude_sp::*;
	use {csc_numeric as csc, csr_numeric as csr};

	impl_binop!({
		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			RT: (Conjugate<Canonical = T>),
			R: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, RT, Rows, Cols>>),
		> Mul<csr::generic::SparseRowMat<R>> for Scale<T>
		{
			type Output = SparseRowMat<I, T, Rows, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: &T, rhs: SparseRowMatRef<'_, I, RT>) -> SparseRowMat<I, T> {
					(Scale::from_ref(lhs) * rhs.transpose()).into_transpose()
				}

				let lhs = &self.0;
				let rhs = rhs.rb();
				imp(lhs, rhs.as_dyn()).into_shape(rhs.nrows(), rhs.ncols())
			}
		}

		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			RT: (Conjugate<Canonical = T>),
			R: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, RT, Rows, Cols>>),
		> Mul<csc::generic::SparseColMat<R>> for Scale<T>
		{
			type Output = SparseColMat<I, T, Rows, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				#[math]
				fn imp<I: Index, T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: &T, rhs: SparseColMatRef<'_, I, RT>) -> SparseColMat<I, T> {
					with_dim!(M, rhs.nrows());
					with_dim!(N, rhs.ncols());

					let rhs = rhs.as_shape(M, N);

					let symbolic = rhs.symbolic().to_owned().unwrap();
					let mut val = alloc::vec::Vec::new();
					val.resize(symbolic.row_idx().len(), zero());

					for j in rhs.ncols().indices() {
						for (val, rhs) in iter::zip(&mut val[symbolic.col_range(j)], rhs.val_of_col(j)) {
							*val = *lhs * Conj::apply(rhs);
						}
					}

					SparseColMat::new(symbolic, val).into_shape(*M, *N)
				}

				let lhs = &self.0;
				let rhs = rhs.rb();
				imp(lhs, rhs.as_dyn()).into_shape(rhs.nrows(), rhs.ncols())
			}
		}
	});

	impl_binop!({
		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			LT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, LT, Rows, Cols>>),
		> Mul<Scale<T>> for csr::generic::SparseRowMat<L>
		{
			type Output = SparseRowMat<I, T, Rows, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>>(rhs: &T, lhs: SparseRowMatRef<'_, I, LT>) -> SparseRowMat<I, T> {
					(lhs.transpose() * Scale::from_ref(rhs)).into_transpose()
				}

				let rhs = &rhs.0;
				let lhs = self.rb();
				imp(rhs, lhs.as_dyn()).into_shape(lhs.nrows(), lhs.ncols())
			}
		}

		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			LT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, LT, Rows, Cols>>),
		> Mul<Scale<T>> for csc::generic::SparseColMat<L>
		{
			type Output = SparseColMat<I, T, Rows, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				#[math]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>>(rhs: &T, lhs: SparseColMatRef<'_, I, LT>) -> SparseColMat<I, T> {
					with_dim!(M, lhs.nrows());
					with_dim!(N, lhs.ncols());

					let lhs = lhs.as_shape(M, N);

					let symbolic = lhs.symbolic().to_owned().unwrap();
					let mut val = alloc::vec::Vec::new();
					val.resize(symbolic.row_idx().len(), zero());

					for j in lhs.ncols().indices() {
						for (val, lhs) in iter::zip(&mut val[symbolic.col_range(j)], lhs.val_of_col(j)) {
							*val = Conj::apply(lhs) * *rhs;
						}
					}

					SparseColMat::new(symbolic, val).into_shape(*M, *N)
				}

				let rhs = &rhs.0;
				let lhs = self.rb();
				imp(rhs, lhs.as_dyn()).into_shape(lhs.nrows(), lhs.ncols())
			}
		}

		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			LT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, LT, Rows, Cols>>),
		> Div<Scale<T>> for csr::generic::SparseRowMat<L>
		{
			type Output = SparseRowMat<I, T, Rows, Cols>;

			fn div(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>>(rhs: &T, lhs: SparseRowMatRef<'_, I, LT>) -> SparseRowMat<I, T> {
					lhs * Scale(recip(rhs))
				}

				let rhs = &rhs.0;
				let lhs = self.rb();
				imp(rhs, lhs.as_dyn()).into_shape(lhs.nrows(), lhs.ncols())
			}
		}

		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			LT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, LT, Rows, Cols>>),
		> Div<Scale<T>> for csc::generic::SparseColMat<L>
		{
			type Output = SparseColMat<I, T, Rows, Cols>;

			fn div(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>>(rhs: &T, lhs: SparseColMatRef<'_, I, LT>) -> SparseColMat<I, T> {
					lhs * Scale(recip(rhs))
				}

				let rhs = &rhs.0;
				let lhs = self.rb();
				imp(rhs, lhs.as_dyn()).into_shape(lhs.nrows(), lhs.ncols())
			}
		}
	});

	impl_op_assign!({
		impl<I: Index, T: ComplexField, Rows: Shape, Cols: Shape, L: (for<'a> ReborrowMut<'a, Target = csr::Mut<'a, I, T, Rows, Cols>>)>
			MulAssign<Scale<T>> for csr::generic::SparseRowMat<L>
		{
			fn mul_assign(&mut self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField>(rhs: &T, lhs: SparseRowMatMut<'_, I, T>) {
					let mut lhs = lhs.transpose_mut();
					lhs *= Scale::from_ref(rhs);
				}

				let rhs = &rhs.0;
				let lhs = self.rb_mut();
				imp(rhs, lhs.as_dyn_mut())
			}
		}

		impl<I: Index, T: ComplexField, Rows: Shape, Cols: Shape, L: (for<'a> ReborrowMut<'a, Target = csc::Mut<'a, I, T, Rows, Cols>>)>
			MulAssign<Scale<T>> for csc::generic::SparseColMat<L>
		{
			fn mul_assign(&mut self, rhs: _) {
				#[track_caller]
				#[math]
				fn imp<I: Index, T: ComplexField>(rhs: &T, lhs: SparseColMatMut<'_, I, T>) {
					with_dim!(M, lhs.nrows());
					with_dim!(N, lhs.ncols());

					let mut lhs = lhs.as_shape_mut(M, N);

					for j in lhs.ncols().indices() {
						for val in lhs.rb_mut().val_of_col_mut(j) {
							*val = *val * *rhs;
						}
					}
				}

				let rhs = &rhs.0;
				let lhs = self.rb_mut();
				imp(rhs, lhs.as_dyn_mut())
			}
		}

		impl<I: Index, T: ComplexField, Rows: Shape, Cols: Shape, L: (for<'a> ReborrowMut<'a, Target = csr::Mut<'a, I, T, Rows, Cols>>)>
			DivAssign<Scale<T>> for csr::generic::SparseRowMat<L>
		{
			fn div_assign(&mut self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField>(rhs: &T, lhs: SparseRowMatMut<'_, I, T>) {
					let mut lhs = lhs.transpose_mut();
					lhs /= Scale::from_ref(rhs);
				}

				let rhs = &rhs.0;
				let lhs = self.rb_mut();
				imp(rhs, lhs.as_dyn_mut())
			}
		}

		impl<I: Index, T: ComplexField, Rows: Shape, Cols: Shape, L: (for<'a> ReborrowMut<'a, Target = csc::Mut<'a, I, T, Rows, Cols>>)>
			DivAssign<Scale<T>> for csc::generic::SparseColMat<L>
		{
			fn div_assign(&mut self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField>(rhs: &T, lhs: SparseColMatMut<'_, I, T>) {
					let mut lhs = lhs.transpose_mut();
					lhs *= Scale(recip(rhs));
				}

				let rhs = &rhs.0;
				let lhs = self.rb_mut();
				imp(rhs, lhs.as_dyn_mut())
			}
		}
	});

	impl_binop!({
		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			RT: (Conjugate<Canonical = T>),
			R: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, RT, Rows, Cols>>),
		> Mul<csr::generic::SparseRowMat<R>> for f64
		{
			type Output = SparseRowMat<I, T, Rows, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: &T, rhs: SparseRowMatRef<'_, I, RT>) -> SparseRowMat<I, T> {
					(Scale::from_ref(lhs) * rhs.transpose()).into_transpose()
				}

				let lhs = &from_f64::<T>(*self);
				let rhs = rhs.rb();
				imp(lhs, rhs.as_dyn()).into_shape(rhs.nrows(), rhs.ncols())
			}
		}

		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			RT: (Conjugate<Canonical = T>),
			R: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, RT, Rows, Cols>>),
		> Mul<csc::generic::SparseColMat<R>> for f64
		{
			type Output = SparseColMat<I, T, Rows, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: &T, rhs: SparseColMatRef<'_, I, RT>) -> SparseColMat<I, T> {
					Scale::from_ref(lhs) * rhs
				}

				let lhs = &from_f64::<T>(*self);
				let rhs = rhs.rb();
				imp(lhs, rhs.as_dyn()).into_shape(rhs.nrows(), rhs.ncols())
			}
		}
	});

	impl_binop!({
		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			LT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, LT, Rows, Cols>>),
		> Mul<f64> for csr::generic::SparseRowMat<L>
		{
			type Output = SparseRowMat<I, T, Rows, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>>(rhs: &T, lhs: SparseRowMatRef<'_, I, LT>) -> SparseRowMat<I, T> {
					(lhs.transpose() * Scale::from_ref(rhs)).into_transpose()
				}

				let rhs = &from_f64::<T>(*rhs);
				let lhs = self.rb();
				imp(rhs, lhs.as_dyn()).into_shape(lhs.nrows(), lhs.ncols())
			}
		}

		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			LT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, LT, Rows, Cols>>),
		> Mul<f64> for csc::generic::SparseColMat<L>
		{
			type Output = SparseColMat<I, T, Rows, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>>(rhs: &T, lhs: SparseColMatRef<'_, I, LT>) -> SparseColMat<I, T> {
					lhs * Scale::from_ref(rhs)
				}

				let rhs = &from_f64::<T>(*rhs);
				let lhs = self.rb();
				imp(rhs, lhs.as_dyn()).into_shape(lhs.nrows(), lhs.ncols())
			}
		}

		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			LT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, LT, Rows, Cols>>),
		> Div<f64> for csr::generic::SparseRowMat<L>
		{
			type Output = SparseRowMat<I, T, Rows, Cols>;

			fn div(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>>(rhs: &T, lhs: SparseRowMatRef<'_, I, LT>) -> SparseRowMat<I, T> {
					lhs * Scale(recip(rhs))
				}

				let rhs = &from_f64::<T>(*rhs);
				let lhs = self.rb();
				imp(rhs, lhs.as_dyn()).into_shape(lhs.nrows(), lhs.ncols())
			}
		}

		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			LT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, LT, Rows, Cols>>),
		> Div<f64> for csc::generic::SparseColMat<L>
		{
			type Output = SparseColMat<I, T, Rows, Cols>;

			fn div(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>>(rhs: &T, lhs: SparseColMatRef<'_, I, LT>) -> SparseColMat<I, T> {
					lhs * Scale(recip(rhs))
				}

				let rhs = &from_f64::<T>(*rhs);
				let lhs = self.rb();
				imp(rhs, lhs.as_dyn()).into_shape(lhs.nrows(), lhs.ncols())
			}
		}
	});

	impl_op_assign!({
		impl<I: Index, T: ComplexField, Rows: Shape, Cols: Shape, L: (for<'a> ReborrowMut<'a, Target = csr::Mut<'a, I, T, Rows, Cols>>)>
			MulAssign<f64> for csr::generic::SparseRowMat<L>
		{
			fn mul_assign(&mut self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField>(rhs: &T, lhs: SparseRowMatMut<'_, I, T>) {
					let mut lhs = lhs.transpose_mut();
					lhs *= Scale::from_ref(rhs);
				}

				let rhs = &from_f64::<T>(*rhs);
				let lhs = self.rb_mut();
				imp(rhs, lhs.as_dyn_mut())
			}
		}

		impl<I: Index, T: ComplexField, Rows: Shape, Cols: Shape, L: (for<'a> ReborrowMut<'a, Target = csc::Mut<'a, I, T, Rows, Cols>>)>
			MulAssign<f64> for csc::generic::SparseColMat<L>
		{
			fn mul_assign(&mut self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField>(rhs: &T, lhs: SparseColMatMut<'_, I, T>) {
					*&mut { lhs } *= Scale::from_ref(rhs)
				}

				let rhs = &from_f64::<T>(*rhs);
				let lhs = self.rb_mut();
				imp(rhs, lhs.as_dyn_mut())
			}
		}

		impl<I: Index, T: ComplexField, Rows: Shape, Cols: Shape, L: (for<'a> ReborrowMut<'a, Target = csr::Mut<'a, I, T, Rows, Cols>>)>
			DivAssign<f64> for csr::generic::SparseRowMat<L>
		{
			fn div_assign(&mut self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField>(rhs: &T, lhs: SparseRowMatMut<'_, I, T>) {
					let mut lhs = lhs.transpose_mut();
					lhs /= Scale::from_ref(rhs);
				}

				let rhs = &from_f64::<T>(*rhs);
				let lhs = self.rb_mut();
				imp(rhs, lhs.as_dyn_mut())
			}
		}

		impl<I: Index, T: ComplexField, Rows: Shape, Cols: Shape, L: (for<'a> ReborrowMut<'a, Target = csc::Mut<'a, I, T, Rows, Cols>>)>
			DivAssign<f64> for csc::generic::SparseColMat<L>
		{
			fn div_assign(&mut self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField>(rhs: &T, lhs: SparseColMatMut<'_, I, T>) {
					let mut lhs = lhs.transpose_mut();
					lhs *= Scale(recip(rhs));
				}

				let rhs = &from_f64::<T>(*rhs);
				let lhs = self.rb_mut();
				imp(rhs, lhs.as_dyn_mut())
			}
		}
	});

	impl_op!({
		impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape, Inner: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, T, Rows, Cols>>)> Neg
			for csr::generic::SparseRowMat<Inner>
		{
			type Output = SparseRowMat<I, T::Canonical, Rows, Cols>;

			fn neg(self) {
				self.rb().transpose().neg().into_transpose()
			}
		}

		impl<I: Index, T: Conjugate, Rows: Shape, Cols: Shape, Inner: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, T, Rows, Cols>>)> Neg
			for csc::generic::SparseColMat<Inner>
		{
			type Output = SparseColMat<I, T::Canonical, Rows, Cols>;

			fn neg(self) {
				#[math]
				#[track_caller]
				fn imp<I: Index, T: Conjugate>(A: SparseColMatRef<'_, I, T>) -> SparseColMat<I, T::Canonical> {
					with_dim!(M, A.nrows());
					with_dim!(N, A.ncols());
					let A = A.as_shape(M, N);

					let symbolic = A.symbolic().to_owned().unwrap();
					let mut val = alloc::vec::Vec::new();
					val.resize(symbolic.row_idx().len(), zero());

					for j in A.ncols().indices() {
						for (val, lhs) in iter::zip(&mut val[symbolic.col_range(j)], A.val_of_col(j)) {
							*val = -Conj::apply(lhs);
						}
					}

					SparseColMat::new(symbolic, val).into_shape(*M, *N)
				}

				let A = self.rb();
				imp(A.as_dyn()).into_shape(A.nrows(), A.ncols())
			}
		}
	});

	impl_op_assign!({
		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			RT: (Conjugate<Canonical = T>),
			L: (for<'a> ReborrowMut<'a, Target = csc::Mut<'a, I, T, Rows, Cols>>),
			R: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, RT, Rows, Cols>>),
		> AddAssign<csc::generic::SparseColMat<R>> for csc::generic::SparseColMat<L>
		{
			fn add_assign(&mut self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: SparseColMatMut<'_, I, T>, rhs: SparseColMatRef<'_, I, RT>) {
					crate::sparse::ops::binary_op_assign_into(lhs, rhs, add_assign_fn::<T, RT>)
				}

				let lhs = self.rb_mut();
				let rhs = rhs.rb();
				imp(lhs.as_dyn_mut(), rhs.as_dyn())
			}
		}

		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			RT: (Conjugate<Canonical = T>),
			L: (for<'a> ReborrowMut<'a, Target = csc::Mut<'a, I, T, Rows, Cols>>),
			R: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, RT, Rows, Cols>>),
		> SubAssign<csc::generic::SparseColMat<R>> for csc::generic::SparseColMat<L>
		{
			fn sub_assign(&mut self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: SparseColMatMut<'_, I, T>, rhs: SparseColMatRef<'_, I, RT>) {
					crate::sparse::ops::binary_op_assign_into(lhs, rhs, sub_assign_fn::<T, RT>)
				}

				let lhs = self.rb_mut();
				let rhs = rhs.rb();
				imp(lhs.as_dyn_mut(), rhs.as_dyn())
			}
		}
	});

	impl_op_assign!({
		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			RT: (Conjugate<Canonical = T>),
			L: (for<'a> ReborrowMut<'a, Target = csr::Mut<'a, I, T, Rows, Cols>>),
			R: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, RT, Rows, Cols>>),
		> AddAssign<csr::generic::SparseRowMat<R>> for csr::generic::SparseRowMat<L>
		{
			fn add_assign(&mut self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: SparseRowMatMut<'_, I, T>, rhs: SparseRowMatRef<'_, I, RT>) {
					crate::sparse::ops::binary_op_assign_into(lhs.transpose_mut(), rhs.transpose(), add_assign_fn::<T, RT>)
				}

				let lhs = self.rb_mut();
				let rhs = rhs.rb();
				imp(lhs.as_dyn_mut(), rhs.as_dyn())
			}
		}

		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			RT: (Conjugate<Canonical = T>),
			L: (for<'a> ReborrowMut<'a, Target = csr::Mut<'a, I, T, Rows, Cols>>),
			R: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, RT, Rows, Cols>>),
		> SubAssign<csr::generic::SparseRowMat<R>> for csr::generic::SparseRowMat<L>
		{
			fn sub_assign(&mut self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, RT: Conjugate<Canonical = T>>(lhs: SparseRowMatMut<'_, I, T>, rhs: SparseRowMatRef<'_, I, RT>) {
					crate::sparse::ops::binary_op_assign_into(lhs.transpose_mut(), rhs.transpose(), sub_assign_fn::<T, RT>)
				}

				let lhs = self.rb_mut();
				let rhs = rhs.rb();
				imp(lhs.as_dyn_mut(), rhs.as_dyn())
			}
		}
	});

	impl_binop!({
		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			LT: (Conjugate<Canonical = T>),
			RT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, LT, Rows, Cols>>),
			R: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, RT, Rows, Cols>>),
		> Add<csc::generic::SparseColMat<R>> for csc::generic::SparseColMat<L>
		{
			type Output = SparseColMat<I, T, Rows, Cols>;

			fn add(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: SparseColMatRef<'_, I, LT>,
					rhs: SparseColMatRef<'_, I, RT>,
				) -> SparseColMat<I, T> {
					crate::sparse::ops::binary_op(lhs, rhs, add_fn::<T, LT, RT>).unwrap()
				}

				let lhs = self.rb();
				let rhs = rhs.rb();
				let (nrows, ncols) = lhs.shape();
				imp(lhs.as_dyn(), rhs.as_dyn()).into_shape(nrows, ncols)
			}
		}

		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			LT: (Conjugate<Canonical = T>),
			RT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, LT, Rows, Cols>>),
			R: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, RT, Rows, Cols>>),
		> Sub<csc::generic::SparseColMat<R>> for csc::generic::SparseColMat<L>
		{
			type Output = SparseColMat<I, T, Rows, Cols>;

			fn sub(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: SparseColMatRef<'_, I, LT>,
					rhs: SparseColMatRef<'_, I, RT>,
				) -> SparseColMat<I, T> {
					crate::sparse::ops::binary_op(lhs, rhs, sub_fn::<T, LT, RT>).unwrap()
				}

				let lhs = self.rb();
				let rhs = rhs.rb();
				let (nrows, ncols) = lhs.shape();
				imp(lhs.as_dyn(), rhs.as_dyn()).into_shape(nrows, ncols)
			}
		}
	});

	impl_binop!({
		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			LT: (Conjugate<Canonical = T>),
			RT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, LT, Rows, Cols>>),
			R: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, RT, Rows, Cols>>),
		> Add<csr::generic::SparseRowMat<R>> for csr::generic::SparseRowMat<L>
		{
			type Output = SparseRowMat<I, T, Rows, Cols>;

			fn add(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: SparseRowMatRef<'_, I, LT>,
					rhs: SparseRowMatRef<'_, I, RT>,
				) -> SparseRowMat<I, T> {
					crate::sparse::ops::binary_op(lhs.transpose(), rhs.transpose(), add_fn::<T, LT, RT>)
						.unwrap()
						.into_transpose()
				}

				let lhs = self.rb();
				let rhs = rhs.rb();
				let (nrows, ncols) = lhs.shape();
				imp(lhs.as_dyn(), rhs.as_dyn()).into_shape(nrows, ncols)
			}
		}

		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			LT: (Conjugate<Canonical = T>),
			RT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, LT, Rows, Cols>>),
			R: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, RT, Rows, Cols>>),
		> Sub<csr::generic::SparseRowMat<R>> for csr::generic::SparseRowMat<L>
		{
			type Output = SparseRowMat<I, T, Rows, Cols>;

			fn sub(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: SparseRowMatRef<'_, I, LT>,
					rhs: SparseRowMatRef<'_, I, RT>,
				) -> SparseRowMat<I, T> {
					crate::sparse::ops::binary_op(lhs.transpose(), rhs.transpose(), sub_fn::<T, LT, RT>)
						.unwrap()
						.into_transpose()
				}

				let lhs = self.rb();
				let rhs = rhs.rb();
				let (nrows, ncols) = lhs.shape();
				imp(lhs.as_dyn(), rhs.as_dyn()).into_shape(nrows, ncols)
			}
		}
	});

	impl_binop!({
		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			Depth: Shape,
			LT: (Conjugate<Canonical = T>),
			RT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, LT, Rows, Depth>>),
			R: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, RT, Depth, Cols>>),
		> Mul<csc::generic::SparseColMat<R>> for csc::generic::SparseColMat<L>
		{
			type Output = SparseColMat<I, T, Rows, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: SparseColMatRef<'_, I, LT>,
					rhs: SparseColMatRef<'_, I, RT>,
				) -> SparseColMat<I, T> {
					let nrows = lhs.nrows();
					let ncols = rhs.ncols();
					linalg_sp::matmul::sparse_sparse_matmul(lhs, rhs, one::<T>(), crate::get_global_parallelism())
						.unwrap()
						.into_shape(nrows, ncols)
				}

				let lhs = self.rb();
				let rhs = rhs.rb();
				let nrows = lhs.nrows();
				let ncols = rhs.ncols();
				imp(lhs.as_dyn(), rhs.as_dyn()).into_shape(nrows, ncols)
			}
		}

		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			Depth: Shape,
			LT: (Conjugate<Canonical = T>),
			RT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, LT, Rows, Depth>>),
			R: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, RT, Depth, Cols>>),
		> Mul<csr::generic::SparseRowMat<R>> for csr::generic::SparseRowMat<L>
		{
			type Output = SparseRowMat<I, T, Rows, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: SparseRowMatRef<'_, I, LT>,
					rhs: SparseRowMatRef<'_, I, RT>,
				) -> SparseRowMat<I, T> {
					(rhs.transpose() * lhs.transpose()).into_transpose()
				}

				let lhs = self.rb();
				let rhs = rhs.rb();
				let nrows = lhs.nrows();
				let ncols = rhs.ncols();
				imp(lhs.as_dyn(), rhs.as_dyn()).into_shape(nrows, ncols)
			}
		}
	});

	impl_binop!({
		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			Depth: Shape,
			LT: (Conjugate<Canonical = T>),
			RT: (Conjugate<Canonical = T>),
			RRStride: Stride,
			RCStride: Stride,
			L: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, LT, Rows, Depth>>),
			R: (for<'a> Reborrow<'a, Target = mat::Ref<'a, RT, Depth, Cols, RRStride, RCStride>>),
		> Mul<mat::generic::Mat<R>> for csc::generic::SparseColMat<L>
		{
			type Output = Mat<T, Rows, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: SparseColMatRef<'_, I, LT>,
					rhs: MatRef<'_, RT>,
				) -> Mat<T> {
					let nrows = lhs.nrows();
					let ncols = rhs.ncols();
					let mut out = Mat::zeros(nrows, ncols);
					linalg_sp::matmul::sparse_dense_matmul(out.rb_mut(), Accum::Add, lhs, rhs, T::one_impl(), crate::get_global_parallelism());
					out
				}

				let lhs = self.rb();
				let rhs = rhs.rb();
				let nrows = lhs.nrows();
				let ncols = rhs.ncols();
				imp(lhs.as_dyn(), rhs.as_dyn().as_dyn_stride()).into_shape(nrows, ncols)
			}
		}

		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			Depth: Shape,
			LT: (Conjugate<Canonical = T>),
			RT: (Conjugate<Canonical = T>),
			RRStride: Stride,
			RCStride: Stride,
			L: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, LT, Rows, Depth>>),
			R: (for<'a> Reborrow<'a, Target = mat::Ref<'a, RT, Depth, Cols, RRStride, RCStride>>),
		> Mul<mat::generic::Mat<R>> for csr::generic::SparseRowMat<L>
		{
			type Output = Mat<T, Rows, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: SparseRowMatRef<'_, I, LT>,
					rhs: MatRef<'_, RT>,
				) -> Mat<T> {
					let nrows = lhs.nrows();
					let ncols = rhs.ncols();
					let mut out = Mat::zeros(nrows, ncols);
					linalg_sp::matmul::dense_sparse_matmul(
						out.rb_mut().transpose_mut(),
						Accum::Add,
						rhs.transpose(),
						lhs.transpose(),
						T::one_impl(),
						crate::get_global_parallelism(),
					);
					out
				}

				let lhs = self.rb();
				let rhs = rhs.rb();
				let nrows = lhs.nrows();
				let ncols = rhs.ncols();
				imp(lhs.as_dyn(), rhs.as_dyn().as_dyn_stride()).into_shape(nrows, ncols)
			}
		}
	});

	impl_binop!({
		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Depth: Shape,
			LT: (Conjugate<Canonical = T>),
			RT: (Conjugate<Canonical = T>),
			RRStride: Stride,
			L: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, LT, Rows, Depth>>),
			R: (for<'a> Reborrow<'a, Target = col::Ref<'a, RT, Depth, RRStride>>),
		> Mul<col::generic::Col<R>> for csc::generic::SparseColMat<L>
		{
			type Output = Col<T, Rows>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: SparseColMatRef<'_, I, LT>,
					rhs: ColRef<'_, RT>,
				) -> Col<T> {
					let nrows = lhs.nrows();
					let mut out = Col::zeros(nrows);
					linalg_sp::matmul::sparse_dense_matmul(
						out.rb_mut().as_mat_mut(),
						Accum::Add,
						lhs,
						rhs.as_mat(),
						T::one_impl(),
						crate::get_global_parallelism(),
					);
					out
				}

				let lhs = self.rb();
				let rhs = rhs.rb();
				let nrows = lhs.nrows();
				imp(lhs.as_dyn(), rhs.as_dyn_rows().as_dyn_stride()).into_row_shape(nrows)
			}
		}

		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Depth: Shape,
			LT: (Conjugate<Canonical = T>),
			RT: (Conjugate<Canonical = T>),
			RRStride: Stride,
			L: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, LT, Rows, Depth>>),
			R: (for<'a> Reborrow<'a, Target = col::Ref<'a, RT, Depth, RRStride>>),
		> Mul<col::generic::Col<R>> for csr::generic::SparseRowMat<L>
		{
			type Output = Col<T, Rows>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: SparseRowMatRef<'_, I, LT>,
					rhs: ColRef<'_, RT>,
				) -> Col<T> {
					let nrows = lhs.nrows();
					let mut out = Col::zeros(nrows);
					linalg_sp::matmul::dense_sparse_matmul(
						out.rb_mut().transpose_mut().as_mat_mut(),
						Accum::Add,
						rhs.transpose().as_mat(),
						lhs.transpose(),
						T::one_impl(),
						crate::get_global_parallelism(),
					);
					out
				}

				let lhs = self.rb();
				let rhs = rhs.rb();
				let nrows = lhs.nrows();
				imp(lhs.as_dyn(), rhs.as_dyn_rows().as_dyn_stride()).into_row_shape(nrows)
			}
		}
	});

	impl_binop!({
		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Dim: Shape,
			LT: (Conjugate<Canonical = T>),
			RT: (Conjugate<Canonical = T>),
			RStride: Stride,
			L: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, LT, Rows, Dim>>),
			R: (for<'a> Reborrow<'a, Target = diag::Ref<'a, RT, Dim, RStride>>),
		> Mul<diag::generic::Diag<R>> for csc::generic::SparseColMat<L>
		{
			type Output = SparseColMat<I, T, Rows, Dim>;

			fn mul(self, rhs: _) {
				#[track_caller]
				#[math]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: SparseColMatRef<'_, I, LT>,
					rhs: DiagRef<'_, RT>,
				) -> SparseColMat<I, T> {
					with_dim!(M, lhs.nrows());
					with_dim!(N, lhs.ncols());

					let lhs = lhs.as_shape(M, N);
					let rhs = rhs.as_shape(N);

					let symbolic = lhs.symbolic().to_owned().unwrap();
					let mut out = alloc::vec::Vec::new();
					out.resize(symbolic.row_idx().len(), T::zero_impl());

					for j in lhs.ncols().indices() {
						let rhs = Conj::apply(&rhs[j]);
						for (out, lhs) in iter::zip(&mut out[symbolic.col_range(j)], lhs.val_of_col(j)) {
							*out = Conj::apply(lhs) * rhs;
						}
					}

					SparseColMat::new(symbolic, out).into_shape(*M, *N)
				}

				let lhs = self.rb();
				let rhs = rhs.rb().column_vector();
				let (nrows, ncols) = lhs.shape();
				imp(lhs.as_dyn(), rhs.as_dyn_rows().as_dyn_stride().as_diagonal()).into_shape(nrows, ncols)
			}
		}

		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Dim: Shape,
			LT: (Conjugate<Canonical = T>),
			RT: (Conjugate<Canonical = T>),
			RStride: Stride,
			L: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, LT, Rows, Dim>>),
			R: (for<'a> Reborrow<'a, Target = diag::Ref<'a, RT, Dim, RStride>>),
		> Mul<diag::generic::Diag<R>> for csr::generic::SparseRowMat<L>
		{
			type Output = SparseRowMat<I, T, Rows, Dim>;

			fn mul(self, rhs: _) {
				#[track_caller]
				#[math]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: SparseRowMatRef<'_, I, LT>,
					rhs: DiagRef<'_, RT>,
				) -> SparseRowMat<I, T> {
					with_dim!(M, lhs.nrows());
					with_dim!(N, lhs.ncols());

					let lhs = lhs.as_shape(M, N);
					let rhs = rhs.as_shape(N);

					let symbolic = lhs.symbolic().to_owned().unwrap();
					let mut out = alloc::vec::Vec::new();
					out.resize(symbolic.col_idx().len(), T::zero_impl());

					for i in lhs.nrows().indices() {
						for ((j, out), lhs) in iter::zip(iter::zip(symbolic.col_idx_of_row(i), &mut out[symbolic.row_range(i)]), lhs.val_of_row(i)) {
							*out = Conj::apply(lhs) * Conj::apply(&rhs[j]);
						}
					}

					SparseRowMat::new(symbolic, out).into_shape(*M, *N)
				}

				let lhs = self.rb();
				let rhs = rhs.rb().column_vector();
				let (nrows, ncols) = lhs.shape();
				imp(lhs.as_dyn(), rhs.as_dyn_rows().as_dyn_stride().as_diagonal()).into_shape(nrows, ncols)
			}
		}
	});

	impl_binop!({
		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Dim: Shape,
			LT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, LT, Rows, Dim>>),
			R: (for<'a> Reborrow<'a, Target = perm::Ref<'a, I, Dim>>),
		> Mul<perm::generic::Perm<R>> for csc::generic::SparseColMat<L>
		{
			type Output = SparseColMat<I, T, Rows, Dim>;

			fn mul(self, rhs: _) {
				#[track_caller]
				#[math]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>>(
					lhs: SparseColMatRef<'_, I, LT>,
					rhs: PermRef<'_, I>,
				) -> SparseColMat<I, T> {
					with_dim!(M, lhs.nrows());
					with_dim!(N, lhs.ncols());

					let lhs = lhs.as_shape(M, N);
					let rhs = rhs.as_shape(N);

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
					.into_shape(*M, *N)
				}

				let lhs = self.rb();
				let rhs = rhs.rb();
				let (nrows, ncols) = lhs.shape();
				imp(lhs.as_dyn(), rhs.as_shape(ncols.unbound())).into_shape(nrows, ncols)
			}
		}

		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Dim: Shape,
			LT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, LT, Rows, Dim>>),
			R: (for<'a> Reborrow<'a, Target = perm::Ref<'a, I, Dim>>),
		> Mul<perm::generic::Perm<R>> for csr::generic::SparseRowMat<L>
		{
			type Output = SparseRowMat<I, T, Rows, Dim>;

			fn mul(self, rhs: _) {
				#[track_caller]
				#[math]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>>(
					lhs: SparseRowMatRef<'_, I, LT>,
					rhs: PermRef<'_, I>,
				) -> SparseRowMat<I, T> {
					with_dim!(M, lhs.nrows());
					with_dim!(N, lhs.ncols());

					let lhs = lhs.as_shape(M, N);
					let rhs = rhs.as_shape(N);

					let symbolic = lhs.symbolic();

					let mut out_row_ptr = alloc::vec::Vec::new();
					let mut out_col_idx = alloc::vec::Vec::new();
					let mut out = alloc::vec::Vec::new();

					out_row_ptr.resize(symbolic.row_ptr().len(), I::truncate(0));
					out_col_idx.resize(symbolic.row_ptr().len(), I::truncate(0));
					out.resize(symbolic.col_idx().len(), T::Canonical::zero_impl());

					let inv = rhs.bound_arrays().0;

					let mut pos = 0usize;
					for i in lhs.nrows().indices() {
						let col_idx = lhs.col_idx_of_row_raw(i);
						let len = col_idx.len();

						for ((out_j, out_v), (lhs_j, lhs_v)) in iter::zip(
							iter::zip(&mut out_col_idx[pos..][..len], &mut out[pos..][..len]),
							iter::zip(lhs.col_idx_of_row(i), lhs.val_of_row(i)),
						) {
							*out_j = *inv[lhs_j];
							*out_v = Conj::apply(lhs_v);
						}

						pos += col_idx.len();
					}

					out_col_idx.truncate(pos);
					out.truncate(pos);

					SparseRowMat::new(
						unsafe { SymbolicSparseRowMat::new_unchecked(symbolic.nrows(), symbolic.ncols(), out_row_ptr, None, out_col_idx) },
						out,
					)
					.into_shape(*M, *N)
				}

				let lhs = self.rb();
				let rhs = rhs.rb();
				let (nrows, ncols) = lhs.shape();
				imp(lhs.as_dyn(), rhs.as_shape(ncols.unbound())).into_shape(nrows, ncols)
			}
		}
	});

	impl_binop!({
		impl<
			I: Index,
			T: ComplexField,
			Cols: Shape,
			Dim: Shape,
			RT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = perm::Ref<'a, I, Dim>>),
			R: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, RT, Dim, Cols>>),
		> Mul<csc::generic::SparseColMat<R>> for perm::generic::Perm<L>
		{
			type Output = SparseColMat<I, T, Dim, Cols>;

			fn mul(self, rhs: _) {
				(rhs.rb().transpose() * self.rb().inverse()).into_transpose()
			}
		}

		impl<
			I: Index,
			T: ComplexField,
			Cols: Shape,
			Dim: Shape,
			RT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = perm::Ref<'a, I, Dim>>),
			R: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, RT, Dim, Cols>>),
		> Mul<csr::generic::SparseRowMat<R>> for perm::generic::Perm<L>
		{
			type Output = SparseRowMat<I, T, Dim, Cols>;

			fn mul(self, rhs: _) {
				(rhs.rb().transpose() * self.rb().inverse()).into_transpose()
			}
		}
	});

	impl_binop!({
		impl<
			I: Index,
			T: ComplexField,
			Cols: Shape,
			Dim: Shape,
			LT: (Conjugate<Canonical = T>),
			RT: (Conjugate<Canonical = T>),
			LStride: Stride,
			L: (for<'a> Reborrow<'a, Target = diag::Ref<'a, LT, Dim, LStride>>),
			R: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, RT, Dim, Cols>>),
		> Mul<csc::generic::SparseColMat<R>> for diag::generic::Diag<L>
		{
			type Output = SparseColMat<I, T, Dim, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: DiagRef<'_, LT>,
					rhs: SparseColMatRef<'_, I, RT>,
				) -> SparseColMat<I, T> {
					(rhs.transpose() * lhs).into_transpose()
				}

				let lhs = self.rb().column_vector();
				let rhs = rhs.rb();
				let (nrows, ncols) = rhs.shape();
				imp(lhs.as_dyn_rows().as_dyn_stride().as_diagonal(), rhs.as_dyn()).into_shape(nrows, ncols)
			}
		}

		impl<
			I: Index,
			T: ComplexField,
			Cols: Shape,
			Dim: Shape,
			LT: (Conjugate<Canonical = T>),
			RT: (Conjugate<Canonical = T>),
			LStride: Stride,
			L: (for<'a> Reborrow<'a, Target = diag::Ref<'a, LT, Dim, LStride>>),
			R: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, RT, Dim, Cols>>),
		> Mul<csr::generic::SparseRowMat<R>> for diag::generic::Diag<L>
		{
			type Output = SparseRowMat<I, T, Dim, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: DiagRef<'_, LT>,
					rhs: SparseRowMatRef<'_, I, RT>,
				) -> SparseRowMat<I, T> {
					(rhs.transpose() * lhs).into_transpose()
				}

				let lhs = self.rb().column_vector();
				let rhs = rhs.rb();
				let (nrows, ncols) = rhs.shape();
				imp(lhs.as_dyn_rows().as_dyn_stride().as_diagonal(), rhs.as_dyn()).into_shape(nrows, ncols)
			}
		}
	});

	impl_binop!({
		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			Depth: Shape,
			LT: (Conjugate<Canonical = T>),
			LRStride: Stride,
			LCStride: Stride,
			RT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = mat::Ref<'a, LT, Rows, Depth, LRStride, LCStride>>),
			R: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, RT, Depth, Cols>>),
		> Mul<csc::generic::SparseColMat<R>> for mat::generic::Mat<L>
		{
			type Output = Mat<T, Rows, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: MatRef<'_, LT>,
					rhs: SparseColMatRef<'_, I, RT>,
				) -> Mat<T> {
					let nrows = lhs.nrows();
					let ncols = rhs.ncols();
					let mut out = Mat::zeros(nrows, ncols);
					linalg_sp::matmul::dense_sparse_matmul(out.rb_mut(), Accum::Add, lhs, rhs, T::one_impl(), crate::get_global_parallelism());
					out
				}

				let lhs = self.rb();
				let rhs = rhs.rb();
				let nrows = lhs.nrows();
				let ncols = rhs.ncols();
				imp(lhs.as_dyn().as_dyn_stride(), rhs.as_dyn()).into_shape(nrows, ncols)
			}
		}

		impl<
			I: Index,
			T: ComplexField,
			Rows: Shape,
			Cols: Shape,
			Depth: Shape,
			LT: (Conjugate<Canonical = T>),
			LRStride: Stride,
			LCStride: Stride,
			RT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = mat::Ref<'a, LT, Rows, Depth, LRStride, LCStride>>),
			R: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, RT, Depth, Cols>>),
		> Mul<csr::generic::SparseRowMat<R>> for mat::generic::Mat<L>
		{
			type Output = Mat<T, Rows, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: MatRef<'_, LT>,
					rhs: SparseRowMatRef<'_, I, RT>,
				) -> Mat<T> {
					let nrows = lhs.nrows();
					let ncols = rhs.ncols();
					let mut out = Mat::zeros(nrows, ncols);
					linalg_sp::matmul::sparse_dense_matmul(
						out.rb_mut().transpose_mut(),
						Accum::Add,
						rhs.transpose(),
						lhs.transpose(),
						T::one_impl(),
						crate::get_global_parallelism(),
					);
					out
				}

				let lhs = self.rb();
				let rhs = rhs.rb();
				let nrows = lhs.nrows();
				let ncols = rhs.ncols();
				imp(lhs.as_dyn().as_dyn_stride(), rhs.as_dyn()).into_shape(nrows, ncols)
			}
		}
	});

	impl_binop!({
		impl<
			I: Index,
			T: ComplexField,
			Cols: Shape,
			Depth: Shape,
			LT: (Conjugate<Canonical = T>),
			LCStride: Stride,
			RT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = row::Ref<'a, LT, Depth, LCStride>>),
			R: (for<'a> Reborrow<'a, Target = csc::Ref<'a, I, RT, Depth, Cols>>),
		> Mul<csc::generic::SparseColMat<R>> for row::generic::Row<L>
		{
			type Output = Row<T, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: RowRef<'_, LT>,
					rhs: SparseColMatRef<'_, I, RT>,
				) -> Row<T> {
					let ncols = rhs.ncols();
					let mut out = Row::zeros(ncols);
					linalg_sp::matmul::dense_sparse_matmul(
						out.rb_mut().as_mat_mut(),
						Accum::Add,
						lhs.as_mat(),
						rhs,
						T::one_impl(),
						crate::get_global_parallelism(),
					);
					out
				}

				let lhs = self.rb();
				let rhs = rhs.rb();
				let ncols = rhs.ncols();
				imp(lhs.as_dyn_cols().as_dyn_stride(), rhs.as_dyn()).into_col_shape(ncols)
			}
		}

		impl<
			I: Index,
			T: ComplexField,
			Cols: Shape,
			Depth: Shape,
			LT: (Conjugate<Canonical = T>),
			LCStride: Stride,
			RT: (Conjugate<Canonical = T>),
			L: (for<'a> Reborrow<'a, Target = row::Ref<'a, LT, Depth, LCStride>>),
			R: (for<'a> Reborrow<'a, Target = csr::Ref<'a, I, RT, Depth, Cols>>),
		> Mul<csr::generic::SparseRowMat<R>> for row::generic::Row<L>
		{
			type Output = Row<T, Cols>;

			fn mul(self, rhs: _) {
				#[track_caller]
				fn imp<I: Index, T: ComplexField, LT: Conjugate<Canonical = T>, RT: Conjugate<Canonical = T>>(
					lhs: RowRef<'_, LT>,
					rhs: SparseRowMatRef<'_, I, RT>,
				) -> Row<T> {
					let ncols = rhs.ncols();
					let mut out = Row::zeros(ncols);
					linalg_sp::matmul::sparse_dense_matmul(
						out.rb_mut().transpose_mut().as_mat_mut(),
						Accum::Add,
						rhs.transpose(),
						lhs.transpose().as_mat(),
						T::one_impl(),
						crate::get_global_parallelism(),
					);
					out
				}

				let lhs = self.rb();
				let rhs = rhs.rb();
				let ncols = rhs.ncols();
				imp(lhs.as_dyn_cols().as_dyn_stride(), rhs.as_dyn()).into_col_shape(ncols)
			}
		}
	});

	#[math]
	fn add_fn<T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>>(lhs: Option<&LhsT>, rhs: Option<&RhsT>) -> T {
		lhs.map(Conj::apply).unwrap_or_else(zero::<T>) + rhs.map(Conj::apply).unwrap_or_else(zero::<T>)
	}

	#[math]
	fn sub_fn<T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>>(lhs: Option<&LhsT>, rhs: Option<&RhsT>) -> T {
		lhs.map(Conj::apply).unwrap_or_else(zero::<T>) - rhs.map(Conj::apply).unwrap_or_else(zero::<T>)
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

	fn rows() -> (Row<f64>, Row<f64>) {
		(row![2.8, -3.3], row![-7.9, 8.3])
	}

	fn cols() -> (Col<f64>, Col<f64>) {
		(col![2.8, -3.3], col![-7.9, 8.3])
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
	fn test_matrix_add() {
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
	fn test_row_add() {
		let (A, B) = rows();
		let expected = row![-5.1, 5.0];

		assert_row_approx_eq(A.as_ref() + B.as_ref(), &expected);
		assert_row_approx_eq(&A + &B, &expected);
		assert_row_approx_eq(A.as_ref() + &B, &expected);
		assert_row_approx_eq(&A + B.as_ref(), &expected);
		assert_row_approx_eq(A.as_ref() + B.clone(), &expected);
		assert_row_approx_eq(A.clone() + B.as_ref(), &expected);
		assert_row_approx_eq(&A + B.clone(), &expected);
		assert_row_approx_eq(A.clone() + &B, &expected);
		assert_row_approx_eq(A + B, &expected);
	}

	#[test]
	fn test_col_add() {
		let (A, B) = cols();
		let expected = col![-5.1, 5.0];

		assert_col_approx_eq(A.as_ref() + B.as_ref(), &expected);
		assert_col_approx_eq(&A + &B, &expected);
		assert_col_approx_eq(A.as_ref() + &B, &expected);
		assert_col_approx_eq(&A + B.as_ref(), &expected);
		assert_col_approx_eq(A.as_ref() + B.clone(), &expected);
		assert_col_approx_eq(A.clone() + B.as_ref(), &expected);
		assert_col_approx_eq(&A + B.clone(), &expected);
		assert_col_approx_eq(A.clone() + &B, &expected);
		assert_col_approx_eq(A + B, &expected);
	}

	#[test]
	fn test_matrix_sub() {
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
	fn test_row_sub() {
		let (A, B) = rows();
		let expected = row![10.7, -11.6];

		assert_row_approx_eq(A.as_ref() - B.as_ref(), &expected);
		assert_row_approx_eq(&A - &B, &expected);
		assert_row_approx_eq(A.as_ref() - &B, &expected);
		assert_row_approx_eq(&A - B.as_ref(), &expected);
		assert_row_approx_eq(A.as_ref() - B.clone(), &expected);
		assert_row_approx_eq(A.clone() - B.as_ref(), &expected);
		assert_row_approx_eq(&A - B.clone(), &expected);
		assert_row_approx_eq(A.clone() - &B, &expected);
		assert_row_approx_eq(A - B, &expected);
	}

	#[test]
	fn test_col_sub() {
		let (A, B) = cols();
		let expected = col![10.7, -11.6];

		assert_col_approx_eq(A.as_ref() - B.as_ref(), &expected);
		assert_col_approx_eq(&A - &B, &expected);
		assert_col_approx_eq(A.as_ref() - &B, &expected);
		assert_col_approx_eq(&A - B.as_ref(), &expected);
		assert_col_approx_eq(A.as_ref() - B.clone(), &expected);
		assert_col_approx_eq(A.clone() - B.as_ref(), &expected);
		assert_col_approx_eq(&A - B.clone(), &expected);
		assert_col_approx_eq(A.clone() - &B, &expected);
		assert_col_approx_eq(A - B, &expected);
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

	fn assert_row_approx_eq(given: Row<f64>, expected: &Row<f64>) {
		assert_eq!(given.nrows(), expected.nrows());
		for i in 0..given.nrows() {
			assert_approx_eq!(given.as_ref()[i], expected.as_ref()[i]);
		}
	}

	fn assert_col_approx_eq(given: Col<f64>, expected: &Col<f64>) {
		assert_eq!(given.ncols(), expected.ncols());
		for i in 0..given.ncols() {
			assert_approx_eq!(given.as_ref()[i], expected.as_ref()[i]);
		}
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
