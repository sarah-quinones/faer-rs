use crate::{get_global_parallelism, internal_prelude::*, ScaleGeneric};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

extern crate alloc;

macro_rules! impl_partial_eq {
    ($lhs: ty, $rhs: ty) => {
        impl<
                C: ComplexContainer,
                LhsC: Container<Canonical = C>,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                LhsT: ConjUnit<Canonical = T>,
                RhsT: ConjUnit<Canonical = T>,
                Rows: Shape,
                Cols: Shape,
            > PartialEq<$rhs> for $lhs
        {
            fn eq(&self, other: &$rhs) -> bool {
                self.as_ref().eq(&other.as_ref())
            }
        }
    };
}

macro_rules! impl_1d_partial_eq {
    ($lhs: ty, $rhs: ty) => {
        impl<
                C: ComplexContainer,
                LhsC: Container<Canonical = C>,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                LhsT: ConjUnit<Canonical = T>,
                RhsT: ConjUnit<Canonical = T>,
                Len: Shape,
            > PartialEq<$rhs> for $lhs
        {
            fn eq(&self, other: &$rhs) -> bool {
                self.as_ref().eq(&other.as_ref())
            }
        }
    };
}

macro_rules! impl_add_sub {
    ($lhs: ty, $rhs: ty, $out: ty) => {
        impl<
                C: ComplexContainer,
                LhsC: Container<Canonical = C>,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                LhsT: ConjUnit<Canonical = T>,
                RhsT: ConjUnit<Canonical = T>,
                Rows: Shape,
                Cols: Shape,
            > Add<$rhs> for $lhs
        {
            type Output = $out;
            #[track_caller]
            fn add(self, other: $rhs) -> Self::Output {
                self.as_ref().add(other.as_ref())
            }
        }

        impl<
                C: ComplexContainer,
                LhsC: Container<Canonical = C>,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                LhsT: ConjUnit<Canonical = T>,
                RhsT: ConjUnit<Canonical = T>,
                Rows: Shape,
                Cols: Shape,
            > Sub<$rhs> for $lhs
        {
            type Output = $out;
            #[track_caller]
            fn sub(self, other: $rhs) -> Self::Output {
                self.as_ref().sub(other.as_ref())
            }
        }
    };
}

macro_rules! impl_add_sub_assign {
    ($lhs: ty, $rhs: ty) => {
        impl<
                LhsC: ComplexContainer,
                RhsC: Container<Canonical = LhsC>,
                LhsT: ComplexField<LhsC, MathCtx: Default>,
                RhsT: ConjUnit<Canonical = LhsT>,
                Rows: Shape,
                Cols: Shape,
            > AddAssign<$rhs> for $lhs
        {
            #[track_caller]
            fn add_assign(&mut self, other: $rhs) {
                self.as_mut().add_assign(other.as_ref())
            }
        }

        impl<
                LhsC: ComplexContainer,
                RhsC: Container<Canonical = LhsC>,
                LhsT: ComplexField<LhsC, MathCtx: Default>,
                RhsT: ConjUnit<Canonical = LhsT>,
                Rows: Shape,
                Cols: Shape,
            > SubAssign<$rhs> for $lhs
        {
            #[track_caller]
            fn sub_assign(&mut self, other: $rhs) {
                self.as_mut().sub_assign(other.as_ref())
            }
        }
    };
}

macro_rules! impl_neg {
    ($mat: ty, $out: ty) => {
        impl<
                C: ComplexContainer,
                CC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                TT: ConjUnit<Canonical = T>,
                Rows: Shape,
                Cols: Shape,
            > Neg for $mat
        {
            type Output = $out;
            #[track_caller]
            fn neg(self) -> Self::Output {
                self.as_ref().neg()
            }
        }
    };
}

macro_rules! impl_1d_add_sub {
    ($lhs: ty, $rhs: ty, $out: ty) => {
        impl<
                C: ComplexContainer,
                LhsC: Container<Canonical = C>,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                LhsT: ConjUnit<Canonical = T>,
                RhsT: ConjUnit<Canonical = T>,
                Len: Shape,
            > Add<$rhs> for $lhs
        {
            type Output = $out;
            #[track_caller]
            fn add(self, other: $rhs) -> Self::Output {
                self.as_ref().add(other.as_ref())
            }
        }

        impl<
                C: ComplexContainer,
                LhsC: Container<Canonical = C>,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                LhsT: ConjUnit<Canonical = T>,
                RhsT: ConjUnit<Canonical = T>,
                Len: Shape,
            > Sub<$rhs> for $lhs
        {
            type Output = $out;
            #[track_caller]
            fn sub(self, other: $rhs) -> Self::Output {
                self.as_ref().sub(other.as_ref())
            }
        }
    };
}

macro_rules! impl_1d_add_sub_assign {
    ($lhs: ty, $rhs: ty) => {
        impl<
                LhsC: ComplexContainer,
                RhsC: Container<Canonical = LhsC>,
                LhsT: ComplexField<LhsC, MathCtx: Default>,
                RhsT: ConjUnit<Canonical = LhsT>,
                Len: Shape,
            > AddAssign<$rhs> for $lhs
        {
            #[track_caller]
            fn add_assign(&mut self, other: $rhs) {
                self.as_mut().add_assign(other.as_ref())
            }
        }

        impl<
                LhsC: ComplexContainer,
                RhsC: Container<Canonical = LhsC>,
                LhsT: ComplexField<LhsC, MathCtx: Default>,
                RhsT: ConjUnit<Canonical = LhsT>,
                Len: Shape,
            > SubAssign<$rhs> for $lhs
        {
            #[track_caller]
            fn sub_assign(&mut self, other: $rhs) {
                self.as_mut().sub_assign(other.as_ref())
            }
        }
    };
}

macro_rules! impl_1d_neg {
    ($mat: ty, $out: ty) => {
        impl<
                C: ComplexContainer,
                CC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                TT: ConjUnit<Canonical = T>,
                Len: Shape,
            > Neg for $mat
        {
            type Output = $out;
            #[track_caller]
            fn neg(self) -> Self::Output {
                self.as_ref().neg()
            }
        }
    };
}

macro_rules! impl_mul {
    ($lhs: ty, $rhs: ty, $out: ty) => {
        impl<
                C: ComplexContainer,
                LhsC: Container<Canonical = C>,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                LhsT: ConjUnit<Canonical = T>,
                RhsT: ConjUnit<Canonical = T>,
                M: Shape,
                N: Shape,
                K: Shape,
            > Mul<$rhs> for $lhs
        {
            type Output = $out;
            #[track_caller]
            fn mul(self, other: $rhs) -> Self::Output {
                self.as_ref().mul(other.as_ref())
            }
        }
    };
}

macro_rules! impl_mul_mat_col {
    ($lhs: ty, $rhs: ty, $out: ty) => {
        impl<
                C: ComplexContainer,
                LhsC: Container<Canonical = C>,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                LhsT: ConjUnit<Canonical = T>,
                RhsT: ConjUnit<Canonical = T>,
                M: Shape,
                K: Shape,
            > Mul<$rhs> for $lhs
        {
            type Output = $out;
            #[track_caller]
            fn mul(self, other: $rhs) -> Self::Output {
                self.as_ref().mul(other.as_ref())
            }
        }
    };
}

macro_rules! impl_mul_row_mat {
    ($lhs: ty, $rhs: ty, $out: ty) => {
        impl<
                C: ComplexContainer,
                LhsC: Container<Canonical = C>,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                LhsT: ConjUnit<Canonical = T>,
                RhsT: ConjUnit<Canonical = T>,
                N: Shape,
                K: Shape,
            > Mul<$rhs> for $lhs
        {
            type Output = $out;
            #[track_caller]
            fn mul(self, other: $rhs) -> Self::Output {
                self.as_ref().mul(other.as_ref())
            }
        }
    };
}
macro_rules! impl_mul_row_col {
    ($lhs: ty, $rhs: ty, $out: ty) => {
        impl<
                C: ComplexContainer,
                LhsC: Container<Canonical = C>,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                LhsT: ConjUnit<Canonical = T>,
                RhsT: ConjUnit<Canonical = T>,
                K: Shape,
            > Mul<$rhs> for $lhs
        {
            type Output = $out;
            #[track_caller]
            fn mul(self, other: $rhs) -> Self::Output {
                self.as_ref().mul(other.as_ref())
            }
        }
    };
}

macro_rules! impl_mul_col_row {
    ($lhs: ty, $rhs: ty, $out: ty) => {
        impl<
                C: ComplexContainer,
                LhsC: Container<Canonical = C>,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                LhsT: ConjUnit<Canonical = T>,
                RhsT: ConjUnit<Canonical = T>,
                M: Shape,
                N: Shape,
            > Mul<$rhs> for $lhs
        {
            type Output = $out;
            #[track_caller]
            fn mul(self, other: $rhs) -> Self::Output {
                self.as_ref().mul(other.as_ref())
            }
        }
    };
}

macro_rules! impl_mul_diag_mat {
    ($lhs: ty, $rhs: ty, $out: ty) => {
        impl<
                C: ComplexContainer,
                LhsC: Container<Canonical = C>,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                LhsT: ConjUnit<Canonical = T>,
                RhsT: ConjUnit<Canonical = T>,
                M: Shape,
                N: Shape,
            > Mul<$rhs> for $lhs
        {
            type Output = $out;
            #[track_caller]
            fn mul(self, other: $rhs) -> Self::Output {
                self.as_ref().mul(other.as_ref())
            }
        }
    };
}

macro_rules! impl_mul_diag_col {
    ($lhs: ty, $rhs: ty, $out: ty) => {
        impl<
                C: ComplexContainer,
                LhsC: Container<Canonical = C>,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                LhsT: ConjUnit<Canonical = T>,
                RhsT: ConjUnit<Canonical = T>,
                M: Shape,
            > Mul<$rhs> for $lhs
        {
            type Output = $out;
            #[track_caller]
            fn mul(self, other: $rhs) -> Self::Output {
                self.as_ref().mul(other.as_ref())
            }
        }
    };
}

macro_rules! impl_mul_mat_diag {
    ($lhs: ty, $rhs: ty, $out: ty) => {
        impl<
                C: ComplexContainer,
                LhsC: Container<Canonical = C>,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                LhsT: ConjUnit<Canonical = T>,
                RhsT: ConjUnit<Canonical = T>,
                M: Shape,
                N: Shape,
            > Mul<$rhs> for $lhs
        {
            type Output = $out;
            #[track_caller]
            fn mul(self, other: $rhs) -> Self::Output {
                self.as_ref().mul(other.as_ref())
            }
        }
    };
}

macro_rules! impl_mul_row_diag {
    ($lhs: ty, $rhs: ty, $out: ty) => {
        impl<
                C: ComplexContainer,
                LhsC: Container<Canonical = C>,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                LhsT: ConjUnit<Canonical = T>,
                RhsT: ConjUnit<Canonical = T>,
                N: Shape,
            > Mul<$rhs> for $lhs
        {
            type Output = $out;
            #[track_caller]
            fn mul(self, other: $rhs) -> Self::Output {
                self.as_ref().mul(other.as_ref())
            }
        }
    };
}

macro_rules! impl_perm {
    ($lhs: ty, $rhs: ty, $out: ty) => {
        impl<
                I: Index,
                C: ComplexContainer,
                CC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                TT: ConjUnit<Canonical = T>,
                Rows: Shape,
                Cols: Shape,
            > Mul<$rhs> for $lhs
        {
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
        impl<
                I: Index,
                C: ComplexContainer,
                CC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                TT: ConjUnit<Canonical = T>,
                Len: Shape,
            > Mul<$rhs> for $lhs
        {
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
        impl<
                C: ComplexContainer,
                LhsC: Container<Canonical = C>,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                LhsT: ConjUnit<Canonical = T>,
                RhsT: ConjUnit<Canonical = T>,
                Rows: Shape,
                Cols: Shape,
            > Mul<$rhs> for $lhs
        {
            type Output = $out;
            #[track_caller]
            fn mul(self, other: $rhs) -> Self::Output {
                self.mul(other.as_ref())
            }
        }
    };
}

macro_rules! impl_mul_scalar {
    ($lhs: ty, $rhs: ty, $out: ty) => {
        impl<
                C: ComplexContainer,
                LhsC: Container<Canonical = C>,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                LhsT: ConjUnit<Canonical = T>,
                RhsT: ConjUnit<Canonical = T>,
                Rows: Shape,
                Cols: Shape,
            > Mul<$rhs> for $lhs
        {
            type Output = $out;
            #[track_caller]
            fn mul(self, other: $rhs) -> Self::Output {
                self.as_ref().mul(other)
            }
        }
    };
}

macro_rules! impl_div_scalar {
    ($lhs: ty, $rhs: ty, $out: ty) => {
        impl<
                C: ComplexContainer,
                LhsC: Container<Canonical = C>,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                LhsT: ConjUnit<Canonical = T>,
                RhsT: ConjUnit<Canonical = T>,
                Rows: Shape,
                Cols: Shape,
            > Div<$rhs> for $lhs
        {
            type Output = $out;
            #[track_caller]
            fn div(self, other: $rhs) -> Self::Output {
                let ctx = &Ctx::<C, T>(T::MathCtx::default());
                self.as_ref().mul(ScaleGeneric::<C, T>(
                    ctx.recip(&Conj::apply_val::<RhsC, RhsT>(ctx, &other.0)),
                ))
            }
        }
    };
}

macro_rules! impl_mul_primitive {
    ($rhs: ty, $out: ty) => {
        impl<
                C: ComplexContainer,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                RhsT: ConjUnit<Canonical = T>,
                Rows: Shape,
                Cols: Shape,
            > Mul<$rhs> for f64
        {
            type Output = $out;
            #[track_caller]
            fn mul(self, other: $rhs) -> Self::Output {
                ScaleGeneric::<C, T>(T::from_f64_impl(&default(), self)).mul(other)
            }
        }

        impl<
                C: ComplexContainer,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                RhsT: ConjUnit<Canonical = T>,
                Rows: Shape,
                Cols: Shape,
            > Mul<f64> for $rhs
        {
            type Output = $out;
            #[track_caller]
            fn mul(self, other: f64) -> Self::Output {
                self.mul(ScaleGeneric::<C, T>(T::from_f64_impl(&default(), other)))
            }
        }
        impl<
                C: ComplexContainer,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                RhsT: ConjUnit<Canonical = T>,
                Rows: Shape,
                Cols: Shape,
            > Div<f64> for $rhs
        {
            type Output = $out;
            #[track_caller]
            fn div(self, other: f64) -> Self::Output {
                self.mul(ScaleGeneric::<C, T>(T::from_f64_impl(
                    &default(),
                    other.recip(),
                )))
            }
        }
    };
}

macro_rules! impl_mul_assign_primitive {
    ($lhs: ty) => {
        impl<
                LhsC: ComplexContainer,
                LhsT: ComplexField<LhsC, MathCtx: Default>,
                Rows: Shape,
                Cols: Shape,
            > MulAssign<f64> for $lhs
        {
            #[track_caller]
            fn mul_assign(&mut self, other: f64) {
                self.mul_assign(ScaleGeneric::<LhsC, LhsT>(LhsT::from_f64_impl(
                    &default(),
                    other,
                )))
            }
        }
        impl<
                LhsC: ComplexContainer,
                LhsT: ComplexField<LhsC, MathCtx: Default>,
                Rows: Shape,
                Cols: Shape,
            > DivAssign<f64> for $lhs
        {
            #[track_caller]
            fn div_assign(&mut self, other: f64) {
                self.mul_assign(ScaleGeneric::<LhsC, LhsT>(LhsT::from_f64_impl(
                    &default(),
                    other.recip(),
                )))
            }
        }
    };
}

macro_rules! impl_mul_assign_scalar {
    ($lhs: ty, $rhs: ty) => {
        impl<
                LhsC: ComplexContainer,
                RhsC: Container<Canonical = LhsC>,
                LhsT: ComplexField<LhsC, MathCtx: Default>,
                RhsT: ConjUnit<Canonical = LhsT>,
                Rows: Shape,
                Cols: Shape,
            > MulAssign<$rhs> for $lhs
        {
            #[track_caller]
            fn mul_assign(&mut self, other: $rhs) {
                self.as_mut().mul_assign(other)
            }
        }
    };
}

macro_rules! impl_div_assign_scalar {
    ($lhs: ty, $rhs: ty) => {
        impl<
                LhsC: ComplexContainer,
                RhsC: Container<Canonical = LhsC>,
                LhsT: ComplexField<LhsC, MathCtx: Default>,
                RhsT: ConjUnit<Canonical = LhsT>,
                Rows: Shape,
                Cols: Shape,
            > DivAssign<$rhs> for $lhs
        {
            #[track_caller]
            fn div_assign(&mut self, other: $rhs) {
                let ctx = &Ctx::<LhsC, LhsT>(LhsT::MathCtx::default());
                self.as_mut()
                    .mul_assign(ScaleGeneric::<LhsC, LhsT>(ctx.recip(&Conj::apply_val::<
                        RhsC,
                        RhsT,
                    >(
                        ctx, &other.0
                    ))))
            }
        }
    };
}

macro_rules! impl_1d_scalar_mul {
    ($lhs: ty, $rhs: ty, $out: ty) => {
        impl<
                C: ComplexContainer,
                LhsC: Container<Canonical = C>,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                LhsT: ConjUnit<Canonical = T>,
                RhsT: ConjUnit<Canonical = T>,
                Len: Shape,
            > Mul<$rhs> for $lhs
        {
            type Output = $out;
            #[track_caller]
            fn mul(self, other: $rhs) -> Self::Output {
                self.mul(other.as_ref())
            }
        }
    };
}

macro_rules! impl_1d_mul_scalar {
    ($lhs: ty, $rhs: ty, $out: ty) => {
        impl<
                C: ComplexContainer,
                LhsC: Container<Canonical = C>,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                LhsT: ConjUnit<Canonical = T>,
                RhsT: ConjUnit<Canonical = T>,
                Len: Shape,
            > Mul<$rhs> for $lhs
        {
            type Output = $out;
            #[track_caller]
            fn mul(self, other: $rhs) -> Self::Output {
                self.as_ref().mul(other)
            }
        }
    };
}

macro_rules! impl_1d_div_scalar {
    ($lhs: ty, $rhs: ty, $out: ty) => {
        impl<
                C: ComplexContainer,
                LhsC: Container<Canonical = C>,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                LhsT: ConjUnit<Canonical = T>,
                RhsT: ConjUnit<Canonical = T>,
                Len: Shape,
            > Div<$rhs> for $lhs
        {
            type Output = $out;
            #[track_caller]
            fn div(self, other: $rhs) -> Self::Output {
                let ctx = &Ctx::<C, T>(T::MathCtx::default());
                self.as_ref().mul(ScaleGeneric::<C, T>(
                    ctx.recip(&Conj::apply_val::<RhsC, RhsT>(ctx, &other.0)),
                ))
            }
        }
    };
}

macro_rules! impl_1d_mul_primitive {
    ($rhs: ty, $out: ty) => {
        impl<
                C: ComplexContainer,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                RhsT: ConjUnit<Canonical = T>,
                Len: Shape,
            > Mul<$rhs> for f64
        {
            type Output = $out;
            #[track_caller]
            fn mul(self, other: $rhs) -> Self::Output {
                ScaleGeneric::<C, T>(T::from_f64_impl(&default(), self)).mul(other)
            }
        }

        impl<
                C: ComplexContainer,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                RhsT: ConjUnit<Canonical = T>,
                Len: Shape,
            > Mul<f64> for $rhs
        {
            type Output = $out;
            #[track_caller]
            fn mul(self, other: f64) -> Self::Output {
                self.mul(ScaleGeneric::<C, T>(T::from_f64_impl(&default(), other)))
            }
        }
        impl<
                C: ComplexContainer,
                RhsC: Container<Canonical = C>,
                T: ComplexField<C, MathCtx: Default>,
                RhsT: ConjUnit<Canonical = T>,
                Len: Shape,
            > Div<f64> for $rhs
        {
            type Output = $out;
            #[track_caller]
            fn div(self, other: f64) -> Self::Output {
                self.mul(ScaleGeneric::<C, T>(T::from_f64_impl(
                    &default(),
                    other.recip(),
                )))
            }
        }
    };
}

macro_rules! impl_1d_mul_assign_primitive {
    ($lhs: ty) => {
        impl<LhsC: ComplexContainer, LhsT: ComplexField<LhsC, MathCtx: Default>, Len: Shape>
            MulAssign<f64> for $lhs
        {
            #[track_caller]
            fn mul_assign(&mut self, other: f64) {
                self.mul_assign(ScaleGeneric::<LhsC, LhsT>(LhsT::from_f64_impl(
                    &default(),
                    other,
                )))
            }
        }
        impl<LhsC: ComplexContainer, LhsT: ComplexField<LhsC, MathCtx: Default>, Len: Shape>
            DivAssign<f64> for $lhs
        {
            #[track_caller]
            fn div_assign(&mut self, other: f64) {
                self.mul_assign(ScaleGeneric::<LhsC, LhsT>(LhsT::from_f64_impl(
                    &default(),
                    other.recip(),
                )))
            }
        }
    };
}

macro_rules! impl_1d_mul_assign_scalar {
    ($lhs: ty, $rhs: ty) => {
        impl<
                LhsC: ComplexContainer,
                RhsC: Container<Canonical = LhsC>,
                LhsT: ComplexField<LhsC, MathCtx: Default>,
                RhsT: ConjUnit<Canonical = LhsT>,
                Len: Shape,
            > MulAssign<$rhs> for $lhs
        {
            #[track_caller]
            fn mul_assign(&mut self, other: $rhs) {
                self.as_mut().mul_assign(other)
            }
        }
    };
}

macro_rules! impl_1d_div_assign_scalar {
    ($lhs: ty, $rhs: ty) => {
        impl<
                LhsC: ComplexContainer,
                RhsC: Container<Canonical = LhsC>,
                LhsT: ComplexField<LhsC, MathCtx: Default>,
                RhsT: ConjUnit<Canonical = LhsT>,
                Len: Shape,
            > DivAssign<$rhs> for $lhs
        {
            #[track_caller]
            fn div_assign(&mut self, other: $rhs) {
                let ctx = &Ctx::<LhsC, LhsT>(LhsT::MathCtx::default());
                self.as_mut()
                    .mul_assign(ScaleGeneric::<LhsC, LhsT>(ctx.recip(&Conj::apply_val::<
                        RhsC,
                        RhsT,
                    >(
                        ctx, &other.0
                    ))))
            }
        }
    };
}

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
        Rows: Shape,
        Cols: Shape,
    > PartialEq<MatRef<'_, RhsC, RhsT, Rows, Cols>> for MatRef<'_, LhsC, LhsT, Rows, Cols>
{
    #[math]
    fn eq(&self, other: &MatRef<'_, RhsC, RhsT, Rows, Cols>) -> bool {
        let lhs = *self;
        let rhs = *other;

        if (lhs.nrows().unbound(), lhs.ncols().unbound())
            != (rhs.nrows().unbound(), rhs.ncols().unbound())
        {
            return false;
        }

        fn imp<
            'M,
            'N,
            C: ComplexContainer,
            LhsC: Container<Canonical = C>,
            RhsC: Container<Canonical = C>,
            T: ComplexField<C, MathCtx: Default>,
            LhsT: ConjUnit<Canonical = T>,
            RhsT: ConjUnit<Canonical = T>,
        >(
            lhs: MatRef<'_, LhsC, LhsT, Dim<'M>, Dim<'N>>,
            rhs: MatRef<'_, RhsC, RhsT, Dim<'M>, Dim<'N>>,
        ) -> bool {
            let ctx = &Ctx::<C, T>(T::MathCtx::default());

            let m = lhs.nrows();
            let n = lhs.ncols();
            for j in n.indices() {
                for i in m.indices() {
                    if !math(
                        Conj::apply::<LhsC, LhsT>(ctx, lhs[(i, j)])
                            == Conj::apply::<RhsC, RhsT>(ctx, rhs[(i, j)]),
                    ) {
                        return false;
                    }
                }
            }

            true
        }

        with_dim!(M, lhs.nrows().unbound());
        with_dim!(N, lhs.ncols().unbound());
        imp(lhs.as_shape(M, N), rhs.as_shape(M, N))
    }
}

// impl_partial_eq!(MatRef<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>);
impl_partial_eq!(
    MatRef<'_, LhsC, LhsT, Rows, Cols>,
    MatMut<'_, RhsC, RhsT, Rows, Cols>
);
impl_partial_eq!(MatRef<'_, LhsC, LhsT, Rows, Cols>, Mat<RhsC, RhsT, Rows, Cols>);

impl_partial_eq!(
    MatMut<'_, LhsC, LhsT, Rows, Cols>,
    MatRef<'_, RhsC, RhsT, Rows, Cols>
);
impl_partial_eq!(
    MatMut<'_, LhsC, LhsT, Rows, Cols>,
    MatMut<'_, RhsC, RhsT, Rows, Cols>
);
impl_partial_eq!(MatMut<'_, LhsC, LhsT, Rows, Cols>, Mat<RhsC, RhsT, Rows, Cols, >);

impl_partial_eq!(Mat<LhsC, LhsT, Rows, Cols,>, MatRef<'_, RhsC, RhsT, Rows, Cols>);
impl_partial_eq!(Mat<LhsC, LhsT, Rows, Cols,>, MatMut<'_, RhsC, RhsT, Rows, Cols>);
impl_partial_eq!(Mat<LhsC, LhsT, Rows, Cols,>, Mat<RhsC, RhsT, Rows, Cols>);

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
        Len: Shape,
    > PartialEq<ColRef<'_, RhsC, RhsT, Len>> for ColRef<'_, LhsC, LhsT, Len>
{
    fn eq(&self, other: &ColRef<'_, RhsC, RhsT, Len>) -> bool {
        self.transpose() == other.transpose()
    }
}

// impl_partial_eq!(ColRef<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>);
impl_1d_partial_eq!(ColRef<'_, LhsC, LhsT, Len>, ColMut<'_, RhsC, RhsT, Len>);
impl_1d_partial_eq!(ColRef<'_, LhsC, LhsT, Len>, Col<RhsC, RhsT, Len>);

impl_1d_partial_eq!(ColMut<'_, LhsC, LhsT, Len>, ColRef<'_, RhsC, RhsT, Len>);
impl_1d_partial_eq!(ColMut<'_, LhsC, LhsT, Len>, ColMut<'_, RhsC, RhsT, Len>);
impl_1d_partial_eq!(ColMut<'_, LhsC, LhsT, Len>, Col<RhsC, RhsT, Len>);

impl_1d_partial_eq!(Col<LhsC, LhsT, Len>, ColRef<'_, RhsC, RhsT, Len>);
impl_1d_partial_eq!(Col<LhsC, LhsT, Len>, ColMut<'_, RhsC, RhsT, Len>);
impl_1d_partial_eq!(Col<LhsC, LhsT, Len>, Col<RhsC, RhsT, Len>);

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
        Len: Shape,
    > PartialEq<RowRef<'_, RhsC, RhsT, Len>> for RowRef<'_, LhsC, LhsT, Len>
{
    #[math]
    fn eq(&self, other: &RowRef<'_, RhsC, RhsT, Len>) -> bool {
        let lhs = *self;
        let rhs = *other;

        if lhs.ncols() != rhs.ncols() {
            return false;
        }

        fn imp<
            'N,
            C: ComplexContainer,
            LhsC: Container<Canonical = C>,
            RhsC: Container<Canonical = C>,
            T: ComplexField<C, MathCtx: Default>,
            LhsT: ConjUnit<Canonical = T>,
            RhsT: ConjUnit<Canonical = T>,
        >(
            lhs: RowRef<'_, LhsC, LhsT, Dim<'N>>,
            rhs: RowRef<'_, RhsC, RhsT, Dim<'N>>,
        ) -> bool {
            let ctx = &Ctx::<C, T>(T::MathCtx::default());

            let n = lhs.ncols();
            for j in n.indices() {
                if !math(
                    Conj::apply::<LhsC, LhsT>(ctx, lhs[j])
                        == Conj::apply::<RhsC, RhsT>(ctx, rhs[j]),
                ) {
                    return false;
                }
            }

            true
        }
        with_dim!(N, lhs.ncols().unbound());
        imp(self.as_col_shape(N), other.as_col_shape(N))
    }
}

// impl_partial_eq!(RowRef<'_, LhsC, LhsT>, RowRef<'_, RhsC, RhsT>);
impl_1d_partial_eq!(RowRef<'_, LhsC, LhsT, Len>, RowMut<'_, RhsC, RhsT, Len>);
impl_1d_partial_eq!(RowRef<'_, LhsC, LhsT, Len>, Row<RhsC, RhsT, Len>);

impl_1d_partial_eq!(RowMut<'_, LhsC, LhsT, Len>, RowRef<'_, RhsC, RhsT, Len>);
impl_1d_partial_eq!(RowMut<'_, LhsC, LhsT, Len>, RowMut<'_, RhsC, RhsT, Len>);
impl_1d_partial_eq!(RowMut<'_, LhsC, LhsT, Len>, Row<RhsC, RhsT, Len>);

impl_1d_partial_eq!(Row<LhsC, LhsT, Len>, RowRef<'_, RhsC, RhsT, Len>);
impl_1d_partial_eq!(Row<LhsC, LhsT, Len>, RowMut<'_, RhsC, RhsT, Len>);
impl_1d_partial_eq!(Row<LhsC, LhsT, Len>, Row<RhsC, RhsT, Len>);

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
        Len: Shape,
    > PartialEq<DiagRef<'_, RhsC, RhsT, Len>> for DiagRef<'_, LhsC, LhsT, Len>
{
    fn eq(&self, other: &DiagRef<'_, RhsC, RhsT, Len>) -> bool {
        self.column_vector().eq(&other.column_vector())
    }
}

// impl_partial_eq!(DiagRef<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>);
impl_1d_partial_eq!(DiagRef<'_, LhsC, LhsT, Len>, DiagMut<'_, RhsC, RhsT, Len>);
impl_1d_partial_eq!(DiagRef<'_, LhsC, LhsT, Len>, Diag<RhsC, RhsT, Len>);

impl_1d_partial_eq!(DiagMut<'_, LhsC, LhsT, Len>, DiagRef<'_, RhsC, RhsT, Len>);
impl_1d_partial_eq!(DiagMut<'_, LhsC, LhsT, Len>, DiagMut<'_, RhsC, RhsT, Len>);
impl_1d_partial_eq!(DiagMut<'_, LhsC, LhsT, Len>, Diag<RhsC, RhsT, Len>);

impl_1d_partial_eq!(Diag<LhsC, LhsT, Len>, DiagRef<'_, RhsC, RhsT, Len>);
impl_1d_partial_eq!(Diag<LhsC, LhsT, Len>, DiagMut<'_, RhsC, RhsT, Len>);
impl_1d_partial_eq!(Diag<LhsC, LhsT, Len>, Diag<RhsC, RhsT, Len>);

impl<I: Index> PartialEq<PermRef<'_, I>> for PermRef<'_, I> {
    #[inline]
    fn eq(&self, other: &PermRef<'_, I>) -> bool {
        self.arrays().0 == other.arrays().0
    }
}
impl<I: Index> PartialEq<PermRef<'_, I>> for Perm<I> {
    #[inline]
    fn eq(&self, other: &PermRef<'_, I>) -> bool {
        self.as_ref() == other.as_ref()
    }
}
impl<I: Index> PartialEq<Perm<I>> for PermRef<'_, I> {
    #[inline]
    fn eq(&self, other: &Perm<I>) -> bool {
        self.as_ref() == other.as_ref()
    }
}
impl<I: Index> PartialEq<Perm<I>> for Perm<I> {
    #[inline]
    fn eq(&self, other: &Perm<I>) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
        Rows: Shape,
        Cols: Shape,
    > Add<MatRef<'_, RhsC, RhsT, Rows, Cols>> for MatRef<'_, LhsC, LhsT, Rows, Cols>
{
    type Output = Mat<C, T, Rows, Cols>;

    #[math]
    #[track_caller]
    fn add(self, rhs: MatRef<'_, RhsC, RhsT, Rows, Cols>) -> Self::Output {
        let ctx = &Ctx::<C, T>::default();
        let lhs = self;
        Assert!(all(lhs.nrows() == rhs.nrows(), lhs.ncols() == rhs.ncols()));
        zipped!(lhs, rhs).mapC::<C, _>(add_fn::<LhsC, RhsC, LhsT, RhsT>(ctx))
    }
}

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
        Rows: Shape,
        Cols: Shape,
    > Sub<MatRef<'_, RhsC, RhsT, Rows, Cols>> for MatRef<'_, LhsC, LhsT, Rows, Cols>
{
    type Output = Mat<C, T, Rows, Cols>;

    #[math]
    #[track_caller]
    fn sub(self, rhs: MatRef<'_, RhsC, RhsT, Rows, Cols>) -> Self::Output {
        let ctx = &Ctx::<C, T>::default();
        let lhs = self;
        let rhs = rhs;
        Assert!(all(lhs.nrows() == rhs.nrows(), lhs.ncols() == rhs.ncols()));
        zipped!(lhs, rhs).mapC::<C, _>(sub_fn::<LhsC, RhsC, LhsT, RhsT>(ctx))
    }
}

impl<
        LhsC: ComplexContainer,
        RhsC: Container<Canonical = LhsC>,
        LhsT: ComplexField<LhsC, MathCtx: Default>,
        RhsT: ConjUnit<Canonical = LhsT>,
        Rows: Shape,
        Cols: Shape,
    > AddAssign<MatRef<'_, RhsC, RhsT, Rows, Cols>> for MatMut<'_, LhsC, LhsT, Rows, Cols>
{
    #[math]
    #[track_caller]
    fn add_assign(&mut self, rhs: MatRef<'_, RhsC, RhsT, Rows, Cols>) {
        let ctx = &Ctx::<LhsC, LhsT>::default();
        help!(LhsC);
        zipped!(self.rb_mut(), rhs).for_each(add_assign_fn::<LhsC, RhsC, _, _>(ctx))
    }
}

impl<
        LhsC: ComplexContainer,
        RhsC: Container<Canonical = LhsC>,
        LhsT: ComplexField<LhsC, MathCtx: Default>,
        RhsT: ConjUnit<Canonical = LhsT>,
        Rows: Shape,
        Cols: Shape,
    > SubAssign<MatRef<'_, RhsC, RhsT, Rows, Cols>> for MatMut<'_, LhsC, LhsT, Rows, Cols>
{
    #[math]
    #[track_caller]
    fn sub_assign(&mut self, rhs: MatRef<'_, RhsC, RhsT, Rows, Cols>) {
        let ctx = &Ctx::<LhsC, LhsT>::default();
        help!(LhsC);
        zipped!(self.rb_mut(), rhs).for_each(sub_assign_fn::<LhsC, RhsC, _, _>(ctx))
    }
}

impl<
        C: ComplexContainer,
        CC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        TT: ConjUnit<Canonical = T>,
        Rows: Shape,
        Cols: Shape,
    > Neg for MatRef<'_, CC, TT, Rows, Cols>
{
    type Output = Mat<C, T, Rows, Cols>;

    #[math]
    fn neg(self) -> Self::Output {
        let ctx = &Ctx::<C, T>::default();
        let this = self;
        zipped!(this).mapC::<C, T>(neg_fn::<CC, _>(ctx))
    }
}

#[inline]
fn add_fn<
    LhsC: Container,
    RhsC: Container<Canonical = LhsC::Canonical>,
    LhsT: ConjUnit<Canonical: ComplexField<LhsC::Canonical>>,
    RhsT: ConjUnit<Canonical = LhsT::Canonical>,
>(
    ctx: &Ctx<LhsC::Canonical, LhsT::Canonical>,
) -> impl '_
       + FnMut(
    linalg::zip::Zip<LhsC::Of<&LhsT>, linalg::zip::Last<RhsC::Of<&RhsT>>>,
) -> <LhsC::Canonical as Container>::Of<LhsT::Canonical> {
    help!(LhsC::Canonical);
    #[inline(always)]
    move |unzipped!(a, b)| {
        LhsT::Canonical::add_impl(
            ctx,
            as_ref!(Conj::apply::<LhsC, LhsT>(ctx, a)),
            as_ref!(Conj::apply::<RhsC, RhsT>(ctx, b)),
        )
    }
}

#[inline]
fn sub_fn<
    LhsC: Container,
    RhsC: Container<Canonical = LhsC::Canonical>,
    LhsT: ConjUnit<Canonical: ComplexField<LhsC::Canonical>>,
    RhsT: ConjUnit<Canonical = LhsT::Canonical>,
>(
    ctx: &Ctx<LhsC::Canonical, LhsT::Canonical>,
) -> impl '_
       + FnMut(
    linalg::zip::Zip<LhsC::Of<&LhsT>, linalg::zip::Last<RhsC::Of<&RhsT>>>,
) -> <LhsC::Canonical as Container>::Of<LhsT::Canonical> {
    help!(LhsC::Canonical);
    #[inline(always)]
    move |unzipped!(a, b)| {
        LhsT::Canonical::sub_impl(
            ctx,
            as_ref!(Conj::apply::<LhsC, LhsT>(ctx, a)),
            as_ref!(Conj::apply::<RhsC, RhsT>(ctx, b)),
        )
    }
}

#[inline]
fn mul_fn<
    LhsC: Container,
    RhsC: Container<Canonical = LhsC::Canonical>,
    LhsT: ConjUnit<Canonical: ComplexField<LhsC::Canonical>>,
    RhsT: ConjUnit<Canonical = LhsT::Canonical>,
>(
    ctx: &Ctx<LhsC::Canonical, LhsT::Canonical>,
) -> impl '_
       + FnMut(
    linalg::zip::Zip<LhsC::Of<&LhsT>, linalg::zip::Last<RhsC::Of<&RhsT>>>,
) -> <LhsC::Canonical as Container>::Of<LhsT::Canonical> {
    help!(LhsC::Canonical);
    #[inline(always)]
    move |unzipped!(a, b)| {
        LhsT::Canonical::mul_impl(
            ctx,
            as_ref!(Conj::apply::<LhsC, LhsT>(ctx, a)),
            as_ref!(Conj::apply::<RhsC, RhsT>(ctx, b)),
        )
    }
}

#[inline]
fn neg_fn<LhsC: Container, LhsT: ConjUnit<Canonical: ComplexField<LhsC::Canonical>>>(
    ctx: &Ctx<LhsC::Canonical, LhsT::Canonical>,
) -> impl '_
       + FnMut(
    linalg::zip::Last<LhsC::Of<&LhsT>>,
) -> <LhsC::Canonical as Container>::Of<LhsT::Canonical> {
    help!(LhsC::Canonical);
    #[inline(always)]
    move |unzipped!(a)| LhsT::Canonical::neg_impl(ctx, as_ref!(Conj::apply::<LhsC, LhsT>(ctx, a)))
}

#[inline]
fn add_assign_fn<
    LhsC: ComplexContainer,
    RhsC: Container<Canonical = LhsC>,
    LhsT: ComplexField<LhsC>,
    RhsT: ConjUnit<Canonical = LhsT>,
>(
    ctx: &Ctx<LhsC::Canonical, LhsT::Canonical>,
) -> impl '_ + FnMut(linalg::zip::Zip<LhsC::Of<&mut LhsT>, linalg::zip::Last<RhsC::Of<&RhsT>>>) {
    help!(LhsC::Canonical);
    #[inline(always)]
    move |unzipped!(mut a, b)| {
        write1!(
            a,
            LhsT::Canonical::add_impl(
                ctx,
                as_ref!(Conj::apply::<LhsC, LhsT>(ctx, rb!(a))),
                as_ref!(Conj::apply::<RhsC, RhsT>(ctx, b)),
            )
        )
    }
}

#[inline]
fn sub_assign_fn<
    LhsC: ComplexContainer,
    RhsC: Container<Canonical = LhsC>,
    LhsT: ComplexField<LhsC>,
    RhsT: ConjUnit<Canonical = LhsT>,
>(
    ctx: &Ctx<LhsC::Canonical, LhsT::Canonical>,
) -> impl '_ + FnMut(linalg::zip::Zip<LhsC::Of<&mut LhsT>, linalg::zip::Last<RhsC::Of<&RhsT>>>) {
    help!(LhsC::Canonical);
    #[inline(always)]
    move |unzipped!(mut a, b)| {
        write1!(
            a,
            LhsT::Canonical::sub_impl(
                ctx,
                as_ref!(Conj::apply::<LhsC, LhsT>(ctx, rb!(a))),
                as_ref!(Conj::apply::<RhsC, RhsT>(ctx, b)),
            )
        )
    }
}

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
        Len: Shape,
    > Add<ColRef<'_, RhsC, RhsT, Len>> for ColRef<'_, LhsC, LhsT, Len>
{
    type Output = Col<C, T, Len>;

    #[math]
    #[track_caller]
    fn add(self, rhs: ColRef<'_, RhsC, RhsT, Len>) -> Self::Output {
        let ctx = &Ctx::<C, T>::default();
        let lhs = self;
        Assert!(all(lhs.nrows() == rhs.nrows(), lhs.ncols() == rhs.ncols()));
        zipped!(lhs, rhs).mapC::<C, _>(add_fn::<LhsC, RhsC, LhsT, RhsT>(ctx))
    }
}

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
        Len: Shape,
    > Sub<ColRef<'_, RhsC, RhsT, Len>> for ColRef<'_, LhsC, LhsT, Len>
{
    type Output = Col<C, T, Len>;

    #[math]
    #[track_caller]
    fn sub(self, rhs: ColRef<'_, RhsC, RhsT, Len>) -> Self::Output {
        let ctx = &Ctx::<C, T>::default();
        let lhs = self;
        Assert!(all(lhs.nrows() == rhs.nrows(), lhs.ncols() == rhs.ncols()));
        zipped!(lhs, rhs).mapC::<C, _>(sub_fn::<LhsC, RhsC, LhsT, RhsT>(ctx))
    }
}

impl<
        LhsC: ComplexContainer,
        RhsC: Container<Canonical = LhsC>,
        LhsT: ComplexField<LhsC, MathCtx: Default>,
        RhsT: ConjUnit<Canonical = LhsT>,
        Len: Shape,
    > AddAssign<ColRef<'_, RhsC, RhsT, Len>> for ColMut<'_, LhsC, LhsT, Len>
{
    #[math]
    #[track_caller]
    fn add_assign(&mut self, rhs: ColRef<'_, RhsC, RhsT, Len>) {
        let ctx = &Ctx::<LhsC, LhsT>::default();
        help!(LhsC);
        zipped!(self.rb_mut(), rhs).for_each(add_assign_fn::<LhsC, RhsC, _, _>(ctx))
    }
}

impl<
        LhsC: ComplexContainer,
        RhsC: Container<Canonical = LhsC>,
        LhsT: ComplexField<LhsC, MathCtx: Default>,
        RhsT: ConjUnit<Canonical = LhsT>,
        Len: Shape,
    > SubAssign<ColRef<'_, RhsC, RhsT, Len>> for ColMut<'_, LhsC, LhsT, Len>
{
    #[math]
    #[track_caller]
    fn sub_assign(&mut self, rhs: ColRef<'_, RhsC, RhsT, Len>) {
        let ctx = &Ctx::<LhsC, LhsT>::default();
        help!(LhsC);
        zipped!(self.rb_mut(), rhs).for_each(sub_assign_fn::<LhsC, RhsC, _, _>(ctx))
    }
}

impl<
        C: ComplexContainer,
        CC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        TT: ConjUnit<Canonical = T>,
        Len: Shape,
    > Neg for ColRef<'_, CC, TT, Len>
{
    type Output = Col<C, T, Len>;

    #[math]
    fn neg(self) -> Self::Output {
        let ctx = &Ctx::<C, T>::default();
        let this = self;
        zipped!(this).mapC::<C, T>(neg_fn::<CC, _>(ctx))
    }
}

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
        Len: Shape,
    > Add<RowRef<'_, RhsC, RhsT, Len>> for RowRef<'_, LhsC, LhsT, Len>
{
    type Output = Row<C, T, Len>;

    #[math]
    #[track_caller]
    fn add(self, rhs: RowRef<'_, RhsC, RhsT, Len>) -> Self::Output {
        let ctx = &Ctx::<C, T>::default();
        let lhs = self;
        Assert!(all(lhs.nrows() == rhs.nrows(), lhs.ncols() == rhs.ncols()));
        zipped!(lhs, rhs).mapC::<C, _>(add_fn::<LhsC, RhsC, LhsT, RhsT>(ctx))
    }
}

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
        Len: Shape,
    > Sub<RowRef<'_, RhsC, RhsT, Len>> for RowRef<'_, LhsC, LhsT, Len>
{
    type Output = Row<C, T, Len>;

    #[math]
    #[track_caller]
    fn sub(self, rhs: RowRef<'_, RhsC, RhsT, Len>) -> Self::Output {
        let ctx = &Ctx::<C, T>::default();
        let lhs = self;
        let rhs = rhs;
        Assert!(all(lhs.nrows() == rhs.nrows(), lhs.ncols() == rhs.ncols()));
        zipped!(lhs, rhs).mapC::<C, _>(sub_fn::<LhsC, RhsC, LhsT, RhsT>(ctx))
    }
}

impl<
        LhsC: ComplexContainer,
        RhsC: Container<Canonical = LhsC>,
        LhsT: ComplexField<LhsC, MathCtx: Default>,
        RhsT: ConjUnit<Canonical = LhsT>,
        Len: Shape,
    > AddAssign<RowRef<'_, RhsC, RhsT, Len>> for RowMut<'_, LhsC, LhsT, Len>
{
    #[math]
    #[track_caller]
    fn add_assign(&mut self, rhs: RowRef<'_, RhsC, RhsT, Len>) {
        let ctx = &Ctx::<LhsC, LhsT>::default();
        help!(LhsC);
        zipped!(self.rb_mut(), rhs).for_each(add_assign_fn::<LhsC, RhsC, _, _>(ctx))
    }
}

impl<
        LhsC: ComplexContainer,
        RhsC: Container<Canonical = LhsC>,
        LhsT: ComplexField<LhsC, MathCtx: Default>,
        RhsT: ConjUnit<Canonical = LhsT>,
        Len: Shape,
    > SubAssign<RowRef<'_, RhsC, RhsT, Len>> for RowMut<'_, LhsC, LhsT, Len>
{
    #[math]
    #[track_caller]
    fn sub_assign(&mut self, rhs: RowRef<'_, RhsC, RhsT, Len>) {
        let ctx = &Ctx::<LhsC, LhsT>::default();
        help!(LhsC);
        zipped!(self.rb_mut(), rhs).for_each(sub_assign_fn::<LhsC, RhsC, _, _>(ctx))
    }
}

impl<
        C: ComplexContainer,
        CC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        TT: ConjUnit<Canonical = T>,
        Len: Shape,
    > Neg for RowRef<'_, CC, TT, Len>
{
    type Output = Row<C, T, Len>;

    #[math]
    fn neg(self) -> Self::Output {
        let ctx = &Ctx::<C, T>::default();
        let this = self;
        zipped!(this).mapC::<C, T>(neg_fn::<CC, _>(ctx))
    }
}
impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
        Len: Shape,
    > Add<DiagRef<'_, RhsC, RhsT, Len>> for DiagRef<'_, LhsC, LhsT, Len>
{
    type Output = Diag<C, T, Len>;

    #[track_caller]
    #[math]
    fn add(self, rhs: DiagRef<'_, RhsC, RhsT, Len>) -> Self::Output {
        (self.column_vector() + rhs.column_vector()).into_diagonal()
    }
}

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
        Len: Shape,
    > Sub<DiagRef<'_, RhsC, RhsT, Len>> for DiagRef<'_, LhsC, LhsT, Len>
{
    type Output = Diag<C, T, Len>;

    #[track_caller]
    #[math]
    fn sub(self, rhs: DiagRef<'_, RhsC, RhsT, Len>) -> Self::Output {
        (self.column_vector() - rhs.column_vector()).into_diagonal()
    }
}

impl<
        LhsC: ComplexContainer,
        RhsC: Container<Canonical = LhsC>,
        LhsT: ComplexField<LhsC, MathCtx: Default>,
        RhsT: ConjUnit<Canonical = LhsT>,
        Len: Shape,
    > AddAssign<DiagRef<'_, RhsC, RhsT, Len>> for DiagMut<'_, LhsC, LhsT, Len>
{
    #[track_caller]
    fn add_assign(&mut self, rhs: DiagRef<'_, RhsC, RhsT, Len>) {
        *&mut (self.rb_mut().column_vector_mut()) += rhs.column_vector()
    }
}

impl<
        LhsC: ComplexContainer,
        RhsC: Container<Canonical = LhsC>,
        LhsT: ComplexField<LhsC, MathCtx: Default>,
        RhsT: ConjUnit<Canonical = LhsT>,
        Len: Shape,
    > SubAssign<DiagRef<'_, RhsC, RhsT, Len>> for DiagMut<'_, LhsC, LhsT, Len>
{
    #[track_caller]
    fn sub_assign(&mut self, rhs: DiagRef<'_, RhsC, RhsT, Len>) {
        *&mut (self.rb_mut().column_vector_mut()) -= rhs.column_vector()
    }
}

impl<
        C: ComplexContainer,
        CC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        TT: ConjUnit<Canonical = T>,
        Len: Shape,
    > Neg for DiagRef<'_, CC, TT, Len>
{
    type Output = Diag<C, T, Len>;

    fn neg(self) -> Self::Output {
        (-self.column_vector()).into_diagonal()
    }
}

// impl_add_sub!(MatRef<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(MatRef<'_, LhsC, LhsT, Rows, Cols>, MatMut<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(MatRef<'_, LhsC, LhsT, Rows, Cols>, Mat<RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(MatRef<'_, LhsC, LhsT, Rows, Cols>, &MatRef<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(MatRef<'_, LhsC, LhsT, Rows, Cols>, &MatMut<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(MatRef<'_, LhsC, LhsT, Rows, Cols>, &Mat<RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(&MatRef<'_, LhsC, LhsT, Rows, Cols>, MatRef<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(&MatRef<'_, LhsC, LhsT, Rows, Cols>, MatMut<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(&MatRef<'_, LhsC, LhsT, Rows, Cols>, Mat<RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(&MatRef<'_, LhsC, LhsT, Rows, Cols>, &MatRef<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(&MatRef<'_, LhsC, LhsT, Rows, Cols>, &MatMut<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(&MatRef<'_, LhsC, LhsT, Rows, Cols>, &Mat<RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);

impl_add_sub!(MatMut<'_, LhsC, LhsT, Rows, Cols>, MatRef<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(MatMut<'_, LhsC, LhsT, Rows, Cols>, MatMut<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(MatMut<'_, LhsC, LhsT, Rows, Cols>, Mat<RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(MatMut<'_, LhsC, LhsT, Rows, Cols>, &MatRef<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(MatMut<'_, LhsC, LhsT, Rows, Cols>, &MatMut<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(MatMut<'_, LhsC, LhsT, Rows, Cols>, &Mat<RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(&MatMut<'_, LhsC, LhsT, Rows, Cols>, MatRef<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(&MatMut<'_, LhsC, LhsT, Rows, Cols>, MatMut<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(&MatMut<'_, LhsC, LhsT, Rows, Cols>, Mat<RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(&MatMut<'_, LhsC, LhsT, Rows, Cols>, &MatRef<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(&MatMut<'_, LhsC, LhsT, Rows, Cols>, &MatMut<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(&MatMut<'_, LhsC, LhsT, Rows, Cols>, &Mat<RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);

impl_add_sub!(Mat<LhsC, LhsT, Rows, Cols>, MatRef<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(Mat<LhsC, LhsT, Rows, Cols>, MatMut<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(Mat<LhsC, LhsT, Rows, Cols>, Mat<RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(Mat<LhsC, LhsT, Rows, Cols>, &MatRef<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(Mat<LhsC, LhsT, Rows, Cols>, &MatMut<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(Mat<LhsC, LhsT, Rows, Cols>, &Mat<RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(&Mat<LhsC, LhsT, Rows, Cols>, MatRef<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(&Mat<LhsC, LhsT, Rows, Cols>, MatMut<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(&Mat<LhsC, LhsT, Rows, Cols>, Mat<RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(&Mat<LhsC, LhsT, Rows, Cols>, &MatRef<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(&Mat<LhsC, LhsT, Rows, Cols>, &MatMut<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_add_sub!(&Mat<LhsC, LhsT, Rows, Cols>, &Mat<RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);

// impl_add_sub_assign!(MatMut<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>);
impl_add_sub_assign!(
    MatMut<'_, LhsC, LhsT, Rows, Cols>,
    MatMut<'_, RhsC, RhsT, Rows, Cols>
);
impl_add_sub_assign!(MatMut<'_, LhsC, LhsT, Rows, Cols>, Mat<RhsC, RhsT, Rows, Cols>);
impl_add_sub_assign!(
    MatMut<'_, LhsC, LhsT, Rows, Cols>,
    &MatRef<'_, RhsC, RhsT, Rows, Cols>
);
impl_add_sub_assign!(
    MatMut<'_, LhsC, LhsT, Rows, Cols>,
    &MatMut<'_, RhsC, RhsT, Rows, Cols>
);
impl_add_sub_assign!(MatMut<'_, LhsC, LhsT, Rows, Cols>, &Mat<RhsC, RhsT, Rows, Cols>);

impl_add_sub_assign!(Mat<LhsC, LhsT, Rows, Cols>, MatRef<'_, RhsC, RhsT, Rows, Cols>);
impl_add_sub_assign!(Mat<LhsC, LhsT, Rows, Cols>, MatMut<'_, RhsC, RhsT, Rows, Cols>);
impl_add_sub_assign!(Mat<LhsC, LhsT, Rows, Cols>, Mat<RhsC, RhsT, Rows, Cols>);
impl_add_sub_assign!(Mat<LhsC, LhsT, Rows, Cols>, &MatRef<'_, RhsC, RhsT, Rows, Cols>);
impl_add_sub_assign!(Mat<LhsC, LhsT, Rows, Cols>, &MatMut<'_, RhsC, RhsT, Rows, Cols>);
impl_add_sub_assign!(Mat<LhsC, LhsT, Rows, Cols>, &Mat<RhsC, RhsT, Rows, Cols>);

// impl_neg!(MatRef<'_, CC, TT>, Mat<C, T>);
impl_neg!(MatMut<'_, CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_neg!(Mat<CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_neg!(&MatRef<'_, CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_neg!(&MatMut<'_, CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_neg!(&Mat<CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);

// impl_add_sub!(ColRef<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
impl_1d_add_sub!(ColRef<'_, LhsC, LhsT, Len>, ColMut<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(ColRef<'_, LhsC, LhsT, Len>, Col<RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(ColRef<'_, LhsC, LhsT, Len>, &ColRef<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(ColRef<'_, LhsC, LhsT, Len>, &ColMut<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(ColRef<'_, LhsC, LhsT, Len>, &Col<RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(&ColRef<'_, LhsC, LhsT, Len>, ColRef<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(&ColRef<'_, LhsC, LhsT, Len>, ColMut<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(&ColRef<'_, LhsC, LhsT, Len>, Col<RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(&ColRef<'_, LhsC, LhsT, Len>, &ColRef<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(&ColRef<'_, LhsC, LhsT, Len>, &ColMut<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(&ColRef<'_, LhsC, LhsT, Len>, &Col<RhsC, RhsT, Len>, Col<C, T, Len>);

impl_1d_add_sub!(ColMut<'_, LhsC, LhsT, Len>, ColRef<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(ColMut<'_, LhsC, LhsT, Len>, ColMut<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(ColMut<'_, LhsC, LhsT, Len>, Col<RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(ColMut<'_, LhsC, LhsT, Len>, &ColRef<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(ColMut<'_, LhsC, LhsT, Len>, &ColMut<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(ColMut<'_, LhsC, LhsT, Len>, &Col<RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(&ColMut<'_, LhsC, LhsT, Len>, ColRef<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(&ColMut<'_, LhsC, LhsT, Len>, ColMut<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(&ColMut<'_, LhsC, LhsT, Len>, Col<RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(&ColMut<'_, LhsC, LhsT, Len>, &ColRef<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(&ColMut<'_, LhsC, LhsT, Len>, &ColMut<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(&ColMut<'_, LhsC, LhsT, Len>, &Col<RhsC, RhsT, Len>, Col<C, T, Len>);

impl_1d_add_sub!(Col<LhsC, LhsT, Len>, ColRef<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(Col<LhsC, LhsT, Len>, ColMut<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(Col<LhsC, LhsT, Len>, Col<RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(Col<LhsC, LhsT, Len>, &ColRef<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(Col<LhsC, LhsT, Len>, &ColMut<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(Col<LhsC, LhsT, Len>, &Col<RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(&Col<LhsC, LhsT, Len>, ColRef<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(&Col<LhsC, LhsT, Len>, ColMut<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(&Col<LhsC, LhsT, Len>, Col<RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(&Col<LhsC, LhsT, Len>, &ColRef<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(&Col<LhsC, LhsT, Len>, &ColMut<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_add_sub!(&Col<LhsC, LhsT, Len>, &Col<RhsC, RhsT, Len>, Col<C, T, Len>);

// impl_add_sub_assign!(ColMut<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>);
impl_1d_add_sub_assign!(ColMut<'_, LhsC, LhsT, Len>, ColMut<'_, RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(ColMut<'_, LhsC, LhsT, Len>, Col<RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(ColMut<'_, LhsC, LhsT, Len>, &ColRef<'_, RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(ColMut<'_, LhsC, LhsT, Len>, &ColMut<'_, RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(ColMut<'_, LhsC, LhsT, Len>, &Col<RhsC, RhsT, Len>);

impl_1d_add_sub_assign!(Col<LhsC, LhsT, Len>, ColRef<'_, RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(Col<LhsC, LhsT, Len>, ColMut<'_, RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(Col<LhsC, LhsT, Len>, Col<RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(Col<LhsC, LhsT, Len>, &ColRef<'_, RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(Col<LhsC, LhsT, Len>, &ColMut<'_, RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(Col<LhsC, LhsT, Len>, &Col<RhsC, RhsT, Len>);

// impl_neg!(ColRef<'_, CC, TT>, Col<C, T>);
impl_1d_neg!(ColMut<'_, CC, TT, Len>, Col<C, T, Len>);
impl_1d_neg!(Col<CC, TT, Len>, Col<C, T, Len>);
impl_1d_neg!(&ColRef<'_, CC, TT, Len>, Col<C, T, Len>);
impl_1d_neg!(&ColMut<'_, CC, TT, Len>, Col<C, T, Len>);
impl_1d_neg!(&Col<CC, TT, Len>, Col<C, T, Len>);

// impl_add_sub!(RowRef<'_, LhsC, LhsT>, RowRef<'_, RhsC, RhsT>, Row<C, T>);
impl_1d_add_sub!(RowRef<'_, LhsC, LhsT, Len>, RowMut<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(RowRef<'_, LhsC, LhsT, Len>, Row<RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(RowRef<'_, LhsC, LhsT, Len>, &RowRef<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(RowRef<'_, LhsC, LhsT, Len>, &RowMut<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(RowRef<'_, LhsC, LhsT, Len>, &Row<RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(&RowRef<'_, LhsC, LhsT, Len>, RowRef<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(&RowRef<'_, LhsC, LhsT, Len>, RowMut<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(&RowRef<'_, LhsC, LhsT, Len>, Row<RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(&RowRef<'_, LhsC, LhsT, Len>, &RowRef<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(&RowRef<'_, LhsC, LhsT, Len>, &RowMut<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(&RowRef<'_, LhsC, LhsT, Len>, &Row<RhsC, RhsT, Len>, Row<C, T, Len>);

impl_1d_add_sub!(RowMut<'_, LhsC, LhsT, Len>, RowRef<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(RowMut<'_, LhsC, LhsT, Len>, RowMut<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(RowMut<'_, LhsC, LhsT, Len>, Row<RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(RowMut<'_, LhsC, LhsT, Len>, &RowRef<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(RowMut<'_, LhsC, LhsT, Len>, &RowMut<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(RowMut<'_, LhsC, LhsT, Len>, &Row<RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(&RowMut<'_, LhsC, LhsT, Len>, RowRef<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(&RowMut<'_, LhsC, LhsT, Len>, RowMut<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(&RowMut<'_, LhsC, LhsT, Len>, Row<RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(&RowMut<'_, LhsC, LhsT, Len>, &RowRef<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(&RowMut<'_, LhsC, LhsT, Len>, &RowMut<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(&RowMut<'_, LhsC, LhsT, Len>, &Row<RhsC, RhsT, Len>, Row<C, T, Len>);

impl_1d_add_sub!(Row<LhsC, LhsT, Len>, RowRef<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(Row<LhsC, LhsT, Len>, RowMut<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(Row<LhsC, LhsT, Len>, Row<RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(Row<LhsC, LhsT, Len>, &RowRef<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(Row<LhsC, LhsT, Len>, &RowMut<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(Row<LhsC, LhsT, Len>, &Row<RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(&Row<LhsC, LhsT, Len>, RowRef<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(&Row<LhsC, LhsT, Len>, RowMut<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(&Row<LhsC, LhsT, Len>, Row<RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(&Row<LhsC, LhsT, Len>, &RowRef<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(&Row<LhsC, LhsT, Len>, &RowMut<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_add_sub!(&Row<LhsC, LhsT, Len>, &Row<RhsC, RhsT, Len>, Row<C, T, Len>);

// impl_1d_add_sub_assign!(RowMut<'_, LhsC, LhsT>, RowRef<'_, RhsC, RhsT>);
impl_1d_add_sub_assign!(RowMut<'_, LhsC, LhsT, Len>, RowMut<'_, RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(RowMut<'_, LhsC, LhsT, Len>, Row<RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(RowMut<'_, LhsC, LhsT, Len>, &RowRef<'_, RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(RowMut<'_, LhsC, LhsT, Len>, &RowMut<'_, RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(RowMut<'_, LhsC, LhsT, Len>, &Row<RhsC, RhsT, Len>);

impl_1d_add_sub_assign!(Row<LhsC, LhsT, Len>, RowRef<'_, RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(Row<LhsC, LhsT, Len>, RowMut<'_, RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(Row<LhsC, LhsT, Len>, Row<RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(Row<LhsC, LhsT, Len>, &RowRef<'_, RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(Row<LhsC, LhsT, Len>, &RowMut<'_, RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(Row<LhsC, LhsT, Len>, &Row<RhsC, RhsT, Len>);

// impl_1d_neg!(RowRef<'_, CC, TT>, Row<C, T>);
impl_1d_neg!(RowMut<'_, CC, TT, Len>, Row<C, T, Len>);
impl_1d_neg!(Row<CC, TT, Len>, Row<C, T, Len>);
impl_1d_neg!(&RowRef<'_, CC, TT, Len>, Row<C, T, Len>);
impl_1d_neg!(&RowMut<'_, CC, TT, Len>, Row<C, T, Len>);
impl_1d_neg!(&Row<CC, TT, Len>, Row<C, T, Len>);

// impl_1d_add_sub!(DiagRef<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
impl_1d_add_sub!(DiagRef<'_, LhsC, LhsT, Len>, DiagMut<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(DiagRef<'_, LhsC, LhsT, Len>, Diag<RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(DiagRef<'_, LhsC, LhsT, Len>, &DiagRef<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(DiagRef<'_, LhsC, LhsT, Len>, &DiagMut<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(DiagRef<'_, LhsC, LhsT, Len>, &Diag<RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(&DiagRef<'_, LhsC, LhsT, Len>, DiagRef<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(&DiagRef<'_, LhsC, LhsT, Len>, DiagMut<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(&DiagRef<'_, LhsC, LhsT, Len>, Diag<RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(&DiagRef<'_, LhsC, LhsT, Len>, &DiagRef<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(&DiagRef<'_, LhsC, LhsT, Len>, &DiagMut<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(&DiagRef<'_, LhsC, LhsT, Len>, &Diag<RhsC, RhsT, Len>, Diag<C, T, Len>);

impl_1d_add_sub!(DiagMut<'_, LhsC, LhsT, Len>, DiagRef<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(DiagMut<'_, LhsC, LhsT, Len>, DiagMut<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(DiagMut<'_, LhsC, LhsT, Len>, Diag<RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(DiagMut<'_, LhsC, LhsT, Len>, &DiagRef<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(DiagMut<'_, LhsC, LhsT, Len>, &DiagMut<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(DiagMut<'_, LhsC, LhsT, Len>, &Diag<RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(&DiagMut<'_, LhsC, LhsT, Len>, DiagRef<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(&DiagMut<'_, LhsC, LhsT, Len>, DiagMut<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(&DiagMut<'_, LhsC, LhsT, Len>, Diag<RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(&DiagMut<'_, LhsC, LhsT, Len>, &DiagRef<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(&DiagMut<'_, LhsC, LhsT, Len>, &DiagMut<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(&DiagMut<'_, LhsC, LhsT, Len>, &Diag<RhsC, RhsT, Len>, Diag<C, T, Len>);

impl_1d_add_sub!(Diag<LhsC, LhsT, Len>, DiagRef<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(Diag<LhsC, LhsT, Len>, DiagMut<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(Diag<LhsC, LhsT, Len>, Diag<RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(Diag<LhsC, LhsT, Len>, &DiagRef<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(Diag<LhsC, LhsT, Len>, &DiagMut<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(Diag<LhsC, LhsT, Len>, &Diag<RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(&Diag<LhsC, LhsT, Len>, DiagRef<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(&Diag<LhsC, LhsT, Len>, DiagMut<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(&Diag<LhsC, LhsT, Len>, Diag<RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(&Diag<LhsC, LhsT, Len>, &DiagRef<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(&Diag<LhsC, LhsT, Len>, &DiagMut<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_add_sub!(&Diag<LhsC, LhsT, Len>, &Diag<RhsC, RhsT, Len>, Diag<C, T, Len>);

// impl_add_sub_assign!(DiagMut<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>);
impl_1d_add_sub_assign!(DiagMut<'_, LhsC, LhsT, Len>, DiagMut<'_, RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(DiagMut<'_, LhsC, LhsT, Len>, Diag<RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(DiagMut<'_, LhsC, LhsT, Len>, &DiagRef<'_, RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(DiagMut<'_, LhsC, LhsT, Len>, &DiagMut<'_, RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(DiagMut<'_, LhsC, LhsT, Len>, &Diag<RhsC, RhsT, Len>);

impl_1d_add_sub_assign!(Diag<LhsC, LhsT, Len>, DiagRef<'_, RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(Diag<LhsC, LhsT, Len>, DiagMut<'_, RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(Diag<LhsC, LhsT, Len>, Diag<RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(Diag<LhsC, LhsT, Len>, &DiagRef<'_, RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(Diag<LhsC, LhsT, Len>, &DiagMut<'_, RhsC, RhsT, Len>);
impl_1d_add_sub_assign!(Diag<LhsC, LhsT, Len>, &Diag<RhsC, RhsT, Len>);

// impl_neg!(DiagRef<'_, CC, TT>, Diag<C, T>);
impl_1d_neg!(DiagMut<'_, CC, TT, Len>, Diag<C, T, Len>);
impl_1d_neg!(Diag<CC, TT, Len>, Diag<C, T, Len>);
impl_1d_neg!(&DiagRef<'_, CC, TT, Len>, Diag<C, T, Len>);
impl_1d_neg!(&DiagMut<'_, CC, TT, Len>, Diag<C, T, Len>);
impl_1d_neg!(&Diag<CC, TT, Len>, Diag<C, T, Len>);

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
    > Mul<ScaleGeneric<RhsC, RhsT>> for ScaleGeneric<LhsC, LhsT>
{
    type Output = ScaleGeneric<C, T>;

    #[inline]
    #[math]
    fn mul(self, rhs: ScaleGeneric<RhsC, RhsT>) -> Self::Output {
        let ctx = &Ctx::<C, T>(T::MathCtx::default());
        ScaleGeneric(math(
            Conj::apply_val::<LhsC, _>(ctx, &self.0) * Conj::apply_val::<RhsC, _>(ctx, &rhs.0),
        ))
    }
}

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
    > Add<ScaleGeneric<RhsC, RhsT>> for ScaleGeneric<LhsC, LhsT>
{
    type Output = ScaleGeneric<C, T>;

    #[inline]
    #[math]
    fn add(self, rhs: ScaleGeneric<RhsC, RhsT>) -> Self::Output {
        let ctx = &Ctx::<C, T>(T::MathCtx::default());
        ScaleGeneric(math(
            Conj::apply_val::<LhsC, _>(ctx, &self.0) + Conj::apply_val::<RhsC, _>(ctx, &rhs.0),
        ))
    }
}

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
    > Sub<ScaleGeneric<RhsC, RhsT>> for ScaleGeneric<LhsC, LhsT>
{
    type Output = ScaleGeneric<C, T>;

    #[inline]
    #[math]
    fn sub(self, rhs: ScaleGeneric<RhsC, RhsT>) -> Self::Output {
        let ctx = &Ctx::<C, T>(T::MathCtx::default());
        ScaleGeneric(math(
            Conj::apply_val::<LhsC, _>(ctx, &self.0) - Conj::apply_val::<RhsC, _>(ctx, &rhs.0),
        ))
    }
}

impl<
        LhsC: ComplexContainer,
        RhsC: Container<Canonical = LhsC>,
        LhsT: ComplexField<LhsC, MathCtx: Default>,
        RhsT: ConjUnit<Canonical = LhsT>,
    > MulAssign<ScaleGeneric<RhsC, RhsT>> for ScaleGeneric<LhsC, LhsT>
{
    #[inline]
    #[math]
    fn mul_assign(&mut self, rhs: ScaleGeneric<RhsC, RhsT>) {
        let ctx = &Ctx::<LhsC, LhsT>(LhsT::MathCtx::default());
        *self = ScaleGeneric(math(
            Conj::apply_val::<LhsC, _>(ctx, &self.0) * Conj::apply_val::<RhsC, _>(ctx, &rhs.0),
        ));
    }
}

impl<
        LhsC: ComplexContainer,
        RhsC: Container<Canonical = LhsC>,
        LhsT: ComplexField<LhsC, MathCtx: Default>,
        RhsT: ConjUnit<Canonical = LhsT>,
    > AddAssign<ScaleGeneric<RhsC, RhsT>> for ScaleGeneric<LhsC, LhsT>
{
    #[inline]
    #[math]
    fn add_assign(&mut self, rhs: ScaleGeneric<RhsC, RhsT>) {
        let ctx = &Ctx::<LhsC, LhsT>(LhsT::MathCtx::default());
        *self = ScaleGeneric(math(
            Conj::apply_val::<LhsC, _>(ctx, &self.0) + Conj::apply_val::<RhsC, _>(ctx, &rhs.0),
        ));
    }
}

impl<
        LhsC: ComplexContainer,
        RhsC: Container<Canonical = LhsC>,
        LhsT: ComplexField<LhsC, MathCtx: Default>,
        RhsT: ConjUnit<Canonical = LhsT>,
    > SubAssign<ScaleGeneric<RhsC, RhsT>> for ScaleGeneric<LhsC, LhsT>
{
    #[inline]
    #[math]
    fn sub_assign(&mut self, rhs: ScaleGeneric<RhsC, RhsT>) {
        let ctx = &Ctx::<LhsC, LhsT>(LhsT::MathCtx::default());
        *self = ScaleGeneric(math(
            Conj::apply_val::<LhsC, _>(ctx, &self.0) - Conj::apply_val::<RhsC, _>(ctx, &rhs.0),
        ));
    }
}

mod matmul {
    use super::*;

    impl<
            C: ComplexContainer,
            LhsC: Container<Canonical = C>,
            RhsC: Container<Canonical = C>,
            T: ComplexField<C, MathCtx: Default>,
            LhsT: ConjUnit<Canonical = T>,
            RhsT: ConjUnit<Canonical = T>,
            M: Shape,
            N: Shape,
            K: Shape,
        > Mul<MatRef<'_, RhsC, RhsT, K, N>> for MatRef<'_, LhsC, LhsT, M, K>
    {
        type Output = Mat<C, T, M, N>;

        #[inline]
        #[track_caller]
        fn mul(self, rhs: MatRef<'_, RhsC, RhsT, K, N>) -> Self::Output {
            let lhs = self;
            Assert!(lhs.ncols() == rhs.nrows());
            let ctx = &Ctx::<C, T>::default();
            let mut out = Mat::zeros_with_ctx(ctx, lhs.nrows(), rhs.ncols());
            help!(C);
            crate::linalg::matmul::matmul(
                ctx,
                out.as_mut(),
                None,
                lhs,
                rhs,
                as_ref!(T::one_impl(ctx)),
                get_global_parallelism(),
            );
            out
        }
    }

    impl<
            C: ComplexContainer,
            LhsC: Container<Canonical = C>,
            RhsC: Container<Canonical = C>,
            T: ComplexField<C, MathCtx: Default>,
            LhsT: ConjUnit<Canonical = T>,
            RhsT: ConjUnit<Canonical = T>,
            M: Shape,
            K: Shape,
        > Mul<ColRef<'_, RhsC, RhsT, K>> for MatRef<'_, LhsC, LhsT, M, K>
    {
        type Output = Col<C, T, M>;

        #[inline]
        #[track_caller]
        fn mul(self, rhs: ColRef<'_, RhsC, RhsT, K>) -> Self::Output {
            let lhs = self;
            Assert!(lhs.ncols() == rhs.nrows());
            let ctx = &Ctx::<C, T>::default();
            let mut out = Col::zeros_with_ctx(ctx, lhs.nrows());
            help!(C);
            crate::linalg::matmul::matmul(
                ctx,
                out.as_mut().as_mat_mut(),
                None,
                lhs,
                rhs.as_mat(),
                as_ref!(T::one_impl(ctx)),
                get_global_parallelism(),
            );
            out
        }
    }

    impl<
            C: ComplexContainer,
            LhsC: Container<Canonical = C>,
            RhsC: Container<Canonical = C>,
            T: ComplexField<C, MathCtx: Default>,
            LhsT: ConjUnit<Canonical = T>,
            RhsT: ConjUnit<Canonical = T>,
            N: Shape,
            K: Shape,
        > Mul<MatRef<'_, RhsC, RhsT, K, N>> for RowRef<'_, LhsC, LhsT, K>
    {
        type Output = Row<C, T, N>;

        #[inline]
        #[track_caller]
        fn mul(self, rhs: MatRef<'_, RhsC, RhsT, K, N>) -> Self::Output {
            let lhs = self;
            Assert!(lhs.ncols() == rhs.nrows());
            let ctx = &Ctx::<C, T>::default();
            let mut out = Row::zeros_with_ctx(ctx, rhs.ncols());
            help!(C);
            crate::linalg::matmul::matmul(
                ctx,
                out.as_mut().as_mat_mut(),
                None,
                lhs.as_mat(),
                rhs,
                as_ref!(T::one_impl(ctx)),
                get_global_parallelism(),
            );
            out
        }
    }

    impl<
            C: ComplexContainer,
            LhsC: Container<Canonical = C>,
            RhsC: Container<Canonical = C>,
            T: ComplexField<C, MathCtx: Default>,
            LhsT: ConjUnit<Canonical = T>,
            RhsT: ConjUnit<Canonical = T>,
            K: Shape,
        > Mul<ColRef<'_, RhsC, RhsT, K>> for RowRef<'_, LhsC, LhsT, K>
    {
        type Output = C::Of<T>;

        #[inline]
        #[track_caller]
        fn mul(self, rhs: ColRef<'_, RhsC, RhsT, K>) -> Self::Output {
            let lhs = self;
            Assert!(lhs.ncols() == rhs.nrows());
            let lhs = lhs.canonical();
            let rhs = rhs.canonical();
            let ctx = &Ctx::<C, T>::default();
            with_dim!(K, lhs.ncols().unbound());
            crate::linalg::matmul::dot::inner_prod(
                ctx,
                lhs.as_col_shape(K),
                Conj::get::<LhsC, LhsT>(),
                rhs.as_row_shape(K),
                Conj::get::<RhsC, RhsT>(),
            )
        }
    }

    impl<
            C: ComplexContainer,
            LhsC: Container<Canonical = C>,
            RhsC: Container<Canonical = C>,
            T: ComplexField<C, MathCtx: Default>,
            LhsT: ConjUnit<Canonical = T>,
            RhsT: ConjUnit<Canonical = T>,
            M: Shape,
            N: Shape,
        > Mul<RowRef<'_, RhsC, RhsT, N>> for ColRef<'_, LhsC, LhsT, M>
    {
        type Output = Mat<C, T, M, N>;

        #[inline]
        #[track_caller]
        fn mul(self, rhs: RowRef<'_, RhsC, RhsT, N>) -> Self::Output {
            let lhs = self;
            Assert!(lhs.ncols() == rhs.nrows());
            let ctx = &Ctx::<C, T>::default();
            let mut out = Mat::zeros_with_ctx(ctx, lhs.nrows(), rhs.ncols());
            help!(C);
            crate::linalg::matmul::matmul(
                ctx,
                out.as_mut(),
                None,
                lhs.as_mat(),
                rhs.as_mat(),
                as_ref!(T::one_impl(ctx)),
                get_global_parallelism(),
            );
            out
        }
    }

    // impl_mul!(MatRef<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(MatRef<'_, LhsC, LhsT, M, K>, MatMut<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(MatRef<'_, LhsC, LhsT, M, K>, Mat<RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(MatRef<'_, LhsC, LhsT, M, K>, &MatRef<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(MatRef<'_, LhsC, LhsT, M, K>, &MatMut<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(MatRef<'_, LhsC, LhsT, M, K>, &Mat<RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(&MatRef<'_, LhsC, LhsT, M, K>, MatRef<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(&MatRef<'_, LhsC, LhsT, M, K>, MatMut<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(&MatRef<'_, LhsC, LhsT, M, K>, Mat<RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(&MatRef<'_, LhsC, LhsT, M, K>, &MatRef<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(&MatRef<'_, LhsC, LhsT, M, K>, &MatMut<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(&MatRef<'_, LhsC, LhsT, M, K>, &Mat<RhsC, RhsT, K, N>, Mat<C, T, M, N>);

    impl_mul!(MatMut<'_, LhsC, LhsT, M, K>, MatRef<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(MatMut<'_, LhsC, LhsT, M, K>, MatMut<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(MatMut<'_, LhsC, LhsT, M, K>, Mat<RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(MatMut<'_, LhsC, LhsT, M, K>, &MatRef<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(MatMut<'_, LhsC, LhsT, M, K>, &MatMut<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(MatMut<'_, LhsC, LhsT, M, K>, &Mat<RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(&MatMut<'_, LhsC, LhsT, M, K>, MatRef<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(&MatMut<'_, LhsC, LhsT, M, K>, MatMut<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(&MatMut<'_, LhsC, LhsT, M, K>, Mat<RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(&MatMut<'_, LhsC, LhsT, M, K>, &MatRef<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(&MatMut<'_, LhsC, LhsT, M, K>, &MatMut<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(&MatMut<'_, LhsC, LhsT, M, K>, &Mat<RhsC, RhsT, K, N>, Mat<C, T, M, N>);

    impl_mul!(Mat<LhsC, LhsT, M, K>, MatRef<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(Mat<LhsC, LhsT, M, K>, MatMut<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(Mat<LhsC, LhsT, M, K>, Mat<RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(Mat<LhsC, LhsT, M, K>, &MatRef<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(Mat<LhsC, LhsT, M, K>, &MatMut<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(Mat<LhsC, LhsT, M, K>, &Mat<RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(&Mat<LhsC, LhsT, M, K>, MatRef<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(&Mat<LhsC, LhsT, M, K>, MatMut<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(&Mat<LhsC, LhsT, M, K>, Mat<RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(&Mat<LhsC, LhsT, M, K>, &MatRef<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(&Mat<LhsC, LhsT, M, K>, &MatMut<'_, RhsC, RhsT, K, N>, Mat<C, T, M, N>);
    impl_mul!(&Mat<LhsC, LhsT, M, K>, &Mat<RhsC, RhsT, K, N>, Mat<C, T, M, N>);

    // impl_mul!(MatRef<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul_mat_col!(MatRef<'_, LhsC, LhsT, M, K>, ColMut<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(MatRef<'_, LhsC, LhsT, M, K>, Col<RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(MatRef<'_, LhsC, LhsT, M, K>, &ColRef<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(MatRef<'_, LhsC, LhsT, M, K>, &ColMut<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(MatRef<'_, LhsC, LhsT, M, K>, &Col<RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(&MatRef<'_, LhsC, LhsT, M, K>, ColRef<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(&MatRef<'_, LhsC, LhsT, M, K>, ColMut<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(&MatRef<'_, LhsC, LhsT, M, K>, Col<RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(&MatRef<'_, LhsC, LhsT, M, K>, &ColRef<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(&MatRef<'_, LhsC, LhsT, M, K>, &ColMut<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(&MatRef<'_, LhsC, LhsT, M, K>, &Col<RhsC, RhsT, K>, Col<C, T, M>);

    impl_mul_mat_col!(MatMut<'_, LhsC, LhsT, M, K>, ColRef<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(MatMut<'_, LhsC, LhsT, M, K>, ColMut<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(MatMut<'_, LhsC, LhsT, M, K>, Col<RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(MatMut<'_, LhsC, LhsT, M, K>, &ColRef<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(MatMut<'_, LhsC, LhsT, M, K>, &ColMut<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(MatMut<'_, LhsC, LhsT, M, K>, &Col<RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(&MatMut<'_, LhsC, LhsT, M, K>, ColRef<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(&MatMut<'_, LhsC, LhsT, M, K>, ColMut<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(&MatMut<'_, LhsC, LhsT, M, K>, Col<RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(&MatMut<'_, LhsC, LhsT, M, K>, &ColRef<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(&MatMut<'_, LhsC, LhsT, M, K>, &ColMut<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(&MatMut<'_, LhsC, LhsT, M, K>, &Col<RhsC, RhsT, K>, Col<C, T, M>);

    impl_mul_mat_col!(Mat<LhsC, LhsT, M, K>, ColRef<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(Mat<LhsC, LhsT, M, K>, ColMut<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(Mat<LhsC, LhsT, M, K>, Col<RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(Mat<LhsC, LhsT, M, K>, &ColRef<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(Mat<LhsC, LhsT, M, K>, &ColMut<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(Mat<LhsC, LhsT, M, K>, &Col<RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(&Mat<LhsC, LhsT, M, K>, ColRef<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(&Mat<LhsC, LhsT, M, K>, ColMut<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(&Mat<LhsC, LhsT, M, K>, Col<RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(&Mat<LhsC, LhsT, M, K>, &ColRef<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(&Mat<LhsC, LhsT, M, K>, &ColMut<'_, RhsC, RhsT, K>, Col<C, T, M>);
    impl_mul_mat_col!(&Mat<LhsC, LhsT, M, K>, &Col<RhsC, RhsT, K>, Col<C, T, M>);

    // impl_mul!(RowRef<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul_row_mat!(RowRef<'_, LhsC, LhsT, K>, MatMut<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(RowRef<'_, LhsC, LhsT, K>, Mat<RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(RowRef<'_, LhsC, LhsT, K>, &MatRef<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(RowRef<'_, LhsC, LhsT, K>, &MatMut<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(RowRef<'_, LhsC, LhsT, K>, &Mat<RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(&RowRef<'_, LhsC, LhsT, K>, MatRef<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(&RowRef<'_, LhsC, LhsT, K>, MatMut<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(&RowRef<'_, LhsC, LhsT, K>, Mat<RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(&RowRef<'_, LhsC, LhsT, K>, &MatRef<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(&RowRef<'_, LhsC, LhsT, K>, &MatMut<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(&RowRef<'_, LhsC, LhsT, K>, &Mat<RhsC, RhsT, K, N>, Row<C, T, N>);

    impl_mul_row_mat!(RowMut<'_, LhsC, LhsT, K>, MatRef<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(RowMut<'_, LhsC, LhsT, K>, MatMut<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(RowMut<'_, LhsC, LhsT, K>, Mat<RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(RowMut<'_, LhsC, LhsT, K>, &MatRef<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(RowMut<'_, LhsC, LhsT, K>, &MatMut<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(RowMut<'_, LhsC, LhsT, K>, &Mat<RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(&RowMut<'_, LhsC, LhsT, K>, MatRef<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(&RowMut<'_, LhsC, LhsT, K>, MatMut<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(&RowMut<'_, LhsC, LhsT, K>, Mat<RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(&RowMut<'_, LhsC, LhsT, K>, &MatRef<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(&RowMut<'_, LhsC, LhsT, K>, &MatMut<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(&RowMut<'_, LhsC, LhsT, K>, &Mat<RhsC, RhsT, K, N>, Row<C, T, N>);

    impl_mul_row_mat!(Row<LhsC, LhsT, K>, MatRef<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(Row<LhsC, LhsT, K>, MatMut<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(Row<LhsC, LhsT, K>, Mat<RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(Row<LhsC, LhsT, K>, &MatRef<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(Row<LhsC, LhsT, K>, &MatMut<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(Row<LhsC, LhsT, K>, &Mat<RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(&Row<LhsC, LhsT, K>, MatRef<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(&Row<LhsC, LhsT, K>, MatMut<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(&Row<LhsC, LhsT, K>, Mat<RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(&Row<LhsC, LhsT, K>, &MatRef<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(&Row<LhsC, LhsT, K>, &MatMut<'_, RhsC, RhsT, K, N>, Row<C, T, N>);
    impl_mul_row_mat!(&Row<LhsC, LhsT, K>, &Mat<RhsC, RhsT, K, N>, Row<C, T, N>);

    // impl_mul!(RowRef<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul_row_col!(
        RowRef<'_, LhsC, LhsT, K>,
        ColMut<'_, RhsC, RhsT, K>,
        C::Of<T>
    );
    impl_mul_row_col!(RowRef<'_, LhsC, LhsT, K>, Col<RhsC, RhsT, K>, C::Of<T>);
    impl_mul_row_col!(
        RowRef<'_, LhsC, LhsT, K>,
        &ColRef<'_, RhsC, RhsT, K>,
        C::Of<T>
    );
    impl_mul_row_col!(
        RowRef<'_, LhsC, LhsT, K>,
        &ColMut<'_, RhsC, RhsT, K>,
        C::Of<T>
    );
    impl_mul_row_col!(RowRef<'_, LhsC, LhsT, K>, &Col<RhsC, RhsT, K>, C::Of<T>);
    impl_mul_row_col!(
        &RowRef<'_, LhsC, LhsT, K>,
        ColRef<'_, RhsC, RhsT, K>,
        C::Of<T>
    );
    impl_mul_row_col!(
        &RowRef<'_, LhsC, LhsT, K>,
        ColMut<'_, RhsC, RhsT, K>,
        C::Of<T>
    );
    impl_mul_row_col!(&RowRef<'_, LhsC, LhsT, K>, Col<RhsC, RhsT, K>, C::Of<T>);
    impl_mul_row_col!(
        &RowRef<'_, LhsC, LhsT, K>,
        &ColRef<'_, RhsC, RhsT, K>,
        C::Of<T>
    );
    impl_mul_row_col!(
        &RowRef<'_, LhsC, LhsT, K>,
        &ColMut<'_, RhsC, RhsT, K>,
        C::Of<T>
    );
    impl_mul_row_col!(&RowRef<'_, LhsC, LhsT, K>, &Col<RhsC, RhsT, K>, C::Of<T>);

    impl_mul_row_col!(
        RowMut<'_, LhsC, LhsT, K>,
        ColRef<'_, RhsC, RhsT, K>,
        C::Of<T>
    );
    impl_mul_row_col!(
        RowMut<'_, LhsC, LhsT, K>,
        ColMut<'_, RhsC, RhsT, K>,
        C::Of<T>
    );
    impl_mul_row_col!(RowMut<'_, LhsC, LhsT, K>, Col<RhsC, RhsT, K>, C::Of<T>);
    impl_mul_row_col!(
        RowMut<'_, LhsC, LhsT, K>,
        &ColRef<'_, RhsC, RhsT, K>,
        C::Of<T>
    );
    impl_mul_row_col!(
        RowMut<'_, LhsC, LhsT, K>,
        &ColMut<'_, RhsC, RhsT, K>,
        C::Of<T>
    );
    impl_mul_row_col!(RowMut<'_, LhsC, LhsT, K>, &Col<RhsC, RhsT, K>, C::Of<T>);
    impl_mul_row_col!(
        &RowMut<'_, LhsC, LhsT, K>,
        ColRef<'_, RhsC, RhsT, K>,
        C::Of<T>
    );
    impl_mul_row_col!(
        &RowMut<'_, LhsC, LhsT, K>,
        ColMut<'_, RhsC, RhsT, K>,
        C::Of<T>
    );
    impl_mul_row_col!(&RowMut<'_, LhsC, LhsT, K>, Col<RhsC, RhsT, K>, C::Of<T>);
    impl_mul_row_col!(
        &RowMut<'_, LhsC, LhsT, K>,
        &ColRef<'_, RhsC, RhsT, K>,
        C::Of<T>
    );
    impl_mul_row_col!(
        &RowMut<'_, LhsC, LhsT, K>,
        &ColMut<'_, RhsC, RhsT, K>,
        C::Of<T>
    );
    impl_mul_row_col!(&RowMut<'_, LhsC, LhsT, K>, &Col<RhsC, RhsT, K>, C::Of<T>);

    impl_mul_row_col!(Row<LhsC, LhsT, K>, ColRef<'_, RhsC, RhsT, K>, C::Of<T>);
    impl_mul_row_col!(Row<LhsC, LhsT, K>, ColMut<'_, RhsC, RhsT, K>, C::Of<T>);
    impl_mul_row_col!(Row<LhsC, LhsT, K>, Col<RhsC, RhsT, K>, C::Of<T>);
    impl_mul_row_col!(Row<LhsC, LhsT, K>, &ColRef<'_, RhsC, RhsT, K>, C::Of<T>);
    impl_mul_row_col!(Row<LhsC, LhsT, K>, &ColMut<'_, RhsC, RhsT, K>, C::Of<T>);
    impl_mul_row_col!(Row<LhsC, LhsT, K>, &Col<RhsC, RhsT, K>, C::Of<T>);
    impl_mul_row_col!(&Row<LhsC, LhsT, K>, ColRef<'_, RhsC, RhsT, K>, C::Of<T>);
    impl_mul_row_col!(&Row<LhsC, LhsT, K>, ColMut<'_, RhsC, RhsT, K>, C::Of<T>);
    impl_mul_row_col!(&Row<LhsC, LhsT, K>, Col<RhsC, RhsT, K>, C::Of<T>);
    impl_mul_row_col!(&Row<LhsC, LhsT, K>, &ColRef<'_, RhsC, RhsT, K>, C::Of<T>);
    impl_mul_row_col!(&Row<LhsC, LhsT, K>, &ColMut<'_, RhsC, RhsT, K>, C::Of<T>);
    impl_mul_row_col!(&Row<LhsC, LhsT, K>, &Col<RhsC, RhsT, K>, C::Of<T>);

    // impl_mul!(ColRef<'_, LhsC, LhsT>, RowRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul_col_row!(ColRef<'_, LhsC, LhsT, M>, RowMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(ColRef<'_, LhsC, LhsT, M>, Row<RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(ColRef<'_, LhsC, LhsT, M>, &RowRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(ColRef<'_, LhsC, LhsT, M>, &RowMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(ColRef<'_, LhsC, LhsT, M>, &Row<RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(&ColRef<'_, LhsC, LhsT, M>, RowRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(&ColRef<'_, LhsC, LhsT, M>, RowMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(&ColRef<'_, LhsC, LhsT, M>, Row<RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(&ColRef<'_, LhsC, LhsT, M>, &RowRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(&ColRef<'_, LhsC, LhsT, M>, &RowMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(&ColRef<'_, LhsC, LhsT, M>, &Row<RhsC, RhsT, N>, Mat<C, T, M, N>);

    impl_mul_col_row!(ColMut<'_, LhsC, LhsT, M>, RowRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(ColMut<'_, LhsC, LhsT, M>, RowMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(ColMut<'_, LhsC, LhsT, M>, Row<RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(ColMut<'_, LhsC, LhsT, M>, &RowRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(ColMut<'_, LhsC, LhsT, M>, &RowMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(ColMut<'_, LhsC, LhsT, M>, &Row<RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(&ColMut<'_, LhsC, LhsT, M>, RowRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(&ColMut<'_, LhsC, LhsT, M>, RowMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(&ColMut<'_, LhsC, LhsT, M>, Row<RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(&ColMut<'_, LhsC, LhsT, M>, &RowRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(&ColMut<'_, LhsC, LhsT, M>, &RowMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(&ColMut<'_, LhsC, LhsT, M>, &Row<RhsC, RhsT, N>, Mat<C, T, M, N>);

    impl_mul_col_row!(Col<LhsC, LhsT, M>, RowRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(Col<LhsC, LhsT, M>, RowMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(Col<LhsC, LhsT, M>, Row<RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(Col<LhsC, LhsT, M>, &RowRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(Col<LhsC, LhsT, M>, &RowMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(Col<LhsC, LhsT, M>, &Row<RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(&Col<LhsC, LhsT, M>, RowRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(&Col<LhsC, LhsT, M>, RowMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(&Col<LhsC, LhsT, M>, Row<RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(&Col<LhsC, LhsT, M>, &RowRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(&Col<LhsC, LhsT, M>, &RowMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_col_row!(&Col<LhsC, LhsT, M>, &Row<RhsC, RhsT, N>, Mat<C, T, M, N>);

    impl<
            C: ComplexContainer,
            LhsC: Container<Canonical = C>,
            RhsC: Container<Canonical = C>,
            T: ComplexField<C, MathCtx: Default>,
            LhsT: ConjUnit<Canonical = T>,
            RhsT: ConjUnit<Canonical = T>,
            M: Shape,
            N: Shape,
        > Mul<MatRef<'_, RhsC, RhsT, M, N>> for DiagRef<'_, LhsC, LhsT, M>
    {
        type Output = Mat<C, T, M, N>;

        #[track_caller]
        #[math]
        fn mul(self, rhs: MatRef<'_, RhsC, RhsT, M, N>) -> Self::Output {
            let lhs = self.column_vector();
            let lhs_dim = lhs.nrows();
            let rhs_nrows = rhs.nrows();
            Assert!(lhs_dim == rhs_nrows);

            let ctx = &Ctx::<C, T>(T::MathCtx::default());
            Mat::from_fn(rhs.nrows(), rhs.ncols(), |i, j| {
                math(
                    Conj::apply::<LhsC, LhsT>(ctx, lhs[i])
                        * Conj::apply::<RhsC, RhsT>(ctx, rhs[(i, j)]),
                )
            })
        }
    }

    // impl_mul!(DiagRef<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul_diag_mat!(DiagRef<'_, LhsC, LhsT, M>, MatMut<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(DiagRef<'_, LhsC, LhsT, M>, Mat<RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(DiagRef<'_, LhsC, LhsT, M>, &MatRef<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(DiagRef<'_, LhsC, LhsT, M>, &MatMut<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(DiagRef<'_, LhsC, LhsT, M>, &Mat<RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(&DiagRef<'_, LhsC, LhsT, M>, MatRef<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(&DiagRef<'_, LhsC, LhsT, M>, MatMut<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(&DiagRef<'_, LhsC, LhsT, M>, Mat<RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(&DiagRef<'_, LhsC, LhsT, M>, &MatRef<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(&DiagRef<'_, LhsC, LhsT, M>, &MatMut<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(&DiagRef<'_, LhsC, LhsT, M>, &Mat<RhsC, RhsT, M, N>, Mat<C, T, M, N>);

    impl_mul_diag_mat!(DiagMut<'_, LhsC, LhsT, M>, MatRef<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(DiagMut<'_, LhsC, LhsT, M>, MatMut<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(DiagMut<'_, LhsC, LhsT, M>, Mat<RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(DiagMut<'_, LhsC, LhsT, M>, &MatRef<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(DiagMut<'_, LhsC, LhsT, M>, &MatMut<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(DiagMut<'_, LhsC, LhsT, M>, &Mat<RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(&DiagMut<'_, LhsC, LhsT, M>, MatRef<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(&DiagMut<'_, LhsC, LhsT, M>, MatMut<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(&DiagMut<'_, LhsC, LhsT, M>, Mat<RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(&DiagMut<'_, LhsC, LhsT, M>, &MatRef<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(&DiagMut<'_, LhsC, LhsT, M>, &MatMut<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(&DiagMut<'_, LhsC, LhsT, M>, &Mat<RhsC, RhsT, M, N>, Mat<C, T, M, N>);

    impl_mul_diag_mat!(Diag<LhsC, LhsT, M>, MatRef<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(Diag<LhsC, LhsT, M>, MatMut<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(Diag<LhsC, LhsT, M>, Mat<RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(Diag<LhsC, LhsT, M>, &MatRef<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(Diag<LhsC, LhsT, M>, &MatMut<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(Diag<LhsC, LhsT, M>, &Mat<RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(&Diag<LhsC, LhsT, M>, MatRef<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(&Diag<LhsC, LhsT, M>, MatMut<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(&Diag<LhsC, LhsT, M>, Mat<RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(&Diag<LhsC, LhsT, M>, &MatRef<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(&Diag<LhsC, LhsT, M>, &MatMut<'_, RhsC, RhsT, M, N>, Mat<C, T, M, N>);
    impl_mul_diag_mat!(&Diag<LhsC, LhsT, M>, &Mat<RhsC, RhsT, M, N>, Mat<C, T, M, N>);

    impl<
            C: ComplexContainer,
            LhsC: Container<Canonical = C>,
            RhsC: Container<Canonical = C>,
            T: ComplexField<C, MathCtx: Default>,
            LhsT: ConjUnit<Canonical = T>,
            RhsT: ConjUnit<Canonical = T>,
            M: Shape,
        > Mul<ColRef<'_, RhsC, RhsT, M>> for DiagRef<'_, LhsC, LhsT, M>
    {
        type Output = Col<C, T, M>;

        #[track_caller]
        #[math]
        fn mul(self, rhs: ColRef<'_, RhsC, RhsT, M>) -> Self::Output {
            let lhs = self.column_vector();
            let lhs_dim = lhs.nrows();
            let rhs_nrows = rhs.nrows();
            Assert!(lhs_dim == rhs_nrows);

            let ctx = &Ctx::<C, T>(T::MathCtx::default());
            zipped!(lhs, rhs).mapC::<C, T>(mul_fn::<LhsC, RhsC, _, _>(ctx))
        }
    }

    // impl_mul!(DiagRef<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul_diag_col!(DiagRef<'_, LhsC, LhsT, M>, ColMut<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(DiagRef<'_, LhsC, LhsT, M>, Col<RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(DiagRef<'_, LhsC, LhsT, M>, &ColRef<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(DiagRef<'_, LhsC, LhsT, M>, &ColMut<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(DiagRef<'_, LhsC, LhsT, M>, &Col<RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(&DiagRef<'_, LhsC, LhsT, M>, ColRef<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(&DiagRef<'_, LhsC, LhsT, M>, ColMut<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(&DiagRef<'_, LhsC, LhsT, M>, Col<RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(&DiagRef<'_, LhsC, LhsT, M>, &ColRef<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(&DiagRef<'_, LhsC, LhsT, M>, &ColMut<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(&DiagRef<'_, LhsC, LhsT, M>, &Col<RhsC, RhsT, M>, Col<C, T, M>);

    impl_mul_diag_col!(DiagMut<'_, LhsC, LhsT, M>, ColRef<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(DiagMut<'_, LhsC, LhsT, M>, ColMut<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(DiagMut<'_, LhsC, LhsT, M>, Col<RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(DiagMut<'_, LhsC, LhsT, M>, &ColRef<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(DiagMut<'_, LhsC, LhsT, M>, &ColMut<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(DiagMut<'_, LhsC, LhsT, M>, &Col<RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(&DiagMut<'_, LhsC, LhsT, M>, ColRef<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(&DiagMut<'_, LhsC, LhsT, M>, ColMut<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(&DiagMut<'_, LhsC, LhsT, M>, Col<RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(&DiagMut<'_, LhsC, LhsT, M>, &ColRef<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(&DiagMut<'_, LhsC, LhsT, M>, &ColMut<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(&DiagMut<'_, LhsC, LhsT, M>, &Col<RhsC, RhsT, M>, Col<C, T, M>);

    impl_mul_diag_col!(Diag<LhsC, LhsT, M>, ColRef<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(Diag<LhsC, LhsT, M>, ColMut<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(Diag<LhsC, LhsT, M>, Col<RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(Diag<LhsC, LhsT, M>, &ColRef<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(Diag<LhsC, LhsT, M>, &ColMut<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(Diag<LhsC, LhsT, M>, &Col<RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(&Diag<LhsC, LhsT, M>, ColRef<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(&Diag<LhsC, LhsT, M>, ColMut<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(&Diag<LhsC, LhsT, M>, Col<RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(&Diag<LhsC, LhsT, M>, &ColRef<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(&Diag<LhsC, LhsT, M>, &ColMut<'_, RhsC, RhsT, M>, Col<C, T, M>);
    impl_mul_diag_col!(&Diag<LhsC, LhsT, M>, &Col<RhsC, RhsT, M>, Col<C, T, M>);

    impl<
            C: ComplexContainer,
            LhsC: Container<Canonical = C>,
            RhsC: Container<Canonical = C>,
            T: ComplexField<C, MathCtx: Default>,
            LhsT: ConjUnit<Canonical = T>,
            RhsT: ConjUnit<Canonical = T>,
            M: Shape,
            N: Shape,
        > Mul<DiagRef<'_, RhsC, RhsT, N>> for MatRef<'_, LhsC, LhsT, M, N>
    {
        type Output = Mat<C, T, M, N>;

        #[math]
        #[track_caller]
        fn mul(self, rhs: DiagRef<'_, RhsC, RhsT, N>) -> Self::Output {
            let lhs = self;
            let rhs = rhs.column_vector();
            let lhs_ncols = lhs.ncols();
            let rhs_dim = rhs.nrows();
            Assert!(lhs_ncols == rhs_dim);

            let ctx = &Ctx::<C, T>(T::MathCtx::default());
            Mat::from_fn(lhs.nrows(), lhs.ncols(), |i, j| {
                (i, j);
                math(
                    Conj::apply::<LhsC, LhsT>(ctx, lhs[(i, j)])
                        * Conj::apply::<RhsC, RhsT>(ctx, rhs[j]),
                )
            })
        }
    }

    // impl_mul!(MatRef<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul_mat_diag!(MatRef<'_, LhsC, LhsT, M, N>, DiagMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(MatRef<'_, LhsC, LhsT, M, N>, Diag<RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(MatRef<'_, LhsC, LhsT, M, N>, &DiagRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(MatRef<'_, LhsC, LhsT, M, N>, &DiagMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(MatRef<'_, LhsC, LhsT, M, N>, &Diag<RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(&MatRef<'_, LhsC, LhsT, M, N>, DiagRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(&MatRef<'_, LhsC, LhsT, M, N>, DiagMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(&MatRef<'_, LhsC, LhsT, M, N>, Diag<RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(&MatRef<'_, LhsC, LhsT, M, N>, &DiagRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(&MatRef<'_, LhsC, LhsT, M, N>, &DiagMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(&MatRef<'_, LhsC, LhsT, M, N>, &Diag<RhsC, RhsT, N>, Mat<C, T, M, N>);

    impl_mul_mat_diag!(MatMut<'_, LhsC, LhsT, M, N>, DiagRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(MatMut<'_, LhsC, LhsT, M, N>, DiagMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(MatMut<'_, LhsC, LhsT, M, N>, Diag<RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(MatMut<'_, LhsC, LhsT, M, N>, &DiagRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(MatMut<'_, LhsC, LhsT, M, N>, &DiagMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(MatMut<'_, LhsC, LhsT, M, N>, &Diag<RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(&MatMut<'_, LhsC, LhsT, M, N>, DiagRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(&MatMut<'_, LhsC, LhsT, M, N>, DiagMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(&MatMut<'_, LhsC, LhsT, M, N>, Diag<RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(&MatMut<'_, LhsC, LhsT, M, N>, &DiagRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(&MatMut<'_, LhsC, LhsT, M, N>, &DiagMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(&MatMut<'_, LhsC, LhsT, M, N>, &Diag<RhsC, RhsT, N>, Mat<C, T, M, N>);

    impl_mul_mat_diag!(Mat<LhsC, LhsT, M, N>, DiagRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(Mat<LhsC, LhsT, M, N>, DiagMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(Mat<LhsC, LhsT, M, N>, Diag<RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(Mat<LhsC, LhsT, M, N>, &DiagRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(Mat<LhsC, LhsT, M, N>, &DiagMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(Mat<LhsC, LhsT, M, N>, &Diag<RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(&Mat<LhsC, LhsT, M, N>, DiagRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(&Mat<LhsC, LhsT, M, N>, DiagMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(&Mat<LhsC, LhsT, M, N>, Diag<RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(&Mat<LhsC, LhsT, M, N>, &DiagRef<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(&Mat<LhsC, LhsT, M, N>, &DiagMut<'_, RhsC, RhsT, N>, Mat<C, T, M, N>);
    impl_mul_mat_diag!(&Mat<LhsC, LhsT, M, N>, &Diag<RhsC, RhsT, N>, Mat<C, T, M, N>);

    impl<
            C: ComplexContainer,
            LhsC: Container<Canonical = C>,
            RhsC: Container<Canonical = C>,
            T: ComplexField<C, MathCtx: Default>,
            LhsT: ConjUnit<Canonical = T>,
            RhsT: ConjUnit<Canonical = T>,
            N: Shape,
        > Mul<DiagRef<'_, RhsC, RhsT, N>> for RowRef<'_, LhsC, LhsT, N>
    {
        type Output = Row<C, T, N>;

        #[math]
        #[track_caller]
        fn mul(self, rhs: DiagRef<'_, RhsC, RhsT, N>) -> Self::Output {
            let lhs = self;
            let rhs = rhs.column_vector().transpose();
            let lhs_ncols = lhs.ncols();
            let rhs_dim = rhs.ncols();
            Assert!(lhs_ncols == rhs_dim);

            let ctx = &Ctx::<C, T>(T::MathCtx::default());
            zipped!(lhs, rhs).mapC::<C, T>(mul_fn::<LhsC, RhsC, _, _>(ctx))
        }
    }

    // impl_mul!(RowRef<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul_row_diag!(RowRef<'_, LhsC, LhsT, N>, DiagMut<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(RowRef<'_, LhsC, LhsT, N>, Diag<RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(RowRef<'_, LhsC, LhsT, N>, &DiagRef<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(RowRef<'_, LhsC, LhsT, N>, &DiagMut<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(RowRef<'_, LhsC, LhsT, N>, &Diag<RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(&RowRef<'_, LhsC, LhsT, N>, DiagRef<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(&RowRef<'_, LhsC, LhsT, N>, DiagMut<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(&RowRef<'_, LhsC, LhsT, N>, Diag<RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(&RowRef<'_, LhsC, LhsT, N>, &DiagRef<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(&RowRef<'_, LhsC, LhsT, N>, &DiagMut<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(&RowRef<'_, LhsC, LhsT, N>, &Diag<RhsC, RhsT, N>, Row<C, T, N>);

    impl_mul_row_diag!(RowMut<'_, LhsC, LhsT, N>, DiagRef<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(RowMut<'_, LhsC, LhsT, N>, DiagMut<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(RowMut<'_, LhsC, LhsT, N>, Diag<RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(RowMut<'_, LhsC, LhsT, N>, &DiagRef<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(RowMut<'_, LhsC, LhsT, N>, &DiagMut<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(RowMut<'_, LhsC, LhsT, N>, &Diag<RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(&RowMut<'_, LhsC, LhsT, N>, DiagRef<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(&RowMut<'_, LhsC, LhsT, N>, DiagMut<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(&RowMut<'_, LhsC, LhsT, N>, Diag<RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(&RowMut<'_, LhsC, LhsT, N>, &DiagRef<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(&RowMut<'_, LhsC, LhsT, N>, &DiagMut<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(&RowMut<'_, LhsC, LhsT, N>, &Diag<RhsC, RhsT, N>, Row<C, T, N>);

    impl_mul_row_diag!(Row<LhsC, LhsT, N>, DiagRef<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(Row<LhsC, LhsT, N>, DiagMut<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(Row<LhsC, LhsT, N>, Diag<RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(Row<LhsC, LhsT, N>, &DiagRef<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(Row<LhsC, LhsT, N>, &DiagMut<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(Row<LhsC, LhsT, N>, &Diag<RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(&Row<LhsC, LhsT, N>, DiagRef<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(&Row<LhsC, LhsT, N>, DiagMut<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(&Row<LhsC, LhsT, N>, Diag<RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(&Row<LhsC, LhsT, N>, &DiagRef<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(&Row<LhsC, LhsT, N>, &DiagMut<'_, RhsC, RhsT, N>, Row<C, T, N>);
    impl_mul_row_diag!(&Row<LhsC, LhsT, N>, &Diag<RhsC, RhsT, N>, Row<C, T, N>);

    impl<
            C: ComplexContainer,
            LhsC: Container<Canonical = C>,
            RhsC: Container<Canonical = C>,
            T: ComplexField<C, MathCtx: Default>,
            LhsT: ConjUnit<Canonical = T>,
            RhsT: ConjUnit<Canonical = T>,
            N: Shape,
        > Mul<DiagRef<'_, RhsC, RhsT, N>> for DiagRef<'_, LhsC, LhsT, N>
    {
        type Output = Diag<C, T, N>;

        #[track_caller]
        #[math]
        fn mul(self, rhs: DiagRef<'_, RhsC, RhsT, N>) -> Self::Output {
            let lhs = self.column_vector();
            let rhs = rhs.column_vector();
            Assert!(lhs.nrows() == rhs.nrows());

            let ctx = &Ctx::<C, T>(T::MathCtx::default());
            zipped!(lhs, rhs)
                .mapC::<C, T>(mul_fn::<LhsC, RhsC, _, _>(ctx))
                .into_diagonal()
        }
    }

    // impl_mul!(DiagRef<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul_row_diag!(DiagRef<'_, LhsC, LhsT, N>, DiagMut<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(DiagRef<'_, LhsC, LhsT, N>, Diag<RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(DiagRef<'_, LhsC, LhsT, N>, &DiagRef<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(DiagRef<'_, LhsC, LhsT, N>, &DiagMut<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(DiagRef<'_, LhsC, LhsT, N>, &Diag<RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(&DiagRef<'_, LhsC, LhsT, N>, DiagRef<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(&DiagRef<'_, LhsC, LhsT, N>, DiagMut<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(&DiagRef<'_, LhsC, LhsT, N>, Diag<RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(&DiagRef<'_, LhsC, LhsT, N>, &DiagRef<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(&DiagRef<'_, LhsC, LhsT, N>, &DiagMut<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(&DiagRef<'_, LhsC, LhsT, N>, &Diag<RhsC, RhsT, N>, Diag<C, T, N>);

    impl_mul_row_diag!(DiagMut<'_, LhsC, LhsT, N>, DiagRef<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(DiagMut<'_, LhsC, LhsT, N>, DiagMut<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(DiagMut<'_, LhsC, LhsT, N>, Diag<RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(DiagMut<'_, LhsC, LhsT, N>, &DiagRef<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(DiagMut<'_, LhsC, LhsT, N>, &DiagMut<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(DiagMut<'_, LhsC, LhsT, N>, &Diag<RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(&DiagMut<'_, LhsC, LhsT, N>, DiagRef<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(&DiagMut<'_, LhsC, LhsT, N>, DiagMut<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(&DiagMut<'_, LhsC, LhsT, N>, Diag<RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(&DiagMut<'_, LhsC, LhsT, N>, &DiagRef<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(&DiagMut<'_, LhsC, LhsT, N>, &DiagMut<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(&DiagMut<'_, LhsC, LhsT, N>, &Diag<RhsC, RhsT, N>, Diag<C, T, N>);

    impl_mul_row_diag!(Diag<LhsC, LhsT, N>, DiagRef<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(Diag<LhsC, LhsT, N>, DiagMut<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(Diag<LhsC, LhsT, N>, Diag<RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(Diag<LhsC, LhsT, N>, &DiagRef<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(Diag<LhsC, LhsT, N>, &DiagMut<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(Diag<LhsC, LhsT, N>, &Diag<RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(&Diag<LhsC, LhsT, N>, DiagRef<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(&Diag<LhsC, LhsT, N>, DiagMut<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(&Diag<LhsC, LhsT, N>, Diag<RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(&Diag<LhsC, LhsT, N>, &DiagRef<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(&Diag<LhsC, LhsT, N>, &DiagMut<'_, RhsC, RhsT, N>, Diag<C, T, N>);
    impl_mul_row_diag!(&Diag<LhsC, LhsT, N>, &Diag<RhsC, RhsT, N>, Diag<C, T, N>);
}

impl<I: Index> Mul<PermRef<'_, I>> for PermRef<'_, I> {
    type Output = Perm<I>;

    #[track_caller]
    fn mul(self, rhs: PermRef<'_, I>) -> Self::Output {
        let lhs = self;
        Assert!(lhs.len() == rhs.len());
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

impl<
        I: Index,
        C: ComplexContainer,
        CC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        TT: ConjUnit<Canonical = T>,
        Rows: Shape,
        Cols: Shape,
    > Mul<MatRef<'_, CC, TT, Rows, Cols>> for PermRef<'_, I, Rows>
{
    type Output = Mat<C, T, Rows, Cols>;

    #[track_caller]
    #[math]
    fn mul(self, rhs: MatRef<'_, CC, TT, Rows, Cols>) -> Self::Output {
        let lhs = self;

        Assert!(lhs.len() == rhs.nrows());
        let ctx = &Ctx::<C, T>(T::MathCtx::default());
        let mut out = Mat::zeros_with_ctx(ctx, rhs.nrows(), rhs.ncols());

        fn imp<
            'ROWS,
            'COLS,
            I: Index,
            C: ComplexContainer,
            CC: Container<Canonical = C>,
            T: ComplexField<C, MathCtx: Default>,
            TT: ConjUnit<Canonical = T>,
        >(
            mut out: MatMut<'_, C, T, Dim<'ROWS>, Dim<'COLS>>,
            lhs: PermRef<'_, I, Dim<'ROWS>>,
            rhs: MatRef<'_, CC, TT, Dim<'ROWS>, Dim<'COLS>>,
        ) {
            let ctx = &Ctx::<C, T>(T::MathCtx::default());
            let fwd = lhs.bound_arrays().0;

            for j in rhs.ncols().indices() {
                for i in rhs.nrows().indices() {
                    let fwd = fwd[i];
                    let rhs = rhs.at(fwd.zx(), j);
                    help!(C);
                    math(write1!(
                        out.as_mut().write(i, j),
                        Conj::apply::<CC, TT>(ctx, rhs)
                    ));
                }
            }
        }

        with_dim!(M, out.nrows().unbound());
        with_dim!(N, out.ncols().unbound());
        imp(
            out.as_mut().as_shape_mut(M, N),
            lhs.as_shape(M),
            rhs.as_shape(M, N),
        );

        out
    }
}

// impl_perm!(PermRef<'_, I>, MatRef<'_, CC, TT>, Mat<C, T>);
impl_perm!(PermRef<'_, I, Rows>, MatMut<'_, CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(PermRef<'_, I, Rows>, Mat<CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(PermRef<'_, I, Rows>, &MatRef<'_, CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(PermRef<'_, I, Rows>, &MatMut<'_, CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(PermRef<'_, I, Rows>, &Mat<CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&PermRef<'_, I, Rows>, MatRef<'_, CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&PermRef<'_, I, Rows>, MatMut<'_, CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&PermRef<'_, I, Rows>, Mat<CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&PermRef<'_, I, Rows>, &MatRef<'_, CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&PermRef<'_, I, Rows>, &MatMut<'_, CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&PermRef<'_, I, Rows>, &Mat<CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);

impl_perm!(Perm<I, Rows>, MatRef<'_, CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(Perm<I, Rows>, MatMut<'_, CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(Perm<I, Rows>, Mat<CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(Perm<I, Rows>, &MatRef<'_, CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(Perm<I, Rows>, &MatMut<'_, CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(Perm<I, Rows>, &Mat<CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&Perm<I, Rows>, MatRef<'_, CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&Perm<I, Rows>, MatMut<'_, CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&Perm<I, Rows>, Mat<CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&Perm<I, Rows>, &MatRef<'_, CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&Perm<I, Rows>, &MatMut<'_, CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&Perm<I, Rows>, &Mat<CC, TT, Rows, Cols>, Mat<C, T, Rows, Cols>);

impl<
        I: Index,
        C: ComplexContainer,
        CC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        TT: ConjUnit<Canonical = T>,
        Len: Shape,
    > Mul<ColRef<'_, CC, TT, Len>> for PermRef<'_, I, Len>
{
    type Output = Col<C, T, Len>;

    #[track_caller]
    #[math]
    fn mul(self, rhs: ColRef<'_, CC, TT, Len>) -> Self::Output {
        let lhs = self;

        let ctx = &Ctx::<C, T>(T::MathCtx::default());
        Assert!(lhs.len() == rhs.nrows());
        let mut out = Col::zeros_with_ctx(ctx, rhs.nrows());

        fn imp<
            'ROWS,
            I: Index,
            C: ComplexContainer,
            CC: Container<Canonical = C>,
            T: ComplexField<C, MathCtx: Default>,
            TT: ConjUnit<Canonical = T>,
        >(
            mut out: ColMut<'_, C, T, Dim<'ROWS>>,
            lhs: PermRef<'_, I, Dim<'ROWS>>,
            rhs: ColRef<'_, CC, TT, Dim<'ROWS>>,
        ) {
            let ctx = &Ctx::<C, T>(T::MathCtx::default());
            let fwd = lhs.bound_arrays().0;

            for i in rhs.nrows().indices() {
                let fwd = fwd[i];
                let rhs = rhs.at(fwd.zx());
                help!(C);
                math(write1!(
                    out.as_mut().at_mut(i),
                    Conj::apply::<CC, TT>(ctx, rhs)
                ));
            }
        }

        with_dim!(M, out.nrows().unbound());
        imp(
            out.as_mut().as_row_shape_mut(M),
            lhs.as_shape(M),
            rhs.as_row_shape(M),
        );

        out
    }
}

// impl_perm!(PermRef<'_, I>, ColRef<'_, CC, TT>, Col<C, T>);
impl_1d_perm!(PermRef<'_, I, Len>, ColMut<'_, CC, TT, Len>, Col<C, T, Len>);
impl_1d_perm!(PermRef<'_, I, Len>, Col<CC, TT, Len>, Col<C, T, Len>);
impl_1d_perm!(PermRef<'_, I, Len>, &ColRef<'_, CC, TT, Len>, Col<C, T, Len>);
impl_1d_perm!(PermRef<'_, I, Len>, &ColMut<'_, CC, TT, Len>, Col<C, T, Len>);
impl_1d_perm!(PermRef<'_, I, Len>, &Col<CC, TT, Len>, Col<C, T, Len>);
impl_1d_perm!(&PermRef<'_, I, Len>, ColRef<'_, CC, TT, Len>, Col<C, T, Len>);
impl_1d_perm!(&PermRef<'_, I, Len>, ColMut<'_, CC, TT, Len>, Col<C, T, Len>);
impl_1d_perm!(&PermRef<'_, I, Len>, Col<CC, TT, Len>, Col<C, T, Len>);
impl_1d_perm!(&PermRef<'_, I, Len>, &ColRef<'_, CC, TT, Len>, Col<C, T, Len>);
impl_1d_perm!(&PermRef<'_, I, Len>, &ColMut<'_, CC, TT, Len>, Col<C, T, Len>);
impl_1d_perm!(&PermRef<'_, I, Len>, &Col<CC, TT, Len>, Col<C, T, Len>);

impl_1d_perm!(Perm<I, Len>, ColRef<'_, CC, TT, Len>, Col<C, T, Len>);
impl_1d_perm!(Perm<I, Len>, ColMut<'_, CC, TT, Len>, Col<C, T, Len>);
impl_1d_perm!(Perm<I, Len>, Col<CC, TT, Len>, Col<C, T, Len>);
impl_1d_perm!(Perm<I, Len>, &ColRef<'_, CC, TT, Len>, Col<C, T, Len>);
impl_1d_perm!(Perm<I, Len>, &ColMut<'_, CC, TT, Len>, Col<C, T, Len>);
impl_1d_perm!(Perm<I, Len>, &Col<CC, TT, Len>, Col<C, T, Len>);
impl_1d_perm!(&Perm<I, Len>, ColRef<'_, CC, TT, Len>, Col<C, T, Len>);
impl_1d_perm!(&Perm<I, Len>, ColMut<'_, CC, TT, Len>, Col<C, T, Len>);
impl_1d_perm!(&Perm<I, Len>, Col<CC, TT, Len>, Col<C, T, Len>);
impl_1d_perm!(&Perm<I, Len>, &ColRef<'_, CC, TT, Len>, Col<C, T, Len>);
impl_1d_perm!(&Perm<I, Len>, &ColMut<'_, CC, TT, Len>, Col<C, T, Len>);
impl_1d_perm!(&Perm<I, Len>, &Col<CC, TT, Len>, Col<C, T, Len>);

impl<
        I: Index,
        C: ComplexContainer,
        CC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        TT: ConjUnit<Canonical = T>,
        Rows: Shape,
        Cols: Shape,
    > Mul<PermRef<'_, I, Cols>> for MatRef<'_, CC, TT, Rows, Cols>
{
    type Output = Mat<C, T, Rows, Cols>;

    #[track_caller]
    #[math]
    fn mul(self, rhs: PermRef<'_, I, Cols>) -> Self::Output {
        let lhs = self;

        Assert!(lhs.ncols() == rhs.len());
        let ctx = &Ctx::<C, T>(T::MathCtx::default());
        let mut out = Mat::zeros_with_ctx(ctx, lhs.nrows(), lhs.ncols());

        fn imp<
            'ROWS,
            'COLS,
            I: Index,
            C: ComplexContainer,
            CC: Container<Canonical = C>,
            T: ComplexField<C, MathCtx: Default>,
            TT: ConjUnit<Canonical = T>,
        >(
            mut out: MatMut<'_, C, T, Dim<'ROWS>, Dim<'COLS>>,
            lhs: MatRef<'_, CC, TT, Dim<'ROWS>, Dim<'COLS>>,
            rhs: PermRef<'_, I, Dim<'COLS>>,
        ) {
            let ctx = &Ctx::<C, T>(T::MathCtx::default());
            let inv = rhs.bound_arrays().1;

            for j in lhs.ncols().indices() {
                let inv = inv[j];
                for i in lhs.nrows().indices() {
                    let lhs = lhs.at(i, inv.zx());

                    help!(C);
                    math(write1!(
                        out.as_mut().at_mut(i, j),
                        Conj::apply::<CC, TT>(ctx, lhs)
                    ));
                }
            }
        }

        with_dim!(M, out.nrows().unbound());
        with_dim!(N, out.ncols().unbound());
        imp(out.as_shape_mut(M, N), lhs.as_shape(M, N), rhs.as_shape(N));

        out
    }
}

// impl_perm!(MatRef<'_, CC, TT>, PermRef<'_, I>, Mat<C, T>);
impl_perm!(MatRef<'_, CC, TT, Rows, Cols>, Perm<I, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(MatRef<'_, CC, TT, Rows, Cols>, &PermRef<'_, I, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(MatRef<'_, CC, TT, Rows, Cols>, &Perm<I, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&MatRef<'_, CC, TT, Rows, Cols>, PermRef<'_, I, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&MatRef<'_, CC, TT, Rows, Cols>, Perm<I, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&MatRef<'_, CC, TT, Rows, Cols>, &PermRef<'_, I, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&MatRef<'_, CC, TT, Rows, Cols>, &Perm<I, Cols>, Mat<C, T, Rows, Cols>);

impl_perm!(MatMut<'_, CC, TT, Rows, Cols>, PermRef<'_, I, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(MatMut<'_, CC, TT, Rows, Cols>, Perm<I, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(MatMut<'_, CC, TT, Rows, Cols>, &PermRef<'_, I, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(MatMut<'_, CC, TT, Rows, Cols>, &Perm<I, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&MatMut<'_, CC, TT, Rows, Cols>, PermRef<'_, I, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&MatMut<'_, CC, TT, Rows, Cols>, Perm<I, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&MatMut<'_, CC, TT, Rows, Cols>, &PermRef<'_, I, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&MatMut<'_, CC, TT, Rows, Cols>, &Perm<I, Cols>, Mat<C, T, Rows, Cols>);

impl_perm!(Mat<CC, TT, Rows, Cols>, PermRef<'_, I, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(Mat<CC, TT, Rows, Cols>, Perm<I, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(Mat<CC, TT, Rows, Cols>, &PermRef<'_, I, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(Mat<CC, TT, Rows, Cols>, &Perm<I, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&Mat<CC, TT, Rows, Cols>, PermRef<'_, I, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&Mat<CC, TT, Rows, Cols>, Perm<I, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&Mat<CC, TT, Rows, Cols>, &PermRef<'_, I, Cols>, Mat<C, T, Rows, Cols>);
impl_perm!(&Mat<CC, TT, Rows, Cols>, &Perm<I, Cols>, Mat<C, T, Rows, Cols>);

impl<
        I: Index,
        C: ComplexContainer,
        CC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        TT: ConjUnit<Canonical = T>,
        Len: Shape,
    > Mul<PermRef<'_, I, Len>> for RowRef<'_, CC, TT, Len>
{
    type Output = Row<C, T, Len>;

    #[track_caller]
    #[math]
    fn mul(self, rhs: PermRef<'_, I, Len>) -> Self::Output {
        let lhs = self;

        Assert!(lhs.ncols() == rhs.len());
        let ctx = &Ctx::<C, T>(T::MathCtx::default());
        let mut out = Row::zeros_with_ctx(ctx, lhs.ncols());

        fn imp<
            'COLS,
            I: Index,
            C: ComplexContainer,
            CC: Container<Canonical = C>,
            T: ComplexField<C, MathCtx: Default>,
            TT: ConjUnit<Canonical = T>,
        >(
            mut out: RowMut<'_, C, T, Dim<'COLS>>,
            lhs: RowRef<'_, CC, TT, Dim<'COLS>>,
            rhs: PermRef<'_, I, Dim<'COLS>>,
        ) {
            let ctx = &Ctx::<C, T>(T::MathCtx::default());
            let inv = rhs.bound_arrays().1;

            for j in lhs.ncols().indices() {
                let inv = inv[j];
                let lhs = lhs.at(inv.zx());

                help!(C);
                math(write1!(
                    out.as_mut().at_mut(j),
                    Conj::apply::<CC, TT>(ctx, lhs)
                ));
            }
        }

        with_dim!(N, out.ncols().unbound());
        imp(
            out.as_col_shape_mut(N),
            lhs.as_col_shape(N),
            rhs.as_shape(N),
        );
        out
    }
}

// impl_perm!(RowRef<'_, CC, TT>, PermRef<'_, I>, Row<C, T>);
impl_1d_perm!(RowRef<'_, CC, TT, Len>, Perm<I, Len>, Row<C, T, Len>);
impl_1d_perm!(RowRef<'_, CC, TT, Len>, &PermRef<'_, I, Len>, Row<C, T, Len>);
impl_1d_perm!(RowRef<'_, CC, TT, Len>, &Perm<I, Len>, Row<C, T, Len>);
impl_1d_perm!(&RowRef<'_, CC, TT, Len>, PermRef<'_, I, Len>, Row<C, T, Len>);
impl_1d_perm!(&RowRef<'_, CC, TT, Len>, Perm<I, Len>, Row<C, T, Len>);
impl_1d_perm!(&RowRef<'_, CC, TT, Len>, &PermRef<'_, I, Len>, Row<C, T, Len>);
impl_1d_perm!(&RowRef<'_, CC, TT, Len>, &Perm<I, Len>, Row<C, T, Len>);

impl_1d_perm!(RowMut<'_, CC, TT, Len>, PermRef<'_, I, Len>, Row<C, T, Len>);
impl_1d_perm!(RowMut<'_, CC, TT, Len>, Perm<I, Len>, Row<C, T, Len>);
impl_1d_perm!(RowMut<'_, CC, TT, Len>, &PermRef<'_, I, Len>, Row<C, T, Len>);
impl_1d_perm!(RowMut<'_, CC, TT, Len>, &Perm<I, Len>, Row<C, T, Len>);
impl_1d_perm!(&RowMut<'_, CC, TT, Len>, PermRef<'_, I, Len>, Row<C, T, Len>);
impl_1d_perm!(&RowMut<'_, CC, TT, Len>, Perm<I, Len>, Row<C, T, Len>);
impl_1d_perm!(&RowMut<'_, CC, TT, Len>, &PermRef<'_, I, Len>, Row<C, T, Len>);
impl_1d_perm!(&RowMut<'_, CC, TT, Len>, &Perm<I, Len>, Row<C, T, Len>);

impl_1d_perm!(Row<CC, TT, Len>, PermRef<'_, I, Len>, Row<C, T, Len>);
impl_1d_perm!(Row<CC, TT, Len>, Perm<I, Len>, Row<C, T, Len>);
impl_1d_perm!(Row<CC, TT, Len>, &PermRef<'_, I, Len>, Row<C, T, Len>);
impl_1d_perm!(Row<CC, TT, Len>, &Perm<I, Len>, Row<C, T, Len>);
impl_1d_perm!(&Row<CC, TT, Len>, PermRef<'_, I, Len>, Row<C, T, Len>);
impl_1d_perm!(&Row<CC, TT, Len>, Perm<I, Len>, Row<C, T, Len>);
impl_1d_perm!(&Row<CC, TT, Len>, &PermRef<'_, I, Len>, Row<C, T, Len>);
impl_1d_perm!(&Row<CC, TT, Len>, &Perm<I, Len>, Row<C, T, Len>);

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
        Rows: Shape,
        Cols: Shape,
    > Mul<ScaleGeneric<RhsC, RhsT>> for MatRef<'_, LhsC, LhsT, Rows, Cols>
{
    type Output = Mat<C, T, Rows, Cols>;

    #[math]
    fn mul(self, rhs: ScaleGeneric<RhsC, RhsT>) -> Self::Output {
        let ctx = &Ctx::<C, T>(T::MathCtx::default());
        let rhs = Conj::apply_val::<RhsC, RhsT>(ctx, &rhs.0);
        let lhs = self;
        zipped!(lhs).mapC::<C, T>(|unzipped!(x)| math(Conj::apply::<LhsC, LhsT>(ctx, x) * rhs))
    }
}

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
        Rows: Shape,
        Cols: Shape,
    > Mul<MatRef<'_, RhsC, RhsT, Rows, Cols>> for ScaleGeneric<LhsC, LhsT>
{
    type Output = Mat<C, T, Rows, Cols>;

    #[math]
    fn mul(self, rhs: MatRef<'_, RhsC, RhsT, Rows, Cols>) -> Self::Output {
        let ctx = &Ctx::<C, T>(T::MathCtx::default());
        let lhs = Conj::apply_val::<LhsC, LhsT>(ctx, &self.0);
        zipped!(rhs).mapC::<C, T>(|unzipped!(x)| math(lhs * Conj::apply::<RhsC, RhsT>(ctx, x)))
    }
}

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
        Len: Shape,
    > Mul<ScaleGeneric<RhsC, RhsT>> for ColRef<'_, LhsC, LhsT, Len>
{
    type Output = Col<C, T, Len>;

    #[math]
    fn mul(self, rhs: ScaleGeneric<RhsC, RhsT>) -> Self::Output {
        let ctx = &Ctx::<C, T>(T::MathCtx::default());
        let rhs = Conj::apply_val::<RhsC, RhsT>(ctx, &rhs.0);
        let lhs = self;
        zipped!(lhs).mapC::<C, T>(|unzipped!(x)| math(Conj::apply::<LhsC, LhsT>(ctx, x) * rhs))
    }
}
impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
        Len: Shape,
    > Mul<ColRef<'_, RhsC, RhsT, Len>> for ScaleGeneric<LhsC, LhsT>
{
    type Output = Col<C, T, Len>;

    #[math]
    fn mul(self, rhs: ColRef<'_, RhsC, RhsT, Len>) -> Self::Output {
        let ctx = &Ctx::<C, T>(T::MathCtx::default());
        let lhs = Conj::apply_val::<LhsC, LhsT>(ctx, &self.0);
        zipped!(rhs).mapC::<C, T>(|unzipped!(x)| math(lhs * Conj::apply::<RhsC, RhsT>(ctx, x)))
    }
}

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
        Len: Shape,
    > Mul<ScaleGeneric<RhsC, RhsT>> for RowRef<'_, LhsC, LhsT, Len>
{
    type Output = Row<C, T, Len>;

    #[math]
    fn mul(self, rhs: ScaleGeneric<RhsC, RhsT>) -> Self::Output {
        let ctx = &Ctx::<C, T>(T::MathCtx::default());
        let rhs = Conj::apply_val::<RhsC, RhsT>(ctx, &rhs.0);
        let lhs = self;
        zipped!(lhs).mapC::<C, T>(|unzipped!(x)| math(Conj::apply::<LhsC, LhsT>(ctx, x) * rhs))
    }
}
impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
        Len: Shape,
    > Mul<RowRef<'_, RhsC, RhsT, Len>> for ScaleGeneric<LhsC, LhsT>
{
    type Output = Row<C, T, Len>;

    #[math]
    fn mul(self, rhs: RowRef<'_, RhsC, RhsT, Len>) -> Self::Output {
        let ctx = &Ctx::<C, T>(T::MathCtx::default());
        let lhs = Conj::apply_val::<LhsC, LhsT>(ctx, &self.0);
        zipped!(rhs).mapC::<C, T>(|unzipped!(x)| math(lhs * Conj::apply::<RhsC, RhsT>(ctx, x)))
    }
}

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
        Len: Shape,
    > Mul<ScaleGeneric<RhsC, RhsT>> for DiagRef<'_, LhsC, LhsT, Len>
{
    type Output = Diag<C, T, Len>;

    fn mul(self, rhs: ScaleGeneric<RhsC, RhsT>) -> Self::Output {
        (self.column_vector() * rhs).into_diagonal()
    }
}
impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
        Len: Shape,
    > Mul<DiagRef<'_, RhsC, RhsT, Len>> for ScaleGeneric<LhsC, LhsT>
{
    type Output = Diag<C, T, Len>;

    fn mul(self, rhs: DiagRef<'_, RhsC, RhsT, Len>) -> Self::Output {
        (self * rhs.column_vector()).into_diagonal()
    }
}

impl_mul_scalar!(MatMut<'_, LhsC, LhsT, Rows, Cols>, ScaleGeneric<RhsC, RhsT>, Mat<C, T, Rows, Cols>);
impl_mul_scalar!(Mat<LhsC, LhsT, Rows, Cols>, ScaleGeneric<RhsC, RhsT>, Mat<C, T, Rows, Cols>);
impl_mul_scalar!(&MatRef<'_, LhsC, LhsT, Rows, Cols>, ScaleGeneric<RhsC, RhsT>, Mat<C, T, Rows, Cols>);
impl_mul_scalar!(&MatMut<'_, LhsC, LhsT, Rows, Cols>, ScaleGeneric<RhsC, RhsT>, Mat<C, T, Rows, Cols>);
impl_mul_scalar!(&Mat<LhsC, LhsT, Rows, Cols>, ScaleGeneric<RhsC, RhsT>, Mat<C, T, Rows, Cols>);

impl_div_scalar!(MatRef<'_, LhsC, LhsT, Rows, Cols>, ScaleGeneric<RhsC, RhsT>, Mat<C, T, Rows, Cols>);
impl_div_scalar!(MatMut<'_, LhsC, LhsT, Rows, Cols>, ScaleGeneric<RhsC, RhsT>, Mat<C, T, Rows, Cols>);
impl_div_scalar!(Mat<LhsC, LhsT, Rows, Cols>, ScaleGeneric<RhsC, RhsT>, Mat<C, T, Rows, Cols>);
impl_div_scalar!(&MatRef<'_, LhsC, LhsT, Rows, Cols>, ScaleGeneric<RhsC, RhsT>, Mat<C, T, Rows, Cols>);
impl_div_scalar!(&MatMut<'_, LhsC, LhsT, Rows, Cols>, ScaleGeneric<RhsC, RhsT>, Mat<C, T, Rows, Cols>);
impl_div_scalar!(&Mat<LhsC, LhsT, Rows, Cols>, ScaleGeneric<RhsC, RhsT>, Mat<C, T, Rows, Cols>);

impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, MatMut<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, Mat<RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &MatRef<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &MatMut<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &Mat<RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);

impl_mul_primitive!(MatRef<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_mul_primitive!(MatMut<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_mul_primitive!(Mat<RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_mul_primitive!(&MatRef<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_mul_primitive!(&MatMut<'_, RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);
impl_mul_primitive!(&Mat<RhsC, RhsT, Rows, Cols>, Mat<C, T, Rows, Cols>);

// impl_mul_scalar!(ColRef<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Col<C, T>);
impl_1d_mul_scalar!(ColMut<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Col<C, T, Len>);
impl_1d_mul_scalar!(Col<LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Col<C, T, Len>);
impl_1d_mul_scalar!(&ColRef<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Col<C, T, Len>);
impl_1d_mul_scalar!(&ColMut<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Col<C, T, Len>);
impl_1d_mul_scalar!(&Col<LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Col<C, T, Len>);

impl_1d_div_scalar!(ColRef<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Col<C, T, Len>);
impl_1d_div_scalar!(ColMut<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Col<C, T, Len>);
impl_1d_div_scalar!(Col<LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Col<C, T, Len>);
impl_1d_div_scalar!(&ColRef<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Col<C, T, Len>);
impl_1d_div_scalar!(&ColMut<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Col<C, T, Len>);
impl_1d_div_scalar!(&Col<LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Col<C, T, Len>);

impl_1d_scalar_mul!(ScaleGeneric<LhsC, LhsT>, ColMut<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_scalar_mul!(ScaleGeneric<LhsC, LhsT>, Col<RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &ColRef<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &ColMut<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &Col<RhsC, RhsT, Len>, Col<C, T, Len>);

impl_1d_mul_primitive!(ColRef<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_mul_primitive!(ColMut<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_mul_primitive!(Col<RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_mul_primitive!(&ColRef<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_mul_primitive!(&ColMut<'_, RhsC, RhsT, Len>, Col<C, T, Len>);
impl_1d_mul_primitive!(&Col<RhsC, RhsT, Len>, Col<C, T, Len>);

impl_1d_mul_scalar!(RowMut<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Row<C, T, Len>);
impl_1d_mul_scalar!(Row<LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Row<C, T, Len>);
impl_1d_mul_scalar!(&RowRef<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Row<C, T, Len>);
impl_1d_mul_scalar!(&RowMut<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Row<C, T, Len>);
impl_1d_mul_scalar!(&Row<LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Row<C, T, Len>);

impl_1d_div_scalar!(RowRef<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Row<C, T, Len>);
impl_1d_div_scalar!(RowMut<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Row<C, T, Len>);
impl_1d_div_scalar!(Row<LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Row<C, T, Len>);
impl_1d_div_scalar!(&RowRef<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Row<C, T, Len>);
impl_1d_div_scalar!(&RowMut<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Row<C, T, Len>);
impl_1d_div_scalar!(&Row<LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Row<C, T, Len>);

impl_1d_scalar_mul!(ScaleGeneric<LhsC, LhsT>, RowMut<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_scalar_mul!(ScaleGeneric<LhsC, LhsT>, Row<RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &RowRef<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &RowMut<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &Row<RhsC, RhsT, Len>, Row<C, T, Len>);

impl_1d_mul_primitive!(RowRef<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_mul_primitive!(RowMut<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_mul_primitive!(Row<RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_mul_primitive!(&RowRef<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_mul_primitive!(&RowMut<'_, RhsC, RhsT, Len>, Row<C, T, Len>);
impl_1d_mul_primitive!(&Row<RhsC, RhsT, Len>, Row<C, T, Len>);

impl_1d_mul_scalar!(DiagMut<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Diag<C, T, Len>);
impl_1d_mul_scalar!(Diag<LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Diag<C, T, Len>);
impl_1d_mul_scalar!(&DiagRef<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Diag<C, T, Len>);
impl_1d_mul_scalar!(&DiagMut<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Diag<C, T, Len>);
impl_1d_mul_scalar!(&Diag<LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Diag<C, T, Len>);

impl_1d_div_scalar!(DiagRef<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Diag<C, T, Len>);
impl_1d_div_scalar!(DiagMut<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Diag<C, T, Len>);
impl_1d_div_scalar!(Diag<LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Diag<C, T, Len>);
impl_1d_div_scalar!(&DiagRef<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Diag<C, T, Len>);
impl_1d_div_scalar!(&DiagMut<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Diag<C, T, Len>);
impl_1d_div_scalar!(&Diag<LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>, Diag<C, T, Len>);

impl_1d_scalar_mul!(ScaleGeneric<LhsC, LhsT>, DiagMut<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_scalar_mul!(ScaleGeneric<LhsC, LhsT>, Diag<RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &Diag<RhsC, RhsT, Len>, Diag<C, T, Len>);

impl_1d_mul_primitive!(DiagRef<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_mul_primitive!(DiagMut<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_mul_primitive!(Diag<RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_mul_primitive!(&DiagRef<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_mul_primitive!(&DiagMut<'_, RhsC, RhsT, Len>, Diag<C, T, Len>);
impl_1d_mul_primitive!(&Diag<RhsC, RhsT, Len>, Diag<C, T, Len>);

impl<
        LhsC: ComplexContainer,
        RhsC: Container<Canonical = LhsC>,
        LhsT: ComplexField<LhsC, MathCtx: Default>,
        RhsT: ConjUnit<Canonical = LhsT>,
        Rows: Shape,
        Cols: Shape,
    > MulAssign<ScaleGeneric<RhsC, RhsT>> for MatMut<'_, LhsC, LhsT, Rows, Cols>
{
    #[math]
    fn mul_assign(&mut self, rhs: ScaleGeneric<RhsC, RhsT>) {
        let ctx = &Ctx::<LhsC, LhsT>(LhsT::MathCtx::default());
        let rhs = Conj::apply_val::<RhsC, RhsT>(ctx, &rhs.0);

        help!(LhsC);
        zipped!(self.rb_mut()).for_each(|unzipped!(mut x)| math(write1!(x, x * rhs)))
    }
}
impl<
        LhsC: ComplexContainer,
        RhsC: Container<Canonical = LhsC>,
        LhsT: ComplexField<LhsC, MathCtx: Default>,
        RhsT: ConjUnit<Canonical = LhsT>,
        Len: Shape,
    > MulAssign<ScaleGeneric<RhsC, RhsT>> for ColMut<'_, LhsC, LhsT, Len>
{
    #[math]
    fn mul_assign(&mut self, rhs: ScaleGeneric<RhsC, RhsT>) {
        let ctx = &Ctx::<LhsC, LhsT>(LhsT::MathCtx::default());
        let rhs = Conj::apply_val::<RhsC, RhsT>(ctx, &rhs.0);

        help!(LhsC);
        zipped!(self.rb_mut()).for_each(|unzipped!(mut x)| math(write1!(x, x * rhs)))
    }
}
impl<
        LhsC: ComplexContainer,
        RhsC: Container<Canonical = LhsC>,
        LhsT: ComplexField<LhsC, MathCtx: Default>,
        RhsT: ConjUnit<Canonical = LhsT>,
        Len: Shape,
    > MulAssign<ScaleGeneric<RhsC, RhsT>> for RowMut<'_, LhsC, LhsT, Len>
{
    #[math]
    fn mul_assign(&mut self, rhs: ScaleGeneric<RhsC, RhsT>) {
        let ctx = &Ctx::<LhsC, LhsT>(LhsT::MathCtx::default());
        let rhs = Conj::apply_val::<RhsC, RhsT>(ctx, &rhs.0);

        help!(LhsC);
        zipped!(self.rb_mut()).for_each(|unzipped!(mut x)| math(write1!(x, x * rhs)))
    }
}
impl<
        LhsC: ComplexContainer,
        RhsC: Container<Canonical = LhsC>,
        LhsT: ComplexField<LhsC, MathCtx: Default>,
        RhsT: ConjUnit<Canonical = LhsT>,
        Len: Shape,
    > MulAssign<ScaleGeneric<RhsC, RhsT>> for DiagMut<'_, LhsC, LhsT, Len>
{
    fn mul_assign(&mut self, rhs: ScaleGeneric<RhsC, RhsT>) {
        let mut this = self.rb_mut().column_vector_mut();
        this *= rhs;
    }
}

impl_mul_assign_scalar!(Mat<LhsC, LhsT, Rows, Cols>, ScaleGeneric<RhsC, RhsT>);
impl_1d_mul_assign_scalar!(Col<LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>);
impl_1d_mul_assign_scalar!(Row<LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>);
impl_1d_mul_assign_scalar!(Diag<LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>);

impl_div_assign_scalar!(MatMut<'_, LhsC, LhsT, Rows, Cols>, ScaleGeneric<RhsC, RhsT>);
impl_div_assign_scalar!(Mat<LhsC, LhsT, Rows, Cols>, ScaleGeneric<RhsC, RhsT>);
impl_1d_div_assign_scalar!(ColMut<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>);
impl_1d_div_assign_scalar!(Col<LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>);
impl_1d_div_assign_scalar!(RowMut<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>);
impl_1d_div_assign_scalar!(Row<LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>);
impl_1d_div_assign_scalar!(DiagMut<'_, LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>);
impl_1d_div_assign_scalar!(Diag<LhsC, LhsT, Len>, ScaleGeneric<RhsC, RhsT>);

impl_mul_assign_primitive!(MatMut<'_, LhsC, LhsT, Rows, Cols>);
impl_mul_assign_primitive!(Mat<LhsC, LhsT, Rows, Cols>);
impl_1d_mul_assign_primitive!(ColMut<'_, LhsC, LhsT, Len>);
impl_1d_mul_assign_primitive!(Col<LhsC, LhsT, Len>);
impl_1d_mul_assign_primitive!(RowMut<'_, LhsC, LhsT, Len>);
impl_1d_mul_assign_primitive!(Row<LhsC, LhsT, Len>);
impl_1d_mul_assign_primitive!(DiagMut<'_, LhsC, LhsT, Len>);
impl_1d_mul_assign_primitive!(Diag<LhsC, LhsT, Len>);

#[cfg(any())]
mod sparse {
    macro_rules! impl_mul_primitive_sparse {
        ($rhs: ty, $out: ty) => {
            impl<I: Index, E: ComplexField, RhsE: Conjugate<Canonical = E>> Mul<$rhs> for f64 {
                type Output = $out;
                #[track_caller]
                fn mul(self, other: $rhs) -> Self::Output {
                    ScaleGeneric(T::from_f64_impl(&default(), self)).mul(other)
                }
            }
            impl<I: Index, E: ComplexField, RhsE: Conjugate<Canonical = E>> Mul<$rhs> for f32 {
                type Output = $out;
                #[track_caller]
                fn mul(self, other: $rhs) -> Self::Output {
                    ScaleGeneric(T::from_f64_impl(&default(), self as f64)).mul(other)
                }
            }

            impl<I: Index, E: ComplexField, RhsE: Conjugate<Canonical = E>> Mul<f64> for $rhs {
                type Output = $out;
                #[track_caller]
                fn mul(self, other: f64) -> Self::Output {
                    self.mul(ScaleGeneric(T::from_f64_impl(&default(), other)))
                }
            }
            impl<I: Index, E: ComplexField, RhsE: Conjugate<Canonical = E>> Mul<f32> for $rhs {
                type Output = $out;
                #[track_caller]
                fn mul(self, other: f32) -> Self::Output {
                    self.mul(ScaleGeneric(T::from_f64_impl(&default(), other as f64)))
                }
            }
            impl<I: Index, E: ComplexField, RhsE: Conjugate<Canonical = E>> Div<f64> for $rhs {
                type Output = $out;
                #[track_caller]
                fn div(self, other: f64) -> Self::Output {
                    self.mul(ScaleGeneric(T::from_f64_impl(&default(), other.recip())))
                }
            }
            impl<I: Index, E: ComplexField, RhsE: Conjugate<Canonical = E>> Div<f32> for $rhs {
                type Output = $out;
                #[track_caller]
                fn div(self, other: f32) -> Self::Output {
                    self.mul(ScaleGeneric(T::from_f64_impl(
                        &default(),
                        other.recip() as f64,
                    )))
                }
            }
        };
    }

    macro_rules! impl_mul_assign_primitive_sparse {
        ($lhs: ty) => {
            impl<I: Index, LhsE: ComplexField> MulAssign<f64> for $lhs {
                #[track_caller]
                fn mul_assign(&mut self, other: f64) {
                    self.mul_assign(ScaleGeneric(LhsT::from_f64_impl(&default(), other)))
                }
            }
            impl<I: Index, LhsE: ComplexField> MulAssign<f32> for $lhs {
                #[track_caller]
                fn mul_assign(&mut self, other: f32) {
                    self.mul_assign(ScaleGeneric(LhsT::from_f64_impl(&default(), other as f64)))
                }
            }
            impl<I: Index, LhsE: ComplexField> DivAssign<f64> for $lhs {
                #[track_caller]
                fn div_assign(&mut self, other: f64) {
                    self.mul_assign(ScaleGeneric(LhsT::from_f64_impl(&default(), other.recip())))
                }
            }
            impl<I: Index, LhsE: ComplexField> DivAssign<f32> for $lhs {
                #[track_caller]
                fn div_assign(&mut self, other: f32) {
                    self.mul_assign(ScaleGeneric(LhsT::from_f64_impl(
                        &default(),
                        other.recip() as f64,
                    )))
                }
            }
        };
    }

    macro_rules! impl_sparse_mul {
        ($lhs: ty, $rhs: ty, $out: ty) => {
            impl<
                    I: Index,
                    E: ComplexField,
                    LhsE: Conjugate<Canonical = E>,
                    RhsE: Conjugate<Canonical = E>,
                > Mul<$rhs> for $lhs
            where
                E: ComplexField,
            {
                type Output = $out;
                #[track_caller]
                fn mul(self, other: $rhs) -> Self::Output {
                    self.as_ref().mul(other.as_ref())
                }
            }
        };
    }

    macro_rules! impl_partial_eq_sparse {
        ($lhs: ty, $rhs: ty) => {
            impl<I: Index, LhsE: Conjugate, RhsE: Conjugate<Canonical = LhsE::Canonical>>
                PartialEq<$rhs> for $lhs
            {
                fn eq(&self, other: &$rhs) -> bool {
                    self.as_ref().eq(&other.as_ref())
                }
            }
        };
    }

    macro_rules! impl_add_sub_sparse {
        ($lhs: ty, $rhs: ty, $out: ty) => {
            impl<
                    I: Index,
                    E: ComplexField,
                    LhsE: Conjugate<Canonical = E>,
                    RhsE: Conjugate<Canonical = E>,
                > Add<$rhs> for $lhs
            {
                type Output = $out;
                #[track_caller]
                fn add(self, other: $rhs) -> Self::Output {
                    self.as_ref().add(other.as_ref())
                }
            }

            impl<
                    I: Index,
                    E: ComplexField,
                    LhsE: Conjugate<Canonical = E>,
                    RhsE: Conjugate<Canonical = E>,
                > Sub<$rhs> for $lhs
            {
                type Output = $out;
                #[track_caller]
                fn sub(self, other: $rhs) -> Self::Output {
                    self.as_ref().sub(other.as_ref())
                }
            }
        };
    }

    macro_rules! impl_add_sub_assign_sparse {
        ($lhs: ty, $rhs: ty) => {
            impl<I: Index, LhsE: ComplexField, RhsE: Conjugate<Canonical = LhsC, LhsT>>
                AddAssign<$rhs> for $lhs
            {
                #[track_caller]
                fn add_assign(&mut self, other: $rhs) {
                    self.as_mut().add_assign(other.as_ref())
                }
            }

            impl<I: Index, LhsE: ComplexField, RhsE: Conjugate<Canonical = LhsC, LhsT>>
                SubAssign<$rhs> for $lhs
            {
                #[track_caller]
                fn sub_assign(&mut self, other: $rhs) {
                    self.as_mut().sub_assign(other.as_ref())
                }
            }
        };
    }

    macro_rules! impl_neg_sparse {
        ($mat: ty, $out: ty) => {
            impl<I: Index, E: Conjugate> Neg for $mat
            where
                E: ComplexField,
            {
                type Output = $out;
                #[track_caller]
                fn neg(self) -> Self::Output {
                    self.as_ref().neg()
                }
            }
        };
    }

    macro_rules! impl_scalar_mul_sparse {
        ($lhs: ty, $rhs: ty, $out: ty) => {
            impl<
                    I: Index,
                    E: ComplexField,
                    LhsE: Conjugate<Canonical = E>,
                    RhsE: Conjugate<Canonical = E>,
                > Mul<$rhs> for $lhs
            {
                type Output = $out;
                #[track_caller]
                fn mul(self, other: $rhs) -> Self::Output {
                    self.mul(other.as_ref())
                }
            }
        };
    }

    macro_rules! impl_mul_scalar_sparse {
        ($lhs: ty, $rhs: ty, $out: ty) => {
            impl<
                    I: Index,
                    E: ComplexField,
                    LhsE: Conjugate<Canonical = E>,
                    RhsE: Conjugate<Canonical = E>,
                > Mul<$rhs> for $lhs
            {
                type Output = $out;
                #[track_caller]
                fn mul(self, other: $rhs) -> Self::Output {
                    self.as_ref().mul(other)
                }
            }
        };
    }

    macro_rules! impl_mul_assign_scalar_sparse {
        ($lhs: ty, $rhs: ty) => {
            impl<I: Index, LhsE: ComplexField, RhsE: Conjugate<Canonical = LhsC, LhsT>>
                MulAssign<$rhs> for $lhs
            {
                #[track_caller]
                fn mul_assign(&mut self, other: $rhs) {
                    self.as_mut().mul_assign(other)
                }
            }
        };
    }

    macro_rules! impl_div_scalar_sparse {
        ($lhs: ty, $rhs: ty, $out: ty) => {
            impl<
                    I: Index,
                    E: ComplexField,
                    LhsE: Conjugate<Canonical = E>,
                    RhsE: Conjugate<Canonical = E>,
                > Div<$rhs> for $lhs
            {
                type Output = $out;
                #[track_caller]
                fn div(self, other: $rhs) -> Self::Output {
                    self.as_ref()
                        .mul(ScaleGeneric(other.0.canonicalize().faer_inv()))
                }
            }
        };
    }

    macro_rules! impl_div_assign_scalar_sparse {
        ($lhs: ty, $rhs: ty) => {
            impl<I: Index, LhsE: ComplexField, RhsE: Conjugate<Canonical = LhsC, LhsT>>
                DivAssign<$rhs> for $lhs
            {
                #[track_caller]
                fn div_assign(&mut self, other: $rhs) {
                    self.as_mut()
                        .mul_assign(ScaleGeneric(other.0.canonicalize().faer_inv()))
                }
            }
        };
    }

    use super::*;
    use crate::sparse::*;
    impl<
            I: Index,
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        > Mul<MatRef<'_, RhsC, RhsT>> for SparseColMatRef<'_, I, LhsC, LhsT>
    {
        type Output = Mat<C, T>;

        #[track_caller]
        fn mul(self, rhs: MatRef<'_, RhsC, RhsT>) -> Self::Output {
            let lhs = self;
            let mut out = Mat::zeros(lhs.nrows(), rhs.ncols());
            crate::sparse::linalg::matmul::sparse_dense_matmul(
                out.as_mut(),
                lhs,
                rhs,
                None,
                E::faer_one(),
                get_global_parallelism(),
            );
            out
        }
    }

    impl<
            I: Index,
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        > Mul<ColRef<'_, RhsC, RhsT>> for SparseColMatRef<'_, I, LhsC, LhsT>
    {
        type Output = Col<C, T>;

        #[track_caller]
        fn mul(self, rhs: ColRef<'_, RhsC, RhsT>) -> Self::Output {
            let lhs = self;
            let mut out = Col::zeros(lhs.nrows());
            crate::sparse::linalg::matmul::sparse_dense_matmul(
                out.as_mut(),
                lhs,
                rhs,
                None,
                E::faer_one(),
                get_global_parallelism(),
            );
            out
        }
    }

    impl<
            I: Index,
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        > Mul<SparseColMatRef<'_, I, RhsC, RhsT>> for MatRef<'_, LhsC, LhsT>
    {
        type Output = Mat<C, T>;

        #[track_caller]
        fn mul(self, rhs: SparseColMatRef<'_, I, RhsC, RhsT>) -> Self::Output {
            let lhs = self;
            let mut out = Mat::zeros(lhs.nrows(), rhs.ncols());
            crate::sparse::linalg::matmul::dense_sparse_matmul(
                out.as_mut(),
                lhs,
                rhs,
                None,
                E::faer_one(),
                get_global_parallelism(),
            );
            out
        }
    }

    impl<
            I: Index,
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        > Mul<SparseColMatRef<'_, I, RhsC, RhsT>> for RowRef<'_, LhsC, LhsT>
    {
        type Output = Row<C, T>;

        #[track_caller]
        fn mul(self, rhs: SparseColMatRef<'_, I, RhsC, RhsT>) -> Self::Output {
            let lhs = self;
            let mut out = Row::zeros(rhs.ncols());
            crate::sparse::linalg::matmul::dense_sparse_matmul(
                out.as_mut(),
                lhs,
                rhs,
                None,
                E::faer_one(),
                get_global_parallelism(),
            );
            out
        }
    }

    impl<
            I: Index,
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        > Mul<MatRef<'_, RhsC, RhsT>> for SparseRowMatRef<'_, I, LhsC, LhsT>
    {
        type Output = Mat<C, T>;

        #[track_caller]
        fn mul(self, rhs: MatRef<'_, RhsC, RhsT>) -> Self::Output {
            let lhs = self;
            let mut out = Mat::zeros(lhs.nrows(), rhs.ncols());
            crate::sparse::linalg::matmul::dense_sparse_matmul(
                out.as_mut().transpose_mut(),
                rhs.transpose(),
                lhs.transpose(),
                None,
                E::faer_one(),
                get_global_parallelism(),
            );
            out
        }
    }

    impl<
            I: Index,
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        > Mul<ColRef<'_, RhsC, RhsT>> for SparseRowMatRef<'_, I, LhsC, LhsT>
    {
        type Output = Col<C, T>;

        #[track_caller]
        fn mul(self, rhs: ColRef<'_, RhsC, RhsT>) -> Self::Output {
            let lhs = self;
            let mut out = Col::zeros(lhs.nrows());
            crate::sparse::linalg::matmul::dense_sparse_matmul(
                out.as_mut().transpose_mut(),
                rhs.transpose(),
                lhs.transpose(),
                None,
                E::faer_one(),
                get_global_parallelism(),
            );
            out
        }
    }

    impl<
            I: Index,
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        > Mul<SparseRowMatRef<'_, I, RhsC, RhsT>> for MatRef<'_, LhsC, LhsT>
    {
        type Output = Mat<C, T>;

        #[track_caller]
        fn mul(self, rhs: SparseRowMatRef<'_, I, RhsC, RhsT>) -> Self::Output {
            let lhs = self;
            let mut out = Mat::zeros(lhs.nrows(), rhs.ncols());
            crate::sparse::linalg::matmul::sparse_dense_matmul(
                out.as_mut().transpose_mut(),
                rhs.transpose(),
                lhs.transpose(),
                None,
                E::faer_one(),
                get_global_parallelism(),
            );
            out
        }
    }

    impl<
            I: Index,
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        > Mul<SparseRowMatRef<'_, I, RhsC, RhsT>> for RowRef<'_, LhsC, LhsT>
    {
        type Output = Row<C, T>;

        #[track_caller]
        fn mul(self, rhs: SparseRowMatRef<'_, I, RhsC, RhsT>) -> Self::Output {
            let lhs = self;
            let mut out = Row::zeros(rhs.ncols());
            crate::sparse::linalg::matmul::sparse_dense_matmul(
                out.as_mut().transpose_mut(),
                rhs.transpose(),
                lhs.transpose(),
                None,
                E::faer_one(),
                get_global_parallelism(),
            );
            out
        }
    }

    // impl_sparse_mul!(SparseColMatRef<'_, I, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        SparseColMatRef<'_, I, LhsC, LhsT>,
        MatMut<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(SparseColMatRef<'_, I, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        SparseColMatRef<'_, I, LhsC, LhsT>,
        &MatRef<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        SparseColMatRef<'_, I, LhsC, LhsT>,
        &MatMut<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(SparseColMatRef<'_, I, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        &SparseColMatRef<'_, I, LhsC, LhsT>,
        MatRef<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        &SparseColMatRef<'_, I, LhsC, LhsT>,
        MatMut<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(&SparseColMatRef<'_, I, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        &SparseColMatRef<'_, I, LhsC, LhsT>,
        &MatRef<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        &SparseColMatRef<'_, I, LhsC, LhsT>,
        &MatMut<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(&SparseColMatRef<'_, I, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);

    impl_sparse_mul!(
        SparseColMatMut<'_, I, LhsC, LhsT>,
        MatRef<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        SparseColMatMut<'_, I, LhsC, LhsT>,
        MatMut<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(SparseColMatMut<'_, I, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        SparseColMatMut<'_, I, LhsC, LhsT>,
        &MatRef<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        SparseColMatMut<'_, I, LhsC, LhsT>,
        &MatMut<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(SparseColMatMut<'_, I, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        &SparseColMatMut<'_, I, LhsC, LhsT>,
        MatRef<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        &SparseColMatMut<'_, I, LhsC, LhsT>,
        MatMut<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(&SparseColMatMut<'_, I, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        &SparseColMatMut<'_, I, LhsC, LhsT>,
        &MatRef<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        &SparseColMatMut<'_, I, LhsC, LhsT>,
        &MatMut<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(&SparseColMatMut<'_, I, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);

    impl_sparse_mul!(SparseColMat<I, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(SparseColMat<I, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(SparseColMat<I, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(SparseColMat<I, LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(SparseColMat<I, LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(SparseColMat<I, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&SparseColMat<I, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&SparseColMat<I, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&SparseColMat<I, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&SparseColMat<I, LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&SparseColMat<I, LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&SparseColMat<I, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);

    // impl_sparse_mul!(SparseRowMatRef<'_, I, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        SparseRowMatRef<'_, I, LhsC, LhsT>,
        MatMut<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(SparseRowMatRef<'_, I, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        SparseRowMatRef<'_, I, LhsC, LhsT>,
        &MatRef<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        SparseRowMatRef<'_, I, LhsC, LhsT>,
        &MatMut<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(SparseRowMatRef<'_, I, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        &SparseRowMatRef<'_, I, LhsC, LhsT>,
        MatRef<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        &SparseRowMatRef<'_, I, LhsC, LhsT>,
        MatMut<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(&SparseRowMatRef<'_, I, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        &SparseRowMatRef<'_, I, LhsC, LhsT>,
        &MatRef<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        &SparseRowMatRef<'_, I, LhsC, LhsT>,
        &MatMut<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(&SparseRowMatRef<'_, I, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);

    impl_sparse_mul!(
        SparseRowMatMut<'_, I, LhsC, LhsT>,
        MatRef<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        SparseRowMatMut<'_, I, LhsC, LhsT>,
        MatMut<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(SparseRowMatMut<'_, I, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        SparseRowMatMut<'_, I, LhsC, LhsT>,
        &MatRef<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        SparseRowMatMut<'_, I, LhsC, LhsT>,
        &MatMut<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(SparseRowMatMut<'_, I, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        &SparseRowMatMut<'_, I, LhsC, LhsT>,
        MatRef<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        &SparseRowMatMut<'_, I, LhsC, LhsT>,
        MatMut<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(&SparseRowMatMut<'_, I, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        &SparseRowMatMut<'_, I, LhsC, LhsT>,
        &MatRef<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        &SparseRowMatMut<'_, I, LhsC, LhsT>,
        &MatMut<'_, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(&SparseRowMatMut<'_, I, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);

    impl_sparse_mul!(SparseRowMat<I, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(SparseRowMat<I, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(SparseRowMat<I, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(SparseRowMat<I, LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(SparseRowMat<I, LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(SparseRowMat<I, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&SparseRowMat<I, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&SparseRowMat<I, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&SparseRowMat<I, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&SparseRowMat<I, LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&SparseRowMat<I, LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&SparseRowMat<I, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);

    // impl_sparse_mul!(SparseColMatRef<'_, I, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(
        SparseColMatRef<'_, I, LhsC, LhsT>,
        ColMut<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(SparseColMatRef<'_, I, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(
        SparseColMatRef<'_, I, LhsC, LhsT>,
        &ColRef<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(
        SparseColMatRef<'_, I, LhsC, LhsT>,
        &ColMut<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(SparseColMatRef<'_, I, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(
        &SparseColMatRef<'_, I, LhsC, LhsT>,
        ColRef<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(
        &SparseColMatRef<'_, I, LhsC, LhsT>,
        ColMut<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(&SparseColMatRef<'_, I, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(
        &SparseColMatRef<'_, I, LhsC, LhsT>,
        &ColRef<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(
        &SparseColMatRef<'_, I, LhsC, LhsT>,
        &ColMut<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(&SparseColMatRef<'_, I, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);

    impl_sparse_mul!(
        SparseColMatMut<'_, I, LhsC, LhsT>,
        ColRef<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(
        SparseColMatMut<'_, I, LhsC, LhsT>,
        ColMut<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(SparseColMatMut<'_, I, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(
        SparseColMatMut<'_, I, LhsC, LhsT>,
        &ColRef<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(
        SparseColMatMut<'_, I, LhsC, LhsT>,
        &ColMut<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(SparseColMatMut<'_, I, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(
        &SparseColMatMut<'_, I, LhsC, LhsT>,
        ColRef<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(
        &SparseColMatMut<'_, I, LhsC, LhsT>,
        ColMut<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(&SparseColMatMut<'_, I, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(
        &SparseColMatMut<'_, I, LhsC, LhsT>,
        &ColRef<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(
        &SparseColMatMut<'_, I, LhsC, LhsT>,
        &ColMut<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(&SparseColMatMut<'_, I, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);

    impl_sparse_mul!(SparseColMat<I, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(SparseColMat<I, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(SparseColMat<I, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(SparseColMat<I, LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(SparseColMat<I, LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(SparseColMat<I, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(&SparseColMat<I, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(&SparseColMat<I, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(&SparseColMat<I, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(&SparseColMat<I, LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(&SparseColMat<I, LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(&SparseColMat<I, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);

    // impl_sparse_mul!(SparseRowMatRef<'_, I, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(
        SparseRowMatRef<'_, I, LhsC, LhsT>,
        ColMut<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(SparseRowMatRef<'_, I, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(
        SparseRowMatRef<'_, I, LhsC, LhsT>,
        &ColRef<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(
        SparseRowMatRef<'_, I, LhsC, LhsT>,
        &ColMut<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(SparseRowMatRef<'_, I, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(
        &SparseRowMatRef<'_, I, LhsC, LhsT>,
        ColRef<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(
        &SparseRowMatRef<'_, I, LhsC, LhsT>,
        ColMut<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(&SparseRowMatRef<'_, I, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(
        &SparseRowMatRef<'_, I, LhsC, LhsT>,
        &ColRef<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(
        &SparseRowMatRef<'_, I, LhsC, LhsT>,
        &ColMut<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(&SparseRowMatRef<'_, I, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);

    impl_sparse_mul!(
        SparseRowMatMut<'_, I, LhsC, LhsT>,
        ColRef<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(
        SparseRowMatMut<'_, I, LhsC, LhsT>,
        ColMut<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(SparseRowMatMut<'_, I, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(
        SparseRowMatMut<'_, I, LhsC, LhsT>,
        &ColRef<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(
        SparseRowMatMut<'_, I, LhsC, LhsT>,
        &ColMut<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(SparseRowMatMut<'_, I, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(
        &SparseRowMatMut<'_, I, LhsC, LhsT>,
        ColRef<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(
        &SparseRowMatMut<'_, I, LhsC, LhsT>,
        ColMut<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(&SparseRowMatMut<'_, I, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(
        &SparseRowMatMut<'_, I, LhsC, LhsT>,
        &ColRef<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(
        &SparseRowMatMut<'_, I, LhsC, LhsT>,
        &ColMut<'_, RhsC, RhsT>,
        Col<C, T>
    );
    impl_sparse_mul!(&SparseRowMatMut<'_, I, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);

    impl_sparse_mul!(SparseRowMat<I, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(SparseRowMat<I, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(SparseRowMat<I, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(SparseRowMat<I, LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(SparseRowMat<I, LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(SparseRowMat<I, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(&SparseRowMat<I, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(&SparseRowMat<I, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(&SparseRowMat<I, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(&SparseRowMat<I, LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(&SparseRowMat<I, LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_sparse_mul!(&SparseRowMat<I, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);

    // impl_sparse_mul!(MatRef<'_, LhsC, LhsT>, SparseColMatRef<'_, I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        MatRef<'_, LhsC, LhsT>,
        SparseColMatMut<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(MatRef<'_, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        MatRef<'_, LhsC, LhsT>,
        &SparseColMatRef<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        MatRef<'_, LhsC, LhsT>,
        &SparseColMatMut<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(MatRef<'_, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        &MatRef<'_, LhsC, LhsT>,
        SparseColMatRef<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        &MatRef<'_, LhsC, LhsT>,
        SparseColMatMut<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(&MatRef<'_, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        &MatRef<'_, LhsC, LhsT>,
        &SparseColMatRef<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        &MatRef<'_, LhsC, LhsT>,
        &SparseColMatMut<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(&MatRef<'_, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, Mat<C, T>);

    impl_sparse_mul!(
        MatMut<'_, LhsC, LhsT>,
        SparseColMatRef<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        MatMut<'_, LhsC, LhsT>,
        SparseColMatMut<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(MatMut<'_, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        MatMut<'_, LhsC, LhsT>,
        &SparseColMatRef<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        MatMut<'_, LhsC, LhsT>,
        &SparseColMatMut<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(MatMut<'_, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        &MatMut<'_, LhsC, LhsT>,
        SparseColMatRef<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        &MatMut<'_, LhsC, LhsT>,
        SparseColMatMut<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(&MatMut<'_, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        &MatMut<'_, LhsC, LhsT>,
        &SparseColMatRef<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        &MatMut<'_, LhsC, LhsT>,
        &SparseColMatMut<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(&MatMut<'_, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, Mat<C, T>);

    impl_sparse_mul!(Mat<LhsC, LhsT>, SparseColMatRef<'_, I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(Mat<LhsC, LhsT>, SparseColMatMut<'_, I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(Mat< LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(Mat<LhsC, LhsT>, &SparseColMatRef<'_, I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(Mat<LhsC, LhsT>, &SparseColMatMut<'_, I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(Mat<LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&Mat<LhsC, LhsT>, SparseColMatRef<'_, I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&Mat<LhsC, LhsT>, SparseColMatMut<'_, I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&Mat<LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&Mat<LhsC, LhsT>, &SparseColMatRef<'_, I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&Mat<LhsC, LhsT>, &SparseColMatMut<'_, I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&Mat<LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, Mat<C, T>);

    // impl_sparse_mul!(RowRef<'_, LhsC, LhsT>, SparseColMatRef<'_, I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(
        RowRef<'_, LhsC, LhsT>,
        SparseColMatMut<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(RowRef<'_, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(
        RowRef<'_, LhsC, LhsT>,
        &SparseColMatRef<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(
        RowRef<'_, LhsC, LhsT>,
        &SparseColMatMut<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(RowRef<'_, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(
        &RowRef<'_, LhsC, LhsT>,
        SparseColMatRef<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(
        &RowRef<'_, LhsC, LhsT>,
        SparseColMatMut<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(&RowRef<'_, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(
        &RowRef<'_, LhsC, LhsT>,
        &SparseColMatRef<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(
        &RowRef<'_, LhsC, LhsT>,
        &SparseColMatMut<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(&RowRef<'_, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, Row<C, T>);

    impl_sparse_mul!(
        RowMut<'_, LhsC, LhsT>,
        SparseColMatRef<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(
        RowMut<'_, LhsC, LhsT>,
        SparseColMatMut<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(RowMut<'_, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(
        RowMut<'_, LhsC, LhsT>,
        &SparseColMatRef<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(
        RowMut<'_, LhsC, LhsT>,
        &SparseColMatMut<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(RowMut<'_, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(
        &RowMut<'_, LhsC, LhsT>,
        SparseColMatRef<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(
        &RowMut<'_, LhsC, LhsT>,
        SparseColMatMut<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(&RowMut<'_, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(
        &RowMut<'_, LhsC, LhsT>,
        &SparseColMatRef<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(
        &RowMut<'_, LhsC, LhsT>,
        &SparseColMatMut<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(&RowMut<'_, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, Row<C, T>);

    impl_sparse_mul!(Row<LhsC, LhsT>, SparseColMatRef<'_, I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(Row<LhsC, LhsT>, SparseColMatMut<'_, I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(Row<LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(Row<LhsC, LhsT>, &SparseColMatRef<'_, I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(Row<LhsC, LhsT>, &SparseColMatMut<'_, I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(Row<LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(&Row<LhsC, LhsT>, SparseColMatRef<'_, I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(&Row<LhsC, LhsT>, SparseColMatMut<'_, I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(&Row<LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(&Row<LhsC, LhsT>, &SparseColMatRef<'_, I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(&Row<LhsC, LhsT>, &SparseColMatMut<'_, I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(&Row<LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, Row<C, T>);

    // impl_sparse_mul!(MatRef<'_, LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        MatRef<'_, LhsC, LhsT>,
        SparseRowMatMut<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(MatRef<'_, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        MatRef<'_, LhsC, LhsT>,
        &SparseRowMatRef<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        MatRef<'_, LhsC, LhsT>,
        &SparseRowMatMut<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(MatRef<'_, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        &MatRef<'_, LhsC, LhsT>,
        SparseRowMatRef<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        &MatRef<'_, LhsC, LhsT>,
        SparseRowMatMut<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(&MatRef<'_, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        &MatRef<'_, LhsC, LhsT>,
        &SparseRowMatRef<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        &MatRef<'_, LhsC, LhsT>,
        &SparseRowMatMut<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(&MatRef<'_, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, Mat<C, T>);

    impl_sparse_mul!(
        MatMut<'_, LhsC, LhsT>,
        SparseRowMatRef<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        MatMut<'_, LhsC, LhsT>,
        SparseRowMatMut<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(MatMut<'_, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        MatMut<'_, LhsC, LhsT>,
        &SparseRowMatRef<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        MatMut<'_, LhsC, LhsT>,
        &SparseRowMatMut<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(MatMut<'_, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        &MatMut<'_, LhsC, LhsT>,
        SparseRowMatRef<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        &MatMut<'_, LhsC, LhsT>,
        SparseRowMatMut<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(&MatMut<'_, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(
        &MatMut<'_, LhsC, LhsT>,
        &SparseRowMatRef<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(
        &MatMut<'_, LhsC, LhsT>,
        &SparseRowMatMut<'_, I, RhsC, RhsT>,
        Mat<C, T>
    );
    impl_sparse_mul!(&MatMut<'_, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, Mat<C, T>);

    impl_sparse_mul!(Mat<LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(Mat<LhsC, LhsT>, SparseRowMatMut<'_, I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(Mat< LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(Mat<LhsC, LhsT>, &SparseRowMatRef<'_, I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(Mat<LhsC, LhsT>, &SparseRowMatMut<'_, I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(Mat<LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&Mat<LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&Mat<LhsC, LhsT>, SparseRowMatMut<'_, I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&Mat<LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&Mat<LhsC, LhsT>, &SparseRowMatRef<'_, I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&Mat<LhsC, LhsT>, &SparseRowMatMut<'_, I, RhsC, RhsT>, Mat<C, T>);
    impl_sparse_mul!(&Mat<LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, Mat<C, T>);

    // impl_sparse_mul!(RowRef<'_, LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(
        RowRef<'_, LhsC, LhsT>,
        SparseRowMatMut<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(RowRef<'_, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(
        RowRef<'_, LhsC, LhsT>,
        &SparseRowMatRef<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(
        RowRef<'_, LhsC, LhsT>,
        &SparseRowMatMut<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(RowRef<'_, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(
        &RowRef<'_, LhsC, LhsT>,
        SparseRowMatRef<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(
        &RowRef<'_, LhsC, LhsT>,
        SparseRowMatMut<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(&RowRef<'_, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(
        &RowRef<'_, LhsC, LhsT>,
        &SparseRowMatRef<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(
        &RowRef<'_, LhsC, LhsT>,
        &SparseRowMatMut<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(&RowRef<'_, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, Row<C, T>);

    impl_sparse_mul!(
        RowMut<'_, LhsC, LhsT>,
        SparseRowMatRef<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(
        RowMut<'_, LhsC, LhsT>,
        SparseRowMatMut<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(RowMut<'_, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(
        RowMut<'_, LhsC, LhsT>,
        &SparseRowMatRef<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(
        RowMut<'_, LhsC, LhsT>,
        &SparseRowMatMut<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(RowMut<'_, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(
        &RowMut<'_, LhsC, LhsT>,
        SparseRowMatRef<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(
        &RowMut<'_, LhsC, LhsT>,
        SparseRowMatMut<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(&RowMut<'_, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(
        &RowMut<'_, LhsC, LhsT>,
        &SparseRowMatRef<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(
        &RowMut<'_, LhsC, LhsT>,
        &SparseRowMatMut<'_, I, RhsC, RhsT>,
        Row<C, T>
    );
    impl_sparse_mul!(&RowMut<'_, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, Row<C, T>);

    impl_sparse_mul!(Row<LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(Row<LhsC, LhsT>, SparseRowMatMut<'_, I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(Row<LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(Row<LhsC, LhsT>, &SparseRowMatRef<'_, I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(Row<LhsC, LhsT>, &SparseRowMatMut<'_, I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(Row<LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(&Row<LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(&Row<LhsC, LhsT>, SparseRowMatMut<'_, I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(&Row<LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(&Row<LhsC, LhsT>, &SparseRowMatRef<'_, I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(&Row<LhsC, LhsT>, &SparseRowMatMut<'_, I, RhsC, RhsT>, Row<C, T>);
    impl_sparse_mul!(&Row<LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, Row<C, T>);

    impl<I: Index, LhsE: Conjugate, RhsE: Conjugate<Canonical = LhsE::Canonical>>
        PartialEq<SparseColMatRef<'_, I, RhsC, RhsT>> for SparseColMatRef<'_, I, LhsC, LhsT>
    {
        fn eq(&self, other: &SparseColMatRef<'_, I, RhsC, RhsT>) -> bool {
            let lhs = *self;
            let rhs = *other;

            if lhs.nrows() != rhs.nrows() || lhs.ncols() != rhs.ncols() {
                return false;
            }

            let n = lhs.ncols();
            let mut equal = true;
            for j in 0..n {
                equal &= lhs.row_indices_of_col_raw(j) == rhs.row_indices_of_col_raw(j);
                if !equal {
                    return false;
                }

                let lhs_val =
                    crate::utils::slice::SliceGroup::<'_, LhsC, LhsT>::new(lhs.values_of_col(j));
                let rhs_val =
                    crate::utils::slice::SliceGroup::<'_, RhsC, RhsT>::new(rhs.values_of_col(j));
                equal &= lhs_val
                    .into_ref_iter()
                    .map(|r| r.read().canonicalize())
                    .eq(rhs_val.into_ref_iter().map(|r| r.read().canonicalize()));

                if !equal {
                    return false;
                }
            }

            equal
        }
    }

    impl<I: Index, LhsE: Conjugate, RhsE: Conjugate<Canonical = LhsE::Canonical>>
        PartialEq<SparseRowMatRef<'_, I, RhsC, RhsT>> for SparseRowMatRef<'_, I, LhsC, LhsT>
    {
        #[inline]
        fn eq(&self, other: &SparseRowMatRef<'_, I, RhsC, RhsT>) -> bool {
            self.transpose() == other.transpose()
        }
    }

    // impl_partial_eq_sparse!(SparseColMatRef<'_, I, LhsC, LhsT>, SparseColMatRef<'_, I, RhsC,
    // RhsT>);
    impl_partial_eq_sparse!(
        SparseColMatRef<'_, I, LhsC, LhsT>,
        SparseColMatMut<'_, I, RhsC, RhsT>
    );
    impl_partial_eq_sparse!(SparseColMatRef<'_, I, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>);
    impl_partial_eq_sparse!(
        SparseColMatMut<'_, I, LhsC, LhsT>,
        SparseColMatRef<'_, I, RhsC, RhsT>
    );
    impl_partial_eq_sparse!(
        SparseColMatMut<'_, I, LhsC, LhsT>,
        SparseColMatMut<'_, I, RhsC, RhsT>
    );
    impl_partial_eq_sparse!(SparseColMatMut<'_, I, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>);
    impl_partial_eq_sparse!(SparseColMat<I, LhsC, LhsT>, SparseColMatRef<'_, I, RhsC, RhsT>);
    impl_partial_eq_sparse!(SparseColMat<I, LhsC, LhsT>, SparseColMatMut<'_, I, RhsC, RhsT>);
    impl_partial_eq_sparse!(SparseColMat<I, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>);

    // impl_partial_eq_sparse!(SparseRowMatRef<'_, I, LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC,
    // RhsT>);
    impl_partial_eq_sparse!(
        SparseRowMatRef<'_, I, LhsC, LhsT>,
        SparseRowMatMut<'_, I, RhsC, RhsT>
    );
    impl_partial_eq_sparse!(SparseRowMatRef<'_, I, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>);
    impl_partial_eq_sparse!(
        SparseRowMatMut<'_, I, LhsC, LhsT>,
        SparseRowMatRef<'_, I, RhsC, RhsT>
    );
    impl_partial_eq_sparse!(
        SparseRowMatMut<'_, I, LhsC, LhsT>,
        SparseRowMatMut<'_, I, RhsC, RhsT>
    );
    impl_partial_eq_sparse!(SparseRowMatMut<'_, I, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>);
    impl_partial_eq_sparse!(SparseRowMat<I, LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC, RhsT>);
    impl_partial_eq_sparse!(SparseRowMat<I, LhsC, LhsT>, SparseRowMatMut<'_, I, RhsC, RhsT>);
    impl_partial_eq_sparse!(SparseRowMat<I, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>);

    impl<
            I: Index,
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        > Add<SparseColMatRef<'_, I, RhsC, RhsT>> for SparseColMatRef<'_, I, LhsC, LhsT>
    {
        type Output = SparseColMat<I, C, T>;
        #[track_caller]
        fn add(self, rhs: SparseColMatRef<'_, I, RhsC, RhsT>) -> Self::Output {
            crate::sparse::ops::add(self, rhs).unwrap()
        }
    }

    impl<
            I: Index,
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        > Sub<SparseColMatRef<'_, I, RhsC, RhsT>> for SparseColMatRef<'_, I, LhsC, LhsT>
    {
        type Output = SparseColMat<I, C, T>;
        #[track_caller]
        fn sub(self, rhs: SparseColMatRef<'_, I, RhsC, RhsT>) -> Self::Output {
            crate::sparse::ops::sub(self, rhs).unwrap()
        }
    }

    impl<
            I: Index,
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        > Add<SparseRowMatRef<'_, I, RhsC, RhsT>> for SparseRowMatRef<'_, I, LhsC, LhsT>
    {
        type Output = SparseRowMat<I, C, T>;
        #[track_caller]
        fn add(self, rhs: SparseRowMatRef<'_, I, RhsC, RhsT>) -> Self::Output {
            (self.transpose() + rhs.transpose()).into_transpose()
        }
    }

    impl<
            I: Index,
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        > Sub<SparseRowMatRef<'_, I, RhsC, RhsT>> for SparseRowMatRef<'_, I, LhsC, LhsT>
    {
        type Output = SparseRowMat<I, C, T>;
        #[track_caller]
        fn sub(self, rhs: SparseRowMatRef<'_, I, RhsC, RhsT>) -> Self::Output {
            (self.transpose() - rhs.transpose()).into_transpose()
        }
    }

    impl<I: Index, LhsE: ComplexField, RhsE: Conjugate<Canonical = LhsC, LhsT>>
        AddAssign<SparseColMatRef<'_, I, RhsC, RhsT>> for SparseColMatMut<'_, I, LhsC, LhsT>
    {
        #[track_caller]
        fn add_assign(&mut self, other: SparseColMatRef<'_, I, RhsC, RhsT>) {
            crate::sparse::ops::add_assign(self.as_mut(), other);
        }
    }

    impl<I: Index, LhsE: ComplexField, RhsE: Conjugate<Canonical = LhsC, LhsT>>
        SubAssign<SparseColMatRef<'_, I, RhsC, RhsT>> for SparseColMatMut<'_, I, LhsC, LhsT>
    {
        #[track_caller]
        fn sub_assign(&mut self, other: SparseColMatRef<'_, I, RhsC, RhsT>) {
            crate::sparse::ops::sub_assign(self.as_mut(), other);
        }
    }

    impl<I: Index, LhsE: ComplexField, RhsE: Conjugate<Canonical = LhsC, LhsT>>
        AddAssign<SparseRowMatRef<'_, I, RhsC, RhsT>> for SparseRowMatMut<'_, I, LhsC, LhsT>
    {
        #[track_caller]
        fn add_assign(&mut self, other: SparseRowMatRef<'_, I, RhsC, RhsT>) {
            crate::sparse::ops::add_assign(self.as_mut().transpose_mut(), other.transpose());
        }
    }

    impl<I: Index, LhsE: ComplexField, RhsE: Conjugate<Canonical = LhsC, LhsT>>
        SubAssign<SparseRowMatRef<'_, I, RhsC, RhsT>> for SparseRowMatMut<'_, I, LhsC, LhsT>
    {
        #[track_caller]
        fn sub_assign(&mut self, other: SparseRowMatRef<'_, I, RhsC, RhsT>) {
            crate::sparse::ops::sub_assign(self.as_mut().transpose_mut(), other.transpose());
        }
    }

    impl<I: Index, E: Conjugate> Neg for SparseColMatRef<'_, I, C, T>
    where
        E: ComplexField,
    {
        type Output = SparseColMat<I, CC, TT>;
        #[track_caller]
        fn neg(self) -> Self::Output {
            let mut out = self.to_owned().unwrap();
            for mut x in crate::utils::slice::SliceGroupMut::<'_, CC, TT>::new(out.values_mut())
                .into_mut_iter()
            {
                x.write(x.read().faer_neg())
            }
            out
        }
    }
    impl<I: Index, E: Conjugate> Neg for SparseRowMatRef<'_, I, C, T>
    where
        E: ComplexField,
    {
        type Output = SparseRowMat<I, CC, TT>;
        #[track_caller]
        fn neg(self) -> Self::Output {
            (-self.transpose()).into_transpose()
        }
    }

    impl<
            I: Index,
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        > Mul<SparseColMatRef<'_, I, RhsC, RhsT>> for ScaleGeneric<LhsC, LhsT>
    {
        type Output = SparseColMat<I, C, T>;
        #[track_caller]
        fn mul(self, rhs: SparseColMatRef<'_, I, RhsC, RhsT>) -> Self::Output {
            let mut out = rhs.to_owned().unwrap();
            for mut x in crate::utils::slice::SliceGroupMut::<'_, C, T>::new(out.values_mut())
                .into_mut_iter()
            {
                x.write(self.0.canonicalize().faer_mul(x.read()))
            }
            out
        }
    }

    impl<
            I: Index,
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        > Mul<ScaleGeneric<RhsC, RhsT>> for SparseColMatRef<'_, I, LhsC, LhsT>
    {
        type Output = SparseColMat<I, C, T>;
        #[track_caller]
        fn mul(self, rhs: ScaleGeneric<RhsC, RhsT>) -> Self::Output {
            let mut out = self.to_owned().unwrap();
            let rhs = rhs.0.canonicalize();
            for mut x in crate::utils::slice::SliceGroupMut::<'_, C, T>::new(out.values_mut())
                .into_mut_iter()
            {
                x.write(x.read().faer_mul(rhs));
            }
            out
        }
    }

    impl<
            I: Index,
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        > Mul<SparseRowMatRef<'_, I, RhsC, RhsT>> for ScaleGeneric<LhsC, LhsT>
    {
        type Output = SparseRowMat<I, C, T>;
        #[track_caller]
        fn mul(self, rhs: SparseRowMatRef<'_, I, RhsC, RhsT>) -> Self::Output {
            self.mul(rhs.transpose()).into_transpose()
        }
    }

    impl<
            I: Index,
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        > Mul<ScaleGeneric<RhsC, RhsT>> for SparseRowMatRef<'_, I, LhsC, LhsT>
    {
        type Output = SparseRowMat<I, C, T>;
        #[track_caller]
        fn mul(self, rhs: ScaleGeneric<RhsC, RhsT>) -> Self::Output {
            self.transpose().mul(rhs).into_transpose()
        }
    }

    impl<I: Index, LhsE: ComplexField, RhsE: Conjugate<Canonical = LhsC, LhsT>>
        MulAssign<ScaleGeneric<RhsC, RhsT>> for SparseColMatMut<'_, I, LhsC, LhsT>
    {
        #[track_caller]
        fn mul_assign(&mut self, rhs: ScaleGeneric<RhsC, RhsT>) {
            let rhs = rhs.0.canonicalize();
            for mut x in crate::utils::slice::SliceGroupMut::<'_, LhsC, LhsT>::new(
                self.as_mut().values_mut(),
            )
            .into_mut_iter()
            {
                x.write(x.read().faer_mul(rhs))
            }
        }
    }

    impl<I: Index, LhsE: ComplexField, RhsE: Conjugate<Canonical = LhsC, LhsT>>
        MulAssign<ScaleGeneric<RhsC, RhsT>> for SparseRowMatMut<'_, I, LhsC, LhsT>
    {
        #[track_caller]
        fn mul_assign(&mut self, rhs: ScaleGeneric<RhsC, RhsT>) {
            self.as_mut()
                .transpose_mut()
                .mul_assign(ScaleGeneric(rhs.0.canonicalize()));
        }
    }

    #[rustfmt::skip]
// impl_add_sub_sparse!(SparseColMatRef<'_, I, LhsC, LhsT>, SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
impl_add_sub_sparse!(SparseColMatRef<'_, I, LhsC, LhsT>, SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(SparseColMatRef<'_, I, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(SparseColMatRef<'_, I, LhsC, LhsT>, &SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(SparseColMatRef<'_, I, LhsC, LhsT>, &SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(SparseColMatRef<'_, I, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(SparseColMatMut<'_, I, LhsC, LhsT>, SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(SparseColMatMut<'_, I, LhsC, LhsT>, SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(SparseColMatMut<'_, I, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(SparseColMatMut<'_, I, LhsC, LhsT>, &SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(SparseColMatMut<'_, I, LhsC, LhsT>, &SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(SparseColMatMut<'_, I, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(SparseColMat<I, LhsC, LhsT>, SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(SparseColMat<I, LhsC, LhsT>, SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(SparseColMat<I, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(SparseColMat<I, LhsC, LhsT>, &SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(SparseColMat<I, LhsC, LhsT>, &SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(SparseColMat<I, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(&SparseColMatRef<'_, I, LhsC, LhsT>, SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(&SparseColMatRef<'_, I, LhsC, LhsT>, SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(&SparseColMatRef<'_, I, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(&SparseColMatRef<'_, I, LhsC, LhsT>, &SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(&SparseColMatRef<'_, I, LhsC, LhsT>, &SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(&SparseColMatRef<'_, I, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(&SparseColMatMut<'_, I, LhsC, LhsT>, SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(&SparseColMatMut<'_, I, LhsC, LhsT>, SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(&SparseColMatMut<'_, I, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(&SparseColMatMut<'_, I, LhsC, LhsT>, &SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(&SparseColMatMut<'_, I, LhsC, LhsT>, &SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(&SparseColMatMut<'_, I, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(&SparseColMat<I, LhsC, LhsT>, SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(&SparseColMat<I, LhsC, LhsT>, SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(&SparseColMat<I, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(&SparseColMat<I, LhsC, LhsT>, &SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(&SparseColMat<I, LhsC, LhsT>, &SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_add_sub_sparse!(&SparseColMat<I, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    #[rustfmt::skip]
// impl_add_sub_sparse!(SparseRowMatRef<'_, I, LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
impl_add_sub_sparse!(SparseRowMatRef<'_, I, LhsC, LhsT>, SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(SparseRowMatRef<'_, I, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(SparseRowMatRef<'_, I, LhsC, LhsT>, &SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(SparseRowMatRef<'_, I, LhsC, LhsT>, &SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(SparseRowMatRef<'_, I, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(SparseRowMatMut<'_, I, LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(SparseRowMatMut<'_, I, LhsC, LhsT>, SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(SparseRowMatMut<'_, I, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(SparseRowMatMut<'_, I, LhsC, LhsT>, &SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(SparseRowMatMut<'_, I, LhsC, LhsT>, &SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(SparseRowMatMut<'_, I, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(SparseRowMat<I, LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(SparseRowMat<I, LhsC, LhsT>, SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(SparseRowMat<I, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(SparseRowMat<I, LhsC, LhsT>, &SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(SparseRowMat<I, LhsC, LhsT>, &SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(SparseRowMat<I, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(&SparseRowMatRef<'_, I, LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(&SparseRowMatRef<'_, I, LhsC, LhsT>, SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(&SparseRowMatRef<'_, I, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(&SparseRowMatRef<'_, I, LhsC, LhsT>, &SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(&SparseRowMatRef<'_, I, LhsC, LhsT>, &SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(&SparseRowMatRef<'_, I, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(&SparseRowMatMut<'_, I, LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(&SparseRowMatMut<'_, I, LhsC, LhsT>, SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(&SparseRowMatMut<'_, I, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(&SparseRowMatMut<'_, I, LhsC, LhsT>, &SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(&SparseRowMatMut<'_, I, LhsC, LhsT>, &SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(&SparseRowMatMut<'_, I, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(&SparseRowMat<I, LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(&SparseRowMat<I, LhsC, LhsT>, SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(&SparseRowMat<I, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(&SparseRowMat<I, LhsC, LhsT>, &SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(&SparseRowMat<I, LhsC, LhsT>, &SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_add_sub_sparse!(&SparseRowMat<I, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);

    // impl_add_sub_assign_sparse!(SparseColMatMut<'_, I, LhsC, LhsT>, SparseColMatRef<'_, I,
    // RhsC, RhsT>);
    impl_add_sub_assign_sparse!(
        SparseColMatMut<'_, I, LhsC, LhsT>,
        SparseColMatMut<'_, I, RhsC, RhsT>
    );
    impl_add_sub_assign_sparse!(SparseColMatMut<'_, I, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>);
    impl_add_sub_assign_sparse!(
        SparseColMatMut<'_, I, LhsC, LhsT>,
        &SparseColMatRef<'_, I, RhsC, RhsT>
    );
    impl_add_sub_assign_sparse!(
        SparseColMatMut<'_, I, LhsC, LhsT>,
        &SparseColMatMut<'_, I, RhsC, RhsT>
    );
    impl_add_sub_assign_sparse!(SparseColMatMut<'_, I, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>);
    impl_add_sub_assign_sparse!(SparseColMat<I, LhsC, LhsT>, SparseColMatRef<'_, I, RhsC, RhsT>);
    impl_add_sub_assign_sparse!(SparseColMat<I, LhsC, LhsT>, SparseColMatMut<'_, I, RhsC, RhsT>);
    impl_add_sub_assign_sparse!(SparseColMat<I, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>);
    impl_add_sub_assign_sparse!(SparseColMat<I, LhsC, LhsT>, &SparseColMatRef<'_, I, RhsC, RhsT>);
    impl_add_sub_assign_sparse!(SparseColMat<I, LhsC, LhsT>, &SparseColMatMut<'_, I, RhsC, RhsT>);
    impl_add_sub_assign_sparse!(SparseColMat<I, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>);
    // impl_add_sub_assign_sparse!(SparseRowMatMut<'_, I, LhsC, LhsT>, SparseRowMatRef<'_, I,
    // RhsC, RhsT>);
    impl_add_sub_assign_sparse!(
        SparseRowMatMut<'_, I, LhsC, LhsT>,
        SparseRowMatMut<'_, I, RhsC, RhsT>
    );
    impl_add_sub_assign_sparse!(SparseRowMatMut<'_, I, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>);
    impl_add_sub_assign_sparse!(
        SparseRowMatMut<'_, I, LhsC, LhsT>,
        &SparseRowMatRef<'_, I, RhsC, RhsT>
    );
    impl_add_sub_assign_sparse!(
        SparseRowMatMut<'_, I, LhsC, LhsT>,
        &SparseRowMatMut<'_, I, RhsC, RhsT>
    );
    impl_add_sub_assign_sparse!(SparseRowMatMut<'_, I, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>);
    impl_add_sub_assign_sparse!(SparseRowMat<I, LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC, RhsT>);
    impl_add_sub_assign_sparse!(SparseRowMat<I, LhsC, LhsT>, SparseRowMatMut<'_, I, RhsC, RhsT>);
    impl_add_sub_assign_sparse!(SparseRowMat<I, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>);
    impl_add_sub_assign_sparse!(SparseRowMat<I, LhsC, LhsT>, &SparseRowMatRef<'_, I, RhsC, RhsT>);
    impl_add_sub_assign_sparse!(SparseRowMat<I, LhsC, LhsT>, &SparseRowMatMut<'_, I, RhsC, RhsT>);
    impl_add_sub_assign_sparse!(SparseRowMat<I, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>);

    // impl_neg_sparse!(SparseColMatRef<'_, I, C, T>, SparseColMat<I, CC, TT>);
    impl_neg_sparse!(SparseColMatMut<'_, I, C, T>, SparseColMat<I, CC, TT>);
    impl_neg_sparse!(SparseColMat<I, C, T>, SparseColMat<I, CC, TT>);
    impl_neg_sparse!(&SparseColMatRef<'_, I, C, T>, SparseColMat<I, CC, TT>);
    impl_neg_sparse!(&SparseColMatMut<'_, I, C, T>, SparseColMat<I, CC, TT>);
    impl_neg_sparse!(&SparseColMat<I, C, T>, SparseColMat<I, CC, TT>);
    // impl_neg_sparse!(SparseRowMatRef<'_, I, C, T>, SparseRowMat<I, CC, TT>);
    impl_neg_sparse!(SparseRowMatMut<'_, I, C, T>, SparseRowMat<I, CC, TT>);
    impl_neg_sparse!(SparseRowMat<I, C, T>, SparseRowMat<I, CC, TT>);
    impl_neg_sparse!(&SparseRowMatRef<'_, I, C, T>, SparseRowMat<I, CC, TT>);
    impl_neg_sparse!(&SparseRowMatMut<'_, I, C, T>, SparseRowMat<I, CC, TT>);
    impl_neg_sparse!(&SparseRowMat<I, C, T>, SparseRowMat<I, CC, TT>);

    // impl_scalar_mul_sparse!(ScaleGeneric<LhsC, LhsT>, SparseColMatRef<'_, I, RhsC, RhsT>,
    // SparseColMat<I, C, T>);
    impl_scalar_mul_sparse!(ScaleGeneric<LhsC, LhsT>, SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_scalar_mul_sparse!(ScaleGeneric<LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_scalar_mul_sparse!(ScaleGeneric<LhsC, LhsT>, &SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_scalar_mul_sparse!(ScaleGeneric<LhsC, LhsT>, &SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_scalar_mul_sparse!(ScaleGeneric<LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);

    impl_mul_primitive_sparse!(SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_mul_primitive_sparse!(SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_mul_primitive_sparse!(SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_mul_primitive_sparse!(&SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_mul_primitive_sparse!(&SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_mul_primitive_sparse!(&SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);

    // impl_scalar_mul_sparse!(ScaleGeneric<LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC, RhsT>,
    // SparseRowMat<I, C, T>);
    impl_scalar_mul_sparse!(ScaleGeneric<LhsC, LhsT>, SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_scalar_mul_sparse!(ScaleGeneric<LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_scalar_mul_sparse!(ScaleGeneric<LhsC, LhsT>, &SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_scalar_mul_sparse!(ScaleGeneric<LhsC, LhsT>, &SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_scalar_mul_sparse!(ScaleGeneric<LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);

    impl_mul_primitive_sparse!(SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_mul_primitive_sparse!(SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_mul_primitive_sparse!(SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_mul_primitive_sparse!(&SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_mul_primitive_sparse!(&SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_mul_primitive_sparse!(&SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);

    // impl_mul_scalar_sparse!(SparseColMatRef<'_, I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>,
    // SparseColMat<I, C, T>);
    impl_mul_scalar_sparse!(SparseColMatMut<'_, I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_mul_scalar_sparse!(SparseColMat<I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_mul_scalar_sparse!(&SparseColMatRef<'_, I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_mul_scalar_sparse!(&SparseColMatMut<'_, I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_mul_scalar_sparse!(&SparseColMat<I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseColMat<I, C, T>);

    impl_div_scalar_sparse!(SparseColMatRef<'_, I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_div_scalar_sparse!(SparseColMatMut<'_, I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_div_scalar_sparse!(SparseColMat<I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_div_scalar_sparse!(&SparseColMatRef<'_, I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_div_scalar_sparse!(&SparseColMatMut<'_, I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_div_scalar_sparse!(&SparseColMat<I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseColMat<I, C, T>);

    // impl_mul_assign_scalar_sparse!(SparseColMatMut<'_, I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);
    impl_mul_assign_scalar_sparse!(SparseColMat<I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);

    impl_div_assign_scalar_sparse!(SparseColMatMut<'_, I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);
    impl_div_assign_scalar_sparse!(SparseColMat<I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);

    impl_mul_assign_primitive_sparse!(SparseColMatMut<'_, I, LhsC, LhsT>);
    impl_mul_assign_primitive_sparse!(SparseColMat<I, LhsC, LhsT>);

    // impl_mul_scalar_sparse!(SparseRowMatRef<'_, I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>,
    // SparseRowMat<I, C, T>);
    impl_mul_scalar_sparse!(SparseRowMatMut<'_, I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_mul_scalar_sparse!(SparseRowMat<I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_mul_scalar_sparse!(&SparseRowMatRef<'_, I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_mul_scalar_sparse!(&SparseRowMatMut<'_, I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_mul_scalar_sparse!(&SparseRowMat<I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseRowMat<I, C, T>);

    impl_div_scalar_sparse!(SparseRowMatRef<'_, I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_div_scalar_sparse!(SparseRowMatMut<'_, I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_div_scalar_sparse!(SparseRowMat<I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_div_scalar_sparse!(&SparseRowMatRef<'_, I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_div_scalar_sparse!(&SparseRowMatMut<'_, I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_div_scalar_sparse!(&SparseRowMat<I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, SparseRowMat<I, C, T>);

    // impl_mul_assign_scalar_sparse!(SparseRowMatMut<'_, I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);
    impl_mul_assign_scalar_sparse!(SparseRowMat<I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);

    impl_div_assign_scalar_sparse!(SparseRowMatMut<'_, I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);
    impl_div_assign_scalar_sparse!(SparseRowMat<I, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);

    impl_mul_assign_primitive_sparse!(SparseRowMatMut<'_, I, LhsC, LhsT>);
    impl_mul_assign_primitive_sparse!(SparseRowMat<I, LhsC, LhsT>);

    impl<
            I: Index,
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        > Mul<SparseColMatRef<'_, I, RhsC, RhsT>> for SparseColMatRef<'_, I, LhsC, LhsT>
    {
        type Output = SparseColMat<I, C, T>;
        #[track_caller]
        fn mul(self, rhs: SparseColMatRef<'_, I, RhsC, RhsT>) -> Self::Output {
            crate::sparse::linalg::matmul::sparse_sparse_matmul(
                self,
                rhs,
                E::faer_one(),
                crate::get_global_parallelism(),
            )
            .unwrap()
        }
    }

    impl<
            I: Index,
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        > Mul<SparseRowMatRef<'_, I, RhsC, RhsT>> for SparseRowMatRef<'_, I, LhsC, LhsT>
    {
        type Output = SparseRowMat<I, C, T>;

        #[track_caller]
        fn mul(self, rhs: SparseRowMatRef<'_, I, RhsC, RhsT>) -> Self::Output {
            (rhs.transpose() * self.transpose()).into_transpose()
        }
    }

    #[rustfmt::skip]
// impl_sparse_mul!(SparseColMatRef<'_, I, LhsC, LhsT>, SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
impl_sparse_mul!(SparseColMatRef<'_, I, LhsC, LhsT>, SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(SparseColMatRef<'_, I, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(SparseColMatRef<'_, I, LhsC, LhsT>, &SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(SparseColMatRef<'_, I, LhsC, LhsT>, &SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(SparseColMatRef<'_, I, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(SparseColMatMut<'_, I, LhsC, LhsT>, SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(SparseColMatMut<'_, I, LhsC, LhsT>, SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(SparseColMatMut<'_, I, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(SparseColMatMut<'_, I, LhsC, LhsT>, &SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(SparseColMatMut<'_, I, LhsC, LhsT>, &SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(SparseColMatMut<'_, I, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(SparseColMat<I, LhsC, LhsT>, SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(SparseColMat<I, LhsC, LhsT>, SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(SparseColMat<I, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(SparseColMat<I, LhsC, LhsT>, &SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(SparseColMat<I, LhsC, LhsT>, &SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(SparseColMat<I, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(&SparseColMatRef<'_, I, LhsC, LhsT>, SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(&SparseColMatRef<'_, I, LhsC, LhsT>, SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(&SparseColMatRef<'_, I, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(&SparseColMatRef<'_, I, LhsC, LhsT>, &SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(&SparseColMatRef<'_, I, LhsC, LhsT>, &SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(&SparseColMatRef<'_, I, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(&SparseColMatMut<'_, I, LhsC, LhsT>, SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(&SparseColMatMut<'_, I, LhsC, LhsT>, SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(&SparseColMatMut<'_, I, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(&SparseColMatMut<'_, I, LhsC, LhsT>, &SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(&SparseColMatMut<'_, I, LhsC, LhsT>, &SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(&SparseColMatMut<'_, I, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(&SparseColMat<I, LhsC, LhsT>, SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(&SparseColMat<I, LhsC, LhsT>, SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(&SparseColMat<I, LhsC, LhsT>, SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(&SparseColMat<I, LhsC, LhsT>, &SparseColMatRef<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(&SparseColMat<I, LhsC, LhsT>, &SparseColMatMut<'_, I, RhsC, RhsT>, SparseColMat<I, C, T>);
    impl_sparse_mul!(&SparseColMat<I, LhsC, LhsT>, &SparseColMat<I, RhsC, RhsT>, SparseColMat<I, C, T>);
    #[rustfmt::skip]
// impl_sparse_mul!(SparseRowMatRef<'_, I, LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
impl_sparse_mul!(SparseRowMatRef<'_, I, LhsC, LhsT>, SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(SparseRowMatRef<'_, I, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(SparseRowMatRef<'_, I, LhsC, LhsT>, &SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(SparseRowMatRef<'_, I, LhsC, LhsT>, &SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(SparseRowMatRef<'_, I, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(SparseRowMatMut<'_, I, LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(SparseRowMatMut<'_, I, LhsC, LhsT>, SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(SparseRowMatMut<'_, I, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(SparseRowMatMut<'_, I, LhsC, LhsT>, &SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(SparseRowMatMut<'_, I, LhsC, LhsT>, &SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(SparseRowMatMut<'_, I, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(SparseRowMat<I, LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(SparseRowMat<I, LhsC, LhsT>, SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(SparseRowMat<I, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(SparseRowMat<I, LhsC, LhsT>, &SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(SparseRowMat<I, LhsC, LhsT>, &SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(SparseRowMat<I, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(&SparseRowMatRef<'_, I, LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(&SparseRowMatRef<'_, I, LhsC, LhsT>, SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(&SparseRowMatRef<'_, I, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(&SparseRowMatRef<'_, I, LhsC, LhsT>, &SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(&SparseRowMatRef<'_, I, LhsC, LhsT>, &SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(&SparseRowMatRef<'_, I, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(&SparseRowMatMut<'_, I, LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(&SparseRowMatMut<'_, I, LhsC, LhsT>, SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(&SparseRowMatMut<'_, I, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(&SparseRowMatMut<'_, I, LhsC, LhsT>, &SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(&SparseRowMatMut<'_, I, LhsC, LhsT>, &SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(&SparseRowMatMut<'_, I, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(&SparseRowMat<I, LhsC, LhsT>, SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(&SparseRowMat<I, LhsC, LhsT>, SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(&SparseRowMat<I, LhsC, LhsT>, SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(&SparseRowMat<I, LhsC, LhsT>, &SparseRowMatRef<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(&SparseRowMat<I, LhsC, LhsT>, &SparseRowMatMut<'_, I, RhsC, RhsT>, SparseRowMat<I, C, T>);
    impl_sparse_mul!(&SparseRowMat<I, LhsC, LhsT>, &SparseRowMat<I, RhsC, RhsT>, SparseRowMat<I, C, T>);
}
#[cfg(test)]
#[allow(non_snake_case)]
mod test {
    use crate::{assert, col::*, mat, mat::*, perm::*, row::*};
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
        use crate::scale;

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
        let pl = Perm::<usize>::new_checked(
            Box::new([5, 1, 4, 0, 2, 3]),
            Box::new([3, 1, 4, 5, 2, 0]),
            6,
        );
        let pr =
            Perm::<usize>::new_checked(Box::new([1, 4, 0, 2, 3]), Box::new([2, 0, 3, 4, 1]), 5);

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

        assert!(
            &pl * pl.as_ref().inverse()
                == PermRef::<'_, usize>::new_checked(&[0, 1, 2, 3, 4, 5], &[0, 1, 2, 3, 4, 5], 6)
        );
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
