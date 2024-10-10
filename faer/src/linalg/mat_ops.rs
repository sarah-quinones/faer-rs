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
        impl<LhsC: ComplexContainer, LhsT: ComplexField<LhsC, MathCtx: Default>> MulAssign<f64>
            for $lhs
        {
            #[track_caller]
            fn mul_assign(&mut self, other: f64) {
                self.mul_assign(ScaleGeneric::<LhsC, LhsT>(LhsT::from_f64_impl(
                    &default(),
                    other,
                )))
            }
        }
        impl<LhsC: ComplexContainer, LhsT: ComplexField<LhsC, MathCtx: Default>> DivAssign<f64>
            for $lhs
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
    > PartialEq<MatRef<'_, RhsC, RhsT>> for MatRef<'_, LhsC, LhsT>
{
    #[math]
    fn eq(&self, other: &MatRef<'_, RhsC, RhsT>) -> bool {
        let lhs = *self;
        let rhs = *other;

        if (lhs.nrows(), lhs.ncols()) != (rhs.nrows(), rhs.ncols()) {
            return false;
        }
        let ctx = &Ctx::<C, T>(T::MathCtx::default());

        let m = lhs.nrows();
        let n = lhs.ncols();
        for j in 0..n {
            for i in 0..m {
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
}

// impl_partial_eq!(MatRef<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>);
impl_partial_eq!(MatRef<'_, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>);
impl_partial_eq!(MatRef<'_, LhsC, LhsT>, Mat<RhsC, RhsT>);

impl_partial_eq!(MatMut<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>);
impl_partial_eq!(MatMut<'_, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>);
impl_partial_eq!(MatMut<'_, LhsC, LhsT>, Mat<RhsC, RhsT>);

impl_partial_eq!(Mat<LhsC, LhsT>, MatRef<'_, RhsC, RhsT>);
impl_partial_eq!(Mat<LhsC, LhsT>, MatMut<'_, RhsC, RhsT>);
impl_partial_eq!(Mat<LhsC, LhsT>, Mat<RhsC, RhsT>);

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
    > PartialEq<ColRef<'_, RhsC, RhsT>> for ColRef<'_, LhsC, LhsT>
{
    fn eq(&self, other: &ColRef<'_, RhsC, RhsT>) -> bool {
        self.transpose() == other.transpose()
    }
}

// impl_partial_eq!(ColRef<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>);
impl_partial_eq!(ColRef<'_, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>);
impl_partial_eq!(ColRef<'_, LhsC, LhsT>, Col<RhsC, RhsT>);

impl_partial_eq!(ColMut<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>);
impl_partial_eq!(ColMut<'_, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>);
impl_partial_eq!(ColMut<'_, LhsC, LhsT>, Col<RhsC, RhsT>);

impl_partial_eq!(Col<LhsC, LhsT>, ColRef<'_, RhsC, RhsT>);
impl_partial_eq!(Col<LhsC, LhsT>, ColMut<'_, RhsC, RhsT>);
impl_partial_eq!(Col<LhsC, LhsT>, Col<RhsC, RhsT>);

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
    > PartialEq<RowRef<'_, RhsC, RhsT>> for RowRef<'_, LhsC, LhsT>
{
    #[math]
    fn eq(&self, other: &RowRef<'_, RhsC, RhsT>) -> bool {
        let lhs = *self;
        let rhs = *other;

        if lhs.ncols() != rhs.ncols() {
            return false;
        }
        let ctx = &Ctx::<C, T>(T::MathCtx::default());

        let n = lhs.ncols();
        for j in 0..n {
            if !math(
                Conj::apply::<LhsC, LhsT>(ctx, lhs[j]) == Conj::apply::<RhsC, RhsT>(ctx, rhs[j]),
            ) {
                return false;
            }
        }

        true
    }
}

// impl_partial_eq!(RowRef<'_, LhsC, LhsT>, RowRef<'_, RhsC, RhsT>);
impl_partial_eq!(RowRef<'_, LhsC, LhsT>, RowMut<'_, RhsC, RhsT>);
impl_partial_eq!(RowRef<'_, LhsC, LhsT>, Row<RhsC, RhsT>);

impl_partial_eq!(RowMut<'_, LhsC, LhsT>, RowRef<'_, RhsC, RhsT>);
impl_partial_eq!(RowMut<'_, LhsC, LhsT>, RowMut<'_, RhsC, RhsT>);
impl_partial_eq!(RowMut<'_, LhsC, LhsT>, Row<RhsC, RhsT>);

impl_partial_eq!(Row<LhsC, LhsT>, RowRef<'_, RhsC, RhsT>);
impl_partial_eq!(Row<LhsC, LhsT>, RowMut<'_, RhsC, RhsT>);
impl_partial_eq!(Row<LhsC, LhsT>, Row<RhsC, RhsT>);

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
    > PartialEq<DiagRef<'_, RhsC, RhsT>> for DiagRef<'_, LhsC, LhsT>
{
    fn eq(&self, other: &DiagRef<'_, RhsC, RhsT>) -> bool {
        self.column_vector().eq(&other.column_vector())
    }
}

// impl_partial_eq!(DiagRef<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>);
impl_partial_eq!(DiagRef<'_, LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>);
impl_partial_eq!(DiagRef<'_, LhsC, LhsT>, Diag<RhsC, RhsT>);

impl_partial_eq!(DiagMut<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>);
impl_partial_eq!(DiagMut<'_, LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>);
impl_partial_eq!(DiagMut<'_, LhsC, LhsT>, Diag<RhsC, RhsT>);

impl_partial_eq!(Diag<LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>);
impl_partial_eq!(Diag<LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>);
impl_partial_eq!(Diag<LhsC, LhsT>, Diag<RhsC, RhsT>);

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
    > Add<MatRef<'_, RhsC, RhsT>> for MatRef<'_, LhsC, LhsT>
{
    type Output = Mat<C, T>;

    #[math]
    #[track_caller]
    fn add(self, rhs: MatRef<'_, RhsC, RhsT>) -> Self::Output {
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
    > Sub<MatRef<'_, RhsC, RhsT>> for MatRef<'_, LhsC, LhsT>
{
    type Output = Mat<C, T>;

    #[math]
    #[track_caller]
    fn sub(self, rhs: MatRef<'_, RhsC, RhsT>) -> Self::Output {
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
    > AddAssign<MatRef<'_, RhsC, RhsT>> for MatMut<'_, LhsC, LhsT>
{
    #[math]
    #[track_caller]
    fn add_assign(&mut self, rhs: MatRef<'_, RhsC, RhsT>) {
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
    > SubAssign<MatRef<'_, RhsC, RhsT>> for MatMut<'_, LhsC, LhsT>
{
    #[math]
    #[track_caller]
    fn sub_assign(&mut self, rhs: MatRef<'_, RhsC, RhsT>) {
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
    > Neg for MatRef<'_, CC, TT>
{
    type Output = Mat<C, T>;

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
    > Add<ColRef<'_, RhsC, RhsT>> for ColRef<'_, LhsC, LhsT>
{
    type Output = Col<C, T>;

    #[math]
    #[track_caller]
    fn add(self, rhs: ColRef<'_, RhsC, RhsT>) -> Self::Output {
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
    > Sub<ColRef<'_, RhsC, RhsT>> for ColRef<'_, LhsC, LhsT>
{
    type Output = Col<C, T>;

    #[math]
    #[track_caller]
    fn sub(self, rhs: ColRef<'_, RhsC, RhsT>) -> Self::Output {
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
    > AddAssign<ColRef<'_, RhsC, RhsT>> for ColMut<'_, LhsC, LhsT>
{
    #[math]
    #[track_caller]
    fn add_assign(&mut self, rhs: ColRef<'_, RhsC, RhsT>) {
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
    > SubAssign<ColRef<'_, RhsC, RhsT>> for ColMut<'_, LhsC, LhsT>
{
    #[math]
    #[track_caller]
    fn sub_assign(&mut self, rhs: ColRef<'_, RhsC, RhsT>) {
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
    > Neg for ColRef<'_, CC, TT>
{
    type Output = Col<C, T>;

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
    > Add<RowRef<'_, RhsC, RhsT>> for RowRef<'_, LhsC, LhsT>
{
    type Output = Row<C, T>;

    #[math]
    #[track_caller]
    fn add(self, rhs: RowRef<'_, RhsC, RhsT>) -> Self::Output {
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
    > Sub<RowRef<'_, RhsC, RhsT>> for RowRef<'_, LhsC, LhsT>
{
    type Output = Row<C, T>;

    #[math]
    #[track_caller]
    fn sub(self, rhs: RowRef<'_, RhsC, RhsT>) -> Self::Output {
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
    > AddAssign<RowRef<'_, RhsC, RhsT>> for RowMut<'_, LhsC, LhsT>
{
    #[math]
    #[track_caller]
    fn add_assign(&mut self, rhs: RowRef<'_, RhsC, RhsT>) {
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
    > SubAssign<RowRef<'_, RhsC, RhsT>> for RowMut<'_, LhsC, LhsT>
{
    #[math]
    #[track_caller]
    fn sub_assign(&mut self, rhs: RowRef<'_, RhsC, RhsT>) {
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
    > Neg for RowRef<'_, CC, TT>
{
    type Output = Row<C, T>;

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
    > Add<DiagRef<'_, RhsC, RhsT>> for DiagRef<'_, LhsC, LhsT>
{
    type Output = Diag<C, T>;

    #[track_caller]
    #[math]
    fn add(self, rhs: DiagRef<'_, RhsC, RhsT>) -> Self::Output {
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
    > Sub<DiagRef<'_, RhsC, RhsT>> for DiagRef<'_, LhsC, LhsT>
{
    type Output = Diag<C, T>;

    #[track_caller]
    #[math]
    fn sub(self, rhs: DiagRef<'_, RhsC, RhsT>) -> Self::Output {
        (self.column_vector() - rhs.column_vector()).into_diagonal()
    }
}

impl<
        LhsC: ComplexContainer,
        RhsC: Container<Canonical = LhsC>,
        LhsT: ComplexField<LhsC, MathCtx: Default>,
        RhsT: ConjUnit<Canonical = LhsT>,
    > AddAssign<DiagRef<'_, RhsC, RhsT>> for DiagMut<'_, LhsC, LhsT>
{
    #[track_caller]
    fn add_assign(&mut self, rhs: DiagRef<'_, RhsC, RhsT>) {
        *&mut (self.rb_mut().column_vector_mut()) += rhs.column_vector()
    }
}

impl<
        LhsC: ComplexContainer,
        RhsC: Container<Canonical = LhsC>,
        LhsT: ComplexField<LhsC, MathCtx: Default>,
        RhsT: ConjUnit<Canonical = LhsT>,
    > SubAssign<DiagRef<'_, RhsC, RhsT>> for DiagMut<'_, LhsC, LhsT>
{
    #[track_caller]
    fn sub_assign(&mut self, rhs: DiagRef<'_, RhsC, RhsT>) {
        *&mut (self.rb_mut().column_vector_mut()) -= rhs.column_vector()
    }
}

impl<
        C: ComplexContainer,
        CC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        TT: ConjUnit<Canonical = T>,
    > Neg for DiagRef<'_, CC, TT>
{
    type Output = Diag<C, T>;

    fn neg(self) -> Self::Output {
        (-self.column_vector()).into_diagonal()
    }
}

// impl_add_sub!(MatRef<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(MatRef<'_, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(MatRef<'_, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(MatRef<'_, LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(MatRef<'_, LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(MatRef<'_, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(&MatRef<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(&MatRef<'_, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(&MatRef<'_, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(&MatRef<'_, LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(&MatRef<'_, LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(&MatRef<'_, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);

impl_add_sub!(MatMut<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(MatMut<'_, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(MatMut<'_, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(MatMut<'_, LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(MatMut<'_, LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(MatMut<'_, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(&MatMut<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(&MatMut<'_, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(&MatMut<'_, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(&MatMut<'_, LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(&MatMut<'_, LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(&MatMut<'_, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);

impl_add_sub!(Mat<LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(Mat<LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(Mat<LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(Mat<LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(Mat<LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(Mat<LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(&Mat<LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(&Mat<LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(&Mat<LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(&Mat<LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(&Mat<LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
impl_add_sub!(&Mat<LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);

// impl_add_sub_assign!(MatMut<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>);
impl_add_sub_assign!(MatMut<'_, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>);
impl_add_sub_assign!(MatMut<'_, LhsC, LhsT>, Mat<RhsC, RhsT>);
impl_add_sub_assign!(MatMut<'_, LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>);
impl_add_sub_assign!(MatMut<'_, LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>);
impl_add_sub_assign!(MatMut<'_, LhsC, LhsT>, &Mat<RhsC, RhsT>);

impl_add_sub_assign!(Mat<LhsC, LhsT>, MatRef<'_, RhsC, RhsT>);
impl_add_sub_assign!(Mat<LhsC, LhsT>, MatMut<'_, RhsC, RhsT>);
impl_add_sub_assign!(Mat<LhsC, LhsT>, Mat<RhsC, RhsT>);
impl_add_sub_assign!(Mat<LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>);
impl_add_sub_assign!(Mat<LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>);
impl_add_sub_assign!(Mat<LhsC, LhsT>, &Mat<RhsC, RhsT>);

// impl_neg!(MatRef<'_, CC, TT>, Mat<C, T>);
impl_neg!(MatMut<'_, CC, TT>, Mat<C, T>);
impl_neg!(Mat<CC, TT>, Mat<C, T>);
impl_neg!(&MatRef<'_, CC, TT>, Mat<C, T>);
impl_neg!(&MatMut<'_, CC, TT>, Mat<C, T>);
impl_neg!(&Mat<CC, TT>, Mat<C, T>);

// impl_add_sub!(ColRef<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(ColRef<'_, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(ColRef<'_, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
impl_add_sub!(ColRef<'_, LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(ColRef<'_, LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(ColRef<'_, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);
impl_add_sub!(&ColRef<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(&ColRef<'_, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(&ColRef<'_, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
impl_add_sub!(&ColRef<'_, LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(&ColRef<'_, LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(&ColRef<'_, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);

impl_add_sub!(ColMut<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(ColMut<'_, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(ColMut<'_, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
impl_add_sub!(ColMut<'_, LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(ColMut<'_, LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(ColMut<'_, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);
impl_add_sub!(&ColMut<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(&ColMut<'_, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(&ColMut<'_, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
impl_add_sub!(&ColMut<'_, LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(&ColMut<'_, LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(&ColMut<'_, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);

impl_add_sub!(Col<LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(Col<LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(Col<LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
impl_add_sub!(Col<LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(Col<LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(Col<LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);
impl_add_sub!(&Col<LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(&Col<LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(&Col<LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
impl_add_sub!(&Col<LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(&Col<LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
impl_add_sub!(&Col<LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);

// impl_add_sub_assign!(ColMut<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>);
impl_add_sub_assign!(ColMut<'_, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>);
impl_add_sub_assign!(ColMut<'_, LhsC, LhsT>, Col<RhsC, RhsT>);
impl_add_sub_assign!(ColMut<'_, LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>);
impl_add_sub_assign!(ColMut<'_, LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>);
impl_add_sub_assign!(ColMut<'_, LhsC, LhsT>, &Col<RhsC, RhsT>);

impl_add_sub_assign!(Col<LhsC, LhsT>, ColRef<'_, RhsC, RhsT>);
impl_add_sub_assign!(Col<LhsC, LhsT>, ColMut<'_, RhsC, RhsT>);
impl_add_sub_assign!(Col<LhsC, LhsT>, Col<RhsC, RhsT>);
impl_add_sub_assign!(Col<LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>);
impl_add_sub_assign!(Col<LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>);
impl_add_sub_assign!(Col<LhsC, LhsT>, &Col<RhsC, RhsT>);

// impl_neg!(ColRef<'_, CC, TT>, Col<C, T>);
impl_neg!(ColMut<'_, CC, TT>, Col<C, T>);
impl_neg!(Col<CC, TT>, Col<C, T>);
impl_neg!(&ColRef<'_, CC, TT>, Col<C, T>);
impl_neg!(&ColMut<'_, CC, TT>, Col<C, T>);
impl_neg!(&Col<CC, TT>, Col<C, T>);

// impl_add_sub!(RowRef<'_, LhsC, LhsT>, RowRef<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(RowRef<'_, LhsC, LhsT>, RowMut<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(RowRef<'_, LhsC, LhsT>, Row<RhsC, RhsT>, Row<C, T>);
impl_add_sub!(RowRef<'_, LhsC, LhsT>, &RowRef<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(RowRef<'_, LhsC, LhsT>, &RowMut<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(RowRef<'_, LhsC, LhsT>, &Row<RhsC, RhsT>, Row<C, T>);
impl_add_sub!(&RowRef<'_, LhsC, LhsT>, RowRef<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(&RowRef<'_, LhsC, LhsT>, RowMut<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(&RowRef<'_, LhsC, LhsT>, Row<RhsC, RhsT>, Row<C, T>);
impl_add_sub!(&RowRef<'_, LhsC, LhsT>, &RowRef<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(&RowRef<'_, LhsC, LhsT>, &RowMut<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(&RowRef<'_, LhsC, LhsT>, &Row<RhsC, RhsT>, Row<C, T>);

impl_add_sub!(RowMut<'_, LhsC, LhsT>, RowRef<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(RowMut<'_, LhsC, LhsT>, RowMut<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(RowMut<'_, LhsC, LhsT>, Row<RhsC, RhsT>, Row<C, T>);
impl_add_sub!(RowMut<'_, LhsC, LhsT>, &RowRef<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(RowMut<'_, LhsC, LhsT>, &RowMut<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(RowMut<'_, LhsC, LhsT>, &Row<RhsC, RhsT>, Row<C, T>);
impl_add_sub!(&RowMut<'_, LhsC, LhsT>, RowRef<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(&RowMut<'_, LhsC, LhsT>, RowMut<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(&RowMut<'_, LhsC, LhsT>, Row<RhsC, RhsT>, Row<C, T>);
impl_add_sub!(&RowMut<'_, LhsC, LhsT>, &RowRef<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(&RowMut<'_, LhsC, LhsT>, &RowMut<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(&RowMut<'_, LhsC, LhsT>, &Row<RhsC, RhsT>, Row<C, T>);

impl_add_sub!(Row<LhsC, LhsT>, RowRef<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(Row<LhsC, LhsT>, RowMut<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(Row<LhsC, LhsT>, Row<RhsC, RhsT>, Row<C, T>);
impl_add_sub!(Row<LhsC, LhsT>, &RowRef<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(Row<LhsC, LhsT>, &RowMut<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(Row<LhsC, LhsT>, &Row<RhsC, RhsT>, Row<C, T>);
impl_add_sub!(&Row<LhsC, LhsT>, RowRef<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(&Row<LhsC, LhsT>, RowMut<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(&Row<LhsC, LhsT>, Row<RhsC, RhsT>, Row<C, T>);
impl_add_sub!(&Row<LhsC, LhsT>, &RowRef<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(&Row<LhsC, LhsT>, &RowMut<'_, RhsC, RhsT>, Row<C, T>);
impl_add_sub!(&Row<LhsC, LhsT>, &Row<RhsC, RhsT>, Row<C, T>);

// impl_add_sub_assign!(RowMut<'_, LhsC, LhsT>, RowRef<'_, RhsC, RhsT>);
impl_add_sub_assign!(RowMut<'_, LhsC, LhsT>, RowMut<'_, RhsC, RhsT>);
impl_add_sub_assign!(RowMut<'_, LhsC, LhsT>, Row<RhsC, RhsT>);
impl_add_sub_assign!(RowMut<'_, LhsC, LhsT>, &RowRef<'_, RhsC, RhsT>);
impl_add_sub_assign!(RowMut<'_, LhsC, LhsT>, &RowMut<'_, RhsC, RhsT>);
impl_add_sub_assign!(RowMut<'_, LhsC, LhsT>, &Row<RhsC, RhsT>);

impl_add_sub_assign!(Row<LhsC, LhsT>, RowRef<'_, RhsC, RhsT>);
impl_add_sub_assign!(Row<LhsC, LhsT>, RowMut<'_, RhsC, RhsT>);
impl_add_sub_assign!(Row<LhsC, LhsT>, Row<RhsC, RhsT>);
impl_add_sub_assign!(Row<LhsC, LhsT>, &RowRef<'_, RhsC, RhsT>);
impl_add_sub_assign!(Row<LhsC, LhsT>, &RowMut<'_, RhsC, RhsT>);
impl_add_sub_assign!(Row<LhsC, LhsT>, &Row<RhsC, RhsT>);

// impl_neg!(RowRef<'_, CC, TT>, Row<C, T>);
impl_neg!(RowMut<'_, CC, TT>, Row<C, T>);
impl_neg!(Row<CC, TT>, Row<C, T>);
impl_neg!(&RowRef<'_, CC, TT>, Row<C, T>);
impl_neg!(&RowMut<'_, CC, TT>, Row<C, T>);
impl_neg!(&Row<CC, TT>, Row<C, T>);

// impl_add_sub!(DiagRef<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(DiagRef<'_, LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(DiagRef<'_, LhsC, LhsT>, Diag<RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(DiagRef<'_, LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(DiagRef<'_, LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(DiagRef<'_, LhsC, LhsT>, &Diag<RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(&DiagRef<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(&DiagRef<'_, LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(&DiagRef<'_, LhsC, LhsT>, Diag<RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(&DiagRef<'_, LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(&DiagRef<'_, LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(&DiagRef<'_, LhsC, LhsT>, &Diag<RhsC, RhsT>, Diag<C, T>);

impl_add_sub!(DiagMut<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(DiagMut<'_, LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(DiagMut<'_, LhsC, LhsT>, Diag<RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(DiagMut<'_, LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(DiagMut<'_, LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(DiagMut<'_, LhsC, LhsT>, &Diag<RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(&DiagMut<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(&DiagMut<'_, LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(&DiagMut<'_, LhsC, LhsT>, Diag<RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(&DiagMut<'_, LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(&DiagMut<'_, LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(&DiagMut<'_, LhsC, LhsT>, &Diag<RhsC, RhsT>, Diag<C, T>);

impl_add_sub!(Diag<LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(Diag<LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(Diag<LhsC, LhsT>, Diag<RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(Diag<LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(Diag<LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(Diag<LhsC, LhsT>, &Diag<RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(&Diag<LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(&Diag<LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(&Diag<LhsC, LhsT>, Diag<RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(&Diag<LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(&Diag<LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
impl_add_sub!(&Diag<LhsC, LhsT>, &Diag<RhsC, RhsT>, Diag<C, T>);

// impl_add_sub_assign!(DiagMut<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>);
impl_add_sub_assign!(DiagMut<'_, LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>);
impl_add_sub_assign!(DiagMut<'_, LhsC, LhsT>, Diag<RhsC, RhsT>);
impl_add_sub_assign!(DiagMut<'_, LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>);
impl_add_sub_assign!(DiagMut<'_, LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>);
impl_add_sub_assign!(DiagMut<'_, LhsC, LhsT>, &Diag<RhsC, RhsT>);

impl_add_sub_assign!(Diag<LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>);
impl_add_sub_assign!(Diag<LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>);
impl_add_sub_assign!(Diag<LhsC, LhsT>, Diag<RhsC, RhsT>);
impl_add_sub_assign!(Diag<LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>);
impl_add_sub_assign!(Diag<LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>);
impl_add_sub_assign!(Diag<LhsC, LhsT>, &Diag<RhsC, RhsT>);

// impl_neg!(DiagRef<'_, CC, TT>, Diag<C, T>);
impl_neg!(DiagMut<'_, CC, TT>, Diag<C, T>);
impl_neg!(Diag<CC, TT>, Diag<C, T>);
impl_neg!(&DiagRef<'_, CC, TT>, Diag<C, T>);
impl_neg!(&DiagMut<'_, CC, TT>, Diag<C, T>);
impl_neg!(&Diag<CC, TT>, Diag<C, T>);

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
        > Mul<MatRef<'_, RhsC, RhsT>> for MatRef<'_, LhsC, LhsT>
    {
        type Output = Mat<C, T>;

        #[inline]
        #[track_caller]
        fn mul(self, rhs: MatRef<'_, RhsC, RhsT>) -> Self::Output {
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
        > Mul<ColRef<'_, RhsC, RhsT>> for MatRef<'_, LhsC, LhsT>
    {
        type Output = Col<C, T>;

        #[inline]
        #[track_caller]
        fn mul(self, rhs: ColRef<'_, RhsC, RhsT>) -> Self::Output {
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
        > Mul<MatRef<'_, RhsC, RhsT>> for RowRef<'_, LhsC, LhsT>
    {
        type Output = Row<C, T>;

        #[inline]
        #[track_caller]
        fn mul(self, rhs: MatRef<'_, RhsC, RhsT>) -> Self::Output {
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
        > Mul<ColRef<'_, RhsC, RhsT>> for RowRef<'_, LhsC, LhsT>
    {
        type Output = C::Of<T>;

        #[inline]
        #[track_caller]
        fn mul(self, rhs: ColRef<'_, RhsC, RhsT>) -> Self::Output {
            let lhs = self;
            Assert!(lhs.ncols() == rhs.nrows());
            let lhs = lhs.canonical();
            let rhs = rhs.canonical();
            let ctx = &Ctx::<C, T>::default();
            with_dim!(K, lhs.ncols());
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
        > Mul<RowRef<'_, RhsC, RhsT>> for ColRef<'_, LhsC, LhsT>
    {
        type Output = Mat<C, T>;

        #[inline]
        #[track_caller]
        fn mul(self, rhs: RowRef<'_, RhsC, RhsT>) -> Self::Output {
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
    impl_mul!(MatRef<'_, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(MatRef<'_, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(MatRef<'_, LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(MatRef<'_, LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(MatRef<'_, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatRef<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatRef<'_, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatRef<'_, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatRef<'_, LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatRef<'_, LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatRef<'_, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);

    impl_mul!(MatMut<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(MatMut<'_, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(MatMut<'_, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(MatMut<'_, LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(MatMut<'_, LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(MatMut<'_, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatMut<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatMut<'_, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatMut<'_, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatMut<'_, LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatMut<'_, LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatMut<'_, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);

    impl_mul!(Mat<LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(Mat<LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(Mat<LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(Mat<LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(Mat<LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(Mat<LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Mat<LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Mat<LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Mat<LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Mat<LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Mat<LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Mat<LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);

    // impl_mul!(MatRef<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(MatRef<'_, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(MatRef<'_, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_mul!(MatRef<'_, LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(MatRef<'_, LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(MatRef<'_, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);
    impl_mul!(&MatRef<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&MatRef<'_, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&MatRef<'_, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_mul!(&MatRef<'_, LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&MatRef<'_, LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&MatRef<'_, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);

    impl_mul!(MatMut<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(MatMut<'_, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(MatMut<'_, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_mul!(MatMut<'_, LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(MatMut<'_, LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(MatMut<'_, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);
    impl_mul!(&MatMut<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&MatMut<'_, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&MatMut<'_, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_mul!(&MatMut<'_, LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&MatMut<'_, LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&MatMut<'_, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);

    impl_mul!(Mat<LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(Mat<LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(Mat<LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_mul!(Mat<LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(Mat<LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(Mat<LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);
    impl_mul!(&Mat<LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&Mat<LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&Mat<LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_mul!(&Mat<LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&Mat<LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&Mat<LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);

    // impl_mul!(RowRef<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(RowRef<'_, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(RowRef<'_, LhsC, LhsT>, Mat<RhsC, RhsT>, Row<C, T>);
    impl_mul!(RowRef<'_, LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(RowRef<'_, LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(RowRef<'_, LhsC, LhsT>, &Mat<RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowRef<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowRef<'_, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowRef<'_, LhsC, LhsT>, Mat<RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowRef<'_, LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowRef<'_, LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowRef<'_, LhsC, LhsT>, &Mat<RhsC, RhsT>, Row<C, T>);

    impl_mul!(RowMut<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(RowMut<'_, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(RowMut<'_, LhsC, LhsT>, Mat<RhsC, RhsT>, Row<C, T>);
    impl_mul!(RowMut<'_, LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(RowMut<'_, LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(RowMut<'_, LhsC, LhsT>, &Mat<RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowMut<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowMut<'_, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowMut<'_, LhsC, LhsT>, Mat<RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowMut<'_, LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowMut<'_, LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowMut<'_, LhsC, LhsT>, &Mat<RhsC, RhsT>, Row<C, T>);

    impl_mul!(Row<LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(Row<LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(Row<LhsC, LhsT>, Mat<RhsC, RhsT>, Row<C, T>);
    impl_mul!(Row<LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(Row<LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(Row<LhsC, LhsT>, &Mat<RhsC, RhsT>, Row<C, T>);
    impl_mul!(&Row<LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&Row<LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&Row<LhsC, LhsT>, Mat<RhsC, RhsT>, Row<C, T>);
    impl_mul!(&Row<LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&Row<LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&Row<LhsC, LhsT>, &Mat<RhsC, RhsT>, Row<C, T>);

    // impl_mul!(RowRef<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(RowRef<'_, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(RowRef<'_, LhsC, LhsT>, Col<RhsC, RhsT>, C::Of<T>);
    impl_mul!(RowRef<'_, LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(RowRef<'_, LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(RowRef<'_, LhsC, LhsT>, &Col<RhsC, RhsT>, C::Of<T>);
    impl_mul!(&RowRef<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(&RowRef<'_, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(&RowRef<'_, LhsC, LhsT>, Col<RhsC, RhsT>, C::Of<T>);
    impl_mul!(&RowRef<'_, LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(&RowRef<'_, LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(&RowRef<'_, LhsC, LhsT>, &Col<RhsC, RhsT>, C::Of<T>);

    impl_mul!(RowMut<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(RowMut<'_, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(RowMut<'_, LhsC, LhsT>, Col<RhsC, RhsT>, C::Of<T>);
    impl_mul!(RowMut<'_, LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(RowMut<'_, LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(RowMut<'_, LhsC, LhsT>, &Col<RhsC, RhsT>, C::Of<T>);
    impl_mul!(&RowMut<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(&RowMut<'_, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(&RowMut<'_, LhsC, LhsT>, Col<RhsC, RhsT>, C::Of<T>);
    impl_mul!(&RowMut<'_, LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(&RowMut<'_, LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(&RowMut<'_, LhsC, LhsT>, &Col<RhsC, RhsT>, C::Of<T>);

    impl_mul!(Row<LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(Row<LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(Row<LhsC, LhsT>, Col<RhsC, RhsT>, C::Of<T>);
    impl_mul!(Row<LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(Row<LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(Row<LhsC, LhsT>, &Col<RhsC, RhsT>, C::Of<T>);
    impl_mul!(&Row<LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(&Row<LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(&Row<LhsC, LhsT>, Col<RhsC, RhsT>, C::Of<T>);
    impl_mul!(&Row<LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(&Row<LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, C::Of<T>);
    impl_mul!(&Row<LhsC, LhsT>, &Col<RhsC, RhsT>, C::Of<T>);

    // impl_mul!(ColRef<'_, LhsC, LhsT>, RowRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(ColRef<'_, LhsC, LhsT>, RowMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(ColRef<'_, LhsC, LhsT>, Row<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(ColRef<'_, LhsC, LhsT>, &RowRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(ColRef<'_, LhsC, LhsT>, &RowMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(ColRef<'_, LhsC, LhsT>, &Row<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&ColRef<'_, LhsC, LhsT>, RowRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&ColRef<'_, LhsC, LhsT>, RowMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&ColRef<'_, LhsC, LhsT>, Row<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&ColRef<'_, LhsC, LhsT>, &RowRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&ColRef<'_, LhsC, LhsT>, &RowMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&ColRef<'_, LhsC, LhsT>, &Row<RhsC, RhsT>, Mat<C, T>);

    impl_mul!(ColMut<'_, LhsC, LhsT>, RowRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(ColMut<'_, LhsC, LhsT>, RowMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(ColMut<'_, LhsC, LhsT>, Row<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(ColMut<'_, LhsC, LhsT>, &RowRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(ColMut<'_, LhsC, LhsT>, &RowMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(ColMut<'_, LhsC, LhsT>, &Row<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&ColMut<'_, LhsC, LhsT>, RowRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&ColMut<'_, LhsC, LhsT>, RowMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&ColMut<'_, LhsC, LhsT>, Row<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&ColMut<'_, LhsC, LhsT>, &RowRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&ColMut<'_, LhsC, LhsT>, &RowMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&ColMut<'_, LhsC, LhsT>, &Row<RhsC, RhsT>, Mat<C, T>);

    impl_mul!(Col<LhsC, LhsT>, RowRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(Col<LhsC, LhsT>, RowMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(Col<LhsC, LhsT>, Row<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(Col<LhsC, LhsT>, &RowRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(Col<LhsC, LhsT>, &RowMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(Col<LhsC, LhsT>, &Row<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Col<LhsC, LhsT>, RowRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Col<LhsC, LhsT>, RowMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Col<LhsC, LhsT>, Row<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Col<LhsC, LhsT>, &RowRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Col<LhsC, LhsT>, &RowMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Col<LhsC, LhsT>, &Row<RhsC, RhsT>, Mat<C, T>);

    impl<
            C: ComplexContainer,
            LhsC: Container<Canonical = C>,
            RhsC: Container<Canonical = C>,
            T: ComplexField<C, MathCtx: Default>,
            LhsT: ConjUnit<Canonical = T>,
            RhsT: ConjUnit<Canonical = T>,
        > Mul<MatRef<'_, RhsC, RhsT>> for DiagRef<'_, LhsC, LhsT>
    {
        type Output = Mat<C, T>;

        #[track_caller]
        #[math]
        fn mul(self, rhs: MatRef<'_, RhsC, RhsT>) -> Self::Output {
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
    impl_mul!(DiagRef<'_, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(DiagRef<'_, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(DiagRef<'_, LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(DiagRef<'_, LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(DiagRef<'_, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&DiagRef<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&DiagRef<'_, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&DiagRef<'_, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&DiagRef<'_, LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&DiagRef<'_, LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&DiagRef<'_, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);

    impl_mul!(DiagMut<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(DiagMut<'_, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(DiagMut<'_, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(DiagMut<'_, LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(DiagMut<'_, LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(DiagMut<'_, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&DiagMut<'_, LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&DiagMut<'_, LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&DiagMut<'_, LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&DiagMut<'_, LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&DiagMut<'_, LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&DiagMut<'_, LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);

    impl_mul!(Diag<LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(Diag<LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(Diag<LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(Diag<LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(Diag<LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(Diag<LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Diag<LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Diag<LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Diag<LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Diag<LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Diag<LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Diag<LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);

    impl<
            C: ComplexContainer,
            LhsC: Container<Canonical = C>,
            RhsC: Container<Canonical = C>,
            T: ComplexField<C, MathCtx: Default>,
            LhsT: ConjUnit<Canonical = T>,
            RhsT: ConjUnit<Canonical = T>,
        > Mul<ColRef<'_, RhsC, RhsT>> for DiagRef<'_, LhsC, LhsT>
    {
        type Output = Col<C, T>;

        #[track_caller]
        #[math]
        fn mul(self, rhs: ColRef<'_, RhsC, RhsT>) -> Self::Output {
            let lhs = self.column_vector();
            let lhs_dim = lhs.nrows();
            let rhs_nrows = rhs.nrows();
            Assert!(lhs_dim == rhs_nrows);

            let ctx = &Ctx::<C, T>(T::MathCtx::default());
            zipped!(lhs, rhs).mapC::<C, T>(mul_fn::<LhsC, RhsC, _, _>(ctx))
        }
    }

    // impl_mul!(DiagRef<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(DiagRef<'_, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(DiagRef<'_, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_mul!(DiagRef<'_, LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(DiagRef<'_, LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(DiagRef<'_, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);
    impl_mul!(&DiagRef<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&DiagRef<'_, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&DiagRef<'_, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_mul!(&DiagRef<'_, LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&DiagRef<'_, LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&DiagRef<'_, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);

    impl_mul!(DiagMut<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(DiagMut<'_, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(DiagMut<'_, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_mul!(DiagMut<'_, LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(DiagMut<'_, LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(DiagMut<'_, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);
    impl_mul!(&DiagMut<'_, LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&DiagMut<'_, LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&DiagMut<'_, LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_mul!(&DiagMut<'_, LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&DiagMut<'_, LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&DiagMut<'_, LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);

    impl_mul!(Diag<LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(Diag<LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(Diag<LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_mul!(Diag<LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(Diag<LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(Diag<LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);
    impl_mul!(&Diag<LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&Diag<LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&Diag<LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
    impl_mul!(&Diag<LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&Diag<LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
    impl_mul!(&Diag<LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);

    impl<
            C: ComplexContainer,
            LhsC: Container<Canonical = C>,
            RhsC: Container<Canonical = C>,
            T: ComplexField<C, MathCtx: Default>,
            LhsT: ConjUnit<Canonical = T>,
            RhsT: ConjUnit<Canonical = T>,
        > Mul<DiagRef<'_, RhsC, RhsT>> for MatRef<'_, LhsC, LhsT>
    {
        type Output = Mat<C, T>;

        #[math]
        #[track_caller]
        fn mul(self, rhs: DiagRef<'_, RhsC, RhsT>) -> Self::Output {
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
    impl_mul!(MatRef<'_, LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(MatRef<'_, LhsC, LhsT>, Diag<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(MatRef<'_, LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(MatRef<'_, LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(MatRef<'_, LhsC, LhsT>, &Diag<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatRef<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatRef<'_, LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatRef<'_, LhsC, LhsT>, Diag<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatRef<'_, LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatRef<'_, LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatRef<'_, LhsC, LhsT>, &Diag<RhsC, RhsT>, Mat<C, T>);

    impl_mul!(MatMut<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(MatMut<'_, LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(MatMut<'_, LhsC, LhsT>, Diag<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(MatMut<'_, LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(MatMut<'_, LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(MatMut<'_, LhsC, LhsT>, &Diag<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatMut<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatMut<'_, LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatMut<'_, LhsC, LhsT>, Diag<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatMut<'_, LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatMut<'_, LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&MatMut<'_, LhsC, LhsT>, &Diag<RhsC, RhsT>, Mat<C, T>);

    impl_mul!(Mat<LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(Mat<LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(Mat<LhsC, LhsT>, Diag<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(Mat<LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(Mat<LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(Mat<LhsC, LhsT>, &Diag<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Mat<LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Mat<LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Mat<LhsC, LhsT>, Diag<RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Mat<LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Mat<LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Mat<C, T>);
    impl_mul!(&Mat<LhsC, LhsT>, &Diag<RhsC, RhsT>, Mat<C, T>);

    impl<
            C: ComplexContainer,
            LhsC: Container<Canonical = C>,
            RhsC: Container<Canonical = C>,
            T: ComplexField<C, MathCtx: Default>,
            LhsT: ConjUnit<Canonical = T>,
            RhsT: ConjUnit<Canonical = T>,
        > Mul<DiagRef<'_, RhsC, RhsT>> for RowRef<'_, LhsC, LhsT>
    {
        type Output = Row<C, T>;

        #[math]
        #[track_caller]
        fn mul(self, rhs: DiagRef<'_, RhsC, RhsT>) -> Self::Output {
            let lhs = self;
            let rhs = rhs.column_vector().transpose();
            let lhs_ncols = lhs.ncols();
            let rhs_dim = rhs.nrows();
            Assert!(lhs_ncols == rhs_dim);

            let ctx = &Ctx::<C, T>(T::MathCtx::default());
            zipped!(lhs, rhs).mapC::<C, T>(mul_fn::<LhsC, RhsC, _, _>(ctx))
        }
    }

    // impl_mul!(RowRef<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(RowRef<'_, LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(RowRef<'_, LhsC, LhsT>, Diag<RhsC, RhsT>, Row<C, T>);
    impl_mul!(RowRef<'_, LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(RowRef<'_, LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(RowRef<'_, LhsC, LhsT>, &Diag<RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowRef<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowRef<'_, LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowRef<'_, LhsC, LhsT>, Diag<RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowRef<'_, LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowRef<'_, LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowRef<'_, LhsC, LhsT>, &Diag<RhsC, RhsT>, Row<C, T>);

    impl_mul!(RowMut<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(RowMut<'_, LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(RowMut<'_, LhsC, LhsT>, Diag<RhsC, RhsT>, Row<C, T>);
    impl_mul!(RowMut<'_, LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(RowMut<'_, LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(RowMut<'_, LhsC, LhsT>, &Diag<RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowMut<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowMut<'_, LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowMut<'_, LhsC, LhsT>, Diag<RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowMut<'_, LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowMut<'_, LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&RowMut<'_, LhsC, LhsT>, &Diag<RhsC, RhsT>, Row<C, T>);

    impl_mul!(Row<LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(Row<LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(Row<LhsC, LhsT>, Diag<RhsC, RhsT>, Row<C, T>);
    impl_mul!(Row<LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(Row<LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(Row<LhsC, LhsT>, &Diag<RhsC, RhsT>, Row<C, T>);
    impl_mul!(&Row<LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&Row<LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&Row<LhsC, LhsT>, Diag<RhsC, RhsT>, Row<C, T>);
    impl_mul!(&Row<LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&Row<LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Row<C, T>);
    impl_mul!(&Row<LhsC, LhsT>, &Diag<RhsC, RhsT>, Row<C, T>);

    impl<
            C: ComplexContainer,
            LhsC: Container<Canonical = C>,
            RhsC: Container<Canonical = C>,
            T: ComplexField<C, MathCtx: Default>,
            LhsT: ConjUnit<Canonical = T>,
            RhsT: ConjUnit<Canonical = T>,
        > Mul<DiagRef<'_, RhsC, RhsT>> for DiagRef<'_, LhsC, LhsT>
    {
        type Output = Diag<C, T>;

        #[track_caller]
        #[math]
        fn mul(self, rhs: DiagRef<'_, RhsC, RhsT>) -> Self::Output {
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
    impl_mul!(DiagRef<'_, LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(DiagRef<'_, LhsC, LhsT>, Diag<RhsC, RhsT>, Diag<C, T>);
    impl_mul!(DiagRef<'_, LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(DiagRef<'_, LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(DiagRef<'_, LhsC, LhsT>, &Diag<RhsC, RhsT>, Diag<C, T>);
    impl_mul!(&DiagRef<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(&DiagRef<'_, LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(&DiagRef<'_, LhsC, LhsT>, Diag<RhsC, RhsT>, Diag<C, T>);
    impl_mul!(&DiagRef<'_, LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(&DiagRef<'_, LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(&DiagRef<'_, LhsC, LhsT>, &Diag<RhsC, RhsT>, Diag<C, T>);

    impl_mul!(DiagMut<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(DiagMut<'_, LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(DiagMut<'_, LhsC, LhsT>, Diag<RhsC, RhsT>, Diag<C, T>);
    impl_mul!(DiagMut<'_, LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(DiagMut<'_, LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(DiagMut<'_, LhsC, LhsT>, &Diag<RhsC, RhsT>, Diag<C, T>);
    impl_mul!(&DiagMut<'_, LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(&DiagMut<'_, LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(&DiagMut<'_, LhsC, LhsT>, Diag<RhsC, RhsT>, Diag<C, T>);
    impl_mul!(&DiagMut<'_, LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(&DiagMut<'_, LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(&DiagMut<'_, LhsC, LhsT>, &Diag<RhsC, RhsT>, Diag<C, T>);

    impl_mul!(Diag<LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(Diag<LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(Diag<LhsC, LhsT>, Diag<RhsC, RhsT>, Diag<C, T>);
    impl_mul!(Diag<LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(Diag<LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(Diag<LhsC, LhsT>, &Diag<RhsC, RhsT>, Diag<C, T>);
    impl_mul!(&Diag<LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(&Diag<LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(&Diag<LhsC, LhsT>, Diag<RhsC, RhsT>, Diag<C, T>);
    impl_mul!(&Diag<LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(&Diag<LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
    impl_mul!(&Diag<LhsC, LhsT>, &Diag<RhsC, RhsT>, Diag<C, T>);
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
    > Mul<MatRef<'_, CC, TT>> for PermRef<'_, I>
{
    type Output = Mat<C, T>;

    #[track_caller]
    #[math]
    fn mul(self, rhs: MatRef<'_, CC, TT>) -> Self::Output {
        let lhs = self;

        Assert!(lhs.len() == rhs.nrows());
        let ctx = &Ctx::<C, T>(T::MathCtx::default());
        let mut out = Mat::zeros_with_ctx(ctx, rhs.nrows(), rhs.ncols());
        let fwd = lhs.arrays().0;

        for j in 0..rhs.ncols() {
            for (i, fwd) in fwd.iter().enumerate() {
                let rhs = rhs.at(fwd.to_signed().zx(), j);
                help!(C);
                math(write1!(
                    out.as_mut().write(i, j),
                    Conj::apply::<CC, TT>(ctx, rhs)
                ));
            }
        }
        out
    }
}

// impl_perm!(PermRef<'_, I>, MatRef<'_, CC, TT>, Mat<C, T>);
impl_perm!(PermRef<'_, I>, MatMut<'_, CC, TT>, Mat<C, T>);
impl_perm!(PermRef<'_, I>, Mat<CC, TT>, Mat<C, T>);
impl_perm!(PermRef<'_, I>, &MatRef<'_, CC, TT>, Mat<C, T>);
impl_perm!(PermRef<'_, I>, &MatMut<'_, CC, TT>, Mat<C, T>);
impl_perm!(PermRef<'_, I>, &Mat<CC, TT>, Mat<C, T>);
impl_perm!(&PermRef<'_, I>, MatRef<'_, CC, TT>, Mat<C, T>);
impl_perm!(&PermRef<'_, I>, MatMut<'_, CC, TT>, Mat<C, T>);
impl_perm!(&PermRef<'_, I>, Mat<CC, TT>, Mat<C, T>);
impl_perm!(&PermRef<'_, I>, &MatRef<'_, CC, TT>, Mat<C, T>);
impl_perm!(&PermRef<'_, I>, &MatMut<'_, CC, TT>, Mat<C, T>);
impl_perm!(&PermRef<'_, I>, &Mat<CC, TT>, Mat<C, T>);

impl_perm!(Perm<I>, MatRef<'_, CC, TT>, Mat<C, T>);
impl_perm!(Perm<I>, MatMut<'_, CC, TT>, Mat<C, T>);
impl_perm!(Perm<I>, Mat<CC, TT>, Mat<C, T>);
impl_perm!(Perm<I>, &MatRef<'_, CC, TT>, Mat<C, T>);
impl_perm!(Perm<I>, &MatMut<'_, CC, TT>, Mat<C, T>);
impl_perm!(Perm<I>, &Mat<CC, TT>, Mat<C, T>);
impl_perm!(&Perm<I>, MatRef<'_, CC, TT>, Mat<C, T>);
impl_perm!(&Perm<I>, MatMut<'_, CC, TT>, Mat<C, T>);
impl_perm!(&Perm<I>, Mat<CC, TT>, Mat<C, T>);
impl_perm!(&Perm<I>, &MatRef<'_, CC, TT>, Mat<C, T>);
impl_perm!(&Perm<I>, &MatMut<'_, CC, TT>, Mat<C, T>);
impl_perm!(&Perm<I>, &Mat<CC, TT>, Mat<C, T>);

impl<
        I: Index,
        C: ComplexContainer,
        CC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        TT: ConjUnit<Canonical = T>,
    > Mul<ColRef<'_, CC, TT>> for PermRef<'_, I>
{
    type Output = Col<C, T>;

    #[track_caller]
    #[math]
    fn mul(self, rhs: ColRef<'_, CC, TT>) -> Self::Output {
        let lhs = self;

        let ctx = &Ctx::<C, T>(T::MathCtx::default());
        Assert!(lhs.len() == rhs.nrows());
        let mut out = Col::zeros_with_ctx(ctx, rhs.nrows());
        let fwd = lhs.arrays().0;

        for (i, fwd) in fwd.iter().enumerate() {
            let rhs = rhs.at(fwd.to_signed().zx());
            help!(C);
            math(write1!(
                out.as_mut().at_mut(i),
                Conj::apply::<CC, TT>(ctx, rhs)
            ));
        }
        out
    }
}

// impl_perm!(PermRef<'_, I>, ColRef<'_, CC, TT>, Col<C, T>);
impl_perm!(PermRef<'_, I>, ColMut<'_, CC, TT>, Col<C, T>);
impl_perm!(PermRef<'_, I>, Col<CC, TT>, Col<C, T>);
impl_perm!(PermRef<'_, I>, &ColRef<'_, CC, TT>, Col<C, T>);
impl_perm!(PermRef<'_, I>, &ColMut<'_, CC, TT>, Col<C, T>);
impl_perm!(PermRef<'_, I>, &Col<CC, TT>, Col<C, T>);
impl_perm!(&PermRef<'_, I>, ColRef<'_, CC, TT>, Col<C, T>);
impl_perm!(&PermRef<'_, I>, ColMut<'_, CC, TT>, Col<C, T>);
impl_perm!(&PermRef<'_, I>, Col<CC, TT>, Col<C, T>);
impl_perm!(&PermRef<'_, I>, &ColRef<'_, CC, TT>, Col<C, T>);
impl_perm!(&PermRef<'_, I>, &ColMut<'_, CC, TT>, Col<C, T>);
impl_perm!(&PermRef<'_, I>, &Col<CC, TT>, Col<C, T>);

impl_perm!(Perm<I>, ColRef<'_, CC, TT>, Col<C, T>);
impl_perm!(Perm<I>, ColMut<'_, CC, TT>, Col<C, T>);
impl_perm!(Perm<I>, Col<CC, TT>, Col<C, T>);
impl_perm!(Perm<I>, &ColRef<'_, CC, TT>, Col<C, T>);
impl_perm!(Perm<I>, &ColMut<'_, CC, TT>, Col<C, T>);
impl_perm!(Perm<I>, &Col<CC, TT>, Col<C, T>);
impl_perm!(&Perm<I>, ColRef<'_, CC, TT>, Col<C, T>);
impl_perm!(&Perm<I>, ColMut<'_, CC, TT>, Col<C, T>);
impl_perm!(&Perm<I>, Col<CC, TT>, Col<C, T>);
impl_perm!(&Perm<I>, &ColRef<'_, CC, TT>, Col<C, T>);
impl_perm!(&Perm<I>, &ColMut<'_, CC, TT>, Col<C, T>);
impl_perm!(&Perm<I>, &Col<CC, TT>, Col<C, T>);

impl<
        I: Index,
        C: ComplexContainer,
        CC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        TT: ConjUnit<Canonical = T>,
    > Mul<PermRef<'_, I>> for MatRef<'_, CC, TT>
{
    type Output = Mat<C, T>;

    #[track_caller]
    #[math]
    fn mul(self, rhs: PermRef<'_, I>) -> Self::Output {
        let lhs = self;

        Assert!(lhs.ncols() == rhs.len());
        let ctx = &Ctx::<C, T>(T::MathCtx::default());
        let mut out = Mat::zeros_with_ctx(ctx, lhs.nrows(), lhs.ncols());
        let inv = rhs.arrays().1;

        for (j, inv) in inv.iter().enumerate() {
            for i in 0..lhs.nrows() {
                let lhs = lhs.at(i, inv.to_signed().zx());

                help!(C);
                math(write1!(
                    out.as_mut().at_mut(i, j),
                    Conj::apply::<CC, TT>(ctx, lhs)
                ));
            }
        }
        out
    }
}

// impl_perm!(MatRef<'_, CC, TT>, PermRef<'_, I>, Mat<C, T>);
impl_perm!(MatRef<'_, CC, TT>, Perm<I>, Mat<C, T>);
impl_perm!(MatRef<'_, CC, TT>, &PermRef<'_, I>, Mat<C, T>);
impl_perm!(MatRef<'_, CC, TT>, &Perm<I>, Mat<C, T>);
impl_perm!(&MatRef<'_, CC, TT>, PermRef<'_, I>, Mat<C, T>);
impl_perm!(&MatRef<'_, CC, TT>, Perm<I>, Mat<C, T>);
impl_perm!(&MatRef<'_, CC, TT>, &PermRef<'_, I>, Mat<C, T>);
impl_perm!(&MatRef<'_, CC, TT>, &Perm<I>, Mat<C, T>);

impl_perm!(MatMut<'_, CC, TT>, PermRef<'_, I>, Mat<C, T>);
impl_perm!(MatMut<'_, CC, TT>, Perm<I>, Mat<C, T>);
impl_perm!(MatMut<'_, CC, TT>, &PermRef<'_, I>, Mat<C, T>);
impl_perm!(MatMut<'_, CC, TT>, &Perm<I>, Mat<C, T>);
impl_perm!(&MatMut<'_, CC, TT>, PermRef<'_, I>, Mat<C, T>);
impl_perm!(&MatMut<'_, CC, TT>, Perm<I>, Mat<C, T>);
impl_perm!(&MatMut<'_, CC, TT>, &PermRef<'_, I>, Mat<C, T>);
impl_perm!(&MatMut<'_, CC, TT>, &Perm<I>, Mat<C, T>);

impl_perm!(Mat<CC, TT>, PermRef<'_, I>, Mat<C, T>);
impl_perm!(Mat<CC, TT>, Perm<I>, Mat<C, T>);
impl_perm!(Mat<CC, TT>, &PermRef<'_, I>, Mat<C, T>);
impl_perm!(Mat<CC, TT>, &Perm<I>, Mat<C, T>);
impl_perm!(&Mat<CC, TT>, PermRef<'_, I>, Mat<C, T>);
impl_perm!(&Mat<CC, TT>, Perm<I>, Mat<C, T>);
impl_perm!(&Mat<CC, TT>, &PermRef<'_, I>, Mat<C, T>);
impl_perm!(&Mat<CC, TT>, &Perm<I>, Mat<C, T>);

impl<
        I: Index,
        C: ComplexContainer,
        CC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        TT: ConjUnit<Canonical = T>,
    > Mul<PermRef<'_, I>> for RowRef<'_, CC, TT>
{
    type Output = Row<C, T>;

    #[track_caller]
    #[math]
    fn mul(self, rhs: PermRef<'_, I>) -> Self::Output {
        let lhs = self;

        Assert!(lhs.ncols() == rhs.len());
        let ctx = &Ctx::<C, T>(T::MathCtx::default());
        let mut out = Row::zeros_with_ctx(ctx, lhs.ncols());
        let inv = rhs.arrays().1;

        for (j, inv) in inv.iter().enumerate() {
            let lhs = lhs.at(inv.to_signed().zx());

            help!(C);
            math(write1!(
                out.as_mut().at_mut(j),
                Conj::apply::<CC, TT>(ctx, lhs)
            ));
        }
        out
    }
}

// impl_perm!(RowRef<'_, CC, TT>, PermRef<'_, I>, Row<C, T>);
impl_perm!(RowRef<'_, CC, TT>, Perm<I>, Row<C, T>);
impl_perm!(RowRef<'_, CC, TT>, &PermRef<'_, I>, Row<C, T>);
impl_perm!(RowRef<'_, CC, TT>, &Perm<I>, Row<C, T>);
impl_perm!(&RowRef<'_, CC, TT>, PermRef<'_, I>, Row<C, T>);
impl_perm!(&RowRef<'_, CC, TT>, Perm<I>, Row<C, T>);
impl_perm!(&RowRef<'_, CC, TT>, &PermRef<'_, I>, Row<C, T>);
impl_perm!(&RowRef<'_, CC, TT>, &Perm<I>, Row<C, T>);

impl_perm!(RowMut<'_, CC, TT>, PermRef<'_, I>, Row<C, T>);
impl_perm!(RowMut<'_, CC, TT>, Perm<I>, Row<C, T>);
impl_perm!(RowMut<'_, CC, TT>, &PermRef<'_, I>, Row<C, T>);
impl_perm!(RowMut<'_, CC, TT>, &Perm<I>, Row<C, T>);
impl_perm!(&RowMut<'_, CC, TT>, PermRef<'_, I>, Row<C, T>);
impl_perm!(&RowMut<'_, CC, TT>, Perm<I>, Row<C, T>);
impl_perm!(&RowMut<'_, CC, TT>, &PermRef<'_, I>, Row<C, T>);
impl_perm!(&RowMut<'_, CC, TT>, &Perm<I>, Row<C, T>);

impl_perm!(Row<CC, TT>, PermRef<'_, I>, Row<C, T>);
impl_perm!(Row<CC, TT>, Perm<I>, Row<C, T>);
impl_perm!(Row<CC, TT>, &PermRef<'_, I>, Row<C, T>);
impl_perm!(Row<CC, TT>, &Perm<I>, Row<C, T>);
impl_perm!(&Row<CC, TT>, PermRef<'_, I>, Row<C, T>);
impl_perm!(&Row<CC, TT>, Perm<I>, Row<C, T>);
impl_perm!(&Row<CC, TT>, &PermRef<'_, I>, Row<C, T>);
impl_perm!(&Row<CC, TT>, &Perm<I>, Row<C, T>);

impl<
        C: ComplexContainer,
        LhsC: Container<Canonical = C>,
        RhsC: Container<Canonical = C>,
        T: ComplexField<C, MathCtx: Default>,
        LhsT: ConjUnit<Canonical = T>,
        RhsT: ConjUnit<Canonical = T>,
    > Mul<ScaleGeneric<RhsC, RhsT>> for MatRef<'_, LhsC, LhsT>
{
    type Output = Mat<C, T>;

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
    > Mul<MatRef<'_, RhsC, RhsT>> for ScaleGeneric<LhsC, LhsT>
{
    type Output = Mat<C, T>;

    #[math]
    fn mul(self, rhs: MatRef<'_, RhsC, RhsT>) -> Self::Output {
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
    > Mul<ScaleGeneric<RhsC, RhsT>> for ColRef<'_, LhsC, LhsT>
{
    type Output = Col<C, T>;

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
    > Mul<ColRef<'_, RhsC, RhsT>> for ScaleGeneric<LhsC, LhsT>
{
    type Output = Col<C, T>;

    #[math]
    fn mul(self, rhs: ColRef<'_, RhsC, RhsT>) -> Self::Output {
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
    > Mul<ScaleGeneric<RhsC, RhsT>> for RowRef<'_, LhsC, LhsT>
{
    type Output = Row<C, T>;

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
    > Mul<RowRef<'_, RhsC, RhsT>> for ScaleGeneric<LhsC, LhsT>
{
    type Output = Row<C, T>;

    #[math]
    fn mul(self, rhs: RowRef<'_, RhsC, RhsT>) -> Self::Output {
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
    > Mul<ScaleGeneric<RhsC, RhsT>> for DiagRef<'_, LhsC, LhsT>
{
    type Output = Diag<C, T>;

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
    > Mul<DiagRef<'_, RhsC, RhsT>> for ScaleGeneric<LhsC, LhsT>
{
    type Output = Diag<C, T>;

    fn mul(self, rhs: DiagRef<'_, RhsC, RhsT>) -> Self::Output {
        (self * rhs.column_vector()).into_diagonal()
    }
}

// impl_mul_scalar!(MatRef<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Mat<C, T>);
impl_mul_scalar!(MatMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Mat<C, T>);
impl_mul_scalar!(Mat<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Mat<C, T>);
impl_mul_scalar!(&MatRef<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Mat<C, T>);
impl_mul_scalar!(&MatMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Mat<C, T>);
impl_mul_scalar!(&Mat<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Mat<C, T>);

impl_div_scalar!(MatRef<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Mat<C, T>);
impl_div_scalar!(MatMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Mat<C, T>);
impl_div_scalar!(Mat<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Mat<C, T>);
impl_div_scalar!(&MatRef<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Mat<C, T>);
impl_div_scalar!(&MatMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Mat<C, T>);
impl_div_scalar!(&Mat<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Mat<C, T>);

// impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, MatRef<'_, RhsC, RhsT>, Mat<C, T>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, MatMut<'_, RhsC, RhsT>, Mat<C, T>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, Mat<RhsC, RhsT>, Mat<C, T>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &MatRef<'_, RhsC, RhsT>, Mat<C, T>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &MatMut<'_, RhsC, RhsT>, Mat<C, T>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &Mat<RhsC, RhsT>, Mat<C, T>);

impl_mul_primitive!(MatRef<'_, RhsC, RhsT>, Mat<C, T>);
impl_mul_primitive!(MatMut<'_, RhsC, RhsT>, Mat<C, T>);
impl_mul_primitive!(Mat<RhsC, RhsT>, Mat<C, T>);
impl_mul_primitive!(&MatRef<'_, RhsC, RhsT>, Mat<C, T>);
impl_mul_primitive!(&MatMut<'_, RhsC, RhsT>, Mat<C, T>);
impl_mul_primitive!(&Mat<RhsC, RhsT>, Mat<C, T>);

// impl_mul_scalar!(ColRef<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Col<C, T>);
impl_mul_scalar!(ColMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Col<C, T>);
impl_mul_scalar!(Col<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Col<C, T>);
impl_mul_scalar!(&ColRef<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Col<C, T>);
impl_mul_scalar!(&ColMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Col<C, T>);
impl_mul_scalar!(&Col<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Col<C, T>);

impl_div_scalar!(ColRef<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Col<C, T>);
impl_div_scalar!(ColMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Col<C, T>);
impl_div_scalar!(Col<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Col<C, T>);
impl_div_scalar!(&ColRef<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Col<C, T>);
impl_div_scalar!(&ColMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Col<C, T>);
impl_div_scalar!(&Col<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Col<C, T>);

// impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, ColRef<'_, RhsC, RhsT>, Col<C, T>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, ColMut<'_, RhsC, RhsT>, Col<C, T>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, Col<RhsC, RhsT>, Col<C, T>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &ColRef<'_, RhsC, RhsT>, Col<C, T>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &ColMut<'_, RhsC, RhsT>, Col<C, T>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &Col<RhsC, RhsT>, Col<C, T>);

impl_mul_primitive!(ColRef<'_, RhsC, RhsT>, Col<C, T>);
impl_mul_primitive!(ColMut<'_, RhsC, RhsT>, Col<C, T>);
impl_mul_primitive!(Col<RhsC, RhsT>, Col<C, T>);
impl_mul_primitive!(&ColRef<'_, RhsC, RhsT>, Col<C, T>);
impl_mul_primitive!(&ColMut<'_, RhsC, RhsT>, Col<C, T>);
impl_mul_primitive!(&Col<RhsC, RhsT>, Col<C, T>);

// impl_mul_scalar!(RowRef<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Row<C, T>);
impl_mul_scalar!(RowMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Row<C, T>);
impl_mul_scalar!(Row<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Row<C, T>);
impl_mul_scalar!(&RowRef<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Row<C, T>);
impl_mul_scalar!(&RowMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Row<C, T>);
impl_mul_scalar!(&Row<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Row<C, T>);

impl_div_scalar!(RowRef<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Row<C, T>);
impl_div_scalar!(RowMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Row<C, T>);
impl_div_scalar!(Row<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Row<C, T>);
impl_div_scalar!(&RowRef<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Row<C, T>);
impl_div_scalar!(&RowMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Row<C, T>);
impl_div_scalar!(&Row<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Row<C, T>);

// impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, RowRef<'_, RhsC, RhsT>, Row<C, T>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, RowMut<'_, RhsC, RhsT>, Row<C, T>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, Row<RhsC, RhsT>, Row<C, T>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &RowRef<'_, RhsC, RhsT>, Row<C, T>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &RowMut<'_, RhsC, RhsT>, Row<C, T>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &Row<RhsC, RhsT>, Row<C, T>);

impl_mul_primitive!(RowRef<'_, RhsC, RhsT>, Row<C, T>);
impl_mul_primitive!(RowMut<'_, RhsC, RhsT>, Row<C, T>);
impl_mul_primitive!(Row<RhsC, RhsT>, Row<C, T>);
impl_mul_primitive!(&RowRef<'_, RhsC, RhsT>, Row<C, T>);
impl_mul_primitive!(&RowMut<'_, RhsC, RhsT>, Row<C, T>);
impl_mul_primitive!(&Row<RhsC, RhsT>, Row<C, T>);

// impl_mul_scalar!(DiagRef<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Diag<C, T>);
impl_mul_scalar!(DiagMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Diag<C, T>);
impl_mul_scalar!(Diag<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Diag<C, T>);
impl_mul_scalar!(&DiagRef<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Diag<C, T>);
impl_mul_scalar!(&DiagMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Diag<C, T>);
impl_mul_scalar!(&Diag<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Diag<C, T>);

impl_div_scalar!(DiagRef<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Diag<C, T>);
impl_div_scalar!(DiagMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Diag<C, T>);
impl_div_scalar!(Diag<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Diag<C, T>);
impl_div_scalar!(&DiagRef<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Diag<C, T>);
impl_div_scalar!(&DiagMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Diag<C, T>);
impl_div_scalar!(&Diag<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>, Diag<C, T>);

// impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, Diag<RhsC, RhsT>, Diag<C, T>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
impl_scalar_mul!(ScaleGeneric<LhsC, LhsT>, &Diag<RhsC, RhsT>, Diag<C, T>);

impl_mul_primitive!(DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
impl_mul_primitive!(DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
impl_mul_primitive!(Diag<RhsC, RhsT>, Diag<C, T>);
impl_mul_primitive!(&DiagRef<'_, RhsC, RhsT>, Diag<C, T>);
impl_mul_primitive!(&DiagMut<'_, RhsC, RhsT>, Diag<C, T>);
impl_mul_primitive!(&Diag<RhsC, RhsT>, Diag<C, T>);

impl<
        LhsC: ComplexContainer,
        RhsC: Container<Canonical = LhsC>,
        LhsT: ComplexField<LhsC, MathCtx: Default>,
        RhsT: ConjUnit<Canonical = LhsT>,
    > MulAssign<ScaleGeneric<RhsC, RhsT>> for MatMut<'_, LhsC, LhsT>
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
    > MulAssign<ScaleGeneric<RhsC, RhsT>> for ColMut<'_, LhsC, LhsT>
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
    > MulAssign<ScaleGeneric<RhsC, RhsT>> for RowMut<'_, LhsC, LhsT>
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
    > MulAssign<ScaleGeneric<RhsC, RhsT>> for DiagMut<'_, LhsC, LhsT>
{
    fn mul_assign(&mut self, rhs: ScaleGeneric<RhsC, RhsT>) {
        let mut this = self.rb_mut().column_vector_mut();
        this *= rhs;
    }
}

// impl_mul_assign_scalar!(MatMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);
impl_mul_assign_scalar!(Mat<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);
// impl_mul_assign_scalar!(ColMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);
impl_mul_assign_scalar!(Col<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);
// impl_mul_assign_scalar!(RowMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);
impl_mul_assign_scalar!(Row<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);
// impl_mul_assign_scalar!(DiagMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);
impl_mul_assign_scalar!(Diag<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);

impl_div_assign_scalar!(MatMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);
impl_div_assign_scalar!(Mat<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);
impl_div_assign_scalar!(ColMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);
impl_div_assign_scalar!(Col<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);
impl_div_assign_scalar!(RowMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);
impl_div_assign_scalar!(Row<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);
impl_div_assign_scalar!(DiagMut<'_, LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);
impl_div_assign_scalar!(Diag<LhsC, LhsT>, ScaleGeneric<RhsC, RhsT>);

impl_mul_assign_primitive!(MatMut<'_, LhsC, LhsT>);
impl_mul_assign_primitive!(Mat<LhsC, LhsT>);
impl_mul_assign_primitive!(ColMut<'_, LhsC, LhsT>);
impl_mul_assign_primitive!(Col<LhsC, LhsT>);
impl_mul_assign_primitive!(RowMut<'_, LhsC, LhsT>);
impl_mul_assign_primitive!(Row<LhsC, LhsT>);
impl_mul_assign_primitive!(DiagMut<'_, LhsC, LhsT>);
impl_mul_assign_primitive!(Diag<LhsC, LhsT>);

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
