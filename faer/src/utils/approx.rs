use faer_traits::RealValue;

use crate::internal_prelude::*;

pub struct ApproxEq<C: ComplexContainer, T: ComplexField<C>> {
    pub ctx: Ctx<C, T>,
    pub abs_tol: RealValue<C, T>,
    pub rel_tol: RealValue<C, T>,
}

pub struct CwiseMat<Cmp>(pub Cmp);

impl<C: ComplexContainer, T: ComplexField<C>> ApproxEq<C, T> {
    #[math]
    #[inline]
    pub fn eps_with(ctx: Ctx<C, T>) -> Self {
        Self {
            abs_tol: math.re(eps() * from_f64(128.0)),
            rel_tol: math.re(eps() * from_f64(128.0)),
            ctx,
        }
    }

    #[inline]
    pub fn eps() -> Self
    where
        T::MathCtx: Default,
    {
        Self::eps_with(ctx())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ApproxEqError;
extern crate alloc;

#[derive(Clone, Debug)]
pub enum CwiseMatError<Rows: Shape, Cols: Shape, Error> {
    DimMismatch,
    Elements(alloc::vec::Vec<(crate::Idx<Rows>, crate::Idx<Cols>, Error)>),
}

impl<
        C: ComplexContainer,
        T: ComplexField<C>,
        Rows: Shape,
        Cols: Shape,
        L: AsMatRef<C = C, T = T, Rows = Rows, Cols = Cols>,
        R: AsMatRef<C = C, T = T, Rows = Rows, Cols = Cols>,
        Error: equator::CmpDisplay<Cmp, C::Of<T>, C::Of<T>>,
        Cmp: equator::Cmp<C::Of<T>, C::Of<T>, Error = Error>,
    > equator::CmpError<CwiseMat<Cmp>, L, R> for CwiseMat<Cmp>
{
    type Error = CwiseMatError<Rows, Cols, Error>;
}

impl<C: ComplexContainer, T: ComplexField<C>> equator::CmpError<ApproxEq<C, T>, C::Of<T>, C::Of<T>>
    for ApproxEq<C, T>
{
    type Error = ApproxEqError;
}

impl<C: ComplexContainer, T: ComplexField<C>>
    equator::CmpDisplay<ApproxEq<C, T>, C::Of<T>, C::Of<T>> for ApproxEqError
{
    #[math]
    fn fmt(
        &self,
        cmp: &ApproxEq<C, T>,
        lhs: &C::Of<T>,
        mut lhs_source: &str,
        lhs_debug: &dyn core::fmt::Debug,
        rhs: &C::Of<T>,
        rhs_source: &str,
        rhs_debug: &dyn core::fmt::Debug,
        f: &mut core::fmt::Formatter,
    ) -> core::fmt::Result {
        let ApproxEq {
            abs_tol,
            rel_tol,
            ctx,
        } = cmp;
        help!(C::Real);
        help2!(C);

        let abs_tol = wrap!(as_ref!(abs_tol));
        let rel_tol = wrap!(as_ref!(rel_tol));

        if let Some(source) = lhs_source.strip_prefix("__skip_prologue") {
            lhs_source = source;
        } else {
            writeln!(
                f,
                "Assertion failed: {lhs_source} ~ {rhs_source}\nwith absolute tolerance = {abs_tol:?}\nwith relative tolerance = {rel_tol:?}"
            )?;
        }

        let lhs = wrap2!(as_ref2!(*lhs));
        let rhs = wrap2!(as_ref2!(*rhs));
        let distance = wrap!(math(abs(lhs.0 - rhs.0)));

        write!(f, "- {lhs_source} = {lhs_debug:?}\n")?;
        write!(f, "- {rhs_source} = {rhs_debug:?}\n")?;
        write!(f, "- distance = {distance:?}")
    }
}

impl<C: ComplexContainer, T: ComplexField<C>> equator::Cmp<C::Of<T>, C::Of<T>> for ApproxEq<C, T> {
    #[math]
    fn test(&self, lhs: &C::Of<T>, rhs: &C::Of<T>) -> Result<(), Self::Error> {
        let Self {
            ctx,
            abs_tol,
            rel_tol,
        } = self;

        let diff = math(abs(lhs - rhs));
        let lhs = math.abs(lhs);
        let rhs = math.abs(rhs);
        let max = if math(lhs > rhs) { lhs } else { rhs };

        if math.re((is_zero(max) && diff <= abs_tol) || (diff <= abs_tol || diff <= rel_tol * max))
        {
            Ok(())
        } else {
            Err(ApproxEqError)
        }
    }
}

impl<
        C: ComplexContainer,
        T: ComplexField<C>,
        Rows: Shape,
        Cols: Shape,
        L: AsMatRef<C = C, T = T, Rows = Rows, Cols = Cols>,
        R: AsMatRef<C = C, T = T, Rows = Rows, Cols = Cols>,
        Error: equator::CmpDisplay<Cmp, C::Of<T>, C::Of<T>>,
        Cmp: equator::Cmp<C::Of<T>, C::Of<T>, Error = Error>,
    > equator::CmpDisplay<CwiseMat<Cmp>, L, R> for CwiseMatError<Rows, Cols, Error>
{
    #[math]
    fn fmt(
        &self,
        cmp: &CwiseMat<Cmp>,
        lhs: &L,
        lhs_source: &str,
        _: &dyn core::fmt::Debug,
        rhs: &R,
        rhs_source: &str,
        _: &dyn core::fmt::Debug,
        f: &mut core::fmt::Formatter,
    ) -> core::fmt::Result {
        let lhs = lhs.as_mat_ref();
        let rhs = rhs.as_mat_ref();
        match self {
            Self::DimMismatch => {
                let lhs_nrows = lhs.nrows();
                let lhs_ncols = lhs.ncols();
                let rhs_nrows = rhs.nrows();
                let rhs_ncols = rhs.ncols();

                writeln!(f, "Assertion failed: {lhs_source} ~ {rhs_source}\n")?;
                write!(f, "- {lhs_source} = Mat[{lhs_nrows:?}, {lhs_ncols:?}]\n")?;
                write!(f, "- {rhs_source} = Mat[{rhs_nrows:?}, {rhs_ncols:?}]")?;
            }

            Self::Elements(indices) => {
                let mut prefix = "";

                for (i, j, e) in indices {
                    let i = *i;
                    let j = *j;
                    help!(C);
                    let lhs = map!(lhs.at(i, j), ptr, ptr.clone());
                    let rhs = map!(rhs.at(i, j), ptr, ptr.clone());
                    e.fmt(
                        &cmp.0,
                        &lhs,
                        &alloc::format!("{prefix}{lhs_source} at ({i:?}, {j:?})"),
                        crate::hacks::hijack_debug(unsafe {
                            crate::hacks::coerce::<&C::Of<T>, &C::OfDebug<T>>(&lhs)
                        }),
                        &rhs,
                        &alloc::format!("{rhs_source} at ({i:?}, {j:?})"),
                        crate::hacks::hijack_debug(unsafe {
                            crate::hacks::coerce::<&C::Of<T>, &C::OfDebug<T>>(&rhs)
                        }),
                        f,
                    )?;
                    write!(f, "\n\n")?;
                    prefix = "__skip_prologue"
                }
            }
        }
        Ok(())
    }
}

impl<
        C: ComplexContainer,
        T: ComplexField<C>,
        Rows: Shape,
        Cols: Shape,
        L: AsMatRef<C = C, T = T, Rows = Rows, Cols = Cols>,
        R: AsMatRef<C = C, T = T, Rows = Rows, Cols = Cols>,
        Error: equator::CmpDisplay<Cmp, C::Of<T>, C::Of<T>>,
        Cmp: equator::Cmp<C::Of<T>, C::Of<T>, Error = Error>,
    > equator::Cmp<L, R> for CwiseMat<Cmp>
{
    fn test(&self, lhs: &L, rhs: &R) -> Result<(), Self::Error> {
        let lhs = lhs.as_mat_ref();
        let rhs = rhs.as_mat_ref();

        if lhs.nrows() != rhs.nrows() || lhs.ncols() != rhs.ncols() {
            return Err(CwiseMatError::DimMismatch);
        }

        let mut indices = alloc::vec::Vec::new();
        for j in 0..lhs.ncols().unbound() {
            let j = lhs.ncols().checked_idx(j);
            for i in 0..lhs.nrows().unbound() {
                let i = lhs.nrows().checked_idx(i);

                help!(C);
                if let Err(err) = self.0.test(
                    &map!(lhs.at(i, j), ptr, ptr.clone()),
                    &map!(rhs.at(i, j), ptr, ptr.clone()),
                ) {
                    indices.push((i, j, err));
                }
            }
        }

        if indices.is_empty() {
            Ok(())
        } else {
            Err(CwiseMatError::Elements(indices))
        }
    }
}
