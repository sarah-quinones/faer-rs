use faer_traits::RealValue;

use crate::internal_prelude::*;

pub struct ApproxEq<T: ComplexField> {
    pub abs_tol: RealValue<T>,
    pub rel_tol: RealValue<T>,
}

pub struct CwiseMat<Cmp>(pub Cmp);

impl<T: ComplexField> ApproxEq<T> {
    #[math]
    #[inline]
    pub fn eps() -> Self {
        Self {
            abs_tol: eps() * from_f64(128.0),
            rel_tol: eps() * from_f64(128.0),
        }
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
        T: ComplexField,
        Rows: Shape,
        Cols: Shape,
        L: AsMatRef<T = T, Rows = Rows, Cols = Cols>,
        R: AsMatRef<T = T, Rows = Rows, Cols = Cols>,
        Error: equator::CmpDisplay<Cmp, T, T>,
        Cmp: equator::Cmp<T, T, Error = Error>,
    > equator::CmpError<CwiseMat<Cmp>, L, R> for CwiseMat<Cmp>
{
    type Error = CwiseMatError<Rows, Cols, Error>;
}

impl<T: ComplexField> equator::CmpError<ApproxEq<T>, T, T> for ApproxEq<T> {
    type Error = ApproxEqError;
}

impl<T: ComplexField> equator::CmpDisplay<ApproxEq<T>, T, T> for ApproxEqError {
    #[math]
    fn fmt(
        &self,
        cmp: &ApproxEq<T>,
        lhs: &T,
        mut lhs_source: &str,
        lhs_debug: &dyn core::fmt::Debug,
        rhs: &T,
        rhs_source: &str,
        rhs_debug: &dyn core::fmt::Debug,
        f: &mut core::fmt::Formatter,
    ) -> core::fmt::Result {
        let ApproxEq { abs_tol, rel_tol } = cmp;

        if let Some(source) = lhs_source.strip_prefix("__skip_prologue") {
            lhs_source = source;
        } else {
            writeln!(
                f,
                "Assertion failed: {lhs_source} ~ {rhs_source}\nwith absolute tolerance = {abs_tol:?}\nwith relative tolerance = {rel_tol:?}"
            )?;
        }

        let distance = abs(*lhs - *rhs);

        write!(f, "- {lhs_source} = {lhs_debug:?}\n")?;
        write!(f, "- {rhs_source} = {rhs_debug:?}\n")?;
        write!(f, "- distance = {distance:?}")
    }
}

impl<T: ComplexField> equator::Cmp<T, T> for ApproxEq<T> {
    #[math]
    fn test(&self, lhs: &T, rhs: &T) -> Result<(), Self::Error> {
        let Self { abs_tol, rel_tol } = self;

        let diff = abs(*lhs - *rhs);
        let max = max(abs(*lhs), abs(*rhs));

        if (max == zero() && diff <= *abs_tol) || (diff <= *abs_tol || diff <= *rel_tol * max) {
            Ok(())
        } else {
            Err(ApproxEqError)
        }
    }
}

impl<
        T: ComplexField,
        Rows: Shape,
        Cols: Shape,
        L: AsMatRef<T = T, Rows = Rows, Cols = Cols>,
        R: AsMatRef<T = T, Rows = Rows, Cols = Cols>,
        Error: equator::CmpDisplay<Cmp, T, T>,
        Cmp: equator::Cmp<T, T, Error = Error>,
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
                    let lhs = lhs.at(i, j).clone();
                    let rhs = rhs.at(i, j).clone();
                    e.fmt(
                        &cmp.0,
                        &lhs,
                        &alloc::format!("{prefix}{lhs_source} at ({i:?}, {j:?})"),
                        crate::hacks::hijack_debug(&lhs),
                        &rhs,
                        &alloc::format!("{rhs_source} at ({i:?}, {j:?})"),
                        crate::hacks::hijack_debug(&rhs),
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
        T: ComplexField,
        Rows: Shape,
        Cols: Shape,
        L: AsMatRef<T = T, Rows = Rows, Cols = Cols>,
        R: AsMatRef<T = T, Rows = Rows, Cols = Cols>,
        Error: equator::CmpDisplay<Cmp, T, T>,
        Cmp: equator::Cmp<T, T, Error = Error>,
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

                if let Err(err) = self.0.test(&lhs.at(i, j).clone(), &rhs.at(i, j).clone()) {
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
