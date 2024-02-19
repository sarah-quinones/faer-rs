use super::*;
use crate::{assert, sparse::__get_unchecked, utils::slice::*};
use core::ops::Range;

/// Symbolic structure view with dimensions equal to the values tied to `('nrows, 'ncols)`,
/// in column-major order.
#[repr(transparent)]
pub struct SymbolicSparseColMatRef<'nrows, 'ncols, 'a, I: Index>(
    Branded<'ncols, Branded<'nrows, crate::sparse::SymbolicSparseColMatRef<'a, I>>>,
);
/// Immutable sparse matrix view with dimensions equal to the values tied to `('nrows,
/// 'ncols)`, in column-major order.
pub struct SparseColMatRef<'nrows, 'ncols, 'a, I: Index, E: Entity> {
    symbolic: SymbolicSparseColMatRef<'nrows, 'ncols, 'a, I>,
    values: SliceGroup<'a, E>,
}
/// Mutable sparse matrix view with dimensions equal to the values tied to `('nrows,
/// 'ncols)`, in column-major order.
pub struct SparseColMatMut<'nrows, 'ncols, 'a, I: Index, E: Entity> {
    symbolic: SymbolicSparseColMatRef<'nrows, 'ncols, 'a, I>,
    values: SliceGroupMut<'a, E>,
}

impl<'nrows, 'ncols, 'a, I: Index> SymbolicSparseColMatRef<'nrows, 'ncols, 'a, I> {
    /// Returns a new symbolic structure after checking that its dimensions match the
    /// dimensions tied to `('nrows, 'ncols)`.
    #[inline]
    pub fn new(
        inner: crate::sparse::SymbolicSparseColMatRef<'a, I>,
        nrows: Size<'nrows>,
        ncols: Size<'ncols>,
    ) -> Self {
        assert!(all(
            inner.nrows() == nrows.into_inner(),
            inner.ncols() == ncols.into_inner(),
        ));
        Self(Branded {
            __marker: PhantomData,
            inner: Branded {
                __marker: PhantomData,
                inner,
            },
        })
    }

    /// Returns the unconstrained symbolic structure.
    #[inline]
    pub fn into_inner(self) -> crate::sparse::SymbolicSparseColMatRef<'a, I> {
        self.0.inner.inner
    }

    /// Returns the number of rows of the matrix.
    #[inline]
    pub fn nrows(&self) -> Size<'nrows> {
        unsafe { Size::new_raw_unchecked(self.0.inner.inner.nrows()) }
    }

    /// Returns the number of columns of the matrix.
    #[inline]
    pub fn ncols(&self) -> Size<'ncols> {
        unsafe { Size::new_raw_unchecked(self.0.inner.inner.ncols()) }
    }

    #[inline]
    #[track_caller]
    #[doc(hidden)]
    pub fn col_range(&self, j: Idx<'ncols, usize>) -> Range<usize> {
        unsafe { self.into_inner().col_range_unchecked(j.into_inner()) }
    }

    /// Returns the row indices in column `j`.
    #[inline]
    #[track_caller]
    pub fn row_indices_of_col_raw(&self, j: Idx<'ncols, usize>) -> &'a [Idx<'nrows, I>] {
        unsafe {
            &*(__get_unchecked(self.into_inner().row_indices(), self.col_range(j)) as *const [I]
                as *const [Idx<'_, I>])
        }
    }

    /// Returns the row indices in column `j`.
    #[inline]
    #[track_caller]
    pub fn row_indices_of_col(
        &self,
        j: Idx<'ncols, usize>,
    ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = Idx<'nrows, usize>> {
        unsafe {
            __get_unchecked(
                self.into_inner().row_indices(),
                self.into_inner().col_range_unchecked(j.into_inner()),
            )
            .iter()
            .map(
                #[inline(always)]
                move |&row| Idx::new_raw_unchecked(row.zx()),
            )
        }
    }
}

impl<'nrows, 'ncols, 'a, I: Index, E: Entity> SparseColMatRef<'nrows, 'ncols, 'a, I, E> {
    /// Returns a new matrix view after checking that its dimensions match the
    /// dimensions tied to `('nrows, 'ncols)`.
    pub fn new(
        inner: crate::sparse::SparseColMatRef<'a, I, E>,
        nrows: Size<'nrows>,
        ncols: Size<'ncols>,
    ) -> Self {
        assert!(all(
            inner.nrows() == nrows.into_inner(),
            inner.ncols() == ncols.into_inner(),
        ));
        Self {
            symbolic: SymbolicSparseColMatRef::new(inner.symbolic(), nrows, ncols),
            values: SliceGroup::new(inner.values()),
        }
    }

    /// Returns the unconstrained matrix.
    #[inline]
    pub fn into_inner(self) -> crate::sparse::SparseColMatRef<'a, I, E> {
        crate::sparse::SparseColMatRef::new(self.symbolic.into_inner(), self.values.into_inner())
    }

    /// Returns the values in column `j`.
    #[inline]
    pub fn values_of_col(&self, j: Idx<'ncols, usize>) -> GroupFor<E, &'a [E::Unit]> {
        unsafe {
            self.values
                .subslice_unchecked(self.col_range(j))
                .into_inner()
        }
    }

    /// Returns the symbolic structure of the matrix.
    #[inline]
    pub fn symbolic(&self) -> SymbolicSparseColMatRef<'nrows, 'ncols, 'a, I> {
        self.symbolic
    }
}

impl<'nrows, 'ncols, 'a, I: Index, E: Entity> SparseColMatMut<'nrows, 'ncols, 'a, I, E> {
    /// Returns a new matrix view after checking that its dimensions match the
    /// dimensions tied to `('nrows, 'ncols)`.
    pub fn new(
        inner: crate::sparse::SparseColMatMut<'a, I, E>,
        nrows: Size<'nrows>,
        ncols: Size<'ncols>,
    ) -> Self {
        assert!(all(
            inner.nrows() == nrows.into_inner(),
            inner.ncols() == ncols.into_inner(),
        ));
        Self {
            symbolic: SymbolicSparseColMatRef::new(inner.symbolic(), nrows, ncols),
            values: SliceGroupMut::new(inner.values_mut()),
        }
    }

    /// Returns the unconstrained matrix.
    #[inline]
    pub fn into_inner(self) -> crate::sparse::SparseColMatMut<'a, I, E> {
        crate::sparse::SparseColMatMut::new(self.symbolic.into_inner(), self.values.into_inner())
    }

    /// Returns the values in column `j`.
    #[inline]
    pub fn values_of_col_mut(&mut self, j: Idx<'ncols, usize>) -> GroupFor<E, &'_ mut [E::Unit]> {
        unsafe {
            let range = self.col_range(j);
            self.values.rb_mut().subslice_unchecked(range).into_inner()
        }
    }

    /// Returns the symbolic structure of the matrix.
    #[inline]
    pub fn symbolic(&self) -> SymbolicSparseColMatRef<'nrows, 'ncols, 'a, I> {
        self.symbolic
    }
}

impl<I: Index, E: Entity> Copy for SparseColMatRef<'_, '_, '_, I, E> {}
impl<I: Index, E: Entity> Clone for SparseColMatRef<'_, '_, '_, I, E> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<I: Index> Copy for SymbolicSparseColMatRef<'_, '_, '_, I> {}
impl<I: Index> Clone for SymbolicSparseColMatRef<'_, '_, '_, I> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'nrows, 'ncols, 'a, I: Index, E: Entity> core::ops::Deref
    for SparseColMatRef<'nrows, 'ncols, 'a, I, E>
{
    type Target = SymbolicSparseColMatRef<'nrows, 'ncols, 'a, I>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.symbolic
    }
}

impl<'nrows, 'ncols, 'a, I: Index, E: Entity> core::ops::Deref
    for SparseColMatMut<'nrows, 'ncols, 'a, I, E>
{
    type Target = SymbolicSparseColMatRef<'nrows, 'ncols, 'a, I>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.symbolic
    }
}

impl<'short, 'nrows, 'ncols, 'a, I: Index, E: Entity> ReborrowMut<'short>
    for SparseColMatRef<'nrows, 'ncols, 'a, I, E>
{
    type Target = SparseColMatRef<'nrows, 'ncols, 'short, I, E>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}

impl<'short, 'nrows, 'ncols, 'a, I: Index, E: Entity> Reborrow<'short>
    for SparseColMatRef<'nrows, 'ncols, 'a, I, E>
{
    type Target = SparseColMatRef<'nrows, 'ncols, 'short, I, E>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}

impl<'nrows, 'ncols, 'a, I: Index, E: Entity> IntoConst
    for SparseColMatRef<'nrows, 'ncols, 'a, I, E>
{
    type Target = SparseColMatRef<'nrows, 'ncols, 'a, I, E>;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}

impl<'short, 'nrows, 'ncols, 'a, I: Index, E: Entity> ReborrowMut<'short>
    for SparseColMatMut<'nrows, 'ncols, 'a, I, E>
{
    type Target = SparseColMatMut<'nrows, 'ncols, 'short, I, E>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        SparseColMatMut::<'nrows, 'ncols, 'short, I, E> {
            symbolic: self.symbolic,
            values: self.values.rb_mut(),
        }
    }
}

impl<'short, 'nrows, 'ncols, 'a, I: Index, E: Entity> Reborrow<'short>
    for SparseColMatMut<'nrows, 'ncols, 'a, I, E>
{
    type Target = SparseColMatRef<'nrows, 'ncols, 'short, I, E>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        SparseColMatRef::<'nrows, 'ncols, 'short, I, E> {
            symbolic: self.symbolic,
            values: self.values.rb(),
        }
    }
}

impl<'nrows, 'ncols, 'a, I: Index, E: Entity> IntoConst
    for SparseColMatMut<'nrows, 'ncols, 'a, I, E>
{
    type Target = SparseColMatRef<'nrows, 'ncols, 'a, I, E>;

    #[inline]
    fn into_const(self) -> Self::Target {
        SparseColMatRef::<'nrows, 'ncols, 'a, I, E> {
            symbolic: self.symbolic,
            values: self.values.into_const(),
        }
    }
}
