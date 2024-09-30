use core::{fmt, marker::PhantomData};
use generativity::Guard;

use crate::{Index, Shape, ShapeIdx, Unbind};

type Invariant<'a> = fn(&'a ()) -> &'a ();

#[derive(Copy, Clone)]
pub struct Partition<'head, 'tail, 'n> {
    pub head: Dim<'head>,
    pub tail: Dim<'tail>,
    __marker: PhantomData<Invariant<'n>>,
}

impl<'head, 'tail, 'n> Partition<'head, 'tail, 'n> {
    #[inline]
    pub const fn midpoint(&self) -> IdxInc<'n> {
        unsafe { IdxInc::new_unbound(self.head.unbound) }
    }
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Dim<'n> {
    unbound: usize,
    __marker: PhantomData<Invariant<'n>>,
}
impl PartialEq for Dim<'_> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        equator::debug_assert!(self.unbound == other.unbound);
        true
    }
}
impl Eq for Dim<'_> {}

impl PartialOrd for Dim<'_> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        equator::debug_assert!(self.unbound == other.unbound);
        Some(core::cmp::Ordering::Equal)
    }
}
impl Ord for Dim<'_> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        equator::debug_assert!(self.unbound == other.unbound);
        core::cmp::Ordering::Equal
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Idx<'n, I: Index = usize> {
    unbound: I,
    __marker: PhantomData<Invariant<'n>>,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct IdxInc<'n, I: Index = usize> {
    unbound: I,
    __marker: PhantomData<Invariant<'n>>,
}

impl fmt::Debug for Dim<'_> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.unbound.fmt(f)
    }
}
impl<I: Index> fmt::Debug for Idx<'_, I> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.unbound.fmt(f)
    }
}
impl<I: Index> fmt::Debug for IdxInc<'_, I> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.unbound.fmt(f)
    }
}

impl<'n, I: Index> PartialEq<Dim<'n>> for Idx<'n, I> {
    #[inline(always)]
    fn eq(&self, other: &Dim<'n>) -> bool {
        equator::debug_assert!(self.unbound.zx() < other.unbound);

        false
    }
}
impl<'n, I: Index> PartialOrd<Dim<'n>> for Idx<'n, I> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Dim<'n>) -> Option<core::cmp::Ordering> {
        equator::debug_assert!(self.unbound.zx() < other.unbound);

        Some(core::cmp::Ordering::Less)
    }
}

impl<'n, I: Index> PartialEq<Dim<'n>> for IdxInc<'n, I> {
    #[inline(always)]
    fn eq(&self, other: &Dim<'n>) -> bool {
        equator::debug_assert!(self.unbound.zx() <= other.unbound);

        self.unbound.zx() == other.unbound
    }
}

impl<'n, I: Index> PartialOrd<Dim<'n>> for IdxInc<'n, I> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Dim<'n>) -> Option<core::cmp::Ordering> {
        equator::debug_assert!(self.unbound.zx() <= other.unbound);

        Some(if self.unbound.zx() == other.unbound {
            core::cmp::Ordering::Equal
        } else {
            core::cmp::Ordering::Less
        })
    }
}

impl<'n> Dim<'n> {
    #[inline(always)]
    pub const unsafe fn new_unbound(dim: usize) -> Self {
        Self {
            unbound: dim,
            __marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn new(dim: usize, guard: Guard<'n>) -> Self {
        _ = guard;
        Self {
            unbound: dim,
            __marker: PhantomData,
        }
    }

    #[inline(always)]
    pub const fn unbound(self) -> usize {
        self.unbound
    }

    #[inline]
    pub const fn partition<'head, 'tail>(
        self,
        midpoint: IdxInc<'n>,
        head: Guard<'head>,
        tail: Guard<'tail>,
    ) -> Partition<'head, 'tail, 'n> {
        _ = (head, tail);
        unsafe {
            Partition {
                head: Dim::new_unbound(midpoint.unbound),
                tail: Dim::new_unbound(self.unbound - midpoint.unbound),
                __marker: PhantomData,
            }
        }
    }

    #[inline]
    pub fn indices(self) -> impl Clone + ExactSizeIterator + DoubleEndedIterator<Item = Idx<'n>> {
        (0..self.unbound).map(|i| unsafe { Idx::new_unbound(i) })
    }
}

impl<'n, I: Index> Idx<'n, I> {
    #[inline(always)]
    pub const unsafe fn new_unbound(idx: I) -> Self {
        Self {
            unbound: idx,
            __marker: PhantomData,
        }
    }

    #[inline(always)]
    pub unsafe fn new_unchecked(idx: I, dim: Dim<'n>) -> Self {
        equator::debug_assert!(idx.zx() < dim.unbound);

        Self {
            unbound: idx,
            __marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn new_checked(idx: I, dim: Dim<'n>) -> Self {
        equator::assert!(idx.zx() < dim.unbound);

        Self {
            unbound: idx,
            __marker: PhantomData,
        }
    }

    #[inline(always)]
    pub const fn unbound(self) -> I {
        self.unbound
    }

    #[inline(always)]
    pub fn zx(self) -> Idx<'n, usize> {
        Idx {
            unbound: self.unbound.zx(),
            __marker: PhantomData,
        }
    }
}
impl<'n> Idx<'n> {
    #[inline(always)]
    pub const fn next(self) -> IdxInc<'n> {
        unsafe { IdxInc::new_unbound(self.unbound + 1) }
    }
}

impl<'n, I: Index> IdxInc<'n, I> {
    #[inline(always)]
    pub const unsafe fn new_unbound(idx: I) -> Self {
        Self {
            unbound: idx,
            __marker: PhantomData,
        }
    }

    #[inline(always)]
    pub unsafe fn new_unchecked(idx: I, dim: Dim<'n>) -> Self {
        equator::debug_assert!(idx.zx() <= dim.unbound);

        Self {
            unbound: idx,
            __marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn new_checked(idx: I, dim: Dim<'n>) -> Self {
        equator::assert!(idx.zx() <= dim.unbound);

        Self {
            unbound: idx,
            __marker: PhantomData,
        }
    }

    #[inline(always)]
    pub const fn unbound(self) -> I {
        self.unbound
    }

    #[inline(always)]
    pub fn zx(self) -> IdxInc<'n, usize> {
        IdxInc {
            unbound: self.unbound.zx(),
            __marker: PhantomData,
        }
    }
}

impl<'n> IdxInc<'n> {
    #[inline]
    pub fn to(
        self,
        upper: IdxInc<'n>,
    ) -> impl Clone + ExactSizeIterator + DoubleEndedIterator<Item = Idx<'n>> {
        (self.unbound..upper.unbound).map(|i| unsafe { Idx::new_unbound(i) })
    }
}

impl Unbind for Dim<'_> {
    #[inline(always)]
    unsafe fn new_unbound(idx: usize) -> Self {
        Self::new_unbound(idx)
    }

    #[inline(always)]
    fn unbound(self) -> usize {
        self.unbound
    }
}
impl<I: Index> Unbind<I> for Idx<'_, I> {
    #[inline(always)]
    unsafe fn new_unbound(idx: I) -> Self {
        Self::new_unbound(idx)
    }

    #[inline(always)]
    fn unbound(self) -> I {
        self.unbound
    }
}
impl<I: Index> Unbind<I> for IdxInc<'_, I> {
    #[inline(always)]
    unsafe fn new_unbound(idx: I) -> Self {
        Self::new_unbound(idx)
    }

    #[inline(always)]
    fn unbound(self) -> I {
        self.unbound
    }
}

impl<'dim> ShapeIdx for Dim<'dim> {
    type Idx<I: Index> = Idx<'dim, I>;
    type IdxInc<I: Index> = IdxInc<'dim, I>;
}

impl<'dim> Shape for Dim<'dim> {}

impl<'n, I: Index> From<Idx<'n, I>> for IdxInc<'n, I> {
    #[inline(always)]
    fn from(value: Idx<'n, I>) -> Self {
        Self {
            unbound: value.unbound,
            __marker: PhantomData,
        }
    }
}
