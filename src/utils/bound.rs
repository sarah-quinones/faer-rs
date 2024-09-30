use core::{fmt, marker::PhantomData};
use generativity::Guard;

use crate::{Shape, Unbind};

type Invariant<'a> = fn(&'a ()) -> &'a ();

#[derive(Copy, Clone)]
pub struct Partition<'head, 'tail, 'n> {
    pub head: Dim<'head>,
    pub tail: Dim<'tail>,
    __marker: PhantomData<Invariant<'n>>,
}

impl<'head, 'tail, 'n> Partition<'head, 'tail, 'n> {
    pub fn midpoint(&self) -> IdxInc<'n> {
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

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Idx<'n> {
    unbound: usize,
    __marker: PhantomData<Invariant<'n>>,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct IdxInc<'n> {
    unbound: usize,
    __marker: PhantomData<Invariant<'n>>,
}

impl fmt::Debug for Dim<'_> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.unbound.fmt(f)
    }
}
impl fmt::Debug for Idx<'_> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.unbound.fmt(f)
    }
}
impl fmt::Debug for IdxInc<'_> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.unbound.fmt(f)
    }
}

impl<'n> PartialEq<Dim<'n>> for Idx<'n> {
    #[inline(always)]
    fn eq(&self, other: &Dim<'n>) -> bool {
        equator::debug_assert!(self.unbound < other.unbound);

        false
    }
}
impl<'n> PartialOrd<Dim<'n>> for Idx<'n> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Dim<'n>) -> Option<core::cmp::Ordering> {
        equator::debug_assert!(self.unbound < other.unbound);

        Some(core::cmp::Ordering::Less)
    }
}

impl<'n> PartialEq<Dim<'n>> for IdxInc<'n> {
    #[inline(always)]
    fn eq(&self, other: &Dim<'n>) -> bool {
        equator::debug_assert!(self.unbound <= other.unbound);

        self.unbound == other.unbound
    }
}

impl<'n> PartialOrd<Dim<'n>> for IdxInc<'n> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Dim<'n>) -> Option<core::cmp::Ordering> {
        equator::debug_assert!(self.unbound <= other.unbound);

        Some(if self.unbound == other.unbound {
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
    pub fn unbound(self) -> usize {
        self.unbound
    }

    pub fn partition<'head, 'tail>(
        self,
        midpoint: IdxInc<'n>,
        head: Guard<'head>,
        tail: Guard<'tail>,
    ) -> Partition<'head, 'tail, 'n> {
        Partition {
            head: Dim::new(midpoint.unbound, head),
            tail: Dim::new(self.unbound - midpoint.unbound, tail),
            __marker: PhantomData,
        }
    }
}

impl<'n> Idx<'n> {
    #[inline(always)]
    pub const unsafe fn new_unbound(idx: usize) -> Self {
        Self {
            unbound: idx,
            __marker: PhantomData,
        }
    }

    #[inline(always)]
    pub unsafe fn new_unchecked(idx: usize, dim: Dim<'n>) -> Self {
        equator::debug_assert!(idx < dim.unbound);

        Self {
            unbound: idx,
            __marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn new_checked(idx: usize, dim: Dim<'n>) -> Self {
        equator::assert!(idx < dim.unbound);

        Self {
            unbound: idx,
            __marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn unbound(self) -> usize {
        self.unbound
    }
}

impl<'n> IdxInc<'n> {
    #[inline(always)]
    pub const unsafe fn new_unbound(idx: usize) -> Self {
        Self {
            unbound: idx,
            __marker: PhantomData,
        }
    }

    #[inline(always)]
    pub unsafe fn new_unchecked(idx: usize, dim: Dim<'n>) -> Self {
        equator::debug_assert!(idx <= dim.unbound);

        Self {
            unbound: idx,
            __marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn new_checked(idx: usize, dim: Dim<'n>) -> Self {
        equator::assert!(idx <= dim.unbound);

        Self {
            unbound: idx,
            __marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn unbound(self) -> usize {
        self.unbound
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
impl Unbind for Idx<'_> {
    #[inline(always)]
    unsafe fn new_unbound(idx: usize) -> Self {
        Self::new_unbound(idx)
    }

    #[inline(always)]
    fn unbound(self) -> usize {
        self.unbound
    }
}
impl Unbind for IdxInc<'_> {
    #[inline(always)]
    unsafe fn new_unbound(idx: usize) -> Self {
        Self::new_unbound(idx)
    }

    #[inline(always)]
    fn unbound(self) -> usize {
        self.unbound
    }
}

impl<'dim> Shape for Dim<'dim> {
    type Idx = Idx<'dim>;
    type IdxInc = IdxInc<'dim>;
}

impl<'n> From<Idx<'n>> for IdxInc<'n> {
    #[inline(always)]
    fn from(value: Idx<'n>) -> Self {
        Self {
            unbound: value.unbound,
            __marker: PhantomData,
        }
    }
}

// pub type One = Dim<'static>;
// pub const ZERO: Idx<'static> = unsafe { Idx::new_unbound(0) };
