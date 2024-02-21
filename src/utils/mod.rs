#[inline(always)]
pub(crate) unsafe fn unchecked_mul(a: usize, b: isize) -> isize {
    let (sum, overflow) = (a as isize).overflowing_mul(b);
    if overflow {
        core::hint::unreachable_unchecked();
    }
    sum
}

#[inline(always)]
pub(crate) unsafe fn unchecked_add(a: isize, b: isize) -> isize {
    let (sum, overflow) = a.overflowing_add(b);
    if overflow {
        core::hint::unreachable_unchecked();
    }
    sum
}

#[doc(hidden)]
pub(crate) trait DivCeil: Sized {
    fn msrv_div_ceil(self, rhs: Self) -> Self;
    fn msrv_checked_next_multiple_of(self, rhs: Self) -> Option<Self>;
}

impl DivCeil for usize {
    #[inline]
    fn msrv_div_ceil(self, rhs: Self) -> Self {
        let d = self / rhs;
        let r = self % rhs;
        if r > 0 {
            d + 1
        } else {
            d
        }
    }

    #[inline]
    fn msrv_checked_next_multiple_of(self, rhs: Self) -> Option<Self> {
        {
            match self.checked_rem(rhs)? {
                0 => Some(self),
                r => self.checked_add(rhs - r),
            }
        }
    }
}

/// Index and matrix types with compile time checks, whichh can replace bound checks at runtime.
pub mod constrained;
/// Simd operations for a specific type satisfying [`ComplexField`](crate::ComplexField).
pub mod simd;
/// Slice types for [entities](crate::Entity).
pub mod slice;
/// Utilities relating to threading and parallelism.
pub mod thread;
/// Vector type for [entities](crate::Entity).
pub mod vec;
