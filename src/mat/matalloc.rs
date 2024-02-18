use core::mem::ManuallyDrop;
use super::*;
use crate::complex_native::*;

#[repr(C)]
pub struct RawMatUnit<T: 'static> {
    pub(crate) ptr: NonNull<T>,
    pub(crate) row_capacity: usize,
    pub(crate) col_capacity: usize,
}

#[repr(C)]
pub(crate) struct MatUnit<T: 'static> {
    pub(crate) raw: RawMatUnit<T>,
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
}

impl<T: 'static> RawMatUnit<T> {
    pub fn new(row_capacity: usize, col_capacity: usize) -> Self {
        let dangling = NonNull::<T>::dangling();
        if core::mem::size_of::<T>() == 0 {
            Self {
                ptr: dangling,
                row_capacity,
                col_capacity,
            }
        } else {
            let cap = row_capacity
                .checked_mul(col_capacity)
                .unwrap_or_else(capacity_overflow);
            let cap_bytes = cap
                .checked_mul(core::mem::size_of::<T>())
                .unwrap_or_else(capacity_overflow);
            if cap_bytes > isize::MAX as usize {
                capacity_overflow::<()>();
            }

            use alloc::alloc::{alloc, handle_alloc_error, Layout};

            let layout = Layout::from_size_align(cap_bytes, align_for::<T>())
                .ok()
                .unwrap_or_else(capacity_overflow);

            let ptr = if layout.size() == 0 {
                dangling
            } else {
                // SAFETY: we checked that layout has non zero size
                let ptr = unsafe { alloc(layout) } as *mut T;
                if ptr.is_null() {
                    handle_alloc_error(layout)
                }
                // SAFETY: we checked that the pointer is not null
                unsafe { NonNull::<T>::new_unchecked(ptr) }
            };

            Self {
                ptr,
                row_capacity,
                col_capacity,
            }
        }
    }
}

impl<T: 'static> Drop for RawMatUnit<T> {
    fn drop(&mut self) {
        use alloc::alloc::{dealloc, Layout};
        // this cannot overflow because we already allocated this much memory
        // self.row_capacity.wrapping_mul(self.col_capacity) may overflow if T is a zst
        // but that's fine since we immediately multiply it by 0.
        let alloc_size =
            self.row_capacity.wrapping_mul(self.col_capacity) * core::mem::size_of::<T>();
        if alloc_size != 0 {
            // SAFETY: pointer was allocated with alloc::alloc::alloc
            unsafe {
                dealloc(
                    self.ptr.as_ptr() as *mut u8,
                    Layout::from_size_align_unchecked(alloc_size, align_for::<T>()),
                );
            }
        }
    }
}

#[repr(C)]
pub struct RawMat<E: Entity> {
    pub(crate) ptr: GroupCopyFor<E, NonNull<E::Unit>>,
    pub(crate) row_capacity: usize,
    pub(crate) col_capacity: usize,
}

impl<E: Entity> RawMat<E> {
    pub fn new(row_capacity: usize, col_capacity: usize) -> Self {
        // allocate the unit matrices
        let group = E::faer_map(E::UNIT, |()| {
            RawMatUnit::<E::Unit>::new(row_capacity, col_capacity)
        });

        let group = E::faer_map(group, core::mem::ManuallyDrop::new);

        Self {
            ptr: into_copy::<E, _>(E::faer_map(group, |mat| mat.ptr)),
            row_capacity,
            col_capacity,
        }
    }
}

impl<E: Entity> Drop for RawMat<E> {
    fn drop(&mut self) {
        drop(E::faer_map(from_copy::<E, _>(self.ptr), |ptr| RawMatUnit {
            ptr,
            row_capacity: self.row_capacity,
            col_capacity: self.col_capacity,
        }));
    }
}

impl<T> MatUnit<T> {
    #[cold]
    pub fn do_reserve_exact(&mut self, mut new_row_capacity: usize, mut new_col_capacity: usize) {
        new_row_capacity = self.raw.row_capacity.max(new_row_capacity);
        new_col_capacity = self.raw.col_capacity.max(new_col_capacity);

        let new_ptr = if self.raw.row_capacity == new_row_capacity
            && self.raw.row_capacity != 0
            && self.raw.col_capacity != 0
        {
            // case 1:
            // we have enough row capacity, and we've already allocated memory.
            // use realloc to get extra column memory

            use alloc::alloc::{handle_alloc_error, realloc, Layout};

            // this shouldn't overflow since we already hold this many bytes
            let old_cap = self.raw.row_capacity * self.raw.col_capacity;
            let old_cap_bytes = old_cap * core::mem::size_of::<T>();

            let new_cap = new_row_capacity
                .checked_mul(new_col_capacity)
                .unwrap_or_else(capacity_overflow);
            let new_cap_bytes = new_cap
                .checked_mul(core::mem::size_of::<T>())
                .unwrap_or_else(capacity_overflow);

            if new_cap_bytes > isize::MAX as usize {
                capacity_overflow::<()>();
            }

            // SAFETY: this shouldn't overflow since we already checked that it's valid during
            // allocation
            let old_layout =
                unsafe { Layout::from_size_align_unchecked(old_cap_bytes, align_for::<T>()) };
            let new_layout = Layout::from_size_align(new_cap_bytes, align_for::<T>())
                .ok()
                .unwrap_or_else(capacity_overflow);

            // SAFETY:
            // * old_ptr is non null and is the return value of some previous call to alloc
            // * old_layout is the same layout that was used to provide the old allocation
            // * new_cap_bytes is non zero since new_row_capacity and new_col_capacity are larger
            // than self.raw.row_capacity and self.raw.col_capacity respectively, and the computed
            // product doesn't overflow.
            // * new_cap_bytes, when rounded up to the nearest multiple of the alignment does not
            // overflow, since we checked that we can create new_layout with it.
            unsafe {
                let old_ptr = self.raw.ptr.as_ptr();
                let new_ptr = realloc(old_ptr as *mut u8, old_layout, new_cap_bytes);
                if new_ptr.is_null() {
                    handle_alloc_error(new_layout);
                }
                new_ptr as *mut T
            }
        } else {
            // case 2:
            // use alloc and move stuff manually.

            // allocate new memory region
            let new_ptr = {
                let m = ManuallyDrop::new(RawMatUnit::<T>::new(new_row_capacity, new_col_capacity));
                m.ptr.as_ptr()
            };

            let old_ptr = self.raw.ptr.as_ptr();

            // copy each column to new matrix
            for j in 0..self.ncols {
                // SAFETY:
                // * pointer offsets can't overflow since they're within an already allocated
                // memory region less than isize::MAX bytes in size.
                // * new and old allocation can't overlap, so copy_nonoverlapping is fine here.
                unsafe {
                    let old_ptr = old_ptr.add(j * self.raw.row_capacity);
                    let new_ptr = new_ptr.add(j * new_row_capacity);
                    core::ptr::copy_nonoverlapping(old_ptr, new_ptr, self.nrows);
                }
            }

            // deallocate old matrix memory
            let _ = RawMatUnit::<T> {
                // SAFETY: this ptr was checked to be non null, or was acquired from a NonNull
                // pointer.
                ptr: unsafe { NonNull::new_unchecked(old_ptr) },
                row_capacity: self.raw.row_capacity,
                col_capacity: self.raw.col_capacity,
            };

            new_ptr
        };
        self.raw.row_capacity = new_row_capacity;
        self.raw.col_capacity = new_col_capacity;
        self.raw.ptr = unsafe { NonNull::<T>::new_unchecked(new_ptr) };
    }
}

#[cold]
fn capacity_overflow_impl() -> ! {
    panic!("capacity overflow")
}

#[inline(always)]
fn capacity_overflow<T>() -> T {
    capacity_overflow_impl();
}

#[inline(always)]
pub fn is_vectorizable<T: 'static>() -> bool {
    coe::is_same::<f32, T>()
        || coe::is_same::<f64, T>()
        || coe::is_same::<c32, T>()
        || coe::is_same::<c64, T>()
        || coe::is_same::<c32conj, T>()
        || coe::is_same::<c64conj, T>()
}

// https://rust-lang.github.io/hashbrown/src/crossbeam_utils/cache_padded.rs.html#128-130
pub const CACHELINE_ALIGN: usize = {
    #[cfg(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "powerpc64",
    ))]
    {
        128
    }
    #[cfg(any(
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "riscv64",
    ))]
    {
        32
    }
    #[cfg(target_arch = "s390x")]
    {
        256
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "powerpc64",
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "riscv64",
        target_arch = "s390x",
    )))]
    {
        64
    }
};

#[inline(always)]
pub fn align_for<T: 'static>() -> usize {
    if is_vectorizable::<T>() {
        Ord::max(
            core::mem::size_of::<T>(),
            Ord::max(core::mem::align_of::<T>(), CACHELINE_ALIGN),
        )
    } else {
        core::mem::align_of::<T>()
    }
}
