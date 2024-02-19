#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(non_snake_case)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]

use core::sync::atomic::AtomicUsize;
use equator::{assert, debug_assert};

extern crate alloc;

pub mod linalg;

pub mod complex_native;

pub use dyn_stack;
pub mod utils;
pub use reborrow;

pub mod col;
pub mod diag;
pub mod mat;
pub mod perm;
pub mod row;
pub mod sparse;

pub use mat::{Mat, MatMut, MatRef};

mod seal;
mod sort;

pub use faer_entity::{ComplexField, Conjugate, Entity, RealField};

/// Specifies whether the triangular lower or upper part of a matrix should be accessed.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Side {
    /// Lower half should be accessed.
    Lower,
    /// Upper half should be accessed.
    Upper,
}

/// Whether a matrix should be implicitly conjugated when read or not.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Conj {
    /// Do conjugate.
    Yes,
    /// Do not conjugate.
    No,
}

impl Conj {
    /// Combine `self` and `other` to create a new conjugation object.
    #[inline]
    pub fn compose(self, other: Conj) -> Conj {
        if self == other {
            Conj::No
        } else {
            Conj::Yes
        }
    }
}

/// Zips together matrix of the same size, so that coefficient-wise operations can be performed on
/// their elements.
///
/// # Note
/// The order in which the matrix elements are traversed is unspecified.
///
/// # Example
/// ```
/// use faer::{mat, unzipped, zipped, Mat};
///
/// let nrows = 2;
/// let ncols = 3;
///
/// let a = mat![[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]];
/// let b = mat![[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]];
/// let mut sum = Mat::<f64>::zeros(nrows, ncols);
///
/// zipped!(sum.as_mut(), a.as_ref(), b.as_ref()).for_each(|unzipped!(mut sum, a, b)| {
///     let a = a.read();
///     let b = b.read();
///     sum.write(a + b);
/// });
///
/// for i in 0..nrows {
///     for j in 0..ncols {
///         assert_eq!(sum.read(i, j), a.read(i, j) + b.read(i, j));
///     }
/// }
/// ```
#[macro_export]
macro_rules! zipped {
    ($head: expr $(,)?) => {
        $crate::linalg::zip::LastEq($crate::linalg::zip::ViewMut::view_mut(&mut { $head }))
    };

    ($head: expr, $($tail: expr),* $(,)?) => {
        $crate::linalg::zip::ZipEq::new($crate::linalg::zip::ViewMut::view_mut(&mut { $head }), $crate::zipped!($($tail,)*))
    };
}

/// Used to undo the zipping by the [`zipped!`] macro.
#[macro_export]
macro_rules! unzipped {
    ($head: pat $(,)?) => {
        $crate::linalg::zip::Last($head)
    };

    ($head: pat, $($tail: pat),* $(,)?) => {
        $crate::linalg::zip::Zip($head, $crate::unzipped!($($tail,)*))
    };
}

#[doc(hidden)]
#[inline(always)]
pub fn ref_to_ptr<T>(ptr: &T) -> *const T {
    ptr
}

#[macro_export]
#[doc(hidden)]
macro_rules! __transpose_impl {
    ([$([$($col:expr),*])*] $($v:expr;)* ) => {
        [$([$($col,)*],)* [$($v,)*]]
    };
    ([$([$($col:expr),*])*] $($v0:expr, $($v:expr),* ;)*) => {
        $crate::__transpose_impl!([$([$($col),*])* [$($v0),*]] $($($v),* ;)*)
    };
}

/// Creates a [`Mat`] containing the arguments.
///
/// ```
/// use faer::mat;
///
/// let matrix = mat![
///     [1.0, 5.0, 9.0],
///     [2.0, 6.0, 10.0],
///     [3.0, 7.0, 11.0],
///     [4.0, 8.0, 12.0f64],
/// ];
///
/// assert_eq!(matrix.read(0, 0), 1.0);
/// assert_eq!(matrix.read(1, 0), 2.0);
/// assert_eq!(matrix.read(2, 0), 3.0);
/// assert_eq!(matrix.read(3, 0), 4.0);
///
/// assert_eq!(matrix.read(0, 1), 5.0);
/// assert_eq!(matrix.read(1, 1), 6.0);
/// assert_eq!(matrix.read(2, 1), 7.0);
/// assert_eq!(matrix.read(3, 1), 8.0);
///
/// assert_eq!(matrix.read(0, 2), 9.0);
/// assert_eq!(matrix.read(1, 2), 10.0);
/// assert_eq!(matrix.read(2, 2), 11.0);
/// assert_eq!(matrix.read(3, 2), 12.0);
/// ```
#[macro_export]
macro_rules! mat {
    () => {
        {
            compile_error!("number of columns in the matrix is ambiguous");
        }
    };

    ($([$($v:expr),* $(,)?] ),* $(,)?) => {
        {
            let data = ::core::mem::ManuallyDrop::new($crate::__transpose_impl!([] $($($v),* ;)*));
            let data = &*data;
            let ncols = data.len();
            let nrows = (*data.get(0).unwrap()).len();

            #[allow(unused_unsafe)]
            unsafe {
                $crate::mat::Mat::<_>::from_fn(nrows, ncols, |i, j| $crate::ref_to_ptr(&data[j][i]).read())
            }
        }
    };
}

#[cfg(feature = "perf-warn")]
#[macro_export]
#[doc(hidden)]
macro_rules! __perf_warn {
    ($name: ident) => {{
        #[inline(always)]
        #[allow(non_snake_case)]
        fn $name() -> &'static ::core::sync::atomic::AtomicBool {
            static $name: ::core::sync::atomic::AtomicBool =
                ::core::sync::atomic::AtomicBool::new(false);
            &$name
        }
        ::core::matches!(
            $name().compare_exchange(
                false,
                true,
                ::core::sync::atomic::Ordering::Relaxed,
                ::core::sync::atomic::Ordering::Relaxed,
            ),
            Ok(_)
        )
    }};
}

/// Convenience function to concatonate a nested list of matrices into a single
/// big ['Mat']. Concatonation pattern follows the numpy.block convention that
/// each sub-list must have an equal number of columns (net) but the boundaries
/// do not need to align. In other words, this sort of thing:
/// ```notcode
///   AAAbb
///   AAAbb
///   cDDDD
/// ```
/// is perfectly acceptable.
#[doc(hidden)]
#[track_caller]
pub fn concat_impl<E: ComplexField>(blocks: &[&[mat::MatRef<'_, E>]]) -> mat::Mat<E> {
    #[inline(always)]
    fn count_total_columns<E: ComplexField>(block_row: &[mat::MatRef<'_, E>]) -> usize {
        let mut out: usize = 0;
        for elem in block_row.iter() {
            out += elem.ncols();
        }
        out
    }

    #[inline(always)]
    #[track_caller]
    fn count_rows<E: ComplexField>(block_row: &[mat::MatRef<'_, E>]) -> usize {
        let mut out: usize = 0;
        for (i, e) in block_row.iter().enumerate() {
            if i.eq(&0) {
                out = e.nrows();
            } else {
                assert!(e.nrows().eq(&out));
            }
        }
        out
    }

    // get size of result while doing checks
    let mut n: usize = 0;
    let mut m: usize = 0;
    for row in blocks.iter() {
        n += count_rows(row);
    }
    for (i, row) in blocks.iter().enumerate() {
        let cols = count_total_columns(row);
        if i.eq(&0) {
            m = cols;
        } else {
            assert!(cols.eq(&m));
        }
    }

    let mut mat = mat::Mat::<E>::zeros(n, m);
    let mut ni: usize = 0;
    let mut mj: usize;
    for row in blocks.iter() {
        mj = 0;

        for elem in row.iter() {
            mat.as_mut()
                .submatrix_mut(ni, mj, elem.nrows(), elem.ncols())
                .copy_from(elem);
            mj += elem.ncols();
        }
        ni += row[0].nrows();
    }

    mat
}

/// Concatenates the matrices in each row horizontally,
/// then concatenates the results vertically.
///
/// `concat![[a0, a1, a2], [b1, b2]]` results in the matrix
/// ```notcode
/// [a0 | a1 | a2][b0 | b1]
/// ```
#[macro_export]
macro_rules! concat {
    () => {
        {
            compile_error!("number of columns in the matrix is ambiguous");
        }
    };

    ($([$($v:expr),* $(,)?] ),* $(,)?) => {
        {
            $crate::concat_impl(&[$(&[$(($v).as_ref(),)*],)*])
        }
    };
}

/// Creates a [`Col`] containing the arguments.
///
/// ```
/// use faer::col;
///
/// let col_vec = col![3.0, 5.0, 7.0, 9.0];
///
/// assert_eq!(col_vec.read(0), 3.0);
/// assert_eq!(col_vec.read(1), 5.0);
/// assert_eq!(col_vec.read(2), 7.0);
/// assert_eq!(col_vec.read(3), 9.0);
/// ```
#[macro_export]
macro_rules! col {
    () => {
        $crate::col::Col::<_>::new()
    };

    ($($v:expr),+ $(,)?) => {{
        let data = &[$($v),+];
        let n = data.len();

        #[allow(unused_unsafe)]
        unsafe {
            $crate::col::Col::<_>::from_fn(n, |i| $crate::ref_to_ptr(&data[i]).read())
        }
    }};
}

/// Creates a [`Row`] containing the arguments.
///
/// ```
/// use faer::row;
///
/// let row_vec = row![3.0, 5.0, 7.0, 9.0];
///
/// assert_eq!(row_vec.read(0), 3.0);
/// assert_eq!(row_vec.read(1), 5.0);
/// assert_eq!(row_vec.read(2), 7.0);
/// assert_eq!(row_vec.read(3), 9.0);
/// ```
#[macro_export]
macro_rules! row {
    () => {
        $crate::row::Row::<_>::new()
    };

    ($($v:expr),+ $(,)?) => {{
        let data = &[$($v),+];
        let n = data.len();

        #[allow(unused_unsafe)]
        unsafe {
            $crate::row::Row::<_>::from_fn(n, |i| $crate::ref_to_ptr(&data[i]).read())
        }
    }};
}

/// Trait for unsigned integers that can be indexed with.
///
/// Always smaller than or equal to `usize`.
pub trait Index:
    seal::Seal
    + core::fmt::Debug
    + core::ops::Not<Output = Self>
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::AddAssign
    + core::ops::SubAssign
    + bytemuck::Pod
    + Eq
    + Ord
    + Send
    + Sync
{
    /// Equally-sized index type with a fixed size (no `usize`).
    type FixedWidth: Index;
    /// Equally-sized signed index type.
    type Signed: SignedIndex;

    /// Truncate `value` to type [`Self`].
    #[must_use]
    #[inline(always)]
    fn truncate(value: usize) -> Self {
        Self::from_signed(<Self::Signed as SignedIndex>::truncate(value))
    }

    /// Zero extend `self`.
    #[must_use]
    #[inline(always)]
    fn zx(self) -> usize {
        self.to_signed().zx()
    }

    /// Convert a reference to a slice of [`Self`] to fixed width types.
    #[inline(always)]
    fn canonicalize(slice: &[Self]) -> &[Self::FixedWidth] {
        bytemuck::cast_slice(slice)
    }

    /// Convert a mutable reference to a slice of [`Self`] to fixed width types.
    #[inline(always)]
    fn canonicalize_mut(slice: &mut [Self]) -> &mut [Self::FixedWidth] {
        bytemuck::cast_slice_mut(slice)
    }

    /// Convert a signed value to an unsigned one.
    #[inline(always)]
    fn from_signed(value: Self::Signed) -> Self {
        bytemuck::cast(value)
    }

    /// Convert an unsigned value to a signed one.
    #[inline(always)]
    fn to_signed(self) -> Self::Signed {
        bytemuck::cast(self)
    }

    /// Sum values while checking for overflow.
    #[inline]
    fn sum_nonnegative(slice: &[Self]) -> Option<Self> {
        Self::Signed::sum_nonnegative(bytemuck::cast_slice(slice)).map(Self::from_signed)
    }
}

/// Trait for signed integers corresponding to the ones satisfying [`Index`].
///
/// Always smaller than or equal to `isize`.
pub trait SignedIndex:
    seal::Seal
    + core::fmt::Debug
    + core::ops::Neg<Output = Self>
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::AddAssign
    + core::ops::SubAssign
    + bytemuck::Pod
    + Eq
    + Ord
    + Send
    + Sync
{
    /// Maximum representable value.
    const MAX: Self;

    /// Truncate `value` to type [`Self`].
    #[must_use]
    fn truncate(value: usize) -> Self;

    /// Zero extend `self`.
    #[must_use]
    fn zx(self) -> usize;
    /// Sign extend `self`.
    #[must_use]
    fn sx(self) -> usize;

    /// Sum nonnegative values while checking for overflow.
    fn sum_nonnegative(slice: &[Self]) -> Option<Self> {
        let mut acc = Self::zeroed();
        for &i in slice {
            if Self::MAX - i < acc {
                return None;
            }
            acc += i;
        }
        Some(acc)
    }
}

#[cfg(any(
    target_pointer_width = "32",
    target_pointer_width = "64",
    target_pointer_width = "128",
))]
impl Index for u32 {
    type FixedWidth = u32;
    type Signed = i32;
}
#[cfg(any(target_pointer_width = "64", target_pointer_width = "128"))]
impl Index for u64 {
    type FixedWidth = u64;
    type Signed = i64;
}
#[cfg(target_pointer_width = "128")]
impl Index for u128 {
    type FixedWidth = u128;
    type Signed = i128;
}

impl Index for usize {
    #[cfg(target_pointer_width = "32")]
    type FixedWidth = u32;
    #[cfg(target_pointer_width = "64")]
    type FixedWidth = u64;
    #[cfg(target_pointer_width = "128")]
    type FixedWidth = u128;

    type Signed = isize;
}

#[cfg(any(
    target_pointer_width = "32",
    target_pointer_width = "64",
    target_pointer_width = "128",
))]
impl SignedIndex for i32 {
    const MAX: Self = Self::MAX;

    #[inline(always)]
    fn truncate(value: usize) -> Self {
        #[allow(clippy::assertions_on_constants)]
        const _: () = {
            core::assert!(i32::BITS <= usize::BITS);
        };
        value as isize as Self
    }

    #[inline(always)]
    fn zx(self) -> usize {
        self as u32 as usize
    }

    #[inline(always)]
    fn sx(self) -> usize {
        self as isize as usize
    }
}

#[cfg(any(target_pointer_width = "64", target_pointer_width = "128"))]
impl SignedIndex for i64 {
    const MAX: Self = Self::MAX;

    #[inline(always)]
    fn truncate(value: usize) -> Self {
        #[allow(clippy::assertions_on_constants)]
        const _: () = {
            core::assert!(i64::BITS <= usize::BITS);
        };
        value as isize as Self
    }

    #[inline(always)]
    fn zx(self) -> usize {
        self as u64 as usize
    }

    #[inline(always)]
    fn sx(self) -> usize {
        self as isize as usize
    }
}

#[cfg(target_pointer_width = "128")]
impl SignedIndex for i128 {
    const MAX: Self = Self::MAX;

    #[inline(always)]
    fn truncate(value: usize) -> Self {
        #[allow(clippy::assertions_on_constants)]
        const _: () = {
            core::assert!(i128::BITS <= usize::BITS);
        };
        value as isize as Self
    }

    #[inline(always)]
    fn zx(self) -> usize {
        self as u128 as usize
    }

    #[inline(always)]
    fn sx(self) -> usize {
        self as isize as usize
    }
}

impl SignedIndex for isize {
    const MAX: Self = Self::MAX;

    #[inline(always)]
    fn truncate(value: usize) -> Self {
        value as isize
    }

    #[inline(always)]
    fn zx(self) -> usize {
        self as usize
    }

    #[inline(always)]
    fn sx(self) -> usize {
        self as usize
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Scale<E>(pub E);

impl<E> Scale<E> {
    #[inline]
    pub fn value(self) -> E {
        self.0
    }
}

#[inline]
pub fn scale<E>(val: E) -> Scale<E> {
    Scale(val)
}

/// Parallelism strategy that can be passed to most of the routines in the library.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Parallelism {
    /// No parallelism.
    ///
    /// The code is executed sequentially on the same thread that calls a function
    /// and passes this argument.
    None,
    /// Rayon parallelism. Only avaialble with the `rayon` feature.
    ///
    /// The code is possibly executed in parallel on the current thread, as well as the currently
    /// active rayon thread pool.
    ///
    /// The contained value represents a hint about the number of threads an implementation should
    /// use, but there is no way to guarantee how many or which threads will be used.
    ///
    /// A value of `0` treated as equivalent to `rayon::current_num_threads()`.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    Rayon(usize),
}

/// 0: Disable
/// 1: None
/// n >= 2: Rayon(n - 2)
///
/// default: Rayon(0)
static GLOBAL_PARALLELISM: AtomicUsize = {
    #[cfg(feature = "rayon")]
    {
        AtomicUsize::new(2)
    }
    #[cfg(not(feature = "rayon"))]
    {
        AtomicUsize::new(1)
    }
};

/// Causes functions that access global parallelism settings to panic.
pub fn disable_global_parallelism() {
    GLOBAL_PARALLELISM.store(0, core::sync::atomic::Ordering::Relaxed);
}

/// Sets the global parallelism settings.
pub fn set_global_parallelism(parallelism: Parallelism) {
    let value = match parallelism {
        Parallelism::None => 1,
        #[cfg(feature = "rayon")]
        Parallelism::Rayon(n) => n.saturating_add(2),
    };
    GLOBAL_PARALLELISM.store(value, core::sync::atomic::Ordering::Relaxed);
}

/// Gets the global parallelism settings.
///
/// # Panics
/// Panics if global parallelism is disabled.
#[track_caller]
pub fn get_global_parallelism() -> Parallelism {
    let value = GLOBAL_PARALLELISM.load(core::sync::atomic::Ordering::Relaxed);
    match value {
        0 => panic!("Global parallelism is disabled."),
        1 => Parallelism::None,
        #[cfg(feature = "rayon")]
        n => Parallelism::Rayon(n - 2),
        #[cfg(not(feature = "rayon"))]
        _ => unreachable!(),
    }
}

/// De-serialization from common matrix file formats.
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub mod io;

#[cfg(test)]
mod tests {
    use col::Col;
    use faer_entity::*;
    use row::Row;

    macro_rules! impl_unit_entity {
        ($ty: ty) => {
            unsafe impl Entity for $ty {
                type Unit = Self;
                type Index = ();
                type SimdUnit<S: pulp::Simd> = ();
                type SimdMask<S: pulp::Simd> = ();
                type SimdIndex<S: pulp::Simd> = ();
                type Group = IdentityGroup;
                type Iter<I: Iterator> = I;

                type PrefixUnit<'a, S: pulp::Simd> = &'a [()];
                type SuffixUnit<'a, S: pulp::Simd> = &'a [()];
                type PrefixMutUnit<'a, S: pulp::Simd> = &'a mut [()];
                type SuffixMutUnit<'a, S: pulp::Simd> = &'a mut [()];

                const N_COMPONENTS: usize = 1;
                const UNIT: GroupCopyFor<Self, ()> = ();

                #[inline(always)]
                fn faer_first<T>(group: GroupFor<Self, T>) -> T {
                    group
                }

                #[inline(always)]
                fn faer_from_units(group: GroupFor<Self, Self::Unit>) -> Self {
                    group
                }

                #[inline(always)]
                fn faer_into_units(self) -> GroupFor<Self, Self::Unit> {
                    self
                }

                #[inline(always)]
                fn faer_as_ref<T>(group: &GroupFor<Self, T>) -> GroupFor<Self, &T> {
                    group
                }

                #[inline(always)]
                fn faer_as_mut<T>(group: &mut GroupFor<Self, T>) -> GroupFor<Self, &mut T> {
                    group
                }

                #[inline(always)]
                fn faer_as_ptr<T>(group: *mut GroupFor<Self, T>) -> GroupFor<Self, *mut T> {
                    group
                }

                #[inline(always)]
                fn faer_map_impl<T, U>(
                    group: GroupFor<Self, T>,
                    f: &mut impl FnMut(T) -> U,
                ) -> GroupFor<Self, U> {
                    (*f)(group)
                }

                #[inline(always)]
                fn faer_map_with_context<Ctx, T, U>(
                    ctx: Ctx,
                    group: GroupFor<Self, T>,
                    f: &mut impl FnMut(Ctx, T) -> (Ctx, U),
                ) -> (Ctx, GroupFor<Self, U>) {
                    (*f)(ctx, group)
                }

                #[inline(always)]
                fn faer_zip<T, U>(
                    first: GroupFor<Self, T>,
                    second: GroupFor<Self, U>,
                ) -> GroupFor<Self, (T, U)> {
                    (first, second)
                }
                #[inline(always)]
                fn faer_unzip<T, U>(
                    zipped: GroupFor<Self, (T, U)>,
                ) -> (GroupFor<Self, T>, GroupFor<Self, U>) {
                    zipped
                }

                #[inline(always)]
                fn faer_into_iter<I: IntoIterator>(
                    iter: GroupFor<Self, I>,
                ) -> Self::Iter<I::IntoIter> {
                    iter.into_iter()
                }
            }
        };
    }

    use super::*;
    use crate::assert;

    #[test]
    fn basic_slice() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let slice = unsafe { mat::from_raw_parts::<'_, f64>(data.as_ptr(), 2, 3, 3, 1) };

        assert!(slice.get(0, 0) == &1.0);
        assert!(slice.get(0, 1) == &2.0);
        assert!(slice.get(0, 2) == &3.0);

        assert!(slice.get(1, 0) == &4.0);
        assert!(slice.get(1, 1) == &5.0);
        assert!(slice.get(1, 2) == &6.0);
    }

    #[test]
    fn empty() {
        {
            let m = Mat::<f64>::new();
            assert!(m.nrows() == 0);
            assert!(m.ncols() == 0);
            assert!(m.row_capacity() == 0);
            assert!(m.col_capacity() == 0);
        }

        {
            let m = Mat::<f64>::with_capacity(100, 120);
            assert!(m.nrows() == 0);
            assert!(m.ncols() == 0);
            assert!(m.row_capacity() == 100);
            assert!(m.col_capacity() == 120);
        }
    }

    #[test]
    fn reserve() {
        let mut m = Mat::<f64>::new();

        m.reserve_exact(0, 0);
        assert!(m.row_capacity() == 0);
        assert!(m.col_capacity() == 0);

        m.reserve_exact(1, 1);
        assert!(m.row_capacity() >= 1);
        assert!(m.col_capacity() == 1);

        m.reserve_exact(2, 0);
        assert!(m.row_capacity() >= 2);
        assert!(m.col_capacity() == 1);

        m.reserve_exact(2, 3);
        assert!(m.row_capacity() >= 2);
        assert!(m.col_capacity() == 3);
    }

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Zst;
    unsafe impl bytemuck::Zeroable for Zst {}
    unsafe impl bytemuck::Pod for Zst {}

    #[test]
    fn reserve_zst() {
        impl_unit_entity!(Zst);

        let mut m = Mat::<Zst>::new();

        m.reserve_exact(0, 0);
        assert!(m.row_capacity() == 0);
        assert!(m.col_capacity() == 0);

        m.reserve_exact(1, 1);
        assert!(m.row_capacity() == 1);
        assert!(m.col_capacity() == 1);

        m.reserve_exact(2, 0);
        assert!(m.row_capacity() == 2);
        assert!(m.col_capacity() == 1);

        m.reserve_exact(2, 3);
        assert!(m.row_capacity() == 2);
        assert!(m.col_capacity() == 3);

        m.reserve_exact(usize::MAX, usize::MAX);
    }

    #[test]
    fn resize() {
        let mut m = Mat::new();
        let f = |i, j| i as f64 - j as f64;
        m.resize_with(2, 3, f);
        assert!(m.read(0, 0) == 0.0);
        assert!(m.read(0, 1) == -1.0);
        assert!(m.read(0, 2) == -2.0);
        assert!(m.read(1, 0) == 1.0);
        assert!(m.read(1, 1) == 0.0);
        assert!(m.read(1, 2) == -1.0);

        m.resize_with(1, 2, f);
        assert!(m.read(0, 0) == 0.0);
        assert!(m.read(0, 1) == -1.0);

        m.resize_with(2, 1, f);
        assert!(m.read(0, 0) == 0.0);
        assert!(m.read(1, 0) == 1.0);

        m.resize_with(1, 2, f);
        assert!(m.read(0, 0) == 0.0);
        assert!(m.read(0, 1) == -1.0);
    }

    #[test]
    fn resize_zst() {
        // miri test
        let mut m = Mat::new();
        let f = |_i, _j| Zst;
        m.resize_with(2, 3, f);
        m.resize_with(1, 2, f);
        m.resize_with(2, 1, f);
        m.resize_with(1, 2, f);
    }

    #[test]
    #[should_panic]
    fn cap_overflow_1() {
        let _ = Mat::<f64>::with_capacity(isize::MAX as usize, 1);
    }

    #[test]
    #[should_panic]
    fn cap_overflow_2() {
        let _ = Mat::<f64>::with_capacity(isize::MAX as usize, isize::MAX as usize);
    }

    #[test]
    fn matrix_macro() {
        let mut x = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        assert!(x[(0, 0)] == 1.0);
        assert!(x[(0, 1)] == 2.0);
        assert!(x[(0, 2)] == 3.0);

        assert!(x[(1, 0)] == 4.0);
        assert!(x[(1, 1)] == 5.0);
        assert!(x[(1, 2)] == 6.0);

        assert!(x[(2, 0)] == 7.0);
        assert!(x[(2, 1)] == 8.0);
        assert!(x[(2, 2)] == 9.0);

        x[(0, 0)] = 13.0;
        assert!(x[(0, 0)] == 13.0);

        assert!(x.get(.., ..) == x);
        assert!(x.get(.., 1..3) == x.as_ref().submatrix(0, 1, 3, 2));
    }

    #[test]
    fn matrix_macro_cplx() {
        use num_complex::Complex;
        let new = Complex::new;
        let mut x = mat![
            [new(1.0, 2.0), new(3.0, 4.0), new(5.0, 6.0)],
            [new(7.0, 8.0), new(9.0, 10.0), new(11.0, 12.0)],
            [new(13.0, 14.0), new(15.0, 16.0), new(17.0, 18.0)]
        ];

        assert!(x.read(0, 0) == Complex::new(1.0, 2.0));
        assert!(x.read(0, 1) == Complex::new(3.0, 4.0));
        assert!(x.read(0, 2) == Complex::new(5.0, 6.0));

        assert!(x.read(1, 0) == Complex::new(7.0, 8.0));
        assert!(x.read(1, 1) == Complex::new(9.0, 10.0));
        assert!(x.read(1, 2) == Complex::new(11.0, 12.0));

        assert!(x.read(2, 0) == Complex::new(13.0, 14.0));
        assert!(x.read(2, 1) == Complex::new(15.0, 16.0));
        assert!(x.read(2, 2) == Complex::new(17.0, 18.0));

        x.write(1, 0, Complex::new(3.0, 2.0));
        assert!(x.read(1, 0) == Complex::new(3.0, 2.0));
    }

    #[test]
    fn matrix_macro_native_cplx() {
        use complex_native::c64 as Complex;

        let new = Complex::new;
        let mut x = mat![
            [new(1.0, 2.0), new(3.0, 4.0), new(5.0, 6.0)],
            [new(7.0, 8.0), new(9.0, 10.0), new(11.0, 12.0)],
            [new(13.0, 14.0), new(15.0, 16.0), new(17.0, 18.0)]
        ];

        assert!(x.read(0, 0) == Complex::new(1.0, 2.0));
        assert!(x.read(0, 1) == Complex::new(3.0, 4.0));
        assert!(x.read(0, 2) == Complex::new(5.0, 6.0));

        assert!(x.read(1, 0) == Complex::new(7.0, 8.0));
        assert!(x.read(1, 1) == Complex::new(9.0, 10.0));
        assert!(x.read(1, 2) == Complex::new(11.0, 12.0));

        assert!(x.read(2, 0) == Complex::new(13.0, 14.0));
        assert!(x.read(2, 1) == Complex::new(15.0, 16.0));
        assert!(x.read(2, 2) == Complex::new(17.0, 18.0));

        x.write(1, 0, Complex::new(3.0, 2.0));
        assert!(x.read(1, 0) == Complex::new(3.0, 2.0));
    }

    #[test]
    fn col_macro() {
        let mut x = col![3.0, 5.0, 7.0, 9.0];

        assert!(x[0] == 3.0);
        assert!(x[1] == 5.0);
        assert!(x[2] == 7.0);
        assert!(x[3] == 9.0);

        x[0] = 13.0;
        assert!(x[0] == 13.0);

        assert!(x.get(..) == x);
    }

    #[test]
    fn col_macro_cplx() {
        use num_complex::Complex;
        let new = Complex::new;
        let mut x = col![new(1.0, 2.0), new(3.0, 4.0), new(5.0, 6.0),];

        assert!(x.read(0) == Complex::new(1.0, 2.0));
        assert!(x.read(1) == Complex::new(3.0, 4.0));
        assert!(x.read(2) == Complex::new(5.0, 6.0));

        x.write(0, Complex::new(3.0, 2.0));
        assert!(x.read(0) == Complex::new(3.0, 2.0));
    }

    #[test]
    fn col_macro_native_cplx() {
        use complex_native::c64 as Complex;

        let new = Complex::new;
        let mut x = col![new(1.0, 2.0), new(3.0, 4.0), new(5.0, 6.0),];

        assert!(x.read(0) == Complex::new(1.0, 2.0));
        assert!(x.read(1) == Complex::new(3.0, 4.0));
        assert!(x.read(2) == Complex::new(5.0, 6.0));

        x.write(0, Complex::new(3.0, 2.0));
        assert!(x.read(0) == Complex::new(3.0, 2.0));
    }

    #[test]
    fn row_macro() {
        let mut x = row![3.0, 5.0, 7.0, 9.0];

        assert!(x[0] == 3.0);
        assert!(x[1] == 5.0);
        assert!(x[2] == 7.0);
        assert!(x[3] == 9.0);

        x.write(0, 13.0);
        assert!(x.read(0) == 13.0);
    }

    #[test]
    fn row_macro_cplx() {
        use num_complex::Complex;

        let new = Complex::new;
        let mut x = row![new(1.0, 2.0), new(3.0, 4.0), new(5.0, 6.0),];

        assert!(x.read(0) == Complex::new(1.0, 2.0));
        assert!(x.read(1) == Complex::new(3.0, 4.0));
        assert!(x.read(2) == Complex::new(5.0, 6.0));

        x.write(0, Complex::new(3.0, 2.0));
        assert!(x.read(0) == Complex::new(3.0, 2.0));
    }

    #[test]
    fn row_macro_native_cplx() {
        use complex_native::c64 as Complex;

        let new = Complex::new;
        let mut x = row![new(1.0, 2.0), new(3.0, 4.0), new(5.0, 6.0),];

        assert!(x.read(0) == new(1.0, 2.0));
        assert!(x.read(1) == new(3.0, 4.0));
        assert!(x.read(2) == new(5.0, 6.0));

        x.write(0, new(3.0, 2.0));
        assert!(x.read(0) == new(3.0, 2.0));
    }

    #[test]
    fn null_col_and_row() {
        let null_col: Col<f64> = col![];
        assert!(null_col == Col::<f64>::new());

        let null_row: Row<f64> = row![];
        assert!(null_row == Row::<f64>::new());
    }

    #[test]
    fn positive_concat_f64() {
        let a0: Mat<f64> = Mat::from_fn(2, 2, |_, _| 1f64);
        let a1: Mat<f64> = Mat::from_fn(2, 3, |_, _| 2f64);
        let a2: Mat<f64> = Mat::from_fn(2, 4, |_, _| 3f64);

        let b0: Mat<f64> = Mat::from_fn(1, 6, |_, _| 4f64);
        let b1: Mat<f64> = Mat::from_fn(1, 3, |_, _| 5f64);

        let c0: Mat<f64> = Mat::from_fn(6, 1, |_, _| 6f64);
        let c1: Mat<f64> = Mat::from_fn(6, 3, |_, _| 7f64);
        let c2: Mat<f64> = Mat::from_fn(6, 2, |_, _| 8f64);
        let c3: Mat<f64> = Mat::from_fn(6, 3, |_, _| 9f64);

        let x = concat_impl(&[
            &[a0.as_ref(), a1.as_ref(), a2.as_ref()],
            &[b0.as_ref(), b1.as_ref()],
            &[c0.as_ref(), c1.as_ref(), c2.as_ref(), c3.as_ref()],
        ]);

        assert!(x == concat![[a0, a1, a2], [b0, b1], [c0, c1, c2, &c3]]);

        assert!(x[(0, 0)] == 1f64);
        assert!(x[(1, 1)] == 1f64);
        assert!(x[(2, 2)] == 4f64);
        assert!(x[(3, 3)] == 7f64);
        assert!(x[(4, 4)] == 8f64);
        assert!(x[(5, 5)] == 8f64);
        assert!(x[(6, 6)] == 9f64);
        assert!(x[(7, 7)] == 9f64);
        assert!(x[(8, 8)] == 9f64);
    }

    #[test]
    fn to_owned_equality() {
        use num_complex::{Complex, Complex as C};
        let mut mf32: Mat<f32> = mat![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
        let mut mf64: Mat<f64> = mat![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
        let mut mf32c: Mat<Complex<f32>> = mat![
            [C::new(1., 1.), C::new(2., 2.), C::new(3., 3.)],
            [C::new(4., 4.), C::new(5., 5.), C::new(6., 6.)],
            [C::new(7., 7.), C::new(8., 8.), C::new(9., 9.)]
        ];
        let mut mf64c: Mat<Complex<f64>> = mat![
            [C::new(1., 1.), C::new(2., 2.), C::new(3., 3.)],
            [C::new(4., 4.), C::new(5., 5.), C::new(6., 6.)],
            [C::new(7., 7.), C::new(8., 8.), C::new(9., 9.)]
        ];

        assert!(mf32.transpose().to_owned().as_ref() == mf32.transpose());
        assert!(mf64.transpose().to_owned().as_ref() == mf64.transpose());
        assert!(mf32c.transpose().to_owned().as_ref() == mf32c.transpose());
        assert!(mf64c.transpose().to_owned().as_ref() == mf64c.transpose());

        assert!(mf32.as_mut().transpose_mut().to_owned().as_ref() == mf32.transpose());
        assert!(mf64.as_mut().transpose_mut().to_owned().as_ref() == mf64.transpose());
        assert!(mf32c.as_mut().transpose_mut().to_owned().as_ref() == mf32c.transpose());
        assert!(mf64c.as_mut().transpose_mut().to_owned().as_ref() == mf64c.transpose());
    }

    #[test]
    fn conj_to_owned_equality() {
        use num_complex::{Complex, Complex as C};
        let mut mf32: Mat<f32> = mat![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
        let mut mf64: Mat<f64> = mat![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
        let mut mf32c: Mat<Complex<f32>> = mat![
            [C::new(1., 1.), C::new(2., 2.), C::new(3., 3.)],
            [C::new(4., 4.), C::new(5., 5.), C::new(6., 6.)],
            [C::new(7., 7.), C::new(8., 8.), C::new(9., 9.)]
        ];
        let mut mf64c: Mat<Complex<f64>> = mat![
            [C::new(1., 1.), C::new(2., 2.), C::new(3., 3.)],
            [C::new(4., 4.), C::new(5., 5.), C::new(6., 6.)],
            [C::new(7., 7.), C::new(8., 8.), C::new(9., 9.)]
        ];

        assert!(mf32.as_ref().adjoint().to_owned().as_ref() == mf32.adjoint());
        assert!(mf64.as_ref().adjoint().to_owned().as_ref() == mf64.adjoint());
        assert!(mf32c.as_ref().adjoint().to_owned().as_ref() == mf32c.adjoint());
        assert!(mf64c.as_ref().adjoint().to_owned().as_ref() == mf64c.adjoint());

        assert!(mf32.as_mut().adjoint_mut().to_owned().as_ref() == mf32.adjoint());
        assert!(mf64.as_mut().adjoint_mut().to_owned().as_ref() == mf64.adjoint());
        assert!(mf32c.as_mut().adjoint_mut().to_owned().as_ref() == mf32c.adjoint());
        assert!(mf64c.as_mut().adjoint_mut().to_owned().as_ref() == mf64c.adjoint());
    }

    #[test]
    fn mat_mul_assign_scalar() {
        let mut x = mat![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]];

        let expected = mat![[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]];
        x *= scale(2.0);
        assert_eq!(x, expected);

        let expected = mat![[0.0, 4.0], [8.0, 12.0], [16.0, 20.0]];
        let mut x_mut = x.as_mut();
        x_mut *= scale(2.0);
        assert_eq!(x, expected);
    }

    #[test]
    fn test_col_slice() {
        let mut matrix = mat![[1.0, 5.0, 9.0], [2.0, 6.0, 10.0], [3.0, 7.0, 11.0f64]];

        assert_eq!(matrix.col_as_slice(1), &[5.0, 6.0, 7.0]);
        assert_eq!(matrix.col_as_slice_mut(0), &[1.0, 2.0, 3.0]);

        matrix
            .col_as_slice_mut(0)
            .copy_from_slice(&[-1.0, -2.0, -3.0]);

        let expected = mat![[-1.0, 5.0, 9.0], [-2.0, 6.0, 10.0], [-3.0, 7.0, 11.0f64]];
        assert_eq!(matrix, expected);
    }

    #[test]
    fn from_slice() {
        let mut slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];

        let expected = mat![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
        let view = mat::from_column_major_slice::<'_, f64>(&slice, 3, 2);
        assert_eq!(expected, view);
        let view = mat::from_column_major_slice::<'_, f64>(&mut slice, 3, 2);
        assert_eq!(expected, view);

        let expected = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let view = mat::from_row_major_slice::<'_, f64>(&slice, 3, 2);
        assert_eq!(expected, view);
        let view = mat::from_row_major_slice::<'_, f64>(&mut slice, 3, 2);
        assert_eq!(expected, view);
    }

    #[test]
    #[should_panic]
    fn from_slice_too_big() {
        let slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0_f64];
        mat::from_column_major_slice::<'_, f64>(&slice, 3, 2);
    }

    #[test]
    #[should_panic]
    fn from_slice_too_small() {
        let slice = [1.0, 2.0, 3.0, 4.0, 5.0_f64];
        mat::from_column_major_slice::<'_, f64>(&slice, 3, 2);
    }

    #[test]
    fn test_is_finite() {
        use complex_native::c32;

        let inf = f32::INFINITY;
        let nan = f32::NAN;

        {
            assert!(<f32 as ComplexField>::faer_is_finite(&1.0));
            assert!(!<f32 as ComplexField>::faer_is_finite(&inf));
            assert!(!<f32 as ComplexField>::faer_is_finite(&-inf));
            assert!(!<f32 as ComplexField>::faer_is_finite(&nan));
        }
        {
            let x = c32::new(1.0, 2.0);
            assert!(<c32 as ComplexField>::faer_is_finite(&x));

            let x = c32::new(inf, 2.0);
            assert!(!<c32 as ComplexField>::faer_is_finite(&x));

            let x = c32::new(1.0, inf);
            assert!(!<c32 as ComplexField>::faer_is_finite(&x));

            let x = c32::new(inf, inf);
            assert!(!<c32 as ComplexField>::faer_is_finite(&x));

            let x = c32::new(nan, 2.0);
            assert!(!<c32 as ComplexField>::faer_is_finite(&x));

            let x = c32::new(1.0, nan);
            assert!(!<c32 as ComplexField>::faer_is_finite(&x));

            let x = c32::new(nan, nan);
            assert!(!<c32 as ComplexField>::faer_is_finite(&x));
        }
    }

    #[test]
    fn test_iter() {
        let mut mat = Mat::from_fn(9, 10, |i, j| (i + j) as f64);
        let mut iter = mat.row_chunks_mut(4);

        let first = iter.next();
        let second = iter.next();
        let last = iter.next();
        let none = iter.next();

        assert!(first == Some(Mat::from_fn(4, 10, |i, j| (i + j) as f64).as_mut()));
        assert!(second == Some(Mat::from_fn(4, 10, |i, j| (i + j + 4) as f64).as_mut()));
        assert!(last == Some(Mat::from_fn(1, 10, |i, j| (i + j + 8) as f64).as_mut()));
        assert!(none == None);
    }

    #[test]
    fn test_norm_l2() {
        let relative_err = |a: f64, b: f64| (a - b).abs() / f64::max(a.abs(), b.abs());

        for (m, n) in [(9, 10), (1023, 5), (42, 1)] {
            for factor in [0.0, 1.0, 1e30, 1e250, 1e-30, 1e-250] {
                let mat = Mat::from_fn(m, n, |i, j| factor * ((i + j) as f64));
                let mut target = 0.0;
                zipped!(mat.as_ref()).for_each(|unzipped!(x)| {
                    target = f64::hypot(*x, target);
                });

                if factor == 0.0 {
                    assert!(mat.norm_l2() == target);
                } else {
                    assert!(relative_err(mat.norm_l2(), target) < 1e-14);
                }
            }
        }

        let mat = Col::from_fn(10000000, |_| 0.3);
        let target = (0.3 * 0.3 * 10000000.0f64).sqrt();
        assert!(relative_err(mat.norm_l2(), target) < 1e-14);
    }

    #[test]
    fn test_sum() {
        let relative_err = |a: f64, b: f64| (a - b).abs() / f64::max(a.abs(), b.abs());

        for (m, n) in [(9, 10), (1023, 5), (42, 1)] {
            for factor in [0.0, 1.0, 1e30, 1e250, 1e-30, 1e-250] {
                let mat = Mat::from_fn(m, n, |i, j| factor * ((i + j) as f64));
                let mut target = 0.0;
                zipped!(mat.as_ref()).for_each(|unzipped!(x)| {
                    target += *x;
                });

                if factor == 0.0 {
                    assert!(mat.sum() == target);
                } else {
                    assert!(relative_err(mat.sum(), target) < 1e-14);
                }
            }
        }

        let mat = Col::from_fn(10000000, |_| 0.3);
        let target = 0.3 * 10000000.0f64;
        assert!(relative_err(mat.sum(), target) < 1e-14);
    }

    #[test]
    fn test_kron_ones() {
        for (m, n, p, q) in [(2, 3, 4, 5), (3, 2, 5, 4), (1, 1, 1, 1)] {
            let a = Mat::from_fn(m, n, |_, _| 1 as f64);
            let b = Mat::from_fn(p, q, |_, _| 1 as f64);
            let expected = Mat::from_fn(m * p, n * q, |_, _| 1 as f64);
            assert!(a.kron(&b) == expected);
        }

        for (m, n, p) in [(2, 3, 4), (3, 2, 5), (1, 1, 1)] {
            let a = Mat::from_fn(m, n, |_, _| 1 as f64);
            let b = Col::from_fn(p, |_| 1 as f64);
            let expected = Mat::from_fn(m * p, n, |_, _| 1 as f64);
            assert!(a.kron(&b) == expected);
            assert!(b.kron(&a) == expected);

            let a = Mat::from_fn(m, n, |_, _| 1 as f64);
            let b = Row::from_fn(p, |_| 1 as f64);
            let expected = Mat::from_fn(m, n * p, |_, _| 1 as f64);
            assert!(a.kron(&b) == expected);
            assert!(b.kron(&a) == expected);
        }

        for (m, n) in [(2, 3), (3, 2), (1, 1)] {
            let a = Row::from_fn(m, |_| 1 as f64);
            let b = Col::from_fn(n, |_| 1 as f64);
            let expected = Mat::from_fn(n, m, |_, _| 1 as f64);
            assert!(a.kron(&b) == expected);
            assert!(b.kron(&a) == expected);

            let c = Row::from_fn(n, |_| 1 as f64);
            let expected = Mat::from_fn(1, m * n, |_, _| 1 as f64);
            assert!(a.kron(&c) == expected);

            let d = Col::from_fn(m, |_| 1 as f64);
            let expected = Mat::from_fn(m * n, 1, |_, _| 1 as f64);
            assert!(d.kron(&b) == expected);
        }
    }

    #[test]
    fn test_col_index() {
        let mut col_32: Col<f32> = Col::from_fn(3, |i| i as f32);
        col_32.as_mut()[1] = 10f32;
        let tval: f32 = (10f32 - col_32[1]).abs();
        assert!(tval < 1e-14);

        let mut col_64: Col<f64> = Col::from_fn(3, |i| i as f64);
        col_64.as_mut()[1] = 10f64;
        let tval: f64 = (10f64 - col_64[1]).abs();
        assert!(tval < 1e-14);
    }

    #[test]
    fn test_row_index() {
        let mut row_32: Row<f32> = Row::from_fn(3, |i| i as f32);
        row_32.as_mut()[1] = 10f32;
        let tval: f32 = (10f32 - row_32[1]).abs();
        assert!(tval < 1e-14);

        let mut row_64: Row<f64> = Row::from_fn(3, |i| i as f64);
        row_64.as_mut()[1] = 10f64;
        let tval: f64 = (10f64 - row_64[1]).abs();
        assert!(tval < 1e-14);
    }
}
