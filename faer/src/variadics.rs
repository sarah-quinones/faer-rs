use crate::{
    assert,
    internal_prelude::*,
    utils::bound::{Segment, SegmentIdx},
};
use core::{
    cell::UnsafeCell,
    fmt,
    marker::PhantomData,
    ops::{Range, RangeFrom, RangeFull, RangeTo},
};

#[macro_export]
#[doc(hidden)]
macro_rules! __list_impl {
    (@ __impl @ () @ ()) => {
        $crate::variadics::Nil
    };

    (@ __impl @ ($($parsed:tt)+) @ ()) => {
        $crate::variadics::Cons {
            head: $($parsed)+,
            tail: $crate::variadics::Nil,
        }
    };

    (@ __impl @ ($($parsed:tt)+) @ (, $($unparsed:tt)*)) => {
        $crate::variadics::Cons {
            head: $($parsed)+,
            tail: $crate::__list_impl![@ __impl @ () @ ($($unparsed)*)],
        }
    };

    (@ __impl @ ($($parsed:tt)*) @ ($unparsed_head: tt $($unparsed_rest:tt)*)) => {
        $crate::__list_impl![@ __impl @ ($($parsed)* $unparsed_head) @ ($($unparsed_rest)*)]
    };
}

/// create or destructure a variadic tuple containing the given values.
#[macro_export]
macro_rules! list {
    ($($t:tt)*) => {
        $crate::__list_impl![@ __impl @ () @ ($($t)*)]
    };
}

/// destructure a variadic tuple containing the given values.
#[macro_export]
macro_rules! pat_list {
    () => {
        $crate::variadics::Nil
    };
    ($head: pat $(, $tail: pat)* $(,)?) => {
        $crate::variadics::Cons {
            head: $head,
            tail: $crate::variadics::pat_list![$($tail,)*],
        }
    };
}

/// type of a variadic tuple containing the given types.
#[macro_export]
macro_rules! List {
    () => {
        $crate::variadics::Nil
    };
    ($head: ty $(, $tail: ty)* $(,)?) => {
        $crate::variadics::Cons::<$head, $crate::List!($($tail,)*)>
    };
}

/// empty tuple.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Nil;

/// non-empty tuple, containing the first element and the rest of the elements as a variadic
/// tuple.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(C)]
pub struct Cons<Head, Tail> {
    /// first element.
    pub head: Head,
    /// variadic tuple of the remaining elements.
    pub tail: Tail,
}

impl DebugList for Nil {
    fn push_debug(this: &Self, debug: &mut fmt::DebugList<'_, '_>) {
        _ = this;
        _ = debug;
    }
}
impl<Head: fmt::Debug, Tail: DebugList> DebugList for Cons<Head, Tail> {
    fn push_debug(this: &Self, debug: &mut fmt::DebugList<'_, '_>) {
        debug.entry(&this.head);
        Tail::push_debug(&this.tail, debug)
    }
}

impl fmt::Debug for Nil {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().finish()
    }
}
impl<Head: fmt::Debug, Tail: DebugList> fmt::Debug for Cons<Head, Tail> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_list();
        <Cons<Head, Tail> as DebugList>::push_debug(self, &mut debug);
        debug.finish()
    }
}

trait DebugList {
    fn push_debug(this: &Self, debug: &mut fmt::DebugList<'_, '_>);
}
use faer_traits::hacks::GhostNode;
pub use list;
pub use pat_list;
pub use List;

pub unsafe trait Separable<'scope, 'dim> {
    type Index;
    type Disjoint;
}

pub unsafe trait SplitsNode<'scope, 'dim, Node> {
    type Output: Separable<'scope, 'dim>;

    fn output(split: Self, start: &mut IdxInc<'dim>, dim: Dim<'dim>, node: Node) -> Self::Output;
}

unsafe impl<'scope, 'dim, 'range> Separable<'scope, 'dim> for Segment<'scope, 'dim, 'range> {
    type Index = SegmentIdx<'scope, 'dim, 'range>;
    type Disjoint = Nil;
}

unsafe impl<'scope, 'dim, 'range> Separable<'scope, 'dim> for SegmentIdx<'scope, 'dim, 'range> {
    type Index = SegmentIdx<'scope, 'dim, 'range>;
    type Disjoint = Nil;
}

unsafe impl<'scope, 'dim> Separable<'scope, 'dim> for Nil {
    type Index = Nil;
    type Disjoint = Nil;
}

unsafe impl<'scope, 'dim, 'range> SplitsNode<'scope, 'dim, Nil> for Nil {
    type Output = Nil;
    fn output(_: Self, _: &mut IdxInc<'dim>, _: Dim<'dim>, _: Nil) -> Self::Output {
        Nil
    }
}

unsafe impl<'scope, 'dim: 'range, 'range> SplitsNode<'scope, 'dim, GhostNode<'scope, 'range, Nil>>
    for Range<IdxInc<'dim>>
{
    type Output = Segment<'scope, 'dim, 'range>;

    #[inline]
    fn output(
        segment: Self,
        start: &mut IdxInc<'dim>,
        dim: Dim<'dim>,
        node: GhostNode<'scope, 'range, Nil>,
    ) -> Self::Output {
        _ = dim;
        _ = node;

        assert!(all(*start <= segment.start, segment.start <= segment.end));
        *start = segment.end;
        unsafe { Segment::new_unbound(segment.start, segment.end) }
    }
}

unsafe impl<'scope, 'dim: 'range, 'range> SplitsNode<'scope, 'dim, GhostNode<'scope, 'range, Nil>>
    for Idx<'dim>
{
    type Output = SegmentIdx<'scope, 'dim, 'range>;

    #[inline]
    fn output(
        segment: Self,
        start: &mut IdxInc<'dim>,
        dim: Dim<'dim>,
        node: GhostNode<'scope, 'range, Nil>,
    ) -> Self::Output {
        _ = dim;
        _ = node;

        assert!(all(*start <= segment.to_incl()));
        *start = segment.next();
        unsafe { SegmentIdx::new_unbound(segment) }
    }
}

unsafe impl<'scope, 'dim: 'range, 'range> SplitsNode<'scope, 'dim, GhostNode<'scope, 'range, Nil>>
    for RangeFrom<IdxInc<'dim>>
{
    type Output = Segment<'scope, 'dim, 'range>;

    #[inline]
    fn output(
        segment: Self,
        start: &mut IdxInc<'dim>,
        dim: Dim<'dim>,
        node: GhostNode<'scope, 'range, Nil>,
    ) -> Self::Output {
        _ = dim;
        _ = node;

        assert!(*start <= segment.start);
        *start = dim.end();
        unsafe { Segment::new_unbound(segment.start, dim.end()) }
    }
}

unsafe impl<'scope, 'dim: 'range, 'range, 'subrange>
    SplitsNode<'scope, 'dim, GhostNode<'scope, 'range, Nil>> for Segment<'scope, '_, 'subrange>
{
    type Output = Segment<'scope, 'dim, 'subrange>;

    #[inline]
    fn output(
        segment: Self,
        start: &mut IdxInc<'dim>,
        dim: Dim<'dim>,
        node: GhostNode<'scope, 'range, Nil>,
    ) -> Self::Output {
        _ = dim;
        _ = node;

        assert!(all(**start <= *segment.start(), *segment.end() <= *dim));
        *start = unsafe { IdxInc::new_unbound(*segment.end()) };
        unsafe {
            Segment::new_unbound(
                IdxInc::new_unbound(*segment.start()),
                IdxInc::new_unbound(*segment.end()),
            )
        }
    }
}

unsafe impl<'scope, 'dim: 'range, 'range> SplitsNode<'scope, 'dim, GhostNode<'scope, 'range, Nil>>
    for RangeTo<IdxInc<'dim>>
{
    type Output = Segment<'scope, 'dim, 'range>;

    #[inline]
    fn output(
        segment: Self,
        start: &mut IdxInc<'dim>,
        dim: Dim<'dim>,
        node: GhostNode<'scope, 'range, Nil>,
    ) -> Self::Output {
        _ = dim;
        _ = node;

        let begin = *start;
        assert!(all(*start <= segment.end));
        *start = segment.end;
        unsafe { Segment::new_unbound(begin, segment.end) }
    }
}

unsafe impl<'scope, 'dim: 'range, 'range> SplitsNode<'scope, 'dim, GhostNode<'scope, 'range, Nil>>
    for RangeFull
{
    type Output = Segment<'scope, 'dim, 'range>;

    #[inline]
    fn output(
        _: Self,
        start: &mut IdxInc<'dim>,
        dim: Dim<'dim>,
        node: GhostNode<'scope, 'range, Nil>,
    ) -> Self::Output {
        _ = dim;
        _ = node;

        let old_start = *start;
        *start = dim.end();
        unsafe { Segment::new_unbound(old_start, dim.end()) }
    }
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Separator<'scope, Head, Tail> {
    __marker: PhantomData<(Invariant<'scope>, fn(Head, Tail))>,
}

unsafe impl<
        'scope,
        'dim: 'range,
        'range,
        Head: Separable<'scope, 'dim>,
        Tail: Separable<'scope, 'dim>,
    > Separable<'scope, 'dim> for Cons<Head, Tail>
{
    type Index = (Head::Index, Tail::Index);

    type Disjoint = (
        Separator<'scope, Head::Index, Tail::Index>,
        Cons<Head::Disjoint, Tail::Disjoint>,
    );
}

unsafe impl<'scope, 'dim, Seg, Disjoint, Output: Separable<'scope, 'dim>> Separable<'scope, 'dim>
    for (Seg, Output, Disjoint)
{
    type Index = Output::Index;
    type Disjoint = Output::Disjoint;
}

unsafe impl<
        'scope,
        'dim,
        NodeHead,
        NodeTail,
        Head: SplitsNode<'scope, 'dim, NodeHead>,
        Tail: SplitsNode<'scope, 'dim, NodeTail>,
    > SplitsNode<'scope, 'dim, Cons<NodeHead, NodeTail>> for Cons<Head, Tail>
{
    type Output = Cons<Head::Output, Tail::Output>;

    #[inline]
    fn output(
        split: Self,
        start: &mut IdxInc<'dim>,
        dim: Dim<'dim>,
        node: Cons<NodeHead, NodeTail>,
    ) -> Self::Output {
        Cons {
            head: Head::output(split.head, start, dim, node.head),
            tail: Tail::output(split.tail, start, dim, node.tail),
        }
    }
}

unsafe impl<
        'scope,
        'dim,
        'range,
        NodeHead,
        NodeTail,
        Head: SplitsNode<'scope, 'dim, NodeHead>,
        Tail: SplitsNode<'scope, 'dim, NodeTail>,
    > SplitsNode<'scope, 'dim, GhostNode<'scope, 'range, Cons<NodeHead, NodeTail>>>
    for Cons<Head, Tail>
{
    type Output = (
        Segment<'scope, 'dim, 'range>,
        Cons<Head::Output, Tail::Output>,
        <Cons<Head::Output, Tail::Output> as Separable<'scope, 'dim>>::Disjoint,
    );

    #[inline]
    fn output(
        split: Self,
        start: &mut IdxInc<'dim>,
        dim: Dim<'dim>,
        node: GhostNode<'scope, 'range, Cons<NodeHead, NodeTail>>,
    ) -> Self::Output {
        let begin = *start;

        let list = Cons {
            head: Head::output(split.head, start, dim, node.child.head),
            tail: Tail::output(split.tail, start, dim, node.child.tail),
        };

        let end = *start;

        const {
            core::assert!(
                size_of::<<Cons<Head::Output, Tail::Output> as Separable<'scope, 'dim>>::Disjoint>(
                ) == 0
            )
        }

        unsafe {
            (
                Segment::new_unbound(begin, end),
                list,
                core::mem::transmute_copy(&()),
            )
        }
    }
}

pub trait ArraySplit<'scope, 'dim: 'a, 'a, Outlives = &'a Self>: Separable<'scope, 'dim> {
    type ArrayRefSegments<T: 'a>: 'a;
    type ArrayMutSegments<T: 'a>: 'a;

    fn array_ref_segments<T>(this: Self, array: &'a Array<'dim, T>) -> Self::ArrayRefSegments<T>;
    fn array_mut_segments<T>(
        this: Self,
        array: &'a mut Array<'dim, T>,
        disjoint: Self::Disjoint,
    ) -> Self::ArrayMutSegments<T>;

    unsafe fn cast_cell<T>(
        segment: Self::ArrayRefSegments<UnsafeCell<T>>,
    ) -> Self::ArrayMutSegments<T>;
}

impl<'scope, 'dim: 'a, 'a> ArraySplit<'scope, 'dim, 'a> for Nil {
    type ArrayRefSegments<T: 'a> = Nil;
    type ArrayMutSegments<T: 'a> = Nil;

    #[inline]
    fn array_ref_segments<T>(_: Self, _: &'a Array<'dim, T>) -> Self::ArrayRefSegments<T> {
        Nil
    }
    #[inline]
    fn array_mut_segments<T>(
        _: Self,
        _: &'a mut Array<'dim, T>,
        _: Self::Disjoint,
    ) -> Self::ArrayMutSegments<T> {
        Nil
    }

    #[inline]
    unsafe fn cast_cell<T: 'a>(
        _: Self::ArrayRefSegments<UnsafeCell<T>>,
    ) -> Self::ArrayMutSegments<T> {
        Nil
    }
}
impl<'scope, 'dim: 'a, 'range, 'a, Tail: ArraySplit<'scope, 'dim, 'a>> ArraySplit<'scope, 'dim, 'a>
    for Cons<Segment<'scope, 'dim, 'range>, Tail>
{
    type ArrayRefSegments<T: 'a> = Cons<&'a Array<'range, T>, Tail::ArrayRefSegments<T>>;
    type ArrayMutSegments<T: 'a> = Cons<&'a mut Array<'range, T>, Tail::ArrayMutSegments<T>>;

    #[inline]
    fn array_ref_segments<T>(this: Self, array: &'a Array<'dim, T>) -> Self::ArrayRefSegments<T> {
        let head = array.segment(this.head);

        Cons {
            head,
            tail: Tail::array_ref_segments(this.tail, array),
        }
    }
    #[inline]
    fn array_mut_segments<T>(
        this: Self,
        array: &'a mut Array<'dim, T>,
        _: Self::Disjoint,
    ) -> Self::ArrayMutSegments<T> {
        let cell = Self::array_ref_segments(this, unsafe {
            &*(array as *mut Array<'dim, T> as *mut Array<'dim, UnsafeCell<T>>)
        });

        unsafe { Self::cast_cell(cell) }
    }

    #[inline]
    unsafe fn cast_cell<T: 'a>(
        segment: Self::ArrayRefSegments<UnsafeCell<T>>,
    ) -> Self::ArrayMutSegments<T> {
        Cons {
            head: unsafe {
                &mut *((*(segment.head as *const Array<'range, UnsafeCell<T>>
                    as *const UnsafeCell<Array<'range, T>>))
                    .get())
            },
            tail: Tail::cast_cell(segment.tail),
        }
    }
}

impl<'scope, 'dim: 'a, 'range, 'a, Tail: ArraySplit<'scope, 'dim, 'a>> ArraySplit<'scope, 'dim, 'a>
    for Cons<SegmentIdx<'scope, 'dim, 'range>, Tail>
{
    type ArrayRefSegments<T: 'a> = Cons<&'a T, Tail::ArrayRefSegments<T>>;
    type ArrayMutSegments<T: 'a> = Cons<&'a mut T, Tail::ArrayMutSegments<T>>;

    #[inline]
    fn array_ref_segments<T>(this: Self, array: &'a Array<'dim, T>) -> Self::ArrayRefSegments<T> {
        let head = &array[unsafe { Idx::new_unbound(*this.head) }];

        Cons {
            head,
            tail: Tail::array_ref_segments(this.tail, array),
        }
    }
    #[inline]
    fn array_mut_segments<T>(
        this: Self,
        array: &'a mut Array<'dim, T>,
        _: Self::Disjoint,
    ) -> Self::ArrayMutSegments<T> {
        let cell = Self::array_ref_segments(this, unsafe {
            &*(array as *mut Array<'dim, T> as *mut Array<'dim, UnsafeCell<T>>)
        });

        unsafe { Self::cast_cell(cell) }
    }

    #[inline]
    unsafe fn cast_cell<T: 'a>(
        segment: Self::ArrayRefSegments<UnsafeCell<T>>,
    ) -> Self::ArrayMutSegments<T> {
        Cons {
            head: unsafe { &mut *(segment.head.get()) },
            tail: Tail::cast_cell(segment.tail),
        }
    }
}

pub trait RowSplit<'scope, 'dim: 'a, 'a, Outlives = &'a Self>: Separable<'scope, 'dim> {
    type MatRefSegments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride>;
    type MatMutSegments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride>;

    type ColRefSegments<C: Container, T: 'a, RStride: Stride>;
    type ColMutSegments<C: Container, T: 'a, RStride: Stride>;

    fn mat_ref_segments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride>(
        this: Self,
        mat: MatRef<'a, C, T, Dim<'dim>, Cols, RStride, CStride>,
    ) -> Self::MatRefSegments<C, T, Cols, RStride, CStride>;
    fn mat_mut_segments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride>(
        this: Self,
        mat: MatMut<'a, C, T, Dim<'dim>, Cols, RStride, CStride>,
        disjoint: Self::Disjoint,
    ) -> Self::MatMutSegments<C, T, Cols, RStride, CStride>;

    fn col_ref_segments<C: Container, T: 'a, RStride: Stride>(
        this: Self,
        mat: ColRef<'a, C, T, Dim<'dim>, RStride>,
    ) -> Self::ColRefSegments<C, T, RStride>;
    fn col_mut_segments<C: Container, T: 'a, RStride: Stride>(
        this: Self,
        mat: ColMut<'a, C, T, Dim<'dim>, RStride>,
        disjoint: Self::Disjoint,
    ) -> Self::ColMutSegments<C, T, RStride>;

    unsafe fn cast_cell<C: Container, T, RStride: Stride>(
        segment: Self::ColRefSegments<C, UnsafeCell<T>, RStride>,
    ) -> Self::ColMutSegments<C, T, RStride>;
}

pub trait ColSplit<'scope, 'dim: 'a, 'a, Outlives = &'a Self>: Separable<'scope, 'dim> {
    type MatRefSegments<C: Container, T: 'a, Rows: 'a + Shape, RStride: Stride, CStride: Stride>;
    type MatMutSegments<C: Container, T: 'a, Rows: 'a + Shape, RStride: Stride, CStride: Stride>;

    type RowRefSegments<C: Container, T: 'a, CStride: Stride>;
    type RowMutSegments<C: Container, T: 'a, CStride: Stride>;

    fn mat_ref_segments<C: Container, T: 'a, Rows: 'a + Shape, RStride: Stride, CStride: Stride>(
        this: Self,
        mat: MatRef<'a, C, T, Rows, Dim<'dim>, RStride, CStride>,
    ) -> Self::MatRefSegments<C, T, Rows, RStride, CStride>;
    fn mat_mut_segments<C: Container, T: 'a, Rows: 'a + Shape, RStride: Stride, CStride: Stride>(
        this: Self,
        mat: MatMut<'a, C, T, Rows, Dim<'dim>, RStride, CStride>,
        disjoint: Self::Disjoint,
    ) -> Self::MatMutSegments<C, T, Rows, RStride, CStride>;

    fn row_ref_segments<C: Container, T: 'a, CStride: Stride>(
        this: Self,
        mat: RowRef<'a, C, T, Dim<'dim>, CStride>,
    ) -> Self::RowRefSegments<C, T, CStride>;
    fn row_mut_segments<C: Container, T: 'a, CStride: Stride>(
        this: Self,
        mat: RowMut<'a, C, T, Dim<'dim>, CStride>,
        disjoint: Self::Disjoint,
    ) -> Self::RowMutSegments<C, T, CStride>;

    unsafe fn cast_cell<C: Container, T, RStride: Stride>(
        segment: Self::RowRefSegments<C, UnsafeCell<T>, RStride>,
    ) -> Self::RowMutSegments<C, T, RStride>;
}

impl<'scope, 'dim: 'a, 'a> RowSplit<'scope, 'dim, 'a> for Nil {
    type MatRefSegments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride> =
        Nil;
    type MatMutSegments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride> =
        Nil;

    type ColRefSegments<C: Container, T: 'a, RStride: Stride> = Nil;
    type ColMutSegments<C: Container, T: 'a, RStride: Stride> = Nil;

    #[inline]
    fn mat_ref_segments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride>(
        _: Self,
        _: MatRef<'a, C, T, Dim<'dim>, Cols, RStride, CStride>,
    ) -> Self::MatRefSegments<C, T, Cols, RStride, CStride> {
        Nil
    }
    #[inline]
    fn mat_mut_segments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride>(
        _: Self,
        _: MatMut<'a, C, T, Dim<'dim>, Cols, RStride, CStride>,
        _: Self::Disjoint,
    ) -> Self::MatMutSegments<C, T, Cols, RStride, CStride> {
        Nil
    }

    #[inline]
    fn col_ref_segments<C: Container, T: 'a, RStride: Stride>(
        _: Self,
        _: ColRef<'a, C, T, Dim<'dim>, RStride>,
    ) -> Self::ColRefSegments<C, T, RStride> {
        Nil
    }
    #[inline]
    fn col_mut_segments<C: Container, T: 'a, RStride: Stride>(
        _: Self,
        _: ColMut<'a, C, T, Dim<'dim>, RStride>,
        _: Self::Disjoint,
    ) -> Self::ColMutSegments<C, T, RStride> {
        Nil
    }

    #[inline]
    unsafe fn cast_cell<C: Container, T: 'a, RStride: Stride>(
        _: Self::ColRefSegments<C, UnsafeCell<T>, RStride>,
    ) -> Self::ColMutSegments<C, T, RStride> {
        Nil
    }
}

impl<'scope, 'dim: 'a, 'a> ColSplit<'scope, 'dim, 'a> for Nil {
    type MatRefSegments<C: Container, T: 'a, Rows: 'a + Shape, RStride: Stride, CStride: Stride> =
        Nil;
    type MatMutSegments<C: Container, T: 'a, Rows: 'a + Shape, RStride: Stride, CStride: Stride> =
        Nil;

    type RowRefSegments<C: Container, T: 'a, RStride: Stride> = Nil;
    type RowMutSegments<C: Container, T: 'a, RStride: Stride> = Nil;

    #[inline]
    fn mat_ref_segments<C: Container, T: 'a, Rows: 'a + Shape, RStride: Stride, CStride: Stride>(
        _: Self,
        _: MatRef<'a, C, T, Rows, Dim<'dim>, RStride, CStride>,
    ) -> Self::MatRefSegments<C, T, Rows, RStride, CStride> {
        Nil
    }
    #[inline]
    fn mat_mut_segments<C: Container, T: 'a, Rows: 'a + Shape, RStride: Stride, CStride: Stride>(
        _: Self,
        _: MatMut<'a, C, T, Rows, Dim<'dim>, RStride, CStride>,
        _: Self::Disjoint,
    ) -> Self::MatMutSegments<C, T, Rows, RStride, CStride> {
        Nil
    }

    #[inline]
    fn row_ref_segments<C: Container, T: 'a, RStride: Stride>(
        _: Self,
        _: RowRef<'a, C, T, Dim<'dim>, RStride>,
    ) -> Self::RowRefSegments<C, T, RStride> {
        Nil
    }
    #[inline]
    fn row_mut_segments<C: Container, T: 'a, RStride: Stride>(
        _: Self,
        _: RowMut<'a, C, T, Dim<'dim>, RStride>,
        _: Self::Disjoint,
    ) -> Self::RowMutSegments<C, T, RStride> {
        Nil
    }

    #[inline]
    unsafe fn cast_cell<C: Container, T: 'a, RStride: Stride>(
        _: Self::RowRefSegments<C, UnsafeCell<T>, RStride>,
    ) -> Self::RowMutSegments<C, T, RStride> {
        Nil
    }
}

type Covariant<'a> = fn() -> &'a ();
type Invariant<'a> = fn(&'a ()) -> &'a ();

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Disjoint<'scope, 'range> {
    __marker: PhantomData<(Invariant<'scope>, Covariant<'range>)>,
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Private<T>(T);

impl<'scope, 'dim: 'a, 'range, 'a, Tail: RowSplit<'scope, 'dim, 'a>> RowSplit<'scope, 'dim, 'a>
    for Cons<Segment<'scope, 'dim, 'range>, Tail>
{
    type MatRefSegments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride> =
        Cons<
            MatRef<'a, C, T, Dim<'range>, Cols, RStride, CStride>,
            Tail::MatRefSegments<C, T, Cols, RStride, CStride>,
        >;
    type MatMutSegments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride> =
        Cons<
            MatMut<'a, C, T, Dim<'range>, Cols, RStride, CStride>,
            Tail::MatMutSegments<C, T, Cols, RStride, CStride>,
        >;
    type ColRefSegments<C: Container, T: 'a, RStride: Stride> =
        Cons<ColRef<'a, C, T, Dim<'range>, RStride>, Tail::ColRefSegments<C, T, RStride>>;
    type ColMutSegments<C: Container, T: 'a, RStride: Stride> =
        Cons<ColMut<'a, C, T, Dim<'range>, RStride>, Tail::ColMutSegments<C, T, RStride>>;

    #[inline]
    fn mat_ref_segments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride>(
        this: Self,
        mat: MatRef<'a, C, T, Dim<'dim>, Cols, RStride, CStride>,
    ) -> Self::MatRefSegments<C, T, Cols, RStride, CStride> {
        let head = mat.row_segment(this.head);
        Cons {
            head,
            tail: Tail::mat_ref_segments(this.tail, mat),
        }
    }

    #[inline]
    fn mat_mut_segments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride>(
        this: Self,
        mat: MatMut<'a, C, T, Dim<'dim>, Cols, RStride, CStride>,
        disjoint: Self::Disjoint,
    ) -> Self::MatMutSegments<C, T, Cols, RStride, CStride> {
        let mat = mat.into_const();
        let head = unsafe { mat.row_segment(this.head).const_cast() };
        Cons {
            head,
            tail: Tail::mat_mut_segments(this.tail, unsafe { mat.const_cast() }, disjoint.1.tail),
        }
    }

    #[inline]
    fn col_ref_segments<C: Container, T: 'a, RStride: Stride>(
        this: Self,
        mat: ColRef<'a, C, T, Dim<'dim>, RStride>,
    ) -> Self::ColRefSegments<C, T, RStride> {
        let head = mat.row_segment(this.head);
        Cons {
            head,
            tail: Tail::col_ref_segments(this.tail, mat),
        }
    }
    #[inline]
    fn col_mut_segments<C: Container, T: 'a, RStride: Stride>(
        this: Self,
        mat: ColMut<'a, C, T, Dim<'dim>, RStride>,
        disjoint: Self::Disjoint,
    ) -> Self::ColMutSegments<C, T, RStride> {
        let mat = mat.into_const();
        let head = unsafe { mat.row_segment(this.head).const_cast() };
        Cons {
            head,
            tail: Tail::col_mut_segments(this.tail, unsafe { mat.const_cast() }, disjoint.1.tail),
        }
    }

    #[inline]
    unsafe fn cast_cell<C: Container, T, RStride: Stride>(
        segment: Self::ColRefSegments<C, UnsafeCell<T>, RStride>,
    ) -> Self::ColMutSegments<C, T, RStride> {
        Cons {
            head: segment.head.const_cast().as_type::<T>(),
            tail: Tail::cast_cell(segment.tail),
        }
    }
}

impl<'scope, 'dim: 'a, 'range, 'a, Tail: RowSplit<'scope, 'dim, 'a>> RowSplit<'scope, 'dim, 'a>
    for Cons<SegmentIdx<'scope, 'dim, 'range>, Tail>
{
    type MatRefSegments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride> =
        Cons<RowRef<'a, C, T, Cols, CStride>, Tail::MatRefSegments<C, T, Cols, RStride, CStride>>;
    type MatMutSegments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride> =
        Cons<RowMut<'a, C, T, Cols, CStride>, Tail::MatMutSegments<C, T, Cols, RStride, CStride>>;
    type ColRefSegments<C: Container, T: 'a, RStride: Stride> =
        Cons<C::Of<&'a T>, Tail::ColRefSegments<C, T, RStride>>;
    type ColMutSegments<C: Container, T: 'a, RStride: Stride> =
        Cons<C::Of<&'a mut T>, Tail::ColMutSegments<C, T, RStride>>;

    #[inline]
    fn mat_ref_segments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride>(
        this: Self,
        mat: MatRef<'a, C, T, Dim<'dim>, Cols, RStride, CStride>,
    ) -> Self::MatRefSegments<C, T, Cols, RStride, CStride> {
        let head = mat.row(this.head.local());
        Cons {
            head,
            tail: Tail::mat_ref_segments(this.tail, mat),
        }
    }

    #[inline]
    fn mat_mut_segments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride>(
        this: Self,
        mat: MatMut<'a, C, T, Dim<'dim>, Cols, RStride, CStride>,
        disjoint: Self::Disjoint,
    ) -> Self::MatMutSegments<C, T, Cols, RStride, CStride> {
        let mat = mat.into_const();
        let head = unsafe { mat.row(this.head.local()).const_cast() };
        Cons {
            head,
            tail: Tail::mat_mut_segments(this.tail, unsafe { mat.const_cast() }, disjoint.1.tail),
        }
    }

    #[inline]
    fn col_ref_segments<C: Container, T: 'a, RStride: Stride>(
        this: Self,
        mat: ColRef<'a, C, T, Dim<'dim>, RStride>,
    ) -> Self::ColRefSegments<C, T, RStride> {
        let head = mat.at(this.head.local());
        Cons {
            head,
            tail: Tail::col_ref_segments(this.tail, mat),
        }
    }
    #[inline]
    fn col_mut_segments<C: Container, T: 'a, RStride: Stride>(
        this: Self,
        mat: ColMut<'a, C, T, Dim<'dim>, RStride>,
        _: Self::Disjoint,
    ) -> Self::ColMutSegments<C, T, RStride> {
        unsafe {
            let cell = Self::col_ref_segments(this, mat.as_type::<UnsafeCell<T>>().into_const());
            Self::cast_cell(cell)
        }
    }

    #[inline]
    unsafe fn cast_cell<C: Container, T, RStride: Stride>(
        segment: Self::ColRefSegments<C, UnsafeCell<T>, RStride>,
    ) -> Self::ColMutSegments<C, T, RStride> {
        help!(C);
        Cons {
            head: map!(segment.head, head, unsafe { &mut *(head.get()) }),
            tail: Tail::cast_cell(segment.tail),
        }
    }
}

impl<'scope, 'dim: 'a, 'range, 'a, Tail: ColSplit<'scope, 'dim, 'a>> ColSplit<'scope, 'dim, 'a>
    for Cons<Segment<'scope, 'dim, 'range>, Tail>
{
    type MatRefSegments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride> =
        Cons<
            MatRef<'a, C, T, Cols, Dim<'range>, RStride, CStride>,
            Tail::MatRefSegments<C, T, Cols, RStride, CStride>,
        >;
    type MatMutSegments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride> =
        Cons<
            MatMut<'a, C, T, Cols, Dim<'range>, RStride, CStride>,
            Tail::MatMutSegments<C, T, Cols, RStride, CStride>,
        >;
    type RowRefSegments<C: Container, T: 'a, CStride: Stride> =
        Cons<RowRef<'a, C, T, Dim<'range>, CStride>, Tail::RowRefSegments<C, T, CStride>>;
    type RowMutSegments<C: Container, T: 'a, CStride: Stride> =
        Cons<RowMut<'a, C, T, Dim<'range>, CStride>, Tail::RowMutSegments<C, T, CStride>>;

    #[inline]
    fn mat_ref_segments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride>(
        this: Self,
        mat: MatRef<'a, C, T, Cols, Dim<'dim>, RStride, CStride>,
    ) -> Self::MatRefSegments<C, T, Cols, RStride, CStride> {
        let head = mat.col_segment(this.head);
        Cons {
            head,
            tail: Tail::mat_ref_segments(this.tail, mat),
        }
    }

    #[inline]
    fn mat_mut_segments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride>(
        this: Self,
        mat: MatMut<'a, C, T, Cols, Dim<'dim>, RStride, CStride>,
        disjoint: Self::Disjoint,
    ) -> Self::MatMutSegments<C, T, Cols, RStride, CStride> {
        let mat = mat.into_const();
        let head = unsafe { mat.col_segment(this.head).const_cast() };
        Cons {
            head,
            tail: Tail::mat_mut_segments(this.tail, unsafe { mat.const_cast() }, disjoint.1.tail),
        }
    }

    #[inline]
    fn row_ref_segments<C: Container, T: 'a, CStride: Stride>(
        this: Self,
        mat: RowRef<'a, C, T, Dim<'dim>, CStride>,
    ) -> Self::RowRefSegments<C, T, CStride> {
        let head = mat.col_segment(this.head);
        Cons {
            head,
            tail: Tail::row_ref_segments(this.tail, mat),
        }
    }
    #[inline]
    fn row_mut_segments<C: Container, T: 'a, CStride: Stride>(
        this: Self,
        mat: RowMut<'a, C, T, Dim<'dim>, CStride>,
        disjoint: Self::Disjoint,
    ) -> Self::RowMutSegments<C, T, CStride> {
        let mat = mat.into_const();
        let head = unsafe { mat.col_segment(this.head).const_cast() };
        Cons {
            head,
            tail: Tail::row_mut_segments(this.tail, unsafe { mat.const_cast() }, disjoint.1.tail),
        }
    }

    #[inline]
    unsafe fn cast_cell<C: Container, T, RStride: Stride>(
        segment: Self::RowRefSegments<C, UnsafeCell<T>, RStride>,
    ) -> Self::RowMutSegments<C, T, RStride> {
        Cons {
            head: segment.head.const_cast().as_type::<T>(),
            tail: Tail::cast_cell(segment.tail),
        }
    }
}

impl<'scope, 'dim: 'a, 'range, 'a, Tail: ColSplit<'scope, 'dim, 'a>> ColSplit<'scope, 'dim, 'a>
    for Cons<SegmentIdx<'scope, 'dim, 'range>, Tail>
{
    type MatRefSegments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride> =
        Cons<ColRef<'a, C, T, Cols, RStride>, Tail::MatRefSegments<C, T, Cols, RStride, CStride>>;
    type MatMutSegments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride> =
        Cons<ColMut<'a, C, T, Cols, RStride>, Tail::MatMutSegments<C, T, Cols, RStride, CStride>>;
    type RowRefSegments<C: Container, T: 'a, RStride: Stride> =
        Cons<C::Of<&'a T>, Tail::RowRefSegments<C, T, RStride>>;
    type RowMutSegments<C: Container, T: 'a, RStride: Stride> =
        Cons<C::Of<&'a mut T>, Tail::RowMutSegments<C, T, RStride>>;

    #[inline]
    fn mat_ref_segments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride>(
        this: Self,
        mat: MatRef<'a, C, T, Cols, Dim<'dim>, RStride, CStride>,
    ) -> Self::MatRefSegments<C, T, Cols, RStride, CStride> {
        let head = mat.col(this.head.local());
        Cons {
            head,
            tail: Tail::mat_ref_segments(this.tail, mat),
        }
    }

    #[inline]
    fn mat_mut_segments<C: Container, T: 'a, Cols: 'a + Shape, RStride: Stride, CStride: Stride>(
        this: Self,
        mat: MatMut<'a, C, T, Cols, Dim<'dim>, RStride, CStride>,
        disjoint: Self::Disjoint,
    ) -> Self::MatMutSegments<C, T, Cols, RStride, CStride> {
        let mat = mat.into_const();
        let head = unsafe { mat.col(this.head.local()).const_cast() };
        Cons {
            head,
            tail: Tail::mat_mut_segments(this.tail, unsafe { mat.const_cast() }, disjoint.1.tail),
        }
    }

    #[inline]
    fn row_ref_segments<C: Container, T: 'a, RStride: Stride>(
        this: Self,
        mat: RowRef<'a, C, T, Dim<'dim>, RStride>,
    ) -> Self::RowRefSegments<C, T, RStride> {
        let head = mat.at(this.head.local());
        Cons {
            head,
            tail: Tail::row_ref_segments(this.tail, mat),
        }
    }
    #[inline]
    fn row_mut_segments<C: Container, T: 'a, RStride: Stride>(
        this: Self,
        mat: RowMut<'a, C, T, Dim<'dim>, RStride>,
        _: Self::Disjoint,
    ) -> Self::RowMutSegments<C, T, RStride> {
        unsafe {
            let cell = Self::row_ref_segments(this, mat.as_type::<UnsafeCell<T>>().into_const());
            Self::cast_cell(cell)
        }
    }

    #[inline]
    unsafe fn cast_cell<C: Container, T, RStride: Stride>(
        segment: Self::RowRefSegments<C, UnsafeCell<T>, RStride>,
    ) -> Self::RowMutSegments<C, T, RStride> {
        help!(C);
        Cons {
            head: map!(segment.head, head, unsafe { &mut *(head.get()) }),
            tail: Tail::cast_cell(segment.tail),
        }
    }
}
