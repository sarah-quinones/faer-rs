use crate::internal_prelude::*;
use core::fmt;
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
macro_rules! l {
    ($($t:tt)*) => {
        $crate::__list_impl![@ __impl @ () @ ($($t)*)]
    };
}

/// type of a variadic tuple containing the given types.
#[macro_export]
macro_rules! L {
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
pub use {L, l};
