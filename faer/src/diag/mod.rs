use crate::internal_prelude::*;

pub(crate) mod diagmut;
pub(crate) mod diagown;
pub(crate) mod diagref;

pub use diagmut::DiagMut as DiagMutGeneric;
pub use diagown::Diag as DiagGeneric;
pub use diagref::DiagRef as DiagRefGeneric;
use faer_traits::Unit;

pub type DiagRef<'a, T, Len = usize, Stride = isize> = DiagRefGeneric<'a, Unit, T, Len, Stride>;
pub type DiagMut<'a, T, Len = usize, Stride = isize> = DiagMutGeneric<'a, Unit, T, Len, Stride>;
pub type Diag<T, Len = usize> = DiagGeneric<Unit, T, Len>;
