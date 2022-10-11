use core::ops::{Add, Mul, Neg};

use assert2::debug_assert as fancy_debug_assert;
use dyn_stack::DynStack;
use num_traits::{One, Zero};
use reborrow::*;

use crate::{mul, MatMut, MatRef};

/// Same as [`matmul`], except that panics become undefined behavior.
#[inline]
pub unsafe fn matmul_unchecked<T>(
    dst_real: MatMut<'_, T>,
    dst_imag: MatMut<'_, T>,
    lhs_real: MatRef<'_, T>,
    lhs_imag: MatRef<'_, T>,
    lhs_conj: bool,
    rhs_real: MatRef<'_, T>,
    rhs_imag: MatRef<'_, T>,
    rhs_conj: bool,
    alpha: Option<&T>,
    beta: &T,
    n_threads: usize,
    stack: DynStack<'_>,
) where
    T: Zero + One + Clone + Send + Sync + 'static,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
{
    fancy_debug_assert!(dst_real.nrows() == dst_imag.nrows());
    fancy_debug_assert!(dst_real.ncols() == dst_imag.ncols());

    fancy_debug_assert!(lhs_real.nrows() == lhs_imag.nrows());
    fancy_debug_assert!(lhs_real.ncols() == lhs_imag.ncols());

    fancy_debug_assert!(rhs_real.nrows() == rhs_imag.nrows());
    fancy_debug_assert!(rhs_real.ncols() == rhs_imag.ncols());

    fancy_debug_assert!(dst_real.nrows() == lhs_real.nrows());
    fancy_debug_assert!(dst_real.ncols() == lhs_real.ncols());
    fancy_debug_assert!(lhs_real.ncols() == rhs_real.nrows());

    let mut dst_real = dst_real;
    let mut dst_imag = dst_imag;

    let mut stack = stack;
    mul::matmul_unchecked(
        dst_real.rb_mut(),
        lhs_real,
        rhs_real,
        alpha,
        beta,
        n_threads,
        stack.rb_mut(),
    );
    let neg = lhs_conj == rhs_conj;
    mul::matmul_unchecked(
        dst_real.rb_mut(),
        lhs_imag,
        rhs_imag,
        Some(&T::one()),
        &if neg { -beta } else { beta.clone() },
        n_threads,
        stack.rb_mut(),
    );
    mul::matmul_unchecked(
        dst_imag.rb_mut(),
        lhs_real,
        rhs_imag,
        alpha,
        &if rhs_conj { -beta } else { beta.clone() },
        n_threads,
        stack.rb_mut(),
    );
    mul::matmul_unchecked(
        dst_imag.rb_mut(),
        lhs_imag,
        rhs_real,
        Some(&T::one()),
        &if lhs_conj { -beta } else { beta.clone() },
        n_threads,
        stack.rb_mut(),
    );
}

pub mod triangular {
    use super::*;
    use crate::mul::triangular::BlockStructure;

    #[inline]
    pub unsafe fn matmul_unchecked<T>(
        dst_real: MatMut<'_, T>,
        dst_imag: MatMut<'_, T>,
        dst_structure: BlockStructure,
        lhs_real: MatRef<'_, T>,
        lhs_imag: MatRef<'_, T>,
        lhs_structure: BlockStructure,
        lhs_conj: bool,
        rhs_real: MatRef<'_, T>,
        rhs_imag: MatRef<'_, T>,
        rhs_structure: BlockStructure,
        rhs_conj: bool,
        alpha: Option<&T>,
        beta: &T,
        n_threads: usize,
        stack: DynStack<'_>,
    ) where
        T: Zero + One + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        fancy_debug_assert!(dst_real.nrows() == dst_imag.nrows());
        fancy_debug_assert!(dst_real.ncols() == dst_imag.ncols());

        fancy_debug_assert!(lhs_real.nrows() == lhs_imag.nrows());
        fancy_debug_assert!(lhs_real.ncols() == lhs_imag.ncols());

        fancy_debug_assert!(rhs_real.nrows() == rhs_imag.nrows());
        fancy_debug_assert!(rhs_real.ncols() == rhs_imag.ncols());

        fancy_debug_assert!(dst_real.nrows() == lhs_real.nrows());
        fancy_debug_assert!(dst_real.ncols() == lhs_real.ncols());
        fancy_debug_assert!(lhs_real.ncols() == rhs_real.nrows());

        let mut dst_real = dst_real;
        let mut dst_imag = dst_imag;

        let mut stack = stack;

        mul::triangular::matmul_unchecked(
            dst_real.rb_mut(),
            dst_structure,
            lhs_real,
            lhs_structure,
            rhs_real,
            rhs_structure,
            alpha,
            beta,
            n_threads,
            stack.rb_mut(),
        );
        let neg = lhs_conj == rhs_conj;
        mul::triangular::matmul_unchecked(
            dst_real.rb_mut(),
            dst_structure,
            lhs_imag,
            lhs_structure,
            rhs_imag,
            rhs_structure,
            Some(&T::one()),
            &if neg { -beta } else { beta.clone() },
            n_threads,
            stack.rb_mut(),
        );
        mul::triangular::matmul_unchecked(
            dst_imag.rb_mut(),
            dst_structure,
            lhs_real,
            lhs_structure,
            rhs_imag,
            rhs_structure,
            alpha,
            &if rhs_conj { -beta } else { beta.clone() },
            n_threads,
            stack.rb_mut(),
        );
        mul::triangular::matmul_unchecked(
            dst_imag.rb_mut(),
            dst_structure,
            lhs_imag,
            lhs_structure,
            rhs_real,
            rhs_structure,
            Some(&T::one()),
            &if lhs_conj { -beta } else { beta.clone() },
            n_threads,
            stack.rb_mut(),
        );
    }
}
