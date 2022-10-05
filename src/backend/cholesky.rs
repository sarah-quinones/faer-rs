use core::ops::Add;
use core::ops::Mul;
use core::ops::Neg;

use crate::backend::mul;
use crate::backend::mul::triangular::BlockStructure;
use crate::backend::solve;
use crate::izip;
use crate::temp_mat_req;
use crate::temp_mat_uninit;
use crate::MatMut;

use assert2::assert as fancy_assert;
use assert2::debug_assert as fancy_debug_assert;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use num_traits::{Inv, One, Zero};
use reborrow::*;

pub fn cholesky_in_place_left_looking_req<T: 'static>(
    dim: usize,
    block_size: usize,
    max_n_threads: usize,
) -> Result<StackReq, SizeOverflow> {
    let n = dim;
    let bs = block_size.min(dim);

    match n {
        0 | 1 => return Ok(StackReq::default()),
        _ => (),
    }

    use mul::triangular::BlockStructure::*;
    StackReq::try_all_of([
        temp_mat_req::<T>(bs, n - bs)?,
        StackReq::try_any_of([
            mul::triangular::mat_x_mat_req::<T>(
                TriangularLower,
                Rectangular,
                Rectangular,
                bs,
                bs,
                n - bs,
                max_n_threads,
            )?,
            cholesky_in_place_left_looking_req::<T>(bs, bs / 2, max_n_threads)?,
            mul::mat_x_mat_req::<T>(n - bs, bs, n - bs, max_n_threads)?,
            solve::triangular::solve_tri_lower_with_implicit_unit_diagonal_in_place_req::<T>(
                bs,
                bs,
                max_n_threads,
            )?,
        ])?,
    ])
}

pub unsafe fn cholesky_in_place_left_looking_unchecked<T>(
    matrix: MatMut<'_, T>,
    block_size: usize,
    n_threads_hint: usize,
    stack: DynStack<'_>,
) where
    T: Zero + One + Clone + Send + Sync + 'static,
    for<'a> &'a T:
        Add<&'a T, Output = T> + Mul<&'a T, Output = T> + Neg<Output = T> + Inv<Output = T>,
{
    let mut matrix = matrix;

    fancy_debug_assert!(
        matrix.ncols() == matrix.nrows(),
        "only square matrices can be decomposed into cholesky factors",
    );

    let n = matrix.nrows();

    match n {
        0 | 1 => return,
        _ => (),
    };

    let mut idx = 0;
    let mut stack = stack;
    loop {
        let block_size = (n - idx).min(block_size);
        let stack = stack.rb_mut();

        // we split L/D rows/cols into 3 sections each
        //     ┌             ┐
        //     | L00         |
        // L = | L10 A11     |
        //     | L20 A21 A22 |
        //     └             ┘
        //     ┌          ┐
        //     | D0       |
        // D = |    D1    |
        //     |       D2 |
        //     └          ┘
        //
        // we already computed L00, L10, L20, and D0. we now compute L11, L21, and D1

        let (top_left, _, bottom_left, bottom_right) = matrix.rb_mut().split_at_unchecked(idx, idx);
        let l00 = top_left.into_const();
        let d0 = l00.diagonal_unchecked();
        let (_, l10, _, l20) = bottom_left.into_const().split_at_unchecked(block_size, 0);
        let (mut a11, _, mut a21, _) = bottom_right.split_at_unchecked(block_size, block_size);

        // reserve space for L10×D0
        crate::temp_mat_uninit! {
            let (mut l10xd0, mut stack) = unsafe { temp_mat_uninit::<T>(block_size, idx, stack) };
        };

        for (l10xd0_col, l10_col, d_factor) in izip!(
            l10xd0.rb_mut().into_col_iter(),
            l10.rb().into_col_iter(),
            d0.into_iter(),
        ) {
            for (l10xd0_elem, l) in izip!(l10xd0_col, l10_col) {
                *l10xd0_elem = l * d_factor;
            }
        }

        let l10xd0 = l10xd0.into_const();

        mul::triangular::mat_x_mat(
            a11.rb_mut(),
            BlockStructure::TriangularLower,
            l10xd0,
            BlockStructure::Rectangular,
            l10.transpose(),
            BlockStructure::Rectangular,
            Some(&T::one()),
            &-&T::one(),
            n_threads_hint,
            stack.rb_mut(),
        );

        cholesky_in_place_left_looking_unchecked(
            a11.rb_mut(),
            block_size / 2,
            n_threads_hint,
            stack.rb_mut(),
        );

        if idx + block_size == n {
            break;
        }

        let ld11 = a11.into_const();
        let l11 = ld11;
        let d1 = ld11.diagonal_unchecked();

        mul::mat_x_mat(
            a21.rb_mut(),
            l20,
            l10xd0.transpose(),
            Some(&T::one()),
            &-&T::one(),
            n_threads_hint,
            stack.rb_mut(),
        );

        solve::triangular::solve_tri_lower_with_implicit_unit_diagonal_in_place_unchecked(
            l11,
            a21.rb_mut().transpose(),
            n_threads_hint,
            stack,
        );

        let l21xd1 = a21;
        for (l21xd1_col, d1_elem) in izip!(l21xd1.into_col_iter(), d1) {
            let d1_elem_inv = d1_elem.inv();
            for l21xd1_elem in l21xd1_col {
                *l21xd1_elem = &*l21xd1_elem * &d1_elem_inv;
            }
        }

        idx += block_size;
    }
}

pub fn cholesky_in_place_right_looking_req<T: 'static>(
    dim: usize,
    max_n_threads: usize,
) -> Result<StackReq, SizeOverflow> {
    if dim < 32 {
        cholesky_in_place_left_looking_req::<T>(dim, 16, max_n_threads)
    } else {
        let bs = dim / 2;
        let rem = dim - bs;
        StackReq::try_any_of([
            solve::triangular::solve_tri_lower_with_implicit_unit_diagonal_in_place_req::<T>(
                bs,
                rem,
                max_n_threads,
            )?,
            StackReq::try_all_of([
                temp_mat_req::<T>(rem, bs)?,
                mul::triangular::mat_x_mat_req::<T>(
                    BlockStructure::TriangularLower,
                    BlockStructure::Rectangular,
                    BlockStructure::Rectangular,
                    rem,
                    rem,
                    bs,
                    max_n_threads,
                )?,
            ])?,
        ])
    }
}

pub unsafe fn cholesky_in_place_right_looking_unchecked<T>(
    matrix: MatMut<'_, T>,
    n_threads_hint: usize,
    stack: DynStack<'_>,
) where
    T: Zero + One + Clone + Neg<Output = T> + Send + Sync + 'static,
    for<'a> &'a T:
        Add<&'a T, Output = T> + Mul<&'a T, Output = T> + Neg<Output = T> + Inv<Output = T>,
{
    fancy_debug_assert!(matrix.nrows() == matrix.ncols());
    let mut matrix = matrix;
    let mut stack = stack;

    let n = matrix.nrows();
    if n < 32 {
        cholesky_in_place_left_looking_unchecked(matrix, 16, n_threads_hint, stack);
    } else {
        let block_size = n / 2;
        let rem = n - block_size;
        let (mut l00, _, mut a10, mut a11) =
            matrix.rb_mut().split_at_unchecked(block_size, block_size);

        cholesky_in_place_right_looking_unchecked(l00.rb_mut(), n_threads_hint, stack.rb_mut());

        let l00 = l00.into_const();
        let d0 = l00.diagonal_unchecked();

        solve::triangular::solve_tri_lower_with_implicit_unit_diagonal_in_place_unchecked(
            l00,
            a10.rb_mut().transpose(),
            n_threads_hint,
            stack.rb_mut(),
        );

        {
            temp_mat_uninit! {
                let (mut l10xd0, stack) = unsafe { temp_mat_uninit::<T>(rem, block_size, stack.rb_mut()) };
            };

            for (l10xd0_col, a10_col, d0_elem) in izip!(
                l10xd0.rb_mut().into_col_iter(),
                a10.rb_mut().into_col_iter(),
                d0,
            ) {
                let d0_elem_inv = d0_elem.inv();
                for (l10xd0_elem, a10_elem) in izip!(l10xd0_col, a10_col) {
                    *l10xd0_elem = a10_elem.clone();
                    *a10_elem = &*a10_elem * &d0_elem_inv;
                }
            }

            mul::triangular::mat_x_mat(
                a11.rb_mut(),
                BlockStructure::TriangularLower,
                a10.into_const(),
                BlockStructure::Rectangular,
                l10xd0.transpose().into_const(),
                BlockStructure::Rectangular,
                Some(&T::one()),
                &-&T::one(),
                n_threads_hint,
                stack,
            );
        }

        cholesky_in_place_right_looking_unchecked(a11, n_threads_hint, stack);
    }
}

#[inline]
pub fn cholesky_in_place_left_looking<T>(
    matrix: MatMut<'_, T>,
    block_size: usize,
    n_threads_hint: usize,
    stack: DynStack<'_>,
) where
    T: Zero + One + Clone + Neg<Output = T> + Send + Sync + 'static,
    for<'a> &'a T:
        Add<&'a T, Output = T> + Mul<&'a T, Output = T> + Neg<Output = T> + Inv<Output = T>,
{
    fancy_assert!(
        matrix.ncols() == matrix.nrows(),
        "only square matrices can be decomposed into cholesky factors",
    );
    unsafe { cholesky_in_place_left_looking_unchecked(matrix, block_size, n_threads_hint, stack) }
}

#[inline]
pub fn cholesky_in_place_right_looking<T>(
    matrix: MatMut<'_, T>,
    n_threads_hint: usize,
    stack: DynStack<'_>,
) where
    T: Zero + One + Clone + Neg<Output = T> + Send + Sync + 'static,
    for<'a> &'a T:
        Add<&'a T, Output = T> + Mul<&'a T, Output = T> + Neg<Output = T> + Inv<Output = T>,
{
    fancy_assert!(
        matrix.ncols() == matrix.nrows(),
        "only square matrices can be decomposed into cholesky factors",
    );
    unsafe { cholesky_in_place_right_looking_unchecked(matrix, n_threads_hint, stack) }
}

#[cfg(test)]
mod tests {
    use dyn_stack::GlobalMemBuffer;

    use super::*;
    use crate::mat;

    #[test]
    pub fn cholesky() {
        #[rustfmt::skip]
        let mut a = mat![
            [1.0, f64::NAN],
            [2.0, 1.0],
        ];

        let dim = a.nrows();

        cholesky_in_place_left_looking(
            a.as_mut(),
            64,
            1,
            DynStack::new(&mut GlobalMemBuffer::new(
                cholesky_in_place_left_looking_req::<f64>(dim, 64, 1).unwrap(),
            )),
        );

        dbg!(a);
    }
}
