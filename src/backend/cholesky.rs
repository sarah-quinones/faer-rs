use crate::backend::mul;
use crate::backend::solve;
use crate::izip;
use crate::temp_mat_req;
use crate::temp_mat_uninit;
use crate::MatMut;

use assert2::assert as fancy_assert;
use assert2::debug_assert as fancy_debug_assert;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use num_traits::{One, Zero};
use reborrow::*;

pub fn cholesky_in_place_left_looking_req<T: 'static>(
    max_dim: usize,
    max_block_size: usize,
    max_n_threads: usize,
) -> Result<StackReq, SizeOverflow> {
    let n = max_dim;
    let bs = max_block_size.min(max_dim);

    match n {
        0 | 1 => return Ok(StackReq::default()),
        _ => (),
    }

    StackReq::try_all_of([
        temp_mat_req::<T>(bs, n - bs)?,
        StackReq::try_any_of([
            mul::triangular::mat_mat_accum_dst_lower_half_only_req::<T>(bs, n - bs, max_n_threads)?,
            cholesky_in_place_left_looking_req::<T>(bs, bs / 2, max_n_threads)?,
            mul::mat_mat_accum_req::<T>(n - bs, bs, n - bs, max_n_threads)?,
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
    T: Zero + One + Clone + core::ops::Neg<Output = T> + Send + Sync + 'static,
    for<'a> &'a T: core::ops::Add<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Mul<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Div<&'a T, Output = T>,
{
    let mut matrix = matrix;

    fancy_debug_assert!(
        matrix.ncols() == matrix.nrows(),
        "only square matrices can be decomposed into cholesky factors",
    );

    let n = matrix.nrows();

    match n {
        0 | 1 => return,
        2 => {
            let d0 = matrix.rb().get_unchecked(0, 0).clone();
            let l10 = matrix.rb_mut().get_unchecked(1, 0);
            *l10 = &*l10 / &d0;
            let l10 = l10.clone();
            let d1 = matrix.rb_mut().get_unchecked(1, 1);
            *d1 = &*d1 + &-(&(&l10 * &d0) * &l10);
        }
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
            let (mut l10xd0, mut stack) = temp_mat_uninit::<T>(block_size, idx, stack);
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

        mul::triangular::mat_mat_accum_dst_lower_half_only_unchecked(
            a11.rb_mut(),
            l10xd0,
            l10.trans(),
            T::one(),
            -T::one(),
            n_threads_hint,
            stack.rb_mut(),
        );

        cholesky_in_place_left_looking(
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

        mul::mat_mat_accum(
            a21.rb_mut(),
            l20,
            l10xd0.trans(),
            T::one(),
            -T::one(),
            n_threads_hint,
            stack.rb_mut(),
        );

        solve::triangular::solve_tri_lower_with_implicit_unit_diagonal_in_place_unchecked(
            l11,
            a21.rb_mut().trans(),
            n_threads_hint,
            stack,
        );

        let l21xd1 = a21;
        for (l21xd1_col, d1_elem) in izip!(l21xd1.into_col_iter(), d1) {
            let d1_elem_inv = &T::one() / d1_elem;
            for l21xd1_elem in l21xd1_col {
                *l21xd1_elem = &*l21xd1_elem * &d1_elem_inv;
            }
        }

        idx += block_size;
    }
}

pub fn cholesky_in_place_right_looking_req<T: 'static>(
    max_dim: usize,
    max_n_threads: usize,
) -> Result<StackReq, SizeOverflow> {
    if max_dim < 32 {
        cholesky_in_place_left_looking_req::<T>(max_dim, 16, max_n_threads)
    } else {
        let block_size = max_dim / 2;
        let rem = max_dim - block_size;
        StackReq::try_any_of([
            solve::triangular::solve_tri_lower_with_implicit_unit_diagonal_in_place_req::<T>(
                block_size,
                rem,
                max_n_threads,
            )?,
            StackReq::try_all_of([
                temp_mat_req::<T>(rem, block_size)?,
                mul::triangular::mat_mat_accum_dst_lower_half_only_req::<T>(
                    rem,
                    block_size,
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
    T: Zero + One + Clone + core::ops::Neg<Output = T> + Send + Sync + 'static,
    for<'a> &'a T: core::ops::Add<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Mul<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Div<&'a T, Output = T>,
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
            a10.rb_mut().trans(),
            n_threads_hint,
            stack.rb_mut(),
        );

        {
            temp_mat_uninit! {
                let (mut l10xd0, stack) = temp_mat_uninit::<T>(rem, block_size, stack.rb_mut());
            };

            for (l10xd0_col, a10_col, d0_elem) in izip!(
                l10xd0.rb_mut().into_col_iter(),
                a10.rb_mut().into_col_iter(),
                d0,
            ) {
                let d0_elem_inv = &T::one() / d0_elem;
                for (l10xd0_elem, a10_elem) in izip!(l10xd0_col, a10_col) {
                    *l10xd0_elem = a10_elem.clone();
                    *a10_elem = &*a10_elem * &d0_elem_inv;
                }
            }

            mul::triangular::mat_mat_accum_dst_lower_half_only_unchecked(
                a11.rb_mut(),
                a10.into_const(),
                l10xd0.trans().into_const(),
                T::one(),
                -T::one(),
                n_threads_hint,
                stack,
            )
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
    T: Zero + One + Clone + core::ops::Neg<Output = T> + Send + Sync + 'static,
    for<'a> &'a T: core::ops::Add<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Mul<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Div<&'a T, Output = T>,
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
    T: Zero + One + Clone + core::ops::Neg<Output = T> + Send + Sync + 'static,
    for<'a> &'a T: core::ops::Add<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Mul<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Div<&'a T, Output = T>,
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
