use core::ops::{Add, Mul, Neg};

use crate::{MatMut, MatRef};

use assert2::{assert as fancy_assert, debug_assert as fancy_debug_assert};
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use num_traits::{Inv, One, Zero};
use reborrow::*;

pub mod triangular {
    use crate::join;

    use super::*;

    #[inline]
    fn split_half(n_threads: usize) -> usize {
        n_threads / 2
    }

    #[inline]
    // we want remainder to be a multiple of register size
    fn blocksize<T: 'static>(n: usize) -> usize {
        let base_rem = n / 2;
        n - if n >= 32 {
            (base_rem + 15) / 16 * 16
        } else if n >= 16 {
            (base_rem + 7) / 8 * 8
        } else if n >= 8 {
            (base_rem + 3) / 4 * 4
        } else {
            base_rem
        }
    }

    #[inline]
    fn recursion_threshold<T: 'static>() -> usize {
        4
    }

    pub fn solve_triangular_in_place_req<T: 'static>(
        _dim: usize,
        _rhs_ncols: usize,
        _n_threads: usize,
    ) -> Result<StackReq, SizeOverflow> {
        Ok(StackReq::default())
    }

    #[track_caller]
    #[inline]
    pub fn solve_lower_triangular_in_place<T>(
        triangular_lower: MatRef<'_, T>,
        rhs: MatMut<'_, T>,
        n_threads: usize,
        stack: DynStack<'_>,
    ) where
        T: Zero + One + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Inv<Output = T>,
    {
        fancy_assert!(triangular_lower.nrows() == triangular_lower.ncols());
        fancy_assert!(rhs.nrows() == triangular_lower.ncols());

        unsafe {
            solve_lower_triangular_in_place_unchecked(triangular_lower, rhs, n_threads, stack);
        }
    }

    #[track_caller]
    #[inline]
    pub fn solve_upper_triangular_in_place<T>(
        triangular_upper: MatRef<'_, T>,
        rhs: MatMut<'_, T>,
        n_threads: usize,
        stack: DynStack<'_>,
    ) where
        T: Zero + One + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Inv<Output = T>,
    {
        fancy_assert!(triangular_upper.nrows() == triangular_upper.ncols());
        fancy_assert!(rhs.nrows() == triangular_upper.ncols());

        unsafe {
            solve_upper_triangular_in_place_unchecked(triangular_upper, rhs, n_threads, stack);
        }
    }

    #[track_caller]
    #[inline]
    pub fn solve_unit_lower_triangular_in_place<T>(
        triangular_lower: MatRef<'_, T>,
        rhs: MatMut<'_, T>,
        n_threads: usize,
        stack: DynStack<'_>,
    ) where
        T: Zero + One + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        fancy_assert!(triangular_lower.nrows() == triangular_lower.ncols());
        fancy_assert!(rhs.nrows() == triangular_lower.ncols());

        unsafe {
            solve_unit_lower_triangular_in_place_unchecked(triangular_lower, rhs, n_threads, stack);
        }
    }

    #[track_caller]
    #[inline]
    pub fn solve_unit_upper_triangular_in_place<T>(
        triangular_upper: MatRef<'_, T>,
        rhs: MatMut<'_, T>,
        n_threads: usize,
        stack: DynStack<'_>,
    ) where
        T: Zero + One + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        fancy_assert!(triangular_upper.nrows() == triangular_upper.ncols());
        fancy_assert!(rhs.nrows() == triangular_upper.ncols());

        unsafe {
            solve_unit_upper_triangular_in_place_unchecked(triangular_upper, rhs, n_threads, stack);
        }
    }

    pub unsafe fn solve_unit_lower_triangular_in_place_unchecked<T>(
        tril: MatRef<'_, T>,
        rhs: MatMut<'_, T>,
        n_threads: usize,
        stack: DynStack<'_>,
    ) where
        T: Zero + One + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        let n = tril.nrows();
        let k = rhs.ncols();

        if n_threads > 1 && k > 64 && n <= 128 {
            let (_, _, rhs_left, rhs_right) = rhs.split_at_unchecked(0, k / 2);
            join(
                |n_threads, stack| {
                    solve_unit_lower_triangular_in_place_unchecked(tril, rhs_left, n_threads, stack)
                },
                |n_threads, stack| {
                    solve_unit_lower_triangular_in_place_unchecked(
                        tril, rhs_right, n_threads, stack,
                    )
                },
                |n_threads| solve_triangular_in_place_req::<T>(n, k / 2, n_threads).unwrap(),
                split_half,
                n_threads,
                stack,
            );
            return;
        }

        fancy_debug_assert!(tril.nrows() == tril.ncols());
        fancy_debug_assert!(rhs.nrows() == tril.ncols());

        if n <= recursion_threshold::<T>() {
            pulp::Arch::new().dispatch(
                #[inline(always)]
                || match n {
                    0 => (),
                    1 => (),
                    2 => {
                        let nl10_div_l11 = &-tril.get_unchecked(1, 0);

                        let (_, x0, _, x1) = rhs.split_at_unchecked(1, 0);
                        let x0 = x0.row_unchecked(0);
                        let x1 = x1.row_unchecked(0);

                        x0.cwise().zip_unchecked(x1).for_each(|x0, x1| {
                            *x1 = &*x1 + &(nl10_div_l11 * &*x0);
                        });
                    }
                    3 => {
                        let nl10_div_l11 = &-tril.get_unchecked(1, 0);
                        let nl20_div_l22 = &-tril.get_unchecked(2, 0);
                        let nl21_div_l22 = &-tril.get_unchecked(2, 1);

                        let (_, x0, _, x1_2) = rhs.split_at_unchecked(1, 0);
                        let (_, x1, _, x2) = x1_2.split_at_unchecked(1, 0);
                        let x0 = x0.row_unchecked(0);
                        let x1 = x1.row_unchecked(0);
                        let x2 = x2.row_unchecked(0);

                        x0.cwise()
                            .zip_unchecked(x1)
                            .zip_unchecked(x2)
                            .for_each(|x0, x1, x2| {
                                let y0 = x0.clone();
                                let mut y1 = x1.clone();
                                let mut y2 = x2.clone();
                                y1 = y1 + nl10_div_l11 * &y0;
                                y2 = y2 + nl20_div_l22 * &y0 + nl21_div_l22 * &y1;
                                *x1 = y1;
                                *x2 = y2;
                            });
                    }
                    4 => {
                        let nl10_div_l11 = &-tril.get_unchecked(1, 0);
                        let nl20_div_l22 = &-tril.get_unchecked(2, 0);
                        let nl21_div_l22 = &-tril.get_unchecked(2, 1);
                        let nl30_div_l33 = &-tril.get_unchecked(3, 0);
                        let nl31_div_l33 = &-tril.get_unchecked(3, 1);
                        let nl32_div_l33 = &-tril.get_unchecked(3, 2);

                        let (_, x0, _, x1_2_3) = rhs.split_at_unchecked(1, 0);
                        let (_, x1, _, x2_3) = x1_2_3.split_at_unchecked(1, 0);
                        let (_, x2, _, x3) = x2_3.split_at_unchecked(1, 0);
                        let x0 = x0.row_unchecked(0);
                        let x1 = x1.row_unchecked(0);
                        let x2 = x2.row_unchecked(0);
                        let x3 = x3.row_unchecked(0);

                        x0.cwise()
                            .zip_unchecked(x1)
                            .zip_unchecked(x2)
                            .zip_unchecked(x3)
                            .for_each(|x0, x1, x2, x3| {
                                let y0 = x0.clone();
                                let mut y1 = x1.clone();
                                let mut y2 = x2.clone();
                                let mut y3 = x3.clone();
                                y1 = y1 + nl10_div_l11 * &y0;
                                y2 = y2 + (nl20_div_l22 * &y0 + nl21_div_l22 * &y1);
                                y3 = (y3 + nl30_div_l33 * &y0)
                                    + (nl31_div_l33 * &y1 + nl32_div_l33 * &y2);
                                *x1 = y1;
                                *x2 = y2;
                                *x3 = y3;
                            });
                    }
                    _ => unreachable!(),
                },
            );
            return;
        }

        let mut stack = stack;
        let bs = blocksize::<T>(n);

        let (tril_top_left, _, tril_bot_left, tril_bot_right) = tril.split_at_unchecked(bs, bs);
        let (_, mut rhs_top, _, mut rhs_bot) = rhs.split_at_unchecked(bs, 0);

        solve_unit_lower_triangular_in_place_unchecked(
            tril_top_left,
            rhs_top.rb_mut(),
            n_threads,
            stack.rb_mut(),
        );

        crate::mul::matmul(
            rhs_bot.rb_mut(),
            tril_bot_left,
            rhs_top.into_const(),
            Some(&T::one()),
            &-&T::one(),
            n_threads,
        );

        solve_unit_lower_triangular_in_place_unchecked(tril_bot_right, rhs_bot, n_threads, stack);
    }

    #[inline]
    pub unsafe fn solve_unit_upper_triangular_in_place_unchecked<T>(
        triu: MatRef<'_, T>,
        rhs: MatMut<'_, T>,
        n_threads: usize,
        stack: DynStack<'_>,
    ) where
        T: Zero + One + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        solve_unit_lower_triangular_in_place_unchecked(
            triu.invert(),
            rhs.invert_rows(),
            n_threads,
            stack,
        );
    }

    pub unsafe fn solve_lower_triangular_in_place_unchecked<T>(
        tril: MatRef<'_, T>,
        rhs: MatMut<'_, T>,
        n_threads: usize,
        stack: DynStack<'_>,
    ) where
        T: Zero + One + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Inv<Output = T>,
    {
        let n = tril.nrows();
        let k = rhs.ncols();

        if n_threads > 1 && k > 64 && n <= 128 {
            let (_, _, rhs_left, rhs_right) = rhs.split_at_unchecked(0, k / 2);
            join(
                |n_threads, stack| {
                    solve_lower_triangular_in_place_unchecked(tril, rhs_left, n_threads, stack)
                },
                |n_threads, stack| {
                    solve_lower_triangular_in_place_unchecked(tril, rhs_right, n_threads, stack)
                },
                |n_threads| solve_triangular_in_place_req::<T>(n, k / 2, n_threads).unwrap(),
                split_half,
                n_threads,
                stack,
            );
            return;
        }

        fancy_debug_assert!(tril.nrows() == tril.ncols());
        fancy_debug_assert!(rhs.nrows() == tril.ncols());

        let n = tril.nrows();

        if n <= recursion_threshold::<T>() {
            pulp::Arch::new().dispatch(
                #[inline(always)]
                || match n {
                    0 => (),
                    1 => {
                        let inv = &tril.get_unchecked(0, 0).inv();
                        let row = rhs.row_unchecked(0);
                        row.cwise().for_each(|e| *e = &*e * inv);
                    }
                    2 => {
                        let l00_inv = &tril.get_unchecked(0, 0).inv();
                        let l11_inv = &tril.get_unchecked(1, 1).inv();
                        let nl10_div_l11 = &-&(tril.get_unchecked(1, 0) * l11_inv);

                        let (_, x0, _, x1) = rhs.split_at_unchecked(1, 0);
                        let x0 = x0.row_unchecked(0);
                        let x1 = x1.row_unchecked(0);

                        x0.cwise().zip_unchecked(x1).for_each(|x0, x1| {
                            *x0 = &*x0 * l00_inv;
                            *x1 = &*x1 * l11_inv + nl10_div_l11 * &*x0;
                        });
                    }
                    3 => {
                        let l00_inv = &tril.get_unchecked(0, 0).inv();
                        let l11_inv = &tril.get_unchecked(1, 1).inv();
                        let l22_inv = &tril.get_unchecked(2, 2).inv();
                        let nl10_div_l11 = &-&(tril.get_unchecked(1, 0) * l11_inv);
                        let nl20_div_l22 = &-&(tril.get_unchecked(2, 0) * l22_inv);
                        let nl21_div_l22 = &-&(tril.get_unchecked(2, 1) * l22_inv);

                        let (_, x0, _, x1_2) = rhs.split_at_unchecked(1, 0);
                        let (_, x1, _, x2) = x1_2.split_at_unchecked(1, 0);
                        let x0 = x0.row_unchecked(0);
                        let x1 = x1.row_unchecked(0);
                        let x2 = x2.row_unchecked(0);

                        x0.cwise()
                            .zip_unchecked(x1)
                            .zip_unchecked(x2)
                            .for_each(|x0, x1, x2| {
                                let mut y0 = x0.clone();
                                let mut y1 = x1.clone();
                                let mut y2 = x2.clone();
                                y0 = &y0 * l00_inv;
                                y1 = &y1 * l11_inv + nl10_div_l11 * &y0;
                                y2 = &y2 * l22_inv + nl20_div_l22 * &y0 + nl21_div_l22 * &y1;
                                *x0 = y0;
                                *x1 = y1;
                                *x2 = y2;
                            });
                    }
                    4 => {
                        let l00_inv = &tril.get_unchecked(0, 0).inv();
                        let l11_inv = &tril.get_unchecked(1, 1).inv();
                        let l22_inv = &tril.get_unchecked(2, 2).inv();
                        let l33_inv = &tril.get_unchecked(3, 3).inv();
                        let nl10_div_l11 = &-&(tril.get_unchecked(1, 0) * l11_inv);
                        let nl20_div_l22 = &-&(tril.get_unchecked(2, 0) * l22_inv);
                        let nl21_div_l22 = &-&(tril.get_unchecked(2, 1) * l22_inv);
                        let nl30_div_l33 = &-&(tril.get_unchecked(3, 0) * l33_inv);
                        let nl31_div_l33 = &-&(tril.get_unchecked(3, 1) * l33_inv);
                        let nl32_div_l33 = &-&(tril.get_unchecked(3, 2) * l33_inv);

                        let (_, x0, _, x1_2_3) = rhs.split_at_unchecked(1, 0);
                        let (_, x1, _, x2_3) = x1_2_3.split_at_unchecked(1, 0);
                        let (_, x2, _, x3) = x2_3.split_at_unchecked(1, 0);
                        let x0 = x0.row_unchecked(0);
                        let x1 = x1.row_unchecked(0);
                        let x2 = x2.row_unchecked(0);
                        let x3 = x3.row_unchecked(0);

                        x0.cwise()
                            .zip_unchecked(x1)
                            .zip_unchecked(x2)
                            .zip_unchecked(x3)
                            .for_each(|x0, x1, x2, x3| {
                                let mut y0 = x0.clone();
                                let mut y1 = x1.clone();
                                let mut y2 = x2.clone();
                                let mut y3 = x3.clone();
                                y0 = &y0 * l00_inv;
                                y1 = &y1 * l11_inv + nl10_div_l11 * &y0;
                                y2 = &y2 * l22_inv + (nl20_div_l22 * &y0 + nl21_div_l22 * &y1);
                                y3 = (&y3 * l33_inv + nl30_div_l33 * &y0)
                                    + (nl31_div_l33 * &y1 + nl32_div_l33 * &y2);
                                *x0 = y0;
                                *x1 = y1;
                                *x2 = y2;
                                *x3 = y3;
                            });
                    }
                    _ => unreachable!(),
                },
            );
            return;
        }

        let mut stack = stack;
        let bs = blocksize::<T>(n);

        let (tril_top_left, _, tril_bot_left, tril_bot_right) = tril.split_at_unchecked(bs, bs);
        let (_, mut rhs_top, _, mut rhs_bot) = rhs.split_at_unchecked(bs, 0);

        solve_lower_triangular_in_place_unchecked(
            tril_top_left,
            rhs_top.rb_mut(),
            n_threads,
            stack.rb_mut(),
        );

        crate::mul::matmul_unchecked(
            rhs_bot.rb_mut(),
            tril_bot_left,
            rhs_top.into_const(),
            Some(&T::one()),
            &-&T::one(),
            n_threads,
        );

        solve_lower_triangular_in_place_unchecked(tril_bot_right, rhs_bot, n_threads, stack);
    }

    #[inline]
    pub unsafe fn solve_upper_triangular_in_place_unchecked<T>(
        triu: MatRef<'_, T>,
        rhs: MatMut<'_, T>,
        n_threads: usize,
        stack: DynStack<'_>,
    ) where
        T: Zero + One + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Inv<Output = T>,
    {
        solve_lower_triangular_in_place_unchecked(
            triu.invert(),
            rhs.invert_rows(),
            n_threads,
            stack,
        );
    }
}
