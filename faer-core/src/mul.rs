use core::ops::{Add, Mul};

use crate::{join, MatMut, MatRef};

use assert2::{assert as fancy_assert, debug_assert as fancy_debug_assert};
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use gemm::{gemm, gemm_req};
use num_traits::{One, Zero};
use reborrow::*;

#[inline]
fn split_half(n_threads: usize) -> usize {
    n_threads / 2
}

/// Computes the memory requirements of [`matmul`].
pub fn matmul_req<T: 'static>(
    dst_rows: usize,
    dst_cols: usize,
    lhs_cols: usize,
    n_threads: usize,
) -> Result<StackReq, SizeOverflow> {
    let m = dst_rows;
    let n = dst_cols;
    let k = lhs_cols;
    gemm_req::<T>(m, n, k, n_threads)
}

/// Same as [`matmul`], except that panics become undefined behavior.
#[inline]
pub unsafe fn matmul_unchecked<T>(
    dst: MatMut<'_, T>,
    lhs: MatRef<'_, T>,
    rhs: MatRef<'_, T>,
    alpha: Option<&T>,
    beta: &T,
    n_threads: usize,
    stack: DynStack<'_>,
) where
    T: Zero + Clone + Send + Sync + 'static,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T>,
{
    fancy_debug_assert!(dst.nrows() == lhs.nrows());
    fancy_debug_assert!(dst.ncols() == rhs.ncols());
    fancy_debug_assert!(lhs.ncols() == rhs.nrows());

    let m = dst.nrows();
    let n = dst.ncols();
    let k = lhs.ncols();

    let dst_col_stride = dst.col_stride();
    let dst_row_stride = dst.row_stride();

    // SAFETY:
    // * matching operand/destination dimensions.
    // * strides were verified during creation of matrix views.
    match alpha {
        Some(alpha) => gemm(
            m,
            n,
            k,
            dst.as_ptr(),
            dst_col_stride,
            dst_row_stride,
            true,
            lhs.as_ptr(),
            lhs.col_stride(),
            lhs.row_stride(),
            rhs.as_ptr(),
            rhs.col_stride(),
            rhs.row_stride(),
            alpha.clone(),
            beta.clone(),
            n_threads,
            stack,
        ),
        None => gemm(
            m,
            n,
            k,
            dst.as_ptr(),
            dst_col_stride,
            dst_row_stride,
            false,
            lhs.as_ptr(),
            lhs.col_stride(),
            lhs.row_stride(),
            rhs.as_ptr(),
            rhs.col_stride(),
            rhs.row_stride(),
            T::zero(),
            beta.clone(),
            n_threads,
            stack,
        ),
    }
}

/// Computes the matrix product `[alpha * dst] + beta * lhs * rhs` and stores the result in `dst`.
///
/// If `alpha` is not provided, he preexisting values in `dst` are not read so it is allowed to be
/// a view over uninitialized values if `T: Copy`.
///
/// # Panics
///
/// Panics if the matrix dimensions are not compatible for matrix multiplication.  
/// i.e.  
///  - `dst.nrows() == lhs.nrows()`
///  - `dst.ncols() == rhs.ncols()`
///  - `lhs.ncols() == rhs.nrows()`
#[track_caller]
#[inline]
pub fn matmul<T>(
    dst: MatMut<'_, T>,
    lhs: MatRef<'_, T>,
    rhs: MatRef<'_, T>,
    alpha: Option<&T>,
    beta: &T,
    n_threads: usize,
    stack: DynStack<'_>,
) where
    T: Zero + Clone + Send + Sync + 'static,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T>,
{
    fancy_assert!(dst.nrows() == lhs.nrows());
    fancy_assert!(dst.ncols() == rhs.ncols());
    fancy_assert!(lhs.ncols() == rhs.nrows());
    unsafe { matmul_unchecked(dst, lhs, rhs, alpha, beta, n_threads, stack) }
}

pub mod triangular {
    use super::*;
    use crate::zip::{ColUninit, MatUninit};
    use crate::{join_req, temp_mat_req, temp_mat_uninit};

    #[repr(u8)]
    #[derive(Copy, Clone, Debug)]
    pub(crate) enum DiagonalKind {
        Zero,
        Unit,
        Generic,
    }

    unsafe fn copy_lower<T: Clone + Zero + One>(
        mut dst: MatMut<'_, T>,
        src: MatRef<'_, T>,
        src_diag: DiagonalKind,
    ) {
        let n = dst.nrows();
        fancy_debug_assert!(n == dst.nrows());
        fancy_debug_assert!(n == dst.ncols());
        fancy_debug_assert!(n == src.nrows());
        fancy_debug_assert!(n == src.ncols());

        let strict = match src_diag {
            DiagonalKind::Zero => {
                for j in 0..n {
                    *dst.rb_mut().ptr_in_bounds_at(j, j) = T::zero();
                }
                true
            }
            DiagonalKind::Unit => {
                for j in 0..n {
                    *dst.rb_mut().ptr_in_bounds_at(j, j) = T::one();
                }
                true
            }
            DiagonalKind::Generic => false,
        };

        MatUninit(dst.rb_mut())
            .cwise()
            .for_each_triangular_upper(true, |dst| {
                *dst = T::zero();
            });
        MatUninit(dst)
            .cwise()
            .zip_unchecked(src)
            .for_each_triangular_lower(strict, |dst, src| {
                *dst = src.clone();
            });
    }

    unsafe fn accum_lower<T: Clone + Zero>(
        dst: MatMut<'_, T>,
        src: MatRef<'_, T>,
        skip_diag: bool,
        alpha: Option<&T>,
    ) where
        for<'a> &'a T: Add<Output = T> + Mul<Output = T>,
    {
        let n = dst.nrows();
        fancy_debug_assert!(n == dst.nrows());
        fancy_debug_assert!(n == dst.ncols());
        fancy_debug_assert!(n == src.nrows());
        fancy_debug_assert!(n == src.ncols());

        match alpha {
            Some(alpha) => {
                dst.cwise()
                    .zip_unchecked(src)
                    .for_each_triangular_lower(skip_diag, |dst, src| {
                        *dst = &(alpha * &*dst) + &*src;
                    });
            }
            None => {
                MatUninit(dst)
                    .cwise()
                    .zip_unchecked(src)
                    .for_each_triangular_lower(skip_diag, |dst, src| {
                        *dst = src.clone();
                    });
            }
        }
    }

    #[inline]
    unsafe fn copy_upper<T: Clone + Zero + One>(
        dst: MatMut<'_, T>,
        src: MatRef<'_, T>,
        src_diag: DiagonalKind,
    ) {
        copy_lower(dst.transpose(), src.transpose(), src_diag)
    }

    #[inline]
    unsafe fn mul<T>(
        dst: MatMut<'_, T>,
        lhs: MatRef<'_, T>,
        rhs: MatRef<'_, T>,
        alpha: Option<&T>,
        beta: &T,
        n_threads: usize,
        transposed: bool,
        stack: DynStack<'_>,
    ) where
        T: Zero + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T>,
    {
        let (dst, lhs, rhs) = if !transposed {
            (dst, lhs, rhs)
        } else {
            (dst.transpose(), rhs.transpose(), lhs.transpose())
        };
        super::matmul_unchecked(dst, lhs, rhs, alpha, beta, n_threads, stack);
    }

    fn mat_x_lower_into_lower_impl_req<T: 'static>(
        n: usize,
        n_threads: usize,
    ) -> Result<StackReq, SizeOverflow> {
        let n_threads = if n * n * n <= 128 * 128 * 128 {
            1
        } else {
            n_threads
        };

        if n <= 16 {
            StackReq::try_all_of([
                temp_mat_req::<T>(n, n)?,
                temp_mat_req::<T>(n, n)?,
                super::matmul_req::<T>(n, n, n, n_threads)?,
            ])
        } else {
            let bs = n / 2;
            let rem = n - bs;

            let temp_dst = if n_threads <= 1 {
                StackReq::default()
            } else {
                temp_mat_req::<T>(rem, bs)?
            };

            temp_dst.try_and(join_req(
                |n_threads| {
                    StackReq::try_any_of([
                        super::matmul_req::<T>(rem, bs, rem, n_threads)?,
                        mat_x_lower_into_lower_impl_req::<T>(rem, n_threads)?,
                    ])
                },
                |n_threads| {
                    StackReq::try_any_of([
                        mat_x_lower_into_lower_impl_req::<T>(bs, n_threads)?,
                        join_req(
                            |n_threads| mat_x_mat_into_lower_impl_req::<T>(bs, rem, n_threads),
                            |n_threads| mat_x_lower_impl_req::<T>(rem, bs, n_threads),
                            split_half,
                            n_threads,
                        )?,
                    ])
                },
                split_half,
                n_threads,
            )?)
        }
    }

    unsafe fn mat_x_lower_into_lower_impl_unchecked<T>(
        dst: MatMut<'_, T>,
        skip_diag: bool,
        lhs: MatRef<'_, T>,
        rhs: MatRef<'_, T>,
        rhs_diag: DiagonalKind,
        alpha: Option<&T>,
        beta: &T,
        n_threads: usize,
        transposed: bool,
        mut stack: DynStack<'_>,
    ) where
        T: Zero + One + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T>,
    {
        let n = dst.nrows();
        fancy_debug_assert!(n == dst.nrows());
        fancy_debug_assert!(n == dst.ncols());
        fancy_debug_assert!(n == lhs.nrows());
        fancy_debug_assert!(n == lhs.ncols());
        fancy_debug_assert!(n == rhs.nrows());
        fancy_debug_assert!(n == rhs.ncols());

        let n_threads = if n * n * n <= 128 * 128 * 128 {
            1
        } else {
            n_threads
        };

        if n <= 16 {
            temp_mat_uninit! {
                let (mut temp_dst, stack) = unsafe { temp_mat_uninit::<T>(n, n, stack) };
                let (mut temp_rhs, stack) = unsafe { temp_mat_uninit::<T>(n, n, stack) };
            }
            copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);
            mul(
                temp_dst.rb_mut(),
                lhs,
                temp_rhs.into_const(),
                None,
                beta,
                n_threads,
                transposed,
                stack,
            );
            accum_lower(dst, temp_dst.into_const(), skip_diag, alpha);
        } else {
            let bs = n / 2;
            let rem = n - bs;

            let (mut dst_top_left, _, mut dst_bot_left, dst_bot_right) =
                dst.split_at_unchecked(bs, bs);
            let (lhs_top_left, lhs_top_right, lhs_bot_left, lhs_bot_right) =
                lhs.split_at_unchecked(bs, bs);
            let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_at_unchecked(bs, bs);

            // lhs_bot_right × rhs_bot_left  => dst_bot_left  | mat × mat => mat |   1
            // lhs_bot_right × rhs_bot_right => dst_bot_right | mat × low => low |   X
            //
            // lhs_top_left  × rhs_top_left  => dst_top_left  | mat × low => low |   X
            // lhs_top_right × rhs_bot_left  => dst_top_left  | mat × mat => low | 1/2
            // lhs_bot_left  × rhs_top_left  => dst_bot_left  | mat × low => mat | 1/2

            if n_threads <= 1 {
                mul(
                    dst_bot_left.rb_mut(),
                    lhs_bot_right,
                    rhs_bot_left,
                    alpha,
                    beta,
                    n_threads,
                    transposed,
                    stack.rb_mut(),
                );
                mat_x_lower_into_lower_impl_unchecked(
                    dst_bot_right,
                    skip_diag,
                    lhs_bot_right,
                    rhs_bot_right,
                    rhs_diag,
                    alpha,
                    beta,
                    n_threads,
                    transposed,
                    stack.rb_mut(),
                );

                mat_x_lower_into_lower_impl_unchecked(
                    dst_top_left.rb_mut(),
                    skip_diag,
                    lhs_top_left,
                    rhs_top_left,
                    rhs_diag,
                    alpha,
                    beta,
                    n_threads,
                    transposed,
                    stack.rb_mut(),
                );
                mat_x_mat_into_lower_impl_unchecked(
                    dst_top_left,
                    skip_diag,
                    lhs_top_right,
                    rhs_bot_left,
                    Some(&T::one()),
                    beta,
                    n_threads,
                    transposed,
                    stack.rb_mut(),
                );
                mat_x_lower_impl_unchecked(
                    dst_bot_left,
                    lhs_bot_left,
                    rhs_top_left,
                    rhs_diag,
                    Some(&T::one()),
                    beta,
                    n_threads,
                    transposed,
                    stack.rb_mut(),
                );
            } else {
                temp_mat_uninit! {
                    let (mut temp_dst_bot_left, stack) = unsafe { temp_mat_uninit::<T>(rem, bs, stack) };
                }
                join(
                    |n_threads, mut stack| {
                        mul(
                            dst_bot_left.rb_mut(),
                            lhs_bot_right,
                            rhs_bot_left,
                            alpha,
                            beta,
                            n_threads,
                            transposed,
                            stack.rb_mut(),
                        );
                        mat_x_lower_into_lower_impl_unchecked(
                            dst_bot_right,
                            skip_diag,
                            lhs_bot_right,
                            rhs_bot_right,
                            rhs_diag,
                            alpha,
                            beta,
                            n_threads,
                            transposed,
                            stack.rb_mut(),
                        );
                    },
                    |n_threads, mut stack| {
                        mat_x_lower_into_lower_impl_unchecked(
                            dst_top_left.rb_mut(),
                            skip_diag,
                            lhs_top_left,
                            rhs_top_left,
                            rhs_diag,
                            alpha,
                            beta,
                            n_threads,
                            transposed,
                            stack.rb_mut(),
                        );
                        join(
                            |n_threads, stack| {
                                mat_x_mat_into_lower_impl_unchecked(
                                    dst_top_left,
                                    skip_diag,
                                    lhs_top_right,
                                    rhs_bot_left,
                                    Some(&T::one()),
                                    beta,
                                    n_threads,
                                    transposed,
                                    stack,
                                )
                            },
                            |n_threads, stack| {
                                mat_x_lower_impl_unchecked(
                                    temp_dst_bot_left.rb_mut(),
                                    lhs_bot_left,
                                    rhs_top_left,
                                    rhs_diag,
                                    None,
                                    beta,
                                    n_threads,
                                    transposed,
                                    stack,
                                )
                            },
                            |n_threads| {
                                mat_x_mat_into_lower_impl_req::<T>(bs, rem, n_threads).unwrap()
                            },
                            split_half,
                            n_threads,
                            stack,
                        );
                    },
                    |n_threads| {
                        StackReq::any_of([
                            super::matmul_req::<T>(rem, bs, rem, n_threads).unwrap(),
                            mat_x_lower_into_lower_impl_req::<T>(rem, n_threads).unwrap(),
                        ])
                    },
                    split_half,
                    n_threads,
                    stack,
                );
                dst_bot_left
                    .cwise()
                    .zip(temp_dst_bot_left.into_const())
                    .for_each(|dst, temp| *dst = &*dst + temp);
            }
        }
    }

    fn mat_x_lower_impl_req<T: 'static>(
        m: usize,
        n: usize,
        n_threads: usize,
    ) -> Result<StackReq, SizeOverflow> {
        let n_threads = if n * n * m <= 128 * 128 * 128 {
            1
        } else {
            n_threads
        };

        if n <= 16 {
            StackReq::try_all_of([
                temp_mat_req::<T>(n, n)?,
                super::matmul_req::<T>(m, n, n, n_threads)?,
            ])
        } else {
            let bs = n / 2;
            let rem = n - bs;

            let temp_dst = if n_threads <= 1 {
                StackReq::empty()
            } else {
                temp_mat_req::<T>(m, bs)?
            };

            temp_dst.try_and(join_req(
                |n_threads| super::matmul_req::<T>(m, bs, rem, n_threads),
                |n_threads| {
                    join_req(
                        |n_threads| mat_x_lower_impl_req::<T>(m, bs, n_threads),
                        |n_threads| mat_x_lower_impl_req::<T>(m, rem, n_threads),
                        split_half,
                        n_threads,
                    )
                },
                split_half,
                n_threads,
            )?)
        }
    }

    unsafe fn mat_x_lower_impl_unchecked<T>(
        dst: MatMut<'_, T>,
        lhs: MatRef<'_, T>,
        rhs: MatRef<'_, T>,
        rhs_diag: DiagonalKind,
        alpha: Option<&T>,
        beta: &T,
        n_threads: usize,
        transposed: bool,
        mut stack: DynStack<'_>,
    ) where
        T: Zero + One + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T>,
    {
        let n = rhs.nrows();
        let m = lhs.nrows();
        fancy_debug_assert!(stack.can_hold(mat_x_lower_impl_req::<T>(m, n, n_threads).unwrap()));
        fancy_debug_assert!(m == lhs.nrows());
        fancy_debug_assert!(n == lhs.ncols());
        fancy_debug_assert!(n == rhs.nrows());
        fancy_debug_assert!(n == rhs.ncols());
        fancy_debug_assert!(m == dst.nrows());
        fancy_debug_assert!(n == dst.ncols());

        let n_threads = if n * n * m <= 128 * 128 * 128 {
            1
        } else {
            n_threads
        };

        if n <= 16 {
            temp_mat_uninit! {
                let (mut temp_rhs, stack) = unsafe { temp_mat_uninit::<T>(n, n, stack) };
            };

            copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);
            let temp_rhs = temp_rhs.into_const();

            mul(
                dst, lhs, temp_rhs, alpha, beta, n_threads, transposed, stack,
            );
        } else {
            // split rhs into 3 sections
            // split lhs and dst into 2 sections

            let bs = n / 2;
            let rem = n - bs;

            let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_at_unchecked(bs, bs);
            let (_, _, lhs_left, lhs_right) = lhs.split_at_unchecked(0, bs);
            let (_, _, mut dst_left, mut dst_right) = dst.split_at_unchecked(0, bs);

            if n_threads <= 1 {
                mat_x_lower_impl_unchecked(
                    dst_left.rb_mut(),
                    lhs_left,
                    rhs_top_left,
                    rhs_diag,
                    alpha,
                    beta,
                    n_threads,
                    transposed,
                    stack.rb_mut(),
                );
                mul(
                    dst_left,
                    lhs_right,
                    rhs_bot_left,
                    Some(&T::one()),
                    beta,
                    n_threads,
                    transposed,
                    stack.rb_mut(),
                );
                mat_x_lower_impl_unchecked(
                    dst_right.rb_mut(),
                    lhs_right,
                    rhs_bot_right,
                    rhs_diag,
                    alpha,
                    beta,
                    n_threads,
                    transposed,
                    stack,
                );
            } else {
                temp_mat_uninit! {
                    let (mut temp_dst, stack) = unsafe { temp_mat_uninit::<T>(m, bs, stack) };
                }

                join(
                    |n_threads, stack| {
                        mul(
                            temp_dst.rb_mut(),
                            lhs_right,
                            rhs_bot_left,
                            None,
                            beta,
                            n_threads,
                            transposed,
                            stack,
                        );
                    },
                    |n_threads, stack| {
                        join(
                            |n_threads, stack| {
                                mat_x_lower_impl_unchecked(
                                    dst_left.rb_mut(),
                                    lhs_left,
                                    rhs_top_left,
                                    rhs_diag,
                                    alpha,
                                    beta,
                                    n_threads,
                                    transposed,
                                    stack,
                                )
                            },
                            |n_threads, stack| {
                                mat_x_lower_impl_unchecked(
                                    dst_right.rb_mut(),
                                    lhs_right,
                                    rhs_bot_right,
                                    rhs_diag,
                                    alpha,
                                    beta,
                                    n_threads,
                                    transposed,
                                    stack,
                                )
                            },
                            |n_threads| mat_x_lower_impl_req::<T>(m, bs, n_threads).unwrap(),
                            split_half,
                            n_threads,
                            stack,
                        )
                    },
                    |n_threads| super::matmul_req::<T>(m, bs, rem, n_threads).unwrap(),
                    split_half,
                    n_threads,
                    stack,
                );

                dst_left
                    .cwise()
                    .zip(temp_dst.into_const())
                    .for_each(|dst, temp| *dst = &*dst + temp);
            }
        }
    }

    fn lower_x_lower_into_lower_impl_req<T: 'static>(
        n: usize,
        n_threads: usize,
    ) -> Result<StackReq, SizeOverflow> {
        let n_threads = if n * n * n <= 128 * 128 * 128 {
            1
        } else {
            n_threads
        };

        if n <= 16 {
            StackReq::try_all_of([
                temp_mat_req::<T>(n, n)?,
                temp_mat_req::<T>(n, n)?,
                temp_mat_req::<T>(n, n)?,
                super::matmul_req::<T>(n, n, n, n_threads)?,
            ])
        } else {
            let bs = n / 2;
            let rem = n - bs;
            let temp_dst = if n_threads <= 1 {
                StackReq::default()
            } else {
                temp_mat_req::<T>(n, n)?
            };

            temp_dst.try_and(join_req(
                |n_threads| {
                    StackReq::try_any_of([
                        lower_x_lower_into_lower_impl_req::<T>(bs, n_threads)?,
                        mat_x_lower_impl_req::<T>(rem, bs, n_threads)?,
                    ])
                },
                |n_threads| {
                    StackReq::try_any_of([
                        lower_x_lower_into_lower_impl_req::<T>(rem, n_threads)?,
                        mat_x_lower_impl_req::<T>(rem, bs, n_threads)?,
                    ])
                },
                split_half,
                n_threads,
            )?)
        }
    }

    unsafe fn lower_x_lower_into_lower_impl_unchecked<T>(
        dst: MatMut<'_, T>,
        skip_diag: bool,
        lhs: MatRef<'_, T>,
        lhs_diag: DiagonalKind,
        rhs: MatRef<'_, T>,
        rhs_diag: DiagonalKind,
        alpha: Option<&T>,
        beta: &T,
        n_threads: usize,
        transposed: bool,
        mut stack: DynStack<'_>,
    ) where
        T: Zero + One + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T>,
    {
        let n = dst.nrows();
        fancy_debug_assert!(n == lhs.nrows());
        fancy_debug_assert!(n == lhs.ncols());
        fancy_debug_assert!(n == rhs.nrows());
        fancy_debug_assert!(n == rhs.ncols());
        fancy_debug_assert!(n == dst.nrows());
        fancy_debug_assert!(n == dst.ncols());

        let n_threads = if n * n * n <= 128 * 128 * 128 {
            1
        } else {
            n_threads
        };

        if n <= 16 {
            temp_mat_uninit! {
                let (mut temp_dst, stack) = unsafe { temp_mat_uninit::<T>(n, n, stack) };
                let (mut temp_lhs, stack) = unsafe { temp_mat_uninit::<T>(n, n, stack) };
                let (mut temp_rhs, stack) = unsafe { temp_mat_uninit::<T>(n, n, stack) };
            };

            copy_lower(temp_lhs.rb_mut(), lhs, lhs_diag);
            copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);

            let temp_lhs = temp_lhs.into_const();
            let temp_rhs = temp_rhs.into_const();
            mul(
                temp_dst.rb_mut(),
                temp_lhs,
                temp_rhs,
                None,
                beta,
                n_threads,
                transposed,
                stack,
            );
            accum_lower(dst, temp_dst.into_const(), skip_diag, alpha);
        } else {
            let bs = n / 2;
            let rem = n - bs;

            let (dst_top_left, _, mut dst_bot_left, dst_bot_right) = dst.split_at_unchecked(bs, bs);
            let (lhs_top_left, _, lhs_bot_left, lhs_bot_right) = lhs.split_at_unchecked(bs, bs);
            let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_at_unchecked(bs, bs);

            // lhs_top_left  × rhs_top_left  => dst_top_left  | low × low => low |   X
            // lhs_bot_left  × rhs_top_left  => dst_bot_left  | mat × low => mat | 1/2
            // lhs_bot_right × rhs_bot_left  => dst_bot_left  | low × mat => mat | 1/2
            // lhs_bot_right × rhs_bot_right => dst_bot_right | low × low => low |   X

            if n_threads <= 1 {
                lower_x_lower_into_lower_impl_unchecked(
                    dst_top_left,
                    skip_diag,
                    lhs_top_left,
                    lhs_diag,
                    rhs_top_left,
                    rhs_diag,
                    alpha,
                    beta,
                    n_threads,
                    transposed,
                    stack.rb_mut(),
                );
                mat_x_lower_impl_unchecked(
                    dst_bot_left.rb_mut(),
                    lhs_bot_left,
                    rhs_top_left,
                    rhs_diag,
                    alpha,
                    beta,
                    n_threads,
                    transposed,
                    stack.rb_mut(),
                );
                mat_x_lower_impl_unchecked(
                    dst_bot_left.invert().transpose(),
                    rhs_bot_left.invert().transpose(),
                    lhs_bot_right.invert().transpose(),
                    lhs_diag,
                    Some(&T::one()),
                    beta,
                    n_threads,
                    !transposed,
                    stack.rb_mut(),
                );
                lower_x_lower_into_lower_impl_unchecked(
                    dst_bot_right,
                    skip_diag,
                    lhs_bot_right,
                    lhs_diag,
                    rhs_bot_right,
                    rhs_diag,
                    alpha,
                    beta,
                    n_threads,
                    transposed,
                    stack,
                )
            } else {
                temp_mat_uninit! {
                    let (mut temp_dst_bot_left, stack) = unsafe { temp_mat_uninit::<T>(rem, bs, stack) };
                }
                join(
                    |n_threads, mut stack| {
                        lower_x_lower_into_lower_impl_unchecked(
                            dst_top_left,
                            skip_diag,
                            lhs_top_left,
                            lhs_diag,
                            rhs_top_left,
                            rhs_diag,
                            alpha,
                            beta,
                            n_threads,
                            transposed,
                            stack.rb_mut(),
                        );
                        mat_x_lower_impl_unchecked(
                            dst_bot_left.rb_mut(),
                            lhs_bot_left,
                            rhs_top_left,
                            rhs_diag,
                            alpha,
                            beta,
                            n_threads,
                            transposed,
                            stack.rb_mut(),
                        );
                    },
                    |n_threads, mut stack| {
                        mat_x_lower_impl_unchecked(
                            temp_dst_bot_left.rb_mut().invert().transpose(),
                            rhs_bot_left.invert().transpose(),
                            lhs_bot_right.invert().transpose(),
                            lhs_diag,
                            None,
                            beta,
                            n_threads,
                            !transposed,
                            stack.rb_mut(),
                        );
                        lower_x_lower_into_lower_impl_unchecked(
                            dst_bot_right,
                            skip_diag,
                            lhs_bot_right,
                            lhs_diag,
                            rhs_bot_right,
                            rhs_diag,
                            alpha,
                            beta,
                            n_threads,
                            transposed,
                            stack,
                        )
                    },
                    |n_threads| {
                        StackReq::any_of([
                            lower_x_lower_into_lower_impl_req::<T>(bs, n_threads).unwrap(),
                            mat_x_lower_impl_req::<T>(rem, bs, n_threads).unwrap(),
                        ])
                    },
                    split_half,
                    n_threads,
                    stack,
                );

                dst_bot_left
                    .cwise()
                    .zip(temp_dst_bot_left.into_const())
                    .for_each(|dst, temp| *dst = &*dst + temp);
            }
        }
    }

    fn upper_x_lower_impl_req<T: 'static>(
        n: usize,
        n_threads: usize,
    ) -> Result<StackReq, SizeOverflow> {
        let n_threads = if n * n * n <= 128 * 128 * 128 {
            1
        } else {
            n_threads
        };

        if n <= 16 {
            StackReq::try_all_of([
                temp_mat_req::<T>(n, n)?,
                temp_mat_req::<T>(n, n)?,
                super::matmul_req::<T>(n, n, n, n_threads)?,
            ])
        } else {
            let bs = n / 2;
            let rem = n - bs;

            join_req(
                |n_threads| {
                    StackReq::try_any_of([
                        super::matmul_req::<T>(bs, bs, rem, n_threads)?,
                        upper_x_lower_impl_req::<T>(bs, n_threads)?,
                    ])
                },
                |n_threads| {
                    StackReq::try_any_of([
                        join_req(
                            |n_threads| mat_x_lower_impl_req::<T>(bs, rem, n_threads),
                            |n_threads| mat_x_lower_impl_req::<T>(rem, bs, n_threads),
                            split_half,
                            n_threads,
                        )?,
                        upper_x_lower_impl_req::<T>(rem, n_threads)?,
                    ])
                },
                split_half,
                n_threads,
            )
        }
    }

    unsafe fn upper_x_lower_impl_unchecked<T>(
        dst: MatMut<'_, T>,
        lhs: MatRef<'_, T>,
        lhs_diag: DiagonalKind,
        rhs: MatRef<'_, T>,
        rhs_diag: DiagonalKind,
        alpha: Option<&T>,
        beta: &T,
        n_threads: usize,
        transposed: bool,
        stack: DynStack<'_>,
    ) where
        T: Zero + One + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T>,
    {
        let n = dst.nrows();
        fancy_debug_assert!(n == lhs.nrows());
        fancy_debug_assert!(n == lhs.ncols());
        fancy_debug_assert!(n == rhs.nrows());
        fancy_debug_assert!(n == rhs.ncols());
        fancy_debug_assert!(n == dst.nrows());
        fancy_debug_assert!(n == dst.ncols());

        let n_threads = if n * n * n <= 128 * 128 * 128 {
            1
        } else {
            n_threads
        };

        if n <= 16 {
            temp_mat_uninit! {
                let (mut temp_lhs, stack) = unsafe { temp_mat_uninit::<T>(n, n, stack) };
                let (mut temp_rhs, stack) = unsafe { temp_mat_uninit::<T>(n, n, stack) };
            };

            copy_upper(temp_lhs.rb_mut(), lhs, lhs_diag);
            copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);

            let temp_lhs = temp_lhs.into_const();
            let temp_rhs = temp_rhs.into_const();
            mul(
                dst, temp_lhs, temp_rhs, alpha, beta, n_threads, transposed, stack,
            );
        } else {
            let bs = n / 2;
            let rem = n - bs;

            let (mut dst_top_left, dst_top_right, dst_bot_left, dst_bot_right) =
                dst.split_at_unchecked(bs, bs);
            let (lhs_top_left, lhs_top_right, _, lhs_bot_right) = lhs.split_at_unchecked(bs, bs);
            let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_at_unchecked(bs, bs);

            // lhs_top_right × rhs_bot_left  => dst_top_left  | mat × mat => mat |   1
            // lhs_top_left  × rhs_top_left  => dst_top_left  | upp × low => mat |   X
            //
            // lhs_top_right × rhs_bot_right => dst_top_right | mat × low => mat | 1/2
            // lhs_bot_right × rhs_bot_left  => dst_bot_left  | upp × mat => mat | 1/2
            // lhs_bot_right × rhs_bot_right => dst_bot_right | upp × low => mat |   X

            join(
                |n_threads, mut stack| {
                    mul(
                        dst_top_left.rb_mut(),
                        lhs_top_right,
                        rhs_bot_left,
                        alpha,
                        beta,
                        n_threads,
                        transposed,
                        stack.rb_mut(),
                    );
                    upper_x_lower_impl_unchecked(
                        dst_top_left,
                        lhs_top_left,
                        lhs_diag,
                        rhs_top_left,
                        rhs_diag,
                        Some(&T::one()),
                        beta,
                        n_threads,
                        transposed,
                        stack.rb_mut(),
                    )
                },
                |n_threads, mut stack| {
                    join(
                        |n_threads, stack| {
                            mat_x_lower_impl_unchecked(
                                dst_top_right,
                                lhs_top_right,
                                rhs_bot_right,
                                rhs_diag,
                                alpha,
                                beta,
                                n_threads,
                                transposed,
                                stack,
                            )
                        },
                        |n_threads, stack| {
                            mat_x_lower_impl_unchecked(
                                dst_bot_left.transpose(),
                                rhs_bot_left.transpose(),
                                lhs_bot_right.transpose(),
                                lhs_diag,
                                alpha,
                                beta,
                                n_threads,
                                !transposed,
                                stack,
                            )
                        },
                        |n_threads| mat_x_lower_impl_req::<T>(bs, rem, n_threads).unwrap(),
                        split_half,
                        n_threads,
                        stack.rb_mut(),
                    );

                    upper_x_lower_impl_unchecked(
                        dst_bot_right,
                        lhs_bot_right,
                        lhs_diag,
                        rhs_bot_right,
                        rhs_diag,
                        alpha,
                        beta,
                        n_threads,
                        transposed,
                        stack,
                    )
                },
                |n_threads| {
                    StackReq::any_of([
                        super::matmul_req::<T>(bs, bs, rem, n_threads).unwrap(),
                        upper_x_lower_impl_req::<T>(bs, n_threads).unwrap(),
                    ])
                },
                split_half,
                n_threads,
                stack,
            );
        }
    }

    fn upper_x_lower_into_lower_impl_req<T: 'static>(
        n: usize,
        n_threads: usize,
    ) -> Result<StackReq, SizeOverflow> {
        let n_threads = if n * n * n <= 128 * 128 * 128 {
            1
        } else {
            n_threads
        };

        if n <= 16 {
            StackReq::try_all_of([
                temp_mat_req::<T>(n, n)?,
                temp_mat_req::<T>(n, n)?,
                temp_mat_req::<T>(n, n)?,
                super::matmul_req::<T>(n, n, n, n_threads)?,
            ])
        } else {
            let bs = n / 2;
            let rem = n - bs;

            join_req(
                |n_threads| {
                    StackReq::try_any_of([
                        mat_x_mat_into_lower_impl_req::<T>(bs, rem, n_threads)?,
                        upper_x_lower_into_lower_impl_req::<T>(bs, n_threads)?,
                    ])
                },
                |n_threads| {
                    StackReq::try_any_of([
                        mat_x_lower_impl_req::<T>(rem, bs, n_threads)?,
                        upper_x_lower_into_lower_impl_req::<T>(rem, n_threads)?,
                    ])
                },
                split_half,
                n_threads,
            )
        }
    }

    unsafe fn upper_x_lower_into_lower_impl_unchecked<T>(
        mut dst: MatMut<'_, T>,
        skip_diag: bool,
        lhs: MatRef<'_, T>,
        lhs_diag: DiagonalKind,
        rhs: MatRef<'_, T>,
        rhs_diag: DiagonalKind,
        alpha: Option<&T>,
        beta: &T,
        n_threads: usize,
        transposed: bool,
        stack: DynStack<'_>,
    ) where
        T: Zero + One + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T>,
    {
        let n = dst.nrows();
        fancy_debug_assert!(n == lhs.nrows());
        fancy_debug_assert!(n == lhs.ncols());
        fancy_debug_assert!(n == rhs.nrows());
        fancy_debug_assert!(n == rhs.ncols());
        fancy_debug_assert!(n == dst.nrows());
        fancy_debug_assert!(n == dst.ncols());

        let n_threads = if n * n * n <= 128 * 128 * 128 {
            1
        } else {
            n_threads
        };

        if n <= 16 {
            temp_mat_uninit! {
                let (mut temp_dst, stack) = unsafe { temp_mat_uninit::<T>(n, n, stack) };
                let (mut temp_lhs, stack) = unsafe { temp_mat_uninit::<T>(n, n, stack) };
                let (mut temp_rhs, stack) = unsafe { temp_mat_uninit::<T>(n, n, stack) };
            };

            copy_upper(temp_lhs.rb_mut(), lhs, lhs_diag);
            copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);

            let temp_lhs = temp_lhs.into_const();
            let temp_rhs = temp_rhs.into_const();
            mul(
                temp_dst.rb_mut(),
                temp_lhs,
                temp_rhs,
                None,
                beta,
                n_threads,
                transposed,
                stack,
            );

            accum_lower(dst.rb_mut(), temp_dst.into_const(), skip_diag, alpha);
        } else {
            let bs = n / 2;
            let rem = n - bs;

            let (mut dst_top_left, _, dst_bot_left, dst_bot_right) = dst.split_at_unchecked(bs, bs);
            let (lhs_top_left, lhs_top_right, _, lhs_bot_right) = lhs.split_at_unchecked(bs, bs);
            let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_at_unchecked(bs, bs);

            // lhs_top_left  × rhs_top_left  => dst_top_left  | upp × low => low |   X
            // lhs_top_right × rhs_bot_left  => dst_top_left  | mat × mat => low | 1/2
            //
            // lhs_bot_right × rhs_bot_left  => dst_bot_left  | upp × mat => mat | 1/2
            // lhs_bot_right × rhs_bot_right => dst_bot_right | upp × low => low |   X

            join(
                |n_threads, mut stack| {
                    mat_x_mat_into_lower_impl_unchecked(
                        dst_top_left.rb_mut(),
                        skip_diag,
                        lhs_top_right,
                        rhs_bot_left,
                        alpha,
                        beta,
                        n_threads,
                        transposed,
                        stack.rb_mut(),
                    );
                    upper_x_lower_into_lower_impl_unchecked(
                        dst_top_left,
                        skip_diag,
                        lhs_top_left,
                        lhs_diag,
                        rhs_top_left,
                        rhs_diag,
                        Some(&T::one()),
                        beta,
                        n_threads,
                        transposed,
                        stack.rb_mut(),
                    )
                },
                |n_threads, mut stack| {
                    mat_x_lower_impl_unchecked(
                        dst_bot_left.transpose(),
                        rhs_bot_left.transpose(),
                        lhs_bot_right.transpose(),
                        lhs_diag,
                        alpha,
                        beta,
                        n_threads,
                        !transposed,
                        stack.rb_mut(),
                    );
                    upper_x_lower_into_lower_impl_unchecked(
                        dst_bot_right,
                        skip_diag,
                        lhs_bot_right,
                        lhs_diag,
                        rhs_bot_right,
                        rhs_diag,
                        alpha,
                        beta,
                        n_threads,
                        transposed,
                        stack,
                    )
                },
                |n_threads| {
                    StackReq::any_of([
                        mat_x_mat_into_lower_impl_req::<T>(bs, rem, n_threads).unwrap(),
                        upper_x_lower_into_lower_impl_req::<T>(bs, n_threads).unwrap(),
                    ])
                },
                split_half,
                n_threads,
                stack,
            );
        }
    }

    fn mat_x_mat_into_lower_impl_req<T: 'static>(
        n: usize,
        k: usize,
        n_threads: usize,
    ) -> Result<StackReq, SizeOverflow> {
        let n_threads = if n * n * k <= 128 * 128 * 128 {
            1
        } else {
            n_threads
        };

        if n <= 16 {
            StackReq::try_all_of([
                temp_mat_req::<T>(n, n)?,
                super::matmul_req::<T>(n, n, k, n_threads)?,
            ])
        } else {
            let bs = n / 2;
            let rem = n - bs;
            join_req(
                |n_threads| super::matmul_req::<T>(rem, bs, k, n_threads),
                |n_threads| {
                    join_req(
                        |n_threads| mat_x_mat_into_lower_impl_req::<T>(bs, k, n_threads),
                        |n_threads| mat_x_mat_into_lower_impl_req::<T>(rem, k, n_threads),
                        split_half,
                        n_threads,
                    )
                },
                split_half,
                n_threads,
            )
        }
    }

    unsafe fn mat_x_mat_into_lower_impl_unchecked<T>(
        dst: MatMut<'_, T>,
        skip_diag: bool,
        lhs: MatRef<'_, T>,
        rhs: MatRef<'_, T>,
        alpha: Option<&T>,
        beta: &T,
        n_threads: usize,
        transposed: bool,
        stack: DynStack<'_>,
    ) where
        T: Zero + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T>,
    {
        fancy_debug_assert!(dst.nrows() == dst.ncols());
        fancy_debug_assert!(dst.nrows() == lhs.nrows());
        fancy_debug_assert!(dst.ncols() == rhs.ncols());
        fancy_debug_assert!(lhs.ncols() == rhs.nrows());

        let n = dst.nrows();
        let k = lhs.ncols();

        let n_threads = if n * n * k <= 128 * 128 * 128 {
            1
        } else {
            n_threads
        };

        if n <= 16 {
            temp_mat_uninit! {
                let (mut temp_dst, stack) = unsafe { temp_mat_uninit::<T>(n, n, stack) };
            };
            mul(
                temp_dst.rb_mut(),
                lhs,
                rhs,
                None,
                beta,
                n_threads,
                transposed,
                stack,
            );
            accum_lower(dst, temp_dst.rb(), skip_diag, alpha)
        } else {
            let bs = n / 2;
            let rem = n - n / 2;
            let (dst_top_left, _, dst_bot_left, dst_bot_right) = dst.split_at_unchecked(bs, bs);
            let (_, lhs_top, _, lhs_bot) = lhs.split_at_unchecked(bs, 0);
            let (_, _, rhs_left, rhs_right) = rhs.split_at_unchecked(0, bs);

            join(
                |n_threads, stack| {
                    mul(
                        dst_bot_left,
                        lhs_bot,
                        rhs_left,
                        alpha,
                        beta,
                        n_threads,
                        transposed,
                        stack,
                    )
                },
                |n_threads, stack| {
                    join(
                        |n_threads, stack| {
                            mat_x_mat_into_lower_impl_unchecked(
                                dst_top_left,
                                skip_diag,
                                lhs_top,
                                rhs_left,
                                alpha,
                                beta,
                                n_threads,
                                transposed,
                                stack,
                            )
                        },
                        |n_threads, stack| {
                            mat_x_mat_into_lower_impl_unchecked(
                                dst_bot_right,
                                skip_diag,
                                lhs_bot,
                                rhs_right,
                                alpha,
                                beta,
                                n_threads,
                                transposed,
                                stack,
                            )
                        },
                        |n_threads| mat_x_mat_into_lower_impl_req::<T>(bs, k, n_threads).unwrap(),
                        split_half,
                        n_threads,
                        stack,
                    )
                },
                |n_threads| super::matmul_req::<T>(rem, bs, k, n_threads).unwrap(),
                split_half,
                n_threads,
                stack,
            );
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub enum BlockStructure {
        Rectangular,
        TriangularLower,
        StrictTriangularLower,
        UnitTriangularLower,
        TriangularUpper,
        StrictTriangularUpper,
        UnitTriangularUpper,
    }

    impl BlockStructure {
        #[inline]
        pub fn is_dense(self) -> bool {
            matches!(self, BlockStructure::Rectangular)
        }

        #[inline]
        pub fn is_lower(self) -> bool {
            use BlockStructure::*;
            matches!(
                self,
                TriangularLower | StrictTriangularLower | UnitTriangularLower
            )
        }

        #[inline]
        pub fn is_upper(self) -> bool {
            use BlockStructure::*;
            matches!(
                self,
                TriangularUpper | StrictTriangularUpper | UnitTriangularUpper
            )
        }

        #[inline]
        pub fn transpose(self) -> Self {
            use BlockStructure::*;
            match self {
                Rectangular => Rectangular,
                TriangularLower => TriangularUpper,
                StrictTriangularLower => StrictTriangularUpper,
                UnitTriangularLower => UnitTriangularUpper,
                TriangularUpper => TriangularLower,
                StrictTriangularUpper => StrictTriangularLower,
                UnitTriangularUpper => UnitTriangularLower,
            }
        }

        #[inline]
        pub(crate) fn diag_kind(self) -> DiagonalKind {
            use BlockStructure::*;
            match self {
                Rectangular | TriangularLower | TriangularUpper => DiagonalKind::Generic,
                StrictTriangularLower | StrictTriangularUpper => DiagonalKind::Zero,
                UnitTriangularLower | UnitTriangularUpper => DiagonalKind::Unit,
            }
        }
    }

    /// Computes the memory requirements of [`matmul`].
    #[inline]
    pub fn matmul_req<T: 'static>(
        dst_structure: BlockStructure,
        lhs_structure: BlockStructure,
        rhs_structure: BlockStructure,
        dst_rows: usize,
        dst_cols: usize,
        lhs_cols: usize,
        n_threads: usize,
    ) -> Result<StackReq, SizeOverflow> {
        let mut dst_structure = dst_structure;
        let mut lhs_structure = lhs_structure;
        let mut rhs_structure = rhs_structure;

        let mut dst_rows = dst_rows;
        let mut dst_cols = dst_cols;

        // if either the lhs or the rhs is triangular
        if rhs_structure.is_lower() {
            // do nothing
        } else if rhs_structure.is_upper() {
            // invert dst, lhs and rhs
            dst_structure = dst_structure.transpose();
            lhs_structure = lhs_structure.transpose();
            rhs_structure = rhs_structure.transpose();
        } else if lhs_structure.is_lower() {
            // invert and transpose
            (lhs_structure, rhs_structure) = (rhs_structure, lhs_structure);
            core::mem::swap(&mut dst_rows, &mut dst_cols);
        } else if lhs_structure.is_upper() {
            // transpose
            dst_structure = dst_structure.transpose();
            (lhs_structure, rhs_structure) = (rhs_structure.transpose(), lhs_structure.transpose());
            core::mem::swap(&mut dst_rows, &mut dst_cols);
        } else {
            // do nothing
        }

        let m = dst_rows;
        let n = dst_cols;
        let k = lhs_cols;

        if dst_structure.is_dense() {
            if lhs_structure.is_dense() && rhs_structure.is_dense() {
                super::matmul_req::<T>(m, n, k, n_threads)
            } else {
                fancy_debug_assert!(rhs_structure.is_lower());
                if lhs_structure.is_dense() {
                    mat_x_lower_impl_req::<T>(m, n, n_threads)
                } else if lhs_structure.is_lower() {
                    lower_x_lower_into_lower_impl_req::<T>(n, n_threads)
                } else {
                    fancy_debug_assert!(lhs_structure.is_upper());
                    upper_x_lower_impl_req::<T>(n, n_threads)
                }
            }
        } else if dst_structure.is_lower() {
            if lhs_structure.is_dense() && rhs_structure.is_dense() {
                mat_x_mat_into_lower_impl_req::<T>(n, k, n_threads)
            } else {
                fancy_debug_assert!(rhs_structure.is_lower());
                if lhs_structure.is_dense() {
                    mat_x_lower_into_lower_impl_req::<T>(n, n_threads)
                } else if lhs_structure.is_lower() {
                    lower_x_lower_into_lower_impl_req::<T>(n, n_threads)
                } else {
                    upper_x_lower_into_lower_impl_req::<T>(n, n_threads)
                }
            }
        } else if lhs_structure.is_dense() && rhs_structure.is_dense() {
            mat_x_mat_into_lower_impl_req::<T>(m, k, n_threads)
        } else {
            fancy_debug_assert!(rhs_structure.is_lower());
            if lhs_structure.is_dense() {
                // lower part of lhs does not contribute to result
                upper_x_lower_into_lower_impl_req::<T>(m, n_threads)
            } else if lhs_structure.is_lower() {
                Ok(StackReq::default())
            } else {
                fancy_debug_assert!(lhs_structure.is_upper());
                upper_x_lower_into_lower_impl_req::<T>(m, n_threads)
            }
        }
    }

    /// Computes the matrix product `[alpha * dst] + beta * lhs * rhs` and stores the result in
    /// `dst`.
    ///
    /// The left hand side and right hand side may be interpreted as triangular depending on the
    /// given corresponding matrix structure.  
    ///
    /// For the destination matrix, the result is:
    /// - fully computed if the structure is rectangular,
    /// - only the triangular half (including the diagonal) is computed if the structure is
    /// triangular,
    /// - only the strict triangular half (excluding the diagonal) is computed if the structure is
    /// strictly triangular or unit triangular.
    ///
    /// If `alpha` is not provided, he preexisting values in `dst` are not read so it is allowed to
    /// be a view over uninitialized values if `T: Copy`.
    ///
    /// # Panics
    ///
    /// Panics if the matrix dimensions are not compatible for matrix multiplication.  
    /// i.e.  
    ///  - `dst.nrows() == lhs.nrows()`
    ///  - `dst.ncols() == rhs.ncols()`
    ///  - `lhs.ncols() == rhs.nrows()`
    ///
    ///  Additionally, matrices that are marked as triangular must be square, i.e., they must have
    ///  the same number of rows and columns.
    #[track_caller]
    #[inline]
    pub fn matmul<T>(
        dst: MatMut<'_, T>,
        dst_structure: BlockStructure,
        lhs: MatRef<'_, T>,
        lhs_structure: BlockStructure,
        rhs: MatRef<'_, T>,
        rhs_structure: BlockStructure,
        alpha: Option<&T>,
        beta: &T,
        n_threads: usize,
        stack: DynStack<'_>,
    ) where
        T: Zero + One + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T>,
    {
        fancy_assert!(dst.nrows() == lhs.nrows());
        fancy_assert!(dst.ncols() == rhs.ncols());
        fancy_assert!(lhs.ncols() == rhs.nrows());

        if !dst_structure.is_dense() {
            fancy_assert!(dst.nrows() == dst.ncols());
        }
        if !lhs_structure.is_dense() {
            fancy_assert!(lhs.nrows() == lhs.ncols());
        }
        if !rhs_structure.is_dense() {
            fancy_assert!(rhs.nrows() == rhs.ncols());
        }

        unsafe {
            matmul_unchecked(
                dst,
                dst_structure,
                lhs,
                lhs_structure,
                rhs,
                rhs_structure,
                alpha,
                beta,
                n_threads,
                stack,
            )
        }
    }

    /// Same as [`matmul`], except that panics become undefined behavior.
    #[inline]
    pub unsafe fn matmul_unchecked<T>(
        dst: MatMut<'_, T>,
        dst_structure: BlockStructure,
        lhs: MatRef<'_, T>,
        lhs_structure: BlockStructure,
        rhs: MatRef<'_, T>,
        rhs_structure: BlockStructure,
        alpha: Option<&T>,
        beta: &T,
        n_threads: usize,
        stack: DynStack<'_>,
    ) where
        T: Zero + One + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T>,
    {
        fancy_debug_assert!(dst.nrows() == lhs.nrows());
        fancy_debug_assert!(dst.ncols() == rhs.ncols());
        fancy_debug_assert!(lhs.ncols() == rhs.nrows());

        if !dst_structure.is_dense() {
            fancy_debug_assert!(dst.nrows() == dst.ncols());
        }
        if !lhs_structure.is_dense() {
            fancy_debug_assert!(lhs.nrows() == lhs.ncols());
        }
        if !rhs_structure.is_dense() {
            fancy_debug_assert!(rhs.nrows() == rhs.ncols());
        }

        let mut dst = dst;
        let mut lhs = lhs;
        let mut rhs = rhs;

        let mut dst_structure = dst_structure;
        let mut lhs_structure = lhs_structure;
        let mut rhs_structure = rhs_structure;

        // if either the lhs or the rhs is triangular
        let transposed = if rhs_structure.is_lower() {
            // do nothing
            false
        } else if rhs_structure.is_upper() {
            // invert dst, lhs and rhs
            dst = dst.invert();
            lhs = lhs.invert();
            rhs = rhs.invert();
            dst_structure = dst_structure.transpose();
            lhs_structure = lhs_structure.transpose();
            rhs_structure = rhs_structure.transpose();
            false
        } else if lhs_structure.is_lower() {
            // invert and transpose
            dst = dst.invert().transpose();
            (lhs, rhs) = (rhs.invert().transpose(), lhs.invert().transpose());
            (lhs_structure, rhs_structure) = (rhs_structure, lhs_structure);
            true
        } else if lhs_structure.is_upper() {
            // transpose
            dst_structure = dst_structure.transpose();
            dst = dst.transpose();
            (lhs, rhs) = (rhs.transpose(), lhs.transpose());
            (lhs_structure, rhs_structure) = (rhs_structure.transpose(), lhs_structure.transpose());
            true
        } else {
            // do nothing
            false
        };

        let clear_upper = |dst: MatMut<'_, T>, skip_diag: bool| match alpha {
            Some(alpha) => dst
                .cwise()
                .for_each_triangular_upper(skip_diag, |dst| *dst = alpha * &*dst),

            None => MatUninit(dst)
                .cwise()
                .for_each_triangular_upper(skip_diag, |dst| *dst = T::zero()),
        };

        let skip_diag = matches!(
            dst_structure,
            BlockStructure::StrictTriangularLower
                | BlockStructure::StrictTriangularUpper
                | BlockStructure::UnitTriangularLower
                | BlockStructure::UnitTriangularUpper
        );
        let lhs_diag = lhs_structure.diag_kind();
        let rhs_diag = rhs_structure.diag_kind();

        if dst_structure.is_dense() {
            if lhs_structure.is_dense() && rhs_structure.is_dense() {
                mul(dst, lhs, rhs, alpha, beta, n_threads, transposed, stack);
            } else {
                fancy_debug_assert!(rhs_structure.is_lower());

                if lhs_structure.is_dense() {
                    mat_x_lower_impl_unchecked(
                        dst, lhs, rhs, rhs_diag, alpha, beta, n_threads, transposed, stack,
                    )
                } else if lhs_structure.is_lower() {
                    clear_upper(dst.rb_mut(), true);
                    lower_x_lower_into_lower_impl_unchecked(
                        dst, false, lhs, lhs_diag, rhs, rhs_diag, alpha, beta, n_threads,
                        transposed, stack,
                    );
                } else {
                    fancy_debug_assert!(lhs_structure.is_upper());
                    upper_x_lower_impl_unchecked(
                        dst, lhs, lhs_diag, rhs, rhs_diag, alpha, beta, n_threads, transposed,
                        stack,
                    )
                }
            }
        } else if dst_structure.is_lower() {
            if lhs_structure.is_dense() && rhs_structure.is_dense() {
                mat_x_mat_into_lower_impl_unchecked(
                    dst, skip_diag, lhs, rhs, alpha, beta, n_threads, transposed, stack,
                )
            } else {
                fancy_debug_assert!(rhs_structure.is_lower());
                if lhs_structure.is_dense() {
                    mat_x_lower_into_lower_impl_unchecked(
                        dst, skip_diag, lhs, rhs, rhs_diag, alpha, beta, n_threads, transposed,
                        stack,
                    );
                } else if lhs_structure.is_lower() {
                    lower_x_lower_into_lower_impl_unchecked(
                        dst, skip_diag, lhs, lhs_diag, rhs, rhs_diag, alpha, beta, n_threads,
                        transposed, stack,
                    )
                } else {
                    upper_x_lower_into_lower_impl_unchecked(
                        dst, skip_diag, lhs, lhs_diag, rhs, rhs_diag, alpha, beta, n_threads,
                        transposed, stack,
                    )
                }
            }
        } else if lhs_structure.is_dense() && rhs_structure.is_dense() {
            mat_x_mat_into_lower_impl_unchecked(
                dst.transpose(),
                skip_diag,
                rhs.transpose(),
                lhs.transpose(),
                alpha,
                beta,
                n_threads,
                !transposed,
                stack,
            )
        } else {
            fancy_debug_assert!(rhs_structure.is_lower());
            if lhs_structure.is_dense() {
                // lower part of lhs does not contribute to result
                upper_x_lower_into_lower_impl_unchecked(
                    dst.transpose(),
                    skip_diag,
                    rhs.transpose(),
                    rhs_diag,
                    lhs.transpose(),
                    lhs_diag,
                    alpha,
                    beta,
                    n_threads,
                    transposed,
                    stack,
                )
            } else if lhs_structure.is_lower() {
                if !skip_diag {
                    match alpha {
                        Some(alpha) => {
                            dst.rb_mut()
                                .diagonal_unchecked()
                                .cwise()
                                .zip(lhs.diagonal_unchecked())
                                .zip(rhs.diagonal_unchecked())
                                .for_each(|dst, lhs, rhs| {
                                    *dst = alpha * (&*dst) + beta * &(&*lhs * &*rhs)
                                });
                        }
                        None => {
                            ColUninit(dst.rb_mut().diagonal_unchecked())
                                .cwise()
                                .zip(lhs.diagonal_unchecked())
                                .zip(rhs.diagonal_unchecked())
                                .for_each(|dst, lhs, rhs| *dst = beta * &(&*lhs * &*rhs));
                        }
                    }
                }
                clear_upper(dst.rb_mut(), true);
            } else {
                fancy_debug_assert!(lhs_structure.is_upper());
                upper_x_lower_into_lower_impl_unchecked(
                    dst.transpose(),
                    skip_diag,
                    rhs.transpose(),
                    rhs_diag,
                    lhs.transpose(),
                    lhs_diag,
                    alpha,
                    beta,
                    n_threads,
                    !transposed,
                    stack,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use dyn_stack::GlobalMemBuffer;
    use rand::random;

    use super::triangular::{BlockStructure, DiagonalKind};
    use super::*;
    use crate::{mat, Mat};

    #[test]
    fn rectangular() {
        let lhs = mat![[1.0, 2.0], [3.0, 4.0]];
        let rhs = mat![[5.0], [7.0]];
        let mut dst = mat![[0.0], [0.0]];

        let mut mem = dyn_stack::GlobalMemBuffer::new(matmul_req::<f64>(2, 1, 2, 4).unwrap());
        let stack = DynStack::new(&mut mem);
        super::matmul(
            dst.as_mut(),
            lhs.as_ref(),
            rhs.as_ref(),
            None,
            &2.0,
            1,
            stack,
        );

        fancy_assert!(dst[(0, 0)] == 38.0);
        fancy_assert!(dst[(1, 0)] == 86.0);
    }

    #[test]
    fn rectangular_accum() {
        let lhs = mat![[1.0, 2.0], [3.0, 4.0]];
        let rhs = mat![[5.0], [7.0]];
        let mut dst = mat![[4.0], [3.0]];

        let mut mem = dyn_stack::GlobalMemBuffer::new(matmul_req::<f64>(2, 1, 2, 4).unwrap());
        let stack = DynStack::new(&mut mem);
        super::matmul(
            dst.as_mut(),
            lhs.as_ref(),
            rhs.as_ref(),
            Some(&-2.0),
            &2.0,
            1,
            stack,
        );

        fancy_assert!(dst[(0, 0)] == 30.0);
        fancy_assert!(dst[(1, 0)] == 80.0);
    }

    fn generate_structured_matrix(
        is_dst: bool,
        nrows: usize,
        ncols: usize,
        structure: BlockStructure,
    ) -> Mat<f64> {
        let mut mat = Mat::new();
        mat.resize_with(|_, _| random(), nrows, ncols);

        if !is_dst {
            let kind = structure.diag_kind();
            if structure.is_lower() {
                for j in 0..ncols {
                    for i in 0..j {
                        mat[(i, j)] = 0.0;
                    }
                }
            } else if structure.is_upper() {
                for j in 0..ncols {
                    for i in j + 1..nrows {
                        mat[(i, j)] = 0.0;
                    }
                }
            }

            match kind {
                triangular::DiagonalKind::Zero => {
                    for i in 0..nrows {
                        mat[(i, i)] = 0.0;
                    }
                }
                triangular::DiagonalKind::Unit => {
                    for i in 0..nrows {
                        mat[(i, i)] = 1.0;
                    }
                }
                triangular::DiagonalKind::Generic => (),
            }
        }
        mat
    }

    fn run_test_problem(
        m: usize,
        n: usize,
        k: usize,
        dst_structure: BlockStructure,
        lhs_structure: BlockStructure,
        rhs_structure: BlockStructure,
    ) {
        let mut dst = generate_structured_matrix(true, m, n, dst_structure);
        let mut dst_target = dst.clone();
        let dst_orig = dst.clone();
        let lhs = generate_structured_matrix(false, m, k, lhs_structure);
        let rhs = generate_structured_matrix(false, k, n, rhs_structure);

        for n_threads in [1, 12] {
            triangular::matmul(
                dst.as_mut(),
                dst_structure,
                lhs.as_ref(),
                lhs_structure,
                rhs.as_ref(),
                rhs_structure,
                None,
                &2.5,
                n_threads,
                DynStack::new(&mut GlobalMemBuffer::new(
                    triangular::matmul_req::<f64>(
                        dst_structure,
                        lhs_structure,
                        rhs_structure,
                        m,
                        n,
                        k,
                        n_threads,
                    )
                    .unwrap(),
                )),
            );

            matmul(
                dst_target.as_mut(),
                lhs.as_ref(),
                rhs.as_ref(),
                None,
                &2.5,
                n_threads,
                DynStack::new(&mut GlobalMemBuffer::new(
                    matmul_req::<f64>(m, n, k, n_threads).unwrap(),
                )),
            );

            if dst_structure.is_dense() {
                for j in 0..n {
                    for i in 0..m {
                        assert_approx_eq!(dst[(i, j)], dst_target[(i, j)]);
                    }
                }
            } else if dst_structure.is_lower() {
                for j in 0..n {
                    if matches!(dst_structure.diag_kind(), DiagonalKind::Generic) {
                        for i in 0..j {
                            assert_eq!(dst[(i, j)], dst_orig[(i, j)]);
                        }
                        for i in j..n {
                            assert_approx_eq!(dst[(i, j)], dst_target[(i, j)]);
                        }
                    } else {
                        for i in 0..=j {
                            assert_eq!(dst[(i, j)], dst_orig[(i, j)]);
                        }
                        for i in j + 1..n {
                            assert_approx_eq!(dst[(i, j)], dst_target[(i, j)]);
                        }
                    }
                }
            } else {
                for j in 0..n {
                    if matches!(dst_structure.diag_kind(), DiagonalKind::Generic) {
                        for i in 0..=j {
                            assert_approx_eq!(dst[(i, j)], dst_target[(i, j)]);
                        }
                        for i in j + 1..n {
                            assert_eq!(dst[(i, j)], dst_orig[(i, j)]);
                        }
                    } else {
                        for i in 0..j {
                            assert_approx_eq!(dst[(i, j)], dst_target[(i, j)]);
                        }
                        for i in j..n {
                            assert_eq!(dst[(i, j)], dst_orig[(i, j)]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn triangular() {
        use BlockStructure::*;
        let structures = [
            Rectangular,
            TriangularLower,
            TriangularUpper,
            StrictTriangularLower,
            StrictTriangularUpper,
            UnitTriangularLower,
            UnitTriangularUpper,
        ];

        for dst in structures {
            for lhs in structures {
                for rhs in structures {
                    for _ in 0..3 {
                        let m = random::<usize>() % 100;
                        let mut n = random::<usize>() % 100;
                        let mut k = random::<usize>() % 100;

                        match (!dst.is_dense(), !lhs.is_dense(), !rhs.is_dense()) {
                            (true, true, _) | (true, _, true) | (_, true, true) => {
                                n = m;
                                k = m;
                            }
                            _ => (),
                        }

                        if !dst.is_dense() {
                            n = m;
                        }

                        if !lhs.is_dense() {
                            k = m;
                        }

                        if !rhs.is_dense() {
                            k = n;
                        }

                        run_test_problem(m, n, k, dst, lhs, rhs);
                    }
                }
            }
        }
    }
}
