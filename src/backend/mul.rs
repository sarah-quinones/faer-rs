use crate::{MatMut, MatRef};

use assert2::assert as fancy_assert;
use assert2::debug_assert as fancy_debug_assert;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use gemm::{gemm, gemm_req};
use num_traits::Zero;
use reborrow::*;

/// Computes the memory requirements of `mul::mat_mat`.
pub fn mat_mat_req<T: 'static>(
    max_dst_rows: usize,
    max_dst_cols: usize,
    max_lhs_cols: usize,
    max_n_threads: usize,
) -> Result<StackReq, SizeOverflow> {
    gemm_req::<T>(max_dst_rows, max_dst_cols, max_lhs_cols, max_n_threads)
}

/// Computes the memory requirements of `mul::mat_mat`.
pub fn mat_mat_accum_req<T: 'static>(
    max_dst_rows: usize,
    max_dst_cols: usize,
    max_lhs_cols: usize,
    max_n_threads: usize,
) -> Result<StackReq, SizeOverflow> {
    gemm_req::<T>(max_dst_rows, max_dst_cols, max_lhs_cols, max_n_threads)
}

/// Computes the matrix product `beta * lhs * rhs` and stores the result in `dst`.
/// `n_threads_hint` is a hint for how many threads should be used, but the implementation
/// may use more or less threads internally.
///
/// The preexisting values in `dst` are not read so it is allowed to be a view over uninitialized
/// values if `T: Copy`.
///
/// # Safety
///
/// Requires that the matrix dimensions be compatible for matrix multiplication.  
/// i.e.  
///  - `dst.nrows() == lhs.nrows()`
///  - `dst.ncols() == rhs.ncols()`
///  - `lhs.ncols() == rhs.nrows()`
#[track_caller]
#[inline]
pub unsafe fn mat_mat_unchecked<T>(
    dst: MatMut<'_, T>,
    lhs: MatRef<'_, T>,
    rhs: MatRef<'_, T>,
    beta: T,
    n_threads_hint: usize,
    stack: DynStack<'_>,
) where
    T: Zero + Send + Sync + 'static,
    for<'a> &'a T: core::ops::Add<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Mul<&'a T, Output = T>,
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
    gemm(
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
        beta,
        n_threads_hint,
        stack,
    );
}

/// Computes the matrix product `beta * lhs * rhs` and stores the result in `dst`.
/// `n_threads_hint` is a hint for how many threads should be used, but the implementation
/// may use more or less threads internally.
///
/// The preexisting values in `dst` are not read so it is allowed to be a view over uninitialized
/// values if `T: Copy`.
///
/// # Panics
///
/// Requires that the matrix dimensions be compatible for matrix multiplication.  
/// i.e.  
///  - `dst.nrows() == lhs.nrows()`
///  - `dst.ncols() == rhs.ncols()`
///  - `lhs.ncols() == rhs.nrows()`
#[track_caller]
#[inline]
pub fn mat_mat<T>(
    dst: MatMut<'_, T>,
    lhs: MatRef<'_, T>,
    rhs: MatRef<'_, T>,
    beta: T,
    n_threads_hint: usize,
    stack: DynStack<'_>,
) where
    T: Zero + Send + Sync + 'static,
    for<'a> &'a T: core::ops::Add<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Mul<&'a T, Output = T>,
{
    fancy_assert!(dst.nrows() == lhs.nrows());
    fancy_assert!(dst.ncols() == rhs.ncols());
    fancy_assert!(lhs.ncols() == rhs.nrows());
    unsafe { mat_mat_unchecked(dst, lhs, rhs, beta, n_threads_hint, stack) }
}

/// Computes the matrix product `alpha * dst + beta * lhs * rhs` and stores the result in `dst`.
/// `n_threads_hint` is a hint for how many threads should be used, but the implementation
/// may use more or less threads internally.
///
/// # Safety
///
/// Requires that the matrix dimensions be compatible for matrix multiplication.  
/// i.e.  
///  - `dst.nrows() == lhs.nrows()`
///  - `dst.ncols() == rhs.ncols()`
///  - `lhs.ncols() == rhs.nrows()`
#[track_caller]
#[inline]
pub unsafe fn mat_mat_accum_unchecked<T>(
    dst: MatMut<'_, T>,
    lhs: MatRef<'_, T>,
    rhs: MatRef<'_, T>,
    alpha: T,
    beta: T,
    n_threads_hint: usize,
    stack: DynStack<'_>,
) where
    T: Zero + Send + Sync + 'static,
    for<'a> &'a T: core::ops::Add<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Mul<&'a T, Output = T>,
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
    gemm(
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
        alpha,
        beta,
        n_threads_hint,
        stack,
    );
}

/// Computes the matrix product `alpha * dst + beta * lhs * rhs` and stores the result in `dst`.
/// `n_threads_hint` is a hint for how many threads should be used, but the implementation
/// may use more or less threads internally.
///
/// # Panics
///
/// Requires that the matrix dimensions be compatible for matrix multiplication.  
/// i.e.  
///  - `dst.nrows() == lhs.nrows()`
///  - `dst.ncols() == rhs.ncols()`
///  - `lhs.ncols() == rhs.nrows()`
#[track_caller]
#[inline]
pub fn mat_mat_accum<T>(
    dst: MatMut<'_, T>,
    lhs: MatRef<'_, T>,
    rhs: MatRef<'_, T>,
    alpha: T,
    beta: T,
    n_threads_hint: usize,
    stack: DynStack<'_>,
) where
    T: Zero + Send + Sync + 'static,
    for<'a> &'a T: core::ops::Add<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Mul<&'a T, Output = T>,
{
    fancy_assert!(dst.nrows() == lhs.nrows());
    fancy_assert!(dst.ncols() == rhs.ncols());
    fancy_assert!(lhs.ncols() == rhs.nrows());

    unsafe { mat_mat_accum_unchecked(dst, lhs, rhs, alpha, beta, n_threads_hint, stack) }
}

pub mod triangular {
    use crate::{izip, temp_mat_req, temp_mat_uninit};

    use super::*;

    pub fn mat_mat_accum_dst_lower_half_only_req<T: 'static>(
        max_dst_dim: usize,
        max_lhs_cols: usize,
        max_n_threads: usize,
    ) -> Result<StackReq, SizeOverflow> {
        let n = max_dst_dim;
        let k = max_lhs_cols;
        let n_threads = max_n_threads;
        if n <= 16 {
            StackReq::try_all_of([
                temp_mat_req::<T>(n, n)?,
                mat_mat_req::<T>(n, n, k, n_threads)?,
            ])
        } else {
            if n_threads <= 1 {
                StackReq::try_any_of([
                    mat_mat_accum_dst_lower_half_only_req::<T>(n / 2, k, n_threads)?,
                    mat_mat_accum_req::<T>(n - n / 2, n / 2, k, n_threads)?,
                    mat_mat_accum_dst_lower_half_only_req::<T>(n - n / 2, k, n_threads)?,
                ])
            } else {
                let bot_left_req = mat_mat_accum_req::<T>(n - n / 2, n / 2, k, n_threads / 2)?;
                let n_threads = n_threads - n_threads / 2;
                StackReq::try_all_of([
                    bot_left_req,
                    if n_threads <= 1 {
                        StackReq::try_any_of([
                            mat_mat_accum_dst_lower_half_only_req::<T>(n / 2, k, n_threads)?,
                            mat_mat_accum_dst_lower_half_only_req::<T>(n - n / 2, k, n_threads)?,
                        ])?
                    } else {
                        StackReq::try_all_of([
                            mat_mat_accum_dst_lower_half_only_req::<T>(n / 2, k, n_threads / 2)?,
                            mat_mat_accum_dst_lower_half_only_req::<T>(
                                n - n / 2,
                                k,
                                n_threads - n_threads / 2,
                            )?,
                        ])?
                    },
                ])
            }
        }
    }

    #[inline]
    pub unsafe fn mat_mat_accum_dst_lower_half_only_unchecked<T>(
        dst: MatMut<'_, T>,
        lhs: MatRef<'_, T>,
        rhs: MatRef<'_, T>,
        alpha: T,
        beta: T,
        n_threads_hint: usize,
        stack: DynStack<'_>,
    ) where
        T: Zero + Clone + Send + Sync + 'static,
        for<'a> &'a T: core::ops::Add<&'a T, Output = T>,
        for<'a> &'a T: core::ops::Mul<&'a T, Output = T>,
    {
        fancy_debug_assert!(dst.nrows() == dst.ncols());
        fancy_debug_assert!(dst.nrows() == lhs.nrows());
        fancy_debug_assert!(dst.ncols() == rhs.ncols());
        fancy_debug_assert!(lhs.ncols() == rhs.nrows());

        let n = dst.nrows();
        let k = lhs.ncols();

        let mut stack = stack;

        let n_threads_hint = if n * n * k <= 128 * 128 * 128 {
            1
        } else {
            n_threads_hint
        };

        if n <= 16 {
            temp_mat_uninit! {
                let (mut temp_dst, stack) = temp_mat_uninit::<T>(n, n, stack);
            };
            mat_mat_unchecked(temp_dst.rb_mut(), lhs, rhs, beta, n_threads_hint, stack);
            for (j, (temp_col, col)) in
                izip!(temp_dst.into_const().into_col_iter(), dst.into_col_iter()).enumerate()
            {
                let (_, temp_col) = temp_col.split_at(j);
                let (_, col) = col.split_at(j);

                for (temp, out) in izip!(temp_col, col) {
                    *out = &(&alpha * &*out) + temp;
                }
            }
        } else {
            let (dst_top_left, _, dst_bot_left, dst_bot_right) =
                dst.split_at_unchecked(n / 2, n / 2);
            let (_, lhs_top, _, lhs_bot) = lhs.split_at_unchecked(n / 2, 0);
            let (_, _, rhs_left, rhs_right) = rhs.split_at_unchecked(n / 2, 0);

            if n_threads_hint <= 1 {
                mat_mat_accum_dst_lower_half_only_unchecked(
                    dst_top_left,
                    lhs_top,
                    rhs_left,
                    alpha.clone(),
                    beta.clone(),
                    n_threads_hint,
                    stack.rb_mut(),
                );
                mat_mat_accum_unchecked(
                    dst_bot_left,
                    lhs_bot,
                    rhs_left,
                    alpha.clone(),
                    beta.clone(),
                    n_threads_hint,
                    stack.rb_mut(),
                );
                mat_mat_accum_dst_lower_half_only_unchecked(
                    dst_bot_right,
                    lhs_bot,
                    rhs_right,
                    alpha,
                    beta,
                    n_threads_hint,
                    stack.rb_mut(),
                );
            } else {
                let bot_left_req =
                    mat_mat_accum_req::<T>(n - n / 2, n / 2, k, n_threads_hint / 2).unwrap();
                let (mut bot_left_stack_mem, stack) = stack.make_aligned_uninit::<u8>(
                    bot_left_req.size_bytes(),
                    bot_left_req.align_bytes(),
                );
                let bot_left_stack = DynStack::new(&mut bot_left_stack_mem);

                rayon::join(
                    || {
                        let n_threads_hint = n_threads_hint / 2;
                        mat_mat_accum_unchecked(
                            dst_bot_left,
                            lhs_bot,
                            rhs_left,
                            alpha.clone(),
                            beta.clone(),
                            n_threads_hint,
                            bot_left_stack,
                        );
                    },
                    || {
                        let n_threads_hint = n_threads_hint - n_threads_hint / 2;
                        if n_threads_hint <= 1 {
                            let mut stack = stack;
                            mat_mat_accum_dst_lower_half_only_unchecked(
                                dst_top_left,
                                lhs_top,
                                rhs_left,
                                alpha.clone(),
                                beta.clone(),
                                n_threads_hint,
                                stack.rb_mut(),
                            );
                            mat_mat_accum_dst_lower_half_only_unchecked(
                                dst_bot_right,
                                lhs_bot,
                                rhs_right,
                                alpha.clone(),
                                beta.clone(),
                                n_threads_hint,
                                stack.rb_mut(),
                            );
                        } else {
                            let top_left_req = mat_mat_accum_dst_lower_half_only_req::<T>(
                                n / 2,
                                k,
                                n_threads_hint / 2,
                            )
                            .unwrap();

                            let (mut top_left_stack_mem, stack) = stack.make_aligned_uninit::<u8>(
                                top_left_req.size_bytes(),
                                top_left_req.align_bytes(),
                            );
                            let top_left_stack = DynStack::new(&mut top_left_stack_mem);

                            rayon::join(
                                || {
                                    mat_mat_accum_dst_lower_half_only_unchecked(
                                        dst_top_left,
                                        lhs_top,
                                        rhs_left,
                                        alpha.clone(),
                                        beta.clone(),
                                        n_threads_hint / 2,
                                        top_left_stack,
                                    );
                                },
                                || {
                                    mat_mat_accum_dst_lower_half_only_unchecked(
                                        dst_bot_right,
                                        lhs_bot,
                                        rhs_right,
                                        alpha.clone(),
                                        beta.clone(),
                                        n_threads_hint - n_threads_hint / 2,
                                        stack,
                                    );
                                },
                            );
                        }
                    },
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat;

    #[test]
    fn mul_mul() {
        let lhs = mat![[1.0, 2.0], [3.0, 4.0]];
        let rhs = mat![[5.0], [7.0]];
        let mut dst = mat![[0.0], [0.0]];

        let mut mem = dyn_stack::GlobalMemBuffer::new(mat_mat_req::<f64>(2, 1, 2, 4).unwrap());
        let stack = DynStack::new(&mut mem);
        mat_mat(dst.as_mut(), lhs.as_ref(), rhs.as_ref(), 2.0, 1, stack);

        fancy_assert!(dst[(0, 0)] == 38.0);
        fancy_assert!(dst[(1, 0)] == 86.0);
    }

    #[test]
    fn mul_mul_accum() {
        let lhs = mat![[1.0, 2.0], [3.0, 4.0]];
        let rhs = mat![[5.0], [7.0]];
        let mut dst = mat![[4.0], [3.0]];

        let mut mem =
            dyn_stack::GlobalMemBuffer::new(mat_mat_accum_req::<f64>(2, 1, 2, 4).unwrap());
        let stack = DynStack::new(&mut mem);
        mat_mat_accum(
            dst.as_mut(),
            lhs.as_ref(),
            rhs.as_ref(),
            -2.0,
            2.0,
            1,
            stack,
        );

        fancy_assert!(dst[(0, 0)] == 30.0);
        fancy_assert!(dst[(1, 0)] == 80.0);
    }
}
