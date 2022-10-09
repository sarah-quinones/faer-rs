use core::any::TypeId;
use core::cmp::Ordering;
use core::ops::Add;
use core::ops::Div;
use core::ops::Mul;
use core::ops::Neg;

use crate::backend::mul;
use crate::backend::mul::triangular::BlockStructure;
use crate::backend::permutation::PermutationIndicesMut;
use crate::backend::solve;
use crate::izip;
use crate::temp_mat_req;
use crate::temp_mat_uninit;
use crate::unreachable_unchecked;
use crate::ColMut;
use crate::MatMut;
use crate::MatRef;

use assert2::assert as fancy_assert;
use assert2::debug_assert as fancy_debug_assert;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use num_traits::sign::Signed;
use num_traits::{Inv, One, Zero};
use pulp::Arch;
use reborrow::*;

fn cholesky_in_place_left_looking_req<T: 'static>(
    dim: usize,
    block_size: usize,
    n_threads: usize,
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
            mul::triangular::matmul_req::<T>(
                TriangularLower,
                Rectangular,
                Rectangular,
                bs,
                bs,
                n - bs,
                n_threads,
            )?,
            cholesky_in_place_left_looking_req::<T>(bs, bs / 2, n_threads)?,
            mul::matmul_req::<T>(n - bs, bs, n - bs, n_threads)?,
            solve::triangular::solve_unit_lower_triangular_in_place_req::<T>(bs, bs, n_threads)?,
        ])?,
    ])
}

unsafe fn cholesky_in_place_left_looking_unchecked<T>(
    matrix: MatMut<'_, T>,
    block_size: usize,
    n_threads: usize,
    stack: DynStack<'_>,
) where
    T: Zero + One + Clone + Send + Sync + 'static,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Inv<Output = T>,
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

        mul::triangular::matmul(
            a11.rb_mut(),
            BlockStructure::TriangularLower,
            l10xd0,
            BlockStructure::Rectangular,
            l10.transpose(),
            BlockStructure::Rectangular,
            Some(&T::one()),
            &-&T::one(),
            n_threads,
            stack.rb_mut(),
        );

        cholesky_in_place_left_looking_unchecked(
            a11.rb_mut(),
            block_size / 2,
            n_threads,
            stack.rb_mut(),
        );

        if idx + block_size == n {
            break;
        }

        let ld11 = a11.into_const();
        let l11 = ld11;
        let d1 = ld11.diagonal_unchecked();

        mul::matmul(
            a21.rb_mut(),
            l20,
            l10xd0.transpose(),
            Some(&T::one()),
            &-&T::one(),
            n_threads,
            stack.rb_mut(),
        );

        solve::triangular::solve_unit_lower_triangular_in_place_unchecked(
            l11,
            a21.rb_mut().transpose(),
            n_threads,
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

/// Computes the memory requirements for a cholesky decomposition of a square matrix of dimension
/// `dim`.
pub fn raw_cholesky_in_place_req<T: 'static>(
    dim: usize,
    n_threads: usize,
) -> Result<StackReq, SizeOverflow> {
    if dim < 32 {
        cholesky_in_place_left_looking_req::<T>(dim, 16, n_threads)
    } else {
        let bs = dim / 2;
        let rem = dim - bs;
        StackReq::try_any_of([
            solve::triangular::solve_unit_lower_triangular_in_place_req::<T>(bs, rem, n_threads)?,
            StackReq::try_all_of([
                temp_mat_req::<T>(rem, bs)?,
                mul::triangular::matmul_req::<T>(
                    BlockStructure::TriangularLower,
                    BlockStructure::Rectangular,
                    BlockStructure::Rectangular,
                    rem,
                    rem,
                    bs,
                    n_threads,
                )?,
            ])?,
        ])
    }
}

unsafe fn cholesky_in_place_unchecked<T>(
    matrix: MatMut<'_, T>,
    n_threads: usize,
    stack: DynStack<'_>,
) where
    T: Zero + One + Clone + Send + Sync + 'static,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Inv<Output = T>,
{
    // right looking cholesky

    fancy_debug_assert!(matrix.nrows() == matrix.ncols());
    let mut matrix = matrix;
    let mut stack = stack;

    let n = matrix.nrows();
    if n < 32 {
        cholesky_in_place_left_looking_unchecked(matrix, 16, n_threads, stack);
    } else {
        let block_size = n / 2;
        let rem = n - block_size;
        let (mut l00, _, mut a10, mut a11) =
            matrix.rb_mut().split_at_unchecked(block_size, block_size);

        cholesky_in_place_unchecked(l00.rb_mut(), n_threads, stack.rb_mut());

        let l00 = l00.into_const();
        let d0 = l00.diagonal_unchecked();

        solve::triangular::solve_unit_lower_triangular_in_place_unchecked(
            l00,
            a10.rb_mut().transpose(),
            n_threads,
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

            mul::triangular::matmul(
                a11.rb_mut(),
                BlockStructure::TriangularLower,
                a10.into_const(),
                BlockStructure::Rectangular,
                l10xd0.transpose().into_const(),
                BlockStructure::Rectangular,
                Some(&T::one()),
                &-&T::one(),
                n_threads,
                stack,
            );
        }

        cholesky_in_place_unchecked(a11, n_threads, stack);
    }
}

/// Computes the cholesky factors `L` and `D` of the input matrix such that `L` is strictly lower
/// triangular, `D` is diagonal, and `L×D×L.transpose() == matrix`, then stores them back in the
/// same matrix.
///
/// The input matrix is interpreted as symmetric and only the lower triangular part is read.
///
/// The matrix `L` is stored in the strictly lower triangular part of the input matrix, and the
/// diagonal elements of `D` are stored on the diagonal.
///
/// The strictly upper triangular part of the matrix is not accessed.
///
/// # Warning
///
/// The cholesky decomposition may have poor numerical stability properties when used with non
/// positive definite matrices. In the general case, it is recommended to first permute the matrix
/// using [`compute_cholesky_permutation`] and
/// [`apply_symmetric_permutation`](crate::backend::symmetric::apply_symmetric_permutation).
///
/// # Panics
///
/// Panics if the input matrix is not square.
#[track_caller]
#[inline]
pub fn raw_cholesky_in_place<T>(matrix: MatMut<'_, T>, n_threads: usize, stack: DynStack<'_>)
where
    T: Zero + One + Clone + Send + Sync + 'static,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Inv<Output = T>,
{
    fancy_assert!(
        matrix.ncols() == matrix.nrows(),
        "only square matrices can be decomposed into cholesky factors",
    );
    unsafe { cholesky_in_place_unchecked(matrix, n_threads, stack) }
}

/// Computes a permutation that reduces the chance of numerical errors during the cholesky
/// factorization, then stores the result in `perm_indices` and `perm_inv_indices`.
#[track_caller]
pub fn compute_cholesky_permutation<'a, T>(
    perm_indices: &'a mut [usize],
    perm_inv_indices: &'a mut [usize],
    matrix: MatRef<'_, T>,
) -> PermutationIndicesMut<'a>
where
    T: Signed + PartialOrd,
{
    let n = matrix.nrows();
    fancy_assert!(
        matrix.nrows() == matrix.ncols(),
        "input matrix must be square",
    );
    fancy_assert!(
        perm_indices.len() == n,
        "length of permutation must be equal to the matrix dimension",
    );
    fancy_assert!(
        perm_inv_indices.len() == n,
        "length of inverse permutation must be equal to the matrix dimension",
    );

    let diag = matrix.diagonal();
    for (i, p) in perm_indices.iter_mut().enumerate() {
        *p = i;
    }

    perm_indices.sort_unstable_by(move |&i, &j| {
        let lhs = unsafe { diag.get_unchecked(i) }.abs();
        let rhs = unsafe { diag.get_unchecked(j) }.abs();
        let cmp = rhs.partial_cmp(&lhs);
        if let Some(cmp) = cmp {
            cmp
        } else {
            Ordering::Equal
        }
    });

    for (i, p) in perm_indices.iter().copied().enumerate() {
        *unsafe { perm_inv_indices.get_unchecked_mut(p) } = i;
    }

    unsafe { PermutationIndicesMut::new_unchecked(perm_indices, perm_inv_indices) }
}

pub fn solve_in_place_req<T: 'static>(
    cholesky_dim: usize,
    rhs_ncols: usize,
    n_threads: usize,
) -> Result<StackReq, SizeOverflow> {
    use super::solve::triangular::*;
    StackReq::try_any_of([
        solve_unit_lower_triangular_in_place_req::<T>(cholesky_dim, rhs_ncols, n_threads)?,
        solve_unit_upper_triangular_in_place_req::<T>(cholesky_dim, rhs_ncols, n_threads)?,
    ])
}

#[track_caller]
pub fn solve_in_place<T>(
    cholesky_factors: MatRef<'_, T>,
    rhs: MatMut<'_, T>,
    n_threads: usize,
    stack: DynStack<'_>,
) where
    T: Zero + One + Clone + Send + Sync + 'static,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Div<Output = T>,
{
    let n = cholesky_factors.nrows();
    let k = rhs.ncols();

    fancy_assert!(cholesky_factors.nrows() == cholesky_factors.ncols());
    fancy_assert!(rhs.nrows() == n);

    let mut rhs = rhs;
    let mut stack = stack;

    crate::backend::solve::triangular::solve_unit_lower_triangular_in_place(
        cholesky_factors,
        rhs.rb_mut(),
        n_threads,
        stack.rb_mut(),
    );

    for j in 0..k {
        for i in 0..n {
            let d = unsafe { cholesky_factors.get_unchecked(i, i) };
            let rhs = unsafe { rhs.rb_mut().get_unchecked(i, j) };
            *rhs = &*rhs / d;
        }
    }

    crate::backend::solve::triangular::solve_unit_upper_triangular_in_place(
        cholesky_factors.transpose(),
        rhs.rb_mut(),
        n_threads,
        stack.rb_mut(),
    );
}

use core::mem::size_of;
use pulp::Simd;
use seq_macro::seq;

macro_rules! generate {
    ($name: ident, $r: tt, $ty: ty, $tys: ty, $splat: ident, $mul_add: ident) => {
        #[inline(always)]
        pub unsafe fn $name<S: Simd, T: 'static>(
            simd: S,
            n: usize,
            l_col: *mut T,
            w: *mut T,
            w_col_stride: isize,
            p_array: *const T,
            mu_array: *const T,
        ) {
            assert_eq!(core::any::TypeId::of::<T>(), core::any::TypeId::of::<$ty>());
            let l_col = l_col as *mut $ty;
            let w = w as *mut $ty;
            let p_array = p_array as *const $ty;
            let mu_array = mu_array as *const $ty;
            let lanes = size_of::<$tys>() / size_of::<$ty>();

            let n_vec = n / lanes;
            let n_rem = n % lanes;

            seq!(I in 0..$r {
                let p~I = -*p_array.add(I);
                let mu~I = *mu_array.add(I);
                let w_col~I = w.offset(I * w_col_stride);
            });

            // vectorized section
            {
                let l_col = l_col as *mut $tys;

                seq!(I in 0..$r {
                    let p~I = simd.$splat(p~I);
                    let mu~I = simd.$splat(mu~I);
                    let w_col~I = w_col~I as *mut $tys;
                });

                for i in 0..n_vec {
                    let mut l = *l_col.add(i);
                    seq!(I in 0..$r {
                        let mut w~I = *w_col~I.add(i);
                    });

                    seq!(I in 0..$r {
                        w~I = simd.$mul_add(p~I, l, w~I);
                        l = simd.$mul_add(mu~I, w~I, l);
                    });

                    l_col.add(i).write(l);
                    seq!(I in 0..$r {
                        w_col~I.add(i).write(w~I);
                    });
                }
            }
            // scalar section
            {
                for i in n - n_rem..n {
                    let mut l = *l_col.add(i);
                    seq!(I in 0..$r {
                        let mut w~I = *w_col~I.add(i);
                    });

                    seq!(I in 0..$r {
                        w~I = $ty::mul_add(p~I, l, w~I);
                        l = $ty::mul_add(mu~I, w~I, l);
                    });

                    l_col.add(i).write(l);
                    seq!(I in 0..$r {
                        w_col~I.add(i).write(w~I);
                    });
                }
            }
        }
    };
}

macro_rules! generate_generic {
    ($name: ident, $r: tt) => {
        #[inline(always)]
        pub unsafe fn $name<S: Simd, T: 'static>(
            _simd: S,
            n: usize,
            l_col: *mut T,
            l_row_stride: isize,
            w: *mut T,
            w_row_stride: isize,
            w_col_stride: isize,
            p_array: *const T,
            mu_array: *const T,
        ) where
            T: Clone,
            for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
        {
            seq!(I in 0..$r {
                let p~I = &-&*p_array.add(I);
                let mu~I = &(*mu_array.add(I)).clone();
                let w_col~I = w.offset(I * w_col_stride);
            });

            for i in 0..n {
                let mut l = (*l_col.offset(i as isize * l_row_stride)).clone();
                seq!(I in 0..$r {
                    let mut w~I = (*w_col~I.offset(i as isize * w_row_stride)).clone();
                });

                seq!(I in 0..$r {
                    w~I = &(p~I * &l) + &w~I;
                    l = &(mu~I * &w~I) + &l;
                });

                *l_col.offset(i as isize * l_row_stride) = l;
                seq!(I in 0..$r {
                    *w_col~I.offset(i as isize * w_row_stride) = w~I;
                });
            }
        }
    };
}

generate_generic!(r1, 1);
generate_generic!(r2, 2);
generate_generic!(r3, 3);
generate_generic!(r4, 4);

generate!(rank_1_f64, 1, f64, S::f64s, f64s_splat, f64s_mul_adde);
generate!(rank_2_f64, 2, f64, S::f64s, f64s_splat, f64s_mul_adde);
generate!(rank_3_f64, 3, f64, S::f64s, f64s_splat, f64s_mul_adde);
generate!(rank_4_f64, 4, f64, S::f64s, f64s_splat, f64s_mul_adde);

generate!(rank_1_f32, 1, f32, S::f32s, f32s_splat, f32s_mul_adde);
generate!(rank_2_f32, 2, f32, S::f32s, f32s_splat, f32s_mul_adde);
generate!(rank_3_f32, 3, f32, S::f32s, f32s_splat, f32s_mul_adde);
generate!(rank_4_f32, 4, f32, S::f32s, f32s_splat, f32s_mul_adde);

struct RankRUpdate<'a, T> {
    ld: MatMut<'a, T>,
    w: MatMut<'a, T>,
    alpha: ColMut<'a, T>,
    r: &'a mut dyn FnMut() -> usize,
}

impl<'a, T> pulp::WithSimd for RankRUpdate<'a, T>
where
    T: Zero + Clone + 'static,
    for<'b> &'b T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Inv<Output = T>,
{
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, s: S) -> Self::Output {
        let RankRUpdate {
            mut ld,
            mut w,
            mut alpha,
            r,
        } = self;
        let n = ld.nrows();
        let k = w.ncols();

        fancy_debug_assert!(ld.ncols() == n);
        fancy_debug_assert!(w.nrows() == n);
        fancy_debug_assert!(alpha.nrows() == k);

        let l_rs = ld.row_stride();
        let w_cs = w.col_stride();
        let w_rs = w.row_stride();

        unsafe {
            for j in 0..n {
                let r = (*r)().min(k);

                let mut r_idx = 0;
                while r_idx < r {
                    let r_chunk = (r - r_idx).min(4);
                    let mut p_array = [T::zero(), T::zero(), T::zero(), T::zero()];
                    let mut mu_array = [T::zero(), T::zero(), T::zero(), T::zero()];

                    let mut dj = ld.rb().get_unchecked(j, j).clone();
                    for k in 0..r_chunk {
                        let p = p_array.get_unchecked_mut(k);
                        let mu = mu_array.get_unchecked_mut(k);
                        let alpha = alpha.rb_mut().get_unchecked(r_idx + k);
                        *p = w.rb().get_unchecked(j, r_idx + k).clone();
                        let new_dj = &dj + &(&(&*alpha * &*p) * &*p);
                        *mu = &(&*alpha * &*p) * &new_dj.inv();
                        *alpha = &*alpha + &-&(&new_dj * &(&*mu * &*mu));

                        dj = new_dj;
                    }
                    *ld.rb_mut().get_unchecked(j, j) = dj;

                    let rem = n - j - 1;

                    let ld_ptr = ld.rb_mut().ptr_at(j + 1, j);
                    let w_ptr = w.rb_mut().ptr_at(j + 1, r_idx);
                    let p = p_array.as_ptr();
                    let mu = mu_array.as_ptr();

                    if TypeId::of::<T>() == TypeId::of::<f64>() && l_rs == 1 && w_rs == 1 {
                        match r_chunk {
                            1 => s.vectorize(|| rank_1_f64(s, rem, ld_ptr, w_ptr, w_cs, p, mu)),
                            2 => s.vectorize(|| rank_2_f64(s, rem, ld_ptr, w_ptr, w_cs, p, mu)),
                            3 => s.vectorize(|| rank_3_f64(s, rem, ld_ptr, w_ptr, w_cs, p, mu)),
                            4 => s.vectorize(|| rank_4_f64(s, rem, ld_ptr, w_ptr, w_cs, p, mu)),
                            _ => unreachable_unchecked(),
                        };
                    } else if TypeId::of::<T>() == TypeId::of::<f32>() && l_rs == 1 && w_rs == 1 {
                        match r_chunk {
                            1 => s.vectorize(|| rank_1_f32(s, rem, ld_ptr, w_ptr, w_cs, p, mu)),
                            2 => s.vectorize(|| rank_2_f32(s, rem, ld_ptr, w_ptr, w_cs, p, mu)),
                            3 => s.vectorize(|| rank_3_f32(s, rem, ld_ptr, w_ptr, w_cs, p, mu)),
                            4 => s.vectorize(|| rank_4_f32(s, rem, ld_ptr, w_ptr, w_cs, p, mu)),
                            _ => unreachable_unchecked(),
                        };
                    } else {
                        match r_chunk {
                            1 => s.vectorize(|| r1(s, rem, ld_ptr, l_rs, w_ptr, w_rs, w_cs, p, mu)),
                            2 => s.vectorize(|| r2(s, rem, ld_ptr, l_rs, w_ptr, w_rs, w_cs, p, mu)),
                            3 => s.vectorize(|| r3(s, rem, ld_ptr, l_rs, w_ptr, w_rs, w_cs, p, mu)),
                            4 => s.vectorize(|| r4(s, rem, ld_ptr, l_rs, w_ptr, w_rs, w_cs, p, mu)),
                            _ => unreachable_unchecked(),
                        };
                    }

                    r_idx += r_chunk;
                }
            }
        }
    }
}

/// Performs a rank-k update in place, while clobbering the inputs.
///
/// Takes the cholesky factors `L` and `D` of a matrix `A`, meaning that `L×D×L.transpose() == A`,
/// a matrix `W` and a column vector `α`, which is interpreted as a diagonal matrix.
///
/// This function computes the cholesky factors of `A + W×diag(α)×W.transpose()`, and stores the
/// result in the storage of the original cholesky factors.
///
/// The matrix `W` and the vector `α` are clobbered, meaning that the values they contain after the
/// function returns are unspecified.
#[track_caller]
pub fn rank_r_update_clobber<T>(
    cholesky_factors: MatMut<'_, T>,
    w: MatMut<'_, T>,
    alpha: ColMut<'_, T>,
) where
    T: Zero + Clone + 'static,
    for<'b> &'b T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Inv<Output = T>,
{
    let n = cholesky_factors.nrows();
    let k = w.ncols();

    fancy_assert!(cholesky_factors.ncols() == n);
    fancy_assert!(w.nrows() == n);
    fancy_assert!(alpha.nrows() == k);

    Arch::new().dispatch(RankRUpdate {
        ld: cholesky_factors,
        w,
        alpha,
        r: &mut || k,
    });
}

fn delete_rows_and_cols_triangular<T: Clone>(mat: MatMut<'_, T>, idx: &[usize]) {
    let mut mat = mat;
    let n = mat.nrows();
    let r = idx.len();
    fancy_debug_assert!(mat.ncols() == n);
    fancy_debug_assert!(r <= n);

    Arch::new().dispatch(|| {
        (0..=r).for_each(|chunk_j| {
            #[rustfmt::skip]
            let j_start = if chunk_j == 0 { 0 } else { idx[chunk_j - 1] + 1 };
            let j_finish = if chunk_j == r { n } else { idx[chunk_j] };

            for j in j_start..j_finish {
                (chunk_j..=r).for_each(|chunk_i| {
                    #[rustfmt::skip]
                    let i_start = if chunk_i == chunk_j { j } else { idx[chunk_i - 1] + 1 };
                    let i_finish = if chunk_i == r { n } else { idx[chunk_i] };

                    if chunk_i != 0 || chunk_j != 0 {
                        for i in i_start..i_finish {
                            unsafe {
                                *mat.rb_mut().get_unchecked(i - chunk_i, j - chunk_j) =
                                    mat.rb().get_unchecked(i, j).clone();
                            }
                        }
                    }
                })
            }
        });
    });
}

fn rank_update_indices(start_col: usize, indices: &[usize]) -> impl FnMut() -> usize + '_ {
    let mut current_col = start_col;
    let mut current_r = 0;
    move || {
        if current_r == indices.len() {
            current_r
        } else {
            while current_col == indices[current_r] - current_r {
                current_r += 1;
                if current_r == indices.len() {
                    return current_r;
                }
            }

            current_col += 1;
            current_r
        }
    }
}

#[track_caller]
pub fn delete_rows_and_cols_clobber_req<T: 'static>(
    dim: usize,
    number_of_rows_to_remove: usize,
) -> Result<StackReq, SizeOverflow> {
    let r = number_of_rows_to_remove;
    StackReq::try_all_of([temp_mat_req::<T>(dim, r)?, temp_mat_req::<T>(r, 1)?])
}

#[track_caller]
pub fn delete_rows_and_cols_clobber<T: Clone>(
    cholesky_factors: MatMut<'_, T>,
    indices: &mut [usize],
    stack: DynStack<'_>,
) where
    T: Zero + Clone + 'static,
    for<'b> &'b T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Inv<Output = T>,
{
    let n = cholesky_factors.nrows();
    let r = indices.len();
    fancy_assert!(cholesky_factors.ncols() == n);
    fancy_assert!(indices.len() < n);

    if r == 0 {
        return;
    }

    indices.sort_unstable();
    for i in 0..r - 1 {
        fancy_assert!(indices[i + 1] > indices[i]);
    }
    fancy_assert!(indices[r - 1] < n);

    // TODO: check that there are no duplicates
    // check that they are in bounds

    let first = indices[0];

    temp_mat_uninit! {
        let (mut w, stack) = unsafe { temp_mat_uninit::<T>(n - first - r, r, stack) };
        let (alpha, _) = unsafe { temp_mat_uninit::<T>(r, 1, stack) };
    }
    let mut alpha = alpha.col(0);

    Arch::new().dispatch(|| {
        for k in 0..r {
            let j = indices[k];
            unsafe {
                *alpha.rb_mut().ptr_in_bounds_at_unchecked(k) =
                    cholesky_factors.rb().get_unchecked(j, j).clone();
            }

            for chunk_i in k..r {
                let chunk_i = chunk_i + 1;
                let i_start = indices[chunk_i - 1] + 1;
                #[rustfmt::skip]
                let i_finish = if chunk_i == r { n } else { indices[chunk_i] };

                for i in i_start..i_finish {
                    unsafe {
                        *w.rb_mut()
                            .ptr_in_bounds_at_unchecked(i - chunk_i - first, k) =
                            cholesky_factors.rb().get_unchecked(i, j).clone();
                    }
                }
            }
        }
    });
    let mut cholesky_factors = cholesky_factors;
    delete_rows_and_cols_triangular(cholesky_factors.rb_mut(), indices);
    // rank_r_update_clobber(cholesky_factors, w, alpha);

    Arch::new().dispatch(RankRUpdate {
        ld: unsafe {
            cholesky_factors.submatrix_unchecked(first, first, n - first - r, n - first - r)
        },
        w,
        alpha,
        r: &mut rank_update_indices(first, indices),
    });
}

#[track_caller]
pub fn insert_rows_and_cols_clobber<T>(
    cholesky_factors_extended: MatMut<'_, T>,
    insertion_index: usize,
    inserted_matrix: MatMut<'_, T>,
    n_threads: usize,
    stack: DynStack<'_>,
) where
    T: Zero + One + Clone + Send + Sync + 'static,
    for<'b> &'b T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Inv<Output = T>,
{
    let new_n = cholesky_factors_extended.nrows();
    let r = inserted_matrix.ncols();

    fancy_assert!(cholesky_factors_extended.ncols() == new_n);
    fancy_assert!(r < new_n);
    let old_n = new_n - r;

    fancy_assert!(insertion_index <= old_n);

    if r == 0 {
        return;
    }

    let mut current_col = old_n;

    let mut ld = cholesky_factors_extended;

    while current_col != insertion_index {
        current_col -= 1;

        unsafe {
            for i in (current_col..old_n).rev() {
                *ld.rb_mut()
                    .ptr_in_bounds_at_unchecked(i + r, current_col + r) =
                    (*ld.rb().ptr_in_bounds_at_unchecked(i, current_col)).clone();
            }
        }
    }

    while current_col != 0 {
        current_col -= 1;
        unsafe {
            for i in (insertion_index..old_n).rev() {
                *ld.rb_mut().ptr_in_bounds_at_unchecked(i + r, current_col) =
                    (*ld.rb().ptr_in_bounds_at_unchecked(i, current_col)).clone();
            }
        }
    }

    let (ld00, _, l_bot_left, ld_bot_right) =
        unsafe { ld.split_at_unchecked(insertion_index, insertion_index) };
    let ld00 = ld00.into_const();
    let d0 = unsafe { ld00.diagonal_unchecked() };

    let (_, mut l10, _, l20) = unsafe { l_bot_left.split_at_unchecked(r, 0) };
    let (mut ld11, _, mut l21, ld22) = unsafe { ld_bot_right.split_at_unchecked(r, r) };

    let (_, mut a01, _, a_bottom) =
        unsafe { inserted_matrix.split_at_unchecked(insertion_index, 0) };
    let (_, a11, _, a21) = unsafe { a_bottom.split_at_unchecked(r, 0) };

    let mut stack = stack;

    solve::triangular::solve_unit_lower_triangular_in_place(
        ld00.rb(),
        a01.rb_mut(),
        n_threads,
        stack.rb_mut(),
    );

    let a10 = a01.rb().transpose();

    for j in 0..insertion_index {
        let d0_inv = unsafe { d0.get_unchecked(j) }.inv();
        for i in 0..r {
            unsafe {
                *l10.rb_mut().ptr_in_bounds_at_unchecked(i, j) = a10.get_unchecked(i, j) * &d0_inv;
            }
        }
    }

    for j in 0..r {
        for i in j..r {
            unsafe {
                *ld11.rb_mut().ptr_in_bounds_at_unchecked(i, j) =
                    a11.rb().get_unchecked(i, j).clone();
            }
        }
    }

    mul::triangular::matmul(
        ld11.rb_mut(),
        BlockStructure::TriangularLower,
        l10.rb(),
        BlockStructure::Rectangular,
        a01.rb(),
        BlockStructure::Rectangular,
        Some(&T::one()),
        &-&T::one(),
        n_threads,
        stack.rb_mut(),
    );

    raw_cholesky_in_place(ld11.rb_mut(), n_threads, stack.rb_mut());
    let ld11 = ld11.into_const();

    let rem = l21.nrows();

    for j in 0..r {
        for i in 0..rem {
            unsafe {
                *l21.rb_mut().ptr_in_bounds_at_unchecked(i, j) =
                    a21.rb().get_unchecked(i, j).clone();
            }
        }
    }

    mul::matmul(
        l21.rb_mut(),
        l20.rb(),
        a01.rb(),
        Some(&T::one()),
        &-&T::one(),
        n_threads,
        stack.rb_mut(),
    );

    solve::triangular::solve_unit_lower_triangular_in_place(
        ld11,
        l21.rb_mut().transpose(),
        n_threads,
        stack.rb_mut(),
    );

    let d1 = ld11.into_const().diagonal();

    for j in 0..r {
        unsafe {
            let d1_inv = &d1.get_unchecked(j).inv();
            for i in 0..rem {
                let dst = l21.rb_mut().get_unchecked(i, j);
                *dst = &*dst * d1_inv;
            }
        }
    }

    let mut alpha = unsafe { a11.col_unchecked(0) };
    let mut w = a21;

    for j in 0..r {
        unsafe {
            *alpha.rb_mut().ptr_in_bounds_at_unchecked(j) = -ld11.rb().get_unchecked(j, j);

            for i in 0..rem {
                *w.rb_mut().ptr_in_bounds_at_unchecked(i, j) = l21.rb().get(i, j).clone();
            }
        }
    }

    rank_r_update_clobber(ld22, w, alpha);
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use dyn_stack::GlobalMemBuffer;
    use rand::random;

    use super::*;
    use crate::{mat, Mat};

    fn reconstruct_matrix(cholesky_factors: MatRef<'_, f64>) -> Mat<f64> {
        let n = cholesky_factors.nrows();

        let mut lxd = Mat::zeros(n, n);
        for j in 0..n {
            let dj = cholesky_factors[(j, j)];
            lxd[(j, j)] = dj;
            for i in j + 1..n {
                lxd[(i, j)] = cholesky_factors[(i, j)] * dj;
            }
        }

        let mut a_reconstructed = Mat::zeros(n, n);

        use mul::triangular::BlockStructure::*;
        mul::triangular::matmul(
            a_reconstructed.as_mut(),
            Rectangular,
            lxd.as_ref(),
            TriangularLower,
            cholesky_factors.transpose(),
            UnitTriangularUpper,
            None,
            &1.0,
            12,
            DynStack::new(&mut GlobalMemBuffer::new(
                mul::triangular::matmul_req::<f64>(
                    Rectangular,
                    TriangularLower,
                    TriangularUpper,
                    n,
                    n,
                    n,
                    12,
                )
                .unwrap(),
            )),
        );

        a_reconstructed
    }

    #[test]
    fn test_roundtrip() {
        let n = 511;
        let mut a = Mat::with_dims(|_, _| random::<f64>(), n, n);
        let a_orig = a.clone();

        raw_cholesky_in_place(
            a.as_mut(),
            12,
            DynStack::new(&mut GlobalMemBuffer::new(
                raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
            )),
        );

        let a_reconstructed = reconstruct_matrix(a.as_ref());

        for j in 0..n {
            for i in j..n {
                assert_approx_eq!(a_reconstructed[(i, j)], a_orig[(i, j)]);
            }
        }
    }

    #[test]
    fn test_solve() {
        let n = 4;
        let k = 5;
        let mut a = Mat::with_dims(|_, _| random::<f64>(), n, n);
        let mut rhs = Mat::with_dims(|_, _| random::<f64>(), n, k);
        let a_orig = a.clone();
        let rhs_orig = rhs.clone();

        raw_cholesky_in_place(
            a.as_mut(),
            12,
            DynStack::new(&mut GlobalMemBuffer::new(
                raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
            )),
        );

        solve_in_place(
            a.as_ref(),
            rhs.as_mut(),
            12,
            DynStack::new(&mut GlobalMemBuffer::new(
                solve_in_place_req::<f64>(n, k, 12).unwrap(),
            )),
        );

        let mut result = Mat::zeros(n, k);
        use mul::triangular::BlockStructure::*;
        mul::triangular::matmul(
            result.as_mut(),
            Rectangular,
            a_orig.as_ref(),
            TriangularLower,
            rhs.as_ref(),
            Rectangular,
            None,
            &1.0,
            12,
            DynStack::new(&mut GlobalMemBuffer::new(
                mul::triangular::matmul_req::<f64>(
                    Rectangular,
                    TriangularLower,
                    Rectangular,
                    n,
                    k,
                    n,
                    12,
                )
                .unwrap(),
            )),
        );

        mul::triangular::matmul(
            result.as_mut(),
            Rectangular,
            a_orig.as_ref().transpose(),
            StrictTriangularUpper,
            rhs.as_ref(),
            Rectangular,
            Some(&1.0),
            &1.0,
            12,
            DynStack::new(&mut GlobalMemBuffer::new(
                mul::triangular::matmul_req::<f64>(
                    Rectangular,
                    StrictTriangularUpper,
                    Rectangular,
                    n,
                    k,
                    n,
                    12,
                )
                .unwrap(),
            )),
        );

        for j in 0..k {
            for i in 0..n {
                assert_approx_eq!(result[(i, j)], rhs_orig[(i, j)]);
            }
        }
    }

    #[test]
    fn test_update() {
        use mul::triangular::BlockStructure::*;
        let random = |_, _| random::<f64>();

        for k in [0, 1, 2, 3, 4, 5] {
            let n = 511;
            let mut a = Mat::with_dims(random, n, n);
            let mut a_updated = a.clone();
            let mut w = Mat::with_dims(random, n, k);
            let mut alpha = Mat::with_dims(random, k, 1);
            let alpha = alpha.as_mut().col(0);

            let mut w_alpha = Mat::zeros(n, k);
            for j in 0..k {
                for i in 0..n {
                    w_alpha[(i, j)] = alpha[j] * w[(i, j)];
                }
            }

            mul::triangular::matmul(
                a_updated.as_mut(),
                TriangularLower,
                w_alpha.as_ref(),
                Rectangular,
                w.as_ref().transpose(),
                Rectangular,
                Some(&1.0),
                &1.0,
                12,
                DynStack::new(&mut GlobalMemBuffer::new(
                    mul::triangular::matmul_req::<f64>(
                        TriangularLower,
                        Rectangular,
                        Rectangular,
                        n,
                        n,
                        k,
                        12,
                    )
                    .unwrap(),
                )),
            );

            raw_cholesky_in_place(
                a.as_mut(),
                12,
                DynStack::new(&mut GlobalMemBuffer::new(
                    raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
                )),
            );

            rank_r_update_clobber(a.as_mut(), w.as_mut(), alpha);

            let a_reconstructed = reconstruct_matrix(a.as_ref());

            for j in 0..n {
                for i in j..n {
                    assert_approx_eq!(a_reconstructed[(i, j)], a_updated[(i, j)], 1e-5);
                }
            }
        }
    }

    #[test]
    fn test_delete() {
        let a_orig = mat![
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 5.0, 0.0, 0.0],
            [3.0, 6.0, 8.0, 0.0],
            [4.0, 7.0, 9.0, 10.0],
        ];

        {
            let mut a = a_orig.clone();
            let n = a.nrows();
            let r = 2;

            raw_cholesky_in_place(
                a.as_mut(),
                12,
                DynStack::new(&mut GlobalMemBuffer::new(
                    raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
                )),
            );

            delete_rows_and_cols_clobber(
                a.as_mut(),
                &mut [1, 3],
                DynStack::new(&mut GlobalMemBuffer::new(
                    delete_rows_and_cols_clobber_req::<f64>(n, r).unwrap(),
                )),
            );

            let a_reconstructed = reconstruct_matrix(a.as_ref().submatrix(0, 0, n - r, n - r));
            assert_approx_eq!(a_reconstructed[(0, 0)], a_orig[(0, 0)]);
            assert_approx_eq!(a_reconstructed[(1, 0)], a_orig[(2, 0)]);
            assert_approx_eq!(a_reconstructed[(1, 1)], a_orig[(2, 2)]);
        }

        {
            let mut a = a_orig.clone();
            let n = a.nrows();
            let r = 2;

            raw_cholesky_in_place(
                a.as_mut(),
                12,
                DynStack::new(&mut GlobalMemBuffer::new(
                    raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
                )),
            );

            delete_rows_and_cols_clobber(
                a.as_mut(),
                &mut [0, 2],
                DynStack::new(&mut GlobalMemBuffer::new(
                    delete_rows_and_cols_clobber_req::<f64>(n, r).unwrap(),
                )),
            );

            let a_reconstructed = reconstruct_matrix(a.as_ref().submatrix(0, 0, n - r, n - r));
            assert_approx_eq!(a_reconstructed[(0, 0)], a_orig[(1, 1)]);
            assert_approx_eq!(a_reconstructed[(1, 0)], a_orig[(3, 1)]);
            assert_approx_eq!(a_reconstructed[(1, 1)], a_orig[(3, 3)]);
        }

        {
            let mut a = a_orig.clone();
            let n = a.nrows();
            let r = 3;

            raw_cholesky_in_place(
                a.as_mut(),
                12,
                DynStack::new(&mut GlobalMemBuffer::new(
                    raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
                )),
            );

            delete_rows_and_cols_clobber(
                a.as_mut(),
                &mut [0, 2, 3],
                DynStack::new(&mut GlobalMemBuffer::new(
                    delete_rows_and_cols_clobber_req::<f64>(n, r).unwrap(),
                )),
            );

            let a_reconstructed = reconstruct_matrix(a.as_ref().submatrix(0, 0, n - r, n - r));
            assert_approx_eq!(a_reconstructed[(0, 0)], a_orig[(1, 1)]);
        }
    }

    #[test]
    fn test_insert() {
        let a_orig = mat![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 5.0, 6.0, 7.0],
            [3.0, 6.0, 8.0, 9.0],
            [4.0, 7.0, 9.0, 10.0],
        ];

        {
            let mut a = a_orig.clone();

            let mut w = mat![
                [11.0, 17.0],
                [12.0, 18.0],
                [13.0, 14.0],
                [14.0, 20.0],
                [15.0, 21.0],
                [16.0, 22.0],
            ];

            let a_new = mat![
                [1.0, 2.0, 11.0, 17.0, 3.0, 4.0],
                [2.0, 5.0, 12.0, 18.0, 6.0, 7.0],
                [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 14.0, 20.0, 21.0, 22.0],
                [3.0, 6.0, 15.0, 21.0, 8.0, 9.0],
                [4.0, 7.0, 16.0, 22.0, 9.0, 10.0],
            ];

            let n = a.nrows();
            let r = w.ncols();
            let position = 2;

            raw_cholesky_in_place(
                a.as_mut(),
                12,
                DynStack::new(&mut GlobalMemBuffer::new(
                    raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
                )),
            );

            a.resize_with(|_, _| 0.0, n + r, n + r);
            insert_rows_and_cols_clobber(
                a.as_mut(),
                position,
                w.as_mut(),
                12,
                DynStack::new(&mut GlobalMemBuffer::new(StackReq::new::<f64>(1024))),
            );

            let a_reconstructed = reconstruct_matrix(a.as_ref());

            for j in 0..n + r {
                for i in 0..n + r {
                    assert_approx_eq!(a_reconstructed[(i, j)], a_new[(i, j)]);
                }
            }
        }

        {
            let mut a = a_orig.clone();

            let mut w = mat![
                [11.0, 12.0],
                [12.0, 18.0],
                [13.0, 19.0],
                [14.0, 20.0],
                [15.0, 21.0],
                [16.0, 22.0],
            ];

            let a_new = mat![
                [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                [12.0, 18.0, 19.0, 20.0, 21.0, 22.0],
                [13.0, 19.0, 1.0, 2.0, 3.0, 4.0],
                [14.0, 20.0, 2.0, 5.0, 6.0, 7.0],
                [15.0, 21.0, 3.0, 6.0, 8.0, 9.0],
                [16.0, 22.0, 4.0, 7.0, 9.0, 10.0],
            ];

            let n = a.nrows();
            let r = w.ncols();
            let position = 0;

            raw_cholesky_in_place(
                a.as_mut(),
                12,
                DynStack::new(&mut GlobalMemBuffer::new(
                    raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
                )),
            );

            a.resize_with(|_, _| 0.0, n + r, n + r);
            insert_rows_and_cols_clobber(
                a.as_mut(),
                position,
                w.as_mut(),
                12,
                DynStack::new(&mut GlobalMemBuffer::new(StackReq::new::<f64>(1024))),
            );

            let a_reconstructed = reconstruct_matrix(a.as_ref());

            for j in 0..n + r {
                for i in 0..n + r {
                    assert_approx_eq!(a_reconstructed[(i, j)], a_new[(i, j)]);
                }
            }
        }

        {
            let mut a = a_orig.clone();

            let mut w = mat![
                [11.0, 17.0],
                [12.0, 18.0],
                [13.0, 19.0],
                [14.0, 20.0],
                [15.0, 16.0],
                [16.0, 22.0],
            ];

            let a_new = mat![
                [1.0, 2.0, 3.0, 4.0, 11.0, 17.0],
                [2.0, 5.0, 6.0, 7.0, 12.0, 18.0],
                [3.0, 6.0, 8.0, 9.0, 13.0, 19.0],
                [4.0, 7.0, 9.0, 10.0, 14.0, 20.0],
                [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0, 16.0, 22.0],
            ];

            let n = a.nrows();
            let r = w.ncols();
            let position = 4;

            raw_cholesky_in_place(
                a.as_mut(),
                12,
                DynStack::new(&mut GlobalMemBuffer::new(
                    raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
                )),
            );

            a.resize_with(|_, _| 0.0, n + r, n + r);
            insert_rows_and_cols_clobber(
                a.as_mut(),
                position,
                w.as_mut(),
                12,
                DynStack::new(&mut GlobalMemBuffer::new(StackReq::new::<f64>(1024))),
            );

            let a_reconstructed = reconstruct_matrix(a.as_ref());

            for j in 0..n + r {
                for i in 0..n + r {
                    assert_approx_eq!(a_reconstructed[(i, j)], a_new[(i, j)]);
                }
            }
        }
    }
}
