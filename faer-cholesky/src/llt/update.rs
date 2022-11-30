use core::{any::TypeId, mem::size_of};

use assert2::{assert as fancy_assert, debug_assert as fancy_debug_assert};
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{
    mul, mul::triangular::BlockStructure, solve, temp_mat_req, temp_mat_uninit, ColMut,
    ComplexField, Conj, MatMut, Parallelism,
};
use pulp::{Arch, Simd};
use reborrow::*;
use seq_macro::seq;

use crate::{
    ldlt_diagonal::update::{delete_rows_and_cols_triangular, rank_update_indices},
    llt::compute::{cholesky_in_place, cholesky_in_place_req},
};

use super::CholeskyError;

macro_rules! generate {
    ($name: ident, $r: tt, $ty: ty, $tys: ty, $splat: ident, $mul_add: ident, $mul: ident) => {
        #[inline(always)]
        unsafe fn $name<S: Simd, T: 'static>(
            simd: S,
            n: usize,
            l_col: *mut T,
            w: *mut T,
            w_col_stride: isize,
            neg_wj_over_ljj_array: *const T,
            alpha_wj_over_nljj_array: *const T,
            nljj_over_ljj_array: *const T,
        ) {
            assert_eq!(core::any::TypeId::of::<T>(), core::any::TypeId::of::<$ty>());
            let l_col = l_col as *mut $ty;
            let w = w as *mut $ty;
            let neg_wj_over_ljj_array = neg_wj_over_ljj_array as *const $ty;
            let nljj_over_ljj_array = nljj_over_ljj_array as *const $ty;
            let alpha_wj_over_nljj_array = alpha_wj_over_nljj_array as *const $ty;
            let lanes = size_of::<$tys>() / size_of::<$ty>();

            let n_vec = n / lanes;
            let n_rem = n % lanes;

            seq!(I in 0..$r {
                let neg_wj_over_ljj~I = *neg_wj_over_ljj_array.add(I);
                let nljj_over_ljj~I = *nljj_over_ljj_array.add(I);
                let alpha_wj_over_nljj~I = *alpha_wj_over_nljj_array.add(I);
                let w_col~I = w.offset(I * w_col_stride);
            });

            // vectorized section
            {
                let l_col = l_col as *mut $tys;

                seq!(I in 0..$r {
                    let neg_wj_over_ljj~I = simd.$splat(neg_wj_over_ljj~I);
                    let nljj_over_ljj~I = simd.$splat(nljj_over_ljj~I);
                    let alpha_wj_over_nljj~I = simd.$splat(alpha_wj_over_nljj~I);
                    let w_col~I = w_col~I as *mut $tys;
                });

                for i in 0..n_vec {
                    let mut l = *l_col.add(i);

                    seq!(I in 0..$r {
                        let mut w~I = *w_col~I.add(i);
                        w~I = simd.$mul_add(neg_wj_over_ljj~I, l, w~I);
                        l = simd.$mul_add(alpha_wj_over_nljj~I, w~I, simd.$mul(nljj_over_ljj~I, l));
                        w_col~I.add(i).write(w~I);
                    });

                    l_col.add(i).write(l);
                }
            }
            // scalar section
            {
                for i in n - n_rem..n {
                    let mut l = *l_col.add(i);

                    seq!(I in 0..$r {
                        let mut w~I = *w_col~I.add(i);
                        w~I = $ty::mul_add(neg_wj_over_ljj~I, l, w~I);
                        l = $ty::mul_add(alpha_wj_over_nljj~I, w~I, nljj_over_ljj~I * l);
                        w_col~I.add(i).write(w~I);
                    });

                    l_col.add(i).write(l);
                }
            }
        }
    };
}

macro_rules! generate_generic {
    ($name: ident, $r: tt) => {
        #[inline(always)]
        unsafe fn $name<S: Simd, T: ComplexField>(
            _simd: S,
            n: usize,
            l_col: *mut T,
            l_row_stride: isize,
            w: *mut T,
            w_row_stride: isize,
            w_col_stride: isize,
            neg_wj_over_ljj_array: *const T,
            alpha_wj_over_nljj_array: *const T,
            nljj_over_ljj_array: *const T,
        ) {
            seq!(I in 0..$r {
                let neg_wj_over_ljj~I = *neg_wj_over_ljj_array.add(I);
                let nljj_over_ljj~I = *nljj_over_ljj_array.add(I);
                let alpha_wj_over_nljj~I = *alpha_wj_over_nljj_array.add(I);
                let w_col~I = w.offset(I * w_col_stride);
            });

            for i in 0..n {
                let mut l = (*l_col.offset(i as isize * l_row_stride)).clone();

                seq!(I in 0..$r {
                    let mut w~I = (*w_col~I.offset(i as isize * w_row_stride)).clone();
                    w~I = (neg_wj_over_ljj~I * l) + w~I;
                    l = (alpha_wj_over_nljj~I * w~I) + (nljj_over_ljj~I * l);
                    *w_col~I.offset(i as isize * w_row_stride) = w~I;
                });

                *l_col.offset(i as isize * l_row_stride) = l;
            }
        }
    };
}

generate_generic!(r1, 1);
generate_generic!(r2, 2);
generate_generic!(r3, 3);
generate_generic!(r4, 4);

#[rustfmt::skip]
generate!(rank_1_f64, 1, f64, S::f64s, f64s_splat, f64s_mul_adde, f64s_mul);
#[rustfmt::skip]
generate!(rank_2_f64, 2, f64, S::f64s, f64s_splat, f64s_mul_adde, f64s_mul);
#[rustfmt::skip]
generate!(rank_3_f64, 3, f64, S::f64s, f64s_splat, f64s_mul_adde, f64s_mul);
#[rustfmt::skip]
generate!(rank_4_f64, 4, f64, S::f64s, f64s_splat, f64s_mul_adde, f64s_mul);

#[rustfmt::skip]
generate!(rank_1_f32, 1, f32, S::f32s, f32s_splat, f32s_mul_adde, f32s_mul);
#[rustfmt::skip]
generate!(rank_2_f32, 2, f32, S::f32s, f32s_splat, f32s_mul_adde, f32s_mul);
#[rustfmt::skip]
generate!(rank_3_f32, 3, f32, S::f32s, f32s_splat, f32s_mul_adde, f32s_mul);
#[rustfmt::skip]
generate!(rank_4_f32, 4, f32, S::f32s, f32s_splat, f32s_mul_adde, f32s_mul);

struct RankRUpdate<'a, T> {
    l: MatMut<'a, T>,
    w: MatMut<'a, T>,
    alpha: ColMut<'a, T>,
    r: &'a mut dyn FnMut() -> usize,
}

impl<'a, T: ComplexField> pulp::WithSimd for RankRUpdate<'a, T> {
    type Output = Result<(), CholeskyError>;

    #[inline(always)]
    fn with_simd<S: Simd>(self, s: S) -> Self::Output {
        // On the Modification of LDLT Factorizations
        // By R. Fletcher and M. J. D. Powell
        // https://www.ams.org/journals/mcom/1974-28-128/S0025-5718-1974-0359297-1/S0025-5718-1974-0359297-1.pdf

        let RankRUpdate {
            mut l,
            mut w,
            mut alpha,
            r,
        } = self;
        let n = l.nrows();
        let k = w.ncols();

        fancy_debug_assert!(l.ncols() == n);
        fancy_debug_assert!(w.nrows() == n);
        fancy_debug_assert!(alpha.nrows() == k);

        let l_rs = l.row_stride();
        let w_cs = w.col_stride();
        let w_rs = w.row_stride();

        unsafe {
            for j in 0..n {
                let r = (*r)().min(k);

                let mut r_idx = 0;
                while r_idx < r {
                    let r_chunk = (r - r_idx).min(4);
                    let mut neg_wj_over_ljj_array = [T::zero(), T::zero(), T::zero(), T::zero()];
                    let mut alpha_wj_over_nljj_array = [T::zero(), T::zero(), T::zero(), T::zero()];
                    let mut nljj_over_ljj_array = [T::zero(), T::zero(), T::zero(), T::zero()];

                    let mut ljj = l.rb().get_unchecked(j, j).clone();
                    for k in 0..r_chunk {
                        let neg_wj_over_ljj = neg_wj_over_ljj_array.get_unchecked_mut(k);
                        let alpha_conj_wj_over_nljj = alpha_wj_over_nljj_array.get_unchecked_mut(k);
                        let nljj_over_ljj = nljj_over_ljj_array.get_unchecked_mut(k);

                        let alpha = alpha.rb_mut().get_unchecked(r_idx + k);
                        let wj = *w.rb().get_unchecked(j, r_idx + k);
                        let alpha_conj_wj = *alpha * wj.conj();

                        let sqr_nljj = ljj * ljj + alpha_conj_wj * wj;
                        if !(sqr_nljj.real() > T::Real::zero()) {
                            return Err(CholeskyError);
                        }
                        let nljj = sqr_nljj.sqrt();
                        let inv_ljj = ljj.inv();
                        let inv_nljj = nljj.inv();

                        *neg_wj_over_ljj = -(wj * inv_ljj);
                        *nljj_over_ljj = nljj * inv_ljj;
                        *alpha_conj_wj_over_nljj = alpha_conj_wj * inv_nljj;
                        *alpha =
                            *alpha - *alpha_conj_wj_over_nljj * (*alpha_conj_wj_over_nljj).conj();

                        ljj = nljj;
                    }
                    *l.rb_mut().get_unchecked(j, j) = ljj;

                    let rem = n - j - 1;

                    let l_ptr = l.rb_mut().ptr_at(j + 1, j);
                    let w_ptr = w.rb_mut().ptr_at(j + 1, r_idx);
                    let neg_wj_over_ljj = neg_wj_over_ljj_array.as_ptr();
                    let alpha_wj_over_nljj = alpha_wj_over_nljj_array.as_ptr();
                    let nljj_over_ljj = nljj_over_ljj_array.as_ptr();

                    if TypeId::of::<T>() == TypeId::of::<f64>() && l_rs == 1 && w_rs == 1 {
                        match r_chunk {
                            1 => s.vectorize(|| {
                                rank_1_f64(
                                    s,
                                    rem,
                                    l_ptr,
                                    w_ptr,
                                    w_cs,
                                    neg_wj_over_ljj,
                                    alpha_wj_over_nljj,
                                    nljj_over_ljj,
                                )
                            }),
                            2 => s.vectorize(|| {
                                rank_2_f64(
                                    s,
                                    rem,
                                    l_ptr,
                                    w_ptr,
                                    w_cs,
                                    neg_wj_over_ljj,
                                    alpha_wj_over_nljj,
                                    nljj_over_ljj,
                                )
                            }),
                            3 => s.vectorize(|| {
                                rank_3_f64(
                                    s,
                                    rem,
                                    l_ptr,
                                    w_ptr,
                                    w_cs,
                                    neg_wj_over_ljj,
                                    alpha_wj_over_nljj,
                                    nljj_over_ljj,
                                )
                            }),
                            4 => s.vectorize(|| {
                                rank_4_f64(
                                    s,
                                    rem,
                                    l_ptr,
                                    w_ptr,
                                    w_cs,
                                    neg_wj_over_ljj,
                                    alpha_wj_over_nljj,
                                    nljj_over_ljj,
                                )
                            }),
                            _ => unreachable!(),
                        };
                    } else if TypeId::of::<T>() == TypeId::of::<f32>() && l_rs == 1 && w_rs == 1 {
                        match r_chunk {
                            1 => s.vectorize(|| {
                                rank_1_f32(
                                    s,
                                    rem,
                                    l_ptr,
                                    w_ptr,
                                    w_cs,
                                    neg_wj_over_ljj,
                                    alpha_wj_over_nljj,
                                    nljj_over_ljj,
                                )
                            }),
                            2 => s.vectorize(|| {
                                rank_2_f32(
                                    s,
                                    rem,
                                    l_ptr,
                                    w_ptr,
                                    w_cs,
                                    neg_wj_over_ljj,
                                    alpha_wj_over_nljj,
                                    nljj_over_ljj,
                                )
                            }),
                            3 => s.vectorize(|| {
                                rank_3_f32(
                                    s,
                                    rem,
                                    l_ptr,
                                    w_ptr,
                                    w_cs,
                                    neg_wj_over_ljj,
                                    alpha_wj_over_nljj,
                                    nljj_over_ljj,
                                )
                            }),
                            4 => s.vectorize(|| {
                                rank_4_f32(
                                    s,
                                    rem,
                                    l_ptr,
                                    w_ptr,
                                    w_cs,
                                    neg_wj_over_ljj,
                                    alpha_wj_over_nljj,
                                    nljj_over_ljj,
                                )
                            }),
                            _ => unreachable!(),
                        };
                    } else {
                        match r_chunk {
                            1 => s.vectorize(|| {
                                r1(
                                    s,
                                    rem,
                                    l_ptr,
                                    l_rs,
                                    w_ptr,
                                    w_rs,
                                    w_cs,
                                    neg_wj_over_ljj,
                                    alpha_wj_over_nljj,
                                    nljj_over_ljj,
                                )
                            }),
                            2 => s.vectorize(|| {
                                r2(
                                    s,
                                    rem,
                                    l_ptr,
                                    l_rs,
                                    w_ptr,
                                    w_rs,
                                    w_cs,
                                    neg_wj_over_ljj,
                                    alpha_wj_over_nljj,
                                    nljj_over_ljj,
                                )
                            }),
                            3 => s.vectorize(|| {
                                r3(
                                    s,
                                    rem,
                                    l_ptr,
                                    l_rs,
                                    w_ptr,
                                    w_rs,
                                    w_cs,
                                    neg_wj_over_ljj,
                                    alpha_wj_over_nljj,
                                    nljj_over_ljj,
                                )
                            }),
                            4 => s.vectorize(|| {
                                r4(
                                    s,
                                    rem,
                                    l_ptr,
                                    l_rs,
                                    w_ptr,
                                    w_rs,
                                    w_cs,
                                    neg_wj_over_ljj,
                                    alpha_wj_over_nljj,
                                    nljj_over_ljj,
                                )
                            }),
                            _ => unreachable!(),
                        };
                    }

                    r_idx += r_chunk;
                }
            }
        }
        Ok(())
    }
}

/// Performs a rank-r update in place, while clobbering the inputs.
///
/// Takes the Cholesky factor $L$ of a matrix $A$, i.e., $LL^* = A$, a matrix $W$ and a column
/// vector $\alpha$, which is interpreted as a diagonal matrix.
///
/// This function computes the Cholesky factor of $A + W\text{Diag}(\alpha)W^*$, and stores the
/// result in the storage of the original cholesky factors.
///
/// The matrix $W$ and the vector $\alpha$ are clobbered, meaning that the values they contain after
/// the function is called are unspecified.
#[track_caller]
pub fn rank_r_update_clobber<T: ComplexField>(
    cholesky_factor: MatMut<'_, T>,
    w: MatMut<'_, T>,
    alpha: ColMut<'_, T>,
) -> Result<(), CholeskyError> {
    let n = cholesky_factor.nrows();
    let k = w.ncols();

    fancy_assert!(cholesky_factor.ncols() == n);
    fancy_assert!(w.nrows() == n);
    fancy_assert!(alpha.nrows() == k);

    Arch::new().dispatch(RankRUpdate {
        l: cholesky_factor,
        w,
        alpha,
        r: &mut || k,
    })
}

/// Computes the size and alignment of required workspace for deleting the rows and columns from a
/// matrix, given its Cholesky decomposition.
#[track_caller]
pub fn delete_rows_and_cols_clobber_req<T: 'static>(
    dim: usize,
    number_of_rows_to_remove: usize,
) -> Result<StackReq, SizeOverflow> {
    let r = number_of_rows_to_remove;
    StackReq::try_all_of([temp_mat_req::<T>(dim, r)?, temp_mat_req::<T>(r, 1)?])
}

/// Deletes `r` rows and columns at the provided indices from the Cholesky factor.
///
/// Takes the Cholesky factor $L$ of a matrix $A$ of dimension `n`, and `r` indices, then computes
/// the Cholesky factor of $A$ with the provided rows and columns deleted from it.
///
/// The result is stored in the top left corner (with dimension `n - r`) of `cholesky_factor`.
///
/// The indices are clobbered, meaning that the values that the slice contains after the function
/// is called are unspecified.
#[track_caller]
pub fn delete_rows_and_cols_clobber<T: ComplexField>(
    cholesky_factor: MatMut<'_, T>,
    indices: &mut [usize],
    stack: DynStack<'_>,
) {
    let n = cholesky_factor.nrows();
    let r = indices.len();
    fancy_assert!(cholesky_factor.ncols() == n);
    fancy_assert!(indices.len() < n);

    if r == 0 {
        return;
    }

    indices.sort_unstable();
    for i in 0..r - 1 {
        fancy_assert!(indices[i + 1] > indices[i]);
    }
    fancy_assert!(indices[r - 1] < n);

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
                *alpha.rb_mut().ptr_in_bounds_at(k) = T::one();
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
                            cholesky_factor.rb().get_unchecked(i, j).clone();
                    }
                }
            }
        }
    });
    let mut cholesky_factor = cholesky_factor;
    delete_rows_and_cols_triangular(cholesky_factor.rb_mut(), indices);

    Arch::new()
        .dispatch(RankRUpdate {
            l: unsafe {
                cholesky_factor.submatrix_unchecked(first, first, n - first - r, n - first - r)
            },
            w,
            alpha,
            r: &mut rank_update_indices(first, indices),
        })
        .unwrap();
}

/// Computes the size and alignment of the required workspace for inserting the rows and columns at
/// the index in the Cholesky factor..
pub fn insert_rows_and_cols_clobber_req<T: 'static>(
    inserted_matrix_ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    cholesky_in_place_req::<T>(inserted_matrix_ncols, parallelism, Default::default())
}

/// Inserts `r` rows and columns at the provided index in the Cholesky factor.
///
/// Takes a matrix, `cholesky_factor_extended`, of dimension `n + r`, containing the Cholesky factor
/// $L$ of a matrix $A$ in its top left corner, of dimension `n`, and computes the Cholesky factor
/// of $A$ with the provided `inserted_matrix` inserted at the position starting at
/// `insertion_index`.
///
/// The inserted matrix is clobbered, meaning that the values it contains after the function
/// is called are unspecified.
#[track_caller]
pub fn insert_rows_and_cols_clobber<T: ComplexField>(
    cholesky_factor_extended: MatMut<'_, T>,
    insertion_index: usize,
    inserted_matrix: MatMut<'_, T>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) -> Result<(), CholeskyError> {
    let new_n = cholesky_factor_extended.nrows();
    let r = inserted_matrix.ncols();

    fancy_assert!(cholesky_factor_extended.nrows() == cholesky_factor_extended.ncols());
    fancy_assert!(cholesky_factor_extended.ncols() == new_n);
    fancy_assert!(r < new_n);
    let old_n = new_n - r;

    fancy_assert!(insertion_index <= old_n);

    if r == 0 {
        return Ok(());
    }

    let mut current_col = old_n;

    let mut ld = cholesky_factor_extended;

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

    let (l00, _, l_bot_left, ld_bot_right) = ld.split_at(insertion_index, insertion_index);
    let l00 = l00.into_const();

    let (_, mut l10, _, l20) = l_bot_left.split_at(r, 0);
    let (mut l11, _, mut l21, ld22) = ld_bot_right.split_at(r, r);

    let (_, mut a01, _, a_bottom) = inserted_matrix.split_at(insertion_index, 0);
    let (_, a11, _, a21) = a_bottom.split_at(r, 0);

    let mut stack = stack;

    solve::solve_lower_triangular_in_place(l00.rb(), Conj::No, a01.rb_mut(), Conj::No, parallelism);

    let a10 = a01.rb().transpose();

    for j in 0..insertion_index {
        for i in 0..r {
            unsafe {
                *l10.rb_mut().ptr_in_bounds_at_unchecked(i, j) = (*a10.get_unchecked(i, j)).conj();
            }
        }
    }

    for j in 0..r {
        for i in j..r {
            unsafe {
                *l11.rb_mut().ptr_in_bounds_at_unchecked(i, j) = *a11.rb().get_unchecked(i, j);
            }
        }
    }

    mul::triangular::matmul(
        l11.rb_mut(),
        BlockStructure::TriangularLower,
        Conj::No,
        l10.rb(),
        BlockStructure::Rectangular,
        Conj::No,
        a01.rb(),
        BlockStructure::Rectangular,
        Conj::No,
        Some(T::one()),
        -T::one(),
        parallelism,
    );

    cholesky_in_place(
        l11.rb_mut(),
        parallelism,
        stack.rb_mut(),
        Default::default(),
    )?;
    let l11 = l11.into_const();

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
        Conj::No,
        l20.rb(),
        Conj::No,
        a01.rb(),
        Conj::No,
        Some(T::one()),
        -T::one(),
        parallelism,
    );

    solve::solve_lower_triangular_in_place(
        l11,
        Conj::Yes,
        l21.rb_mut().transpose(),
        Conj::No,
        parallelism,
    );

    let mut alpha = a11.col(0);
    let mut w = a21;

    for j in 0..r {
        unsafe {
            *alpha.rb_mut().ptr_in_bounds_at_unchecked(j) = -T::one();

            for i in 0..rem {
                *w.rb_mut().ptr_in_bounds_at_unchecked(i, j) = l21.rb().get(i, j).clone();
            }
        }
    }

    rank_r_update_clobber(ld22, w, alpha)
}
