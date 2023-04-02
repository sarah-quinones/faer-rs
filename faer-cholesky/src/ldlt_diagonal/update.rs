use assert2::{assert as fancy_assert, debug_assert as fancy_debug_assert};
use core::{any::TypeId, mem::size_of};
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{
    mul, mul::triangular::BlockStructure, solve, temp_mat_req, temp_mat_uninit, ColMut,
    ComplexField, Conj, MatMut, Parallelism,
};
use pulp::Arch;
use reborrow::*;
use seq_macro::seq;

use crate::ldlt_diagonal::compute::{raw_cholesky_in_place, raw_cholesky_in_place_req};

macro_rules! generate {
    ($name: ident, $r: tt, $ty: ty, $tys: ty, $splat: ident, $mul_add: ident) => {
        unsafe fn $name(
            arch: pulp::Arch,
            n: usize,
            l_col: *mut $ty,
            w: *mut $ty,
            w_col_stride: isize,
            p_array: *const $ty,
            beta_array: *const $ty,
        ) {
            struct Impl {
                n: usize,
                l_col: *mut $ty,
                w: *mut $ty,
                w_col_stride: isize,
                p_array: *const $ty,
                beta_array: *const $ty,
            }
            impl pulp::WithSimd for Impl {
                type Output = ();

                #[inline(always)]
                fn with_simd<S: pulp::Simd>(self, simd: S) {
                    unsafe {
                        let Self { n, l_col, w, w_col_stride, p_array, beta_array } = self;

                        let l_col = l_col as *mut $ty;
                        let w = w as *mut $ty;
                        let p_array = p_array as *const $ty;
                        let beta_array = beta_array as *const $ty;
                        let lanes = size_of::<$tys>() / size_of::<$ty>();

                        let n_vec = n / lanes;
                        let n_rem = n % lanes;

                        seq!(I in 0..$r {
                            let p~I = -*p_array.add(I);
                            let beta~I = *beta_array.add(I);
                            let w_col~I = w.offset(I * w_col_stride);
                        });

                        // vectorized section
                        {
                            let l_col = l_col as *mut $tys;

                            seq!(I in 0..$r {
                                let p~I = simd.$splat(p~I);
                                let beta~I = simd.$splat(beta~I);
                                let w_col~I = w_col~I as *mut $tys;
                            });

                            for i in 0..n_vec {
                                let mut l = *l_col.add(i);
                                seq!(I in 0..$r {
                                    let mut w~I = *w_col~I.add(i);
                                });

                                seq!(I in 0..$r {
                                    w~I = simd.$mul_add(p~I, l, w~I);
                                    l = simd.$mul_add(beta~I, w~I, l);
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
                                    l = $ty::mul_add(beta~I, w~I, l);
                                });

                                l_col.add(i).write(l);
                                seq!(I in 0..$r {
                                    w_col~I.add(i).write(w~I);
                                });
                            }
                        }
                    }
                }
            }
            arch.dispatch(Impl { n, l_col, w, w_col_stride, p_array, beta_array })
        }
    };
}

macro_rules! generate_generic {
    ($name: ident, $r: tt) => {
        unsafe fn $name<T: ComplexField>(
            n: usize,
            l_col: *mut T,
            l_row_stride: isize,
            w: *mut T,
            w_row_stride: isize,
            w_col_stride: isize,
            p_array: *const T,
            beta_array: *const T,
        ) {
            seq!(I in 0..$r {
                let p~I = (*p_array.add(I)).neg();
                let beta~I = (*beta_array.add(I)).clone();
                let w_col~I = w.offset(I * w_col_stride);
            });

            for i in 0..n {
                let mut l = (*l_col.offset(i as isize * l_row_stride)).clone();
                seq!(I in 0..$r {
                    let mut w~I = (*w_col~I.offset(i as isize * w_row_stride)).clone();
                });

                seq!(I in 0..$r {
                    w~I = (p~I.mul(&l)).add(&w~I);
                    l = (beta~I.mul(&w~I)).add(&l);
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

impl<'a, T: ComplexField> RankRUpdate<'a, T> {
    fn run(self) {
        // On the Modification of LDLT Factorizations
        // By R. Fletcher and M. J. D. Powell
        // https://www.ams.org/journals/mcom/1974-28-128/S0025-5718-1974-0359297-1/S0025-5718-1974-0359297-1.pdf

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

        let arch = pulp::Arch::new();
        unsafe {
            for j in 0..n {
                let r = (*r)().min(k);

                let mut r_idx = 0;
                while r_idx < r {
                    let r_chunk = (r - r_idx).min(4);
                    let mut p_array = [T::zero(), T::zero(), T::zero(), T::zero()];
                    let mut beta_array = [T::zero(), T::zero(), T::zero(), T::zero()];

                    let mut dj = ld.rb().get_unchecked(j, j).clone();
                    for k in 0..r_chunk {
                        let p = p_array.get_unchecked_mut(k);
                        let beta = beta_array.get_unchecked_mut(k);
                        let alpha = alpha.rb_mut().get_unchecked(r_idx + k);

                        *p = w.rb().get_unchecked(j, r_idx + k).clone();
                        let alpha_conj_p = (*alpha).mul(&p.conj());
                        let new_dj = dj.add(&alpha_conj_p.mul(p));
                        *beta = alpha_conj_p.mul(&new_dj.inv());
                        *alpha = (*alpha).sub(&new_dj.mul(&beta.mul(&beta.conj())));

                        dj = new_dj;
                    }
                    *ld.rb_mut().get_unchecked(j, j) = dj;

                    let rem = n - j - 1;

                    let l_ptr = ld.rb_mut().ptr_at(j + 1, j);
                    let w_ptr = w.rb_mut().ptr_at(j + 1, r_idx);
                    let p = p_array.as_ptr();
                    let beta = beta_array.as_ptr();

                    if TypeId::of::<T>() == TypeId::of::<f64>() && l_rs == 1 && w_rs == 1 {
                        match r_chunk {
                            1 => rank_1_f64(
                                arch, rem, l_ptr as _, w_ptr as _, w_cs, p as _, beta as _,
                            ),
                            2 => rank_2_f64(
                                arch, rem, l_ptr as _, w_ptr as _, w_cs, p as _, beta as _,
                            ),
                            3 => rank_3_f64(
                                arch, rem, l_ptr as _, w_ptr as _, w_cs, p as _, beta as _,
                            ),
                            4 => rank_4_f64(
                                arch, rem, l_ptr as _, w_ptr as _, w_cs, p as _, beta as _,
                            ),
                            _ => unreachable!(),
                        };
                    } else if TypeId::of::<T>() == TypeId::of::<f32>() && l_rs == 1 && w_rs == 1 {
                        match r_chunk {
                            1 => rank_1_f32(
                                arch, rem, l_ptr as _, w_ptr as _, w_cs, p as _, beta as _,
                            ),
                            2 => rank_2_f32(
                                arch, rem, l_ptr as _, w_ptr as _, w_cs, p as _, beta as _,
                            ),
                            3 => rank_3_f32(
                                arch, rem, l_ptr as _, w_ptr as _, w_cs, p as _, beta as _,
                            ),
                            4 => rank_4_f32(
                                arch, rem, l_ptr as _, w_ptr as _, w_cs, p as _, beta as _,
                            ),
                            _ => unreachable!(),
                        };
                    } else {
                        match r_chunk {
                            1 => r1(rem, l_ptr, l_rs, w_ptr, w_rs, w_cs, p, beta),
                            2 => r2(rem, l_ptr, l_rs, w_ptr, w_rs, w_cs, p, beta),
                            3 => r3(rem, l_ptr, l_rs, w_ptr, w_rs, w_cs, p, beta),
                            4 => r4(rem, l_ptr, l_rs, w_ptr, w_rs, w_cs, p, beta),
                            _ => unreachable!(),
                        };
                    }

                    r_idx += r_chunk;
                }
            }
        }
    }
}

/// Performs a rank-r update in place, while clobbering the inputs.
///
/// Takes the Cholesky factors $L$ and $D$ of a matrix $A$, meaning that $LDL^H = A$,
/// a matrix $W$ and a column vector $\alpha$, which is interpreted as a diagonal matrix.
///
/// This function computes the cholesky factors of $A + W\text{Diag}(\alpha)W^H$, and stores the
/// result in the storage of the original cholesky factors.
///
/// The matrix $W$ and the vector $\alpha$ are clobbered, meaning that the values they contain after
/// the function is called are unspecified.
#[track_caller]
pub fn rank_r_update_clobber<T: ComplexField>(
    cholesky_factors: MatMut<'_, T>,
    w: MatMut<'_, T>,
    alpha: ColMut<'_, T>,
) {
    let n = cholesky_factors.nrows();
    let k = w.ncols();

    fancy_assert!(cholesky_factors.ncols() == n);
    fancy_assert!(w.nrows() == n);
    fancy_assert!(alpha.nrows() == k);

    RankRUpdate {
        ld: cholesky_factors,
        w,
        alpha,
        r: &mut || k,
    }
    .run();
}

pub(crate) fn delete_rows_and_cols_triangular<T: Clone>(mat: MatMut<'_, T>, idx: &[usize]) {
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

pub(crate) fn rank_update_indices(
    start_col: usize,
    indices: &[usize],
) -> impl FnMut() -> usize + '_ {
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
/// Takes the Cholesky factors $L$ and $D$ of a matrix $A$ of dimension `n`, and `r` indices, then
/// computes the Cholesky factor of $A$ with the provided rows and columns deleted from it.
///
/// The result is stored in the top left corner (with dimension `n - r`) of `cholesky_factor`.
///
/// The indices are clobbered, meaning that the values that the slice contains after the function
/// is called are unspecified.
#[track_caller]
pub fn delete_rows_and_cols_clobber<T: ComplexField>(
    cholesky_factors: MatMut<'_, T>,
    indices: &mut [usize],
    stack: DynStack<'_>,
) {
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

    let first = indices[0];

    let (mut w, stack) = unsafe { temp_mat_uninit::<T>(n - first - r, r, stack) };
    let (mut alpha, _) = unsafe { temp_mat_uninit::<T>(r, 1, stack) };
    let mut w = w.as_mut();
    let alpha = alpha.as_mut();
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

    RankRUpdate {
        ld: unsafe {
            cholesky_factors.submatrix_unchecked(first, first, n - first - r, n - first - r)
        },
        w,
        alpha,
        r: &mut rank_update_indices(first, indices),
    }
    .run();
}

/// Computes the size and alignment of the required workspace for inserting the rows and columns at
/// the index in the Cholesky factor..
pub fn insert_rows_and_cols_clobber_req<T: 'static>(
    inserted_matrix_ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    raw_cholesky_in_place_req::<T>(inserted_matrix_ncols, parallelism, Default::default())
}

/// Inserts `r` rows and columns at the provided index in the Cholesky factor.
///
/// Takes a matrix, `cholesky_factor_extended`, of dimension `n + r`, containing the Cholesky
/// factors $L$ and $D$ of a matrix $A$ in its top left corner, of dimension `n`, and computes the
/// Cholesky factor of $A$ with the provided `inserted_matrix` inserted at the position starting at
/// `insertion_index`.
///
/// The inserted matrix is clobbered, meaning that the values it contains after the function
/// is called are unspecified.
#[track_caller]
pub fn insert_rows_and_cols_clobber<T: ComplexField>(
    cholesky_factors_extended: MatMut<'_, T>,
    insertion_index: usize,
    inserted_matrix: MatMut<'_, T>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
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

    solve::solve_unit_lower_triangular_in_place(
        ld00.rb(),
        Conj::No,
        a01.rb_mut(),
        Conj::No,
        parallelism,
    );

    let a01 = a01.rb();

    for j in 0..insertion_index {
        let d0_inv = unsafe { d0.get_unchecked(j) }.inv();
        for i in 0..r {
            unsafe {
                *l10.rb_mut().ptr_in_bounds_at_unchecked(i, j) =
                    a01.get_unchecked(j, i).conj().mul(&d0_inv);
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
        Conj::No,
        l10.rb(),
        BlockStructure::Rectangular,
        Conj::No,
        a01.rb(),
        BlockStructure::Rectangular,
        Conj::No,
        Some(T::one()),
        T::one().neg(),
        parallelism,
    );

    raw_cholesky_in_place(
        ld11.rb_mut(),
        parallelism,
        stack.rb_mut(),
        Default::default(),
    );
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
        Conj::No,
        l20.rb(),
        Conj::No,
        a01.rb(),
        Conj::No,
        Some(T::one()),
        T::one().neg(),
        parallelism,
    );

    solve::solve_unit_lower_triangular_in_place(
        ld11,
        Conj::Yes,
        l21.rb_mut().transpose(),
        Conj::No,
        parallelism,
    );

    let d1 = ld11.into_const().diagonal();

    for j in 0..r {
        unsafe {
            let d1_inv = (*d1.get_unchecked(j)).inv();
            for i in 0..rem {
                let dst = l21.rb_mut().get_unchecked(i, j);
                *dst = dst.mul(&d1_inv);
            }
        }
    }

    let mut alpha = unsafe { a11.col_unchecked(0) };
    let mut w = a21;

    for j in 0..r {
        unsafe {
            *alpha.rb_mut().ptr_in_bounds_at_unchecked(j) = ld11.rb().get_unchecked(j, j).neg();

            for i in 0..rem {
                *w.rb_mut().ptr_in_bounds_at_unchecked(i, j) = l21.rb().get_unchecked(i, j).neg();
            }
        }
    }

    rank_r_update_clobber(ld22, w, alpha);
}
