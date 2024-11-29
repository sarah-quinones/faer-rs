use super::temp_mat_scratch;
use crate::{
    col::ColRef,
    internal_prelude::*,
    mat::{MatMut, MatRef},
    row::RowRef,
    utils::{bound::Dim, simd::SimdCtx},
    Conj, ContiguousFwd, Par, Shape, Stride,
};
use dyn_stack::{DynStack, GlobalMemBuffer};
use equator::assert;
use faer_macros::math;
use faer_traits::{ByRef, ComplexField, Conjugate};
use generativity::make_guard;
use pulp::Simd;
use reborrow::*;

pub mod triangular;

pub mod dot {
    use super::*;
    use faer_traits::SimdArch;

    pub fn inner_prod<K: Shape, T: ComplexField>(
        lhs: RowRef<T, K>,
        conj_lhs: Conj,
        rhs: ColRef<T, K>,
        conj_rhs: Conj,
    ) -> T {
        #[math]
        pub fn imp<'K, T: ComplexField>(
            lhs: RowRef<T, Dim<'K>>,
            conj_lhs: Conj,
            rhs: ColRef<T, Dim<'K>>,
            conj_rhs: Conj,
        ) -> T {
            if const { T::SIMD_CAPABILITIES.is_simd() } {
                if let (Some(lhs), Some(rhs)) = (lhs.try_as_row_major(), rhs.try_as_col_major()) {
                    inner_prod_slice::<T>(lhs.ncols(), lhs.transpose(), conj_lhs, rhs, conj_rhs)
                } else {
                    inner_prod_schoolbook(lhs, conj_lhs, rhs, conj_rhs)
                }
            } else {
                inner_prod_schoolbook(lhs, conj_lhs, rhs, conj_rhs)
            }
        }

        with_dim!(K, lhs.ncols().unbound());

        imp(lhs.as_col_shape(K), conj_lhs, rhs.as_row_shape(K), conj_rhs)
    }

    #[inline(always)]
    #[math]
    fn inner_prod_slice<'K, T: ComplexField>(
        len: Dim<'K>,
        lhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
        conj_lhs: Conj,
        rhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
        conj_rhs: Conj,
    ) -> T {
        struct Impl<'a, 'K, T: ComplexField> {
            len: Dim<'K>,
            lhs: ColRef<'a, T, Dim<'K>, ContiguousFwd>,
            conj_lhs: Conj,
            rhs: ColRef<'a, T, Dim<'K>, ContiguousFwd>,
            conj_rhs: Conj,
        }
        impl<'a, 'K, T: ComplexField> pulp::WithSimd for Impl<'_, '_, T> {
            type Output = T;

            #[inline(always)]
            fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                let Self {
                    len,
                    lhs,
                    conj_lhs,
                    rhs,
                    conj_rhs,
                } = self;

                let simd = SimdCtx::new(T::simd_ctx(simd), len);

                let mut tmp = if conj_lhs == conj_rhs {
                    inner_prod_no_conj_simd::<T, S>(simd, lhs, rhs)
                } else {
                    inner_prod_conj_lhs_simd::<T, S>(simd, lhs, rhs)
                };

                if conj_rhs == Conj::Yes {
                    tmp = conj(tmp);
                }
                tmp
            }
        }

        T::Arch::default().dispatch(Impl::<_> {
            len,
            lhs,
            rhs,
            conj_lhs,
            conj_rhs,
        })
    }

    #[inline(always)]
    pub fn inner_prod_no_conj_simd<'K, T: ComplexField, S: Simd>(
        simd: SimdCtx<'K, T, S>,
        lhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
        rhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
    ) -> T {
        let mut acc0 = simd.zero();
        let mut acc1 = simd.zero();
        let mut acc2 = simd.zero();
        let mut acc3 = simd.zero();

        let (head, idx4, idx, tail) = simd.batch_indices::<4>();

        if let Some(i0) = head {
            let l0 = simd.read(lhs, i0);
            let r0 = simd.read(rhs, i0);

            acc0 = simd.mul_add(l0, r0, acc0);
        }
        for [i0, i1, i2, i3] in idx4 {
            let l0 = simd.read(lhs, i0);
            let l1 = simd.read(lhs, i1);
            let l2 = simd.read(lhs, i2);
            let l3 = simd.read(lhs, i3);

            let r0 = simd.read(rhs, i0);
            let r1 = simd.read(rhs, i1);
            let r2 = simd.read(rhs, i2);
            let r3 = simd.read(rhs, i3);

            acc0 = simd.mul_add(l0, r0, acc0);
            acc1 = simd.mul_add(l1, r1, acc1);
            acc2 = simd.mul_add(l2, r2, acc2);
            acc3 = simd.mul_add(l3, r3, acc3);
        }
        for i0 in idx {
            let l0 = simd.read(lhs, i0);
            let r0 = simd.read(rhs, i0);

            acc0 = simd.mul_add(l0, r0, acc0);
        }
        if let Some(i0) = tail {
            let l0 = simd.read(lhs, i0);
            let r0 = simd.read(rhs, i0);

            acc0 = simd.mul_add(l0, r0, acc0);
        }
        acc0 = simd.add(acc0, acc1);
        acc2 = simd.add(acc2, acc3);
        acc0 = simd.add(acc0, acc2);

        simd.reduce_sum(acc0)
    }

    #[inline(always)]
    pub fn inner_prod_conj_lhs_simd<'K, T: ComplexField, S: Simd>(
        simd: SimdCtx<'K, T, S>,
        lhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
        rhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
    ) -> T {
        let mut acc0 = simd.zero();
        let mut acc1 = simd.zero();
        let mut acc2 = simd.zero();
        let mut acc3 = simd.zero();

        let (head, idx4, idx, tail) = simd.batch_indices::<4>();

        if let Some(i0) = head {
            let l0 = simd.read(lhs, i0);
            let r0 = simd.read(rhs, i0);

            acc0 = simd.conj_mul_add(l0, r0, acc0);
        }
        for [i0, i1, i2, i3] in idx4 {
            let l0 = simd.read(lhs, i0);
            let l1 = simd.read(lhs, i1);
            let l2 = simd.read(lhs, i2);
            let l3 = simd.read(lhs, i3);

            let r0 = simd.read(rhs, i0);
            let r1 = simd.read(rhs, i1);
            let r2 = simd.read(rhs, i2);
            let r3 = simd.read(rhs, i3);

            acc0 = simd.conj_mul_add(l0, r0, acc0);
            acc1 = simd.conj_mul_add(l1, r1, acc1);
            acc2 = simd.conj_mul_add(l2, r2, acc2);
            acc3 = simd.conj_mul_add(l3, r3, acc3);
        }
        for i0 in idx {
            let l0 = simd.read(lhs, i0);
            let r0 = simd.read(rhs, i0);

            acc0 = simd.conj_mul_add(l0, r0, acc0);
        }
        if let Some(i0) = tail {
            let l0 = simd.read(lhs, i0);
            let r0 = simd.read(rhs, i0);

            acc0 = simd.conj_mul_add(l0, r0, acc0);
        }
        acc0 = simd.add(acc0, acc1);
        acc2 = simd.add(acc2, acc3);
        acc0 = simd.add(acc0, acc2);

        simd.reduce_sum(acc0)
    }

    #[math]
    pub fn inner_prod_schoolbook<'K, T: ComplexField>(
        lhs: RowRef<'_, T, Dim<'K>>,
        conj_lhs: Conj,
        rhs: ColRef<'_, T, Dim<'K>>,
        conj_rhs: Conj,
    ) -> T {
        let mut acc = zero();

        for k in lhs.ncols().indices() {
            if const { T::IS_REAL } {
                acc = lhs[k] * rhs[k] + acc;
            } else {
                match (conj_lhs, conj_rhs) {
                    (Conj::No, Conj::No) => {
                        acc = lhs[k] * rhs[k] + acc;
                    }
                    (Conj::No, Conj::Yes) => {
                        acc = lhs[k] * conj(rhs[k]) + acc;
                    }
                    (Conj::Yes, Conj::No) => {
                        acc = conj(lhs[k]) * rhs[k] + acc;
                    }
                    (Conj::Yes, Conj::Yes) => {
                        acc = conj(lhs[k] * rhs[k]) + acc;
                    }
                }
            }
        }

        acc
    }
}

mod matvec_rowmajor {
    use super::*;
    use crate::col::ColMut;
    use faer_traits::SimdArch;

    #[math]
    pub fn matvec<'M, 'K, T: ComplexField>(
        dst: ColMut<'_, T, Dim<'M>>,
        beta: Accum,
        lhs: MatRef<'_, T, Dim<'M>, Dim<'K>, isize, ContiguousFwd>,
        conj_lhs: Conj,
        rhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
        conj_rhs: Conj,
        alpha: &T,
        par: Par,
    ) {
        core::assert!(const { T::SIMD_CAPABILITIES.is_simd() });

        match par {
            Par::Seq => {
                pub struct Impl<'a, 'M, 'K, T: ComplexField> {
                    dst: ColMut<'a, T, Dim<'M>>,
                    beta: Accum,
                    lhs: MatRef<'a, T, Dim<'M>, Dim<'K>, isize, ContiguousFwd>,
                    conj_lhs: Conj,
                    rhs: ColRef<'a, T, Dim<'K>, ContiguousFwd>,
                    conj_rhs: Conj,
                    alpha: &'a T,
                }

                impl<'a, 'M, 'K, T: ComplexField> pulp::WithSimd for Impl<'a, 'M, 'K, T> {
                    type Output = ();

                    #[inline(always)]
                    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                        let Self {
                            dst,
                            beta,
                            lhs,
                            conj_lhs,
                            rhs,
                            conj_rhs,
                            alpha,
                        } = self;
                        let simd = T::simd_ctx(simd);
                        let mut dst = dst;

                        let K = lhs.ncols();
                        let simd = SimdCtx::new(simd, K);
                        for i in lhs.nrows().indices() {
                            let dst = dst.rb_mut().at_mut(i);
                            let lhs = lhs.row(i);
                            let rhs = rhs;
                            let mut tmp = if conj_lhs == conj_rhs {
                                dot::inner_prod_no_conj_simd::<T, S>(simd, lhs.transpose(), rhs)
                            } else {
                                dot::inner_prod_conj_lhs_simd::<T, S>(simd, lhs.transpose(), rhs)
                            };

                            if conj_rhs == Conj::Yes {
                                tmp = conj(tmp);
                            }
                            tmp = *alpha * tmp;
                            if let Accum::Add = beta {
                                tmp = *dst + tmp;
                            }
                            *dst = tmp;
                        }
                    }
                }

                T::Arch::default().dispatch(Impl {
                    dst,
                    beta,
                    lhs,
                    conj_lhs,
                    rhs,
                    conj_rhs,
                    alpha,
                });
            }
            #[cfg(feature = "rayon")]
            Par::Rayon(nthreads) => {
                let nthreads = nthreads.get();

                use rayon::prelude::*;
                dst.par_partition_mut(nthreads)
                    .zip_eq(lhs.par_row_partition(nthreads))
                    .for_each(|(dst, lhs)| {
                        make_guard!(M);
                        let nrows = dst.nrows().bind(M);
                        let dst = dst.as_row_shape_mut(nrows);
                        let lhs = lhs.as_row_shape(nrows);

                        matvec(dst, beta, lhs, conj_lhs, rhs, conj_rhs, alpha, Par::Seq);
                    })
            }
        }
    }
}

mod matvec_colmajor {
    use super::*;
    use crate::{
        col::ColMut, linalg::temp_mat_uninit, mat::AsMatMut, unzipped, utils::bound::IdxInc, zipped,
    };
    use faer_traits::SimdArch;

    #[math]
    pub fn matvec<'M, 'K, T: ComplexField>(
        dst: ColMut<'_, T, Dim<'M>, ContiguousFwd>,
        beta: Accum,
        lhs: MatRef<'_, T, Dim<'M>, Dim<'K>, ContiguousFwd, isize>,
        conj_lhs: Conj,
        rhs: ColRef<'_, T, Dim<'K>>,
        conj_rhs: Conj,
        alpha: &T,
        par: Par,
    ) {
        core::assert!(const { T::SIMD_CAPABILITIES.is_simd() });

        match par {
            Par::Seq => {
                pub struct Impl<'a, 'M, 'K, T: ComplexField> {
                    dst: ColMut<'a, T, Dim<'M>, ContiguousFwd>,
                    beta: Accum,
                    lhs: MatRef<'a, T, Dim<'M>, Dim<'K>, ContiguousFwd, isize>,
                    conj_lhs: Conj,
                    rhs: ColRef<'a, T, Dim<'K>>,
                    conj_rhs: Conj,
                    alpha: &'a T,
                }

                impl<'a, 'M, 'K, T: ComplexField> pulp::WithSimd for Impl<'a, 'M, 'K, T> {
                    type Output = ();

                    #[inline(always)]
                    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                        let Self {
                            dst,
                            beta,
                            lhs,
                            conj_lhs,
                            rhs,
                            conj_rhs,
                            alpha,
                        } = self;

                        let simd = T::simd_ctx(simd);

                        let M = lhs.nrows();
                        let simd = SimdCtx::<T, S>::new(simd, M);
                        let (head, body, tail) = simd.indices();

                        let mut dst = dst;
                        match beta {
                            Accum::Add => {}
                            Accum::Replace => {
                                let mut dst = dst.rb_mut();
                                if let Some(i) = head {
                                    simd.write(dst.rb_mut(), i, simd.zero());
                                }
                                for i in body.clone() {
                                    simd.write(dst.rb_mut(), i, simd.zero());
                                }
                                if let Some(i) = tail {
                                    simd.write(dst.rb_mut(), i, simd.zero());
                                }
                            }
                        }

                        for j in lhs.ncols().indices() {
                            let mut dst = dst.rb_mut();
                            let lhs = lhs.col(j);
                            let rhs = rhs.at(j);
                            let rhs = if conj_rhs == Conj::Yes {
                                conj(*rhs)
                            } else {
                                copy(*rhs)
                            };
                            let rhs = rhs * *alpha;

                            let vrhs = simd.splat(&rhs);
                            if conj_lhs == Conj::Yes {
                                if let Some(i) = head {
                                    let y = simd.read(dst.rb(), i);
                                    let x = simd.read(lhs, i);
                                    simd.write(dst.rb_mut(), i, simd.conj_mul_add(x, vrhs, y));
                                }
                                for i in body.clone() {
                                    let y = simd.read(dst.rb(), i);
                                    let x = simd.read(lhs, i);
                                    simd.write(dst.rb_mut(), i, simd.conj_mul_add(x, vrhs, y));
                                }
                                if let Some(i) = tail {
                                    let y = simd.read(dst.rb(), i);
                                    let x = simd.read(lhs, i);
                                    simd.write(dst.rb_mut(), i, simd.conj_mul_add(x, vrhs, y));
                                }
                            } else {
                                if let Some(i) = head {
                                    let y = simd.read(dst.rb(), i);
                                    let x = simd.read(lhs, i);
                                    simd.write(dst.rb_mut(), i, simd.mul_add(x, vrhs, y));
                                }
                                for i in body.clone() {
                                    let y = simd.read(dst.rb(), i);
                                    let x = simd.read(lhs, i);
                                    simd.write(dst.rb_mut(), i, simd.mul_add(x, vrhs, y));
                                }
                                if let Some(i) = tail {
                                    let y = simd.read(dst.rb(), i);
                                    let x = simd.read(lhs, i);
                                    simd.write(dst.rb_mut(), i, simd.mul_add(x, vrhs, y));
                                }
                            }
                        }
                    }
                }

                T::Arch::default().dispatch(Impl {
                    dst,
                    lhs,
                    conj_lhs,
                    rhs,
                    conj_rhs,
                    beta,
                    alpha,
                })
            }
            #[cfg(feature = "rayon")]
            Par::Rayon(nthreads) => {
                use rayon::prelude::*;
                let nthreads = nthreads.get();
                let mut mem = GlobalMemBuffer::new(
                    temp_mat_scratch::<T>(dst.nrows().unbound(), nthreads).unwrap(),
                );
                let stack = DynStack::new(&mut mem);

                let (mut tmp, _) =
                    unsafe { temp_mat_uninit::<T, _, _>(dst.nrows(), nthreads, stack) };
                let mut tmp = tmp.as_mat_mut().try_as_col_major_mut().unwrap();

                let mut dst = dst;
                make_guard!(Z);
                let Z = 0usize.bind(Z);
                let z = IdxInc::new_checked(0, lhs.ncols());

                tmp.rb_mut()
                    .par_col_iter_mut()
                    .zip_eq(lhs.par_col_partition(nthreads))
                    .zip_eq(rhs.par_partition(nthreads))
                    .for_each(|((dst, lhs), rhs)| {
                        make_guard!(K);
                        let K = lhs.ncols().bind(K);
                        let lhs = lhs.as_col_shape(K);
                        let rhs = rhs.as_row_shape(K);

                        matvec(
                            dst,
                            Accum::Replace,
                            lhs,
                            conj_lhs,
                            rhs,
                            conj_rhs,
                            alpha,
                            Par::Seq,
                        );
                    });

                matvec(
                    dst.rb_mut(),
                    beta,
                    lhs.subcols(z, Z),
                    conj_lhs,
                    rhs.subrows(z, Z),
                    conj_rhs,
                    &zero(),
                    Par::Seq,
                );
                for j in 0..nthreads {
                    zipped!(dst.rb_mut(), tmp.rb().col(j))
                        .for_each(|unzipped!(dst, src)| *dst = *dst + *src)
                }
            }
        }
    }
}

#[math]
fn matmul_imp<'M, 'N, 'K, T: ComplexField>(
    dst: MatMut<'_, T, Dim<'M>, Dim<'N>>,
    beta: Accum,
    lhs: MatRef<'_, T, Dim<'M>, Dim<'K>>,
    conj_lhs: Conj,
    rhs: MatRef<'_, T, Dim<'K>, Dim<'N>>,
    conj_rhs: Conj,
    alpha: &T,
    par: Par,
) {
    let mut dst = dst;

    let M = dst.nrows();
    let N = dst.ncols();
    let K = lhs.ncols();

    let mut lhs = lhs;
    let mut rhs = rhs;

    if const { T::SIMD_CAPABILITIES.is_simd() } {
        if dst.row_stride() < 0 {
            dst = dst.reverse_rows_mut();
            lhs = lhs.reverse_rows();
        }
        if dst.col_stride() < 0 {
            dst = dst.reverse_cols_mut();
            rhs = rhs.reverse_cols();
        }

        if dst.ncols().unbound() == 1 {
            let first = dst.ncols().check(0);
            if let (Some(dst), Some(lhs)) =
                (dst.rb_mut().try_as_col_major_mut(), lhs.try_as_col_major())
            {
                matvec_colmajor::matvec(
                    dst.col_mut(first),
                    beta,
                    lhs,
                    conj_lhs,
                    rhs.col(first),
                    conj_rhs,
                    alpha,
                    par,
                );
                return;
            }

            if let (Some(rhs), Some(lhs)) = (rhs.try_as_col_major(), lhs.try_as_row_major()) {
                matvec_rowmajor::matvec(
                    dst.col_mut(first),
                    beta,
                    lhs,
                    conj_lhs,
                    rhs.col(first),
                    conj_rhs,
                    alpha,
                    par,
                );
                return;
            }
        }
        if dst.nrows().unbound() == 1 {
            let mut dst = dst.rb_mut().transpose_mut();
            let (rhs, lhs) = (lhs.transpose(), rhs.transpose());
            let (conj_rhs, conj_lhs) = (conj_lhs, conj_rhs);

            let first = dst.ncols().check(0);
            if let (Some(dst), Some(lhs)) =
                (dst.rb_mut().try_as_col_major_mut(), lhs.try_as_col_major())
            {
                matvec_colmajor::matvec(
                    dst.col_mut(first),
                    beta,
                    lhs,
                    conj_lhs,
                    rhs.col(first),
                    conj_rhs,
                    alpha,
                    par,
                );
                return;
            }

            if let (Some(rhs), Some(lhs)) = (rhs.try_as_col_major(), lhs.try_as_row_major()) {
                matvec_rowmajor::matvec(
                    dst.col_mut(first),
                    beta,
                    lhs,
                    conj_lhs,
                    rhs.col(first),
                    conj_rhs,
                    alpha,
                    par,
                );
                return;
            }
        }
        macro_rules! gemm_call {
            ($ty: ty) => {
                unsafe {
                    let dst = core::mem::transmute_copy::<
                        MatMut<'_, T, Dim<'M>, Dim<'N>>,
                        MatMut<'_, $ty, Dim<'M>, Dim<'N>>,
                    >(&dst);
                    let lhs = core::mem::transmute_copy::<
                        MatRef<'_, T, Dim<'M>, Dim<'K>>,
                        MatRef<'_, $ty, Dim<'M>, Dim<'K>>,
                    >(&lhs);
                    let rhs = core::mem::transmute_copy::<
                        MatRef<'_, T, Dim<'K>, Dim<'N>>,
                        MatRef<'_, $ty, Dim<'K>, Dim<'N>>,
                    >(&rhs);
                    let alpha = core::mem::transmute_copy::<&T, &$ty>(&alpha);

                    gemm::gemm(
                        M.unbound(),
                        N.unbound(),
                        K.unbound(),
                        dst.as_ptr_mut(),
                        dst.col_stride(),
                        dst.row_stride(),
                        beta != Accum::Replace,
                        lhs.as_ptr(),
                        lhs.col_stride(),
                        lhs.row_stride(),
                        rhs.as_ptr(),
                        rhs.col_stride(),
                        rhs.row_stride(),
                        match beta {
                            Accum::Replace => core::mem::zeroed(),
                            Accum::Add => 1.0.into(),
                        },
                        *alpha,
                        false,
                        conj_lhs == Conj::Yes,
                        conj_rhs == Conj::Yes,
                        match par {
                            Par::Seq => gemm::Parallelism::None,
                            #[cfg(feature = "rayon")]
                            Par::Rayon(nthreads) => gemm::Parallelism::Rayon(nthreads.get()),
                        },
                    )
                };
                return;
            };
        }

        if const { T::IS_NATIVE_F64 } {
            gemm_call!(f64);
        }
        if const { T::IS_NATIVE_C64 } {
            gemm_call!(num_complex::Complex<f64>);
        }
        if const { T::IS_NATIVE_F32 } {
            gemm_call!(f32);
        }
        if const { T::IS_NATIVE_C32 } {
            gemm_call!(num_complex::Complex<f32>);
        }
    }

    match par {
        Par::Seq => {
            for j in dst.ncols().indices() {
                for i in dst.nrows().indices() {
                    let dst = dst.rb_mut().at_mut(i, j);

                    let mut acc =
                        dot::inner_prod_schoolbook(lhs.row(i), conj_lhs, rhs.col(j), conj_rhs);
                    acc = *alpha * acc;
                    if let Accum::Add = beta {
                        acc = *dst + acc;
                    }
                    *dst = acc;
                }
            }
        }
        #[cfg(feature = "rayon")]
        Par::Rayon(nthreads) => {
            use rayon::prelude::*;
            let nthreads = nthreads.get();

            let m = *dst.nrows();
            let n = *dst.ncols();
            let task_count = m * n;
            let task_per_thread = task_count.div_ceil(nthreads);

            let dst = dst.rb();
            (0..nthreads).into_par_iter().for_each(|tid| {
                let task_idx = tid * task_per_thread;
                if task_idx >= task_count {
                    return;
                }
                let ntasks = Ord::min(task_per_thread, task_count - task_idx);

                for ij in 0..ntasks {
                    let ij = task_idx + ij;
                    let i = dst.nrows().check(ij % m);
                    let j = dst.ncols().check(ij / m);

                    let dst = unsafe { dst.const_cast().at_mut(i, j) };

                    let mut acc =
                        dot::inner_prod_schoolbook(lhs.row(i), conj_lhs, rhs.col(j), conj_rhs);
                    acc = *alpha * acc;

                    if let Accum::Add = beta {
                        acc = *dst + acc;
                    }
                    *dst = acc;
                }
            });
        }
    }
}

#[track_caller]
fn precondition<M: Shape, N: Shape, K: Shape>(
    dst_nrows: M,
    dst_ncols: N,
    lhs_nrows: M,
    lhs_ncols: K,
    rhs_nrows: K,
    rhs_ncols: N,
) {
    assert!(all(
        dst_nrows == lhs_nrows,
        dst_ncols == rhs_ncols,
        lhs_ncols == rhs_nrows,
    ));
}

#[track_caller]
#[inline]
pub fn matmul<
    T: ComplexField,
    LhsT: Conjugate<Canonical = T>,
    RhsT: Conjugate<Canonical = T>,
    M: Shape,
    N: Shape,
    K: Shape,
>(
    dst: MatMut<'_, T, M, N, impl Stride, impl Stride>,
    beta: Accum,
    lhs: MatRef<'_, LhsT, M, K, impl Stride, impl Stride>,
    rhs: MatRef<'_, RhsT, K, N, impl Stride, impl Stride>,
    alpha: T,
    par: Par,
) {
    precondition(
        dst.nrows(),
        dst.ncols(),
        lhs.nrows(),
        lhs.ncols(),
        rhs.nrows(),
        rhs.ncols(),
    );

    make_guard!(M);
    make_guard!(N);
    make_guard!(K);
    let M = dst.nrows().bind(M);
    let N = dst.ncols().bind(N);
    let K = lhs.ncols().bind(K);

    matmul_imp(
        dst.as_dyn_stride_mut().as_shape_mut(M, N),
        beta,
        lhs.as_dyn_stride().canonical().as_shape(M, K),
        const { Conj::get::<LhsT>() },
        rhs.as_dyn_stride().canonical().as_shape(K, N),
        const { Conj::get::<RhsT>() },
        &alpha,
        par,
    );
}

#[track_caller]
#[inline]
pub fn matmul_with_conj<T: ComplexField, M: Shape, N: Shape, K: Shape>(
    dst: MatMut<'_, T, M, N, impl Stride, impl Stride>,
    beta: Accum,
    lhs: MatRef<'_, T, M, K, impl Stride, impl Stride>,
    conj_lhs: Conj,
    rhs: MatRef<'_, T, K, N, impl Stride, impl Stride>,
    conj_rhs: Conj,
    alpha: T,
    par: Par,
) {
    precondition(
        dst.nrows(),
        dst.ncols(),
        lhs.nrows(),
        lhs.ncols(),
        rhs.nrows(),
        rhs.ncols(),
    );

    make_guard!(M);
    make_guard!(N);
    make_guard!(K);
    let M = dst.nrows().bind(M);
    let N = dst.ncols().bind(N);
    let K = lhs.ncols().bind(K);

    matmul_imp(
        dst.as_dyn_stride_mut().as_shape_mut(M, N),
        beta,
        lhs.as_dyn_stride().canonical().as_shape(M, K),
        conj_lhs,
        rhs.as_dyn_stride().canonical().as_shape(K, N),
        conj_rhs,
        &alpha,
        par,
    );
}

#[cfg(test)]
mod tests {
    use crate::c32;
    use std::num::NonZeroUsize;

    use super::{
        triangular::{BlockStructure, DiagonalKind},
        *,
    };
    use crate::{
        assert,
        mat::{Mat, MatMut, MatRef},
        stats::prelude::*,
    };

    #[test]
    #[ignore = "takes too long"]
    fn test_matmul() {
        let rng = &mut StdRng::seed_from_u64(0);

        if option_env!("CI") == Some("true") {
            // too big for CI
            return;
        }

        let betas = [Accum::Replace, Accum::Add];

        #[cfg(not(miri))]
        let bools = [false, true];
        #[cfg(not(miri))]
        let alphas = [c32::ONE, c32::ZERO, c32::new(21.04, -12.13)];
        #[cfg(not(miri))]
        let par = [Par::Seq, Par::Rayon(NonZeroUsize::new(4).unwrap())];
        #[cfg(not(miri))]
        let conjs = [Conj::Yes, Conj::No];

        #[cfg(miri)]
        let bools = [true];
        #[cfg(miri)]
        let alphas = [c32::new(0.3218, -1.217489)];
        #[cfg(miri)]
        let par = [Par::Seq];
        #[cfg(miri)]
        let conjs = [Conj::Yes];

        let big0 = 127;
        let big1 = 128;
        let big2 = 129;

        let mid0 = 15;
        let mid1 = 16;
        let mid2 = 17;
        for (m, n, k) in [
            (big0, big1, 5),
            (big1, big0, 5),
            (big0, big2, 5),
            (big2, big0, 5),
            (mid0, mid0, 5),
            (mid1, mid1, 5),
            (mid2, mid2, 5),
            (mid0, mid1, 5),
            (mid1, mid0, 5),
            (mid0, mid2, 5),
            (mid2, mid0, 5),
            (mid0, 1, 1),
            (1, mid0, 1),
            (1, 1, mid0),
            (1, mid0, mid0),
            (mid0, 1, mid0),
            (mid0, mid0, 1),
            (1, 1, 1),
        ] {
            let distribution = ComplexDistribution::new(StandardNormal, StandardNormal);
            let a = CwiseMatDistribution {
                nrows: m,
                ncols: k,
                dist: distribution,
            }
            .rand::<Mat<c32>>(rng);
            let b = CwiseMatDistribution {
                nrows: k,
                ncols: n,
                dist: distribution,
            }
            .rand::<Mat<c32>>(rng);
            let mut acc_init = CwiseMatDistribution {
                nrows: m,
                ncols: n,
                dist: distribution,
            }
            .rand::<Mat<c32>>(rng);

            let a = a.as_ref();
            let b = b.as_ref();

            for reverse_acc_cols in bools {
                for reverse_acc_rows in bools {
                    for reverse_b_cols in bools {
                        for reverse_b_rows in bools {
                            for reverse_a_cols in bools {
                                for reverse_a_rows in bools {
                                    for a_colmajor in bools {
                                        for b_colmajor in bools {
                                            for acc_colmajor in bools {
                                                let a = if a_colmajor { a } else { a.transpose() };
                                                let mut a =
                                                    if a_colmajor { a } else { a.transpose() };

                                                let b = if b_colmajor { b } else { b.transpose() };
                                                let mut b =
                                                    if b_colmajor { b } else { b.transpose() };

                                                if reverse_a_rows {
                                                    a = a.reverse_rows();
                                                }
                                                if reverse_a_cols {
                                                    a = a.reverse_cols();
                                                }
                                                if reverse_b_rows {
                                                    b = b.reverse_rows();
                                                }
                                                if reverse_b_cols {
                                                    b = b.reverse_cols();
                                                }
                                                for conj_a in conjs {
                                                    for conj_b in conjs {
                                                        for parallelism in par {
                                                            for beta in betas {
                                                                for alpha in alphas {
                                                                    test_matmul_impl(
                                                                        reverse_acc_cols,
                                                                        reverse_acc_rows,
                                                                        acc_colmajor,
                                                                        m,
                                                                        n,
                                                                        conj_a,
                                                                        conj_b,
                                                                        parallelism,
                                                                        beta,
                                                                        alpha,
                                                                        acc_init.as_mut(),
                                                                        a,
                                                                        b,
                                                                    );
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[math]
    fn matmul_with_conj_fallback<T: Copy + ComplexField>(
        acc: MatMut<'_, T>,
        a: MatRef<'_, T>,
        conj_a: Conj,
        b: MatRef<'_, T>,
        conj_b: Conj,
        beta: Accum,
        alpha: T,
    ) {
        let m = acc.nrows();
        let n = acc.ncols();
        let k = a.ncols();

        let job = |idx: usize| {
            let i = idx % m;
            let j = idx / m;
            let acc = acc.rb().submatrix(i, j, 1, 1);
            let mut acc = unsafe { acc.const_cast() };

            let mut local_acc = zero::<T>();
            for depth in 0..k {
                let a = a.at(i, depth);
                let b = b.at(depth, j);
                local_acc = local_acc
                    + match conj_a {
                        Conj::Yes => conj(*a),
                        Conj::No => copy(*a),
                    } * match conj_b {
                        Conj::Yes => conj(*b),
                        Conj::No => copy(*b),
                    }
            }
            match beta {
                Accum::Add => *acc.rb_mut().at_mut(0, 0) = acc[(0, 0)] + local_acc * alpha,
                Accum::Replace => *acc.rb_mut().at_mut(0, 0) = local_acc * alpha,
            }
        };

        for i in 0..m * n {
            job(i);
        }
    }

    #[math]
    fn test_matmul_impl(
        reverse_acc_cols: bool,
        reverse_acc_rows: bool,
        acc_colmajor: bool,
        m: usize,
        n: usize,
        conj_a: Conj,
        conj_b: Conj,
        parallelism: Par,
        beta: Accum,
        alpha: c32,
        acc_init: MatMut<c32>,
        a: MatRef<c32>,
        b: MatRef<c32>,
    ) {
        let acc = if acc_colmajor {
            acc_init
        } else {
            acc_init.transpose_mut()
        };

        let mut acc = if acc_colmajor {
            acc
        } else {
            acc.transpose_mut()
        };
        if reverse_acc_rows {
            acc = acc.reverse_rows_mut();
        }
        if reverse_acc_cols {
            acc = acc.reverse_cols_mut();
        }
        let mut target = acc.rb().to_owned();

        matmul_with_conj(acc.rb_mut(), beta, a, conj_a, b, conj_b, alpha, parallelism);
        matmul_with_conj_fallback(target.as_mut(), a, conj_a, b, conj_b, beta, alpha);

        for j in 0..n {
            for i in 0..m {
                let acc = *acc.rb().at(i, j);
                let target = *target.as_ref().at(i, j);
                assert!(abs(acc.re - target.re) < 1e-3);
                assert!(abs(acc.im - target.im) < 1e-3);
            }
        }
    }

    fn generate_structured_matrix(
        is_dst: bool,
        nrows: usize,
        ncols: usize,
        structure: BlockStructure,
    ) -> Mat<f64> {
        let rng = &mut StdRng::seed_from_u64(0);
        let mut mat = CwiseMatDistribution {
            nrows,
            ncols,
            dist: StandardNormal,
        }
        .rand::<Mat<f64>>(rng);

        if !is_dst {
            let kind = structure.diag_kind();
            if structure.is_lower() {
                for j in 0..ncols {
                    for i in 0..j {
                        *mat.as_mut().at_mut(i, j) = 0.0;
                    }
                }
            } else if structure.is_upper() {
                for j in 0..ncols {
                    for i in j + 1..nrows {
                        *mat.as_mut().at_mut(i, j) = 0.0;
                    }
                }
            }

            match kind {
                triangular::DiagonalKind::Zero => {
                    for i in 0..nrows {
                        *mat.as_mut().at_mut(i, i) = 0.0;
                    }
                }
                triangular::DiagonalKind::Unit => {
                    for i in 0..nrows {
                        *mat.as_mut().at_mut(i, i) = 1.0;
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
        let mut dst_target = dst.as_ref().to_owned();
        let dst_orig = dst.as_ref().to_owned();
        let lhs = generate_structured_matrix(false, m, k, lhs_structure);
        let rhs = generate_structured_matrix(false, k, n, rhs_structure);

        for parallelism in [Par::Seq, Par::rayon(8)] {
            triangular::matmul_with_conj(
                dst.as_mut(),
                dst_structure,
                Accum::Replace,
                lhs.as_ref(),
                lhs_structure,
                Conj::No,
                rhs.as_ref(),
                rhs_structure,
                Conj::No,
                2.5,
                parallelism,
            );

            matmul_with_conj(
                dst_target.as_mut(),
                Accum::Replace,
                lhs.as_ref(),
                Conj::No,
                rhs.as_ref(),
                Conj::No,
                2.5,
                parallelism,
            );

            if dst_structure.is_dense() {
                for j in 0..n {
                    for i in 0..m {
                        assert!(
                            (dst.as_ref().at(i, j) - dst_target.as_ref().at(i, j)).abs() < 1e-10
                        );
                    }
                }
            } else if dst_structure.is_lower() {
                for j in 0..n {
                    if matches!(dst_structure.diag_kind(), DiagonalKind::Generic) {
                        for i in 0..j {
                            assert!(
                                (dst.as_ref().at(i, j) - dst_orig.as_ref().at(i, j)).abs() < 1e-10
                            );
                        }
                        for i in j..n {
                            assert!(
                                (dst.as_ref().at(i, j) - dst_target.as_ref().at(i, j)).abs()
                                    < 1e-10
                            );
                        }
                    } else {
                        for i in 0..=j {
                            assert!(
                                (dst.as_ref().at(i, j) - dst_orig.as_ref().at(i, j)).abs() < 1e-10
                            );
                        }
                        for i in j + 1..n {
                            assert!(
                                (dst.as_ref().at(i, j) - dst_target.as_ref().at(i, j)).abs()
                                    < 1e-10
                            );
                        }
                    }
                }
            } else {
                for j in 0..n {
                    if matches!(dst_structure.diag_kind(), DiagonalKind::Generic) {
                        for i in 0..=j {
                            assert!(
                                (dst.as_ref().at(i, j) - dst_target.as_ref().at(i, j)).abs()
                                    < 1e-10
                            );
                        }
                        for i in j + 1..n {
                            assert!(
                                (dst.as_ref().at(i, j) - dst_orig.as_ref().at(i, j)).abs() < 1e-10
                            );
                        }
                    } else {
                        for i in 0..j {
                            assert!(
                                (dst.as_ref().at(i, j) - dst_target.as_ref().at(i, j)).abs()
                                    < 1e-10
                            );
                        }
                        for i in j..n {
                            assert!(
                                (dst.as_ref().at(i, j) - dst_orig.as_ref().at(i, j)).abs() < 1e-10
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_triangular() {
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
                    #[cfg(not(miri))]
                    let big = 100;

                    #[cfg(miri)]
                    let big = 31;
                    for _ in 0..3 {
                        let m = rand::random::<usize>() % big;
                        let mut n = rand::random::<usize>() % big;
                        let mut k = rand::random::<usize>() % big;

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
