use crate::{
    complex_native::*,
    mat::MatRef,
    utils::{simd::*, slice::*},
};
use faer_entity::*;

#[inline(always)]
fn norm_max_contiguous<E: RealField>(data: MatRef<'_, E>) -> E {
    struct Impl<'a, E: RealField> {
        data: MatRef<'a, E>,
    }

    impl<E: RealField> pulp::WithSimd for Impl<'_, E> {
        type Output = E;

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self { data } = self;
            let m = data.nrows();
            let n = data.ncols();

            let offset = SimdFor::<E, S>::new(simd).align_offset_ptr(data.as_ptr(), m);

            let simd = SimdFor::<E, S>::new(simd);

            let zero = simd.splat(E::faer_zero());

            let mut acc0 = zero;
            let mut acc1 = zero;
            let mut acc2 = zero;
            let mut acc3 = zero;
            for j in 0..n {
                let col = SliceGroup::<'_, E>::new(data.try_get_contiguous_col(j));
                let (head, body, tail) = simd.as_aligned_simd(col, offset);
                let (body4, body1) = body.as_arrays::<4>();

                let head = simd.abs(head.read_or(zero));
                acc0 = simd.select(simd.greater_than(head, acc0), head, acc0);

                for [x0, x1, x2, x3] in body4.into_ref_iter().map(RefGroup::unzip) {
                    let x0 = simd.abs(x0.get());
                    let x1 = simd.abs(x1.get());
                    let x2 = simd.abs(x2.get());
                    let x3 = simd.abs(x3.get());
                    acc0 = simd.select(simd.greater_than(x0, acc0), x0, acc0);
                    acc1 = simd.select(simd.greater_than(x1, acc1), x1, acc1);
                    acc2 = simd.select(simd.greater_than(x2, acc2), x2, acc2);
                    acc3 = simd.select(simd.greater_than(x3, acc3), x3, acc3);
                }

                for x0 in body1.into_ref_iter() {
                    let x0 = simd.abs(x0.get());
                    acc0 = simd.select(simd.greater_than(x0, acc0), x0, acc0);
                }

                let tail = simd.abs(tail.read_or(zero));
                acc3 = simd.select(simd.greater_than(tail, acc3), tail, acc3);
            }
            acc0 = simd.select(simd.greater_than(acc0, acc1), acc0, acc1);
            acc2 = simd.select(simd.greater_than(acc2, acc3), acc2, acc3);
            acc0 = simd.select(simd.greater_than(acc0, acc2), acc0, acc2);

            let acc0 = from_copy::<E, _>(acc0);
            let acc = SliceGroup::<'_, E>::new(E::faer_map(
                E::faer_as_ref(&acc0),
                #[inline(always)]
                |acc| bytemuck::cast_slice::<_, <E as Entity>::Unit>(core::slice::from_ref(acc)),
            ));
            let mut acc_scalar = E::faer_zero();
            for x in acc.into_ref_iter() {
                let x = x.read();
                acc_scalar = if acc_scalar > x { acc_scalar } else { x };
            }
            acc_scalar
        }
    }

    E::Simd::default().dispatch(Impl { data })
}

pub fn norm_max<E: ComplexField>(mut mat: MatRef<'_, E>) -> E::Real {
    if mat.ncols() > 1 && mat.col_stride().unsigned_abs() < mat.row_stride().unsigned_abs() {
        mat = mat.transpose();
    }
    if mat.row_stride() < 0 {
        mat = mat.reverse_rows();
    }

    if mat.nrows() == 0 || mat.ncols() == 0 {
        E::Real::faer_zero()
    } else {
        let m = mat.nrows();
        let n = mat.ncols();

        if mat.row_stride() == 1 {
            if coe::is_same::<E, c32>() {
                let mat: MatRef<'_, c32> = coe::coerce(mat);
                let mat = unsafe {
                    crate::mat::from_raw_parts(
                        mat.as_ptr() as *const f32,
                        2 * mat.nrows(),
                        mat.ncols(),
                        1,
                        2 * mat.col_stride(),
                    )
                };
                return coe::coerce_static(norm_max_contiguous::<f32>(mat));
            }
            if coe::is_same::<E, c64>() {
                let mat: MatRef<'_, c64> = coe::coerce(mat);
                let mat = unsafe {
                    crate::mat::from_raw_parts(
                        mat.as_ptr() as *const f64,
                        2 * mat.nrows(),
                        mat.ncols(),
                        1,
                        2 * mat.col_stride(),
                    )
                };
                return coe::coerce_static(norm_max_contiguous::<f64>(mat));
            }
            if coe::is_same::<E, num_complex::Complex<E::Real>>() {
                let mat: MatRef<'_, num_complex::Complex<E::Real>> = coe::coerce(mat);
                let num_complex::Complex { re, im } = mat.real_imag();
                let re = norm_max_contiguous(re);
                let im = norm_max_contiguous(im);
                return if re > im { re } else { im };
            }
            if coe::is_same::<E, E::Real>() {
                let mat: MatRef<'_, E::Real> = coe::coerce(mat);
                return norm_max_contiguous(mat);
            }
        }

        let mut acc = E::Real::faer_zero();
        for j in 0..n {
            for i in 0..m {
                let val = mat.read(i, j);
                let re = val.faer_real();
                let im = val.faer_imag();
                acc = if re > acc { re } else { acc };
                acc = if im > acc { im } else { acc };
            }
        }
        acc
    }
}
