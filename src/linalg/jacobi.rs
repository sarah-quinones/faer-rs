use crate::{
    assert, unzipped,
    utils::{simd::*, slice::*},
    zipped, ColMut, RealField, RowMut,
};
use faer_entity::{pulp, SimdCtx, SimdGroupFor};
use reborrow::*;

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct JacobiRotation<T> {
    pub c: T,
    pub s: T,
}

unsafe impl<T: bytemuck::Zeroable> bytemuck::Zeroable for JacobiRotation<T> {}
unsafe impl<T: bytemuck::Pod> bytemuck::Pod for JacobiRotation<T> {}

impl<E: RealField> JacobiRotation<E> {
    #[inline]
    pub fn make_givens(p: E, q: E) -> Self {
        if q == E::faer_zero() {
            Self {
                c: if p < E::faer_zero() {
                    E::faer_one().faer_neg()
                } else {
                    E::faer_one()
                },
                s: E::faer_zero(),
            }
        } else if p == E::faer_zero() {
            Self {
                c: E::faer_zero(),
                s: if q < E::faer_zero() {
                    E::faer_one().faer_neg()
                } else {
                    E::faer_one()
                },
            }
        } else if p.faer_abs() > q.faer_abs() {
            let t = q.faer_div(p);
            let mut u = E::faer_one().faer_add(t.faer_abs2()).faer_sqrt();
            if p < E::faer_zero() {
                u = u.faer_neg();
            }
            let c = u.faer_inv();
            let s = t.faer_neg().faer_mul(c);

            Self { c, s }
        } else {
            let t = p.faer_div(q);
            let mut u = E::faer_one().faer_add(t.faer_abs2()).faer_sqrt();
            if q < E::faer_zero() {
                u = u.faer_neg();
            }
            let s = u.faer_inv().faer_neg();
            let c = t.faer_neg().faer_mul(s);

            Self { c, s }
        }
    }

    #[inline]
    pub fn from_triplet(x: E, y: E, z: E) -> Self {
        let abs_y = y.faer_abs();
        let two_abs_y = abs_y.faer_add(abs_y);
        if two_abs_y == E::faer_zero() {
            Self {
                c: E::faer_one(),
                s: E::faer_zero(),
            }
        } else {
            let tau = (x.faer_sub(z)).faer_mul(two_abs_y.faer_inv());
            let w = ((tau.faer_mul(tau)).faer_add(E::faer_one())).faer_sqrt();
            let t = if tau > E::faer_zero() {
                (tau.faer_add(w)).faer_inv()
            } else {
                (tau.faer_sub(w)).faer_inv()
            };

            let neg_sign_y = if y > E::faer_zero() {
                E::faer_one().faer_neg()
            } else {
                E::faer_one()
            };
            let n = (t.faer_mul(t).faer_add(E::faer_one()))
                .faer_sqrt()
                .faer_inv();

            Self {
                c: n,
                s: neg_sign_y.faer_mul(t).faer_mul(n),
            }
        }
    }

    #[inline]
    pub fn apply_on_the_left_2x2(&self, m00: E, m01: E, m10: E, m11: E) -> (E, E, E, E) {
        let Self { c, s } = *self;
        (
            m00.faer_mul(c).faer_add(m10.faer_mul(s)),
            m01.faer_mul(c).faer_add(m11.faer_mul(s)),
            s.faer_neg().faer_mul(m00).faer_add(c.faer_mul(m10)),
            s.faer_neg().faer_mul(m01).faer_add(c.faer_mul(m11)),
        )
    }

    #[inline]
    pub fn apply_on_the_right_2x2(&self, m00: E, m01: E, m10: E, m11: E) -> (E, E, E, E) {
        let (r00, r01, r10, r11) = self.transpose().apply_on_the_left_2x2(m00, m10, m01, m11);
        (r00, r10, r01, r11)
    }

    #[inline]
    pub fn apply_on_the_left_in_place(&self, x: RowMut<'_, E>, y: RowMut<'_, E>) {
        self.apply_on_the_left_in_place_arch(E::Simd::default(), x, y);
    }

    #[inline(never)]
    fn apply_on_the_left_in_place_fallback(&self, x: RowMut<'_, E>, y: RowMut<'_, E>) {
        let Self { c, s } = *self;
        zipped!(__rw, x, y).for_each(move |unzipped!(mut x, mut y)| {
            let x_ = x.read();
            let y_ = y.read();
            x.write(c.faer_mul(x_).faer_add(s.faer_mul(y_)));
            y.write(s.faer_neg().faer_mul(x_).faer_add(c.faer_mul(y_)));
        });
    }

    #[inline(always)]
    pub fn apply_on_the_right_in_place_with_simd_and_offset<S: pulp::Simd>(
        &self,
        simd: S,
        offset: pulp::Offset<E::SimdMask<S>>,
        x: ColMut<'_, E>,
        y: ColMut<'_, E>,
    ) {
        self.transpose()
            .apply_on_the_left_in_place_with_simd_and_offset(
                simd,
                offset,
                x.transpose_mut(),
                y.transpose_mut(),
            );
    }

    #[inline(always)]
    pub fn apply_on_the_left_in_place_with_simd_and_offset<S: pulp::Simd>(
        &self,
        simd: S,
        offset: pulp::Offset<E::SimdMask<S>>,
        x: RowMut<'_, E>,
        y: RowMut<'_, E>,
    ) {
        let Self { c, s } = *self;
        assert!(all(x.nrows() == 1, y.nrows() == 1, x.ncols() == y.ncols()));

        if c == E::faer_one() && s == E::faer_zero() {
            return;
        }

        if x.col_stride() != 1 || y.col_stride() != 1 {
            self.apply_on_the_left_in_place_fallback(x, y);
            return;
        }

        let simd = SimdFor::<E, S>::new(simd);

        let x = SliceGroupMut::<'_, E>::new(x.transpose_mut().try_as_slice_mut().unwrap());
        let y = SliceGroupMut::<'_, E>::new(y.transpose_mut().try_as_slice_mut().unwrap());

        let c = simd.splat(c);
        let s = simd.splat(s);

        let (x_head, x_body, x_tail) = simd.as_aligned_simd_mut(x, offset);
        let (y_head, y_body, y_tail) = simd.as_aligned_simd_mut(y, offset);

        #[inline(always)]
        fn process<E: RealField, S: pulp::Simd>(
            simd: SimdFor<E, S>,
            mut x: impl Write<Output = SimdGroupFor<E, S>>,
            mut y: impl Write<Output = SimdGroupFor<E, S>>,
            c: SimdGroupFor<E, S>,
            s: SimdGroupFor<E, S>,
        ) {
            let zero = simd.splat(E::faer_zero());
            let x_ = x.read_or(zero);
            let y_ = y.read_or(zero);
            x.write(simd.mul_add_e(c, x_, simd.mul(s, y_)));
            y.write(simd.mul_add_e(c, y_, simd.neg(simd.mul(s, x_))));
        }

        process(simd, x_head, y_head, c, s);
        for (x, y) in x_body.into_mut_iter().zip(y_body.into_mut_iter()) {
            process(simd, x, y, c, s);
        }
        process(simd, x_tail, y_tail, c, s);
    }

    #[inline]
    pub fn apply_on_the_left_in_place_arch(
        &self,
        arch: E::Simd,
        x: RowMut<'_, E>,
        y: RowMut<'_, E>,
    ) {
        struct ApplyOnLeft<'a, E: RealField> {
            c: E,
            s: E,
            x: RowMut<'a, E>,
            y: RowMut<'a, E>,
        }

        impl<E: RealField> pulp::WithSimd for ApplyOnLeft<'_, E> {
            type Output = ();

            #[inline(always)]
            fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                let Self { x, y, c, s } = self;
                assert!(all(x.nrows() == 1, y.nrows() == 1, x.ncols() == y.ncols()));

                if c == E::faer_one() && s == E::faer_zero() {
                    return;
                }

                let simd = SimdFor::<E, S>::new(simd);

                let x = SliceGroupMut::<'_, E>::new(x.transpose_mut().try_as_slice_mut().unwrap());
                let y = SliceGroupMut::<'_, E>::new(y.transpose_mut().try_as_slice_mut().unwrap());

                let offset = simd.align_offset(x.rb());

                let c = simd.splat(c);
                let s = simd.splat(s);

                let (x_head, x_body, x_tail) = simd.as_aligned_simd_mut(x, offset);
                let (y_head, y_body, y_tail) = simd.as_aligned_simd_mut(y, offset);

                #[inline(always)]
                fn process<E: RealField, S: pulp::Simd>(
                    simd: SimdFor<E, S>,
                    mut x: impl Write<Output = SimdGroupFor<E, S>>,
                    mut y: impl Write<Output = SimdGroupFor<E, S>>,
                    c: SimdGroupFor<E, S>,
                    s: SimdGroupFor<E, S>,
                ) {
                    let zero = simd.splat(E::faer_zero());
                    let x_ = x.read_or(zero);
                    let y_ = y.read_or(zero);
                    x.write(simd.mul_add_e(c, x_, simd.mul(s, y_)));
                    y.write(simd.mul_add_e(c, y_, simd.neg(simd.mul(s, x_))));
                }

                process(simd, x_head, y_head, c, s);
                for (x, y) in x_body.into_mut_iter().zip(y_body.into_mut_iter()) {
                    process(simd, x, y, c, s);
                }
                process(simd, x_tail, y_tail, c, s);
            }
        }

        let Self { c, s } = *self;

        let mut x = x;
        let mut y = y;

        if x.col_stride() == 1 && y.col_stride() == 1 {
            arch.dispatch(ApplyOnLeft::<'_, E> { c, s, x, y });
        } else {
            zipped!(__rw, x, y).for_each(move |unzipped!(mut x, mut y)| {
                let x_ = x.read();
                let y_ = y.read();
                x.write(c.faer_mul(x_).faer_add(s.faer_mul(y_)));
                y.write(s.faer_neg().faer_mul(x_).faer_add(c.faer_mul(y_)));
            });
        }
    }

    #[inline]
    pub fn apply_on_the_right_in_place(&self, x: ColMut<'_, E>, y: ColMut<'_, E>) {
        self.transpose()
            .apply_on_the_left_in_place(x.transpose_mut(), y.transpose_mut());
    }

    #[inline]
    pub fn apply_on_the_right_in_place_arch(
        &self,
        arch: E::Simd,
        x: ColMut<'_, E>,
        y: ColMut<'_, E>,
    ) {
        self.transpose().apply_on_the_left_in_place_arch(
            arch,
            x.transpose_mut(),
            y.transpose_mut(),
        );
    }

    #[inline]
    pub fn transpose(&self) -> Self {
        Self {
            c: self.c,
            s: self.s.faer_neg(),
        }
    }
}

impl<E: RealField> core::ops::Mul for JacobiRotation<E> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            c: self.c.faer_mul(rhs.c).faer_sub(self.s.faer_mul(rhs.s)),
            s: self.c.faer_mul(rhs.s).faer_add(self.s.faer_mul(rhs.c)),
        }
    }
}
