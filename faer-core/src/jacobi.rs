use crate::{zipped, MatMut, RealField};
use faer_entity::SimdCtx;

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
    pub fn apply_on_the_left_in_place(&self, x: MatMut<'_, E>, y: MatMut<'_, E>) {
        struct ApplyOnLeft<'a, E: RealField> {
            c: E,
            s: E,
            x: MatMut<'a, E>,
            y: MatMut<'a, E>,
        }

        impl<E: RealField> pulp::WithSimd for ApplyOnLeft<'_, E> {
            type Output = ();

            #[inline(always)]
            fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                let Self { x, y, c, s } = self;
                assert!(x.nrows() == 1);
                assert!(y.nrows() == 1);
                assert_eq!(x.ncols(), y.ncols());

                if c == E::faer_one() && s == E::faer_zero() {
                    return;
                }

                let n = x.ncols();
                let x = E::faer_map(
                    x.as_ptr(),
                    #[inline(always)]
                    |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, n) },
                );
                let y = E::faer_map(
                    y.as_ptr(),
                    #[inline(always)]
                    |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, n) },
                );

                let c = E::faer_simd_splat(simd, c);
                let s = E::faer_simd_splat(simd, s);

                let (x_head, x_tail) = crate::simd::slice_as_mut_simd::<E, S>(x);
                let (y_head, y_tail) = crate::simd::slice_as_mut_simd::<E, S>(y);

                for (x, y) in E::faer_into_iter(x_head).zip(E::faer_into_iter(y_head)) {
                    let mut x_ = E::faer_deref(E::faer_rb(E::faer_as_ref(&x)));
                    let mut y_ = E::faer_deref(E::faer_rb(E::faer_as_ref(&y)));

                    (x_, y_) = (
                        E::faer_simd_mul_adde(
                            simd,
                            E::faer_copy(&c),
                            E::faer_copy(&x_),
                            E::faer_simd_mul(simd, E::faer_copy(&s), E::faer_copy(&y_)),
                        ),
                        E::faer_simd_mul_adde(
                            simd,
                            E::faer_copy(&c),
                            E::faer_copy(&y_),
                            E::faer_simd_neg(
                                simd,
                                E::faer_simd_mul(simd, E::faer_copy(&s), E::faer_copy(&x_)),
                            ),
                        ),
                    );

                    E::faer_map(
                        E::faer_zip(x, x_),
                        #[inline(always)]
                        |(x, x_)| *x = x_,
                    );
                    E::faer_map(
                        E::faer_zip(y, y_),
                        #[inline(always)]
                        |(y, y_)| *y = y_,
                    );
                }

                let mut x_ = E::faer_partial_load(simd, E::faer_rb(E::faer_as_ref(&x_tail)));
                let mut y_ = E::faer_partial_load(simd, E::faer_rb(E::faer_as_ref(&y_tail)));

                (x_, y_) = (
                    E::faer_simd_mul_adde(
                        simd,
                        E::faer_copy(&c),
                        E::faer_copy(&x_),
                        E::faer_simd_mul(simd, E::faer_copy(&s), E::faer_copy(&y_)),
                    ),
                    E::faer_simd_mul_adde(
                        simd,
                        E::faer_copy(&c),
                        E::faer_copy(&y_),
                        E::faer_simd_neg(
                            simd,
                            E::faer_simd_mul(simd, E::faer_copy(&s), E::faer_copy(&x_)),
                        ),
                    ),
                );

                E::faer_partial_store(simd, x_tail, x_);
                E::faer_partial_store(simd, y_tail, y_);
            }
        }

        let Self { c, s } = *self;
        if x.col_stride() == 1 && y.col_stride() == 1 {
            E::Simd::default().dispatch(ApplyOnLeft::<'_, E> { c, s, x, y });
        } else {
            zipped!(x, y).for_each(move |mut x, mut y| {
                let x_ = x.read();
                let y_ = y.read();
                x.write(c.faer_mul(x_).faer_add(s.faer_mul(y_)));
                y.write(s.faer_neg().faer_mul(x_).faer_add(c.faer_mul(y_)));
            });
        }
    }

    #[inline]
    pub fn apply_on_the_left_in_place_arch(
        &self,
        arch: E::Simd,
        x: MatMut<'_, E>,
        y: MatMut<'_, E>,
    ) {
        struct ApplyOnLeft<'a, E: RealField> {
            c: E,
            s: E,
            x: MatMut<'a, E>,
            y: MatMut<'a, E>,
        }

        impl<E: RealField> pulp::WithSimd for ApplyOnLeft<'_, E> {
            type Output = ();

            #[inline(always)]
            fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                let Self { x, y, c, s } = self;
                assert!(x.nrows() == 1);
                assert!(y.nrows() == 1);
                assert_eq!(x.ncols(), y.ncols());

                if c == E::faer_one() && s == E::faer_zero() {
                    return;
                }

                let n = x.ncols();
                let x = E::faer_map(
                    x.as_ptr(),
                    #[inline(always)]
                    |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, n) },
                );
                let y = E::faer_map(
                    y.as_ptr(),
                    #[inline(always)]
                    |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, n) },
                );

                let c = E::faer_simd_splat(simd, c);
                let s = E::faer_simd_splat(simd, s);

                let (x_head, x_tail) = crate::simd::slice_as_mut_simd::<E, S>(x);
                let (y_head, y_tail) = crate::simd::slice_as_mut_simd::<E, S>(y);

                for (x, y) in E::faer_into_iter(x_head).zip(E::faer_into_iter(y_head)) {
                    let mut x_ = E::faer_deref(E::faer_rb(E::faer_as_ref(&x)));
                    let mut y_ = E::faer_deref(E::faer_rb(E::faer_as_ref(&y)));

                    (x_, y_) = (
                        E::faer_simd_mul_adde(
                            simd,
                            E::faer_copy(&c),
                            E::faer_copy(&x_),
                            E::faer_simd_mul(simd, E::faer_copy(&s), E::faer_copy(&y_)),
                        ),
                        E::faer_simd_mul_adde(
                            simd,
                            E::faer_copy(&c),
                            E::faer_copy(&y_),
                            E::faer_simd_neg(
                                simd,
                                E::faer_simd_mul(simd, E::faer_copy(&s), E::faer_copy(&x_)),
                            ),
                        ),
                    );

                    E::faer_map(
                        E::faer_zip(x, x_),
                        #[inline(always)]
                        |(x, x_)| *x = x_,
                    );
                    E::faer_map(
                        E::faer_zip(y, y_),
                        #[inline(always)]
                        |(y, y_)| *y = y_,
                    );
                }

                let mut x_ = E::faer_partial_load(simd, E::faer_rb(E::faer_as_ref(&x_tail)));
                let mut y_ = E::faer_partial_load(simd, E::faer_rb(E::faer_as_ref(&y_tail)));

                (x_, y_) = (
                    E::faer_simd_mul_adde(
                        simd,
                        E::faer_copy(&c),
                        E::faer_copy(&x_),
                        E::faer_simd_mul(simd, E::faer_copy(&s), E::faer_copy(&y_)),
                    ),
                    E::faer_simd_mul_adde(
                        simd,
                        E::faer_copy(&c),
                        E::faer_copy(&y_),
                        E::faer_simd_neg(
                            simd,
                            E::faer_simd_mul(simd, E::faer_copy(&s), E::faer_copy(&x_)),
                        ),
                    ),
                );

                E::faer_partial_store(simd, x_tail, x_);
                E::faer_partial_store(simd, y_tail, y_);
            }
        }

        let Self { c, s } = *self;
        if x.col_stride() == 1 && y.col_stride() == 1 {
            arch.dispatch(ApplyOnLeft::<'_, E> { c, s, x, y });
        } else {
            zipped!(x, y).for_each(move |mut x, mut y| {
                let x_ = x.read();
                let y_ = y.read();
                x.write(c.faer_mul(x_).faer_add(s.faer_mul(y_)));
                y.write(s.faer_neg().faer_mul(x_).faer_add(c.faer_mul(y_)));
            });
        }
    }

    #[inline]
    pub fn apply_on_the_right_in_place(&self, x: MatMut<'_, E>, y: MatMut<'_, E>) {
        self.transpose()
            .apply_on_the_left_in_place(x.transpose(), y.transpose());
    }

    #[inline]
    pub fn apply_on_the_right_in_place_arch(
        &self,
        arch: E::Simd,
        x: MatMut<'_, E>,
        y: MatMut<'_, E>,
    ) {
        self.transpose()
            .apply_on_the_left_in_place_arch(arch, x.transpose(), y.transpose());
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
