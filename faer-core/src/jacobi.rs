use crate::{zipped, MatMut, RealField};

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
        if q == E::zero() {
            Self {
                c: if p < E::zero() {
                    E::one().neg()
                } else {
                    E::one()
                },
                s: E::zero(),
            }
        } else if p == E::zero() {
            Self {
                c: E::zero(),
                s: if q < E::zero() {
                    E::one().neg()
                } else {
                    E::one()
                },
            }
        } else if p.abs() > q.abs() {
            let t = q.div(p);
            let mut u = E::one().add(t.abs2()).sqrt();
            if p < E::zero() {
                u = u.neg();
            }
            let c = u.inv();
            let s = t.neg().mul(c);

            Self { c, s }
        } else {
            let t = p.div(q);
            let mut u = E::one().add(t.abs2()).sqrt();
            if q < E::zero() {
                u = u.neg();
            }
            let s = u.inv().neg();
            let c = t.neg().mul(s);

            Self { c, s }
        }
    }

    #[inline]
    pub fn from_triplet(x: E, y: E, z: E) -> Self {
        let abs_y = y.abs();
        let two_abs_y = abs_y.add(abs_y);
        if two_abs_y == E::zero() {
            Self {
                c: E::one(),
                s: E::zero(),
            }
        } else {
            let tau = (x.sub(z)).mul(two_abs_y.inv());
            let w = ((tau.mul(tau)).add(E::one())).sqrt();
            let t = if tau > E::zero() {
                (tau.add(w)).inv()
            } else {
                (tau.sub(w)).inv()
            };

            let neg_sign_y = if y > E::zero() {
                E::one().neg()
            } else {
                E::one()
            };
            let n = (t.mul(t).add(E::one())).sqrt().inv();

            Self {
                c: n,
                s: neg_sign_y.mul(t).mul(n),
            }
        }
    }

    #[inline]
    pub fn apply_on_the_left_2x2(&self, m00: E, m01: E, m10: E, m11: E) -> (E, E, E, E) {
        let Self { c, s } = *self;
        (
            m00.mul(c).add(m10.mul(s)),
            m01.mul(c).add(m11.mul(s)),
            s.neg().mul(m00).add(c.mul(m10)),
            s.neg().mul(m01).add(c.mul(m11)),
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

                if c == E::one() && s == E::zero() {
                    return;
                }

                let n = x.ncols();
                let x = E::map(
                    x.as_ptr(),
                    #[inline(always)]
                    |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, n) },
                );
                let y = E::map(
                    y.as_ptr(),
                    #[inline(always)]
                    |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, n) },
                );

                let c = E::simd_splat(simd, c);
                let s = E::simd_splat(simd, s);

                let (x_head, x_tail) = crate::simd::slice_as_mut_simd::<E, S>(x);
                let (y_head, y_tail) = crate::simd::slice_as_mut_simd::<E, S>(y);

                for (x, y) in E::into_iter(x_head).zip(E::into_iter(y_head)) {
                    let mut x_ = E::deref(E::rb(E::as_ref(&x)));
                    let mut y_ = E::deref(E::rb(E::as_ref(&y)));

                    (x_, y_) = (
                        E::simd_mul_adde(
                            simd,
                            E::copy(&c),
                            E::copy(&x_),
                            E::simd_mul(simd, E::copy(&s), E::copy(&y_)),
                        ),
                        E::simd_mul_adde(
                            simd,
                            E::copy(&c),
                            E::copy(&y_),
                            E::simd_neg(simd, E::simd_mul(simd, E::copy(&s), E::copy(&x_))),
                        ),
                    );

                    E::map(
                        E::zip(x, x_),
                        #[inline(always)]
                        |(x, x_)| *x = x_,
                    );
                    E::map(
                        E::zip(y, y_),
                        #[inline(always)]
                        |(y, y_)| *y = y_,
                    );
                }

                let mut x_ = E::partial_load(simd, E::rb(E::as_ref(&x_tail)));
                let mut y_ = E::partial_load(simd, E::rb(E::as_ref(&y_tail)));

                (x_, y_) = (
                    E::simd_mul_adde(
                        simd,
                        E::copy(&c),
                        E::copy(&x_),
                        E::simd_mul(simd, E::copy(&s), E::copy(&y_)),
                    ),
                    E::simd_mul_adde(
                        simd,
                        E::copy(&c),
                        E::copy(&y_),
                        E::simd_neg(simd, E::simd_mul(simd, E::copy(&s), E::copy(&x_))),
                    ),
                );

                E::partial_store(simd, x_tail, x_);
                E::partial_store(simd, y_tail, y_);
            }
        }

        let Self { c, s } = *self;
        if E::HAS_SIMD && x.col_stride() == 1 && y.col_stride() == 1 {
            pulp::Arch::new().dispatch(ApplyOnLeft::<'_, E> { c, s, x, y });
        } else {
            zipped!(x, y).for_each(move |mut x, mut y| {
                let x_ = x.read();
                let y_ = y.read();
                x.write(c.mul(x_).add(s.mul(y_)));
                y.write(s.neg().mul(x_).add(c.mul(y_)));
            });
        }
    }

    #[inline]
    pub fn apply_on_the_left_in_place_arch(
        &self,
        arch: pulp::Arch,
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

                if c == E::one() && s == E::zero() {
                    return;
                }

                let n = x.ncols();
                let x = E::map(
                    x.as_ptr(),
                    #[inline(always)]
                    |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, n) },
                );
                let y = E::map(
                    y.as_ptr(),
                    #[inline(always)]
                    |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, n) },
                );

                let c = E::simd_splat(simd, c);
                let s = E::simd_splat(simd, s);

                let (x_head, x_tail) = crate::simd::slice_as_mut_simd::<E, S>(x);
                let (y_head, y_tail) = crate::simd::slice_as_mut_simd::<E, S>(y);

                for (x, y) in E::into_iter(x_head).zip(E::into_iter(y_head)) {
                    let mut x_ = E::deref(E::rb(E::as_ref(&x)));
                    let mut y_ = E::deref(E::rb(E::as_ref(&y)));

                    (x_, y_) = (
                        E::simd_mul_adde(
                            simd,
                            E::copy(&c),
                            E::copy(&x_),
                            E::simd_mul(simd, E::copy(&s), E::copy(&y_)),
                        ),
                        E::simd_mul_adde(
                            simd,
                            E::copy(&c),
                            E::copy(&y_),
                            E::simd_neg(simd, E::simd_mul(simd, E::copy(&s), E::copy(&x_))),
                        ),
                    );

                    E::map(
                        E::zip(x, x_),
                        #[inline(always)]
                        |(x, x_)| *x = x_,
                    );
                    E::map(
                        E::zip(y, y_),
                        #[inline(always)]
                        |(y, y_)| *y = y_,
                    );
                }

                let mut x_ = E::partial_load(simd, E::rb(E::as_ref(&x_tail)));
                let mut y_ = E::partial_load(simd, E::rb(E::as_ref(&y_tail)));

                (x_, y_) = (
                    E::simd_mul_adde(
                        simd,
                        E::copy(&c),
                        E::copy(&x_),
                        E::simd_mul(simd, E::copy(&s), E::copy(&y_)),
                    ),
                    E::simd_mul_adde(
                        simd,
                        E::copy(&c),
                        E::copy(&y_),
                        E::simd_neg(simd, E::simd_mul(simd, E::copy(&s), E::copy(&x_))),
                    ),
                );

                E::partial_store(simd, x_tail, x_);
                E::partial_store(simd, y_tail, y_);
            }
        }

        let Self { c, s } = *self;
        if E::HAS_SIMD && x.col_stride() == 1 && y.col_stride() == 1 {
            arch.dispatch(ApplyOnLeft::<'_, E> { c, s, x, y });
        } else {
            zipped!(x, y).for_each(move |mut x, mut y| {
                let x_ = x.read();
                let y_ = y.read();
                x.write(c.mul(x_).add(s.mul(y_)));
                y.write(s.neg().mul(x_).add(c.mul(y_)));
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
        arch: pulp::Arch,
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
            s: self.s.neg(),
        }
    }
}

impl<E: RealField> core::ops::Mul for JacobiRotation<E> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            c: self.c.mul(rhs.c).sub(self.s.mul(rhs.s)),
            s: self.c.mul(rhs.s).add(self.s.mul(rhs.c)),
        }
    }
}
