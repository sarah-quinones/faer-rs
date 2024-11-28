use crate::internal_prelude::*;

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct JacobiRotation<T> {
    pub c: T,
    pub s: T,
}

impl<T: ComplexField> JacobiRotation<T> {
    #[math]
    pub fn make_givens(p: T, q: T) -> Self
    where
        T: RealField,
    {
        {
            let p = real(p);
            let q = real(q);

            if q == zero() {
                let c = if p < zero() { -one() } else { one() };
                let s = zero();
                let c = from_real(c);
                let s = from_real(s);

                Self { c, s }
            } else if p == zero() {
                let c = zero();
                let s = if q < zero() { one() } else { -one() };
                let c = from_real(c);
                let s = from_real(s);

                Self { c, s }
            } else if abs(p) > abs(q) {
                let t = q / p;
                let mut u = hypot(one(), t);
                if p < zero() {
                    u = -u;
                }
                let c = recip(u);
                let s = -t * c;

                let c = from_real(c);
                let s = from_real(s);

                Self { c, s }
            } else {
                let t = p / q;
                let mut u = hypot(one(), t);
                if q < zero() {
                    u = -u;
                }
                let s = -recip(u);
                let c = -t * s;

                let c = from_real(c);
                let s = from_real(s);

                Self { c, s }
            }
        }
    }

    #[math]
    pub fn rotg(p: T, q: T) -> (Self, T) {
        if const { T::IS_REAL } {
            {
                let p = real(p);
                let q = real(q);

                if q == zero() {
                    let c = one();
                    let s = zero();
                    let c = from_real(c);
                    let s = from_real(s);

                    (Self { c, s }, from_real(p))
                } else if p == zero() {
                    let c = zero();
                    let s = one();
                    let c = from_real(c);
                    let s = from_real(s);

                    (Self { c, s }, from_real(q))
                } else {
                    let safmin = min_positive();
                    let safmax = recip(safmin);
                    let a = p;
                    let b = q;

                    let scl = min(safmax, max(safmin, max(abs(a), abs(b))));
                    let r = scl * (sqrt(abs2(a / scl) + abs2(b / scl)));
                    let r = if abs(a) > abs(b) {
                        if a > zero() {
                            r
                        } else {
                            -r
                        }
                    } else {
                        if b > zero() {
                            r
                        } else {
                            -r
                        }
                    };
                    let c = from_real(a / r);
                    let s = from_real(b / r);
                    let r = from_real(r);

                    (Self { c, s }, r)
                }
            }
        } else {
            {
                let a = p;
                let b = q;

                let eps = eps();
                let sml = min_positive();
                let big = max_positive();
                let rtmin = sqrt(div(sml, eps));
                let rtmax = recip(rtmin);

                if b == zero() {
                    return (
                        Self {
                            c: one(),
                            s: zero(),
                        },
                        one(),
                    );
                }

                let (c, s, r);

                if a == zero() {
                    c = zero();
                    let g1 = max(abs(real(b)), abs(imag(b)));
                    if g1 > rtmin && g1 < rtmax {
                        // Use unscaled algorithm
                        let g2 = abs2(b);

                        let d = sqrt(g2);
                        s = mul_real(conj(b), recip(d));
                        r = from_real(d);
                    } else {
                        // Use scaled algorithm
                        let u = min(big, max(sml, g1));
                        let uu = recip(u);
                        let gs = mul_real(b, uu);
                        let g2 = abs2(gs);
                        let d = sqrt(g2);
                        s = mul_real(conj(gs), recip(d));
                        r = from_real(mul(d, u));
                    }
                } else {
                    let f1 = max(abs(real(a)), abs(imag(a)));
                    let g1 = max(abs(real(b)), abs(imag(b)));
                    if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
                        // Use unscaled algorithm
                        let f2 = abs2(a);
                        let g2 = abs2(b);
                        let h2 = add(f2, g2);
                        let d = if f2 > rtmin && h2 < rtmax {
                            sqrt(mul(f2, h2))
                        } else {
                            mul(sqrt(f2), sqrt(h2))
                        };
                        let p = recip(d);
                        c = from_real(mul(f2, p));
                        s = conj(b) * mul_real(a, p);

                        r = mul_real(a, mul(h2, p));
                    } else {
                        // Use scaled algorithm
                        let u = min(big, max(sml, max(f1, g1)));
                        let uu = recip(u);
                        let gs = mul_real(b, uu);
                        let g2 = abs2(gs);
                        let (f2, h2, w);
                        let fs;
                        if mul(f1, uu) < rtmin {
                            // a is not well-scaled when scaled by g1.
                            let v = min(big, max(sml, f1));
                            let vv = recip(v);
                            w = mul(v, uu);
                            fs = mul_real(a, vv);
                            f2 = abs2(fs);
                            h2 = add(mul(mul(f2, w), w), g2);
                        } else {
                            // Otherwise use the same scaling for a and b.
                            w = one();
                            fs = mul_real(a, uu);
                            f2 = abs2(fs);
                            h2 = add(f2, g2);
                        }
                        let d = if f2 > rtmin && h2 < rtmax {
                            sqrt(mul(f2, h2))
                        } else {
                            mul(sqrt(f2), sqrt(h2))
                        };
                        let p = recip(d);
                        c = from_real(mul(mul(f2, p), w));
                        s = conj(gs) * mul_real(fs, p);
                        r = mul_real(mul_real(fs, mul(h2, p)), u);
                    }
                }

                (Self { c, s }, r)
            }
        }
    }

    #[inline]
    #[math]
    pub fn apply_on_the_left_2x2(&self, m00: T, m01: T, m10: T, m11: T) -> (T, T, T, T) {
        let Self { c, s } = self;

        (
            m00 * *c + m10 * *s,
            m01 * *c + m11 * *s,
            *c * m10 - conj(*s) * m00,
            *c * m11 - conj(*s) * m01,
        )
    }

    #[inline]
    pub fn apply_on_the_right_2x2(&self, m00: T, m01: T, m10: T, m11: T) -> (T, T, T, T) {
        let (r00, r01, r10, r11) = self.adjoint().apply_on_the_left_2x2(m00, m10, m01, m11);
        (r00, r10, r01, r11)
    }

    #[inline]
    fn apply_on_the_left_in_place_impl<'N>(
        &self,

        (x, y): (RowMut<'_, T, Dim<'N>>, RowMut<'_, T, Dim<'N>>),
    ) {
        let mut x = x;
        let mut y = y;
        if const { T::SIMD_CAPABILITIES.is_simd() } {
            struct Impl<'a, 'N, T: ComplexField> {
                this: &'a JacobiRotation<T>,
                x: RowMut<'a, T, Dim<'N>, ContiguousFwd>,
                y: RowMut<'a, T, Dim<'N>, ContiguousFwd>,
            }
            impl<T: ComplexField> pulp::WithSimd for Impl<'_, '_, T> {
                type Output = ();

                #[inline(always)]
                fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                    let Self { this, x, y } = self;
                    let simd = SimdCtx::new(T::simd_ctx(simd), x.ncols());
                    this.apply_on_the_left_in_place_with_simd(simd, x, y);
                }
            }

            if x.col_stride() < 0 && y.col_stride() < 0 {
                x = x.reverse_cols_mut();
                y = y.reverse_cols_mut();
            }

            if let (Some(x), Some(y)) = (
                x.rb_mut().try_as_row_major_mut(),
                y.rb_mut().try_as_row_major_mut(),
            ) {
                T::Arch::default().dispatch(Impl { this: self, x, y });
                return;
            }
        }
        self.apply_on_the_left_in_place_fallback(x, y);
    }

    #[inline]
    pub fn apply_on_the_left_in_place<N: Shape>(
        &self,

        (x, y): (RowMut<'_, T, N>, RowMut<'_, T, N>),
    ) {
        with_dim!(N, x.ncols().unbound());
        self.apply_on_the_left_in_place_impl((x.as_col_shape_mut(N), y.as_col_shape_mut(N)));
    }

    #[inline]
    pub fn apply_on_the_right_in_place<N: Shape>(
        &self,

        (x, y): (ColMut<'_, T, N>, ColMut<'_, T, N>),
    ) {
        with_dim!(N, x.nrows().unbound());

        let x = x.as_row_shape_mut(N);
        let y = y.as_row_shape_mut(N);

        self.transpose()
            .apply_on_the_left_in_place((x.transpose_mut(), y.transpose_mut()));
    }

    #[inline(never)]
    #[math]
    fn apply_on_the_left_in_place_fallback<'N>(
        &self,

        x: RowMut<'_, T, Dim<'N>>,
        y: RowMut<'_, T, Dim<'N>>,
    ) {
        let Self { c, s } = self;
        zipped!(x, y).for_each(move |unzipped!(x, y)| {
            let x_ = *c * *x - conj(*s) * *y;
            let y_ = *c * *y + *s * *x;
            *x = x_;
            *y = y_;
        });
    }

    #[inline(always)]
    pub fn apply_on_the_right_in_place_with_simd<'N, S: pulp::Simd>(
        &self,
        simd: SimdCtx<'N, T, S>,

        x: ColMut<'_, T, Dim<'N>, ContiguousFwd>,
        y: ColMut<'_, T, Dim<'N>, ContiguousFwd>,
    ) {
        self.transpose().apply_on_the_left_in_place_with_simd(
            simd,
            x.transpose_mut(),
            y.transpose_mut(),
        );
    }

    #[math]
    #[inline(always)]
    pub fn apply_on_the_left_in_place_with_simd<'N, S: pulp::Simd>(
        &self,
        simd: SimdCtx<'N, T, S>,

        x: RowMut<'_, T, Dim<'N>, ContiguousFwd>,
        y: RowMut<'_, T, Dim<'N>, ContiguousFwd>,
    ) {
        let Self { c, s } = self;

        if *c == one() && *s == zero() {
            return;
        };

        let mut x = x.transpose_mut();
        let mut y = y.transpose_mut();

        let c = simd.splat_real(&real(*c));
        let s = simd.splat(&s);

        let (head, body, tail) = simd.indices();

        if let Some(i) = head {
            let mut xx = simd.read(x.rb(), i);
            let mut yy = simd.read(y.rb(), i);

            (xx, yy) = (
                simd.conj_mul_add(simd.neg(s), yy, simd.mul_real(xx, c)),
                simd.mul_add(s, xx, simd.mul_real(yy, c)),
            );

            simd.write(x.rb_mut(), i, xx);
            simd.write(y.rb_mut(), i, yy);
        }
        for i in body {
            let mut xx = simd.read(x.rb(), i);
            let mut yy = simd.read(y.rb(), i);

            (xx, yy) = (
                simd.conj_mul_add(simd.neg(s), yy, simd.mul_real(xx, c)),
                simd.mul_add(s, xx, simd.mul_real(yy, c)),
            );

            simd.write(x.rb_mut(), i, xx);
            simd.write(y.rb_mut(), i, yy);
        }
        if let Some(i) = tail {
            let mut xx = simd.read(x.rb(), i);
            let mut yy = simd.read(y.rb(), i);

            (xx, yy) = (
                simd.conj_mul_add(simd.neg(s), yy, simd.mul_real(xx, c)),
                simd.mul_add(s, xx, simd.mul_real(yy, c)),
            );

            simd.write(x.rb_mut(), i, xx);
            simd.write(y.rb_mut(), i, yy);
        }
    }

    #[inline]
    #[math]
    pub fn adjoint(&self) -> Self {
        Self {
            c: copy(self.c),
            s: -conj(self.s),
        }
    }

    #[inline]
    #[math]
    pub fn conjugate(&self) -> Self {
        Self {
            c: copy(self.c),
            s: conj(self.s),
        }
    }

    #[inline]
    #[math]
    pub fn transpose(&self) -> Self {
        Self {
            c: copy(self.c),
            s: -self.s,
        }
    }
}
