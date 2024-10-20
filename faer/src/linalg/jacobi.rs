use crate::internal_prelude::*;

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct JacobiRotation<C: Container, T> {
    pub c: C::Of<T>,
    pub s: C::Of<T>,
}

impl<C: ComplexContainer, T: ComplexField<C>> JacobiRotation<C, T> {
    #[math]
    pub fn make_givens(ctx: &Ctx<C, T>, p: C::Of<T>, q: C::Of<T>) -> Self
    where
        C: RealContainer,
        T: RealField<C>,
    {
        math({
            let p = cx.real(p);
            let q = cx.real(q);

            if is_zero(q) {
                let c = if lt_zero(p) { -one() } else { one() };
                let s = zero();
                let c = cx.from_real(c);
                let s = cx.from_real(s);

                Self { c, s }
            } else if is_zero(p) {
                let c = zero();
                let s = if lt_zero(q) { -one() } else { one() };
                let c = cx.from_real(c);
                let s = cx.from_real(s);

                Self { c, s }
            } else if abs(p) > abs(q) {
                let t = q / p;
                let mut u = hypot(one(), t);
                if lt_zero(p) {
                    u = -u;
                }
                let c = recip(u);
                let s = -t * c;

                let c = cx.from_real(c);
                let s = cx.from_real(s);

                Self { c, s }
            } else {
                let t = p / q;
                let mut u = hypot(one(), t);
                if lt_zero(q) {
                    u = -u;
                }
                let s = -recip(u);
                let c = -t * s;

                let c = cx.from_real(c);
                let s = cx.from_real(s);

                Self { c, s }
            }
        })
    }

    #[math]
    pub fn rotg(ctx: &Ctx<C, T>, p: C::Of<T>, q: C::Of<T>) -> (Self, C::Of<T>) {
        if const { T::IS_REAL } {
            math.re({
                let p = cx.real(p);
                let q = cx.real(q);

                if is_zero(q) {
                    let c = one();
                    let s = zero();
                    let c = cx.from_real(c);
                    let s = cx.from_real(s);

                    (Self { c, s }, cx.from_real(p))
                } else if is_zero(p) {
                    let c = zero();
                    let s = one();
                    let c = cx.from_real(c);
                    let s = cx.from_real(s);

                    (Self { c, s }, cx.from_real(q))
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
                    let c = cx.from_real(a / r);
                    let s = cx.from_real(b / r);
                    let r = cx.from_real(r);

                    (Self { c, s }, r)
                }
            })
        } else {
            math({
                let a = p;
                let b = q;

                let eps = eps();
                let sml = min_positive();
                let big = max_positive();
                let rtmin = re.sqrt(re.div(sml, eps));
                let rtmax = re.recip(rtmin);

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
                    let g1 = max(re.abs(real(b)), re.abs(imag(b)));
                    if g1 > rtmin && g1 < rtmax {
                        // Use unscaled algorithm
                        let g2 = abs2(b);

                        let d = re.sqrt(g2);
                        s = mul_real(conj(b), re.recip(d));
                        r = from_real(d);
                    } else {
                        // Use scaled algorithm
                        let u = min(big, max(sml, g1));
                        let uu = re.recip(u);
                        let gs = mul_real(b, uu);
                        let g2 = abs2(gs);
                        let d = re.sqrt(g2);
                        s = mul_real(conj(gs), re.recip(d));
                        r = from_real(re.mul(d, u));
                    }
                } else {
                    let f1 = max(re.abs(real(a)), re.abs(imag(a)));
                    let g1 = max(re.abs(real(b)), re.abs(imag(b)));
                    if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
                        // Use unscaled algorithm
                        let f2 = abs2(a);
                        let g2 = abs2(b);
                        let h2 = re.add(f2, g2);
                        let d = if f2 > rtmin && h2 < rtmax {
                            re.sqrt(re.mul(f2, h2))
                        } else {
                            re.mul(re.sqrt(f2), re.sqrt(h2))
                        };
                        let p = re.recip(d);
                        c = from_real(re.mul(f2, p));
                        s = conj(b) * mul_real(a, p);

                        r = mul_real(a, re.mul(h2, p));
                    } else {
                        // Use scaled algorithm
                        let u = min(big, max(sml, max(f1, g1)));
                        let uu = re.recip(u);
                        let gs = mul_real(b, uu);
                        let g2 = abs2(gs);
                        let (f2, h2, w);
                        let fs;
                        if re.mul(f1, uu) < rtmin {
                            // a is not well-scaled when scaled by g1.
                            let v = min(big, max(sml, f1));
                            let vv = re.recip(v);
                            w = re.mul(v, uu);
                            fs = mul_real(a, vv);
                            f2 = abs2(fs);
                            h2 = re.add(re.mul(re.mul(f2, w), w), g2);
                        } else {
                            // Otherwise use the same scaling for a and b.
                            w = re.one();
                            fs = mul_real(a, uu);
                            f2 = abs2(fs);
                            h2 = re.add(f2, g2);
                        }
                        let d = if f2 > rtmin && h2 < rtmax {
                            re.sqrt(re.mul(f2, h2))
                        } else {
                            re.mul(re.sqrt(f2), re.sqrt(h2))
                        };
                        let p = re.recip(d);
                        c = from_real(re.mul(re.mul(f2, p), w));
                        s = conj(gs) * mul_real(fs, p);
                        r = mul_real(mul_real(fs, re.mul(h2, p)), u);
                    }
                }

                (Self { c, s }, r)
            })
        }
    }

    #[inline]
    #[math]
    pub fn apply_on_the_left_2x2(
        &self,
        ctx: &Ctx<C, T>,
        m00: C::Of<T>,
        m01: C::Of<T>,
        m10: C::Of<T>,
        m11: C::Of<T>,
    ) -> (C::Of<T>, C::Of<T>, C::Of<T>, C::Of<T>) {
        let Self { c, s } = self;

        math((
            m00 * c + m10 * s,
            m01 * c + m11 * s,
            c * m10 - conj(s) * m00,
            c * m11 - conj(s) * m01,
        ))
    }

    #[inline]
    pub fn apply_on_the_right_2x2(
        &self,
        ctx: &Ctx<C, T>,
        m00: C::Of<T>,
        m01: C::Of<T>,
        m10: C::Of<T>,
        m11: C::Of<T>,
    ) -> (C::Of<T>, C::Of<T>, C::Of<T>, C::Of<T>) {
        let (r00, r01, r10, r11) = self
            .adjoint(ctx)
            .apply_on_the_left_2x2(ctx, m00, m10, m01, m11);
        (r00, r10, r01, r11)
    }

    #[inline]
    fn apply_on_the_left_in_place_impl<'N>(
        &self,
        ctx: &Ctx<C, T>,
        (x, y): (RowMut<'_, C, T, Dim<'N>>, RowMut<'_, C, T, Dim<'N>>),
    ) {
        let mut x = x;
        let mut y = y;
        if const { T::SIMD_CAPABILITIES.is_simd() } {
            struct Impl<'a, 'N, C: ComplexContainer, T: ComplexField<C>> {
                this: &'a JacobiRotation<C, T>,
                ctx: &'a Ctx<C, T>,
                x: RowMut<'a, C, T, Dim<'N>, ContiguousFwd>,
                y: RowMut<'a, C, T, Dim<'N>, ContiguousFwd>,
            }
            impl<C: ComplexContainer, T: ComplexField<C>> pulp::WithSimd for Impl<'_, '_, C, T> {
                type Output = ();

                #[inline(always)]
                fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                    let Self { this, ctx, x, y } = self;
                    let simd = SimdCtx::new(T::simd_ctx(ctx, simd), x.ncols());
                    this.apply_on_the_left_in_place_with_simd(simd, ctx, x, y);
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
                T::Arch::default().dispatch(Impl {
                    this: self,
                    ctx,
                    x,
                    y,
                });
                return;
            }
        }
        self.apply_on_the_left_in_place_fallback(ctx, x, y);
    }

    #[inline]
    pub fn apply_on_the_left_in_place<N: Shape>(
        &self,
        ctx: &Ctx<C, T>,
        (x, y): (RowMut<'_, C, T, N>, RowMut<'_, C, T, N>),
    ) {
        with_dim!(N, x.ncols().unbound());
        self.apply_on_the_left_in_place_impl(ctx, (x.as_col_shape_mut(N), y.as_col_shape_mut(N)));
    }

    #[inline]
    pub fn apply_on_the_right_in_place<N: Shape>(
        &self,
        ctx: &Ctx<C, T>,
        (x, y): (ColMut<'_, C, T, N>, ColMut<'_, C, T, N>),
    ) {
        with_dim!(N, x.nrows().unbound());

        let x = x.as_row_shape_mut(N);
        let y = y.as_row_shape_mut(N);

        self.transpose(ctx)
            .apply_on_the_left_in_place(ctx, (x.transpose_mut(), y.transpose_mut()));
    }

    #[inline(never)]
    #[math]
    fn apply_on_the_left_in_place_fallback<'N>(
        &self,
        ctx: &Ctx<C, T>,
        x: RowMut<'_, C, T, Dim<'N>>,
        y: RowMut<'_, C, T, Dim<'N>>,
    ) {
        help!(C);
        let Self { c, s } = self;
        math(zipped!(x, y).for_each(move |unzipped!(mut x, mut y)| {
            let x_ = c * x - conj(s) * y;
            let y_ = c * y + s * x;
            write1!(x, x_);
            write1!(y, y_);
        }));
    }

    #[inline(always)]
    pub fn apply_on_the_right_in_place_with_simd<'N, S: pulp::Simd>(
        &self,
        simd: SimdCtx<'N, C, T, S>,
        ctx: &Ctx<C, T>,
        x: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
        y: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    ) {
        self.transpose(ctx).apply_on_the_left_in_place_with_simd(
            simd,
            ctx,
            x.transpose_mut(),
            y.transpose_mut(),
        );
    }

    #[math]
    #[inline(always)]
    pub fn apply_on_the_left_in_place_with_simd<'N, S: pulp::Simd>(
        &self,
        simd: SimdCtx<'N, C, T, S>,
        ctx: &Ctx<C, T>,
        x: RowMut<'_, C, T, Dim<'N>, ContiguousFwd>,
        y: RowMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    ) {
        let Self { c, s } = self;
        help!(C);
        help2!(C::Real);

        math(if c == one() && s == zero() {
            return;
        });

        let mut x = x.transpose_mut();
        let mut y = y.transpose_mut();

        let c = simd.splat_real(as_ref2!(math.real(c)));
        let s = simd.splat(as_ref!(s));

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
    pub fn adjoint(&self, ctx: &Ctx<C, T>) -> Self {
        math(Self {
            c: copy(self.c),
            s: -conj(self.s),
        })
    }

    #[inline]
    #[math]
    pub fn conjugate(&self, ctx: &Ctx<C, T>) -> Self {
        math(Self {
            c: copy(self.c),
            s: conj(self.s),
        })
    }

    #[inline]
    #[math]
    pub fn transpose(&self, ctx: &Ctx<C, T>) -> Self {
        math(Self {
            c: copy(self.c),
            s: -self.s,
        })
    }
}
