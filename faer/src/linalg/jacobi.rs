use crate::internal_prelude::*;

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct JacobiRotation<C: ComplexContainer, T: ComplexField<C>> {
    pub c: C::Of<T>,
    pub s: C::Of<T>,
}

impl<C: RealContainer, T: RealField<C>> JacobiRotation<C, T> {
    #[inline]
    #[math]
    pub fn make_givens(ctx: &Ctx<C, T>, p: C::Of<T>, q: C::Of<T>) -> Self {
        math({
            if is_zero(q) {
                Self {
                    c: if lt_zero(p) { -one() } else { one() },
                    s: zero(),
                }
            } else if is_zero(p) {
                Self {
                    c: zero(),
                    s: if lt_zero(q) { -one() } else { one() },
                }
            } else if abs(p) > abs(q) {
                let t = q / p;
                let mut u = hypot(one(), t);
                if lt_zero(p) {
                    u = -u;
                }
                let c = recip(u);
                let s = -t * c;

                Self { c, s }
            } else {
                let t = p / q;
                let mut u = hypot(one(), t);
                if lt_zero(q) {
                    u = -u;
                }
                let s = -recip(u);
                let c = -t * s;

                Self { c, s }
            }
        })
    }

    #[inline]
    #[math]
    pub fn from_triplet(ctx: &Ctx<C, T>, x: C::Of<T>, y: C::Of<T>, z: C::Of<T>) -> Self {
        math({
            let abs_y = abs(y);
            let two_abs_y = abs_y + abs_y;
            if is_zero(two_abs_y) {
                Self {
                    c: one(),
                    s: zero(),
                }
            } else {
                let tau = (x - z) * recip(two_abs_y);
                let w = hypot(one(), tau);
                let t = if gt_zero(tau) {
                    recip(tau + w)
                } else {
                    recip(tau - w)
                };

                let neg_sign_y = if gt_zero(y) { -one() } else { one() };
                let n = recip(hypot(one(), t));

                Self {
                    c: copy(n),
                    s: neg_sign_y * t * n,
                }
            }
        })
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
            c * m10 - s * m00,
            c * m11 - s * m01,
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
            .transpose(ctx)
            .apply_on_the_left_2x2(ctx, m00, m10, m01, m11);
        (r00, r10, r01, r11)
    }

    #[inline]
    pub fn apply_on_the_left_in_place<'N>(
        &self,
        ctx: &Ctx<C, T>,
        x: RowMut<'_, C, T, Dim<'N>>,
        y: RowMut<'_, C, T, Dim<'N>>,
    ) {
        let mut x = x;
        let mut y = y;
        if const { T::SIMD_CAPABILITIES.is_simd() } {
            struct Impl<'a, 'N, C: RealContainer, T: RealField<C>> {
                this: &'a JacobiRotation<C, T>,
                ctx: &'a Ctx<C, T>,
                x: RowMut<'a, C, T, Dim<'N>, ContiguousFwd>,
                y: RowMut<'a, C, T, Dim<'N>, ContiguousFwd>,
            }
            impl<C: RealContainer, T: RealField<C>> pulp::WithSimd for Impl<'_, '_, C, T> {
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
    pub fn apply_on_the_right_in_place<'N>(
        &self,
        ctx: &Ctx<C, T>,
        x: ColMut<'_, C, T, Dim<'N>>,
        y: ColMut<'_, C, T, Dim<'N>>,
    ) {
        self.transpose(ctx)
            .apply_on_the_left_in_place(ctx, x.transpose_mut(), y.transpose_mut());
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
            let x_ = c * x + s * y;
            let y_ = c * y - s * x;
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

        math(if c == one() && s == zero() {
            return;
        });

        let mut x = x.transpose_mut();
        let mut y = y.transpose_mut();

        let c = simd.splat(as_ref!(c));
        let s = simd.splat(as_ref!(s));

        let (head, body, tail) = simd.indices();

        if let Some(i) = head {
            let mut xx = simd.read(x.rb(), i);
            let mut yy = simd.read(y.rb(), i);

            xx = simd.mul_add(c, xx, simd.mul(s, yy));
            yy = simd.mul_add(c, yy, simd.neg(simd.mul(s, xx)));

            simd.write(x.rb_mut(), i, xx);
            simd.write(y.rb_mut(), i, yy);
        }
        for i in body {
            let mut xx = simd.read(x.rb(), i);
            let mut yy = simd.read(y.rb(), i);

            xx = simd.mul_add(c, xx, simd.mul(s, yy));
            yy = simd.mul_add(c, yy, simd.neg(simd.mul(s, xx)));

            simd.write(x.rb_mut(), i, xx);
            simd.write(y.rb_mut(), i, yy);
        }
        if let Some(i) = tail {
            let mut xx = simd.read(x.rb(), i);
            let mut yy = simd.read(y.rb(), i);

            xx = simd.mul_add(c, xx, simd.mul(s, yy));
            yy = simd.mul_add(c, yy, simd.neg(simd.mul(s, xx)));

            simd.write(x.rb_mut(), i, xx);
            simd.write(y.rb_mut(), i, yy);
        }
    }

    #[inline]
    #[math]
    pub fn transpose(&self, ctx: &Ctx<C, T>) -> Self {
        math(Self {
            c: copy(self.c),
            s: -self.s,
        })
    }

    #[inline]
    #[math]
    pub fn mul(&self, ctx: &Ctx<C, T>, rhs: &Self) -> Self {
        math(Self {
            c: self.c * rhs.c - self.s * rhs.s,
            s: self.c * rhs.s + self.s * rhs.c,
        })
    }
}
