use crate::{zipped, MatMut, RealField};

#[derive(Copy, Clone, Debug)]
pub struct JacobiRotation<T> {
    pub c: T,
    pub s: T,
}

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
            let t = q.div(&p);
            let mut u = E::one().add(&t.abs2()).sqrt();
            if p < E::zero() {
                u = u.neg();
            }
            let c = u.inv();
            let s = t.neg().mul(&c);

            Self { c, s }
        } else {
            let t = p.div(&q);
            let mut u = E::one().add(&t.abs2()).sqrt();
            if q < E::zero() {
                u = u.neg();
            }
            let s = u.inv().neg();
            let c = t.neg().mul(&s);

            Self { c, s }
        }
    }

    #[inline]
    pub fn from_triplet(x: E, y: E, z: E) -> Self {
        let abs_y = y.abs();
        let two_abs_y = abs_y.add(&abs_y);
        if two_abs_y == E::zero() {
            Self {
                c: E::one(),
                s: E::zero(),
            }
        } else {
            let tau = (x.sub(&z)).mul(&two_abs_y.inv());
            let w = ((tau.mul(&tau)).add(&E::one())).sqrt();
            let t = if tau > E::zero() {
                (tau.add(&w)).inv()
            } else {
                (tau.sub(&w)).inv()
            };

            let neg_sign_y = if y > E::zero() {
                E::one().neg()
            } else {
                E::one()
            };
            let n = (t.mul(&t).add(&E::one())).sqrt().inv();

            Self {
                c: n.clone(),
                s: neg_sign_y.mul(&t).mul(&n),
            }
        }
    }

    #[inline]
    pub fn apply_on_the_left_2x2(&self, m00: E, m01: E, m10: E, m11: E) -> (E, E, E, E) {
        let Self { c, s } = self;
        (
            m00.mul(c).add(&m10.mul(s)),
            m01.mul(c).add(&m11.mul(s)),
            s.neg().mul(&m00).add(&c.mul(&m10)),
            s.neg().mul(&m01).add(&c.mul(&m11)),
        )
    }

    #[inline]
    pub fn apply_on_the_right_2x2(&self, m00: E, m01: E, m10: E, m11: E) -> (E, E, E, E) {
        let (r00, r01, r10, r11) = self.transpose().apply_on_the_left_2x2(m00, m10, m01, m11);
        (r00, r10, r01, r11)
    }

    #[inline]
    pub fn apply_on_the_left_in_place(&self, x: MatMut<'_, E>, y: MatMut<'_, E>) {
        pulp::Arch::new().dispatch(
            #[inline(always)]
            move || {
                assert!(x.nrows() == 1);

                let Self { c, s } = self;
                if *c == E::one() && *s == E::zero() {
                    return;
                }

                zipped!(x, y).for_each(move |mut x, mut y| {
                    let x_ = x.read();
                    let y_ = y.read();
                    x.write(c.mul(&x_).add(&s.mul(&y_)));
                    y.write(s.neg().mul(&x_).add(&c.mul(&y_)));
                });
            },
        )
    }

    #[inline]
    pub fn apply_on_the_right_in_place(&self, x: MatMut<'_, E>, y: MatMut<'_, E>) {
        self.transpose()
            .apply_on_the_left_in_place(x.transpose(), y.transpose());
    }

    #[inline]
    pub fn transpose(&self) -> Self {
        Self {
            c: self.c.clone(),
            s: self.s.neg(),
        }
    }
}

impl<E: RealField> core::ops::Mul for JacobiRotation<E> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            c: self.c.mul(&rhs.c).sub(&self.s.mul(&rhs.s)),
            s: self.c.mul(&rhs.s).add(&self.s.mul(&rhs.c)),
        }
    }
}
