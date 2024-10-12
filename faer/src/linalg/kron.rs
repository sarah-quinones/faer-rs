use crate::internal_prelude::*;

#[track_caller]
#[math]
pub fn kron<C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    dst: MatMut<C, T>,
    lhs: MatRef<C, T>,
    rhs: MatRef<C, T>,
) {
    // pull the lever kron

    let mut dst = dst;
    let mut lhs = lhs;
    let mut rhs = rhs;
    if dst.col_stride().unsigned_abs() < dst.row_stride().unsigned_abs() {
        dst = dst.transpose_mut();
        lhs = lhs.transpose();
        rhs = rhs.transpose();
    }

    Assert!(Some(dst.nrows()) == lhs.nrows().checked_mul(rhs.nrows()));
    Assert!(Some(dst.ncols()) == lhs.ncols().checked_mul(rhs.ncols()));

    for lhs_j in 0..lhs.ncols() {
        for lhs_i in 0..lhs.nrows() {
            let lhs_val = lhs.at(lhs_i, lhs_j);
            let mut dst = dst.rb_mut().submatrix_mut(
                lhs_i * rhs.nrows(),
                lhs_j * rhs.ncols(),
                rhs.nrows(),
                rhs.ncols(),
            );

            for rhs_j in 0..rhs.ncols() {
                for rhs_i in 0..rhs.nrows() {
                    // SAFETY: Bounds have been checked.
                    unsafe {
                        let rhs_val = rhs.at_unchecked(rhs_i, rhs_j);
                        help!(C);
                        write1!(
                            dst.rb_mut().at_mut_unchecked(rhs_i, rhs_j),
                            math(lhs_val * rhs_val)
                        );
                    }
                }
            }
        }
    }
    // the other lever
}

#[cfg(test)]
mod tests {
    use super::kron;
    use crate::{assert, internal_prelude::*, Col, Mat, Row};

    #[test]
    fn test_kron_ones() {
        for (m, n, p, q) in [(2, 3, 4, 5), (3, 2, 5, 4), (1, 1, 1, 1)] {
            let a = Mat::from_fn(m, n, |_, _| 1 as f64);
            let b = Mat::from_fn(p, q, |_, _| 1 as f64);
            let expected = Mat::from_fn(m * p, n * q, |_, _| 1 as f64);
            let mut out =
                Mat::zeros_with_ctx(&default(), a.nrows() * b.nrows(), a.ncols() * b.ncols());
            kron(&default(), out.as_mut(), a.as_ref(), b.as_ref());
            assert!(out == expected);
        }

        for (m, n, p) in [(2, 3, 4), (3, 2, 5), (1, 1, 1)] {
            let a = Mat::from_fn(m, n, |_, _| 1 as f64);
            let b = Col::from_fn(p, |_| 1 as f64);
            let expected = Mat::from_fn(m * p, n, |_, _| 1 as f64);
            let mut out =
                Mat::zeros_with_ctx(&default(), a.nrows() * b.nrows(), a.ncols() * b.ncols());
            kron(&default(), out.as_mut(), a.as_ref(), b.as_ref().as_mat());
            assert!(out == expected);
            let mut out =
                Mat::zeros_with_ctx(&default(), b.nrows() * a.nrows(), b.ncols() * a.ncols());
            kron(&default(), out.as_mut(), b.as_ref().as_mat(), a.as_ref());
            assert!(out == expected);

            let a = Mat::from_fn(m, n, |_, _| 1 as f64);
            let b = Row::from_fn(p, |_| 1 as f64);
            let expected = Mat::from_fn(m, n * p, |_, _| 1 as f64);
            let mut out =
                Mat::zeros_with_ctx(&default(), a.nrows() * b.nrows(), a.ncols() * b.ncols());
            kron(&default(), out.as_mut(), a.as_ref(), b.as_ref().as_mat());
            assert!(out == expected);
            let mut out =
                Mat::zeros_with_ctx(&default(), b.nrows() * a.nrows(), b.ncols() * a.ncols());
            kron(&default(), out.as_mut(), b.as_ref().as_mat(), a.as_ref());
            assert!(out == expected);
        }

        for (m, n) in [(2, 3), (3, 2), (1, 1)] {
            let a = Row::from_fn(m, |_| 1 as f64);
            let b = Col::from_fn(n, |_| 1 as f64);
            let expected = Mat::from_fn(n, m, |_, _| 1 as f64);
            let mut out =
                Mat::zeros_with_ctx(&default(), a.nrows() * b.nrows(), a.ncols() * b.ncols());
            kron(
                &default(),
                out.as_mut(),
                a.as_ref().as_mat(),
                b.as_ref().as_mat(),
            );
            assert!(out == expected);
            let mut out =
                Mat::zeros_with_ctx(&default(), b.nrows() * a.nrows(), b.ncols() * a.ncols());
            kron(
                &default(),
                out.as_mut(),
                b.as_ref().as_mat(),
                a.as_ref().as_mat(),
            );
            assert!(out == expected);

            let c = Row::from_fn(n, |_| 1 as f64);
            let expected = Mat::from_fn(1, m * n, |_, _| 1 as f64);
            let mut out =
                Mat::zeros_with_ctx(&default(), a.nrows() * c.nrows(), a.ncols() * c.ncols());
            kron(
                &default(),
                out.as_mut(),
                a.as_ref().as_mat(),
                c.as_ref().as_mat(),
            );
            assert!(out == expected);

            let d = Col::from_fn(m, |_| 1 as f64);
            let expected = Mat::from_fn(m * n, 1, |_, _| 1 as f64);
            let mut out =
                Mat::zeros_with_ctx(&default(), d.nrows() * b.nrows(), d.ncols() * b.ncols());
            kron(
                &default(),
                out.as_mut(),
                d.as_ref().as_mat(),
                b.as_ref().as_mat(),
            );
            assert!(out == expected);
        }
    }
}