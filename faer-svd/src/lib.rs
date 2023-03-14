use core::{
    any::TypeId,
    mem::{transmute, transmute_copy},
};
use dyn_stack::DynStack;
use faer_core::{
    householder::{
        apply_block_householder_sequence_on_the_left_in_place, upgrade_householder_factor,
    },
    temp_mat_uninit, zip, ColMut, ComplexField, Conj, MatMut, MatRef, Parallelism, RealField,
};
use reborrow::*;

#[doc(hidden)]
pub mod bidiag;
#[doc(hidden)]
pub mod bidiag_real_svd;
#[doc(hidden)]
pub mod jacobi;

fn compute_real_svd<T: RealField>(
    matrix: MatRef<'_, T>,
    s: ColMut<'_, T>,
    u: Option<MatMut<'_, T>>,
    v: Option<MatMut<'_, T>>,
    epsilon: T,
    zero_threshold: T,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let mut stack = stack;
    let mut u = u;
    let mut v = v;

    let do_transpose = matrix.ncols() > matrix.nrows();

    let matrix = if do_transpose {
        matrix.transpose()
    } else {
        matrix
    };

    if do_transpose {
        core::mem::swap(&mut u, &mut v);
    }

    let m = matrix.nrows();
    let n = matrix.ncols();
    if n == 0 {
        return;
    }

    let householder_blocksize = 32;

    temp_mat_uninit! {
        let (mut bid, stack) = unsafe { temp_mat_uninit::<T>(m, n, stack.rb_mut()) };
        let (mut householder_left, stack) = unsafe { temp_mat_uninit::<T>(householder_blocksize, n, stack) };
        let (mut householder_right, mut stack) = unsafe { temp_mat_uninit::<T>(householder_blocksize, n - 1, stack) };
    }

    zip!(bid.rb_mut(), matrix).for_each(|dst, src| *dst = *src);

    bidiag::bidiagonalize_in_place(
        bid.rb_mut(),
        householder_left.rb_mut().row(0).transpose(),
        householder_right.rb_mut().row(0).transpose(),
        parallelism,
        stack.rb_mut(),
    );

    let bid = bid.into_const();

    let (mut diag, stack) = stack.make_with(n, |i| bid[(i, i)]);
    let (mut subdiag, stack) = stack.make_with(n, |i| {
        if i < n - 1 {
            bid[(i, i + 1)]
        } else {
            T::zero()
        }
    });

    temp_mat_uninit! {
        let (mut u_b, stack) = unsafe { temp_mat_uninit::<T>(if v.is_some() { n + 1 } else { 2 }, n + 1, stack) };
        let (mut v_b, mut stack) = unsafe { temp_mat_uninit::<T>(n, if u.is_some() { n } else { 0 }, stack) };
    }

    let jacobi_fallback_threshold = 4;

    let mut j_base = 0;
    while j_base < n {
        let bs = householder_blocksize.min(n - j_base);
        let mut householder = householder_left.rb_mut().submatrix(0, j_base, bs, bs);
        let essentials = bid.submatrix(j_base, j_base, m - j_base, bs);
        for j in 0..bs {
            householder[(j, j)] = householder[(0, j)];
        }
        upgrade_householder_factor(householder, essentials, bs, 1, parallelism);
        j_base += bs;
    }
    let mut j_base = 0;
    while j_base < n - 1 {
        let bs = householder_blocksize.min(n - 1 - j_base);
        let mut householder = householder_right.rb_mut().submatrix(0, j_base, bs, bs);
        let full_essentials = bid.submatrix(0, 1, m, n - 1).transpose();
        let essentials = full_essentials.submatrix(j_base, j_base, n - 1 - j_base, bs);
        for j in 0..bs {
            householder[(j, j)] = householder[(0, j)];
        }
        upgrade_householder_factor(householder, essentials, bs, 1, parallelism);
        j_base += bs;
    }

    bidiag_real_svd::bidiag_svd(
        &mut diag,
        &mut subdiag,
        u_b.rb_mut(),
        v.is_some().then_some(v_b.rb_mut()),
        jacobi_fallback_threshold,
        epsilon,
        zero_threshold,
        parallelism,
        stack.rb_mut(),
    );

    for (s, val) in s.into_iter().zip(&*diag) {
        *s = *val;
    }

    if let Some(mut u) = u {
        let ncols = u.ncols();
        zip!(
            u.rb_mut().submatrix(0, 0, n, n),
            v_b.rb().submatrix(0, 0, n, n),
        )
        .for_each(|dst, src| *dst = *src);

        zip!(u.rb_mut().submatrix(n, 0, m - n, ncols)).for_each(|x| *x = T::zero());
        zip!(u.rb_mut().submatrix(0, n, n, ncols - n)).for_each(|x| *x = T::zero());
        zip!(u.rb_mut().submatrix(n, n, ncols - n, ncols - n).diagonal())
            .for_each(|x| *x = T::one());

        apply_block_householder_sequence_on_the_left_in_place(
            bid,
            householder_left.rb(),
            Conj::No,
            u,
            Conj::No,
            parallelism,
            stack.rb_mut(),
        );
    };
    if let Some(mut v) = v {
        zip!(
            v.rb_mut().submatrix(0, 0, n, n),
            u_b.rb().submatrix(0, 0, n, n),
        )
        .for_each(|dst, src| *dst = *src);

        apply_block_householder_sequence_on_the_left_in_place(
            bid.submatrix(0, 1, m, n - 1).transpose(),
            householder_right.rb(),
            Conj::No,
            v.submatrix(1, 0, n - 1, n),
            Conj::No,
            parallelism,
            stack.rb_mut(),
        );
    }
}

pub fn compute_svd<T: ComplexField>(
    matrix: MatRef<'_, T>,
    s: ColMut<'_, T>,
    u: Option<MatMut<'_, T>>,
    v: Option<MatMut<'_, T>>,
    epsilon: T,
    zero_threshold: T,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    if TypeId::of::<T>() == TypeId::of::<T::Real>() {
        unsafe {
            let matrix: MatRef<'_, T::Real> = transmute(matrix);
            compute_real_svd(
                matrix,
                transmute(s),
                transmute(u),
                transmute(v),
                transmute_copy(&epsilon),
                transmute_copy(&zero_threshold),
                parallelism,
                stack,
            );
        }
    } else {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use faer_core::Mat;

    macro_rules! placeholder_stack {
        () => {
            ::dyn_stack::DynStack::new(&mut ::dyn_stack::GlobalMemBuffer::new(
                ::dyn_stack::StackReq::new::<f64>(1024 * 1024 * 1024),
            ))
        };
    }

    #[test]
    fn test_real() {
        for (m, n) in [(15, 10), (10, 15), (10, 10), (15, 15)] {
            let mat = Mat::with_dims(|_, _| rand::random::<f64>(), m, n);
            let size = m.min(n);

            let mut s = Mat::zeros(m, n);
            let mut u = Mat::zeros(m, m);
            let mut v = Mat::zeros(n, n);

            compute_svd(
                mat.as_ref(),
                s.as_mut().submatrix(0, 0, size, size).diagonal(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                f64::EPSILON,
                f64::MIN_POSITIVE,
                Parallelism::None,
                placeholder_stack!(),
            );

            let reconstructed = &u * &s * v.transpose();

            for j in 0..n {
                for i in 0..m {
                    assert_approx_eq!(reconstructed[(i, j)], mat[(i, j)], 1e-10);
                }
            }
        }
    }
}
