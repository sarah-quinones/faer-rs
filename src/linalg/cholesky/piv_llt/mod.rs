use crate::{
    linalg::{
        cholesky::llt::CholeskyError, matmul::triangular::BlockStructure, temp_mat_req,
        temp_mat_uninit,
    },
    perm::PermRef,
    prelude::*,
    ComplexField, Index, Parallelism, RealField, SignedIndex,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use reborrow::*;

pub mod compute {
    use super::*;
    use equator::assert;

    #[derive(Copy, Clone, Debug)]
    #[non_exhaustive]
    pub struct PivLltParams {
        pub blocksize: usize,
    }

    impl Default for PivLltParams {
        #[inline]
        fn default() -> Self {
            Self { blocksize: 128 }
        }
    }

    #[derive(Copy, Clone, Debug)]
    #[non_exhaustive]
    pub struct PivLltInfo {
        pub rank: usize,
    }

    #[inline]
    pub fn cholesky_in_place_req<I: Index, E: ComplexField>(
        dim: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        _ = parallelism;
        temp_mat_req::<E::Real>(dim, 2)
    }

    #[track_caller]
    pub fn cholesky_in_place<'out, I: Index, E: ComplexField>(
        a: MatMut<'_, E>,
        perm: &'out mut [I],
        perm_inv: &'out mut [I],
        parallelism: Parallelism,
        stack: &mut PodStack,
        params: PivLltParams,
    ) -> Result<(PivLltInfo, PermRef<'out, I>), CholeskyError> {
        assert!(a.nrows() == a.ncols());
        let n = a.nrows();
        assert!(n <= I::Signed::MAX.zx());
        let mut rank = n;
        'exit: {
            if n > 0 {
                let mut a = a;
                for (i, p) in perm.iter_mut().enumerate() {
                    *p = I::truncate(i);
                }

                let (work1, stack) = temp_mat_uninit::<E::Real>(n, 1, stack);
                let (work2, _) = temp_mat_uninit::<E::Real>(n, 1, stack);
                let mut dot_products = work1.col_mut(0);
                let mut diagonals = work2.col_mut(0);

                let mut ajj = E::Real::faer_zero();
                let mut pvt = 0usize;

                for i in 0..n {
                    let aii = a.read(i, i).faer_real();
                    if aii < E::Real::faer_zero() || aii.faer_is_nan() {
                        return Err(CholeskyError {
                            non_positive_definite_minor: 0,
                        });
                    }
                    if aii > ajj {
                        ajj = aii;
                        pvt = i;
                    }
                }

                let tol = E::Real::faer_epsilon()
                    .faer_mul(E::Real::faer_from_f64(n as f64))
                    .faer_mul(ajj);

                let mut k = 0usize;
                while k < n {
                    let bs = Ord::min(n - k, params.blocksize);

                    for i in k..n {
                        dot_products.write(i, E::Real::faer_zero());
                    }

                    for j in k..k + bs {
                        if j == k {
                            for i in j..n {
                                diagonals.write(i, a.read(i, i).faer_real());
                            }
                        } else {
                            for i in j..n {
                                dot_products.write(
                                    i,
                                    dot_products.read(i).faer_add(a.read(i, j - 1).faer_abs2()),
                                );
                                diagonals.write(
                                    i,
                                    a.read(i, i).faer_real().faer_sub(dot_products.read(i)),
                                );
                            }
                        }

                        if j > 0 {
                            pvt = j;
                            ajj = E::Real::faer_zero();
                            for i in j..n {
                                let aii = diagonals.read(i).faer_real();
                                if aii.faer_is_nan() {
                                    return Err(CholeskyError {
                                        non_positive_definite_minor: j,
                                    });
                                }
                                if aii > ajj {
                                    pvt = i;
                                    ajj = aii;
                                }
                            }
                            if ajj < tol {
                                rank = j;
                                a.write(j, j, E::faer_from_real(ajj));
                                break 'exit;
                            }
                        }

                        if pvt != j {
                            a.write(pvt, pvt, a.read(j, j));
                            crate::perm::swap_rows_idx(a.rb_mut().get_mut(.., ..j), j, pvt);
                            crate::perm::swap_cols_idx(a.rb_mut().get_mut(pvt + 1.., ..), j, pvt);
                            unsafe {
                                zipped!(
                                    a.rb().get(j + 1..pvt, j).const_cast(),
                                    a.rb().get(pvt, j + 1..pvt).const_cast().transpose_mut(),
                                )
                            }
                            .for_each(|unzipped!(mut a, mut b)| {
                                let a_ = a.read().faer_conj();
                                let b_ = b.read().faer_conj();
                                a.write(b_);
                                b.write(a_);
                            });
                            a.write(pvt, j, a.read(pvt, j).faer_conj());

                            let tmp = dot_products.read(j);
                            dot_products.write(j, dot_products.read(pvt));
                            dot_products.write(pvt, tmp);
                            perm.swap(j, pvt);
                        }

                        ajj = ajj.faer_sqrt();
                        a.write(j, j, E::faer_from_real(ajj));
                        unsafe {
                            crate::linalg::matmul::matmul(
                                a.rb().get(j + 1.., j).const_cast(),
                                a.rb().get(j + 1.., k..j),
                                a.rb().get(j, k..j).adjoint(),
                                Some(E::faer_one()),
                                E::faer_one().faer_neg(),
                                parallelism,
                            );
                        }
                        let ajj = ajj.faer_inv();
                        zipped!(a.rb_mut().get_mut(j + 1.., j))
                            .for_each(|unzipped!(mut x)| x.write(x.read().faer_scale_real(ajj)));
                    }

                    crate::linalg::matmul::triangular::matmul(
                        unsafe { a.rb().get(k + bs.., k + bs..).const_cast() },
                        BlockStructure::TriangularLower,
                        a.rb().get(k + bs.., k..k + bs),
                        BlockStructure::Rectangular,
                        a.rb().get(k + bs.., k..k + bs).adjoint(),
                        BlockStructure::Rectangular,
                        Some(E::faer_one()),
                        E::faer_one().faer_neg(),
                        parallelism,
                    );

                    k += bs;
                }
                rank = n;
            }
        }

        for (i, p) in perm.iter().enumerate() {
            perm_inv[p.zx()] = I::truncate(i);
        }

        unsafe { Ok((PivLltInfo { rank }, PermRef::new_unchecked(perm, perm_inv))) }
    }
}
