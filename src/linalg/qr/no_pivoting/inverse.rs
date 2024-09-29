use crate::{
    assert,
    linalg::{
        householder::apply_block_householder_sequence_transpose_on_the_right_in_place_with_conj,
        temp_mat_req, temp_mat_uninit, triangular_inverse::invert_upper_triangular,
    },
    unzipped, zipped, ComplexField, Conj, Entity, MatMut, MatRef, Parallelism,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use reborrow::*;

/// Computes the inverse of a matrix, given its QR decomposition,
/// and stores the result in `dst`.
///
/// # Panics
///
/// - Panics if `qr_factors` is not a square matrix.
/// - Panics if the number of columns of `householder_factor` isn't the same as the minimum of the
/// number of rows and the number of columns of `qr_factors`.
/// - Panics if the block size is zero.
/// - Panics if `dst` doesn't have the same shape as `qr_factors`.
/// - Panics if the provided memory in `stack` is insufficient (see [`invert_req`]).
#[track_caller]
pub fn invert<E: ComplexField>(
    dst: MatMut<'_, E>,
    qr_factors: MatRef<'_, E>,
    householder_factor: MatRef<'_, E>,
    parallelism: Parallelism,
    stack: &mut PodStack,
) {
    assert!(all(
        qr_factors.nrows() == qr_factors.ncols(),
        dst.ncols() == qr_factors.ncols(),
        dst.nrows() == qr_factors.nrows(),
        householder_factor.ncols() == Ord::min(qr_factors.nrows(), qr_factors.ncols()),
        householder_factor.nrows() > 0
    ));

    let mut dst = dst;

    // invert R
    invert_upper_triangular(dst.rb_mut(), qr_factors, parallelism);

    // zero bottom part
    zipped!(dst.rb_mut())
        .for_each_triangular_lower(crate::linalg::zip::Diag::Skip, |unzipped!(mut dst)| {
            dst.write(E::faer_zero())
        });

    apply_block_householder_sequence_transpose_on_the_right_in_place_with_conj(
        qr_factors,
        householder_factor,
        Conj::Yes,
        dst.rb_mut(),
        parallelism,
        stack,
    );
}

/// Computes the inverse of a matrix, given its QR decomposition,
/// and stores the result in `qr_factors`.
///
/// # Panics
///
/// - Panics if `qr_factors` is not a square matrix.
/// - Panics if the number of columns of `householder_factor` isn't the same as the minimum of the
/// number of rows and the number of columns of `qr_factors`.
/// - Panics if the block size is zero.
/// - Panics if the provided memory in `stack` is insufficient (see [`invert_in_place_req`]).
#[track_caller]
pub fn invert_in_place<E: ComplexField>(
    qr_factors: MatMut<'_, E>,
    householder_factor: MatRef<'_, E>,
    parallelism: Parallelism,
    stack: &mut PodStack,
) {
    let (mut dst, stack) = temp_mat_uninit::<E>(qr_factors.nrows(), qr_factors.ncols(), stack);
    let mut dst = dst.as_mut();

    invert(
        dst.rb_mut(),
        qr_factors.rb(),
        householder_factor,
        parallelism,
        stack,
    );

    zipped!(qr_factors, dst.rb()).for_each(|unzipped!(mut dst, src)| dst.write(src.read()));
}

/// Computes the size and alignment of required workspace for computing the inverse of a
/// matrix out of place, given its QR decomposition.
pub fn invert_req<E: Entity>(
    qr_nrows: usize,
    qr_ncols: usize,
    blocksize: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = qr_nrows;
    let _ = parallelism;
    temp_mat_req::<E>(blocksize, qr_ncols)
}

/// Computes the size and alignment of required workspace for computing the inverse of a
/// matrix in place, given its QR decomposition.
pub fn invert_in_place_req<E: Entity>(
    qr_nrows: usize,
    qr_ncols: usize,
    blocksize: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_all_of([
        temp_mat_req::<E>(qr_nrows, qr_ncols)?,
        invert_req::<E>(qr_nrows, qr_ncols, blocksize, parallelism)?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assert,
        complex_native::c64,
        linalg::{
            matmul::matmul,
            qr::no_pivoting::compute::{qr_in_place, qr_in_place_req, recommended_blocksize},
        },
        Mat,
    };
    use assert_approx_eq::assert_approx_eq;
    use rand::prelude::*;
    use std::cell::RefCell;

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::PodStack::new(&mut ::dyn_stack::GlobalPodBuffer::new($req.unwrap()))
        };
    }

    type E = c64;

    thread_local! {
        static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(0));
    }

    fn random_value() -> E {
        RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            let rng = &mut *rng;
            E {
                re: rng.gen(),
                im: rng.gen(),
            }
        })
    }

    #[test]
    fn test_invert() {
        for n in [31, 32, 48, 65] {
            let mat = Mat::from_fn(n, n, |_, _| random_value());
            let blocksize = recommended_blocksize::<E>(n, n);
            let mut qr = mat.clone();
            let mut householder_factor = Mat::zeros(blocksize, n);

            let parallelism = crate::Parallelism::Rayon(0);

            qr_in_place(
                qr.as_mut(),
                householder_factor.as_mut(),
                parallelism,
                make_stack!(qr_in_place_req::<E>(
                    n,
                    n,
                    blocksize,
                    parallelism,
                    Default::default()
                )),
                Default::default(),
            );

            let mut inv = Mat::zeros(n, n);
            invert(
                inv.as_mut(),
                qr.as_ref(),
                householder_factor.as_ref(),
                parallelism,
                make_stack!(invert_req::<E>(n, n, blocksize, parallelism)),
            );

            let mut eye = Mat::zeros(n, n);
            matmul(
                eye.as_mut(),
                inv.as_ref(),
                mat.as_ref(),
                None,
                E::faer_one(),
                Parallelism::None,
            );

            for i in 0..n {
                for j in 0..n {
                    let target = if i == j {
                        E::faer_one()
                    } else {
                        E::faer_zero()
                    };
                    assert_approx_eq!(eye.read(i, j), target);
                }
            }
        }
    }
}
