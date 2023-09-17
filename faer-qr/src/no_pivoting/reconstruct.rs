use assert2::assert;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{
    householder::apply_block_householder_sequence_on_the_left_in_place_with_conj, temp_mat_req,
    temp_mat_uninit, zipped, ComplexField, Conj, Entity, MatMut, MatRef, Parallelism,
};
use reborrow::*;

/// Computes the reconstructed matrix, given its QR decomposition, and stores the
/// result in `dst`.
///
/// # Panics
///
/// - Panics if the number of columns of `householder_factor` isn't the same as the minimum of the
/// number of rows and the number of columns of `qr_factors`.
/// - Panics if the block size is zero.
/// - Panics if `dst` doesn't have the same shape as `qr_factors`.
/// - Panics if the provided memory in `stack` is insufficient (see [`reconstruct_req`]).
#[track_caller]
pub fn reconstruct<E: ComplexField>(
    dst: MatMut<'_, E>,
    qr_factors: MatRef<'_, E>,
    householder_factor: MatRef<'_, E>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    assert!((dst.nrows(), dst.ncols()) == (qr_factors.nrows(), qr_factors.ncols()));
    assert!(householder_factor.ncols() == Ord::min(qr_factors.nrows(), qr_factors.ncols()));
    assert!(householder_factor.nrows() > 0);

    let mut dst = dst;

    // copy R
    zipped!(dst.rb_mut(), qr_factors)
        .for_each_triangular_upper(faer_core::zip::Diag::Include, |mut dst, src| {
            dst.write(src.read())
        });

    // zero bottom part
    zipped!(dst.rb_mut())
        .for_each_triangular_lower(faer_core::zip::Diag::Skip, |mut dst| dst.write(E::zero()));

    apply_block_householder_sequence_on_the_left_in_place_with_conj(
        qr_factors,
        householder_factor,
        Conj::No,
        dst.rb_mut(),
        parallelism,
        stack,
    );
}

/// Computes the reconstructed matrix, given its QR decomposition, and stores the
/// result in `qr_factors`.
///
/// # Panics
///
/// - Panics if the number of columns of `householder_factor` isn't the same as the minimum of the
/// number of rows and the number of columns of `qr_factors`.
/// - Panics if the block size is zero.
/// - Panics if the provided memory in `stack` is insufficient (see [`reconstruct_in_place_req`]).
#[track_caller]
pub fn reconstruct_in_place<E: ComplexField>(
    qr_factors: MatMut<'_, E>,
    householder_factor: MatRef<'_, E>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let (mut dst, stack) =
        unsafe { temp_mat_uninit::<E>(qr_factors.nrows(), qr_factors.ncols(), stack) };
    let mut dst = dst.as_mut();

    reconstruct(
        dst.rb_mut(),
        qr_factors.rb(),
        householder_factor,
        parallelism,
        stack,
    );

    zipped!(qr_factors, dst.rb()).for_each(|mut dst, src| dst.write(src.read()));
}

/// Computes the size and alignment of required workspace for reconstructing a matrix out of place,
/// given its QR decomposition.
pub fn reconstruct_req<E: Entity>(
    qr_nrows: usize,
    qr_ncols: usize,
    blocksize: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = qr_nrows;
    let _ = parallelism;
    temp_mat_req::<E>(blocksize, qr_ncols)
}

/// Computes the size and alignment of required workspace for reconstructing a matrix in place,
/// given its QR decomposition.
pub fn reconstruct_in_place_req<E: Entity>(
    qr_nrows: usize,
    qr_ncols: usize,
    blocksize: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_all_of([
        temp_mat_req::<E>(qr_nrows, qr_ncols)?,
        reconstruct_req::<E>(qr_nrows, qr_ncols, blocksize, parallelism)?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::no_pivoting::compute::{qr_in_place, qr_in_place_req, recommended_blocksize};
    use assert2::assert;
    use assert_approx_eq::assert_approx_eq;
    use faer_core::{c64, Mat};
    use rand::prelude::*;
    use std::cell::RefCell;

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::DynStack::new(&mut ::dyn_stack::GlobalMemBuffer::new($req.unwrap()))
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
    fn test_reconstruct() {
        for n in [31, 32, 48, 65] {
            let mat = Mat::with_dims(n, n, |_, _| random_value());
            let blocksize = recommended_blocksize::<E>(n, n);
            let mut qr = mat.clone();
            let mut householder_factor = Mat::zeros(blocksize, n);

            let parallelism = faer_core::Parallelism::Rayon(0);

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

            let mut reconstructed = Mat::zeros(n, n);
            reconstruct(
                reconstructed.as_mut(),
                qr.as_ref(),
                householder_factor.as_ref(),
                parallelism,
                make_stack!(reconstruct_req::<E>(n, n, blocksize, parallelism)),
            );

            for i in 0..n {
                for j in 0..n {
                    assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j));
                }
            }
        }
    }
}
