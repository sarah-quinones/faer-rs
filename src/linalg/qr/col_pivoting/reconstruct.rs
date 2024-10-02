use crate::{
    assert,
    linalg::{
        householder::apply_block_householder_sequence_on_the_left_in_place_with_conj, temp_mat_req,
        temp_mat_uninit,
    },
    perm::{permute_cols_in_place, permute_cols_in_place_req, PermRef},
    unzipped, zipped_rw, ComplexField, Conj, Entity, Index, MatMut, MatRef, Parallelism,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use reborrow::*;

/// Computes the reconstructed matrix, given its QR decomposition, and stores the
/// result in `dst`.
///
/// # Panics
///
/// - Panics if the number of columns of `householder_factor` isn't the same as the minimum of the
/// number of rows and the number of columns of `qr_factors`.
/// - Panics if the block size is zero.
/// - Panics if `col_perm` doesn't have the same dimension as `qr_factors`.
/// - Panics if `dst` doesn't have the same shape as `qr_factors`.
/// - Panics if the provided memory in `stack` is insufficient (see [`reconstruct_req`]).
#[track_caller]
pub fn reconstruct<I: Index, E: ComplexField>(
    dst: MatMut<'_, E>,
    qr_factors: MatRef<'_, E>,
    householder_factor: MatRef<'_, E>,
    col_perm: PermRef<'_, I>,
    parallelism: Parallelism,
    stack: &mut PodStack,
) {
    assert!((dst.nrows(), dst.ncols()) == (qr_factors.nrows(), qr_factors.ncols()));
    assert!(householder_factor.ncols() == Ord::min(qr_factors.nrows(), qr_factors.ncols()));
    assert!(householder_factor.nrows() > 0);

    let mut dst = dst;
    let mut stack = stack;

    // copy R
    zipped_rw!(dst.rb_mut(), qr_factors).for_each_triangular_upper(
        crate::linalg::zip::Diag::Include,
        |unzipped!(mut dst, src)| dst.write(src.read()),
    );

    // zero bottom part
    zipped_rw!(dst.rb_mut())
        .for_each_triangular_lower(crate::linalg::zip::Diag::Skip, |unzipped!(mut dst)| {
            dst.write(E::faer_zero())
        });

    apply_block_householder_sequence_on_the_left_in_place_with_conj(
        qr_factors,
        householder_factor,
        Conj::No,
        dst.rb_mut(),
        parallelism,
        stack.rb_mut(),
    );

    permute_cols_in_place(dst, col_perm.inverse(), stack);
}

/// Computes the reconstructed matrix, given its QR decomposition, and stores the
/// result in `rhs`.
///
/// # Panics
///
/// - Panics if the number of columns of `householder_factor` isn't the same as the minimum of the
/// number of rows and the number of columns of `qr_factors`.
/// - Panics if the block size is zero.
/// - Panics if `col_perm` doesn't have the same dimension as `qr_factors`.
/// - Panics if the provided memory in `stack` is insufficient (see [`reconstruct_in_place_req`]).
#[track_caller]
pub fn reconstruct_in_place<I: Index, E: ComplexField>(
    qr_factors: MatMut<'_, E>,
    householder_factor: MatRef<'_, E>,
    col_perm: PermRef<'_, I>,
    parallelism: Parallelism,
    stack: &mut PodStack,
) {
    let (mut dst, stack) = temp_mat_uninit::<E>(qr_factors.nrows(), qr_factors.ncols(), stack);
    let mut dst = dst.as_mut();

    reconstruct(
        dst.rb_mut(),
        qr_factors.rb(),
        householder_factor,
        col_perm,
        parallelism,
        stack,
    );

    zipped_rw!(qr_factors, dst.rb()).for_each(|unzipped!(mut dst, src)| dst.write(src.read()));
}

/// Computes the size and alignment of required workspace for reconstructing a matrix out of place,
/// given its QR decomposition with column pivoting.
pub fn reconstruct_req<I: Index, E: Entity>(
    qr_nrows: usize,
    qr_ncols: usize,
    blocksize: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = qr_nrows;
    let _ = parallelism;
    StackReq::try_any_of([
        temp_mat_req::<E>(blocksize, qr_ncols)?,
        permute_cols_in_place_req::<I, E>(qr_nrows, qr_ncols)?,
    ])
}

/// Computes the size and alignment of required workspace for reconstructing a matrix in place,
/// given its QR decomposition with column pivoting.
pub fn reconstruct_in_place_req<I: Index, E: Entity>(
    qr_nrows: usize,
    qr_ncols: usize,
    blocksize: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_all_of([
        temp_mat_req::<E>(qr_nrows, qr_ncols)?,
        reconstruct_req::<I, E>(qr_nrows, qr_ncols, blocksize, parallelism)?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assert,
        complex_native::c64,
        linalg::qr::col_pivoting::compute::{qr_in_place, qr_in_place_req, recommended_blocksize},
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
            E::new(rng.gen(), rng.gen())
        })
    }

    #[test]
    fn test_reconstruct() {
        for n in [31, 32, 48, 65] {
            let mat = Mat::from_fn(n, n, |_, _| random_value());
            let blocksize = recommended_blocksize::<E>(n, n);
            let mut qr = mat.clone();
            let mut householder_factor = Mat::zeros(blocksize, n);

            let parallelism = crate::Parallelism::Rayon(0);
            let mut perm = vec![0usize; n];
            let mut perm_inv = vec![0; n];

            let (_, perm) = qr_in_place(
                qr.as_mut(),
                householder_factor.as_mut(),
                &mut perm,
                &mut perm_inv,
                parallelism,
                make_stack!(qr_in_place_req::<usize, E>(
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
                perm.rb(),
                parallelism,
                make_stack!(reconstruct_req::<usize, E>(n, n, blocksize, parallelism)),
            );

            for i in 0..n {
                for j in 0..n {
                    assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j));
                }
            }
        }
    }
}
