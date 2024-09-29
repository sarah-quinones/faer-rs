use crate::{
    assert,
    linalg::{matmul::triangular, temp_mat_req, temp_mat_uninit},
    perm::PermRef,
    ComplexField, Entity, Index, MatMut, MatRef, Parallelism,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use reborrow::*;
use triangular::BlockStructure;

#[track_caller]
fn reconstruct_impl<I: Index, E: ComplexField>(
    dst: MatMut<'_, E>,
    lu_factors: Option<MatRef<'_, E>>,
    row_perm: PermRef<'_, I>,
    parallelism: Parallelism,
    stack: &mut PodStack,
) {
    let lu_factors = match lu_factors {
        Some(lu_factors) => lu_factors,
        None => dst.rb(),
    };

    let m = lu_factors.nrows();
    let n = lu_factors.ncols();
    let size = Ord::min(m, n);

    let (mut lu, _) = temp_mat_uninit::<E>(m, n, stack);
    let mut lu = lu.as_mut();

    let (l_top, _, l_bot, _) = lu_factors.split_at(size, size);
    let (u_left, u_right, _, _) = lu_factors.split_at(size, size);

    let (lu_topleft, lu_topright, lu_botleft, _) = lu.rb_mut().split_at_mut(size, size);

    triangular::matmul(
        lu_topleft,
        BlockStructure::Rectangular,
        l_top,
        BlockStructure::UnitTriangularLower,
        u_left,
        BlockStructure::TriangularUpper,
        None,
        E::faer_one(),
        parallelism,
    );
    triangular::matmul(
        lu_topright,
        BlockStructure::Rectangular,
        l_top,
        BlockStructure::UnitTriangularLower,
        u_right,
        BlockStructure::Rectangular,
        None,
        E::faer_one(),
        parallelism,
    );
    triangular::matmul(
        lu_botleft,
        BlockStructure::Rectangular,
        l_bot,
        BlockStructure::Rectangular,
        u_left,
        BlockStructure::TriangularUpper,
        None,
        E::faer_one(),
        parallelism,
    );

    crate::perm::permute_rows(dst, lu.rb(), row_perm.inverse());
}

/// Computes the reconstructed matrix, given its partial pivoting LU decomposition,
/// and stores the result in `dst`.
///
/// # Panics
///
/// - Panics if the row permutation doesn't have the same dimension as the number of rows of the
/// matrix.
/// - Panics if the destination shape doesn't match the shape of the matrix.
/// - Panics if the provided memory in `stack` is insufficient (see [`reconstruct_req`]).
#[track_caller]
pub fn reconstruct<I: Index, E: ComplexField>(
    dst: MatMut<'_, E>,
    lu_factors: MatRef<'_, E>,
    row_perm: PermRef<'_, I>,
    parallelism: Parallelism,
    stack: &mut PodStack,
) {
    assert!((dst.nrows(), dst.ncols()) == (lu_factors.nrows(), lu_factors.ncols()));
    assert!(row_perm.len() == lu_factors.nrows());
    reconstruct_impl(dst, Some(lu_factors), row_perm, parallelism, stack)
}

/// Computes the reconstructed matrix, given its partial pivoting LU decomposition, and stores the
/// result in `lu_factors`.
///
/// # Panics
///
/// - Panics if the row permutation doesn't have the same dimension as the number of rows of the
/// matrix.
/// - Panics if the provided memory in `stack` is insufficient (see [`reconstruct_in_place_req`]).
#[track_caller]
pub fn reconstruct_in_place<I: Index, E: ComplexField>(
    lu_factors: MatMut<'_, E>,
    row_perm: PermRef<'_, I>,
    parallelism: Parallelism,
    stack: &mut PodStack,
) {
    assert!(row_perm.len() == lu_factors.nrows());
    reconstruct_impl(lu_factors, None, row_perm, parallelism, stack)
}

/// Computes the size and alignment of required workspace for reconstructing a matrix out of place,
/// given its partial pivoting LU decomposition.
pub fn reconstruct_req<I: Index, E: Entity>(
    nrows: usize,
    ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = parallelism;
    temp_mat_req::<E>(nrows, ncols)
}

/// Computes the size and alignment of required workspace for reconstructing a matrix in place,
/// given its partial pivoting LU decomposition.
pub fn reconstruct_in_place_req<I: Index, E: Entity>(
    nrows: usize,
    ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    reconstruct_req::<I, E>(nrows, ncols, parallelism)
}
