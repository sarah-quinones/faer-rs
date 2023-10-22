#[cfg(feature = "std")]
use assert2::{assert, debug_assert};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_core::{
    join_raw,
    mul::triangular,
    permutation::{Index, PermutationRef, SignedIndex},
    temp_mat_req, temp_mat_uninit, ComplexField, Entity, MatMut, MatRef, Parallelism,
};
use reborrow::*;
use triangular::BlockStructure;

fn invert_impl<I: Index, E: ComplexField>(
    mut dst: MatMut<'_, E>,
    lu_factors: Option<MatRef<'_, E>>,
    row_perm: PermutationRef<'_, I, E>,
    col_perm: PermutationRef<'_, I, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    let lu_factors = match lu_factors {
        Some(lu_factors) => lu_factors,
        None => dst.rb(),
    };

    let m = lu_factors.nrows();
    let n = lu_factors.ncols();
    debug_assert!(m == n);
    debug_assert!(dst.nrows() == n);
    debug_assert!(dst.ncols() == n);

    let (mut inv_lu, stack) = temp_mat_uninit::<E>(m, n, stack);
    let (mut inv, _) = temp_mat_uninit::<E>(m, n, stack);
    let inv_lu = inv_lu.as_mut();
    let mut inv = inv.as_mut();

    // SAFETY: even though the matrices alias, only the strictly lower triangular part of l_inv is
    // accessed and only the upper triangular part of u_inv is accessed.
    let l_inv = unsafe { inv_lu.rb().const_cast() };
    let u_inv = unsafe { inv_lu.rb().const_cast() };

    join_raw(
        |parallelism| {
            faer_core::inverse::invert_unit_lower_triangular(l_inv, lu_factors, parallelism)
        },
        |parallelism| faer_core::inverse::invert_upper_triangular(u_inv, lu_factors, parallelism),
        parallelism,
    );

    let l_inv = inv_lu.rb();
    let u_inv = inv_lu.rb();

    triangular::matmul(
        inv.rb_mut(),
        BlockStructure::Rectangular,
        u_inv,
        BlockStructure::TriangularUpper,
        l_inv,
        BlockStructure::UnitTriangularLower,
        None,
        E::faer_one(),
        parallelism,
    );

    let col_p = row_perm.into_arrays().1;
    let row_p = col_perm.into_arrays().1;
    assert!(row_p.len() == n);
    assert!(col_p.len() == n);
    unsafe {
        if dst.row_stride().abs() <= dst.col_stride().abs() {
            for j in 0..n {
                let jj = col_p.get_unchecked(j).to_signed().zx();
                for i in 0..m {
                    let ii = row_p.get_unchecked(i).to_signed().zx();
                    dst.write_unchecked(i, j, inv.read_unchecked(ii, jj));
                }
            }
        } else {
            for i in 0..m {
                let ii = row_p.get_unchecked(i).to_signed().zx();
                for j in 0..n {
                    let jj = col_p.get_unchecked(j).to_signed().zx();
                    dst.write_unchecked(i, j, inv.read_unchecked(ii, jj));
                }
            }
        }
    }
}

/// Computes the inverse of a matrix, given its full pivoting LU decomposition,
/// and stores the result in `dst`.
///
/// # Panics
///
/// - Panics if the LU factors are not a square matrix.
/// - Panics if the row permutation doesn't have the same dimension as the matrix.
/// - Panics if the column permutation doesn't have the same dimension as the matrix.
/// - Panics if the destination shape doesn't match the shape of the matrix.
/// - Panics if the provided memory in `stack` is insufficient (see [`invert_req`]).
#[track_caller]
pub fn invert<I: Index, E: ComplexField>(
    dst: MatMut<'_, E>,
    lu_factors: MatRef<'_, E>,
    row_perm: PermutationRef<'_, I, E>,
    col_perm: PermutationRef<'_, I, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    let n = lu_factors.nrows();
    assert!(lu_factors.nrows() == lu_factors.ncols());
    assert!((row_perm.len(), col_perm.len()) == (n, n));
    assert!((dst.nrows(), dst.ncols()) == (n, n));
    invert_impl(
        dst,
        Some(lu_factors),
        row_perm,
        col_perm,
        parallelism,
        stack,
    )
}

/// Computes the inverse of a matrix, given its full pivoting LU decomposition,
/// and stores the result in `lu_factors`.
///
/// # Panics
///
/// - Panics if the LU factors are not a square matrix.
/// - Panics if the row permutation doesn't have the same dimension as the matrix.
/// - Panics if the column permutation doesn't have the same dimension as the matrix.
/// - Panics if the provided memory in `stack` is insufficient (see [`invert_in_place_req`]).
#[track_caller]
pub fn invert_in_place<I: Index, E: ComplexField>(
    lu_factors: MatMut<'_, E>,
    row_perm: PermutationRef<'_, I, E>,
    col_perm: PermutationRef<'_, I, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    let n = lu_factors.nrows();
    assert!(lu_factors.nrows() == lu_factors.ncols());
    assert!((row_perm.len(), col_perm.len()) == (n, n));
    invert_impl(lu_factors, None, row_perm, col_perm, parallelism, stack)
}

/// Computes the size and alignment of required workspace for computing the inverse of a
/// matrix out of place, given its full pivoting LU decomposition.
pub fn invert_req<I: Index, E: Entity>(
    nrows: usize,
    ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = parallelism;
    temp_mat_req::<E>(nrows, ncols)?.try_and(temp_mat_req::<E>(nrows, ncols)?)
}

/// Computes the size and alignment of required workspace for computing the inverse of a
/// matrix in place, given its full pivoting LU decomposition.
pub fn invert_in_place_req<I: Index, E: Entity>(
    nrows: usize,
    ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = parallelism;
    temp_mat_req::<E>(nrows, ncols)?.try_and(temp_mat_req::<E>(nrows, ncols)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::full_pivoting::compute::{lu_in_place, lu_in_place_req};
    use assert2::assert;
    use assert_approx_eq::assert_approx_eq;
    use faer_core::{mul::matmul, Mat, Parallelism};
    use rand::random;

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::PodStack::new(&mut ::dyn_stack::GlobalPodBuffer::new($req.unwrap()))
        };
    }

    #[test]
    fn test_inverse() {
        (0..32).chain((1..16).map(|i| i * 32)).for_each(|n| {
            let mat = Mat::from_fn(n, n, |_, _| random::<f64>());
            let mut lu = mat.clone();
            let mut row_perm = vec![0; n];
            let mut row_perm_inv = vec![0usize; n];
            let mut col_perm = vec![0; n];
            let mut col_perm_inv = vec![0; n];
            let (_, row_perm, col_perm) = lu_in_place(
                lu.as_mut(),
                &mut row_perm,
                &mut row_perm_inv,
                &mut col_perm,
                &mut col_perm_inv,
                Parallelism::Rayon(0),
                make_stack!(lu_in_place_req::<usize, f64>(
                    n,
                    n,
                    Parallelism::Rayon(0),
                    Default::default()
                )),
                Default::default(),
            );
            let mut inv = Mat::zeros(n, n);
            invert(
                inv.as_mut(),
                lu.as_ref(),
                row_perm.rb(),
                col_perm.rb(),
                Parallelism::Rayon(0),
                make_stack!(invert_req::<usize, f64>(n, n, Parallelism::Rayon(0))),
            );

            let mut prod = Mat::zeros(n, n);
            matmul(
                prod.as_mut(),
                mat.as_ref(),
                inv.as_ref(),
                None,
                1.0,
                Parallelism::Rayon(0),
            );

            for j in 0..n {
                for i in 0..n {
                    let target = if i == j { 1.0 } else { 0.0 };
                    assert_approx_eq!(prod.read(i, j), target);
                }
            }
        });
    }
}
