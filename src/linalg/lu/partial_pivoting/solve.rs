use crate::{
    linalg::{temp_mat_req, temp_mat_uninit, triangular_solve::*},
    perm::{permute_rows, PermRef},
    unzipped, zipped, ComplexField, Conj, Entity, Index, MatMut, MatRef, Parallelism,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use reborrow::*;

fn solve_impl<I: Index, E: ComplexField>(
    lu_factors: MatRef<'_, E>,
    conj_lhs: Conj,
    row_perm: PermRef<'_, I>,
    dst: MatMut<'_, E>,
    rhs: Option<MatRef<'_, E>>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    // LU = P(row_fwd) × A

    // P(row_inv) ConjA?(LU) X = ConjB?(B)
    // X = ConjA?(U)^-1 ConjA?(L)^-1 P(row_fwd) ConjB?(B)

    let n = lu_factors.ncols();
    let k = dst.ncols();

    let (mut temp, _) = temp_mat_uninit::<E>(n, k, stack);
    let mut temp = temp.as_mut();

    // temp <- P(row_fwd) B
    let src = match rhs {
        Some(rhs) => rhs,
        None => dst.rb(),
    };
    permute_rows(temp.rb_mut(), src, row_perm);

    // temp <- ConjA?(L)^-1 P(row_fwd) ConjB?(B)
    solve_unit_lower_triangular_in_place_with_conj(
        lu_factors,
        conj_lhs,
        temp.rb_mut(),
        parallelism,
    );

    // temp <- ConjA?(U)^-1 ConjA?(L)^-1 P(row_fwd) B
    solve_upper_triangular_in_place_with_conj(lu_factors, conj_lhs, temp.rb_mut(), parallelism);

    // dst <- ConjA?(U)^-1 ConjA?(L)^-1 P(row_fwd) B
    zipped!(dst, temp.rb()).for_each(|unzipped!(mut dst, tmp)| dst.write(tmp.read()));
}

fn solve_transpose_impl<I: Index, E: ComplexField>(
    lu_factors: MatRef<'_, E>,
    conj_lhs: Conj,
    row_perm: PermRef<'_, I>,
    dst: MatMut<'_, E>,
    rhs: Option<MatRef<'_, E>>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    // LU = P(row_fwd) × A × P(col_inv)

    // (P(row_inv) ConjA?(LU))^T X = ConjB?(B)
    // ConjA?(U)^T ConjA?(L)^T P(row_fwd) X = ConjB?(B)
    // X = P(row_inv) ConjA?(L).T^-1 ConjA?(U).T^-1 ConjB?(B)

    let n = lu_factors.ncols();
    let k = dst.ncols();

    let (mut temp, _) = temp_mat_uninit::<E>(n, k, stack);
    let mut temp = temp.as_mut();

    // temp <- P(col_fwd) B
    let src = match rhs {
        Some(rhs) => rhs,
        None => dst.rb(),
    };
    zipped!(temp.rb_mut(), src).for_each(|unzipped!(mut dst, tmp)| dst.write(tmp.read()));

    // temp <- ConjA?(U).T^-1 P(col_fwd) ConjB?(B)
    solve_lower_triangular_in_place_with_conj(
        lu_factors.transpose(),
        conj_lhs,
        temp.rb_mut(),
        parallelism,
    );

    // temp <- ConjA?(L).T^-1 ConjA?(U).T^-1 P(row_fwd) B
    solve_unit_upper_triangular_in_place_with_conj(
        lu_factors.transpose(),
        conj_lhs,
        temp.rb_mut(),
        parallelism,
    );

    // dst <- P(row_inv) ConjA?(L).T^-1 ConjA?(U).T^-1 P(col_fwd) ConjB?(B)
    permute_rows(dst, temp.rb(), row_perm.inverse());
}

/// Computes the size and alignment of required workspace for solving a linear system defined by a
/// matrix in place, given its full pivoting LU decomposition.
pub fn solve_in_place_req<I: Index, E: Entity>(
    lu_nrows: usize,
    lu_ncols: usize,
    rhs_ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = lu_ncols;
    let _ = parallelism;
    temp_mat_req::<E>(lu_nrows, rhs_ncols)
}

/// Computes the size and alignment of required workspace for solving a linear system defined by a
/// matrix out of place, given its full pivoting LU decomposition.
pub fn solve_req<I: Index, E: Entity>(
    lu_nrows: usize,
    lu_ncols: usize,
    rhs_ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = lu_ncols;
    let _ = parallelism;
    temp_mat_req::<E>(lu_nrows, rhs_ncols)
}

/// Computes the size and alignment of required workspace for solving a linear system defined by
/// the transpose of a matrix in place, given its full pivoting LU decomposition.
pub fn solve_transpose_in_place_req<I: Index, E: Entity>(
    lu_nrows: usize,
    lu_ncols: usize,
    rhs_ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = lu_ncols;
    let _ = parallelism;
    temp_mat_req::<E>(lu_nrows, rhs_ncols)
}

/// Computes the size and alignment of required workspace for solving a linear system defined by
/// the transpose of a matrix out of place, given its full pivoting LU decomposition.
pub fn solve_transpose_req<I: Index, E: Entity>(
    lu_nrows: usize,
    lu_ncols: usize,
    rhs_ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = lu_ncols;
    let _ = parallelism;
    temp_mat_req::<E>(lu_nrows, rhs_ncols)
}

/// Given the LU factors of a matrix $A$ and a matrix $B$ stored in `rhs`, this function computes
/// the solution of the linear system:
/// $$\text{Op}_A(A)X = B.$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
///
/// The solution of the linear system is stored in `dst`.
///
/// # Panics
///
/// - Panics if `lu_factors` is not a square matrix.
/// - Panics if `row_perm` doesn't have the same dimension as `lu_factors`.
/// - Panics if `rhs` doesn't have the same number of rows as the dimension of `lu_factors`.
/// - Panics if `rhs` and `dst` don't have the same shape.
/// - Panics if the provided memory in `stack` is insufficient (see [`solve_req`]).
pub fn solve<I: Index, E: ComplexField>(
    dst: MatMut<'_, E>,
    lu_factors: MatRef<'_, E>,
    conj_lhs: Conj,
    row_perm: PermRef<'_, I>,
    rhs: MatRef<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    solve_impl(
        lu_factors,
        conj_lhs,
        row_perm,
        dst,
        Some(rhs),
        parallelism,
        stack,
    )
}

/// Given the LU factors of a matrix $A$ and a matrix $B$ stored in `rhs`, this function computes
/// the solution of the linear system:
/// $$\text{Op}_A(A)X = B.$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
///
/// The solution of the linear system is stored in `rhs`.
///
/// # Panics
///
/// - Panics if `lu_factors` is not a square matrix.
/// - Panics if `row_perm` doesn't have the same dimension as `lu_factors`.
/// - Panics if `rhs` doesn't have the same number of rows as the dimension of `lu_factors`.
/// - Panics if the provided memory in `stack` is insufficient (see [`solve_in_place_req`]).
pub fn solve_in_place<I: Index, E: ComplexField>(
    lu_factors: MatRef<'_, E>,
    conj_lhs: Conj,
    row_perm: PermRef<'_, I>,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    solve_impl(
        lu_factors,
        conj_lhs,
        row_perm,
        rhs,
        None,
        parallelism,
        stack,
    );
}

/// Given the LU factors of a matrix $A$ and a matrix $B$ stored in `rhs`, this function computes
/// the solution of the linear system:
/// $$\text{Op}_A(A)^\top X = B.$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
///
/// The solution of the linear system is stored in `dst`.
///
/// # Panics
///
/// - Panics if `lu_factors` is not a square matrix.
/// - Panics if `row_perm` doesn't have the same dimension as `lu_factors`.
/// - Panics if `rhs` doesn't have the same number of rows as the dimension of `lu_factors`.
/// - Panics if `rhs` and `dst` don't have the same shape.
/// - Panics if the provided memory in `stack` is insufficient (see [`solve_transpose_req`]).
pub fn solve_transpose<I: Index, E: ComplexField>(
    dst: MatMut<'_, E>,
    lu_factors: MatRef<'_, E>,
    conj_lhs: Conj,
    row_perm: PermRef<'_, I>,
    rhs: MatRef<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    solve_transpose_impl(
        lu_factors,
        conj_lhs,
        row_perm,
        dst,
        Some(rhs),
        parallelism,
        stack,
    )
}

/// Given the LU factors of a matrix $A$ and a matrix $B$ stored in `rhs`, this function computes
/// the solution of the linear system:
/// $$\text{Op}_A(A)^\top X = B.$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
/// The solution of the linear system is stored in `rhs`.
///
/// # Panics
///
/// - Panics if `lu_factors` is not a square matrix.
/// - Panics if `row_perm` doesn't have the same dimension as `lu_factors`.
/// - Panics if `rhs` doesn't have the same number of rows as the dimension of `lu_factors`.
/// - Panics if the provided memory in `stack` is insufficient (see
///   [`solve_transpose_in_place_req`]).
pub fn solve_transpose_in_place<I: Index, E: ComplexField>(
    lu_factors: MatRef<'_, E>,
    conj_lhs: Conj,
    row_perm: PermRef<'_, I>,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    solve_transpose_impl(
        lu_factors,
        conj_lhs,
        row_perm,
        rhs,
        None,
        parallelism,
        stack,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assert,
        complex_native::{c32, c64},
        linalg::{
            lu::partial_pivoting::compute::{lu_in_place, lu_in_place_req},
            matmul::matmul_with_conj,
        },
        Mat,
    };
    use std::cell::RefCell;

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::PodStack::new(&mut ::dyn_stack::GlobalPodBuffer::new($req.unwrap()))
        };
    }

    fn test_solve<E: ComplexField>(mut gen: impl FnMut() -> E, epsilon: E::Real) {
        (0..32).chain((1..8).map(|i| i * 32)).for_each(|n| {
            for conj_lhs in [Conj::No, Conj::Yes] {
                let a = Mat::from_fn(n, n, |_, _| gen());
                let mut lu = a.clone();
                let a = a.as_ref();
                let mut lu = lu.as_mut();

                let k = 32;
                let rhs = Mat::from_fn(n, k, |_, _| gen());
                let rhs = rhs.as_ref();
                let mut sol = Mat::<E>::zeros(n, k);
                let mut sol = sol.as_mut();

                let mut row_perm = vec![0_usize; n];
                let mut row_perm_inv = vec![0_usize; n];

                let parallelism = Parallelism::Rayon(0);

                let (_, row_perm) = lu_in_place(
                    lu.rb_mut(),
                    &mut row_perm,
                    &mut row_perm_inv,
                    parallelism,
                    make_stack!(lu_in_place_req::<usize, E>(
                        n,
                        n,
                        parallelism,
                        Default::default()
                    )),
                    Default::default(),
                );

                solve(
                    sol.rb_mut(),
                    lu.rb(),
                    conj_lhs,
                    row_perm.rb(),
                    rhs,
                    parallelism,
                    make_stack!(solve_req::<usize, E>(n, n, k, parallelism)),
                );

                let mut rhs_reconstructed = Mat::zeros(n, k);
                let mut rhs_reconstructed = rhs_reconstructed.as_mut();

                matmul_with_conj(
                    rhs_reconstructed.rb_mut(),
                    a,
                    conj_lhs,
                    sol.rb(),
                    Conj::No,
                    None,
                    E::faer_one(),
                    parallelism,
                );

                for j in 0..k {
                    for i in 0..n {
                        assert!(
                            (rhs_reconstructed.read(i, j).faer_sub(rhs.read(i, j))).faer_abs()
                                < epsilon
                        )
                    }
                }
            }
        });
    }

    fn test_solve_transpose<E: ComplexField>(mut gen: impl FnMut() -> E, epsilon: E::Real) {
        (0..32).chain((1..8).map(|i| i * 32)).for_each(|n| {
            for conj_lhs in [Conj::No, Conj::Yes] {
                let a = Mat::from_fn(n, n, |_, _| gen());
                let mut lu = a.clone();
                let a = a.as_ref();
                let mut lu = lu.as_mut();

                let k = 32;
                let rhs = Mat::from_fn(n, k, |_, _| gen());
                let rhs = rhs.as_ref();
                let mut sol = Mat::<E>::zeros(n, k);
                let mut sol = sol.as_mut();

                let mut row_perm = vec![0_usize; n];
                let mut row_perm_inv = vec![0_usize; n];

                let parallelism = Parallelism::Rayon(0);

                let (_, row_perm) = lu_in_place(
                    lu.rb_mut(),
                    &mut row_perm,
                    &mut row_perm_inv,
                    parallelism,
                    make_stack!(lu_in_place_req::<usize, E>(
                        n,
                        n,
                        parallelism,
                        Default::default()
                    )),
                    Default::default(),
                );

                solve_transpose(
                    sol.rb_mut(),
                    lu.rb(),
                    conj_lhs,
                    row_perm.rb(),
                    rhs,
                    parallelism,
                    make_stack!(solve_transpose_req::<usize, E>(n, n, k, parallelism)),
                );

                let mut rhs_reconstructed = Mat::zeros(n, k);
                let mut rhs_reconstructed = rhs_reconstructed.as_mut();

                matmul_with_conj(
                    rhs_reconstructed.rb_mut(),
                    a.transpose(),
                    conj_lhs,
                    sol.rb(),
                    Conj::No,
                    None,
                    E::faer_one(),
                    parallelism,
                );

                for j in 0..k {
                    for i in 0..n {
                        assert!(
                            (rhs_reconstructed.read(i, j).faer_sub(rhs.read(i, j))).faer_abs()
                                < epsilon
                        )
                    }
                }
            }
        });
    }
    use rand::prelude::*;
    thread_local! {
        static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(0));
    }

    fn random_f64() -> f64 {
        RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            let rng = &mut *rng;
            rng.gen()
        })
    }
    fn random_f32() -> f32 {
        RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            let rng = &mut *rng;
            rng.gen()
        })
    }

    fn random_c64() -> c64 {
        c64 {
            re: random_f64(),
            im: random_f64(),
        }
    }
    fn random_c32() -> c32 {
        c32 {
            re: random_f32(),
            im: random_f32(),
        }
    }

    #[test]
    fn test_solve_f64() {
        test_solve(random_f64, 1e-6_f64);
        test_solve_transpose(random_f64, 1e-6_f64);
    }

    #[test]
    fn test_solve_f32() {
        test_solve(random_f32, 1e-1_f32);
        test_solve_transpose(random_f32, 1e-1_f32);
    }

    #[test]
    fn test_solve_c64() {
        test_solve(random_c64, 1e-6_f64);
        test_solve_transpose(random_c64, 1e-6_f64);
    }

    #[test]
    fn test_solve_c32() {
        test_solve(random_c32, 2e-1_f32);
        test_solve_transpose(random_c32, 2e-1_f32);
    }
}
