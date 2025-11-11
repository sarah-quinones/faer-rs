//! Sylvester equation solvers
//!
//! This module provides solvers for Sylvester equations of the form:
//! - Standard Sylvester: $AX - XB = C$
//! - Generalized Sylvester: $AXB^T + CXD^T = E$
//!
//! The solvers use the Bartels-Stewart algorithm with Schur decomposition for general cases,
//! and direct analytical methods for small (2×2) cases.
//!
//! # Testing methodology
//!
//! Tests follow LAPACK DGET32 methodology for DLASY2:
//!
//! - **Residual verification**: Normalized residual computed as `||TL*X - X*TR - scale*B|| / max(smlnum, smlnum*||X||, eps*||T||*||X||)` must be O(1)
//! - **Tolerance**: Test ratio threshold of 20.0 (LAPACK standard for condition estimation routines)
//! - **Scale validation**: Scale factor must satisfy `0 < scale ≤ 1`
//! - **Extreme magnitudes**: Tests with `sqrt(smlnum)` and `sqrt(bignum)` exercise overflow prevention
//! - **Near-singular cases**: Matrices with elements near machine epsilon verify numerical stability
//! - **Quasi-triangular blocks**: 2×2 blocks with subdiagonal elements simulate Schur form inputs
//! - **Precision testing**: Both f32 and f64 tested with type-specific machine epsilon
//!
//! where `smlnum = min_positive / eps`, `bignum = 1 / smlnum`, and `eps` is machine precision.

use crate::assert;
use crate::internal_prelude::*;
use faer_traits::RealField;

/// Solves the 2×2 Sylvester equation $TL \cdot X - X \cdot TR = B$
///
/// This function solves the small-scale Sylvester equation where $TL$ and $TR$ are 2×2 matrices,
/// typically in upper quasi-triangular form (from Schur decomposition). The solution $X$ is
/// computed using direct analytical methods with careful scaling for numerical stability.
///
/// The equation is solved by converting it to a 4×4 linear system and using partial pivoting
/// with scaling to avoid overflow. The solution may be scaled down by the returned scale factor
/// to maintain numerical stability.
///
/// # Mathematical formulation
///
/// Given 2×2 matrices $TL$, $TR$, and $B$, find 2×2 matrix $X$ such that:
/// $$TL \cdot X - X \cdot TR = \text{scale} \cdot B$$
///
/// where $\text{scale} \leq 1$ is chosen to prevent overflow.
///
/// # Inputs
///
/// - `tl`: Left 2×2 matrix (typically from Schur form)
/// - `tr`: Right 2×2 matrix (typically from Schur form)
/// - `b`: Right-hand side 2×2 matrix
/// - `x`: Output 2×2 matrix to store the solution (will be overwritten)
///
/// # Returns
///
/// Returns the scaling factor $\text{scale} \in (0, 1]$ applied to the solution. If `scale = 1.0`,
/// no scaling was needed. If `scale < 1.0`, the solution has been scaled down to avoid overflow,
/// and the true solution is $X / \text{scale}$.
///
/// # Panics
///
/// Panics if:
/// - `tl` is not a 2×2 matrix
/// - `tr` is not a 2×2 matrix
/// - `b` is not a 2×2 matrix
/// - `x` is not a 2×2 matrix
///
/// # Example
///
/// ```
/// use faer::linalg::sylvester::solve_sylvester_2x2;
/// use faer::Mat;
///
/// let tl = Mat::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 0.5 });
/// let tr = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.1 });
/// let b = Mat::from_fn(2, 2, |i, j| (i + j) as f64);
/// let mut x = Mat::zeros(2, 2);
///
/// let scale = solve_sylvester_2x2(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_mut());
///
/// println!("Solution scaling factor: {}", scale);
/// ```
///
/// # Algorithm
///
/// The implementation follows the LAPACK DLASY2 algorithm:
/// 1. Convert the Sylvester equation to a 4×4 linear system by vectorization
/// 2. Apply LU factorization with partial pivoting
/// 3. Scale the right-hand side if necessary to prevent overflow
/// 4. Solve the triangular system via back-substitution
/// 5. Unpack the solution back into 2×2 matrix form
///
/// # Numerical stability
///
/// The algorithm uses several techniques to ensure numerical stability:
/// - Machine epsilon-based thresholding for near-singular matrices
/// - Scaling to prevent overflow in intermediate computations
/// - Partial pivoting to minimize growth factor
///
/// # References
///
/// - Anderson et al., "LAPACK Users' Guide", 3rd ed., SIAM, 1999
/// - Bartels-Stewart algorithm for Sylvester equations
pub fn solve_sylvester_2x2<T: RealField>(tl: MatRef<'_, T>, tr: MatRef<'_, T>, b: MatRef<'_, T>, x: MatMut<'_, T>) -> T {
	let mut x = x;

	assert!(
		all(
			tl.nrows() == 2,
			tr.nrows() == 2,
			tl.ncols() == 2,
			tr.ncols() == 2,
			b.nrows() == 2,
			b.ncols() == 2,
			x.nrows() == 2,
			x.ncols() == 2,
		),
		"solve_sylvester_2x2 requires all matrices to be 2×2"
	);

	let ref eps = eps::<T>();

	let ref smlnum = min_positive::<T>() / eps;

	stack_mat!(btmp, 4, 1, T);

	stack_mat!(tmp, 4, 1, T);

	stack_mat!(t16, 4, 4, T);

	let mut jpiv = [0usize; 4];

	let smin = tr[(0, 0)]
		.abs1()
		.fmax(tr[(0, 1)].abs1())
		.fmax(tr[(1, 0)].abs1())
		.fmax(tr[(1, 1)].abs1())
		.fmax(tl[(0, 0)].abs1())
		.fmax(tl[(0, 1)].abs1())
		.fmax(tl[(1, 0)].abs1())
		.fmax(tl[(1, 1)].abs1());

	let smin = (eps * smin).fmax(smlnum);

	t16[(0, 0)] = &tl[(0, 0)] - &tr[(0, 0)];

	t16[(1, 1)] = &tl[(1, 1)] - &tr[(0, 0)];

	t16[(2, 2)] = &tl[(0, 0)] - &tr[(1, 1)];

	t16[(3, 3)] = &tl[(1, 1)] - &tr[(1, 1)];

	t16[(0, 1)] = tl[(0, 1)].copy();

	t16[(1, 0)] = tl[(1, 0)].copy();

	t16[(2, 3)] = tl[(0, 1)].copy();

	t16[(3, 2)] = tl[(1, 0)].copy();

	t16[(0, 2)] = -&tr[(1, 0)];

	t16[(1, 3)] = -&tr[(1, 0)];

	t16[(2, 0)] = -&tr[(0, 1)];

	t16[(3, 1)] = -&tr[(0, 1)];

	btmp[(0, 0)] = b[(0, 0)].copy();

	btmp[(1, 0)] = b[(1, 0)].copy();

	btmp[(2, 0)] = b[(0, 1)].copy();

	btmp[(3, 0)] = b[(1, 1)].copy();

	let (mut ipsv, mut jpsv);

	#[allow(clippy::needless_range_loop)]
	for i in 0..3usize {
		ipsv = i;

		jpsv = i;

		let mut xmax = zero();

		for ip in i..4 {
			for jp in i..4 {
				if t16[(ip, jp)].abs1() >= xmax {
					xmax = t16[(ip, jp)].abs1();

					ipsv = ip;

					jpsv = jp;
				}
			}
		}

		if ipsv != i {
			crate::perm::swap_rows_idx(t16.rb_mut(), ipsv, i);

			let temp = btmp[(i, 0)].copy();

			btmp[(i, 0)] = btmp[(ipsv, 0)].copy();

			btmp[(ipsv, 0)] = temp;
		}

		if jpsv != i {
			crate::perm::swap_cols_idx(t16.rb_mut(), jpsv, i);
		}

		jpiv[i] = jpsv;

		if t16[(i, i)].abs1() < smin {
			t16[(i, i)] = smin.copy();
		}

		for j in i + 1..4 {
			t16[(j, i)] = &t16[(j, i)] / &t16[(i, i)];

			btmp[(j, 0)] = &btmp[(j, 0)] - &t16[(j, i)] * &btmp[(i, 0)];

			for k in i + 1..4 {
				t16[(j, k)] = &t16[(j, k)] - &t16[(j, i)] * &t16[(i, k)];
			}
		}
	}

	if t16[(3, 3)].abs1() < smin {
		t16[(3, 3)] = smin.copy();
	}

	let mut scale = one::<T>();

	let ref eight = from_f64::<T>(8.0);

	if (eight * smlnum) * btmp[(0, 0)].abs1() > t16[(0, 0)].abs1()
		|| (eight * smlnum) * btmp[(1, 0)].abs1() > t16[(1, 1)].abs1()
		|| (eight * smlnum) * btmp[(2, 0)].abs1() > t16[(2, 2)].abs1()
		|| (eight * smlnum) * btmp[(3, 0)].abs1() > t16[(3, 3)].abs1()
	{
		scale = from_f64::<T>(0.125)
			/ btmp[(0, 0)]
				.abs1()
				.fmax(btmp[(1, 0)].abs1())
				.fmax(btmp[(2, 0)].abs1())
				.fmax(btmp[(3, 0)].abs1());

		btmp[(0, 0)] *= &scale;

		btmp[(1, 0)] *= &scale;

		btmp[(2, 0)] *= &scale;

		btmp[(3, 0)] *= &scale;
	}

	for i in 0..4usize {
		let k = 3 - i;

		let ref temp = t16[(k, k)].recip();

		tmp[(k, 0)] = &btmp[(k, 0)] * temp;

		for j in k + 1..4 {
			tmp[(k, 0)] = &tmp[(k, 0)] - temp * &t16[(k, j)] * &tmp[(j, 0)];
		}
	}

	for i in 0..3usize {
		if jpiv[2 - i] != 2 - i {
			let temp = tmp[(2 - i, 0)].copy();

			tmp[(2 - i, 0)] = tmp[(jpiv[2 - i], 0)].copy();

			tmp[(jpiv[2 - i], 0)] = temp;
		}
	}

	x[(0, 0)] = tmp[(0, 0)].copy();

	x[(1, 0)] = tmp[(1, 0)].copy();

	x[(0, 1)] = tmp[(2, 0)].copy();

	x[(1, 1)] = tmp[(3, 0)].copy();

	scale
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::Mat;
	use crate::assert;

	fn max_norm<T: RealField>(m: MatRef<'_, T>) -> T {
		let mut norm = zero::<T>();
		for i in 0..m.nrows() {
			for j in 0..m.ncols() {
				norm = norm.fmax(m[(i, j)].abs1());
			}
		}
		norm
	}

	fn matrix_sum_abs<T: RealField>(m: MatRef<'_, T>) -> T {
		let mut sum = zero::<T>();
		for i in 0..m.nrows() {
			for j in 0..m.ncols() {
				sum = &sum + &m[(i, j)].abs1();
			}
		}
		sum
	}

	fn compute_xnorm<T: RealField>(x: MatRef<'_, T>) -> T {
		let row0_sum = x[(0, 0)].abs1() + x[(0, 1)].abs1();
		let row1_sum = x[(1, 0)].abs1() + x[(1, 1)].abs1();
		row0_sum.fmax(row1_sum)
	}

	fn verify_solution<T: RealField>(tl: MatRef<'_, T>, tr: MatRef<'_, T>, b: MatRef<'_, T>, x: MatRef<'_, T>, scale: T) -> bool {
		let ref eps = eps::<T>();
		let ref smlnum = min_positive::<T>() / eps;

		let scaled_b = Mat::from_fn(2, 2, |i, j| &scale * &b[(i, j)]);
		let residual = &tl * &x - &x * &tr - &scaled_b;
		let res_norm = max_norm(residual.as_ref());

		let xnorm = compute_xnorm(x);
		let tnrm = matrix_sum_abs(tl).fmax(matrix_sum_abs(tr));

		let den = smlnum.fmax(smlnum * &xnorm).fmax(&tnrm * eps * &xnorm);
		let ratio = &res_norm / &den;

		ratio < from_f64::<T>(20.0) && scale > zero::<T>() && scale <= one::<T>()
	}

	#[test]
	fn test_solve_sylvester_2x2_upper_triangular_11() {
		let tl = Mat::from_fn(2, 2, |i, j| if i <= j { (i + j + 1) as f64 } else { 0.0 });
		let tr = Mat::from_fn(2, 2, |i, j| if i <= j { (j - i + 1) as f64 } else { 0.0 });
		let b = Mat::from_fn(2, 2, |_, _| 1.0);
		let mut x = Mat::zeros(2, 2);

		let scale = solve_sylvester_2x2(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_mut());

		assert!(verify_solution(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_ref(), scale));
	}

	#[test]
	fn test_solve_sylvester_2x2_general_dense() {
		let tl = Mat::from_fn(2, 2, |i, j| (2 * i + j + 1) as f64 / 10.0);
		let tr = Mat::from_fn(2, 2, |i, j| (i + 3 * j + 2) as f64 / 10.0);
		let b = Mat::from_fn(2, 2, |i, j| ((i + 1) * (j + 1)) as f64);
		let mut x = Mat::zeros(2, 2);

		let scale = solve_sylvester_2x2(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_mut());

		assert!(verify_solution(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_ref(), scale));
	}

	#[test]
	fn test_solve_sylvester_2x2_near_singular() {
		let eps_sqrt = eps::<f64>().sqrt();
		let tl = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { eps_sqrt });
		let tr = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { eps_sqrt });
		let b = Mat::from_fn(2, 2, |i, j| (i + j + 1) as f64);
		let mut x = Mat::zeros(2, 2);

		let scale = solve_sylvester_2x2(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_mut());

		assert!(verify_solution(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_ref(), scale));
	}

	#[test]
	fn test_solve_sylvester_2x2_small_magnitude() {
		let smlnum = min_positive::<f64>() / eps::<f64>();
		let small_val = smlnum.sqrt();
		let tl = Mat::from_fn(2, 2, |i, j| if i == j { small_val } else { small_val * 0.5 });
		let tr = Mat::from_fn(2, 2, |i, j| if i == j { small_val } else { small_val * 0.1 });
		let b = Mat::from_fn(2, 2, |i, j| (i + j + 1) as f64);
		let mut x = Mat::zeros(2, 2);

		let scale = solve_sylvester_2x2(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_mut());

		assert!(verify_solution(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_ref(), scale));
	}

	#[test]
	fn test_solve_sylvester_2x2_large_magnitude() {
		let smlnum = min_positive::<f64>() / eps::<f64>();
		let bignum = smlnum.recip();
		let large_val = bignum.sqrt();
		let tl = Mat::from_fn(2, 2, |i, j| if i == j { large_val } else { large_val * 0.5 });
		let tr = Mat::from_fn(2, 2, |i, j| if i == j { large_val } else { large_val * 0.1 });
		let b = Mat::from_fn(2, 2, |i, j| large_val * (i + j + 1) as f64);
		let mut x = Mat::zeros(2, 2);

		let scale = solve_sylvester_2x2(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_mut());

		assert!(verify_solution(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_ref(), scale));
	}

	#[test]
	fn test_solve_sylvester_2x2_overflow_prevention() {
		let smlnum = min_positive::<f64>() / eps::<f64>();
		let bignum = smlnum.recip();
		let tl = Mat::from_fn(2, 2, |i, j| if i == j { smlnum.sqrt() } else { 0.0 });
		let tr = Mat::from_fn(2, 2, |i, j| if i == j { smlnum.sqrt() } else { 0.0 });
		let b = Mat::from_fn(2, 2, |_, _| bignum.sqrt());
		let mut x = Mat::zeros(2, 2);

		let scale = solve_sylvester_2x2(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_mut());

		assert!(scale < 1.0);
		assert!(verify_solution(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_ref(), scale));
	}

	#[test]
	fn test_solve_sylvester_2x2_quasi_triangular_2x2_block() {
		let tl = Mat::from_fn(2, 2, |i, j| {
			if i == 0 && j == 0 {
				2.0
			} else if i == 1 && j == 1 {
				2.0
			} else if i == 0 && j == 1 {
				1.0
			} else {
				-1.0
			}
		});
		let tr = Mat::from_fn(2, 2, |i, j| {
			if i == 0 && j == 0 {
				3.0
			} else if i == 1 && j == 1 {
				3.0
			} else if i == 0 && j == 1 {
				0.5
			} else {
				-0.5
			}
		});
		let b = Mat::from_fn(2, 2, |i, j| ((i + 1) * (j + 1)) as f64);
		let mut x = Mat::zeros(2, 2);

		let scale = solve_sylvester_2x2(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_mut());

		assert!(verify_solution(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_ref(), scale));
	}

	#[test]
	fn test_solve_sylvester_2x2_mixed_magnitudes() {
		let smlnum = min_positive::<f64>() / eps::<f64>();
		let bignum = smlnum.recip();
		let small_val = smlnum.sqrt();
		let large_val = bignum.sqrt();

		let tl = Mat::from_fn(2, 2, |i, j| if i == j { large_val } else { small_val });
		let tr = Mat::from_fn(2, 2, |i, j| if i == j { small_val } else { large_val });
		let b = Mat::from_fn(2, 2, |i, j| if (i + j) % 2 == 0 { large_val } else { small_val });
		let mut x = Mat::zeros(2, 2);

		let scale = solve_sylvester_2x2(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_mut());

		assert!(verify_solution(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_ref(), scale));
	}

	#[test]
	fn test_solve_sylvester_2x2_well_conditioned() {
		let tl = Mat::from_fn(2, 2, |i, j| if i == j { 5.0 } else { 1.0 });
		let tr = Mat::from_fn(2, 2, |i, j| if i == j { 3.0 } else { 0.5 });
		let b = Mat::from_fn(2, 2, |i, j| (i + j + 1) as f64);
		let mut x = Mat::zeros(2, 2);

		let scale = solve_sylvester_2x2(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_mut());

		assert!((scale - 1.0).abs() < eps::<f64>());
		assert!(verify_solution(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_ref(), scale));
	}

	#[test]
	fn test_solve_sylvester_2x2_f32_precision() {
		let tl = Mat::from_fn(2, 2, |i, j| if i == j { 3.0f32 } else { 0.5f32 });
		let tr = Mat::from_fn(2, 2, |i, j| if i == j { 2.0f32 } else { 0.1f32 });
		let b = Mat::from_fn(2, 2, |i, j| (i + j + 1) as f32);
		let mut x = Mat::zeros(2, 2);

		let scale = solve_sylvester_2x2(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_mut());

		assert!(verify_solution(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_ref(), scale));
	}

	#[test]
	fn test_solve_sylvester_2x2_f32_extreme_magnitude() {
		let smlnum = min_positive::<f32>() / eps::<f32>();
		let small_val = smlnum.sqrt();

		let tl = Mat::from_fn(2, 2, |i, j| if i == j { small_val } else { 0.0f32 });
		let tr = Mat::from_fn(2, 2, |i, j| if i == j { small_val } else { 0.0f32 });
		let b = Mat::from_fn(2, 2, |_, _| 1.0f32);
		let mut x = Mat::zeros(2, 2);

		let scale = solve_sylvester_2x2(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_mut());

		assert!(verify_solution(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_ref(), scale));
	}

	#[test]
	#[should_panic(expected = "solve_sylvester_2x2 requires all matrices to be 2×2")]
	fn test_solve_sylvester_2x2_wrong_size() {
		let tl: Mat<f64> = Mat::zeros(3, 3);
		let tr: Mat<f64> = Mat::zeros(2, 2);
		let b: Mat<f64> = Mat::zeros(2, 2);
		let mut x: Mat<f64> = Mat::zeros(2, 2);

		solve_sylvester_2x2(tl.as_ref(), tr.as_ref(), b.as_ref(), x.as_mut());
	}
}
