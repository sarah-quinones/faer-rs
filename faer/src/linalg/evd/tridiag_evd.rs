use super::EvdError;
use crate::internal_prelude::*;
use crate::perm::swap_cols_idx;
use crate::utils::thread::join_raw;
use linalg::householder::*;
use linalg::jacobi::JacobiRotation;
use linalg::matmul::{dot, matmul};
use linalg::svd::bidiag_svd::secular_eq_root_finder;

pub(crate) fn qr_algorithm<T: RealField>(
	diag: ColMut<'_, T, usize, ContiguousFwd>,
	offdiag: ColMut<'_, T, usize, ContiguousFwd>,
	u: Option<MatMut<'_, T, usize, usize>>,
) -> Result<(), EvdError> {
	let n = diag.nrows();

	let mut u = u;

	let mut diag = diag;

	let mut offdiag = offdiag;

	if let Some(mut u) = u.rb_mut() {
		u.fill(zero());

		u.diagonal_mut().fill(one());
	}

	if n <= 1 {
		return Ok(());
	}

	let ref eps = eps::<T>();

	let ref sml = min_positive::<T>();

	if n == 2 {
		let ref a = diag[0].copy();

		let ref d = diag[1].copy();

		let ref b = offdiag[0].copy();

		let ref half = from_f64::<T>(0.5);

		let ref t0 = (a - d).hypot(b.mul_pow2(from_f64::<T>(2.0))).mul_pow2(half);

		let ref t1 = (a + d) * half;

		let ref r0 = t1 - t0;

		let ref r1 = t1 + t0;

		let ref tol = r0.abs().fmax(r1.abs()) * eps;

		if let Some(mut u) = u.rb_mut() {
			if r1 - r0 <= *tol {
				u[(0, 0)] = one();

				u[(1, 0)] = zero();

				u[(0, 1)] = zero();

				u[(1, 1)] = one();
			} else if b.abs() <= *tol {
				if diag[0] < diag[1] {
					u[(0, 0)] = one();

					u[(1, 0)] = zero();

					u[(0, 1)] = zero();

					u[(1, 1)] = one();
				} else {
					u[(0, 0)] = zero();

					u[(1, 0)] = one();

					u[(0, 1)] = one();

					u[(1, 1)] = zero();
				}
			} else {
				let tau = ((d - a) / b).mul_pow2(half);

				let mut t = (tau.abs() + tau.hypot(one::<T>())).recip();

				if tau < zero() {
					t = -t;
				}

				let ref c = t.hypot(one::<T>());

				let ref s = c * t;

				let ref r = c.hypot(s);

				let ref c = c / r;

				let ref s = s / r;

				let ref r0_r = (c * a - s * b) / c;

				if (r0 - r0_r).abs() < r1 - r0_r {
					u[(0, 0)] = c.copy();

					u[(1, 0)] = -s;

					u[(0, 1)] = s.copy();

					u[(1, 1)] = c.copy();
				} else {
					u[(0, 1)] = c.copy();

					u[(1, 1)] = -s;

					u[(0, 0)] = s.copy();

					u[(1, 0)] = c.copy();
				}
			}
		}

		diag[0] = r0.copy();

		diag[1] = r1.copy();

		return Ok(());
	}

	let ref max = diag.norm_max().fmax(offdiag.norm_max());

	if *max == zero() {
		return Ok(());
	}

	let ref max_inv = max.recip();

	for x in diag.rb_mut().iter_mut() {
		*x *= max_inv;
	}

	for x in offdiag.rb_mut().iter_mut() {
		*x *= max_inv;
	}

	let mut end = n - 1;

	let mut start = 0;

	let max_iters = Ord::max(30, nbits::<T>() / 2).saturating_mul(n).saturating_mul(n);

	for iter in 0..max_iters {
		for i in start..end {
			if offdiag[i].abs() < *sml || offdiag[i].abs() < eps * &diag[i].hypot(&diag[i + 1]) {
				offdiag[i] = zero();
			}
		}

		while end > 0 && (offdiag[end - 1] == zero()) {
			end -= 1;
		}

		if end == 0 {
			break;
		}

		if iter + 1 == max_iters {
			for x in diag.rb_mut().iter_mut() {
				*x *= max;
			}

			for x in offdiag.rb_mut().iter_mut() {
				*x *= max;
			}

			return Err(EvdError::NoConvergence);
		}

		start = end - 1;

		while start > 0 && !(offdiag[start - 1] == zero()) {
			start -= 1;
		}

		{
			let td = (&diag[end - 1] - &diag[end]).mul_pow2(from_f64::<T>(0.5));

			let ref e = offdiag[end - 1].copy();

			let mut mu = diag[end].copy();

			if td == zero() {
				mu -= e.abs();
			} else if *e != zero() {
				let e2 = e.abs2();

				let h = td.hypot(e);

				let h = if td > zero() { h.copy() } else { -h };

				if e2 == zero() {
					mu = mu - e / ((td + h) / e)
				} else {
					mu = mu - e2 / (td + h)
				};
			}

			let mut x = &diag[start] - mu;

			let mut z = offdiag[start].copy();

			let mut k = start;

			while k < end && z != zero() {
				let rot = JacobiRotation::make_givens(x.copy(), z.copy());

				let ref sdk = &rot.s * &diag[k] + &rot.c * &offdiag[k];

				let ref dkp1 = &rot.s * &offdiag[k] + &rot.c * &diag[k + 1];

				diag[k] = &rot.c * (&rot.c * &diag[k] - &rot.s * &offdiag[k]) - &rot.s * (&rot.c * &offdiag[k] - &rot.s * &diag[k + 1]);

				diag[k + 1] = &rot.s * sdk + &rot.c * dkp1;

				offdiag[k] = &rot.c * sdk - &rot.s * dkp1;

				if k > start {
					offdiag[k - 1] = &rot.c * &offdiag[k - 1] - &rot.s * &z;
				}

				x = offdiag[k].copy();

				if k < end - 1 {
					z = -&rot.s * &offdiag[k + 1];

					offdiag[k + 1] = &rot.c * &offdiag[k + 1];
				}

				if let Some(u) = u.rb_mut() {
					rot.apply_on_the_right_in_place(u.two_cols_mut(k + 1, k));
				}

				k += 1;
			}
		}
	}

	for i in 0..n - 1 {
		let mut idx = i;

		let mut min = diag[i].copy();

		for k in i + 1..n {
			if diag[k] < min {
				idx = k;

				min = diag[k].copy();
			}
		}

		if idx != i {
			let (a, b) = (diag[i].copy(), diag[idx].copy());

			diag[i] = b;

			diag[idx] = a;

			if let Some(mut u) = u.rb_mut() {
				swap_cols_idx(u.rb_mut(), i, idx);
			}
		}
	}

	for x in diag.rb_mut().iter_mut() {
		*x *= max;
	}

	Ok(())
}

fn secular_eq<T: RealField>(
	ref shift: T,
	ref mu: T,
	d: ColRef<'_, T, usize, ContiguousFwd>,
	z: ColRef<'_, T, usize, ContiguousFwd>,
	rho_recip: T,
) -> T {
	with_dim!(n, d.nrows());

	let d = d.as_row_shape(n);

	let z = z.as_row_shape(n);

	let mut res = rho_recip;

	for i in n.indices() {
		let ref d = d[i].copy();

		let ref z = z[i].copy();

		res += z * (z / ((d - shift) - mu));
	}

	res
}

fn batch_secular_eq<const N: usize, T: RealField>(
	ref shift: &[T; N],
	ref mu: &[T; N],
	d: ColRef<'_, T, usize, ContiguousFwd>,
	z: ColRef<'_, T, usize, ContiguousFwd>,
	ref rho_recip: T,
) -> [T; N] {
	with_dim!(n, d.nrows());

	let d = d.as_row_shape(n);

	let z = z.as_row_shape(n);

	let mut res = [(); N].map(|_| rho_recip.copy());

	for i in n.indices() {
		let ref d = d[i].copy();

		let ref z = z[i].copy();

		for ((res, mu), shift) in res.iter_mut().zip(mu.iter()).zip(shift.iter()) {
			*res += z * (z / ((d - shift) - mu));
		}
	}

	res
}

fn compute_eigenvalues<T: RealField>(
	mut mus: ColMut<'_, T, usize, ContiguousFwd>,
	mut shifts: ColMut<'_, T, usize, ContiguousFwd>,
	d: ColRef<'_, T, usize, ContiguousFwd>,
	z: ColRef<'_, T, usize, ContiguousFwd>,
	rho: T,
	non_deflated: usize,
) {
	let n = non_deflated;

	let full_n = d.nrows();

	let ref rho = rho;

	let ref rho_recip = rho.recip();

	for i in 0..n {
		let left = d[i].copy();

		let last = i == n - 1;

		let right = if last { &d[i] + rho * z.squared_norm_l2() } else { d[i + 1].copy() };

		let d = d.subrows(0, n);

		let z = z.subrows(0, n);

		let (shift, mu) = secular_eq_root_finder(
			&|shift, mu| secular_eq(shift, mu, d, z, rho_recip.copy()),
			&|shift, mu| batch_secular_eq(shift, mu, d, z, rho_recip.copy()),
			left,
			right,
			last,
		);

		shifts[i] = shift;

		mus[i] = mu;
	}

	for i in n..full_n {
		shifts[i] = zero();

		mus[i] = d[i].copy();
	}
}

fn divide_and_conquer_recurse<T: RealField>(
	mut diag: ColMut<'_, T, usize, ContiguousFwd>,
	mut offdiag: ColMut<'_, T, usize, ContiguousFwd>,
	mut u: MatMut<'_, T, usize, usize>,
	par: Par,
	pl_before: &mut [usize],
	pl_after: &mut [usize],
	pr: &mut [usize],
	run_info: &mut [usize],
	mut z: ColMut<'_, T, usize, ContiguousFwd>,
	mut permuted_diag: ColMut<'_, T, usize, ContiguousFwd>,
	mut permuted_z: ColMut<'_, T, usize, ContiguousFwd>,
	mut householder: ColMut<'_, T, usize, ContiguousFwd>,
	mut mus: ColMut<'_, T, usize, ContiguousFwd>,
	mut shifts: ColMut<'_, T, usize, ContiguousFwd>,
	mut repaired_u: MatMut<'_, T, usize, usize, ContiguousFwd>,
	mut tmp: MatMut<'_, T, usize, usize, ContiguousFwd>,
	qr_fallback_threshold: usize,
) -> Result<(), EvdError> {
	let n = diag.nrows();

	let qr_fallback_threshold = Ord::max(qr_fallback_threshold, 4);

	if n <= qr_fallback_threshold {
		return qr_algorithm(diag, offdiag, Some(u));
	}

	let n1 = n / 2;

	let mut rho = offdiag[n1 - 1].copy();

	let (mut diag0, mut diag1) = diag.rb_mut().split_at_row_mut(n1);

	let (offdiag0, offdiag1) = offdiag.rb_mut().split_at_row_mut(n1 - 1);

	let offdiag1 = offdiag1.split_at_row_mut(1).1;

	diag0[n1 - 1] -= rho.abs();

	diag1[0] -= rho.abs();

	let (mut u0, _, _, mut u1) = u.rb_mut().split_at_mut(n1, n1);

	{
		let (pl_before0, pl_before1) = pl_before.split_at_mut(n1);

		let (pl_after0, pl_after1) = pl_after.split_at_mut(n1);

		let (pr0, pr1) = pr.split_at_mut(n1);

		let (run_info0, run_info1) = run_info.split_at_mut(n1);

		let (z0, z1) = z.rb_mut().split_at_row_mut(n1);

		let (permuted_diag0, permuted_diag1) = permuted_diag.rb_mut().split_at_row_mut(n1);

		let (permuted_z0, permuted_z1) = permuted_z.rb_mut().split_at_row_mut(n1);

		let (householder0, householder1) = householder.rb_mut().split_at_row_mut(n1);

		let (mus0, mus1) = mus.rb_mut().split_at_row_mut(n1);

		let (shift0, shift1) = shifts.rb_mut().split_at_row_mut(n1);

		let (repaired_u0, _, _, repaired_u1) = repaired_u.rb_mut().split_at_mut(n1, n1);

		let (tmp0, _, _, tmp1) = tmp.rb_mut().split_at_mut(n1, n1);

		let mut r0 = Ok(());

		let mut r1 = Ok(());

		join_raw(
			|par| {
				r0 = divide_and_conquer_recurse(
					diag0,
					offdiag0,
					u0.rb_mut(),
					par,
					pl_before0,
					pl_after0,
					pr0,
					run_info0,
					z0,
					permuted_diag0,
					permuted_z0,
					householder0,
					mus0,
					shift0,
					repaired_u0,
					tmp0,
					qr_fallback_threshold,
				);
			},
			|par| {
				r1 = divide_and_conquer_recurse(
					diag1,
					offdiag1,
					u1.rb_mut(),
					par,
					pl_before1,
					pl_after1,
					pr1,
					run_info1,
					z1,
					permuted_diag1,
					permuted_z1,
					householder1,
					mus1,
					shift1,
					repaired_u1,
					tmp1,
					qr_fallback_threshold,
				);
			},
			par,
		);

		r0?;

		r1?;
	}

	let mut repaired_u = repaired_u.subrows_mut(0, n);

	let mut tmp = tmp.subrows_mut(0, n);

	let (mut z0, mut z1) = z.rb_mut().split_at_row_mut(n1);

	z0.copy_from(u0.rb().row(n1 - 1).transpose());

	if rho < zero() {
		z!(z1.rb_mut(), u1.rb().row(0).transpose()).for_each(|uz!(z, u)| *z = -u);
	} else {
		z1.copy_from(u1.rb().row(0).transpose());
	}

	let ref inv_sqrt2 = from_f64::<T>(0.5).sqrt();

	z!(z.rb_mut()).for_each(|uz!(x)| *x *= inv_sqrt2);

	rho = rho.abs().mul_pow2(from_f64::<T>(2.0));

	{
		let mut i = 0usize;

		let mut j = 0usize;

		for p in &mut *pl_before {
			if i == n1 {
				*p = n1 + j;

				j += 1;
			} else if (j == n - n1) || diag[i] < diag[n1 + j] {
				*p = i;

				i += 1;
			} else {
				*p = n1 + j;

				j += 1;
			}
		}
	}

	for (i, &pl_before) in pl_before.iter().enumerate() {
		permuted_diag[i] = diag[pl_before].copy();
	}

	for (i, &pl_before) in pl_before.iter().enumerate() {
		permuted_z[i] = z[pl_before].copy();
	}

	let ref dmax = permuted_diag.norm_max();

	let ref zmax = permuted_z.norm_max();

	let ref eps = eps::<T>();

	let ref tol = from_f64::<T>(8.0) * eps * dmax.fmax(zmax);

	if &rho * zmax <= *tol {
		let (mut tmp_tl, mut tmp_tr, mut tmp_bl, mut tmp_br) = tmp.rb_mut().split_at_mut(n1, n1);

		tmp_tl.copy_from(u0.rb());

		tmp_br.copy_from(u1.rb());

		tmp_tr.fill(zero());

		tmp_bl.fill(zero());

		for (j, &pl_before) in pl_before.iter().enumerate() {
			u.rb_mut().col_mut(j).copy_from(tmp.rb().col(pl_before));
		}

		for (j, diag) in diag.iter_mut().enumerate() {
			*diag = permuted_diag[j].copy();
		}

		return Ok(());
	}

	for i in 0..n {
		let ref zi = permuted_z[i].copy();

		if (&rho * zi).abs() <= *tol {
			permuted_z[i] = zero();
		}
	}

	let mut applied_householder = false;

	let mut idx = 0;

	while idx < n {
		let mut run_len = 1;

		let ref d_prev = permuted_diag[idx].copy();

		while idx + run_len < n {
			if &permuted_diag[idx + run_len] - d_prev <= *tol {
				permuted_diag[idx + run_len] = d_prev.copy();

				run_len += 1;
			} else {
				break;
			}
		}

		run_info[idx..][..run_len].fill(run_len);

		if run_len > 1 {
			applied_householder = true;

			let mut householder = householder.rb_mut().subrows_mut(idx, run_len);

			let mut permuted_z = permuted_z.rb_mut().subrows_mut(idx, run_len);

			householder.copy_from(permuted_z.rb());

			let (tail, head) = householder.rb_mut().split_at_row_mut(run_len - 1);

			let head = head.at_mut(0);

			let HouseholderInfo { tau, .. } = make_householder_in_place(head, tail.as_dyn_stride_mut().reverse_rows_mut());

			permuted_z.fill(zero());

			permuted_z[run_len - 1] = head.copy();

			*head = tau;
		}

		idx += run_len;
	}

	let mut writer_deflated = 0;

	let mut writer_non_deflated = 0;

	for reader in 0..n {
		let z = permuted_z[reader].copy();

		let d = permuted_diag[reader].copy();

		if z == zero() {
			diag[writer_deflated] = d;

			pr[writer_deflated] = reader;

			writer_deflated += 1;
		} else {
			permuted_z[writer_non_deflated] = z;

			permuted_diag[writer_non_deflated] = d;

			pl_after[writer_non_deflated] = reader;

			writer_non_deflated += 1;
		}
	}

	let non_deflated = writer_non_deflated;

	let deflated = writer_deflated;

	for i in 0..deflated {
		permuted_diag[non_deflated + i] = diag[i].copy();

		pl_after[non_deflated + i] = pr[i];
	}

	compute_eigenvalues(
		mus.rb_mut(),
		shifts.rb_mut(),
		permuted_diag.rb(),
		permuted_z.rb(),
		rho.copy(),
		non_deflated,
	);

	for i in 0..non_deflated {
		let ref di = permuted_diag[i].copy();

		let ref mu_i = mus[i].copy();

		let ref shift_i = shifts[i].copy();

		let mut prod = mu_i + (shift_i - di);

		(0..i).chain(i + 1..non_deflated).for_each(|k| {
			let ref dk = permuted_diag[k].copy();

			let ref mu_k = mus[k].copy();

			let ref shift_k = shifts[k].copy();

			let ref numerator = mu_k + (shift_k - di);

			let ref denominator = dk - di;

			prod *= numerator / denominator;
		});

		let prod = prod.abs().sqrt();

		let zi = permuted_z.rb_mut().at_mut(i);

		let new_zi = if *zi < zero() { -prod } else { prod };

		*zi = new_zi;
	}

	let mut eigenvals = z;

	for i in 0..n {
		eigenvals[i] = &mus[i] + &shifts[i];
	}

	for (i, p) in pr.iter_mut().enumerate() {
		*p = i;
	}

	pr.sort_unstable_by(|&i, &j| match PartialOrd::partial_cmp(&eigenvals[i], &eigenvals[j]) {
		Some(ord) => ord,
		None => {
			if eigenvals[i].is_nan() && eigenvals[j].is_nan() {
				core::cmp::Ordering::Equal
			} else if eigenvals[i].is_nan() {
				core::cmp::Ordering::Greater
			} else {
				core::cmp::Ordering::Less
			}
		},
	});

	if !applied_householder {
		for p in pl_after.iter_mut() {
			*p = pl_before[*p];
		}
	}

	for (j, &pj) in pr.iter().enumerate() {
		if pj >= non_deflated {
			repaired_u.rb_mut().col_mut(j).fill(zero());

			repaired_u[(pl_after[pj], j)] = one();
		} else {
			let ref mu_j = mus[pj].copy();

			let ref shift_j = shifts[pj].copy();

			for (i, &pl_after) in pl_after[..non_deflated].iter().enumerate() {
				let ref zi = permuted_z[i].copy();

				let ref di = permuted_diag[i].copy();

				repaired_u[(pl_after, j)] = zi / ((di - shift_j) - mu_j);
			}

			for &pl_after in &pl_after[non_deflated..non_deflated + deflated] {
				repaired_u[(pl_after, j)] = zero();
			}

			let ref inv_norm = repaired_u.rb().col(j).norm_l2().recip();

			z!(repaired_u.rb_mut().col_mut(j)).for_each(|unzip!(x)| *x *= inv_norm);
		}
	}

	if applied_householder {
		let mut idx = 0;

		while idx < n {
			let run_len = run_info[idx];

			if run_len > 1 {
				let mut householder = householder.rb_mut().subrows_mut(idx, run_len);

				let tau = householder[run_len - 1].copy();

				householder[run_len - 1] = one();

				let householder = householder.rb();

				let mut repaired_u = repaired_u.rb_mut().subrows_mut(idx, run_len);

				let ref tau_inv = tau.recip();

				for j in 0..n {
					let mut col = repaired_u.rb_mut().col_mut(j);

					let ref dot = tau_inv * dot::inner_prod(householder.as_dyn_stride().transpose(), Conj::No, col.rb().as_dyn_stride(), Conj::No);

					z!(col.rb_mut(), householder).for_each(|uz!(u, h)| *u -= dot * h);
				}
			}

			idx += run_len;
		}

		for j in 0..n {
			for (i, &pl_before) in pl_before.iter().enumerate() {
				tmp[(pl_before, j)] = repaired_u[(i, j)].copy();
			}
		}

		core::mem::swap(&mut repaired_u, &mut tmp);
	}

	let (repaired_u_top, repaired_u_bot) = repaired_u.rb().split_at_row(n1);

	let (tmp_top, tmp_bot) = tmp.rb_mut().split_at_row_mut(n1);

	crate::utils::thread::join_raw(
		|par| matmul(tmp_top, Accum::Replace, u0.rb(), repaired_u_top, one(), par),
		|par| matmul(tmp_bot, Accum::Replace, u1.rb(), repaired_u_bot, one(), par),
		par,
	);

	u.copy_from(tmp.rb());

	for i in 0..n {
		diag[i] = &mus[pr[i]] + &shifts[pr[i]];
	}

	Ok(())
}

pub(crate) fn divide_and_conquer<T: RealField>(
	diag: ColMut<'_, T, usize, ContiguousFwd>,
	offdiag: ColMut<'_, T, usize, ContiguousFwd>,
	u: MatMut<'_, T, usize, usize>,
	par: Par,
	stack: &mut MemStack,
	qr_fallback_threshold: usize,
) -> Result<(), EvdError> {
	let n = diag.nrows();

	let (mut pl_before, stack) = stack.make_with(n, |_| 0usize);

	let (mut pl_after, stack) = stack.make_with(n, |_| 0usize);

	let (mut pr, stack) = stack.make_with(n, |_| 0usize);

	let (mut run_info, stack) = stack.make_with(n, |_| 0usize);

	let (mut z, stack) = unsafe { temp_mat_uninit(n, 1, stack) };

	let (mut permuted_diag, stack) = unsafe { temp_mat_uninit(n, 1, stack) };

	let (mut permuted_z, stack) = unsafe { temp_mat_uninit(n, 1, stack) };

	let (mut householder, stack) = unsafe { temp_mat_uninit(n, 1, stack) };

	let (mut mus, stack) = unsafe { temp_mat_uninit(n, 1, stack) };

	let (mut shifts, stack) = unsafe { temp_mat_uninit(n, 1, stack) };

	let (mut repaired_u, stack) = unsafe { temp_mat_uninit(n, n, stack) };

	let (mut tmp, _) = unsafe { temp_mat_uninit(n, n, stack) };

	divide_and_conquer_recurse(
		diag,
		offdiag,
		u,
		par,
		&mut pl_before,
		&mut pl_after,
		&mut pr,
		&mut run_info,
		z.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap(),
		permuted_diag.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap(),
		permuted_z.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap(),
		householder.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap(),
		mus.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap(),
		shifts.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap(),
		repaired_u.as_mat_mut().try_as_col_major_mut().unwrap(),
		tmp.as_mat_mut().try_as_col_major_mut().unwrap(),
		qr_fallback_threshold,
	)
}

pub(crate) fn divide_and_conquer_scratch<T: ComplexField>(n: usize, par: Par) -> StackReq {
	let _ = par;

	StackReq::all_of(&[
		StackReq::new::<usize>(n).array(4),
		temp_mat_scratch::<T>(n, 1).array(6),
		temp_mat_scratch::<T>(n, n).array(2),
	])
}

#[cfg(test)]

mod evd_qr_tests {

	use super::*;
	use crate::utils::approx::*;
	use crate::{ColMut, Mat, assert};
	use dyn_stack::MemBuffer;

	#[track_caller]

	fn test_qr(diag: &[f64], offdiag: &[f64]) {
		let n = diag.len();

		let mut u = Mat::full(n, n, f64::NAN);

		let s = {
			let mut diag = diag.to_vec();

			let mut offdiag = offdiag.to_vec();

			qr_algorithm(
				ColMut::from_slice_mut(&mut diag).try_as_col_major_mut().unwrap(),
				ColMut::from_slice_mut(&mut offdiag).try_as_col_major_mut().unwrap(),
				Some(u.as_mut()),
			)
			.unwrap();

			Mat::from_fn(n, n, |i, j| if i == j { diag[i] } else { 0.0 })
		};

		let reconstructed = &u * &s * u.transpose();

		for j in 0..n {
			for i in 0..n {
				let target = if i == j {
					diag[j]
				} else if i == j + 1 {
					offdiag[j]
				} else if j == i + 1 {
					offdiag[i]
				} else {
					0.0
				};

				let approx_eq = ApproxEq::<f64>::eps();

				assert!(reconstructed[(i, j)] ~ target);
			}
		}
	}

	#[track_caller]

	fn test_dc(diag: &[f64], offdiag: &[f64]) {
		let n = diag.len();

		let mut u = Mat::full(n, n, f64::NAN);

		let s = {
			let mut diag = diag.to_vec();

			let mut offdiag = offdiag.to_vec();

			divide_and_conquer(
				ColMut::from_slice_mut(&mut diag).try_as_col_major_mut().unwrap(),
				ColMut::from_slice_mut(&mut offdiag).try_as_col_major_mut().unwrap(),
				u.as_mut(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(divide_and_conquer_scratch::<f64>(n, Par::Seq))),
				4,
			)
			.unwrap();

			Mat::from_fn(n, n, |i, j| if i == j { diag[i] } else { 0.0 })
		};

		let reconstructed = &u * &s * u.transpose();

		for j in 0..n {
			for i in 0..n {
				let target = if i == j {
					diag[j]
				} else if i == j + 1 {
					offdiag[j]
				} else if j == i + 1 {
					offdiag[i]
				} else {
					0.0
				};

				let approx_eq = ApproxEq::<f64>::eps();

				assert!(reconstructed[(i, j)] ~ target);
			}
		}
	}

	#[test]

	fn test_evd_2_0() {
		let diag = [1.0, 1.0];

		let offdiag = [0.0];

		test_qr(&diag, &offdiag);

		test_dc(&diag, &offdiag);
	}

	#[test]

	fn test_evd_2_1() {
		let diag = [1.0, 1.0];

		let offdiag = [0.5213289];

		test_qr(&diag, &offdiag);

		test_dc(&diag, &offdiag);
	}

	#[test]

	fn test_evd_3() {
		let diag = [1.79069356, 1.20930644, 1.0];

		let offdiag = [-4.06813537e-01, 0.0];

		test_qr(&diag, &offdiag);

		test_dc(&diag, &offdiag);
	}

	#[test]

	fn test_evd_5() {
		let diag = [1.95069537, 2.44845332, 2.56957029, 3.03128102, 1.0];

		let offdiag = [-7.02200909e-01, -1.11661820e+00, -6.81418803e-01, 0.0];

		test_qr(&diag, &offdiag);

		test_dc(&diag, &offdiag);
	}

	#[test]

	fn test_evd_wilkinson() {
		let diag = [3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0];

		let offdiag = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

		test_qr(&diag, &offdiag);

		test_dc(&diag, &offdiag);
	}

	#[test]

	fn test_glued_wilkinson() {
		let diag = [3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0];

		let x = 1e-6;

		let offdiag = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, x, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

		test_qr(&diag, &offdiag);

		test_dc(&diag, &offdiag);
	}

	#[test]

	fn test_gh_82() {
		let diag = [
			0.0,
			0.0,
			1.0769230769230773,
			-0.4290761869709236,
			-0.8278050499098524,
			0.07994922044020283,
			-0.35579623371016944,
			0.6487378508167678,
			-0.9347442346214521,
			-0.08624745233962683,
			-0.4999243909534632,
			1.3708277457481026,
			-0.2592167303689501,
			-0.5929351972647323,
			-0.5863220906879729,
			0.15069873027683844,
			0.2449309426221532,
			0.5599151389028441,
			0.440084861097156,
			9.811634162559901e-17,
		];

		let offdiag = [
			1.7320508075688772,
			2.081665999466133,
			2.0303418353670932,
			1.2463948607107287,
			1.5895840148470526,
			1.3810057029812097,
			1.265168346300635,
			0.8941431038915991,
			1.007512301091709,
			0.5877505835309086,
			1.0370970338888965,
			0.8628932798233644,
			1.1935059937001073,
			1.1614143449715744,
			0.41040224297074174,
			0.561318309959268,
			3.1807090401145072e-15,
			0.4963971959331084,
			1.942890293094024e-16,
		];

		test_qr(&diag, &offdiag);

		test_dc(&diag, &offdiag);
	}

	#[test]

	fn test_gh_82_mini() {
		let diag = [1.0000000000000002, 1.0000000000000002];

		let offdiag = [7.216449660063518e-16];

		test_qr(&diag, &offdiag);

		test_dc(&diag, &offdiag);
	}
}
