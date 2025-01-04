use crate::internal_prelude_sp::*;
use crate::{assert, debug_assert};

/// Assuming `tril` is a lower triangular matrix, solves the equation `Op(tril) * X = rhs`, and
/// stores the result in `rhs`, where `Op` is either the conjugate or the identity depending on the
/// value of `conj_tril`.
///
/// # Note
/// The matrix indices need not be sorted, but
/// the diagonal element is assumed to be the first stored element in each column.
#[track_caller]
#[math]
pub fn solve_lower_triangular_in_place<I: Index, T: ComplexField>(
	tril: SparseColMatRef<'_, I, T>,
	conj_tril: Conj,
	diag_tril: DiagStatus,
	rhs: MatMut<'_, T>,
	par: Par,
) {
	let _ = par;
	assert!(all(tril.nrows() == tril.ncols(), rhs.nrows() == tril.nrows()));

	with_dim!(N, rhs.nrows());
	with_dim!(K, rhs.ncols());

	let mut x = rhs.as_shape_mut(N, K);
	let l = tril.as_shape(N, N);

	let mut k = IdxInc::ZERO;
	while let Some(k0) = K.try_check(*k) {
		let k1 = K.try_check(*k + 1);
		let k2 = K.try_check(*k + 2);
		let k3 = K.try_check(*k + 3);

		match (k1, k2, k3) {
			(Some(_), Some(_), Some(k3)) => {
				let mut x = x.rb_mut().get_mut(.., k..k3.next()).col_iter_mut();
				let (Some(mut x0), Some(mut x1), Some(mut x2), Some(mut x3)) = (x.next(), x.next(), x.next(), x.next()) else {
					panic!()
				};

				for j in N.indices() {
					let mut l = iter::zip(l.row_idx_of_col(j), l.val_of_col(j));
					let (i, d) = (l.next().unwrap());
					debug_assert!(i == j);

					let x0j;
					let x1j;
					let x2j;
					let x3j;
					match diag_tril {
						DiagStatus::Unit => {
							x0j = copy(x0[j]);
							x1j = copy(x1[j]);
							x2j = copy(x2[j]);
							x3j = copy(x3[j]);
						},
						DiagStatus::Generic => {
							let d = conj_tril.apply_rt(&recip(*d));
							x0j = x0[j] * d;
							x1j = x1[j] * d;
							x2j = x2[j] * d;
							x3j = x3[j] * d;
							x0[j] = copy(x0j);
							x1[j] = copy(x1j);
							x2[j] = copy(x2j);
							x3[j] = copy(x3j);
						},
					}

					for (i, lij) in l {
						let lij = conj_tril.apply_rt(lij);
						x0[i] = x0[i] - lij * x0j;
						x1[i] = x1[i] - lij * x1j;
						x2[i] = x2[i] - lij * x2j;
						x3[i] = x3[i] - lij * x3j;
					}
				}
				k = k3.next();
			},
			(Some(_), Some(k2), _) => {
				let mut x = x.rb_mut().get_mut(.., k..k2.next()).col_iter_mut();
				let (Some(mut x0), Some(mut x1), Some(mut x2)) = (x.next(), x.next(), x.next()) else {
					panic!()
				};

				for j in N.indices() {
					let mut l = iter::zip(l.row_idx_of_col(j), l.val_of_col(j));
					let (i, d) = (l.next().unwrap());
					debug_assert!(i == j);

					let x0j;
					let x1j;
					let x2j;
					match diag_tril {
						DiagStatus::Unit => {
							x0j = copy(x0[j]);
							x1j = copy(x1[j]);
							x2j = copy(x2[j]);
						},
						DiagStatus::Generic => {
							let d = conj_tril.apply_rt(&recip(*d));
							x0j = x0[j] * d;
							x1j = x1[j] * d;
							x2j = x2[j] * d;
							x0[j] = copy(x0j);
							x1[j] = copy(x1j);
							x2[j] = copy(x2j);
						},
					}

					for (i, lij) in l {
						let lij = conj_tril.apply_rt(lij);
						x0[i] = x0[i] - lij * x0j;
						x1[i] = x1[i] - lij * x1j;
						x2[i] = x2[i] - lij * x2j;
					}
				}
				k = k2.next();
			},
			(Some(k1), _, _) => {
				let mut x = x.rb_mut().get_mut(.., k..k1.next()).col_iter_mut();
				let (Some(mut x0), Some(mut x1)) = (x.next(), x.next()) else { panic!() };

				for j in N.indices() {
					let mut l = iter::zip(l.row_idx_of_col(j), l.val_of_col(j));
					let (i, d) = (l.next().unwrap());
					debug_assert!(i == j);

					let x0j;
					let x1j;
					match diag_tril {
						DiagStatus::Unit => {
							x0j = copy(x0[j]);
							x1j = copy(x1[j]);
						},
						DiagStatus::Generic => {
							let d = conj_tril.apply_rt(&recip(*d));
							x0j = x0[j] * d;
							x1j = x1[j] * d;
							x0[j] = copy(x0j);
							x1[j] = copy(x1j);
						},
					}

					for (i, lij) in l {
						let lij = conj_tril.apply_rt(lij);
						x0[i] = x0[i] - lij * x0j;
						x1[i] = x1[i] - lij * x1j;
					}
				}
				k = k1.next();
			},
			(_, _, _) => {
				let mut x0 = x.rb_mut().get_mut(.., k0);

				for j in N.indices() {
					let mut l = iter::zip(l.row_idx_of_col(j), l.val_of_col(j));
					let (i, d) = (l.next().unwrap());
					debug_assert!(i == j);

					let x0j;
					match diag_tril {
						DiagStatus::Unit => {
							x0j = copy(x0[j]);
						},
						DiagStatus::Generic => {
							let d = conj_tril.apply_rt(&recip(*d));
							x0j = x0[j] * d;
							x0[j] = copy(x0j);
						},
					}

					for (i, lij) in l {
						let lij = conj_tril.apply_rt(lij);
						x0[i] = x0[i] - lij * x0j;
					}
				}
				k = k0.next();
			},
		}
	}
}

/// Assuming `tril` is a lower triangular matrix, solves the equation `Op(tril) * X = rhs`, and
/// stores the result in `rhs`, where `Op` is either the conjugate or the identity depending on the
/// value of `conj_tril`.
///
/// # Note
/// The matrix indices need not be sorted, but
/// the diagonal element is assumed to be the first stored element in each column.
#[track_caller]
#[math]
pub fn ldlt_scale_solve_unit_lower_triangular_transpose_in_place<I: Index, T: ComplexField>(
	tril: SparseColMatRef<'_, I, T>,
	conj_tril: Conj,
	rhs: MatMut<'_, T>,
	par: Par,
) {
	let _ = par;
	assert!(all(tril.nrows() == tril.ncols(), rhs.nrows() == tril.nrows()));

	with_dim!(N, rhs.nrows());
	with_dim!(K, rhs.ncols());

	let mut x = rhs.as_shape_mut(N, K);
	let l = tril.as_shape(N, N);

	let mut k = IdxInc::ZERO;
	while let Some(k0) = K.try_check(*k) {
		let k1 = K.try_check(*k + 1);
		let k2 = K.try_check(*k + 2);
		let k3 = K.try_check(*k + 3);

		match (k1, k2, k3) {
			(Some(_), Some(_), Some(k3)) => {
				let mut x = x.rb_mut().get_mut(.., k..k3.next()).col_iter_mut();
				let (Some(mut x0), Some(mut x1), Some(mut x2), Some(mut x3)) = (x.next(), x.next(), x.next(), x.next()) else {
					panic!()
				};

				for j in N.indices().rev() {
					let mut li = l.row_idx_of_col(j);
					let mut lv = l.val_of_col(j).iter();
					let first = (li.next().zip(lv.next()));

					let mut acc0a = zero::<T>();
					let mut acc1a = zero::<T>();
					let mut acc2a = zero::<T>();
					let mut acc3a = zero::<T>();

					for (i, lij) in iter::zip(li.rev(), lv.rev()) {
						let lij = conj_tril.apply_rt(lij);
						acc0a = acc0a + lij * x0[i];
						acc1a = acc1a + lij * x1[i];
						acc2a = acc2a + lij * x2[i];
						acc3a = acc3a + lij * x3[i];
					}

					let (i, d) = first.unwrap();
					debug_assert!(i == j);
					let d = conj_tril.apply_rt(&recip(*d));

					x0[j] = x0[j] * d - acc0a;
					x1[j] = x1[j] * d - acc1a;
					x2[j] = x2[j] * d - acc2a;
					x3[j] = x3[j] * d - acc3a;
				}
				k = k3.next();
			},
			(Some(_), Some(k2), _) => {
				let mut x = x.rb_mut().get_mut(.., k..k2.next()).col_iter_mut();
				let (Some(mut x0), Some(mut x1), Some(mut x2)) = (x.next(), x.next(), x.next()) else {
					panic!()
				};

				for j in N.indices().rev() {
					let mut li = l.row_idx_of_col(j);
					let mut lv = l.val_of_col(j).iter();
					let first = (li.next().zip(lv.next()));

					let mut acc0a = zero::<T>();
					let mut acc1a = zero::<T>();
					let mut acc2a = zero::<T>();

					for (i, lij) in iter::zip(li.rev(), lv.rev()) {
						let lij = conj_tril.apply_rt(lij);
						acc0a = acc0a + lij * x0[i];
						acc1a = acc1a + lij * x1[i];
						acc2a = acc2a + lij * x2[i];
					}

					let (i, d) = first.unwrap();
					debug_assert!(i == j);
					let d = conj_tril.apply_rt(&recip(*d));

					x0[j] = x0[j] * d - acc0a;
					x1[j] = x1[j] * d - acc1a;
					x2[j] = x2[j] * d - acc2a;
				}

				k = k2.next();
			},
			(Some(k1), _, _) => {
				let mut x = x.rb_mut().get_mut(.., k..k1.next()).col_iter_mut();
				let (Some(mut x0), Some(mut x1)) = (x.next(), x.next()) else { panic!() };

				for j in N.indices().rev() {
					let mut li = l.row_idx_of_col(j);
					let mut lv = l.val_of_col(j).iter();
					let first = (li.next().zip(lv.next()));

					let mut acc0a = zero::<T>();
					let mut acc1a = zero::<T>();

					for (i, lij) in iter::zip(li.rev(), lv.rev()) {
						let lij = conj_tril.apply_rt(lij);
						acc0a = acc0a + lij * x0[i];
						acc1a = acc1a + lij * x1[i];
					}

					let (i, d) = first.unwrap();
					debug_assert!(i == j);
					let d = conj_tril.apply_rt(&recip(*d));

					x0[j] = x0[j] * d - acc0a;
					x1[j] = x1[j] * d - acc1a;
				}

				k = k1.next();
			},
			(_, _, _) => {
				let mut x0 = x.rb_mut().get_mut(.., k0);

				for j in N.indices().rev() {
					let mut li = l.row_idx_of_col(j);
					let mut lv = l.val_of_col(j).iter();
					let first = (li.next().zip(lv.next()));

					let mut acc0a = zero::<T>();

					for (i, lij) in iter::zip(li.rev(), lv.rev()) {
						let lij = conj_tril.apply_rt(lij);
						acc0a = acc0a + lij * x0[i];
					}

					let (i, d) = first.unwrap();
					debug_assert!(i == j);
					let d = conj_tril.apply_rt(&recip(*d));

					x0[j] = x0[j] * d - acc0a;
				}

				k = k0.next();
			},
		}
	}
}

/// Assuming `tril` is a lower triangular matrix, solves the equation `Op(tril).transpose() * X =
/// rhs`, and stores the result in `rhs`, where `Op` is either the conjugate or the identity
/// depending on the value of `conj_tril`.
///
/// # Note
/// The matrix indices need not be sorted, but
/// the diagonal element is assumed to be the first stored element in each column.
#[track_caller]
#[math]
pub fn solve_lower_triangular_transpose_in_place<I: Index, T: ComplexField>(
	tril: SparseColMatRef<'_, I, T>,
	conj_tril: Conj,
	diag_tril: DiagStatus,
	rhs: MatMut<'_, T>,
	par: Par,
) {
	let _ = par;
	assert!(all(tril.nrows() == tril.ncols(), rhs.nrows() == tril.nrows()));

	with_dim!(N, rhs.nrows());
	with_dim!(K, rhs.ncols());

	let mut x = rhs.as_shape_mut(N, K);
	let l = tril.as_shape(N, N);

	let mut k = IdxInc::ZERO;
	while let Some(k0) = K.try_check(*k) {
		let k1 = K.try_check(*k + 1);
		let k2 = K.try_check(*k + 2);
		let k3 = K.try_check(*k + 3);

		match (k1, k2, k3) {
			(Some(_), Some(_), Some(k3)) => {
				let mut x = x.rb_mut().get_mut(.., k..k3.next()).col_iter_mut();
				let (Some(mut x0), Some(mut x1), Some(mut x2), Some(mut x3)) = (x.next(), x.next(), x.next(), x.next()) else {
					panic!()
				};

				for j in N.indices().rev() {
					let mut li = l.row_idx_of_col(j);
					let mut lv = l.val_of_col(j).iter();
					let first = (li.next().zip(lv.next()));

					let mut acc0a = zero::<T>();
					let mut acc1a = zero::<T>();
					let mut acc2a = zero::<T>();
					let mut acc3a = zero::<T>();

					for (i, lij) in iter::zip(li.rev(), lv.rev()) {
						let lij = conj_tril.apply_rt(lij);
						acc0a = acc0a + lij * x0[i];
						acc1a = acc1a + lij * x1[i];
						acc2a = acc2a + lij * x2[i];
						acc3a = acc3a + lij * x3[i];
					}

					let mut x0j = x0[j] - acc0a;
					let mut x1j = x1[j] - acc1a;
					let mut x2j = x2[j] - acc2a;
					let mut x3j = x3[j] - acc3a;

					let (i, d) = first.unwrap();
					debug_assert!(i == j);
					match diag_tril {
						DiagStatus::Unit => {},
						DiagStatus::Generic => {
							let d = conj_tril.apply_rt(&recip(*d));
							x0j = x0j * d;
							x1j = x1j * d;
							x2j = x2j * d;
							x3j = x3j * d;
						},
					}

					x0[j] = x0j;
					x1[j] = x1j;
					x2[j] = x2j;
					x3[j] = x3j;
				}
				k = k3.next();
			},
			(Some(_), Some(k2), _) => {
				let mut x = x.rb_mut().get_mut(.., k..k2.next()).col_iter_mut();
				let (Some(mut x0), Some(mut x1), Some(mut x2)) = (x.next(), x.next(), x.next()) else {
					panic!()
				};

				for j in N.indices().rev() {
					let mut li = l.row_idx_of_col(j);
					let mut lv = l.val_of_col(j).iter();
					let first = (li.next().zip(lv.next()));

					let mut acc0a = zero::<T>();
					let mut acc1a = zero::<T>();
					let mut acc2a = zero::<T>();

					for (i, lij) in iter::zip(li.rev(), lv.rev()) {
						let lij = conj_tril.apply_rt(lij);
						acc0a = acc0a + lij * x0[i];
						acc1a = acc1a + lij * x1[i];
						acc2a = acc2a + lij * x2[i];
					}

					let mut x0j = x0[j] - acc0a;
					let mut x1j = x1[j] - acc1a;
					let mut x2j = x2[j] - acc2a;

					let (i, d) = first.unwrap();
					debug_assert!(i == j);
					match diag_tril {
						DiagStatus::Unit => {},
						DiagStatus::Generic => {
							let d = conj_tril.apply_rt(&recip(*d));
							x0j = x0j * d;
							x1j = x1j * d;
							x2j = x2j * d;
						},
					}

					x0[j] = x0j;
					x1[j] = x1j;
					x2[j] = x2j;
				}

				k = k2.next();
			},
			(Some(k1), _, _) => {
				let mut x = x.rb_mut().get_mut(.., k..k1.next()).col_iter_mut();
				let (Some(mut x0), Some(mut x1)) = (x.next(), x.next()) else { panic!() };

				for j in N.indices().rev() {
					let mut li = l.row_idx_of_col(j);
					let mut lv = l.val_of_col(j).iter();
					let first = (li.next().zip(lv.next()));

					let mut acc0a = zero::<T>();
					let mut acc1a = zero::<T>();

					for (i, lij) in iter::zip(li.rev(), lv.rev()) {
						let lij = conj_tril.apply_rt(lij);
						acc0a = acc0a + lij * x0[i];
						acc1a = acc1a + lij * x1[i];
					}

					let mut x0j = x0[j] - acc0a;
					let mut x1j = x1[j] - acc1a;

					let (i, d) = first.unwrap();
					debug_assert!(i == j);
					match diag_tril {
						DiagStatus::Unit => {},
						DiagStatus::Generic => {
							let d = conj_tril.apply_rt(&recip(*d));
							x0j = x0j * d;
							x1j = x1j * d;
						},
					}

					x0[j] = x0j;
					x1[j] = x1j;
				}

				k = k1.next();
			},
			(_, _, _) => {
				let mut x0 = x.rb_mut().get_mut(.., k0);

				for j in N.indices().rev() {
					let mut li = l.row_idx_of_col(j);
					let mut lv = l.val_of_col(j).iter();
					let first = (li.next().zip(lv.next()));

					let mut acc0a = zero::<T>();

					for (i, lij) in iter::zip(li.rev(), lv.rev()) {
						let lij = conj_tril.apply_rt(lij);
						acc0a = acc0a + lij * x0[i];
					}

					let mut x0j = x0[j] - acc0a;

					let (i, d) = first.unwrap();
					debug_assert!(i == j);
					match diag_tril {
						DiagStatus::Unit => {},
						DiagStatus::Generic => {
							let d = conj_tril.apply_rt(&recip(*d));
							x0j = x0j * d;
						},
					}

					x0[j] = x0j;
				}

				k = k0.next();
			},
		}
	}
}

/// Assuming `triu` is an upper triangular matrix, solves the equation `Op(triu) * X = rhs`, and
/// stores the result in `rhs`, where `Op` is either the conjugate or the identity
/// depending on the value of `conj_triu`.
///
/// # Note
/// The matrix indices need not be sorted, but
/// the diagonal element is assumed to be the last stored element in each column.
#[track_caller]
#[math]
pub fn solve_upper_triangular_in_place<I: Index, T: ComplexField>(
	triu: SparseColMatRef<'_, I, T>,
	conj_triu: Conj,
	diag_triu: DiagStatus,
	rhs: MatMut<'_, T>,
	par: Par,
) {
	let _ = par;

	assert!(all(triu.nrows() == triu.ncols(), rhs.nrows() == triu.nrows()));
	with_dim!(N, rhs.nrows());
	with_dim!(K, rhs.ncols());

	let mut x = rhs.as_shape_mut(N, K);
	let u = triu.as_shape(N, N);

	let mut k = IdxInc::ZERO;
	while let Some(k0) = K.try_check(*k) {
		let k1 = K.try_check(*k + 1);
		let k2 = K.try_check(*k + 2);
		let k3 = K.try_check(*k + 3);

		match (k1, k2, k3) {
			(Some(_), Some(_), Some(k3)) => {
				let mut x = x.rb_mut().get_mut(.., k..k3.next()).col_iter_mut();
				let (Some(mut x0), Some(mut x1), Some(mut x2), Some(mut x3)) = (x.next(), x.next(), x.next(), x.next()) else {
					panic!()
				};

				for j in N.indices().rev() {
					let mut u = iter::zip(u.row_idx_of_col(j).rev(), u.val_of_col(j).iter().rev());

					let (i, d) = (u.next().unwrap());
					debug_assert!(i == j);

					let x0j;
					let x1j;
					let x2j;
					let x3j;
					match diag_triu {
						DiagStatus::Unit => {
							x0j = copy(x0[j]);
							x1j = copy(x1[j]);
							x2j = copy(x2[j]);
							x3j = copy(x3[j]);
						},
						DiagStatus::Generic => {
							let d = conj_triu.apply_rt(&recip(*d));
							x0j = x0[j] * d;
							x1j = x1[j] * d;
							x2j = x2[j] * d;
							x3j = x3[j] * d;
							x0[j] = copy(x0j);
							x1[j] = copy(x1j);
							x2[j] = copy(x2j);
							x3[j] = copy(x3j);
						},
					}

					for (i, u) in u {
						let uij = conj_triu.apply_rt(u);
						x0[i] = x0[i] - uij * x0j;
						x1[i] = x1[i] - uij * x1j;
						x2[i] = x2[i] - uij * x2j;
						x3[i] = x3[i] - uij * x3j;
					}
				}
				k = k3.next();
			},
			(Some(_), Some(k2), _) => {
				let mut x = x.rb_mut().get_mut(.., k..k2.next()).col_iter_mut();
				let (Some(mut x0), Some(mut x1), Some(mut x2)) = (x.next(), x.next(), x.next()) else {
					panic!()
				};

				for j in N.indices().rev() {
					let mut u = iter::zip(u.row_idx_of_col(j).rev(), u.val_of_col(j).iter().rev());

					let (i, d) = (u.next().unwrap());
					debug_assert!(i == j);

					let x0j;
					let x1j;
					let x2j;
					match diag_triu {
						DiagStatus::Unit => {
							x0j = copy(x0[j]);
							x1j = copy(x1[j]);
							x2j = copy(x2[j]);
						},
						DiagStatus::Generic => {
							let d = conj_triu.apply_rt(&recip(*d));
							x0j = x0[j] * d;
							x1j = x1[j] * d;
							x2j = x2[j] * d;
							x0[j] = copy(x0j);
							x1[j] = copy(x1j);
							x2[j] = copy(x2j);
						},
					}

					for (i, u) in u {
						let uij = conj_triu.apply_rt(u);
						x0[i] = x0[i] - uij * x0j;
						x1[i] = x1[i] - uij * x1j;
						x2[i] = x2[i] - uij * x2j;
					}
				}
				k = k2.next();
			},
			(Some(k1), _, _) => {
				let mut x = x.rb_mut().get_mut(.., k..k1.next()).col_iter_mut();
				let (Some(mut x0), Some(mut x1)) = (x.next(), x.next()) else { panic!() };

				for j in N.indices().rev() {
					let mut u = iter::zip(u.row_idx_of_col(j).rev(), u.val_of_col(j).iter().rev());

					let (i, d) = (u.next().unwrap());
					debug_assert!(i == j);

					let x0j;
					let x1j;
					match diag_triu {
						DiagStatus::Unit => {
							x0j = copy(x0[j]);
							x1j = copy(x1[j]);
						},
						DiagStatus::Generic => {
							let d = conj_triu.apply_rt(&recip(*d));
							x0j = x0[j] * d;
							x1j = x1[j] * d;
							x0[j] = copy(x0j);
							x1[j] = copy(x1j);
						},
					}

					for (i, u) in u {
						let uij = conj_triu.apply_rt(u);
						x0[i] = x0[i] - uij * x0j;
						x1[i] = x1[i] - uij * x1j;
					}
				}
				k = k1.next();
			},
			(_, _, _) => {
				let mut x0 = x.rb_mut().get_mut(.., k0);

				for j in N.indices().rev() {
					let mut u = iter::zip(u.row_idx_of_col(j).rev(), u.val_of_col(j).iter().rev());

					let (i, d) = (u.next().unwrap());
					debug_assert!(i == j);

					let x0j;
					match diag_triu {
						DiagStatus::Unit => {
							x0j = copy(x0[j]);
						},
						DiagStatus::Generic => {
							let d = conj_triu.apply_rt(&recip(*d));
							x0j = x0[j] * d;
							x0[j] = copy(x0j);
						},
					}

					for (i, u) in u {
						let uij = conj_triu.apply_rt(u);
						x0[i] = x0[i] - uij * x0j;
					}
				}
				k = k0.next();
			},
		}
	}
}

/// Assuming `triu` is an upper triangular matrix, solves the equation `Op(triu).transpose() * X =
/// rhs`, and stores the result in `rhs`, where `Op` is either the conjugate or the identity
/// depending on the value of `conj_triu`.
///
/// # Note
/// The matrix indices need not be sorted, but
/// the diagonal element is assumed to be the first stored element in each column.
#[track_caller]
#[math]
pub fn solve_upper_triangular_transpose_in_place<I: Index, T: ComplexField>(
	triu: SparseColMatRef<'_, I, T>,
	conj_triu: Conj,
	diag_triu: DiagStatus,
	rhs: MatMut<'_, T>,
	par: Par,
) {
	let _ = par;
	assert!(all(triu.nrows() == triu.ncols(), rhs.nrows() == triu.nrows()));

	with_dim!(N, rhs.nrows());
	with_dim!(K, rhs.ncols());

	let mut x = rhs.as_shape_mut(N, K);
	let u = triu.as_shape(N, N);

	let mut k = IdxInc::ZERO;
	while let Some(k0) = K.try_check(*k) {
		let k1 = K.try_check(*k + 1);
		let k2 = K.try_check(*k + 2);
		let k3 = K.try_check(*k + 3);

		match (k1, k2, k3) {
			(Some(_), Some(_), Some(k3)) => {
				let mut x = x.rb_mut().get_mut(.., k..k3.next()).col_iter_mut();
				let (Some(mut x0), Some(mut x1), Some(mut x2), Some(mut x3)) = (x.next(), x.next(), x.next(), x.next()) else {
					panic!()
				};

				for j in N.indices() {
					let mut ui = u.row_idx_of_col(j);
					let mut uv = u.val_of_col(j).iter();
					let first = ui.next_back().zip(uv.next_back());

					let mut acc0a = zero::<T>();
					let mut acc1a = zero::<T>();
					let mut acc2a = zero::<T>();
					let mut acc3a = zero::<T>();

					for (i, uij) in iter::zip(ui, uv) {
						let uij = conj_triu.apply_rt(uij);
						acc0a = acc0a + uij * x0[i];
						acc1a = acc1a + uij * x1[i];
						acc2a = acc2a + uij * x2[i];
						acc3a = acc3a + uij * x3[i];
					}

					let mut x0j = x0[j] - acc0a;
					let mut x1j = x1[j] - acc1a;
					let mut x2j = x2[j] - acc2a;
					let mut x3j = x3[j] - acc3a;

					let (i, d) = first.unwrap();
					debug_assert!(i == j);
					match diag_triu {
						DiagStatus::Unit => {},
						DiagStatus::Generic => {
							let d = conj_triu.apply_rt(&recip(*d));
							x0j = x0j * d;
							x1j = x1j * d;
							x2j = x2j * d;
							x3j = x3j * d;
						},
					}

					x0[j] = x0j;
					x1[j] = x1j;
					x2[j] = x2j;
					x3[j] = x3j;
				}
				k = k3.next();
			},
			(Some(_), Some(k2), _) => {
				let mut x = x.rb_mut().get_mut(.., k..k2.next()).col_iter_mut();
				let (Some(mut x0), Some(mut x1), Some(mut x2)) = (x.next(), x.next(), x.next()) else {
					panic!()
				};

				for j in N.indices().rev() {
					let mut ui = u.row_idx_of_col(j);
					let mut uv = u.val_of_col(j).iter();
					let first = ui.next_back().zip(uv.next_back());

					let mut acc0a = zero::<T>();
					let mut acc1a = zero::<T>();
					let mut acc2a = zero::<T>();

					for (i, uij) in iter::zip(ui, uv) {
						let uij = conj_triu.apply_rt(uij);
						acc0a = acc0a + uij * x0[i];
						acc1a = acc1a + uij * x1[i];
						acc2a = acc2a + uij * x2[i];
					}

					let mut x0j = x0[j] - acc0a;
					let mut x1j = x1[j] - acc1a;
					let mut x2j = x2[j] - acc2a;

					let (i, d) = first.unwrap();
					debug_assert!(i == j);
					match diag_triu {
						DiagStatus::Unit => {},
						DiagStatus::Generic => {
							let d = conj_triu.apply_rt(&recip(*d));
							x0j = x0j * d;
							x1j = x1j * d;
							x2j = x2j * d;
						},
					}

					x0[j] = x0j;
					x1[j] = x1j;
					x2[j] = x2j;
				}

				k = k2.next();
			},
			(Some(k1), _, _) => {
				let mut x = x.rb_mut().get_mut(.., k..k1.next()).col_iter_mut();
				let (Some(mut x0), Some(mut x1)) = (x.next(), x.next()) else { panic!() };

				for j in N.indices().rev() {
					let mut ui = u.row_idx_of_col(j);
					let mut uv = u.val_of_col(j).iter();
					let first = ui.next_back().zip(uv.next_back());

					let mut acc0a = zero::<T>();
					let mut acc1a = zero::<T>();

					for (i, uij) in iter::zip(ui, uv) {
						let uij = conj_triu.apply_rt(uij);
						acc0a = acc0a + uij * x0[i];
						acc1a = acc1a + uij * x1[i];
					}

					let mut x0j = x0[j] - acc0a;
					let mut x1j = x1[j] - acc1a;

					let (i, d) = first.unwrap();
					debug_assert!(i == j);
					match diag_triu {
						DiagStatus::Unit => {},
						DiagStatus::Generic => {
							let d = conj_triu.apply_rt(&recip(*d));
							x0j = x0j * d;
							x1j = x1j * d;
						},
					}

					x0[j] = x0j;
					x1[j] = x1j;
				}

				k = k1.next();
			},
			(_, _, _) => {
				let mut x0 = x.rb_mut().get_mut(.., k0);

				for j in N.indices().rev() {
					let mut ui = u.row_idx_of_col(j);
					let mut uv = u.val_of_col(j).iter();
					let first = ui.next_back().zip(uv.next_back());

					let mut acc0a = zero::<T>();

					for (i, uij) in iter::zip(ui, uv) {
						let uij = conj_triu.apply_rt(uij);
						acc0a = acc0a + uij * x0[i];
					}

					let mut x0j = x0[j] - acc0a;

					let (i, d) = first.unwrap();
					debug_assert!(i == j);
					match diag_triu {
						DiagStatus::Unit => {},
						DiagStatus::Generic => {
							let d = conj_triu.apply_rt(&recip(*d));
							x0j = x0j * d;
						},
					}

					x0[j] = x0j;
				}

				k = k0.next();
			},
		}
	}
}
