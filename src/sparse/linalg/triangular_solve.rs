use crate::{
    assert,
    sparse::SparseColMatRef,
    utils::{bound, slice::*},
    Conj, Index, MatMut, Parallelism,
};
use core::iter;
use faer_entity::ComplexField;
use reborrow::*;

// FIXME: unsound get_unchecked(1..) calls

/// Assuming `self` is a lower triangular matrix, solves the equation `Op(self) * X = rhs`, and
/// stores the result in `rhs`, where `Op` is either the conjugate or the identity depending on the
/// value of `conj`.
///
/// # Note
/// The matrix indices need not be sorted, but
/// the diagonal element is assumed to be the first stored element in each column.
pub fn solve_lower_triangular_in_place<I: Index, E: ComplexField>(
    l: SparseColMatRef<'_, I, E>,
    conj: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    let _ = parallelism;
    assert!(l.nrows() == l.ncols());
    assert!(rhs.nrows() == l.nrows());

    let slice_group = SliceGroup::<'_, E>::new;

    with_dim!(N, rhs.nrows());
    with_dim!(K, rhs.ncols());
    let mut x = rhs.as_shape_mut(N, K);
    let l = l.as_shape(N, N);

    let mut k = bound::IdxInc::<usize>::zero();
    while let Some(k0) = K.try_check(*k) {
        let k1 = K.try_check(*k + 1);
        let k2 = K.try_check(*k + 2);
        let k3 = K.try_check(*k + 3);

        if let Some(k3) = k3 {
            let mut x = x.rb_mut().subcols_range_mut(k..k3.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1), Some(mut x2), Some(mut x3)) =
                (x.next(), x.next(), x.next(), x.next())
            else {
                panic!()
            };

            for j in N.indices() {
                let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                let d = if conj == Conj::Yes { d.faer_conj() } else { d };

                let xj0 = x0.read(j).faer_mul(d);
                x0.write(j, xj0);
                let xj1 = x1.read(j).faer_mul(d);
                x1.write(j, xj1);
                let xj2 = x2.read(j).faer_mul(d);
                x2.write(j, xj2);
                let xj3 = x3.read(j).faer_mul(d);
                x3.write(j, xj3);

                let row_ind = l.row_indices_of_col_raw(j);
                let len = row_ind.len();
                for (i, lij) in iter::zip(
                    &row_ind[1..],
                    slice_group(l.values_of_col(j))
                        .subslice(1..len)
                        .into_ref_iter(),
                ) {
                    let i = i.zx();
                    let lij = lij.read();
                    let lij = if conj == Conj::Yes {
                        lij.faer_conj()
                    } else {
                        lij
                    };
                    x0.write(i, x0.read(i).faer_sub(lij.faer_mul(xj0)));
                    x1.write(i, x1.read(i).faer_sub(lij.faer_mul(xj1)));
                    x2.write(i, x2.read(i).faer_sub(lij.faer_mul(xj2)));
                    x3.write(i, x3.read(i).faer_sub(lij.faer_mul(xj3)));
                }
            }
            k = k3.next();
        } else if let Some(k2) = k2 {
            let mut x = x.rb_mut().subcols_range_mut(k..k2.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1), Some(mut x2)) = (x.next(), x.next(), x.next()) else {
                panic!()
            };

            for j in N.indices() {
                let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                let d = if conj == Conj::Yes { d.faer_conj() } else { d };

                let xj0 = x0.read(j).faer_mul(d);
                x0.write(j, xj0);
                let xj1 = x1.read(j).faer_mul(d);
                x1.write(j, xj1);
                let xj2 = x2.read(j).faer_mul(d);
                x2.write(j, xj2);

                let row_ind = l.row_indices_of_col_raw(j);
                let len = row_ind.len();
                for (i, lij) in iter::zip(
                    &row_ind[1..],
                    slice_group(l.values_of_col(j))
                        .subslice(1..len)
                        .into_ref_iter(),
                ) {
                    let i = i.zx();
                    let lij = lij.read();
                    let lij = if conj == Conj::Yes {
                        lij.faer_conj()
                    } else {
                        lij
                    };
                    x0.write(i, x0.read(i).faer_sub(lij.faer_mul(xj0)));
                    x1.write(i, x1.read(i).faer_sub(lij.faer_mul(xj1)));
                    x2.write(i, x2.read(i).faer_sub(lij.faer_mul(xj2)));
                }
            }
            k = k2.next();
        } else if let Some(k1) = k1 {
            let mut x = x.rb_mut().subcols_range_mut(k..k1.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1)) = (x.next(), x.next()) else {
                panic!()
            };

            for j in N.indices() {
                let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                let d = if conj == Conj::Yes { d.faer_conj() } else { d };

                let xj0 = x0.read(j).faer_mul(d);
                x0.write(j, xj0);
                let xj1 = x1.read(j).faer_mul(d);
                x1.write(j, xj1);

                let row_ind = l.row_indices_of_col_raw(j);
                let len = row_ind.len();
                for (i, lij) in iter::zip(
                    &row_ind[1..],
                    slice_group(l.values_of_col(j))
                        .subslice(1..len)
                        .into_ref_iter(),
                ) {
                    let i = i.zx();
                    let lij = lij.read();
                    let lij = if conj == Conj::Yes {
                        lij.faer_conj()
                    } else {
                        lij
                    };
                    x0.write(i, x0.read(i).faer_sub(lij.faer_mul(xj0)));
                    x1.write(i, x1.read(i).faer_sub(lij.faer_mul(xj1)));
                }
            }
            k = k1.next();
        } else {
            let mut x0 = x.rb_mut().col_mut(k0);

            for j in N.indices() {
                let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                let d = if conj == Conj::Yes { d.faer_conj() } else { d };

                let xj0 = x0.read(j).faer_mul(d);
                x0.write(j, xj0);

                let row_ind = l.row_indices_of_col_raw(j);
                let len = row_ind.len();
                for (i, lij) in iter::zip(
                    &row_ind[1..],
                    slice_group(l.values_of_col(j))
                        .subslice(1..len)
                        .into_ref_iter(),
                ) {
                    let i = i.zx();
                    let lij = lij.read();
                    let lij = if conj == Conj::Yes {
                        lij.faer_conj()
                    } else {
                        lij
                    };
                    x0.write(i, x0.read(i).faer_sub(lij.faer_mul(xj0)));
                }
            }
            k = k0.next();
        }
    }
}

/// Assuming `self` is a lower triangular matrix, solves the equation `Op(self).transpose() * X =
/// rhs`, and stores the result in `rhs`, where `Op` is either the conjugate or the identity
/// depending on the value of `conj`.
///
/// # Note
/// The matrix indices need not be sorted, but
/// the diagonal element is assumed to be the first stored element in each column.
pub fn solve_lower_triangular_transpose_in_place<I: Index, E: ComplexField>(
    l: SparseColMatRef<'_, I, E>,
    conj: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    let _ = parallelism;
    assert!(l.nrows() == l.ncols());
    assert!(rhs.nrows() == l.nrows());

    let slice_group = SliceGroup::<'_, E>::new;

    with_dim!(N, rhs.nrows());
    with_dim!(K, rhs.ncols());
    let mut x = rhs.as_shape_mut(N, K);
    let l = l.as_shape(N, N);

    let mut k = bound::IdxInc::<usize>::zero();
    while let Some(k0) = K.try_check(*k) {
        let k1 = K.try_check(*k + 1);
        let k2 = K.try_check(*k + 2);
        let k3 = K.try_check(*k + 3);

        if let Some(k3) = k3 {
            let mut x = x.rb_mut().subcols_range_mut(k..k3.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1), Some(mut x2), Some(mut x3)) =
                (x.next(), x.next(), x.next(), x.next())
            else {
                panic!()
            };

            for j in N.indices().rev() {
                let mut acc0a = E::faer_zero();
                let mut acc1a = E::faer_zero();
                let mut acc2a = E::faer_zero();
                let mut acc3a = E::faer_zero();

                let row_ind = l.row_indices_of_col_raw(j);
                let len = row_ind.len();
                assert!(len >= 1);
                for (i, lij) in iter::zip(
                    &row_ind[1..],
                    slice_group(l.values_of_col(j))
                        .subslice(1..len)
                        .into_ref_iter(),
                ) {
                    let i = i.zx();
                    let lij = lij.read();
                    let lij = if conj == Conj::Yes {
                        lij.faer_conj()
                    } else {
                        lij
                    };
                    acc0a = acc0a.faer_add(lij.faer_mul(x0.read(i)));
                    acc1a = acc1a.faer_add(lij.faer_mul(x1.read(i)));
                    acc2a = acc2a.faer_add(lij.faer_mul(x2.read(i)));
                    acc3a = acc3a.faer_add(lij.faer_mul(x3.read(i)));
                }

                let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                let d = if conj == Conj::Yes { d.faer_conj() } else { d };
                x0.write(j, x0.read(j).faer_sub(acc0a).faer_mul(d));
                x1.write(j, x1.read(j).faer_sub(acc1a).faer_mul(d));
                x2.write(j, x2.read(j).faer_sub(acc2a).faer_mul(d));
                x3.write(j, x3.read(j).faer_sub(acc3a).faer_mul(d));
            }
            k = k3.next();
        } else if let Some(k2) = k2 {
            let mut x = x.rb_mut().subcols_range_mut(k..k2.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1), Some(mut x2)) = (x.next(), x.next(), x.next()) else {
                panic!()
            };

            for j in N.indices().rev() {
                let mut acc0a = E::faer_zero();
                let mut acc1a = E::faer_zero();
                let mut acc2a = E::faer_zero();

                let row_ind = l.row_indices_of_col_raw(j);
                let len = row_ind.len();
                assert!(len >= 1);
                for (i, lij) in iter::zip(
                    &row_ind[1..],
                    slice_group(l.values_of_col(j))
                        .subslice(1..len)
                        .into_ref_iter(),
                ) {
                    let i = i.zx();
                    let lij = lij.read();
                    let lij = if conj == Conj::Yes {
                        lij.faer_conj()
                    } else {
                        lij
                    };
                    acc0a = acc0a.faer_add(lij.faer_mul(x0.read(i)));
                    acc1a = acc1a.faer_add(lij.faer_mul(x1.read(i)));
                    acc2a = acc2a.faer_add(lij.faer_mul(x2.read(i)));
                }

                let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                let d = if conj == Conj::Yes { d.faer_conj() } else { d };
                x0.write(j, x0.read(j).faer_sub(acc0a).faer_mul(d));
                x1.write(j, x1.read(j).faer_sub(acc1a).faer_mul(d));
                x2.write(j, x2.read(j).faer_sub(acc2a).faer_mul(d));
            }
            k = k2.next();
        } else if let Some(k1) = k1 {
            let mut x = x.rb_mut().subcols_range_mut(k..k1.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1)) = (x.next(), x.next()) else {
                panic!()
            };

            for j in N.indices().rev() {
                let mut acc0a = E::faer_zero();
                let mut acc1a = E::faer_zero();

                let row_ind = l.row_indices_of_col_raw(j);
                let len = row_ind.len();
                assert!(len >= 1);
                for (i, lij) in iter::zip(
                    &row_ind[1..],
                    slice_group(l.values_of_col(j))
                        .subslice(1..len)
                        .into_ref_iter(),
                ) {
                    let i = i.zx();
                    let lij = lij.read();
                    let lij = if conj == Conj::Yes {
                        lij.faer_conj()
                    } else {
                        lij
                    };
                    acc0a = acc0a.faer_add(lij.faer_mul(x0.read(i)));
                    acc1a = acc1a.faer_add(lij.faer_mul(x1.read(i)));
                }

                let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                let d = if conj == Conj::Yes { d.faer_conj() } else { d };
                x0.write(j, x0.read(j).faer_sub(acc0a).faer_mul(d));
                x1.write(j, x1.read(j).faer_sub(acc1a).faer_mul(d));
            }
            k = k1.next();
        } else {
            let mut x0 = x.rb_mut().col_mut(k0);

            for j in N.indices().rev() {
                let mut acc0a = E::faer_zero();

                let row_ind = l.row_indices_of_col_raw(j);
                let len = row_ind.len();
                assert!(len >= 1);
                for (i, lij) in iter::zip(
                    &row_ind[1..],
                    slice_group(l.values_of_col(j))
                        .subslice(1..len)
                        .into_ref_iter(),
                ) {
                    let i = i.zx();
                    let lij = lij.read();
                    let lij = if conj == Conj::Yes {
                        lij.faer_conj()
                    } else {
                        lij
                    };
                    acc0a = acc0a.faer_add(lij.faer_mul(x0.read(i)));
                }

                let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                let d = if conj == Conj::Yes { d.faer_conj() } else { d };
                x0.write(j, x0.read(j).faer_sub(acc0a).faer_mul(d));
            }
            k = k0.next();
        }
    }
}

/// Assuming `self` is a unit lower triangular matrix, solves the equation `Op(self) * X = rhs`, and
/// stores the result in `rhs`, where `Op` is either the conjugate or the identity depending on the
/// value of `conj`.
///
/// # Note
/// The matrix indices need not be sorted, but
/// the diagonal element is assumed to be the first stored element in each column.
pub fn solve_unit_lower_triangular_in_place<I: Index, E: ComplexField>(
    l: SparseColMatRef<'_, I, E>,
    conj: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    let _ = parallelism;
    assert!(l.nrows() == l.ncols());
    assert!(rhs.nrows() == l.nrows());

    let slice_group = SliceGroup::<'_, E>::new;

    with_dim!(N, rhs.nrows());
    with_dim!(K, rhs.ncols());

    let mut x = rhs.as_shape_mut(N, K);
    let l = l.as_shape(N, N);

    let mut k = bound::IdxInc::<usize>::zero();
    while let Some(k0) = K.try_check(*k) {
        let k1 = K.try_check(*k + 1);
        let k2 = K.try_check(*k + 2);
        let k3 = K.try_check(*k + 3);

        if let Some(k3) = k3 {
            let mut x = x.rb_mut().subcols_range_mut(k..k3.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1), Some(mut x2), Some(mut x3)) =
                (x.next(), x.next(), x.next(), x.next())
            else {
                panic!()
            };

            for j in N.indices() {
                let xj0 = x0.read(j);
                let xj1 = x1.read(j);
                let xj2 = x2.read(j);
                let xj3 = x3.read(j);

                let row_ind = l.row_indices_of_col_raw(j);
                let len = row_ind.len();
                assert!(len >= 1);
                for (i, lij) in unsafe {
                    iter::zip(
                        row_ind.get_unchecked(1..),
                        slice_group(l.values_of_col(j))
                            .subslice_unchecked(1..len)
                            .into_ref_iter(),
                    )
                } {
                    let i = i.zx();
                    let lij = lij.read();
                    let lij = if conj == Conj::Yes {
                        lij.faer_conj()
                    } else {
                        lij
                    };
                    x0.write(i, x0.read(i).faer_sub(lij.faer_mul(xj0)));
                    x1.write(i, x1.read(i).faer_sub(lij.faer_mul(xj1)));
                    x2.write(i, x2.read(i).faer_sub(lij.faer_mul(xj2)));
                    x3.write(i, x3.read(i).faer_sub(lij.faer_mul(xj3)));
                }
            }
            k = k3.next();
        } else if let Some(k2) = k2 {
            let mut x = x.rb_mut().subcols_range_mut(k..k2.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1), Some(mut x2)) = (x.next(), x.next(), x.next()) else {
                panic!()
            };

            for j in N.indices() {
                let xj0 = x0.read(j);
                let xj1 = x1.read(j);
                let xj2 = x2.read(j);

                let row_ind = l.row_indices_of_col_raw(j);
                let len = row_ind.len();
                assert!(len >= 1);
                for (i, lij) in unsafe {
                    iter::zip(
                        row_ind.get_unchecked(1..),
                        slice_group(l.values_of_col(j))
                            .subslice_unchecked(1..len)
                            .into_ref_iter(),
                    )
                } {
                    let i = i.zx();
                    let lij = lij.read();
                    let lij = if conj == Conj::Yes {
                        lij.faer_conj()
                    } else {
                        lij
                    };
                    x0.write(i, x0.read(i).faer_sub(lij.faer_mul(xj0)));
                    x1.write(i, x1.read(i).faer_sub(lij.faer_mul(xj1)));
                    x2.write(i, x2.read(i).faer_sub(lij.faer_mul(xj2)));
                }
            }
            k = k2.next();
        } else if let Some(k1) = k1 {
            let mut x = x.rb_mut().subcols_range_mut(k..k1.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1)) = (x.next(), x.next()) else {
                panic!()
            };

            for j in N.indices() {
                let xj0 = x0.read(j);
                let xj1 = x1.read(j);

                let row_ind = l.row_indices_of_col_raw(j);
                let len = row_ind.len();
                assert!(len >= 1);
                for (i, lij) in unsafe {
                    iter::zip(
                        row_ind.get_unchecked(1..),
                        slice_group(l.values_of_col(j))
                            .subslice_unchecked(1..len)
                            .into_ref_iter(),
                    )
                } {
                    let i = i.zx();
                    let lij = lij.read();
                    let lij = if conj == Conj::Yes {
                        lij.faer_conj()
                    } else {
                        lij
                    };
                    x0.write(i, x0.read(i).faer_sub(lij.faer_mul(xj0)));
                    x1.write(i, x1.read(i).faer_sub(lij.faer_mul(xj1)));
                }
            }
            k = k1.next();
        } else {
            let mut x0 = x.rb_mut().col_mut(k0);

            for j in N.indices() {
                let xj0 = x0.read(j);

                let row_ind = l.row_indices_of_col_raw(j);
                let len = row_ind.len();
                assert!(len >= 1);
                for (i, lij) in unsafe {
                    iter::zip(
                        row_ind.get_unchecked(1..),
                        slice_group(l.values_of_col(j))
                            .subslice_unchecked(1..len)
                            .into_ref_iter(),
                    )
                } {
                    let i = i.zx();
                    let lij = lij.read();
                    let lij = if conj == Conj::Yes {
                        lij.faer_conj()
                    } else {
                        lij
                    };
                    x0.write(i, x0.read(i).faer_sub(lij.faer_mul(xj0)));
                }
            }
            k = k0.next();
        }
    }
}

/// Assuming `self` is a unit lower triangular matrix, solves the equation `Op(self).transpose() * X
/// = rhs`, and stores the result in `rhs`, where `Op` is either the conjugate or the identity
/// depending on the value of `conj`.
///
/// # Note
/// The matrix indices need not be sorted, but
/// the diagonal element is assumed to be the first stored element in each column.
pub fn solve_unit_lower_triangular_transpose_in_place<I: Index, E: ComplexField>(
    l: SparseColMatRef<'_, I, E>,
    conj: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    let _ = parallelism;
    assert!(l.nrows() == l.ncols());
    assert!(rhs.nrows() == l.nrows());

    let slice_group = SliceGroup::<'_, E>::new;

    with_dim!(N, rhs.nrows());
    with_dim!(K, rhs.ncols());

    let mut x = rhs.as_shape_mut(N, K);
    let l = l.as_shape(N, N);

    let mut k = bound::IdxInc::<usize>::zero();
    while let Some(k0) = K.try_check(*k) {
        let k1 = K.try_check(*k + 1);
        let k2 = K.try_check(*k + 2);
        let k3 = K.try_check(*k + 3);

        if let Some(k3) = k3 {
            let mut x = x.rb_mut().subcols_range_mut(k..k3.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1), Some(mut x2), Some(mut x3)) =
                (x.next(), x.next(), x.next(), x.next())
            else {
                panic!()
            };

            for j in N.indices().rev() {
                let mut acc0a = E::faer_zero();
                let mut acc1a = E::faer_zero();
                let mut acc2a = E::faer_zero();
                let mut acc3a = E::faer_zero();

                let row_ind = l.row_indices_of_col_raw(j);
                let len = row_ind.len();
                assert!(len >= 1);
                for (i, lij) in unsafe {
                    iter::zip(
                        row_ind.get_unchecked(1..),
                        slice_group(l.values_of_col(j))
                            .subslice_unchecked(1..len)
                            .into_ref_iter(),
                    )
                } {
                    let i = i.zx();
                    let lij = lij.read();
                    let lij = if conj == Conj::Yes {
                        lij.faer_conj()
                    } else {
                        lij
                    };
                    acc0a = acc0a.faer_add(lij.faer_mul(x0.read(i)));
                    acc1a = acc1a.faer_add(lij.faer_mul(x1.read(i)));
                    acc2a = acc2a.faer_add(lij.faer_mul(x2.read(i)));
                    acc3a = acc3a.faer_add(lij.faer_mul(x3.read(i)));
                }

                x0.write(j, x0.read(j).faer_sub(acc0a));
                x1.write(j, x1.read(j).faer_sub(acc1a));
                x2.write(j, x2.read(j).faer_sub(acc2a));
                x3.write(j, x3.read(j).faer_sub(acc3a));
            }
            k = k3.next();
        } else if let Some(k2) = k2 {
            let mut x = x.rb_mut().subcols_range_mut(k..k2.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1), Some(mut x2)) = (x.next(), x.next(), x.next()) else {
                panic!()
            };

            for j in N.indices().rev() {
                let mut acc0a = E::faer_zero();
                let mut acc1a = E::faer_zero();
                let mut acc2a = E::faer_zero();

                let row_ind = l.row_indices_of_col_raw(j);
                let len = row_ind.len();
                assert!(len >= 1);
                for (i, lij) in unsafe {
                    iter::zip(
                        row_ind.get_unchecked(1..),
                        slice_group(l.values_of_col(j))
                            .subslice_unchecked(1..len)
                            .into_ref_iter(),
                    )
                } {
                    let i = i.zx();
                    let lij = lij.read();
                    let lij = if conj == Conj::Yes {
                        lij.faer_conj()
                    } else {
                        lij
                    };
                    acc0a = acc0a.faer_add(lij.faer_mul(x0.read(i)));
                    acc1a = acc1a.faer_add(lij.faer_mul(x1.read(i)));
                    acc2a = acc2a.faer_add(lij.faer_mul(x2.read(i)));
                }

                x0.write(j, x0.read(j).faer_sub(acc0a));
                x1.write(j, x1.read(j).faer_sub(acc1a));
                x2.write(j, x2.read(j).faer_sub(acc2a));
            }
            k = k2.next();
        } else if let Some(k1) = k1 {
            let mut x = x.rb_mut().subcols_range_mut(k..k1.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1)) = (x.next(), x.next()) else {
                panic!()
            };

            for j in N.indices().rev() {
                let mut acc0a = E::faer_zero();
                let mut acc1a = E::faer_zero();

                let row_ind = l.row_indices_of_col_raw(j);
                let len = row_ind.len();
                assert!(len >= 1);
                for (i, lij) in unsafe {
                    iter::zip(
                        row_ind.get_unchecked(1..),
                        slice_group(l.values_of_col(j))
                            .subslice_unchecked(1..len)
                            .into_ref_iter(),
                    )
                } {
                    let i = i.zx();
                    let lij = lij.read();
                    let lij = if conj == Conj::Yes {
                        lij.faer_conj()
                    } else {
                        lij
                    };
                    acc0a = acc0a.faer_add(lij.faer_mul(x0.read(i)));
                    acc1a = acc1a.faer_add(lij.faer_mul(x1.read(i)));
                }

                x0.write(j, x0.read(j).faer_sub(acc0a));
                x1.write(j, x1.read(j).faer_sub(acc1a));
            }
            k = k1.next();
        } else {
            let mut x0 = x.rb_mut().col_mut(k0);

            for j in N.indices().rev() {
                let mut acc0a = E::faer_zero();

                let row_ind = l.row_indices_of_col_raw(j);
                let len = row_ind.len();
                assert!(len >= 1);
                for (i, lij) in unsafe {
                    iter::zip(
                        row_ind.get_unchecked(1..),
                        slice_group(l.values_of_col(j))
                            .subslice_unchecked(1..len)
                            .into_ref_iter(),
                    )
                } {
                    let i = i.zx();
                    let lij = lij.read();
                    let lij = if conj == Conj::Yes {
                        lij.faer_conj()
                    } else {
                        lij
                    };
                    acc0a = acc0a.faer_add(lij.faer_mul(x0.read(i)));
                }

                x0.write(j, x0.read(j).faer_sub(acc0a));
            }
            k = k0.next();
        }
    }
}

#[doc(hidden)]
pub fn ldlt_scale_solve_unit_lower_triangular_transpose_in_place<I: Index, E: ComplexField>(
    l: SparseColMatRef<'_, I, E>,
    conj: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    let _ = parallelism;
    assert!(l.nrows() == l.ncols());
    assert!(rhs.nrows() == l.nrows());

    let slice_group = SliceGroup::<'_, E>::new;

    with_dim!(N, rhs.nrows());
    with_dim!(K, rhs.ncols());

    let mut x = rhs.as_shape_mut(N, K);
    let l = l.as_shape(N, N);

    let mut k = bound::IdxInc::<usize>::zero();
    while let Some(k0) = K.try_check(*k) {
        let k1 = K.try_check(*k + 1);
        let k2 = K.try_check(*k + 2);
        let k3 = K.try_check(*k + 3);

        if let Some(k3) = k3 {
            let mut x = x.rb_mut().subcols_range_mut(k..k3.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1), Some(mut x2), Some(mut x3)) =
                (x.next(), x.next(), x.next(), x.next())
            else {
                panic!()
            };

            for j in N.indices().rev() {
                let d = slice_group(l.values_of_col(j))
                    .read(0)
                    .faer_real()
                    .faer_inv();

                let mut acc0a = E::faer_zero();
                let mut acc1a = E::faer_zero();
                let mut acc2a = E::faer_zero();
                let mut acc3a = E::faer_zero();

                let row_ind = l.row_indices_of_col_raw(j);
                let len = row_ind.len();
                assert!(len >= 1);
                for (i, lij) in unsafe {
                    iter::zip(
                        row_ind.get_unchecked(1..),
                        slice_group(l.values_of_col(j))
                            .subslice_unchecked(1..len)
                            .into_ref_iter(),
                    )
                } {
                    let i = i.zx();
                    let lij = lij.read();
                    let lij = if conj == Conj::Yes {
                        lij.faer_conj()
                    } else {
                        lij
                    };
                    acc0a = acc0a.faer_add(lij.faer_mul(x0.read(i)));
                    acc1a = acc1a.faer_add(lij.faer_mul(x1.read(i)));
                    acc2a = acc2a.faer_add(lij.faer_mul(x2.read(i)));
                    acc3a = acc3a.faer_add(lij.faer_mul(x3.read(i)));
                }

                x0.write(j, x0.read(j).faer_scale_real(d).faer_sub(acc0a));
                x1.write(j, x1.read(j).faer_scale_real(d).faer_sub(acc1a));
                x2.write(j, x2.read(j).faer_scale_real(d).faer_sub(acc2a));
                x3.write(j, x3.read(j).faer_scale_real(d).faer_sub(acc3a));
            }
            k = k3.next();
        } else if let Some(k2) = k2 {
            let mut x = x.rb_mut().subcols_range_mut(k..k2.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1), Some(mut x2)) = (x.next(), x.next(), x.next()) else {
                panic!()
            };

            for j in N.indices().rev() {
                let d = slice_group(l.values_of_col(j))
                    .read(0)
                    .faer_real()
                    .faer_inv();

                let mut acc0a = E::faer_zero();
                let mut acc1a = E::faer_zero();
                let mut acc2a = E::faer_zero();

                let row_ind = l.row_indices_of_col_raw(j);
                let len = row_ind.len();
                assert!(len >= 1);
                for (i, lij) in unsafe {
                    iter::zip(
                        row_ind.get_unchecked(1..),
                        slice_group(l.values_of_col(j))
                            .subslice_unchecked(1..len)
                            .into_ref_iter(),
                    )
                } {
                    let i = i.zx();
                    let lij = lij.read();
                    let lij = if conj == Conj::Yes {
                        lij.faer_conj()
                    } else {
                        lij
                    };
                    acc0a = acc0a.faer_add(lij.faer_mul(x0.read(i)));
                    acc1a = acc1a.faer_add(lij.faer_mul(x1.read(i)));
                    acc2a = acc2a.faer_add(lij.faer_mul(x2.read(i)));
                }

                x0.write(j, x0.read(j).faer_scale_real(d).faer_sub(acc0a));
                x1.write(j, x1.read(j).faer_scale_real(d).faer_sub(acc1a));
                x2.write(j, x2.read(j).faer_scale_real(d).faer_sub(acc2a));
            }
            k = k2.next();
        } else if let Some(k1) = k1 {
            let mut x = x.rb_mut().subcols_range_mut(k..k1.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1)) = (x.next(), x.next()) else {
                panic!()
            };

            for j in N.indices().rev() {
                let d = slice_group(l.values_of_col(j))
                    .read(0)
                    .faer_real()
                    .faer_inv();

                let mut acc0a = E::faer_zero();
                let mut acc1a = E::faer_zero();

                let row_ind = l.row_indices_of_col_raw(j);
                let len = row_ind.len();
                assert!(len >= 1);
                for (i, lij) in unsafe {
                    iter::zip(
                        row_ind.get_unchecked(1..),
                        slice_group(l.values_of_col(j))
                            .subslice_unchecked(1..len)
                            .into_ref_iter(),
                    )
                } {
                    let i = i.zx();
                    let lij = lij.read();
                    let lij = if conj == Conj::Yes {
                        lij.faer_conj()
                    } else {
                        lij
                    };
                    acc0a = acc0a.faer_add(lij.faer_mul(x0.read(i)));
                    acc1a = acc1a.faer_add(lij.faer_mul(x1.read(i)));
                }

                x0.write(j, x0.read(j).faer_scale_real(d).faer_sub(acc0a));
                x1.write(j, x1.read(j).faer_scale_real(d).faer_sub(acc1a));
            }
            k = k1.next();
        } else {
            let mut x0 = x.rb_mut().col_mut(k0);

            for j in N.indices().rev() {
                let d = slice_group(l.values_of_col(j))
                    .read(0)
                    .faer_real()
                    .faer_inv();

                let mut acc0a = E::faer_zero();

                let row_ind = l.row_indices_of_col_raw(j);
                let len = row_ind.len();
                assert!(len >= 1);
                for (i, lij) in unsafe {
                    iter::zip(
                        row_ind.get_unchecked(1..),
                        slice_group(l.values_of_col(j))
                            .subslice_unchecked(1..len)
                            .into_ref_iter(),
                    )
                } {
                    let i = i.zx();
                    let lij = lij.read();
                    let lij = if conj == Conj::Yes {
                        lij.faer_conj()
                    } else {
                        lij
                    };
                    acc0a = acc0a.faer_add(lij.faer_mul(x0.read(i)));
                }

                x0.write(j, x0.read(j).faer_scale_real(d).faer_sub(acc0a));
            }
            k = k0.next();
        }
    }
}

/// Assuming `self` is an upper triangular matrix, solves the equation `Op(self) * X = rhs`, and
/// stores the result in `rhs`, where `Op` is either the conjugate or the identity
/// depending on the value of `conj`.
///
/// # Note
/// The matrix indices need not be sorted, but
/// the diagonal element is assumed to be the last stored element in each column.
pub fn solve_upper_triangular_in_place<I: Index, E: ComplexField>(
    u: SparseColMatRef<'_, I, E>,
    conj: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    let _ = parallelism;
    assert!(u.nrows() == u.ncols());
    assert!(rhs.nrows() == u.nrows());

    with_dim!(N, rhs.nrows());
    with_dim!(K, rhs.ncols());

    let mut x = rhs.as_shape_mut(N, K);
    let u = u.as_shape(N, N);

    let mut k = bound::IdxInc::<usize>::zero();
    while let Some(k0) = K.try_check(*k) {
        let k1 = K.try_check(*k + 1);
        let k2 = K.try_check(*k + 2);
        let k3 = K.try_check(*k + 3);

        if let Some(k3) = k3 {
            let mut x = x.rb_mut().subcols_range_mut(k..k3.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1), Some(mut x2), Some(mut x3)) =
                (x.next(), x.next(), x.next(), x.next())
            else {
                panic!()
            };

            for j in N.indices().rev() {
                let ui = u.row_indices_of_col_raw(j);
                let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                let u_inv = ux.read(ui.len() - 1).faer_inv();
                let u_inv = if conj == Conj::Yes {
                    u_inv.faer_conj()
                } else {
                    u_inv
                };
                let xj0 = x0.read(j).faer_mul(u_inv);
                let xj1 = x1.read(j).faer_mul(u_inv);
                let xj2 = x2.read(j).faer_mul(u_inv);
                let xj3 = x3.read(j).faer_mul(u_inv);
                x0.write(j, xj0);
                x1.write(j, xj1);
                x2.write(j, xj2);
                x3.write(j, xj3);

                for (i, u) in iter::zip(
                    &ui[..ui.len() - 1],
                    ux.subslice(0..ui.len() - 1).into_ref_iter(),
                ) {
                    let i = i.zx();
                    let u = if conj == Conj::Yes {
                        u.read().faer_conj()
                    } else {
                        u.read()
                    };

                    x0.write(i, x0.read(i).faer_sub(E::faer_mul(u, xj0)));
                    x1.write(i, x1.read(i).faer_sub(E::faer_mul(u, xj1)));
                    x2.write(i, x2.read(i).faer_sub(E::faer_mul(u, xj2)));
                    x3.write(i, x3.read(i).faer_sub(E::faer_mul(u, xj3)));
                }
            }
            k = k3.next();
        } else if let Some(k2) = k2 {
            let mut x = x.rb_mut().subcols_range_mut(k..k2.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1), Some(mut x2)) = (x.next(), x.next(), x.next()) else {
                panic!()
            };

            for j in N.indices().rev() {
                let ui = u.row_indices_of_col_raw(j);
                let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                let u_inv = ux.read(ui.len() - 1).faer_inv();
                let u_inv = if conj == Conj::Yes {
                    u_inv.faer_conj()
                } else {
                    u_inv
                };
                let xj0 = x0.read(j).faer_mul(u_inv);
                let xj1 = x1.read(j).faer_mul(u_inv);
                let xj2 = x2.read(j).faer_mul(u_inv);
                x0.write(j, xj0);
                x1.write(j, xj1);
                x2.write(j, xj2);

                for (i, u) in iter::zip(
                    &ui[..ui.len() - 1],
                    ux.subslice(0..ui.len() - 1).into_ref_iter(),
                ) {
                    let i = i.zx();
                    let u = if conj == Conj::Yes {
                        u.read().faer_conj()
                    } else {
                        u.read()
                    };

                    x0.write(i, x0.read(i).faer_sub(E::faer_mul(u, xj0)));
                    x1.write(i, x1.read(i).faer_sub(E::faer_mul(u, xj1)));
                    x2.write(i, x2.read(i).faer_sub(E::faer_mul(u, xj2)));
                }
            }
            k = k2.next();
        } else if let Some(k1) = k1 {
            let mut x = x.rb_mut().subcols_range_mut(k..k1.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1)) = (x.next(), x.next()) else {
                panic!()
            };

            for j in N.indices().rev() {
                let ui = u.row_indices_of_col_raw(j);
                let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                let u_inv = ux.read(ui.len() - 1).faer_inv();
                let u_inv = if conj == Conj::Yes {
                    u_inv.faer_conj()
                } else {
                    u_inv
                };
                let xj0 = x0.read(j).faer_mul(u_inv);
                let xj1 = x1.read(j).faer_mul(u_inv);
                x0.write(j, xj0);
                x1.write(j, xj1);

                for (i, u) in iter::zip(
                    &ui[..ui.len() - 1],
                    ux.subslice(0..ui.len() - 1).into_ref_iter(),
                ) {
                    let i = i.zx();
                    let u = if conj == Conj::Yes {
                        u.read().faer_conj()
                    } else {
                        u.read()
                    };

                    x0.write(i, x0.read(i).faer_sub(E::faer_mul(u, xj0)));
                    x1.write(i, x1.read(i).faer_sub(E::faer_mul(u, xj1)));
                }
            }
            k = k1.next();
        } else {
            let mut x0 = x.rb_mut().col_mut(k0);

            for j in N.indices().rev() {
                let ui = u.row_indices_of_col_raw(j);
                let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                let u_inv = ux.read(ui.len() - 1).faer_inv();
                let u_inv = if conj == Conj::Yes {
                    u_inv.faer_conj()
                } else {
                    u_inv
                };
                let xj = x0.read(j).faer_mul(u_inv);
                x0.write(j, xj);

                for (i, u) in iter::zip(
                    &ui[..ui.len() - 1],
                    ux.subslice(0..ui.len() - 1).into_ref_iter(),
                ) {
                    let i = i.zx();
                    let u = if conj == Conj::Yes {
                        u.read().faer_conj()
                    } else {
                        u.read()
                    };

                    x0.write(i, x0.read(i).faer_sub(E::faer_mul(u, xj)));
                }
            }
            k = k0.next();
        }
    }
}

/// Assuming `self` is an upper triangular matrix, solves the equation `Op(self).transpose() * X =
/// rhs`, and stores the result in `rhs`, where `Op` is either the conjugate or the identity
/// depending on the value of `conj`.
///
/// # Note
/// The matrix indices need not be sorted, but
/// the diagonal element is assumed to be the last stored element in each column.
pub fn solve_upper_triangular_transpose_in_place<I: Index, E: ComplexField>(
    u: SparseColMatRef<'_, I, E>,
    conj: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    let _ = parallelism;
    assert!(u.nrows() == u.ncols());
    assert!(rhs.nrows() == u.nrows());

    with_dim!(N, rhs.nrows());
    with_dim!(K, rhs.ncols());

    let mut x = rhs.as_shape_mut(N, K);
    let u = u.as_shape(N, N);

    let mut k = bound::IdxInc::<usize>::zero();
    while let Some(k0) = K.try_check(*k) {
        let k1 = K.try_check(*k + 1);
        let k2 = K.try_check(*k + 2);
        let k3 = K.try_check(*k + 3);

        if let Some(k3) = k3 {
            let mut x = x.rb_mut().subcols_range_mut(k..k3.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1), Some(mut x2), Some(mut x3)) =
                (x.next(), x.next(), x.next(), x.next())
            else {
                panic!()
            };

            for j in N.indices() {
                let ui = u.row_indices_of_col_raw(j);
                let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                let u_inv = ux.read(ui.len() - 1).faer_inv();
                let u_inv = if conj == Conj::Yes {
                    u_inv.faer_conj()
                } else {
                    u_inv
                };

                let mut acc0a = E::faer_zero();
                let mut acc1a = E::faer_zero();
                let mut acc2a = E::faer_zero();
                let mut acc3a = E::faer_zero();

                let rows = &ui[..ui.len() - 1];
                let values = ux.subslice(0..ui.len() - 1);

                for (i, uij) in iter::zip(rows, values.into_ref_iter()) {
                    let uija = uij.read();
                    let uija = if conj == Conj::Yes {
                        uija.faer_conj()
                    } else {
                        uija
                    };
                    acc0a = acc0a.faer_add(uija.faer_mul(x0.read(i.zx())));
                    acc1a = acc1a.faer_add(uija.faer_mul(x1.read(i.zx())));
                    acc2a = acc2a.faer_add(uija.faer_mul(x2.read(i.zx())));
                    acc3a = acc3a.faer_add(uija.faer_mul(x3.read(i.zx())));
                }

                x0.write(j, x0.read(j).faer_sub(acc0a).faer_mul(u_inv));
                x1.write(j, x1.read(j).faer_sub(acc1a).faer_mul(u_inv));
                x2.write(j, x2.read(j).faer_sub(acc2a).faer_mul(u_inv));
                x3.write(j, x3.read(j).faer_sub(acc3a).faer_mul(u_inv));
            }
            k = k3.next();
        } else if let Some(k2) = k2 {
            let mut x = x.rb_mut().subcols_range_mut(k..k2.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1), Some(mut x2)) = (x.next(), x.next(), x.next()) else {
                panic!()
            };

            for j in N.indices() {
                let ui = u.row_indices_of_col_raw(j);
                let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                let u_inv = ux.read(ui.len() - 1).faer_inv();
                let u_inv = if conj == Conj::Yes {
                    u_inv.faer_conj()
                } else {
                    u_inv
                };

                let mut acc0a = E::faer_zero();
                let mut acc1a = E::faer_zero();
                let mut acc2a = E::faer_zero();

                let rows = &ui[..ui.len() - 1];
                let values = ux.subslice(0..ui.len() - 1);

                for (i, uij) in iter::zip(rows, values.into_ref_iter()) {
                    let uija = uij.read();
                    let uija = if conj == Conj::Yes {
                        uija.faer_conj()
                    } else {
                        uija
                    };
                    acc0a = acc0a.faer_add(uija.faer_mul(x0.read(i.zx())));
                    acc1a = acc1a.faer_add(uija.faer_mul(x1.read(i.zx())));
                    acc2a = acc2a.faer_add(uija.faer_mul(x2.read(i.zx())));
                }

                x0.write(j, x0.read(j).faer_sub(acc0a).faer_mul(u_inv));
                x1.write(j, x1.read(j).faer_sub(acc1a).faer_mul(u_inv));
                x2.write(j, x2.read(j).faer_sub(acc2a).faer_mul(u_inv));
            }
            k = k2.next();
        } else if let Some(k1) = k1 {
            let mut x = x.rb_mut().subcols_range_mut(k..k1.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1)) = (x.next(), x.next()) else {
                panic!()
            };

            for j in N.indices() {
                let ui = u.row_indices_of_col_raw(j);
                let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                let u_inv = ux.read(ui.len() - 1).faer_inv();
                let u_inv = if conj == Conj::Yes {
                    u_inv.faer_conj()
                } else {
                    u_inv
                };

                let mut acc0a = E::faer_zero();
                let mut acc0b = E::faer_zero();
                let mut acc1a = E::faer_zero();
                let mut acc1b = E::faer_zero();

                let a = 0;
                let b = 1;

                let rows_head = ui[..ui.len() - 1].chunks_exact(2);
                let rows_tail = rows_head.remainder();
                let (values_head, values_tail) = ux.subslice(0..ui.len() - 1).into_chunks_exact(2);

                for (i, uij) in iter::zip(rows_head, values_head) {
                    let uija = uij.read(a);
                    let uijb = uij.read(b);
                    let uija = if conj == Conj::Yes {
                        uija.faer_conj()
                    } else {
                        uija
                    };
                    let uijb = if conj == Conj::Yes {
                        uijb.faer_conj()
                    } else {
                        uijb
                    };
                    acc0a = acc0a.faer_add(uija.faer_mul(x0.read(i[a].zx())));
                    acc0b = acc0b.faer_add(uijb.faer_mul(x0.read(i[b].zx())));
                    acc1a = acc1a.faer_add(uija.faer_mul(x1.read(i[a].zx())));
                    acc1b = acc1b.faer_add(uijb.faer_mul(x1.read(i[b].zx())));
                }

                for (i, uij) in iter::zip(rows_tail, values_tail.into_ref_iter()) {
                    let uija = uij.read();
                    let uija = if conj == Conj::Yes {
                        uija.faer_conj()
                    } else {
                        uija
                    };
                    acc0a = acc0a.faer_add(uija.faer_mul(x0.read(i.zx())));
                    acc1a = acc1a.faer_add(uija.faer_mul(x1.read(i.zx())));
                }

                x0.write(
                    j,
                    x0.read(j).faer_sub(acc0a.faer_add(acc0b)).faer_mul(u_inv),
                );
                x1.write(
                    j,
                    x1.read(j).faer_sub(acc1a.faer_add(acc1b)).faer_mul(u_inv),
                );
            }
            k = k1.next();
        } else {
            let mut x0 = x.rb_mut().col_mut(k0);

            for j in N.indices() {
                let ui = u.row_indices_of_col_raw(j);
                let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                let u_inv = ux.read(ui.len() - 1).faer_inv();
                let u_inv = if conj == Conj::Yes {
                    u_inv.faer_conj()
                } else {
                    u_inv
                };

                let mut acc0a = E::faer_zero();
                let mut acc0b = E::faer_zero();
                let mut acc0c = E::faer_zero();
                let mut acc0d = E::faer_zero();

                let a = 0;
                let b = 1;
                let c = 2;
                let d = 3;

                let rows_head = ui[..ui.len() - 1].chunks_exact(4);
                let rows_tail = rows_head.remainder();
                let (values_head, values_tail) = ux.subslice(0..ui.len() - 1).into_chunks_exact(4);

                for (i, uij) in iter::zip(rows_head, values_head) {
                    let uija = uij.read(a);
                    let uijb = uij.read(b);
                    let uijc = uij.read(c);
                    let uijd = uij.read(d);
                    let uija = if conj == Conj::Yes {
                        uija.faer_conj()
                    } else {
                        uija
                    };
                    let uijb = if conj == Conj::Yes {
                        uijb.faer_conj()
                    } else {
                        uijb
                    };
                    let uijc = if conj == Conj::Yes {
                        uijc.faer_conj()
                    } else {
                        uijc
                    };
                    let uijd = if conj == Conj::Yes {
                        uijd.faer_conj()
                    } else {
                        uijd
                    };
                    acc0a = acc0a.faer_add(uija.faer_mul(x0.read(i[a].zx())));
                    acc0b = acc0b.faer_add(uijb.faer_mul(x0.read(i[b].zx())));
                    acc0c = acc0c.faer_add(uijc.faer_mul(x0.read(i[c].zx())));
                    acc0d = acc0d.faer_add(uijd.faer_mul(x0.read(i[d].zx())));
                }

                for (i, uij) in iter::zip(rows_tail, values_tail.into_ref_iter()) {
                    let uija = uij.read();
                    let uija = if conj == Conj::Yes {
                        uija.faer_conj()
                    } else {
                        uija
                    };
                    acc0a = acc0a.faer_add(uija.faer_mul(x0.read(i.zx())));
                }

                x0.write(
                    j,
                    x0.read(j)
                        .faer_sub(acc0a.faer_add(acc0b).faer_add(acc0c.faer_add(acc0d)))
                        .faer_mul(u_inv),
                );
            }
            k = k0.next();
        }
    }
}

/// Assuming `self` is a unit upper triangular matrix, solves the equation `Op(self) * X =
/// rhs`, and stores the result in `rhs`, where `Op` is either the conjugate or the identity
/// depending on the value of `conj`.
///
/// # Note
/// The matrix indices need not be sorted, but
/// the diagonal element is assumed to be the last stored element in each column.
pub fn solve_unit_upper_triangular_in_place<I: Index, E: ComplexField>(
    u: SparseColMatRef<'_, I, E>,
    conj: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    let _ = parallelism;
    assert!(u.nrows() == u.ncols());
    assert!(rhs.nrows() == u.nrows());

    with_dim!(N, rhs.nrows());
    with_dim!(K, rhs.ncols());

    let mut x = rhs.as_shape_mut(N, K);
    let u = u.as_shape(N, N);

    let mut k = bound::IdxInc::<usize>::zero();
    while let Some(k0) = K.try_check(*k) {
        let k1 = K.try_check(*k + 1);
        let k2 = K.try_check(*k + 2);
        let k3 = K.try_check(*k + 3);

        if let Some(k3) = k3 {
            let mut x = x.rb_mut().subcols_range_mut(k..k3.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1), Some(mut x2), Some(mut x3)) =
                (x.next(), x.next(), x.next(), x.next())
            else {
                panic!()
            };

            for j in N.indices().rev() {
                let ui = u.row_indices_of_col_raw(j);
                let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                let xj0 = x0.read(j);
                let xj1 = x1.read(j);
                let xj2 = x2.read(j);
                let xj3 = x3.read(j);

                for (i, u) in iter::zip(
                    &ui[..ui.len() - 1],
                    ux.subslice(0..ui.len() - 1).into_ref_iter(),
                ) {
                    let i = i.zx();
                    let u = if conj == Conj::Yes {
                        u.read().faer_conj()
                    } else {
                        u.read()
                    };

                    x0.write(i, x0.read(i).faer_sub(E::faer_mul(u, xj0)));
                    x1.write(i, x1.read(i).faer_sub(E::faer_mul(u, xj1)));
                    x2.write(i, x2.read(i).faer_sub(E::faer_mul(u, xj2)));
                    x3.write(i, x3.read(i).faer_sub(E::faer_mul(u, xj3)));
                }
            }
            k = k3.next();
        } else if let Some(k2) = k2 {
            let mut x = x.rb_mut().subcols_range_mut(k..k2.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1), Some(mut x2)) = (x.next(), x.next(), x.next()) else {
                panic!()
            };

            for j in N.indices().rev() {
                let ui = u.row_indices_of_col_raw(j);
                let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                let xj0 = x0.read(j);
                let xj1 = x1.read(j);
                let xj2 = x2.read(j);

                for (i, u) in iter::zip(
                    &ui[..ui.len() - 1],
                    ux.subslice(0..ui.len() - 1).into_ref_iter(),
                ) {
                    let i = i.zx();
                    let u = if conj == Conj::Yes {
                        u.read().faer_conj()
                    } else {
                        u.read()
                    };

                    x0.write(i, x0.read(i).faer_sub(E::faer_mul(u, xj0)));
                    x1.write(i, x1.read(i).faer_sub(E::faer_mul(u, xj1)));
                    x2.write(i, x2.read(i).faer_sub(E::faer_mul(u, xj2)));
                }
            }
            k = k2.next();
        } else if let Some(k1) = k1 {
            let mut x = x.rb_mut().subcols_range_mut(k..k1.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1)) = (x.next(), x.next()) else {
                panic!()
            };

            for j in N.indices().rev() {
                let ui = u.row_indices_of_col_raw(j);
                let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                let xj0 = x0.read(j);
                let xj1 = x1.read(j);

                for (i, u) in iter::zip(
                    &ui[..ui.len() - 1],
                    ux.subslice(0..ui.len() - 1).into_ref_iter(),
                ) {
                    let i = i.zx();
                    let u = if conj == Conj::Yes {
                        u.read().faer_conj()
                    } else {
                        u.read()
                    };

                    x0.write(i, x0.read(i).faer_sub(E::faer_mul(u, xj0)));
                    x1.write(i, x1.read(i).faer_sub(E::faer_mul(u, xj1)));
                }
            }
            k = k1.next();
        } else {
            let mut x0 = x.rb_mut().col_mut(k0);

            for j in N.indices().rev() {
                let ui = u.row_indices_of_col_raw(j);
                let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                let xj = x0.read(j);

                for (i, u) in iter::zip(
                    &ui[..ui.len() - 1],
                    ux.subslice(0..ui.len() - 1).into_ref_iter(),
                ) {
                    let i = i.zx();
                    let u = if conj == Conj::Yes {
                        u.read().faer_conj()
                    } else {
                        u.read()
                    };

                    x0.write(i, x0.read(i).faer_sub(E::faer_mul(u, xj)));
                }
            }
            k = k0.next();
        }
    }
}

/// Assuming `self` is a unit upper triangular matrix, solves the equation `Op(self).transpose() * X
/// = rhs`, and stores the result in `rhs`, where `Op` is either the conjugate or the identity
/// depending on the value of `conj`.
///
/// # Note
/// The matrix indices need not be sorted, but
/// the diagonal element is assumed to be the last stored element in each column.
pub fn solve_unit_upper_triangular_transpose_in_place<I: Index, E: ComplexField>(
    u: SparseColMatRef<'_, I, E>,
    conj: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    let _ = parallelism;
    assert!(u.nrows() == u.ncols());
    assert!(rhs.nrows() == u.nrows());

    with_dim!(N, rhs.nrows());
    with_dim!(K, rhs.ncols());

    let mut x = rhs.as_shape_mut(N, K);
    let u = u.as_shape(N, N);

    let mut k = bound::IdxInc::<usize>::zero();
    while let Some(k0) = K.try_check(*k) {
        let k1 = K.try_check(*k + 1);
        let k2 = K.try_check(*k + 2);
        let k3 = K.try_check(*k + 3);

        if let Some(k3) = k3 {
            let mut x = x.rb_mut().subcols_range_mut(k..k3.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1), Some(mut x2), Some(mut x3)) =
                (x.next(), x.next(), x.next(), x.next())
            else {
                panic!()
            };

            for j in N.indices() {
                let ui = u.row_indices_of_col_raw(j);
                let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                let mut acc0a = E::faer_zero();
                let mut acc1a = E::faer_zero();
                let mut acc2a = E::faer_zero();
                let mut acc3a = E::faer_zero();

                let a = 0;

                let rows_head = ui[..ui.len() - 1].chunks_exact(4);
                let rows_tail = rows_head.remainder();
                let (values_head, values_tail) = ux.subslice(0..ui.len() - 1).into_chunks_exact(4);

                for (i, uij) in iter::zip(rows_head, values_head) {
                    let uija = uij.read(a);
                    let uija = if conj == Conj::Yes {
                        uija.faer_conj()
                    } else {
                        uija
                    };
                    acc0a = acc0a.faer_add(uija.faer_mul(x0.read(i[a].zx())));
                    acc1a = acc1a.faer_add(uija.faer_mul(x1.read(i[a].zx())));
                    acc2a = acc2a.faer_add(uija.faer_mul(x2.read(i[a].zx())));
                    acc3a = acc3a.faer_add(uija.faer_mul(x3.read(i[a].zx())));
                }

                for (i, uij) in iter::zip(rows_tail, values_tail.into_ref_iter()) {
                    let uija = uij.read();
                    let uija = if conj == Conj::Yes {
                        uija.faer_conj()
                    } else {
                        uija
                    };
                    acc0a = acc0a.faer_add(uija.faer_mul(x0.read(i.zx())));
                    acc1a = acc1a.faer_add(uija.faer_mul(x1.read(i.zx())));
                    acc2a = acc2a.faer_add(uija.faer_mul(x2.read(i.zx())));
                    acc3a = acc3a.faer_add(uija.faer_mul(x3.read(i.zx())));
                }

                x0.write(j, x0.read(j).faer_sub(acc0a));
                x1.write(j, x1.read(j).faer_sub(acc1a));
                x2.write(j, x2.read(j).faer_sub(acc2a));
                x3.write(j, x3.read(j).faer_sub(acc3a));
            }
            k = k3.next();
        } else if let Some(k2) = k2 {
            let mut x = x.rb_mut().subcols_range_mut(k..k2.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1), Some(mut x2)) = (x.next(), x.next(), x.next()) else {
                panic!()
            };

            for j in N.indices() {
                let ui = u.row_indices_of_col_raw(j);
                let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                let mut acc0a = E::faer_zero();
                let mut acc1a = E::faer_zero();
                let mut acc2a = E::faer_zero();

                let a = 0;

                let rows_head = ui[..ui.len() - 1].chunks_exact(4);
                let rows_tail = rows_head.remainder();
                let (values_head, values_tail) = ux.subslice(0..ui.len() - 1).into_chunks_exact(4);

                for (i, uij) in iter::zip(rows_head, values_head) {
                    let uija = uij.read(a);
                    let uija = if conj == Conj::Yes {
                        uija.faer_conj()
                    } else {
                        uija
                    };
                    acc0a = acc0a.faer_add(uija.faer_mul(x0.read(i[a].zx())));
                    acc1a = acc1a.faer_add(uija.faer_mul(x1.read(i[a].zx())));
                    acc2a = acc2a.faer_add(uija.faer_mul(x2.read(i[a].zx())));
                }

                for (i, uij) in iter::zip(rows_tail, values_tail.into_ref_iter()) {
                    let uija = uij.read();
                    let uija = if conj == Conj::Yes {
                        uija.faer_conj()
                    } else {
                        uija
                    };
                    acc0a = acc0a.faer_add(uija.faer_mul(x0.read(i.zx())));
                    acc1a = acc1a.faer_add(uija.faer_mul(x1.read(i.zx())));
                    acc2a = acc2a.faer_add(uija.faer_mul(x2.read(i.zx())));
                }

                x0.write(j, x0.read(j).faer_sub(acc0a));
                x1.write(j, x1.read(j).faer_sub(acc1a));
                x2.write(j, x2.read(j).faer_sub(acc2a));
            }
            k = k2.next();
        } else if let Some(k1) = k1 {
            let mut x = x.rb_mut().subcols_range_mut(k..k1.next()).col_iter_mut();
            let (Some(mut x0), Some(mut x1)) = (x.next(), x.next()) else {
                panic!()
            };

            for j in N.indices() {
                let ui = u.row_indices_of_col_raw(j);
                let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                let mut acc0a = E::faer_zero();
                let mut acc0b = E::faer_zero();
                let mut acc1a = E::faer_zero();
                let mut acc1b = E::faer_zero();

                let a = 0;
                let b = 1;

                let rows_head = ui[..ui.len() - 1].chunks_exact(4);
                let rows_tail = rows_head.remainder();
                let (values_head, values_tail) = ux.subslice(0..ui.len() - 1).into_chunks_exact(4);

                for (i, uij) in iter::zip(rows_head, values_head) {
                    let uija = uij.read(a);
                    let uijb = uij.read(b);
                    let uija = if conj == Conj::Yes {
                        uija.faer_conj()
                    } else {
                        uija
                    };
                    let uijb = if conj == Conj::Yes {
                        uijb.faer_conj()
                    } else {
                        uijb
                    };
                    acc0a = acc0a.faer_add(uija.faer_mul(x0.read(i[a].zx())));
                    acc0b = acc0b.faer_add(uijb.faer_mul(x0.read(i[b].zx())));
                    acc1a = acc1a.faer_add(uija.faer_mul(x1.read(i[a].zx())));
                    acc1b = acc1b.faer_add(uijb.faer_mul(x1.read(i[b].zx())));
                }

                for (i, uij) in iter::zip(rows_tail, values_tail.into_ref_iter()) {
                    let uija = uij.read();
                    let uija = if conj == Conj::Yes {
                        uija.faer_conj()
                    } else {
                        uija
                    };
                    acc0a = acc0a.faer_add(uija.faer_mul(x0.read(i.zx())));
                    acc1a = acc1a.faer_add(uija.faer_mul(x1.read(i.zx())));
                }

                x0.write(j, x0.read(j).faer_sub(acc0a.faer_add(acc0b)));
                x1.write(j, x1.read(j).faer_sub(acc1a.faer_add(acc1b)));
            }
            k = k1.next();
        } else {
            let mut x0 = x.rb_mut().col_mut(k0);

            for j in N.indices() {
                let ui = u.row_indices_of_col_raw(j);
                let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                let mut acc0a = E::faer_zero();
                let mut acc0b = E::faer_zero();
                let mut acc0c = E::faer_zero();
                let mut acc0d = E::faer_zero();

                let a = 0;
                let b = 1;
                let c = 2;
                let d = 3;

                let rows_head = ui[..ui.len() - 1].chunks_exact(4);
                let rows_tail = rows_head.remainder();
                let (values_head, values_tail) = ux.subslice(0..ui.len() - 1).into_chunks_exact(4);

                for (i, uij) in iter::zip(rows_head, values_head) {
                    let uija = uij.read(a);
                    let uijb = uij.read(b);
                    let uijc = uij.read(c);
                    let uijd = uij.read(d);
                    let uija = if conj == Conj::Yes {
                        uija.faer_conj()
                    } else {
                        uija
                    };
                    let uijb = if conj == Conj::Yes {
                        uijb.faer_conj()
                    } else {
                        uijb
                    };
                    let uijc = if conj == Conj::Yes {
                        uijc.faer_conj()
                    } else {
                        uijc
                    };
                    let uijd = if conj == Conj::Yes {
                        uijd.faer_conj()
                    } else {
                        uijd
                    };
                    acc0a = acc0a.faer_add(uija.faer_mul(x0.read(i[a].zx())));
                    acc0b = acc0b.faer_add(uijb.faer_mul(x0.read(i[b].zx())));
                    acc0c = acc0c.faer_add(uijc.faer_mul(x0.read(i[c].zx())));
                    acc0d = acc0d.faer_add(uijd.faer_mul(x0.read(i[d].zx())));
                }

                for (i, uij) in iter::zip(rows_tail, values_tail.into_ref_iter()) {
                    let uija = uij.read();
                    let uija = if conj == Conj::Yes {
                        uija.faer_conj()
                    } else {
                        uija
                    };
                    acc0a = acc0a.faer_add(uija.faer_mul(x0.read(i.zx())));
                }

                x0.write(
                    j,
                    x0.read(j)
                        .faer_sub(acc0a.faer_add(acc0b).faer_add(acc0c.faer_add(acc0d))),
                );
            }
            k = k0.next();
        }
    }
}
