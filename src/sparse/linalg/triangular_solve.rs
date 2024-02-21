use crate::{assert, sparse::SparseColMatRef, utils::slice::*, Conj, Index, MatMut, Parallelism};
use core::iter::zip;
use faer_entity::ComplexField;

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

    crate::utils::constrained::Size::with2(
        rhs.nrows(),
        rhs.ncols(),
        #[inline(always)]
        |N, K| {
            let mut x = crate::utils::constrained::mat::MatMut::new(rhs, N, K);
            let l = crate::utils::constrained::sparse::SparseColMatRef::new(l, N, N);

            let mut k = 0usize;
            while k < *K {
                let bs = Ord::min(*K - k, 4);
                match bs {
                    1 => {
                        let k0 = K.check(k);

                        for j in N.indices() {
                            let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                            let d = if conj == Conj::Yes { d.faer_conj() } else { d };

                            let xj0 = x.read(j, k0).faer_mul(d);
                            x.write(j, k0, xj0);

                            for (i, lij) in zip(
                                l.row_indices_of_col(j),
                                slice_group(l.values_of_col(j)).into_ref_iter(),
                            )
                            .skip(1)
                            {
                                let lij = lij.read();
                                let lij = if conj == Conj::Yes {
                                    lij.faer_conj()
                                } else {
                                    lij
                                };
                                x.write(i, k0, x.read(i, k0).faer_sub(lij.faer_mul(xj0)));
                            }
                        }
                    }
                    2 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);
                        for j in N.indices() {
                            let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                            let d = if conj == Conj::Yes { d.faer_conj() } else { d };

                            let xj0 = x.read(j, k0).faer_mul(d);
                            x.write(j, k0, xj0);
                            let xj1 = x.read(j, k1).faer_mul(d);
                            x.write(j, k1, xj1);

                            for (i, lij) in zip(
                                l.row_indices_of_col(j),
                                slice_group(l.values_of_col(j)).into_ref_iter(),
                            )
                            .skip(1)
                            {
                                let lij = lij.read();
                                let lij = if conj == Conj::Yes {
                                    lij.faer_conj()
                                } else {
                                    lij
                                };
                                x.write(i, k0, x.read(i, k0).faer_sub(lij.faer_mul(xj0)));
                                x.write(i, k1, x.read(i, k1).faer_sub(lij.faer_mul(xj1)));
                            }
                        }
                    }
                    3 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);
                        let k2 = K.check(k + 2);
                        for j in N.indices() {
                            let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                            let d = if conj == Conj::Yes { d.faer_conj() } else { d };

                            let xj0 = x.read(j, k0).faer_mul(d);
                            x.write(j, k0, xj0);
                            let xj1 = x.read(j, k1).faer_mul(d);
                            x.write(j, k1, xj1);
                            let xj2 = x.read(j, k2).faer_mul(d);
                            x.write(j, k2, xj2);

                            for (i, lij) in zip(
                                l.row_indices_of_col(j),
                                slice_group(l.values_of_col(j)).into_ref_iter(),
                            )
                            .skip(1)
                            {
                                let lij = lij.read();
                                let lij = if conj == Conj::Yes {
                                    lij.faer_conj()
                                } else {
                                    lij
                                };
                                x.write(i, k0, x.read(i, k0).faer_sub(lij.faer_mul(xj0)));
                                x.write(i, k1, x.read(i, k1).faer_sub(lij.faer_mul(xj1)));
                                x.write(i, k2, x.read(i, k2).faer_sub(lij.faer_mul(xj2)));
                            }
                        }
                    }
                    4 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);
                        let k2 = K.check(k + 2);
                        let k3 = K.check(k + 3);
                        for j in N.indices() {
                            let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                            let d = if conj == Conj::Yes { d.faer_conj() } else { d };

                            let xj0 = x.read(j, k0).faer_mul(d);
                            x.write(j, k0, xj0);
                            let xj1 = x.read(j, k1).faer_mul(d);
                            x.write(j, k1, xj1);
                            let xj2 = x.read(j, k2).faer_mul(d);
                            x.write(j, k2, xj2);
                            let xj3 = x.read(j, k3).faer_mul(d);
                            x.write(j, k3, xj3);

                            for (i, lij) in zip(
                                l.row_indices_of_col(j),
                                slice_group(l.values_of_col(j)).into_ref_iter(),
                            )
                            .skip(1)
                            {
                                let lij = lij.read();
                                let lij = if conj == Conj::Yes {
                                    lij.faer_conj()
                                } else {
                                    lij
                                };
                                x.write(i, k0, x.read(i, k0).faer_sub(lij.faer_mul(xj0)));
                                x.write(i, k1, x.read(i, k1).faer_sub(lij.faer_mul(xj1)));
                                x.write(i, k2, x.read(i, k2).faer_sub(lij.faer_mul(xj2)));
                                x.write(i, k3, x.read(i, k3).faer_sub(lij.faer_mul(xj3)));
                            }
                        }
                    }
                    _ => unreachable!(),
                }
                k += bs;
            }
        },
    );
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

    crate::utils::constrained::Size::with2(
        rhs.nrows(),
        rhs.ncols(),
        #[inline(always)]
        |N, K| {
            let mut x = crate::utils::constrained::mat::MatMut::new(rhs, N, K);
            let l = crate::utils::constrained::sparse::SparseColMatRef::new(l, N, N);

            let mut k = 0usize;
            while k < *K {
                let bs = Ord::min(*K - k, 4);
                match bs {
                    1 => {
                        let k0 = K.check(k);

                        for j in N.indices().rev() {
                            let mut acc0a = E::faer_zero();
                            let mut acc0b = E::faer_zero();
                            let mut acc0c = E::faer_zero();
                            let mut acc0d = E::faer_zero();

                            let a = 0;
                            let b = 1;
                            let c = 2;
                            let d = 3;

                            let nrows = l.row_indices_of_col_raw(j).len();
                            let rows_head = l.row_indices_of_col_raw(j)[1..].chunks_exact(4);
                            let rows_tail = rows_head.remainder();
                            let (values_head, values_tail) = slice_group(l.values_of_col(j))
                                .subslice(1..nrows)
                                .into_chunks_exact(4);

                            for (i, lij) in zip(rows_head, values_head) {
                                let lija = lij.read(a);
                                let lijb = lij.read(b);
                                let lijc = lij.read(c);
                                let lijd = lij.read(d);
                                let lija = if conj == Conj::Yes {
                                    lija.faer_conj()
                                } else {
                                    lija
                                };
                                let lijb = if conj == Conj::Yes {
                                    lijb.faer_conj()
                                } else {
                                    lijb
                                };
                                let lijc = if conj == Conj::Yes {
                                    lijc.faer_conj()
                                } else {
                                    lijc
                                };
                                let lijd = if conj == Conj::Yes {
                                    lijd.faer_conj()
                                } else {
                                    lijd
                                };
                                acc0a = acc0a.faer_add(lija.faer_mul(x.read(i[a].zx(), k0)));
                                acc0b = acc0b.faer_add(lijb.faer_mul(x.read(i[b].zx(), k0)));
                                acc0c = acc0c.faer_add(lijc.faer_mul(x.read(i[c].zx(), k0)));
                                acc0d = acc0d.faer_add(lijd.faer_mul(x.read(i[d].zx(), k0)));
                            }

                            for (i, lij) in zip(rows_tail, values_tail.into_ref_iter()) {
                                let lija = lij.read();
                                let lija = if conj == Conj::Yes {
                                    lija.faer_conj()
                                } else {
                                    lija
                                };
                                acc0a = acc0a.faer_add(lija.faer_mul(x.read(i.zx(), k0)));
                            }

                            let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                            let d = if conj == Conj::Yes { d.faer_conj() } else { d };
                            x.write(
                                j,
                                k0,
                                x.read(j, k0)
                                    .faer_sub(acc0a.faer_add(acc0b).faer_add(acc0c.faer_add(acc0d)))
                                    .faer_mul(d),
                            );
                        }
                    }
                    2 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);

                        for j in N.indices().rev() {
                            let mut acc0a = E::faer_zero();
                            let mut acc0b = E::faer_zero();
                            let mut acc1a = E::faer_zero();
                            let mut acc1b = E::faer_zero();

                            let a = 0;
                            let b = 1;

                            let nrows = l.row_indices_of_col_raw(j).len();
                            let rows_head = l.row_indices_of_col_raw(j)[1..].chunks_exact(2);
                            let rows_tail = rows_head.remainder();
                            let (values_head, values_tail) = slice_group(l.values_of_col(j))
                                .subslice(1..nrows)
                                .into_chunks_exact(2);

                            for (i, lij) in zip(rows_head, values_head) {
                                let lija = lij.read(a);
                                let lijb = lij.read(b);
                                let lija = if conj == Conj::Yes {
                                    lija.faer_conj()
                                } else {
                                    lija
                                };
                                let lijb = if conj == Conj::Yes {
                                    lijb.faer_conj()
                                } else {
                                    lijb
                                };
                                acc0a = acc0a.faer_add(lija.faer_mul(x.read(i[a].zx(), k0)));
                                acc0b = acc0b.faer_add(lijb.faer_mul(x.read(i[b].zx(), k0)));
                                acc1a = acc1a.faer_add(lija.faer_mul(x.read(i[a].zx(), k1)));
                                acc1b = acc1b.faer_add(lijb.faer_mul(x.read(i[b].zx(), k1)));
                            }

                            for (i, lij) in zip(rows_tail, values_tail.into_ref_iter()) {
                                let lija = lij.read();
                                let lija = if conj == Conj::Yes {
                                    lija.faer_conj()
                                } else {
                                    lija
                                };
                                acc0a = acc0a.faer_add(lija.faer_mul(x.read(i.zx(), k0)));
                                acc1a = acc1a.faer_add(lija.faer_mul(x.read(i.zx(), k1)));
                            }

                            let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                            let d = if conj == Conj::Yes { d.faer_conj() } else { d };
                            x.write(
                                j,
                                k0,
                                x.read(j, k0).faer_sub(acc0a.faer_add(acc0b)).faer_mul(d),
                            );
                            x.write(
                                j,
                                k1,
                                x.read(j, k1).faer_sub(acc1a.faer_add(acc1b)).faer_mul(d),
                            );
                        }
                    }
                    3 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);
                        let k2 = K.check(k + 2);

                        for j in N.indices().rev() {
                            let mut acc0a = E::faer_zero();
                            let mut acc1a = E::faer_zero();
                            let mut acc2a = E::faer_zero();

                            for (i, lij) in zip(
                                l.row_indices_of_col(j),
                                slice_group(l.values_of_col(j)).into_ref_iter(),
                            )
                            .skip(1)
                            {
                                let lij = lij.read();
                                let lij = if conj == Conj::Yes {
                                    lij.faer_conj()
                                } else {
                                    lij
                                };
                                acc0a = acc0a.faer_add(lij.faer_mul(x.read(i, k0)));
                                acc1a = acc1a.faer_add(lij.faer_mul(x.read(i, k1)));
                                acc2a = acc2a.faer_add(lij.faer_mul(x.read(i, k2)));
                            }

                            let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                            let d = if conj == Conj::Yes { d.faer_conj() } else { d };
                            x.write(j, k0, x.read(j, k0).faer_sub(acc0a).faer_mul(d));
                            x.write(j, k1, x.read(j, k1).faer_sub(acc1a).faer_mul(d));
                            x.write(j, k2, x.read(j, k2).faer_sub(acc2a).faer_mul(d));
                        }
                    }
                    4 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);
                        let k2 = K.check(k + 2);
                        let k3 = K.check(k + 3);

                        for j in N.indices().rev() {
                            let mut acc0a = E::faer_zero();
                            let mut acc1a = E::faer_zero();
                            let mut acc2a = E::faer_zero();
                            let mut acc3a = E::faer_zero();

                            for (i, lij) in zip(
                                l.row_indices_of_col(j),
                                slice_group(l.values_of_col(j)).into_ref_iter(),
                            )
                            .skip(1)
                            {
                                let lij = lij.read();
                                let lij = if conj == Conj::Yes {
                                    lij.faer_conj()
                                } else {
                                    lij
                                };
                                acc0a = acc0a.faer_add(lij.faer_mul(x.read(i, k0)));
                                acc1a = acc1a.faer_add(lij.faer_mul(x.read(i, k1)));
                                acc2a = acc2a.faer_add(lij.faer_mul(x.read(i, k2)));
                                acc3a = acc3a.faer_add(lij.faer_mul(x.read(i, k3)));
                            }

                            let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                            let d = if conj == Conj::Yes { d.faer_conj() } else { d };
                            x.write(j, k0, x.read(j, k0).faer_sub(acc0a).faer_mul(d));
                            x.write(j, k1, x.read(j, k1).faer_sub(acc1a).faer_mul(d));
                            x.write(j, k2, x.read(j, k2).faer_sub(acc2a).faer_mul(d));
                            x.write(j, k3, x.read(j, k3).faer_sub(acc3a).faer_mul(d));
                        }
                    }
                    _ => unreachable!(),
                }
                k += bs;
            }
        },
    );
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

    crate::utils::constrained::Size::with2(
        rhs.nrows(),
        rhs.ncols(),
        #[inline(always)]
        |N, K| {
            let mut x = crate::utils::constrained::mat::MatMut::new(rhs, N, K);
            let l = crate::utils::constrained::sparse::SparseColMatRef::new(l, N, N);

            let mut k = 0usize;
            while k < *K {
                let bs = Ord::min(*K - k, 4);
                match bs {
                    1 => {
                        let k0 = K.check(k);

                        for j in N.indices() {
                            let xj0 = x.read(j, k0);

                            for (i, lij) in zip(
                                l.row_indices_of_col(j),
                                slice_group(l.values_of_col(j)).into_ref_iter(),
                            )
                            .skip(1)
                            {
                                let lij = lij.read();
                                let lij = if conj == Conj::Yes {
                                    lij.faer_conj()
                                } else {
                                    lij
                                };
                                x.write(i, k0, x.read(i, k0).faer_sub(lij.faer_mul(xj0)));
                            }
                        }
                    }
                    2 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);
                        for j in N.indices() {
                            let xj0 = x.read(j, k0);
                            let xj1 = x.read(j, k1);

                            for (i, lij) in zip(
                                l.row_indices_of_col(j),
                                slice_group(l.values_of_col(j)).into_ref_iter(),
                            )
                            .skip(1)
                            {
                                let lij = lij.read();
                                let lij = if conj == Conj::Yes {
                                    lij.faer_conj()
                                } else {
                                    lij
                                };
                                x.write(i, k0, x.read(i, k0).faer_sub(lij.faer_mul(xj0)));
                                x.write(i, k1, x.read(i, k1).faer_sub(lij.faer_mul(xj1)));
                            }
                        }
                    }
                    3 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);
                        let k2 = K.check(k + 2);
                        for j in N.indices() {
                            let xj0 = x.read(j, k0);
                            let xj1 = x.read(j, k1);
                            let xj2 = x.read(j, k2);

                            for (i, lij) in zip(
                                l.row_indices_of_col(j),
                                slice_group(l.values_of_col(j)).into_ref_iter(),
                            )
                            .skip(1)
                            {
                                let lij = lij.read();
                                let lij = if conj == Conj::Yes {
                                    lij.faer_conj()
                                } else {
                                    lij
                                };
                                x.write(i, k0, x.read(i, k0).faer_sub(lij.faer_mul(xj0)));
                                x.write(i, k1, x.read(i, k1).faer_sub(lij.faer_mul(xj1)));
                                x.write(i, k2, x.read(i, k2).faer_sub(lij.faer_mul(xj2)));
                            }
                        }
                    }
                    4 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);
                        let k2 = K.check(k + 2);
                        let k3 = K.check(k + 3);
                        for j in N.indices() {
                            let xj0 = x.read(j, k0);
                            let xj1 = x.read(j, k1);
                            let xj2 = x.read(j, k2);
                            let xj3 = x.read(j, k3);

                            for (i, lij) in zip(
                                l.row_indices_of_col(j),
                                slice_group(l.values_of_col(j)).into_ref_iter(),
                            )
                            .skip(1)
                            {
                                let lij = lij.read();
                                let lij = if conj == Conj::Yes {
                                    lij.faer_conj()
                                } else {
                                    lij
                                };
                                x.write(i, k0, x.read(i, k0).faer_sub(lij.faer_mul(xj0)));
                                x.write(i, k1, x.read(i, k1).faer_sub(lij.faer_mul(xj1)));
                                x.write(i, k2, x.read(i, k2).faer_sub(lij.faer_mul(xj2)));
                                x.write(i, k3, x.read(i, k3).faer_sub(lij.faer_mul(xj3)));
                            }
                        }
                    }
                    _ => unreachable!(),
                }

                k += bs;
            }
        },
    );
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

    crate::utils::constrained::Size::with2(
        rhs.nrows(),
        rhs.ncols(),
        #[inline(always)]
        |N, K| {
            let mut x = crate::utils::constrained::mat::MatMut::new(rhs, N, K);
            let l = crate::utils::constrained::sparse::SparseColMatRef::new(l, N, N);

            let mut k = 0usize;
            while k < *K {
                let bs = Ord::min(*K - k, 4);
                match bs {
                    1 => {
                        let k0 = K.check(k);

                        for j in N.indices().rev() {
                            let mut acc0a = E::faer_zero();
                            let mut acc0b = E::faer_zero();
                            let mut acc0c = E::faer_zero();
                            let mut acc0d = E::faer_zero();

                            let a = 0;
                            let b = 1;
                            let c = 2;
                            let d = 3;

                            let nrows = l.row_indices_of_col_raw(j).len();
                            let rows_head = l.row_indices_of_col_raw(j)[1..].chunks_exact(4);
                            let rows_tail = rows_head.remainder();
                            let (values_head, values_tail) = slice_group(l.values_of_col(j))
                                .subslice(1..nrows)
                                .into_chunks_exact(4);

                            for (i, lij) in zip(rows_head, values_head) {
                                let lija = lij.read(a);
                                let lijb = lij.read(b);
                                let lijc = lij.read(c);
                                let lijd = lij.read(d);
                                let lija = if conj == Conj::Yes {
                                    lija.faer_conj()
                                } else {
                                    lija
                                };
                                let lijb = if conj == Conj::Yes {
                                    lijb.faer_conj()
                                } else {
                                    lijb
                                };
                                let lijc = if conj == Conj::Yes {
                                    lijc.faer_conj()
                                } else {
                                    lijc
                                };
                                let lijd = if conj == Conj::Yes {
                                    lijd.faer_conj()
                                } else {
                                    lijd
                                };
                                acc0a = acc0a.faer_add(lija.faer_mul(x.read(i[a].zx(), k0)));
                                acc0b = acc0b.faer_add(lijb.faer_mul(x.read(i[b].zx(), k0)));
                                acc0c = acc0c.faer_add(lijc.faer_mul(x.read(i[c].zx(), k0)));
                                acc0d = acc0d.faer_add(lijd.faer_mul(x.read(i[d].zx(), k0)));
                            }

                            for (i, lij) in zip(rows_tail, values_tail.into_ref_iter()) {
                                let lija = lij.read();
                                let lija = if conj == Conj::Yes {
                                    lija.faer_conj()
                                } else {
                                    lija
                                };
                                acc0a = acc0a.faer_add(lija.faer_mul(x.read(i.zx(), k0)));
                            }

                            x.write(
                                j,
                                k0,
                                x.read(j, k0).faer_sub(
                                    acc0a.faer_add(acc0b).faer_add(acc0c.faer_add(acc0d)),
                                ),
                            );
                        }
                    }
                    2 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);

                        for j in N.indices().rev() {
                            let mut acc0a = E::faer_zero();
                            let mut acc0b = E::faer_zero();
                            let mut acc1a = E::faer_zero();
                            let mut acc1b = E::faer_zero();

                            let a = 0;
                            let b = 1;

                            let nrows = l.row_indices_of_col_raw(j).len();
                            let rows_head = l.row_indices_of_col_raw(j)[1..].chunks_exact(2);
                            let rows_tail = rows_head.remainder();
                            let (values_head, values_tail) = slice_group(l.values_of_col(j))
                                .subslice(1..nrows)
                                .into_chunks_exact(2);

                            for (i, lij) in zip(rows_head, values_head) {
                                let lija = lij.read(a);
                                let lijb = lij.read(b);
                                let lija = if conj == Conj::Yes {
                                    lija.faer_conj()
                                } else {
                                    lija
                                };
                                let lijb = if conj == Conj::Yes {
                                    lijb.faer_conj()
                                } else {
                                    lijb
                                };
                                acc0a = acc0a.faer_add(lija.faer_mul(x.read(i[a].zx(), k0)));
                                acc0b = acc0b.faer_add(lijb.faer_mul(x.read(i[b].zx(), k0)));
                                acc1a = acc1a.faer_add(lija.faer_mul(x.read(i[a].zx(), k1)));
                                acc1b = acc1b.faer_add(lijb.faer_mul(x.read(i[b].zx(), k1)));
                            }

                            for (i, lij) in zip(rows_tail, values_tail.into_ref_iter()) {
                                let lija = lij.read();
                                let lija = if conj == Conj::Yes {
                                    lija.faer_conj()
                                } else {
                                    lija
                                };
                                acc0a = acc0a.faer_add(lija.faer_mul(x.read(i.zx(), k0)));
                                acc1a = acc1a.faer_add(lija.faer_mul(x.read(i.zx(), k1)));
                            }

                            x.write(j, k0, x.read(j, k0).faer_sub(acc0a.faer_add(acc0b)));
                            x.write(j, k1, x.read(j, k1).faer_sub(acc1a.faer_add(acc1b)));
                        }
                    }
                    3 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);
                        let k2 = K.check(k + 2);

                        for j in N.indices().rev() {
                            let mut acc0a = E::faer_zero();
                            let mut acc1a = E::faer_zero();
                            let mut acc2a = E::faer_zero();

                            for (i, lij) in zip(
                                l.row_indices_of_col(j),
                                slice_group(l.values_of_col(j)).into_ref_iter(),
                            )
                            .skip(1)
                            {
                                let lij = lij.read();
                                let lij = if conj == Conj::Yes {
                                    lij.faer_conj()
                                } else {
                                    lij
                                };
                                acc0a = acc0a.faer_add(lij.faer_mul(x.read(i, k0)));
                                acc1a = acc1a.faer_add(lij.faer_mul(x.read(i, k1)));
                                acc2a = acc2a.faer_add(lij.faer_mul(x.read(i, k2)));
                            }

                            x.write(j, k0, x.read(j, k0).faer_sub(acc0a));
                            x.write(j, k1, x.read(j, k1).faer_sub(acc1a));
                            x.write(j, k2, x.read(j, k2).faer_sub(acc2a));
                        }
                    }
                    4 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);
                        let k2 = K.check(k + 2);
                        let k3 = K.check(k + 3);

                        for j in N.indices().rev() {
                            let mut acc0a = E::faer_zero();
                            let mut acc1a = E::faer_zero();
                            let mut acc2a = E::faer_zero();
                            let mut acc3a = E::faer_zero();

                            for (i, lij) in zip(
                                l.row_indices_of_col(j),
                                slice_group(l.values_of_col(j)).into_ref_iter(),
                            )
                            .skip(1)
                            {
                                let lij = lij.read();
                                let lij = if conj == Conj::Yes {
                                    lij.faer_conj()
                                } else {
                                    lij
                                };
                                acc0a = acc0a.faer_add(lij.faer_mul(x.read(i, k0)));
                                acc1a = acc1a.faer_add(lij.faer_mul(x.read(i, k1)));
                                acc2a = acc2a.faer_add(lij.faer_mul(x.read(i, k2)));
                                acc3a = acc3a.faer_add(lij.faer_mul(x.read(i, k3)));
                            }

                            x.write(j, k0, x.read(j, k0).faer_sub(acc0a));
                            x.write(j, k1, x.read(j, k1).faer_sub(acc1a));
                            x.write(j, k2, x.read(j, k2).faer_sub(acc2a));
                            x.write(j, k3, x.read(j, k3).faer_sub(acc3a));
                        }
                    }
                    _ => unreachable!(),
                }
                k += bs;
            }
        },
    );
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

    crate::utils::constrained::Size::with2(
        rhs.nrows(),
        rhs.ncols(),
        #[inline(always)]
        |N, K| {
            let mut x = crate::utils::constrained::mat::MatMut::new(rhs, N, K);
            let u = crate::utils::constrained::sparse::SparseColMatRef::new(u, N, N);

            let mut k = 0usize;
            while k < *K {
                let bs = Ord::min(*K - k, 4);
                match bs {
                    1 => {
                        let k0 = K.check(k);

                        for j in N.indices().rev() {
                            let ui = u.row_indices_of_col_raw(j);
                            let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                            let u_inv = ux.read(ui.len() - 1).faer_inv();
                            let u_inv = if conj == Conj::Yes {
                                u_inv.faer_conj()
                            } else {
                                u_inv
                            };
                            let xj = x.read(j, k0).faer_mul(u_inv);
                            x.write(j, k0, xj);

                            for (i, u) in zip(
                                &ui[..ui.len() - 1],
                                ux.subslice(0..ui.len() - 1).into_ref_iter(),
                            ) {
                                let i = i.zx();
                                let u = if conj == Conj::Yes {
                                    u.read().faer_conj()
                                } else {
                                    u.read()
                                };

                                x.write(i, k0, x.read(i, k0).faer_sub(E::faer_mul(u, xj)));
                            }
                        }
                    }
                    2 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);

                        for j in N.indices().rev() {
                            let ui = u.row_indices_of_col_raw(j);
                            let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                            let u_inv = ux.read(ui.len() - 1).faer_inv();
                            let u_inv = if conj == Conj::Yes {
                                u_inv.faer_conj()
                            } else {
                                u_inv
                            };
                            let xj0 = x.read(j, k0).faer_mul(u_inv);
                            let xj1 = x.read(j, k1).faer_mul(u_inv);
                            x.write(j, k0, xj0);
                            x.write(j, k1, xj1);

                            for (i, u) in zip(
                                &ui[..ui.len() - 1],
                                ux.subslice(0..ui.len() - 1).into_ref_iter(),
                            ) {
                                let i = i.zx();
                                let u = if conj == Conj::Yes {
                                    u.read().faer_conj()
                                } else {
                                    u.read()
                                };

                                x.write(i, k0, x.read(i, k0).faer_sub(E::faer_mul(u, xj0)));
                                x.write(i, k1, x.read(i, k1).faer_sub(E::faer_mul(u, xj1)));
                            }
                        }
                    }
                    3 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);
                        let k2 = K.check(k + 2);

                        for j in N.indices().rev() {
                            let ui = u.row_indices_of_col_raw(j);
                            let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                            let u_inv = ux.read(ui.len() - 1).faer_inv();
                            let u_inv = if conj == Conj::Yes {
                                u_inv.faer_conj()
                            } else {
                                u_inv
                            };
                            let xj0 = x.read(j, k0).faer_mul(u_inv);
                            let xj1 = x.read(j, k1).faer_mul(u_inv);
                            let xj2 = x.read(j, k2).faer_mul(u_inv);
                            x.write(j, k0, xj0);
                            x.write(j, k1, xj1);
                            x.write(j, k2, xj2);

                            for (i, u) in zip(
                                &ui[..ui.len() - 1],
                                ux.subslice(0..ui.len() - 1).into_ref_iter(),
                            ) {
                                let i = i.zx();
                                let u = if conj == Conj::Yes {
                                    u.read().faer_conj()
                                } else {
                                    u.read()
                                };

                                x.write(i, k0, x.read(i, k0).faer_sub(E::faer_mul(u, xj0)));
                                x.write(i, k1, x.read(i, k1).faer_sub(E::faer_mul(u, xj1)));
                                x.write(i, k2, x.read(i, k2).faer_sub(E::faer_mul(u, xj2)));
                            }
                        }
                    }
                    4 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);
                        let k2 = K.check(k + 2);
                        let k3 = K.check(k + 3);

                        for j in N.indices().rev() {
                            let ui = u.row_indices_of_col_raw(j);
                            let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                            let u_inv = ux.read(ui.len() - 1).faer_inv();
                            let u_inv = if conj == Conj::Yes {
                                u_inv.faer_conj()
                            } else {
                                u_inv
                            };
                            let xj0 = x.read(j, k0).faer_mul(u_inv);
                            let xj1 = x.read(j, k1).faer_mul(u_inv);
                            let xj2 = x.read(j, k2).faer_mul(u_inv);
                            let xj3 = x.read(j, k3).faer_mul(u_inv);
                            x.write(j, k0, xj0);
                            x.write(j, k1, xj1);
                            x.write(j, k2, xj2);
                            x.write(j, k3, xj3);

                            for (i, u) in zip(
                                &ui[..ui.len() - 1],
                                ux.subslice(0..ui.len() - 1).into_ref_iter(),
                            ) {
                                let i = i.zx();
                                let u = if conj == Conj::Yes {
                                    u.read().faer_conj()
                                } else {
                                    u.read()
                                };

                                x.write(i, k0, x.read(i, k0).faer_sub(E::faer_mul(u, xj0)));
                                x.write(i, k1, x.read(i, k1).faer_sub(E::faer_mul(u, xj1)));
                                x.write(i, k2, x.read(i, k2).faer_sub(E::faer_mul(u, xj2)));
                                x.write(i, k3, x.read(i, k3).faer_sub(E::faer_mul(u, xj3)));
                            }
                        }
                    }
                    _ => unreachable!(),
                }
                k += bs;
            }
        },
    );
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

    crate::utils::constrained::Size::with2(
        rhs.nrows(),
        rhs.ncols(),
        #[inline(always)]
        |N, K| {
            let mut x = crate::utils::constrained::mat::MatMut::new(rhs, N, K);
            let u = crate::utils::constrained::sparse::SparseColMatRef::new(u, N, N);

            let mut k = 0usize;
            while k < *K {
                let bs = Ord::min(*K - k, 4);
                match bs {
                    1 => {
                        let k0 = K.check(k);
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
                            let (values_head, values_tail) =
                                ux.subslice(0..ui.len() - 1).into_chunks_exact(4);

                            for (i, uij) in zip(rows_head, values_head) {
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
                                acc0a = acc0a.faer_add(uija.faer_mul(x.read(i[a].zx(), k0)));
                                acc0b = acc0b.faer_add(uijb.faer_mul(x.read(i[b].zx(), k0)));
                                acc0c = acc0c.faer_add(uijc.faer_mul(x.read(i[c].zx(), k0)));
                                acc0d = acc0d.faer_add(uijd.faer_mul(x.read(i[d].zx(), k0)));
                            }

                            for (i, uij) in zip(rows_tail, values_tail.into_ref_iter()) {
                                let uija = uij.read();
                                let uija = if conj == Conj::Yes {
                                    uija.faer_conj()
                                } else {
                                    uija
                                };
                                acc0a = acc0a.faer_add(uija.faer_mul(x.read(i.zx(), k0)));
                            }

                            x.write(
                                j,
                                k0,
                                x.read(j, k0)
                                    .faer_sub(acc0a.faer_add(acc0b).faer_add(acc0c.faer_add(acc0d)))
                                    .faer_mul(u_inv),
                            );
                        }
                    }
                    2 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);

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
                            let (values_head, values_tail) =
                                ux.subslice(0..ui.len() - 1).into_chunks_exact(2);

                            for (i, uij) in zip(rows_head, values_head) {
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
                                acc0a = acc0a.faer_add(uija.faer_mul(x.read(i[a].zx(), k0)));
                                acc0b = acc0b.faer_add(uijb.faer_mul(x.read(i[b].zx(), k0)));
                                acc1a = acc1a.faer_add(uija.faer_mul(x.read(i[a].zx(), k1)));
                                acc1b = acc1b.faer_add(uijb.faer_mul(x.read(i[b].zx(), k1)));
                            }

                            for (i, uij) in zip(rows_tail, values_tail.into_ref_iter()) {
                                let uija = uij.read();
                                let uija = if conj == Conj::Yes {
                                    uija.faer_conj()
                                } else {
                                    uija
                                };
                                acc0a = acc0a.faer_add(uija.faer_mul(x.read(i.zx(), k0)));
                                acc1a = acc1a.faer_add(uija.faer_mul(x.read(i.zx(), k1)));
                            }

                            x.write(
                                j,
                                k0,
                                x.read(j, k0)
                                    .faer_sub(acc0a.faer_add(acc0b))
                                    .faer_mul(u_inv),
                            );
                            x.write(
                                j,
                                k1,
                                x.read(j, k1)
                                    .faer_sub(acc1a.faer_add(acc1b))
                                    .faer_mul(u_inv),
                            );
                        }
                    }
                    3 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);
                        let k2 = K.check(k + 2);

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

                            for (i, uij) in zip(rows, values.into_ref_iter()) {
                                let uija = uij.read();
                                let uija = if conj == Conj::Yes {
                                    uija.faer_conj()
                                } else {
                                    uija
                                };
                                acc0a = acc0a.faer_add(uija.faer_mul(x.read(i.zx(), k0)));
                                acc1a = acc1a.faer_add(uija.faer_mul(x.read(i.zx(), k1)));
                                acc2a = acc2a.faer_add(uija.faer_mul(x.read(i.zx(), k2)));
                            }

                            x.write(j, k0, x.read(j, k0).faer_sub(acc0a).faer_mul(u_inv));
                            x.write(j, k1, x.read(j, k1).faer_sub(acc1a).faer_mul(u_inv));
                            x.write(j, k2, x.read(j, k2).faer_sub(acc2a).faer_mul(u_inv));
                        }
                    }
                    4 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);
                        let k2 = K.check(k + 2);
                        let k3 = K.check(k + 3);

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

                            for (i, uij) in zip(rows, values.into_ref_iter()) {
                                let uija = uij.read();
                                let uija = if conj == Conj::Yes {
                                    uija.faer_conj()
                                } else {
                                    uija
                                };
                                acc0a = acc0a.faer_add(uija.faer_mul(x.read(i.zx(), k0)));
                                acc1a = acc1a.faer_add(uija.faer_mul(x.read(i.zx(), k1)));
                                acc2a = acc2a.faer_add(uija.faer_mul(x.read(i.zx(), k2)));
                                acc3a = acc3a.faer_add(uija.faer_mul(x.read(i.zx(), k3)));
                            }

                            x.write(j, k0, x.read(j, k0).faer_sub(acc0a).faer_mul(u_inv));
                            x.write(j, k1, x.read(j, k1).faer_sub(acc1a).faer_mul(u_inv));
                            x.write(j, k2, x.read(j, k2).faer_sub(acc2a).faer_mul(u_inv));
                            x.write(j, k3, x.read(j, k3).faer_sub(acc3a).faer_mul(u_inv));
                        }
                    }
                    _ => unreachable!(),
                }
                k += bs;
            }
        },
    );
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

    crate::utils::constrained::Size::with2(
        rhs.nrows(),
        rhs.ncols(),
        #[inline(always)]
        |N, K| {
            let mut x = crate::utils::constrained::mat::MatMut::new(rhs, N, K);
            let u = crate::utils::constrained::sparse::SparseColMatRef::new(u, N, N);

            let mut k = 0usize;
            while k < *K {
                let bs = Ord::min(*K - k, 4);
                match bs {
                    1 => {
                        let k0 = K.check(k);

                        for j in N.indices().rev() {
                            let ui = u.row_indices_of_col_raw(j);
                            let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                            let xj = x.read(j, k0);

                            for (i, u) in zip(
                                &ui[..ui.len() - 1],
                                ux.subslice(0..ui.len() - 1).into_ref_iter(),
                            ) {
                                let i = i.zx();
                                let u = if conj == Conj::Yes {
                                    u.read().faer_conj()
                                } else {
                                    u.read()
                                };

                                x.write(i, k0, x.read(i, k0).faer_sub(E::faer_mul(u, xj)));
                            }
                        }
                    }
                    2 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);

                        for j in N.indices().rev() {
                            let ui = u.row_indices_of_col_raw(j);
                            let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                            let xj0 = x.read(j, k0);
                            let xj1 = x.read(j, k1);

                            for (i, u) in zip(
                                &ui[..ui.len() - 1],
                                ux.subslice(0..ui.len() - 1).into_ref_iter(),
                            ) {
                                let i = i.zx();
                                let u = if conj == Conj::Yes {
                                    u.read().faer_conj()
                                } else {
                                    u.read()
                                };

                                x.write(i, k0, x.read(i, k0).faer_sub(E::faer_mul(u, xj0)));
                                x.write(i, k1, x.read(i, k1).faer_sub(E::faer_mul(u, xj1)));
                            }
                        }
                    }
                    3 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);
                        let k2 = K.check(k + 2);

                        for j in N.indices().rev() {
                            let ui = u.row_indices_of_col_raw(j);
                            let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                            let xj0 = x.read(j, k0);
                            let xj1 = x.read(j, k1);
                            let xj2 = x.read(j, k2);

                            for (i, u) in zip(
                                &ui[..ui.len() - 1],
                                ux.subslice(0..ui.len() - 1).into_ref_iter(),
                            ) {
                                let i = i.zx();
                                let u = if conj == Conj::Yes {
                                    u.read().faer_conj()
                                } else {
                                    u.read()
                                };

                                x.write(i, k0, x.read(i, k0).faer_sub(E::faer_mul(u, xj0)));
                                x.write(i, k1, x.read(i, k1).faer_sub(E::faer_mul(u, xj1)));
                                x.write(i, k2, x.read(i, k2).faer_sub(E::faer_mul(u, xj2)));
                            }
                        }
                    }
                    4 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);
                        let k2 = K.check(k + 2);
                        let k3 = K.check(k + 3);

                        for j in N.indices().rev() {
                            let ui = u.row_indices_of_col_raw(j);
                            let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                            let xj0 = x.read(j, k0);
                            let xj1 = x.read(j, k1);
                            let xj2 = x.read(j, k2);
                            let xj3 = x.read(j, k3);

                            for (i, u) in zip(
                                &ui[..ui.len() - 1],
                                ux.subslice(0..ui.len() - 1).into_ref_iter(),
                            ) {
                                let i = i.zx();
                                let u = if conj == Conj::Yes {
                                    u.read().faer_conj()
                                } else {
                                    u.read()
                                };

                                x.write(i, k0, x.read(i, k0).faer_sub(E::faer_mul(u, xj0)));
                                x.write(i, k1, x.read(i, k1).faer_sub(E::faer_mul(u, xj1)));
                                x.write(i, k2, x.read(i, k2).faer_sub(E::faer_mul(u, xj2)));
                                x.write(i, k3, x.read(i, k3).faer_sub(E::faer_mul(u, xj3)));
                            }
                        }
                    }
                    _ => unreachable!(),
                }
                k += bs;
            }
        },
    );
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

    crate::utils::constrained::Size::with2(
        rhs.nrows(),
        rhs.ncols(),
        #[inline(always)]
        |N, K| {
            let mut x = crate::utils::constrained::mat::MatMut::new(rhs, N, K);
            let u = crate::utils::constrained::sparse::SparseColMatRef::new(u, N, N);

            let mut k = 0usize;
            while k < *K {
                let bs = Ord::min(*K - k, 4);
                match bs {
                    1 => {
                        let k0 = K.check(k);
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
                            let (values_head, values_tail) =
                                ux.subslice(0..ui.len() - 1).into_chunks_exact(4);

                            for (i, uij) in zip(rows_head, values_head) {
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
                                acc0a = acc0a.faer_add(uija.faer_mul(x.read(i[a].zx(), k0)));
                                acc0b = acc0b.faer_add(uijb.faer_mul(x.read(i[b].zx(), k0)));
                                acc0c = acc0c.faer_add(uijc.faer_mul(x.read(i[c].zx(), k0)));
                                acc0d = acc0d.faer_add(uijd.faer_mul(x.read(i[d].zx(), k0)));
                            }

                            for (i, uij) in zip(rows_tail, values_tail.into_ref_iter()) {
                                let uija = uij.read();
                                let uija = if conj == Conj::Yes {
                                    uija.faer_conj()
                                } else {
                                    uija
                                };
                                acc0a = acc0a.faer_add(uija.faer_mul(x.read(i.zx(), k0)));
                            }

                            x.write(
                                j,
                                k0,
                                x.read(j, k0).faer_sub(
                                    acc0a.faer_add(acc0b).faer_add(acc0c.faer_add(acc0d)),
                                ),
                            );
                        }
                    }
                    2 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);

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
                            let (values_head, values_tail) =
                                ux.subslice(0..ui.len() - 1).into_chunks_exact(4);

                            for (i, uij) in zip(rows_head, values_head) {
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
                                acc0a = acc0a.faer_add(uija.faer_mul(x.read(i[a].zx(), k0)));
                                acc0b = acc0b.faer_add(uijb.faer_mul(x.read(i[b].zx(), k0)));
                                acc1a = acc1a.faer_add(uija.faer_mul(x.read(i[a].zx(), k1)));
                                acc1b = acc1b.faer_add(uijb.faer_mul(x.read(i[b].zx(), k1)));
                            }

                            for (i, uij) in zip(rows_tail, values_tail.into_ref_iter()) {
                                let uija = uij.read();
                                let uija = if conj == Conj::Yes {
                                    uija.faer_conj()
                                } else {
                                    uija
                                };
                                acc0a = acc0a.faer_add(uija.faer_mul(x.read(i.zx(), k0)));
                                acc1a = acc1a.faer_add(uija.faer_mul(x.read(i.zx(), k1)));
                            }

                            x.write(j, k0, x.read(j, k0).faer_sub(acc0a.faer_add(acc0b)));
                            x.write(j, k1, x.read(j, k1).faer_sub(acc1a.faer_add(acc1b)));
                        }
                    }
                    3 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);
                        let k2 = K.check(k + 2);

                        for j in N.indices() {
                            let ui = u.row_indices_of_col_raw(j);
                            let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

                            let mut acc0a = E::faer_zero();
                            let mut acc1a = E::faer_zero();
                            let mut acc2a = E::faer_zero();

                            let a = 0;

                            let rows_head = ui[..ui.len() - 1].chunks_exact(4);
                            let rows_tail = rows_head.remainder();
                            let (values_head, values_tail) =
                                ux.subslice(0..ui.len() - 1).into_chunks_exact(4);

                            for (i, uij) in zip(rows_head, values_head) {
                                let uija = uij.read(a);
                                let uija = if conj == Conj::Yes {
                                    uija.faer_conj()
                                } else {
                                    uija
                                };
                                acc0a = acc0a.faer_add(uija.faer_mul(x.read(i[a].zx(), k0)));
                                acc1a = acc1a.faer_add(uija.faer_mul(x.read(i[a].zx(), k1)));
                                acc2a = acc2a.faer_add(uija.faer_mul(x.read(i[a].zx(), k2)));
                            }

                            for (i, uij) in zip(rows_tail, values_tail.into_ref_iter()) {
                                let uija = uij.read();
                                let uija = if conj == Conj::Yes {
                                    uija.faer_conj()
                                } else {
                                    uija
                                };
                                acc0a = acc0a.faer_add(uija.faer_mul(x.read(i.zx(), k0)));
                                acc1a = acc1a.faer_add(uija.faer_mul(x.read(i.zx(), k1)));
                                acc2a = acc2a.faer_add(uija.faer_mul(x.read(i.zx(), k2)));
                            }

                            x.write(j, k0, x.read(j, k0).faer_sub(acc0a));
                            x.write(j, k1, x.read(j, k1).faer_sub(acc1a));
                            x.write(j, k2, x.read(j, k2).faer_sub(acc2a));
                        }
                    }
                    4 => {
                        let k0 = K.check(k);
                        let k1 = K.check(k + 1);
                        let k2 = K.check(k + 2);
                        let k3 = K.check(k + 3);

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
                            let (values_head, values_tail) =
                                ux.subslice(0..ui.len() - 1).into_chunks_exact(4);

                            for (i, uij) in zip(rows_head, values_head) {
                                let uija = uij.read(a);
                                let uija = if conj == Conj::Yes {
                                    uija.faer_conj()
                                } else {
                                    uija
                                };
                                acc0a = acc0a.faer_add(uija.faer_mul(x.read(i[a].zx(), k0)));
                                acc1a = acc1a.faer_add(uija.faer_mul(x.read(i[a].zx(), k1)));
                                acc2a = acc2a.faer_add(uija.faer_mul(x.read(i[a].zx(), k2)));
                                acc3a = acc3a.faer_add(uija.faer_mul(x.read(i[a].zx(), k3)));
                            }

                            for (i, uij) in zip(rows_tail, values_tail.into_ref_iter()) {
                                let uija = uij.read();
                                let uija = if conj == Conj::Yes {
                                    uija.faer_conj()
                                } else {
                                    uija
                                };
                                acc0a = acc0a.faer_add(uija.faer_mul(x.read(i.zx(), k0)));
                                acc1a = acc1a.faer_add(uija.faer_mul(x.read(i.zx(), k1)));
                                acc2a = acc2a.faer_add(uija.faer_mul(x.read(i.zx(), k2)));
                                acc3a = acc3a.faer_add(uija.faer_mul(x.read(i.zx(), k3)));
                            }

                            x.write(j, k0, x.read(j, k0).faer_sub(acc0a));
                            x.write(j, k1, x.read(j, k1).faer_sub(acc1a));
                            x.write(j, k2, x.read(j, k2).faer_sub(acc2a));
                            x.write(j, k3, x.read(j, k3).faer_sub(acc3a));
                        }
                    }
                    _ => unreachable!(),
                }
                k += bs;
            }
        },
    );
}
