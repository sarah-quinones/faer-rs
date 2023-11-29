use core::iter::zip;
use faer_core::{
    assert, group_helpers::SliceGroup, permutation::Index, sparse::SparseColMatRef, Conj, MatMut,
    Parallelism,
};
use faer_entity::ComplexField;
use reborrow::*;

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

    let n = l.ncols();
    let mut x = rhs;
    for mut x in x.rb_mut().col_chunks_mut(4) {
        match x.ncols() {
            1 => {
                for j in 0..n {
                    let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                    let d = if conj == Conj::Yes { d.faer_conj() } else { d };

                    let xj0 = x.read(j, 0).faer_mul(d);
                    x.write(j, 0, xj0);

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
                        x.write(i, 0, x.read(i, 0).faer_sub(lij.faer_mul(xj0)));
                    }
                }
            }
            2 => {
                for j in 0..n {
                    let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                    let d = if conj == Conj::Yes { d.faer_conj() } else { d };

                    let xj0 = x.read(j, 0).faer_mul(d);
                    x.write(j, 0, xj0);
                    let xj1 = x.read(j, 1).faer_mul(d);
                    x.write(j, 1, xj1);

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
                        x.write(i, 0, x.read(i, 0).faer_sub(lij.faer_mul(xj0)));
                        x.write(i, 1, x.read(i, 1).faer_sub(lij.faer_mul(xj1)));
                    }
                }
            }
            3 => {
                for j in 0..n {
                    let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                    let d = if conj == Conj::Yes { d.faer_conj() } else { d };

                    let xj0 = x.read(j, 0).faer_mul(d);
                    x.write(j, 0, xj0);
                    let xj1 = x.read(j, 1).faer_mul(d);
                    x.write(j, 1, xj1);
                    let xj2 = x.read(j, 2).faer_mul(d);
                    x.write(j, 2, xj2);

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
                        x.write(i, 0, x.read(i, 0).faer_sub(lij.faer_mul(xj0)));
                        x.write(i, 1, x.read(i, 1).faer_sub(lij.faer_mul(xj1)));
                        x.write(i, 2, x.read(i, 2).faer_sub(lij.faer_mul(xj2)));
                    }
                }
            }
            4 => {
                for j in 0..n {
                    let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                    let d = if conj == Conj::Yes { d.faer_conj() } else { d };

                    let xj0 = x.read(j, 0).faer_mul(d);
                    x.write(j, 0, xj0);
                    let xj1 = x.read(j, 1).faer_mul(d);
                    x.write(j, 1, xj1);
                    let xj2 = x.read(j, 2).faer_mul(d);
                    x.write(j, 2, xj2);
                    let xj3 = x.read(j, 3).faer_mul(d);
                    x.write(j, 3, xj3);

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
                        x.write(i, 0, x.read(i, 0).faer_sub(lij.faer_mul(xj0)));
                        x.write(i, 1, x.read(i, 1).faer_sub(lij.faer_mul(xj1)));
                        x.write(i, 2, x.read(i, 2).faer_sub(lij.faer_mul(xj2)));
                        x.write(i, 3, x.read(i, 3).faer_sub(lij.faer_mul(xj3)));
                    }
                }
            }
            _ => unreachable!(),
        }
    }
}

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

    let n = l.ncols();
    let mut x = rhs;
    for mut x in x.rb_mut().col_chunks_mut(4) {
        match x.ncols() {
            1 => {
                for j in (0..n).rev() {
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
                        acc0a = acc0a.faer_add(lija.faer_mul(x.read(i[a].zx(), 0)));
                        acc0b = acc0b.faer_add(lijb.faer_mul(x.read(i[b].zx(), 0)));
                        acc0c = acc0c.faer_add(lijc.faer_mul(x.read(i[c].zx(), 0)));
                        acc0d = acc0d.faer_add(lijd.faer_mul(x.read(i[d].zx(), 0)));
                    }

                    for (i, lij) in zip(rows_tail, values_tail.into_ref_iter()) {
                        let lija = lij.read();
                        let lija = if conj == Conj::Yes {
                            lija.faer_conj()
                        } else {
                            lija
                        };
                        acc0a = acc0a.faer_add(lija.faer_mul(x.read(i.zx(), 0)));
                    }

                    let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                    let d = if conj == Conj::Yes { d.faer_conj() } else { d };
                    x.write(
                        j,
                        0,
                        x.read(j, 0)
                            .faer_sub(acc0a.faer_add(acc0b).faer_add(acc0c.faer_add(acc0d)))
                            .faer_mul(d),
                    );
                }
            }
            2 => {
                for j in (0..n).rev() {
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
                        acc0a = acc0a.faer_add(lija.faer_mul(x.read(i[a].zx(), 0)));
                        acc0b = acc0b.faer_add(lijb.faer_mul(x.read(i[b].zx(), 0)));
                        acc1a = acc1a.faer_add(lija.faer_mul(x.read(i[a].zx(), 1)));
                        acc1b = acc1b.faer_add(lijb.faer_mul(x.read(i[b].zx(), 1)));
                    }

                    for (i, lij) in zip(rows_tail, values_tail.into_ref_iter()) {
                        let lija = lij.read();
                        let lija = if conj == Conj::Yes {
                            lija.faer_conj()
                        } else {
                            lija
                        };
                        acc0a = acc0a.faer_add(lija.faer_mul(x.read(i.zx(), 0)));
                        acc1a = acc1a.faer_add(lija.faer_mul(x.read(i.zx(), 1)));
                    }

                    let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                    let d = if conj == Conj::Yes { d.faer_conj() } else { d };
                    x.write(
                        j,
                        0,
                        x.read(j, 0).faer_sub(acc0a.faer_add(acc0b)).faer_mul(d),
                    );
                    x.write(
                        j,
                        1,
                        x.read(j, 1).faer_sub(acc1a.faer_add(acc1b)).faer_mul(d),
                    );
                }
            }
            3 => {
                for j in (0..n).rev() {
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
                        acc0a = acc0a.faer_add(lij.faer_mul(x.read(i, 0)));
                        acc1a = acc1a.faer_add(lij.faer_mul(x.read(i, 1)));
                        acc2a = acc2a.faer_add(lij.faer_mul(x.read(i, 2)));
                    }

                    let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                    let d = if conj == Conj::Yes { d.faer_conj() } else { d };
                    x.write(j, 0, x.read(j, 0).faer_sub(acc0a).faer_mul(d));
                    x.write(j, 1, x.read(j, 1).faer_sub(acc1a).faer_mul(d));
                    x.write(j, 2, x.read(j, 2).faer_sub(acc2a).faer_mul(d));
                }
            }
            4 => {
                for j in (0..n).rev() {
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
                        acc0a = acc0a.faer_add(lij.faer_mul(x.read(i, 0)));
                        acc1a = acc1a.faer_add(lij.faer_mul(x.read(i, 1)));
                        acc2a = acc2a.faer_add(lij.faer_mul(x.read(i, 2)));
                        acc3a = acc3a.faer_add(lij.faer_mul(x.read(i, 3)));
                    }

                    let d = slice_group(l.values_of_col(j)).read(0).faer_inv();
                    let d = if conj == Conj::Yes { d.faer_conj() } else { d };
                    x.write(j, 0, x.read(j, 0).faer_sub(acc0a).faer_mul(d));
                    x.write(j, 1, x.read(j, 1).faer_sub(acc1a).faer_mul(d));
                    x.write(j, 2, x.read(j, 2).faer_sub(acc2a).faer_mul(d));
                    x.write(j, 3, x.read(j, 3).faer_sub(acc3a).faer_mul(d));
                }
            }
            _ => unreachable!(),
        }
    }
}

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

    let n = l.ncols();
    let mut x = rhs;
    for mut x in x.rb_mut().col_chunks_mut(4) {
        match x.ncols() {
            1 => {
                for j in 0..n {
                    let xj0 = x.read(j, 0);

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
                        x.write(i, 0, x.read(i, 0).faer_sub(lij.faer_mul(xj0)));
                    }
                }
            }
            2 => {
                for j in 0..n {
                    let xj0 = x.read(j, 0);
                    let xj1 = x.read(j, 1);

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
                        x.write(i, 0, x.read(i, 0).faer_sub(lij.faer_mul(xj0)));
                        x.write(i, 1, x.read(i, 1).faer_sub(lij.faer_mul(xj1)));
                    }
                }
            }
            3 => {
                for j in 0..n {
                    let xj0 = x.read(j, 0);
                    let xj1 = x.read(j, 1);
                    let xj2 = x.read(j, 2);

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
                        x.write(i, 0, x.read(i, 0).faer_sub(lij.faer_mul(xj0)));
                        x.write(i, 1, x.read(i, 1).faer_sub(lij.faer_mul(xj1)));
                        x.write(i, 2, x.read(i, 2).faer_sub(lij.faer_mul(xj2)));
                    }
                }
            }
            4 => {
                for j in 0..n {
                    let xj0 = x.read(j, 0);
                    let xj1 = x.read(j, 1);
                    let xj2 = x.read(j, 2);
                    let xj3 = x.read(j, 3);

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
                        x.write(i, 0, x.read(i, 0).faer_sub(lij.faer_mul(xj0)));
                        x.write(i, 1, x.read(i, 1).faer_sub(lij.faer_mul(xj1)));
                        x.write(i, 2, x.read(i, 2).faer_sub(lij.faer_mul(xj2)));
                        x.write(i, 3, x.read(i, 3).faer_sub(lij.faer_mul(xj3)));
                    }
                }
            }
            _ => unreachable!(),
        }
    }
}

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

    let n = l.ncols();
    let mut x = rhs;
    for mut x in x.rb_mut().col_chunks_mut(4) {
        match x.ncols() {
            1 => {
                for j in (0..n).rev() {
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
                        acc0a = acc0a.faer_add(lija.faer_mul(x.read(i[a].zx(), 0)));
                        acc0b = acc0b.faer_add(lijb.faer_mul(x.read(i[b].zx(), 0)));
                        acc0c = acc0c.faer_add(lijc.faer_mul(x.read(i[c].zx(), 0)));
                        acc0d = acc0d.faer_add(lijd.faer_mul(x.read(i[d].zx(), 0)));
                    }

                    for (i, lij) in zip(rows_tail, values_tail.into_ref_iter()) {
                        let lija = lij.read();
                        let lija = if conj == Conj::Yes {
                            lija.faer_conj()
                        } else {
                            lija
                        };
                        acc0a = acc0a.faer_add(lija.faer_mul(x.read(i.zx(), 0)));
                    }

                    x.write(
                        j,
                        0,
                        x.read(j, 0)
                            .faer_sub(acc0a.faer_add(acc0b).faer_add(acc0c.faer_add(acc0d))),
                    );
                }
            }
            2 => {
                for j in (0..n).rev() {
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
                        acc0a = acc0a.faer_add(lija.faer_mul(x.read(i[a].zx(), 0)));
                        acc0b = acc0b.faer_add(lijb.faer_mul(x.read(i[b].zx(), 0)));
                        acc1a = acc1a.faer_add(lija.faer_mul(x.read(i[a].zx(), 1)));
                        acc1b = acc1b.faer_add(lijb.faer_mul(x.read(i[b].zx(), 1)));
                    }

                    for (i, lij) in zip(rows_tail, values_tail.into_ref_iter()) {
                        let lija = lij.read();
                        let lija = if conj == Conj::Yes {
                            lija.faer_conj()
                        } else {
                            lija
                        };
                        acc0a = acc0a.faer_add(lija.faer_mul(x.read(i.zx(), 0)));
                        acc1a = acc1a.faer_add(lija.faer_mul(x.read(i.zx(), 1)));
                    }

                    x.write(j, 0, x.read(j, 0).faer_sub(acc0a.faer_add(acc0b)));
                    x.write(j, 1, x.read(j, 1).faer_sub(acc1a.faer_add(acc1b)));
                }
            }
            3 => {
                for j in (0..n).rev() {
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
                        acc0a = acc0a.faer_add(lij.faer_mul(x.read(i, 0)));
                        acc1a = acc1a.faer_add(lij.faer_mul(x.read(i, 1)));
                        acc2a = acc2a.faer_add(lij.faer_mul(x.read(i, 2)));
                    }

                    x.write(j, 0, x.read(j, 0).faer_sub(acc0a));
                    x.write(j, 1, x.read(j, 1).faer_sub(acc1a));
                    x.write(j, 2, x.read(j, 2).faer_sub(acc2a));
                }
            }
            4 => {
                for j in (0..n).rev() {
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
                        acc0a = acc0a.faer_add(lij.faer_mul(x.read(i, 0)));
                        acc1a = acc1a.faer_add(lij.faer_mul(x.read(i, 1)));
                        acc2a = acc2a.faer_add(lij.faer_mul(x.read(i, 2)));
                        acc3a = acc3a.faer_add(lij.faer_mul(x.read(i, 3)));
                    }

                    x.write(j, 0, x.read(j, 0).faer_sub(acc0a));
                    x.write(j, 1, x.read(j, 1).faer_sub(acc1a));
                    x.write(j, 2, x.read(j, 2).faer_sub(acc2a));
                    x.write(j, 3, x.read(j, 3).faer_sub(acc3a));
                }
            }
            _ => unreachable!(),
        }
    }
}

pub fn solve_upper_triangular_in_place<I: Index, E: ComplexField>(
    u: SparseColMatRef<'_, I, E>,
    conj: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    let _ = parallelism;
    assert!(u.nrows() == u.ncols());
    assert!(rhs.nrows() == u.nrows());

    let n = u.ncols();
    let mut x = rhs;
    for j in (0..n).rev() {
        let ui = u.row_indices_of_col_raw(j);
        let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

        let u_inv = ux.read(ui.len() - 1).faer_inv();
        let u_inv = if conj == Conj::Yes {
            u_inv.faer_conj()
        } else {
            u_inv
        };
        for k in 0..x.ncols() {
            let xj = x.read(j, k).faer_mul(u_inv);
            x.write(j, k, xj);

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

                x.write(i, k, x.read(i, k).faer_sub(E::faer_mul(u, xj)));
            }
        }
    }
}

pub fn solve_upper_triangular_transpose_in_place<I: Index, E: ComplexField>(
    u: SparseColMatRef<'_, I, E>,
    conj: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    let _ = parallelism;
    assert!(u.nrows() == u.ncols());
    assert!(rhs.nrows() == u.nrows());

    let n = u.ncols();
    let mut x = rhs;
    for j in 0..n {
        let ui = u.row_indices_of_col_raw(j);
        let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

        let u_inv = ux.read(ui.len() - 1).faer_inv();
        let u_inv = if conj == Conj::Yes {
            u_inv.faer_conj()
        } else {
            u_inv
        };
        for k in 0..x.ncols() {
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
                acc0a = acc0a.faer_add(uija.faer_mul(x.read(i[a].zx(), k)));
                acc0b = acc0b.faer_add(uijb.faer_mul(x.read(i[b].zx(), k)));
                acc0c = acc0c.faer_add(uijc.faer_mul(x.read(i[c].zx(), k)));
                acc0d = acc0d.faer_add(uijd.faer_mul(x.read(i[d].zx(), k)));
            }

            for (i, uij) in zip(rows_tail, values_tail.into_ref_iter()) {
                let uija = uij.read();
                let uija = if conj == Conj::Yes {
                    uija.faer_conj()
                } else {
                    uija
                };
                acc0a = acc0a.faer_add(uija.faer_mul(x.read(i.zx(), k)));
            }

            x.write(
                j,
                k,
                x.read(j, k)
                    .faer_sub(acc0a.faer_add(acc0b).faer_add(acc0c.faer_add(acc0d)))
                    .faer_mul(u_inv),
            );
        }
    }
}

pub fn solve_unit_upper_triangular_in_place<I: Index, E: ComplexField>(
    u: SparseColMatRef<'_, I, E>,
    conj: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    let _ = parallelism;
    assert!(u.nrows() == u.ncols());
    assert!(rhs.nrows() == u.nrows());

    let n = u.ncols();
    let mut x = rhs;
    for j in (0..n).rev() {
        let ui = u.row_indices_of_col_raw(j);
        let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

        for k in 0..x.ncols() {
            let xj = x.read(j, k);

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

                x.write(i, k, x.read(i, k).faer_sub(E::faer_mul(u, xj)));
            }
        }
    }
}

pub fn solve_unit_upper_triangular_transpose_in_place<I: Index, E: ComplexField>(
    u: SparseColMatRef<'_, I, E>,
    conj: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
) {
    let _ = parallelism;
    assert!(u.nrows() == u.ncols());
    assert!(rhs.nrows() == u.nrows());

    let n = u.ncols();
    let mut x = rhs;
    for j in 0..n {
        let ui = u.row_indices_of_col_raw(j);
        let ux = SliceGroup::<'_, E>::new(u.values_of_col(j));

        for k in 0..x.ncols() {
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
                acc0a = acc0a.faer_add(uija.faer_mul(x.read(i[a].zx(), k)));
                acc0b = acc0b.faer_add(uijb.faer_mul(x.read(i[b].zx(), k)));
                acc0c = acc0c.faer_add(uijc.faer_mul(x.read(i[c].zx(), k)));
                acc0d = acc0d.faer_add(uijd.faer_mul(x.read(i[d].zx(), k)));
            }

            for (i, uij) in zip(rows_tail, values_tail.into_ref_iter()) {
                let uija = uij.read();
                let uija = if conj == Conj::Yes {
                    uija.faer_conj()
                } else {
                    uija
                };
                acc0a = acc0a.faer_add(uija.faer_mul(x.read(i.zx(), k)));
            }

            x.write(
                j,
                k,
                x.read(j, k)
                    .faer_sub(acc0a.faer_add(acc0b).faer_add(acc0c.faer_add(acc0d))),
            );
        }
    }
}
