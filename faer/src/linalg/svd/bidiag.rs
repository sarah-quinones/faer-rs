use crate::internal_prelude::*;

pub fn bidiag_in_place_scratch<C: ComplexContainer, T: ComplexField<C>>(
    nrows: usize,
    ncols: usize,
    par: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_all_of([
        temp_mat_scratch::<C, T>(nrows, 1)?,
        temp_mat_scratch::<C, T>(nrows, par.degree())?,
        temp_mat_scratch::<C, T>(ncols, 1)?,
    ])
}

#[math]
pub fn bidiag_in_place<'M, 'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'M>, Dim<'N>>,
    H_left: RowMut<'_, C, T, Dim<'M>>,
    H_right: RowMut<'_, C, T, Dim<'N>>,
    par: Parallelism,
    stack: &mut DynStack,
) {
    let nthreads = par.degree();
    let m = A.nrows();
    let n = A.ncols();

    let (mut y, stack) = unsafe { temp_mat_uninit(ctx, n, 1, stack) };
    let (mut z, stack) = unsafe { temp_mat_uninit(ctx, m, 1, stack) };
    let (mut z_tmp, stack) = unsafe { temp_mat_uninit(ctx, m, nthreads, stack) };

    let y = y.as_mat_mut().col_mut(0);
    let z = z.as_mat_mut().col_mut(0);
    let z_tmp = z_tmp.as_mat_mut();
    let mut tl = math(zero());
    let mut tr = math(zero());
    let mut a01 = math(zero());

    let mut A = A;
    let mut Hl = H_left;
    let mut Hr = H_right;
    for k in 0..Ord::min(*m, *n) {
        let ki = m.idx(k);
        let kj = n.idx(k);

        let A = A.rb_mut();
        ghost_tree!(ROWS(TOP, I, BOT), COLS(LEFT, J, RIGHT), {
            let (rows, row_split @ list![top, _, _], rows_x) =
                m.split(list![..ki.into(), ki, ..], ROWS);
            let (_, col_split, cols_x) = n.split(list![top, kj, ..], COLS);

            let list![A0, A1, A2] = A.row_segments_mut(row_split, rows_x);
        });
    }
}
