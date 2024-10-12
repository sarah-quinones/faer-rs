use crate::internal_prelude::*;

pub fn solve_in_place_scratch<C: ComplexContainer, T: ComplexField<C>>(
    dim: usize,
    rhs_ncols: usize,
    par: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    _ = (dim, rhs_ncols, par);
    Ok(StackReq::empty())
}

#[math]
pub fn solve_in_place_with_conj<'N, 'K, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    L: MatRef<'_, C, T, Dim<'N>, Dim<'N>>,
    conj_lhs: Conj,
    rhs: MatMut<'_, C, T, Dim<'N>, Dim<'K>>,
    par: Parallelism,
    stack: &mut DynStack,
) {
    _ = stack;
    let mut rhs = rhs;
    linalg::triangular_solve::solve_lower_triangular_in_place_with_conj(
        ctx,
        L,
        conj_lhs,
        rhs.rb_mut(),
        par,
    );

    linalg::triangular_solve::solve_upper_triangular_in_place_with_conj(
        ctx,
        L.transpose(),
        conj_lhs.compose(Conj::Yes),
        rhs.rb_mut(),
        par,
    );
}
