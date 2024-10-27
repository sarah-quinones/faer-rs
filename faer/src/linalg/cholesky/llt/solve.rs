use crate::internal_prelude::*;

pub fn solve_in_place_scratch<T: ComplexField>(
    dim: usize,
    rhs_ncols: usize,
    par: Par,
) -> Result<StackReq, SizeOverflow> {
    _ = (dim, rhs_ncols, par);
    Ok(StackReq::empty())
}

#[math]
pub fn solve_in_place_with_conj<'N, 'K, T: ComplexField>(
    L: MatRef<'_, T, Dim<'N>, Dim<'N>>,
    conj_lhs: Conj,
    rhs: MatMut<'_, T, Dim<'N>, Dim<'K>>,
    par: Par,
    stack: &mut DynStack,
) {
    _ = stack;
    let mut rhs = rhs;
    linalg::triangular_solve::solve_lower_triangular_in_place_with_conj(
        L,
        conj_lhs,
        rhs.rb_mut(),
        par,
    );

    linalg::triangular_solve::solve_upper_triangular_in_place_with_conj(
        L.transpose(),
        conj_lhs.compose(Conj::Yes),
        rhs.rb_mut(),
        par,
    );
}
