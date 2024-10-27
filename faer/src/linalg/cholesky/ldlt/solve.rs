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
    LD_factors: MatRef<'_, T, Dim<'N>, Dim<'N>>,
    conj_lhs: Conj,
    rhs: MatMut<'_, T, Dim<'N>, Dim<'K>>,
    par: Par,
    stack: &mut DynStack,
) {
    _ = stack;

    let N = rhs.nrows();
    let K = rhs.ncols();
    let mut rhs = rhs;
    linalg::triangular_solve::solve_unit_lower_triangular_in_place_with_conj(
        LD_factors,
        conj_lhs,
        rhs.rb_mut(),
        par,
    );

    for j in K.indices() {
        for i in N.indices() {
            let d = recip(real(LD_factors[(i, i)]));
            rhs[(i, j)] = mul_real(rhs[(i, j)], d);
        }
    }

    linalg::triangular_solve::solve_unit_upper_triangular_in_place_with_conj(
        LD_factors.transpose(),
        conj_lhs.compose(Conj::Yes),
        rhs.rb_mut(),
        par,
    );
}
