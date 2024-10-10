use crate::internal_prelude::*;

const LINEAR_IMPL_THRESHOLD: usize = 128;

fn real_imag<C: RealContainer, T: RealField<C>>(
    mat: MatRef<num_complex::Complex<C>, T, usize, usize, ContiguousFwd>,
) -> (
    MatRef<C, T, usize, usize, ContiguousFwd>,
    MatRef<C, T, usize, usize, ContiguousFwd>,
) {
    unsafe {
        (
            MatRef::from_raw_parts(
                mat.as_ptr().re,
                mat.nrows(),
                mat.ncols(),
                mat.row_stride(),
                mat.col_stride(),
            ),
            MatRef::from_raw_parts(
                mat.as_ptr().im,
                mat.nrows(),
                mat.ncols(),
                mat.row_stride(),
                mat.col_stride(),
            ),
        )
    }
}

pub mod norm_l1;
pub mod norm_l2;
pub mod norm_l2_sqr;
pub mod norm_max;
pub mod sum;
