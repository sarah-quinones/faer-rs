use faer_core::{Entity, MatMut, MatRef};

use crate::impls::{CblasInt, CblasLayout};

#[inline(always)]
pub unsafe fn from_blas<'a, E: Entity>(
    layout: CblasLayout,
    ptr: E::Group<*const E::Unit>,
    nrows: CblasInt,
    ncols: CblasInt,
    leading_dim: CblasInt,
) -> MatRef<'a, E> {
    let stride = Stride::from_leading_dim(layout, leading_dim);
    MatRef::<E>::from_raw_parts(
        ptr,
        nrows as usize,
        ncols as usize,
        stride.row as isize,
        stride.col as isize,
    )
}

#[inline(always)]
pub unsafe fn from_blas_mut<'a, E: Entity>(
    layout: CblasLayout,
    ptr: E::Group<*mut E::Unit>,
    nrows: CblasInt,
    ncols: CblasInt,
    leading_dim: CblasInt,
) -> MatMut<'a, E> {
    let stride = Stride::from_leading_dim(layout, leading_dim);
    MatMut::<E>::from_raw_parts(
        ptr,
        nrows as usize,
        ncols as usize,
        stride.row as isize,
        stride.col as isize,
    )
}

#[inline(always)]
pub unsafe fn from_blas_vec<'a, E: Entity>(
    ptr: E::Group<*const E::Unit>,
    n: CblasInt,
    inc: CblasInt,
) -> MatRef<'a, E> {
    MatRef::<E>::from_raw_parts(ptr, n as usize, 1, inc as isize, 0)
}

#[inline(always)]
pub unsafe fn from_blas_vec_mut<'a, E: Entity>(
    ptr: E::Group<*mut E::Unit>,
    n: CblasInt,
    inc: CblasInt,
) -> MatMut<'a, E> {
    MatMut::<E>::from_raw_parts(ptr, n as usize, 1, inc as isize, 0)
}

pub struct Stride {
    pub row: isize,
    pub col: isize,
}
impl Stride {
    #[inline(always)]
    pub fn from_leading_dim(layout: CblasLayout, leading_dim: CblasInt) -> Self {
        match layout {
            CblasLayout::RowMajor => Self {
                col: 1,
                row: leading_dim as isize,
            },
            CblasLayout::ColMajor => Self {
                col: leading_dim as isize,
                row: 1,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::impls::CblasLayout;

    use super::{from_blas, from_blas_vec};

    #[test]
    fn test_row_major() {
        /*
            | 0.11 0.12 0.13 |
            | 0.21 0.22 0.23 |
            In row major order
        */
        let m = 2;
        let n = 3;
        let lda = 3;
        let a: [f64; 6] = [0.11, 0.12, 0.13, 0.21, 0.22, 0.23];
        let result = unsafe { from_blas::<f64>(CblasLayout::RowMajor, a.as_ptr(), m, n, lda) };
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 3);
        assert_eq!(*result.get(0, 0), 0.11);
        assert_eq!(*result.get(0, 2), 0.13);
        assert_eq!(*result.get(1, 2), 0.23);
    }

    #[test]
    fn test_col_major() {
        /*
            | 0.11 0.12 0.13 |
            | 0.21 0.22 0.23 |
            In col major order
        */
        let m = 2;
        let n = 3;
        let lda = 2;
        let a: [f64; 6] = [0.11, 0.21, 0.12, 0.22, 0.13, 0.23];
        let result = unsafe { from_blas::<f64>(CblasLayout::ColMajor, a.as_ptr(), m, n, lda) };
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 3);
        assert_eq!(*result.get(0, 0), 0.11);
        assert_eq!(*result.get(0, 2), 0.13);
        assert_eq!(*result.get(1, 2), 0.23);
    }

    #[test]
    fn test_mat_excess_storage() {
        /*
            | 0.11 0.12 0.13 | 0.0 0.0
            | 0.21 0.22 0.23 | 0.0 0.0
            In row major order, where 0s are not part of the matrix
        */
        let m = 2;
        let n = 3;
        let lda = 5;
        let a: [f64; 10] = [0.11, 0.12, 0.13, 0.0, 0.0, 0.21, 0.22, 0.23, 0.0, 0.0];
        let result = unsafe { from_blas::<f64>(CblasLayout::RowMajor, a.as_ptr(), m, n, lda) };
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 3);
        assert_eq!(*result.get(0, 0), 0.11);
        assert_eq!(*result.get(0, 2), 0.13);
        assert_eq!(*result.get(1, 2), 0.23);
    }

    #[test]
    fn test_vec() {
        /*
            [ 0.1 0.2 0.3 ]
        */
        let n = 3;
        let xinc = 1;
        let x: [f64; 3] = [0.1, 0.2, 0.3];
        let result = unsafe { from_blas_vec::<f64>(x.as_ptr(), n, xinc) };
        assert_eq!(*result.get(2, 0), 0.3);

        /*
            [ 0.1 /0.0/ 0.2 /0.0/ 0.3 ]
            where 0s are excess storage
        */
        let n = 3;
        let xinc = 2;
        let x_excess: [f64; 5] = [0.1, 0.0, 0.2, 0.0, 0.3];
        let result_excess = unsafe { from_blas_vec::<f64>(x_excess.as_ptr(), n, xinc) };
        assert_eq!(result, result_excess);
    }
}
