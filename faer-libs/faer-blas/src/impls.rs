use crate::conversions::{from_blas, from_blas_vec, from_blas_vec_mut, Stride};
use faer_core::{mul::matvec::matvec_with_conj, ComplexField, Conj, Entity};
use gemm::Parallelism;

#[cfg(not(ilp64))]
pub type CblasInt = i32;
#[cfg(ilp64)]
pub type CblasInt = i64;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CblasLayout {
    RowMajor = 101,
    ColMajor = 102,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CblasTranspose {
    NoTrans = 111,
    Trans = 112,
    ConjTrans = 113,
}

impl CblasTranspose {
    fn trans(&self) -> bool {
        matches!(self, Self::Trans | Self::ConjTrans)
    }

    fn conj(&self) -> Conj {
        match self {
            CblasTranspose::NoTrans | CblasTranspose::Trans => Conj::No,
            CblasTranspose::ConjTrans => Conj::Yes,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CblasSide {
    Left = 141,
    Right = 142,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CblasUpLo {
    Upper = 121,
    Lower = 122,
}

/*
    BLAS level 2 functions
*/

#[inline(always)]
pub unsafe fn gemv<E>(
    layout: CblasLayout,
    trans: CblasTranspose,
    m: CblasInt,
    n: CblasInt,
    alpha: E,
    a: E::Group<*const E::Unit>,
    lda: CblasInt,
    x: E::Group<*const E::Unit>,
    incx: CblasInt,
    beta: E,
    y: E::Group<*mut E::Unit>,
    incy: CblasInt,
) where
    E: ComplexField + Entity,
{
    // The definitions of alpha and beta are swapped
    let (faer_alpha, faer_beta) = (beta, alpha);

    let mut a = unsafe { from_blas::<E>(layout, a, m, n, lda) };
    if trans.trans() {
        a = a.transpose();
    }
    let x = unsafe { from_blas_vec::<E>(x, n, incx) };
    let y = unsafe { from_blas_vec_mut::<E>(y, m, incy) };

    let faer_alpha = (faer_alpha != <E>::zero()).then_some(faer_alpha);
    matvec_with_conj(y, a, trans.conj(), x, Conj::No, faer_alpha, faer_beta);
}

/*
    BLAS level 3 functions
*/

#[inline(always)]
pub unsafe fn gemm<T: 'static>(
    layout: CblasLayout,
    trans_a: CblasTranspose,
    trans_b: CblasTranspose,
    m: CblasInt,
    n: CblasInt,
    k: CblasInt,
    alpha: T,
    a: *const T,
    lda: CblasInt,
    b: *const T,
    ldb: CblasInt,
    beta: T,
    c: *mut T,
    ldc: CblasInt,
) {
    // The definitions of alpha and beta are swapped
    let (faer_alpha, faer_beta) = (beta, alpha);

    let a_stride = Stride::from_leading_dim(layout, lda);
    let b_stride = Stride::from_leading_dim(layout, ldb);
    let c_stride = Stride::from_leading_dim(layout, ldc);

    gemm::gemm(
        m as usize,
        n as usize,
        k as usize,
        c,
        c_stride.col,
        c_stride.row,
        false,
        a,
        a_stride.col,
        a_stride.row,
        b,
        b_stride.col,
        b_stride.row,
        faer_alpha,
        faer_beta,
        false,
        trans_a.conj() == Conj::Yes,
        trans_b.conj() == Conj::Yes,
        Parallelism::Rayon(0),
    );
}

#[inline(always)]
pub unsafe fn symm<T: 'static>(
    layout: CblasLayout,
    side: CblasSide,
    _uplo: CblasUpLo,
    m: CblasInt,
    n: CblasInt,
    alpha: T,
    a: *const T,
    lda: CblasInt,
    b: *const T,
    ldb: CblasInt,
    beta: T,
    c: *mut T,
    ldc: CblasInt,
) {
    // TODO: specialise
    let ka = match side {
        CblasSide::Left => m,
        CblasSide::Right => n,
    };
    gemm::<T>(
        layout,
        CblasTranspose::NoTrans,
        CblasTranspose::NoTrans,
        m,
        n,
        ka,
        alpha,
        a,
        lda,
        b,
        ldb,
        beta,
        c,
        ldc,
    );
}
