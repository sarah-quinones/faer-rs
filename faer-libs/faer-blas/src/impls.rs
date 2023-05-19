use crate::conversions::{from_blas, from_blas_mut, from_blas_vec, from_blas_vec_mut};
use faer_core::{
    mul::{matmul_with_conj, matvec::matvec_with_conj},
    ComplexField, Conj, Entity, Parallelism,
};

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
    pub fn trans(&self) -> bool {
        matches!(self, Self::Trans | Self::ConjTrans)
    }

    pub fn conj(&self) -> Conj {
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
pub unsafe fn gemm<E>(
    layout: CblasLayout,
    trans_a: CblasTranspose,
    trans_b: CblasTranspose,
    m: CblasInt,
    n: CblasInt,
    k: CblasInt,
    alpha: E,
    a: E::Group<*const E::Unit>,
    lda: CblasInt,
    b: E::Group<*const E::Unit>,
    ldb: CblasInt,
    beta: E,
    c: E::Group<*mut E::Unit>,
    ldc: CblasInt,
) where
    E: ComplexField + Entity,
{
    // The definitions of alpha and beta are swapped
    let (faer_alpha, faer_beta) = (beta, alpha);

    // Note that m, n, k are post-op (mathematical, not storage) dimensions. trans_a(a) is m-by-k,
    // and trans_b(b) is k-by-n.

    let a = if !trans_a.trans() {
        from_blas::<E>(layout, a, m, k, lda)
    } else {
        from_blas::<E>(layout, a, k, m, lda).transpose()
    };

    let b = if !trans_b.trans() {
        from_blas::<E>(layout, b, k, n, ldb)
    } else {
        from_blas::<E>(layout, b, n, k, ldb).transpose()
    };

    let c = from_blas_mut::<E>(layout, c, m, n, ldc);

    let faer_alpha = (faer_alpha != <E>::zero()).then_some(faer_alpha);
    matmul_with_conj(
        c,
        a,
        trans_a.conj(),
        b,
        trans_b.conj(),
        faer_alpha,
        faer_beta,
        Parallelism::Rayon(0),
    );
}

#[inline(always)]
pub unsafe fn symm<E>(
    layout: CblasLayout,
    side: CblasSide,
    _uplo: CblasUpLo,
    m: CblasInt,
    n: CblasInt,
    alpha: E,
    a: E::Group<*const E::Unit>,
    lda: CblasInt,
    b: E::Group<*const E::Unit>,
    ldb: CblasInt,
    beta: E,
    c: E::Group<*mut E::Unit>,
    ldc: CblasInt,
) where
    E: ComplexField + Entity,
{
    // TODO: specialise
    let (ka, left, right) = match side {
        CblasSide::Left => (m, a, b),
        CblasSide::Right => (n, b, a),
    };
    gemm::<E>(
        layout,
        CblasTranspose::NoTrans,
        CblasTranspose::NoTrans,
        m,
        n,
        ka,
        alpha,
        left,
        lda,
        right,
        ldb,
        beta,
        c,
        ldc,
    );
}
