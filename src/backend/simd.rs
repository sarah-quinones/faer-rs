pub trait Simd {
    const F32_IN_REG: usize;
    const F64_IN_REG: usize;
    type RegF32;
    type RegF64;

    unsafe fn add_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32;
    unsafe fn sub_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32;
    unsafe fn mul_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32;
    unsafe fn div_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32;

    unsafe fn add_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64;
    unsafe fn sub_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64;
    unsafe fn mul_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64;
    unsafe fn div_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64;

    #[inline]
    unsafe fn fma_f32(a: Self::RegF32, b: Self::RegF32, c: Self::RegF32) -> Self::RegF32 {
        Self::add_f32(Self::mul_f32(a, b), c)
    }
    #[inline]
    unsafe fn fma_f64(a: Self::RegF64, b: Self::RegF64, c: Self::RegF64) -> Self::RegF64 {
        Self::add_f64(Self::mul_f64(a, b), c)
    }
}

pub struct Scalar;
pub struct ScalarFma;

#[rustfmt::skip]
impl Simd for Scalar {
    const F32_IN_REG: usize = 1;
    const F64_IN_REG: usize = 1;

    type RegF32 = f32;
    type RegF64 = f64;

    #[inline] unsafe fn add_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32 { a + b }
    #[inline] unsafe fn sub_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32 { a - b }
    #[inline] unsafe fn mul_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32 { a * b }
    #[inline] unsafe fn div_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32 { a / b }

    #[inline] unsafe fn add_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64 { a + b }
    #[inline] unsafe fn sub_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64 { a - b }
    #[inline] unsafe fn mul_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64 { a * b }
    #[inline] unsafe fn div_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64 { a / b }
}

#[rustfmt::skip]
impl Simd for ScalarFma {
    const F32_IN_REG: usize = 1;
    const F64_IN_REG: usize = 1;

    type RegF32 = f32;
    type RegF64 = f64;

    #[inline] unsafe fn add_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32 { a + b }
    #[inline] unsafe fn sub_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32 { a - b }
    #[inline] unsafe fn mul_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32 { a * b }
    #[inline] unsafe fn div_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32 { a / b }

    #[inline] unsafe fn add_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64 { a + b }
    #[inline] unsafe fn sub_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64 { a - b }
    #[inline] unsafe fn mul_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64 { a * b }
    #[inline] unsafe fn div_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64 { a / b }

    #[inline] unsafe fn fma_f32(a: Self::RegF32, b: Self::RegF32, c: Self::RegF32) -> Self::RegF32 { f32::mul_add(a, b, c) }
    #[inline] unsafe fn fma_f64(a: Self::RegF64, b: Self::RegF64, c: Self::RegF64) -> Self::RegF64 { f64::mul_add(a, b, c) }

}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86 {
    use super::Simd;

    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    pub struct Sse2;
    pub struct Avx;
    pub struct Fma;
    #[cfg(feature = "nightly")]
    pub struct Avx512f;

    #[rustfmt::skip]
    impl Simd for Sse2 {
        const F32_IN_REG: usize = 4;
        const F64_IN_REG: usize = 2;

        type RegF32 = __m128;
        type RegF64 = __m128d;

        #[inline] unsafe fn add_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32 { _mm_add_ps(a, b) }
        #[inline] unsafe fn sub_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32 { _mm_sub_ps(a, b) }
        #[inline] unsafe fn mul_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32 { _mm_mul_ps(a, b) }
        #[inline] unsafe fn div_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32 { _mm_div_ps(a, b) }

        #[inline] unsafe fn add_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64 { _mm_add_pd(a, b) }
        #[inline] unsafe fn sub_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64 { _mm_sub_pd(a, b) }
        #[inline] unsafe fn mul_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64 { _mm_mul_pd(a, b) }
        #[inline] unsafe fn div_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64 { _mm_div_pd(a, b) }
    }

    #[rustfmt::skip]
    impl Simd for Avx {
        const F32_IN_REG: usize = 8;
        const F64_IN_REG: usize = 4;

        type RegF32 = __m256;
        type RegF64 = __m256d;

        #[inline] unsafe fn add_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32 { _mm256_add_ps(a, b) }
        #[inline] unsafe fn sub_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32 { _mm256_sub_ps(a, b) }
        #[inline] unsafe fn mul_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32 { _mm256_mul_ps(a, b) }
        #[inline] unsafe fn div_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32 { _mm256_div_ps(a, b) }

        #[inline] unsafe fn add_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64 { _mm256_add_pd(a, b) }
        #[inline] unsafe fn sub_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64 { _mm256_sub_pd(a, b) }
        #[inline] unsafe fn mul_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64 { _mm256_mul_pd(a, b) }
        #[inline] unsafe fn div_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64 { _mm256_div_pd(a, b) }
    }

    #[rustfmt::skip]
    impl Simd for Fma {
        const F32_IN_REG: usize = 8;
        const F64_IN_REG: usize = 4;

        type RegF32 = __m256;
        type RegF64 = __m256d;

        #[inline] unsafe fn add_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32 { _mm256_add_ps(a, b) }
        #[inline] unsafe fn sub_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32 { _mm256_sub_ps(a, b) }
        #[inline] unsafe fn mul_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32 { _mm256_mul_ps(a, b) }
        #[inline] unsafe fn div_f32(a: Self::RegF32, b: Self::RegF32) -> Self::RegF32 { _mm256_div_ps(a, b) }

        #[inline] unsafe fn add_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64 { _mm256_add_pd(a, b) }
        #[inline] unsafe fn sub_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64 { _mm256_sub_pd(a, b) }
        #[inline] unsafe fn mul_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64 { _mm256_mul_pd(a, b) }
        #[inline] unsafe fn div_f64(a: Self::RegF64, b: Self::RegF64) -> Self::RegF64 { _mm256_div_pd(a, b) }

        #[inline] unsafe fn fma_f32(a: Self::RegF32, b: Self::RegF32, c: Self::RegF32) -> Self::RegF32 { _mm256_fmadd_ps(a, b, c) }
        #[inline] unsafe fn fma_f64(a: Self::RegF64, b: Self::RegF64, c: Self::RegF64) -> Self::RegF64 { _mm256_fmadd_pd(a, b, c) }
    }
}
