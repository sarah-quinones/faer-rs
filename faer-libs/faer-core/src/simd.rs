pub use faer_entity::{
    one_simd_as_slice, simd_as_slice, simd_as_slice_unit, simd_index_as_slice, slice_as_mut_simd,
    slice_as_simd,
};

fn sum_i32_scalar(slice: &[i32]) -> Option<i32> {
    let mut overflow = false;
    let mut sum = 0i32;
    for &v in slice {
        let o;
        (sum, o) = i32::overflowing_add(sum, v);
        overflow |= o;
    }
    (!overflow).then_some(sum)
}
fn sum_i64_scalar(slice: &[i64]) -> Option<i64> {
    let mut overflow = false;
    let mut sum = 0i64;
    for &v in slice {
        let o;
        (sum, o) = i64::overflowing_add(sum, v);
        overflow |= o;
    }
    (!overflow).then_some(sum)
}

pub fn sum_i32(slice: &[i32]) -> Option<i32> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(simd) = pulp::x86::V3::try_new() {
        return x86::sum_i32_v3(simd, slice);
    }
    sum_i32_scalar(slice)
}

pub fn sum_i64(slice: &[i64]) -> Option<i64> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(simd) = pulp::x86::V3::try_new() {
        return x86::sum_i64_v3(simd, slice);
    }
    sum_i64_scalar(slice)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use pulp::{x86::*, *};
    pub fn sum_i32_v3(simd: V3, slice: &[i32]) -> Option<i32> {
        struct Impl<'a> {
            simd: V3,
            slice: &'a [i32],
        }

        impl pulp::NullaryFnOnce for Impl<'_> {
            type Output = Option<i32>;

            #[inline(always)]
            fn call(self) -> Self::Output {
                let Self { simd, slice } = self;
                let (head, tail) = V3::i32s_as_simd(slice);

                let zero = simd.splat_i32x8(0);
                let mut sum = zero;
                let mut overflow = simd.splat_m32x8(m32::new(false));

                for &v in head {
                    sum = simd.wrapping_add_i32x8(sum, v);
                    overflow = simd.or_m32x8(overflow, simd.cmp_lt_i32x8(sum, zero));
                }

                if overflow != simd.splat_m32x8(m32::new(false)) {
                    return None;
                }

                i32::checked_add(
                    sum_i32_scalar(tail)?,
                    sum_i32_scalar(bytemuck::cast_slice(&[sum]))?,
                )
            }
        }

        simd.vectorize(Impl { simd, slice })
    }

    pub fn sum_i64_v3(simd: V3, slice: &[i64]) -> Option<i64> {
        struct Impl<'a> {
            simd: V3,
            slice: &'a [i64],
        }

        impl pulp::NullaryFnOnce for Impl<'_> {
            type Output = Option<i64>;

            #[inline(always)]
            fn call(self) -> Self::Output {
                let Self { simd, slice } = self;
                let (head, tail) = V3::i64s_as_simd(slice);

                let zero = simd.splat_i64x4(0);
                let mut sum = zero;
                let mut overflow = simd.splat_m64x4(m64::new(false));

                for &v in head {
                    sum = simd.wrapping_add_i64x4(sum, v);
                    overflow = simd.or_m64x4(overflow, simd.cmp_lt_i64x4(sum, zero));
                }

                if overflow != simd.splat_m64x4(m64::new(false)) {
                    return None;
                }

                i64::checked_add(
                    sum_i64_scalar(tail)?,
                    sum_i64_scalar(bytemuck::cast_slice(&[sum]))?,
                )
            }
        }

        simd.vectorize(Impl { simd, slice })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::assert;

    #[test]
    fn test_sum_i32() {
        let array = vec![100_000_000i32; 1000];
        assert!(sum_i32(&array) == None);

        let array = vec![1_000_000i32; 1000];
        assert!(sum_i32(&array) == Some(1_000_000_000i32));
    }

    #[test]
    fn test_sum_i64() {
        let array = vec![i64::MAX / 100; 1000];
        assert!(sum_i64(&array) == None);

        let array = vec![100_000_000i64; 1000];
        assert!(sum_i64(&array) == Some(100_000_000_000i64));

        let array = vec![1_000_000i64; 1000];
        assert!(sum_i64(&array) == Some(1_000_000_000i64));
    }
}
