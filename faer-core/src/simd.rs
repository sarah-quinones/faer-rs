use crate::ComplexField;
use pulp::Simd;

#[inline(always)]
pub fn slice_as_simd<E: ComplexField, S: Simd>(
    slice: E::Group<&[E::Unit]>,
) -> (E::Group<&[E::SimdUnit<S>]>, E::Group<&[E::Unit]>) {
    let (a_head, a_tail) = E::unzip(E::map(
        slice,
        #[inline(always)]
        |slice| E::slice_as_simd::<S>(slice),
    ));
    (a_head, a_tail)
}

#[inline(always)]
pub fn slice_as_mut_simd<E: ComplexField, S: Simd>(
    slice: E::Group<&mut [E::Unit]>,
) -> (E::Group<&mut [E::SimdUnit<S>]>, E::Group<&mut [E::Unit]>) {
    let (a_head, a_tail) = E::unzip(E::map(
        slice,
        #[inline(always)]
        |slice| E::slice_as_mut_simd::<S>(slice),
    ));
    (a_head, a_tail)
}

#[inline(always)]
pub fn simd_as_slice_unit<E: ComplexField, S: Simd>(values: &[E::SimdUnit<S>]) -> &[E::Unit] {
    unsafe {
        core::slice::from_raw_parts(
            values.as_ptr() as *const E::Unit,
            values.len()
                * (core::mem::size_of::<E::SimdUnit<S>>() / core::mem::size_of::<E::Unit>()),
        )
    }
}

#[inline(always)]
pub fn simd_as_slice<E: ComplexField, S: Simd>(
    values: E::Group<&[E::SimdUnit<S>]>,
) -> E::Group<&[E::Unit]> {
    E::map(
        values,
        #[inline(always)]
        |values| simd_as_slice_unit::<E, S>(values),
    )
}

#[inline(always)]
pub fn one_simd_as_slice<E: ComplexField, S: Simd>(
    values: E::Group<&E::SimdUnit<S>>,
) -> E::Group<&[E::Unit]> {
    E::map(
        values,
        #[inline(always)]
        |values| simd_as_slice_unit::<E, S>(core::slice::from_ref(values)),
    )
}
