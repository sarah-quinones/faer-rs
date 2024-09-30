use crate::{
    assert, debug_assert,
    linalg::{
        entity,
        entity::{from_copy, GroupFor, SimdCtx, SimdGroupFor, SimdUnitFor},
        matmul::triangular::BlockStructure,
        temp_mat_req, temp_mat_uninit, triangular_solve as solve,
    },
    solvers::CholeskyError,
    sparse::linalg::cholesky::LltRegularization,
    unzipped,
    utils::{simd::*, slice::*, DivCeil},
    zipped, ComplexField, MatMut, Parallelism,
};
use core::{convert::Infallible, marker::PhantomData};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_entity::*;
use pulp::Simd;
use reborrow::*;

#[inline(always)]
fn first_elem<E: ComplexField, S: Simd>(values: SimdGroupFor<E, S>) -> E {
    E::faer_from_units(E::faer_map(
        from_copy::<E, _>(values),
        #[inline(always)]
        |x| pulp::cast_lossy::<SimdUnitFor<E, S>, E::Unit>(x),
    ))
}

#[inline(always)]
fn as_simd<E: ComplexField, S: Simd>(
    slice: GroupFor<E, &[E::Unit]>,
) -> (GroupFor<E, &[SimdUnitFor<E, S>]>, GroupFor<E, &[E::Unit]>) {
    let slice = SliceGroup::<E>::new(slice);

    let lanes = core::mem::size_of::<entity::SimdUnitFor<E, S>>() / core::mem::size_of::<E::Unit>();
    let len = slice.len();

    let mut prefix = len / lanes * lanes;
    if len % lanes == 0 {
        prefix -= lanes;
    }

    let (head, tail) = slice.split_at(prefix);
    let head = entity::slice_as_simd::<E, S>(head.into_inner()).0;
    (head, tail.into_inner())
}

#[inline(always)]
fn as_simd_mut<E: ComplexField, S: Simd>(
    slice: GroupFor<E, &mut [E::Unit]>,
) -> (
    GroupFor<E, &mut [SimdUnitFor<E, S>]>,
    GroupFor<E, &mut [E::Unit]>,
) {
    let slice = SliceGroupMut::<E>::new(slice);

    let lanes = core::mem::size_of::<entity::SimdUnitFor<E, S>>() / core::mem::size_of::<E::Unit>();
    let len = slice.len();

    let mut prefix = len / lanes * lanes;
    if len % lanes == 0 {
        prefix -= lanes;
    }

    let (head, tail) = slice.split_at(prefix);
    let head = entity::slice_as_mut_simd::<E, S>(head.into_inner()).0;
    (head, tail.into_inner())
}

pub(crate) trait ProcessDiag<E: ComplexField> {
    type Error;
    type This;

    fn new(&self) -> (Self::This, bool);
    fn process(
        this: (&Self::This, bool),
        diag: E::Real,
        idx: usize,
        count: &mut usize,
    ) -> Result<E::Real, Self::Error>;
    fn mul_diag(diag: E::Real, factor: E) -> E;
}

impl<E: ComplexField> ProcessDiag<E> for LltRegularization<E> {
    type Error = CholeskyError;
    type This = (E::Real, E::Real);

    #[inline]
    fn process(
        (&(eps, delta), enable): (&Self::This, bool),
        mut diag: E::Real,
        idx: usize,
        count: &mut usize,
    ) -> Result<E::Real, Self::Error> {
        if enable {
            if diag < eps {
                diag = delta;
                *count += 1;
            }
        }
        if diag > E::Real::faer_zero() {
            Ok(diag.faer_sqrt())
        } else {
            Err(CholeskyError {
                non_positive_definite_minor: idx,
            })
        }
    }

    #[inline]
    fn new(&self) -> (Self::This, bool) {
        (
            (
                self.dynamic_regularization_epsilon,
                self.dynamic_regularization_delta,
            ),
            self.dynamic_regularization_epsilon != E::Real::faer_zero(),
        )
    }

    #[inline]
    fn mul_diag(_: E::Real, factor: E) -> E {
        factor
    }
}

impl<'a, E: ComplexField> ProcessDiag<E> for LdltRegularization<'a, E> {
    type Error = Infallible;
    type This = (Option<&'a [i8]>, E::Real, E::Real);

    #[inline]
    fn process(
        (&(signs, eps, delta), enable): (&Self::This, bool),
        mut diag: E::Real,
        idx: usize,
        count: &mut usize,
    ) -> Result<E::Real, Self::Error> {
        if enable {
            if let Some(signs) = signs {
                if signs[idx] > 0 && diag <= eps {
                    diag = delta;
                    *count += 1;
                } else if signs[idx] < 0 && diag >= eps.faer_neg() {
                    diag = -delta;
                    *count += 1;
                }
            } else if diag.faer_abs() <= eps {
                if diag < E::Real::faer_zero() {
                    diag = -delta;
                } else {
                    diag = delta;
                }
                *count += 1;
            }
        }
        Ok(diag)
    }

    #[inline]
    fn new(&self) -> (Self::This, bool) {
        (
            (
                self.dynamic_regularization_signs,
                self.dynamic_regularization_epsilon,
                self.dynamic_regularization_delta,
            ),
            self.dynamic_regularization_epsilon != E::Real::faer_zero(),
        )
    }

    #[inline]
    fn mul_diag(diag: E::Real, factor: E) -> E {
        factor.faer_scale_real(diag)
    }
}

pub(crate) fn new_cholesky<E: ComplexField, P: ProcessDiag<E>>(
    A: MatMut<'_, E>,
    process: &P,
    stack: &mut PodStack,
) -> Result<usize, P::Error> {
    struct Impl<'a, E: ComplexField, P> {
        A: MatMut<'a, E>,
        stack: &'a mut PodStack,
        process: &'a P,
    }
    impl<E: ComplexField, P: ProcessDiag<E>> pulp::WithSimd for Impl<'_, E, P> {
        type Output = Result<usize, P::Error>;

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let Self { A, stack, process } = self;

            let n = A.nrows();

            let lanes =
                core::mem::size_of::<entity::SimdUnitFor<E, S>>() / core::mem::size_of::<E::Unit>();

            let stride = n.msrv_div_ceil(lanes);
            assert!(stride <= 4);

            unsafe {
                let mut A = A;
                let (stack, slice) = E::faer_map_with_context(stack, E::UNIT, &mut |stack, ()| {
                    let (slice, stack) = stack.make_aligned_raw::<entity::SimdUnitFor<E, S>>(
                        stride * n,
                        crate::mat::matalloc::CACHELINE_ALIGN,
                    );
                    (stack, slice)
                });
                let (diag_slice, _) = stack.make_raw::<E::Real>(n);

                let mut slice = SliceGroupMut::<E, entity::SimdUnitFor<E, S>>::new(slice);

                #[inline(always)]
                unsafe fn in_place_cholesky<E: ComplexField, S: Simd, P: ProcessDiag<E>>(
                    simd: S,
                    n: usize,
                    stride: usize,
                    mut slice: SliceGroupMut<E, entity::SimdUnitFor<E, S>>,
                    diag_slice: &mut [E::Real],
                    process: (&P::This, bool),
                ) -> Result<usize, P::Error> {
                    let simd_real = SimdFor::<E::Real, _>::new(simd);
                    let simd = SimdFor::<E, _>::new(simd);

                    let lanes = core::mem::size_of::<entity::SimdUnitFor<E, S>>()
                        / core::mem::size_of::<E::Unit>();
                    let mut length = lanes * stride;
                    let mut j = 0usize;

                    let mut count = 0;

                    if length > lanes {
                        while length > 3 * lanes {
                            let mut slice =
                                slice.rb_mut().subslice_unchecked(stride - 4..stride * n);

                            let mut Aj0 = slice.rb().get_unchecked(stride * j + 0).get();
                            let mut Aj1 = slice.rb().get_unchecked(stride * j + 1).get();
                            let mut Aj2 = slice.rb().get_unchecked(stride * j + 2).get();
                            let mut Aj3 = slice.rb().get_unchecked(stride * j + 3).get();

                            for k in 0..j {
                                let Ak0 = slice.rb().get_unchecked(stride * k + 0).get();
                                let Ak1 = slice.rb().get_unchecked(stride * k + 1).get();
                                let Ak2 = slice.rb().get_unchecked(stride * k + 2).get();
                                let Ak3 = slice.rb().get_unchecked(stride * k + 3).get();

                                let factor = simd.rotate_left(Ak0, j);
                                let factor = simd.splat(-P::mul_diag(
                                    *diag_slice.get_unchecked(k),
                                    first_elem::<E, S>(factor).faer_conj(),
                                ));

                                Aj0 = simd.mul_add_e(factor, Ak0, Aj0);
                                Aj1 = simd.mul_add_e(factor, Ak1, Aj1);
                                Aj2 = simd.mul_add_e(factor, Ak2, Aj2);
                                Aj3 = simd.mul_add_e(factor, Ak3, Aj3);
                            }

                            let diag = simd.rotate_left(Aj0, j);
                            let diag = first_elem::<E, S>(diag);
                            let diag = P::process(process, diag.faer_real(), j, &mut count)?;
                            *diag_slice.get_unchecked_mut(j) = diag;

                            let inv = simd_real.splat(diag.faer_inv());

                            Aj0 = simd.scale_real(inv, Aj0);
                            Aj1 = simd.scale_real(inv, Aj1);
                            Aj2 = simd.scale_real(inv, Aj2);
                            Aj3 = simd.scale_real(inv, Aj3);

                            slice.rb_mut().get_unchecked_mut(stride * j + 0).set(Aj0);
                            slice.rb_mut().get_unchecked_mut(stride * j + 1).set(Aj1);
                            slice.rb_mut().get_unchecked_mut(stride * j + 2).set(Aj2);
                            slice.rb_mut().get_unchecked_mut(stride * j + 3).set(Aj3);

                            length -= 1;
                            j += 1;
                        }

                        while length > 2 * lanes {
                            let mut slice =
                                slice.rb_mut().subslice_unchecked(stride - 3..stride * n);

                            let mut Aj0 = slice.rb().get_unchecked(stride * j + 0).get();
                            let mut Aj1 = slice.rb().get_unchecked(stride * j + 1).get();
                            let mut Aj2 = slice.rb().get_unchecked(stride * j + 2).get();

                            for k in 0..j {
                                let Ak0 = slice.rb().get_unchecked(stride * k + 0).get();
                                let Ak1 = slice.rb().get_unchecked(stride * k + 1).get();
                                let Ak2 = slice.rb().get_unchecked(stride * k + 2).get();

                                let factor = simd.rotate_left(Ak0, j);
                                let factor = simd.splat(-P::mul_diag(
                                    *diag_slice.get_unchecked(k),
                                    first_elem::<E, S>(factor).faer_conj(),
                                ));

                                Aj0 = simd.mul_add_e(factor, Ak0, Aj0);
                                Aj1 = simd.mul_add_e(factor, Ak1, Aj1);
                                Aj2 = simd.mul_add_e(factor, Ak2, Aj2);
                            }

                            let diag = simd.rotate_left(Aj0, j);
                            let diag = first_elem::<E, S>(diag);
                            let diag = P::process(process, diag.faer_real(), j, &mut count)?;
                            *diag_slice.get_unchecked_mut(j) = diag;

                            let inv = simd_real.splat(diag.faer_inv());

                            Aj0 = simd.scale_real(inv, Aj0);
                            Aj1 = simd.scale_real(inv, Aj1);
                            Aj2 = simd.scale_real(inv, Aj2);

                            slice.rb_mut().get_unchecked_mut(stride * j + 0).set(Aj0);
                            slice.rb_mut().get_unchecked_mut(stride * j + 1).set(Aj1);
                            slice.rb_mut().get_unchecked_mut(stride * j + 2).set(Aj2);

                            length -= 1;
                            j += 1;
                        }

                        while length > 1 * lanes {
                            let mut slice =
                                slice.rb_mut().subslice_unchecked(stride - 2..stride * n);

                            let mut Aj0 = slice.rb().get_unchecked(stride * j + 0).get();
                            let mut Aj1 = slice.rb().get_unchecked(stride * j + 1).get();

                            for k in 0..j {
                                let Ak0 = slice.rb().get_unchecked(stride * k + 0).get();
                                let Ak1 = slice.rb().get_unchecked(stride * k + 1).get();

                                let factor = simd.rotate_left(Ak0, j);
                                let factor = simd.splat(-P::mul_diag(
                                    *diag_slice.get_unchecked(k),
                                    first_elem::<E, S>(factor).faer_conj(),
                                ));

                                Aj0 = simd.mul_add_e(factor, Ak0, Aj0);
                                Aj1 = simd.mul_add_e(factor, Ak1, Aj1);
                            }

                            let diag = simd.rotate_left(Aj0, j);
                            let diag = first_elem::<E, S>(diag);
                            let diag = P::process(process, diag.faer_real(), j, &mut count)?;
                            *diag_slice.get_unchecked_mut(j) = diag;

                            let inv = simd_real.splat(diag.faer_inv());

                            Aj0 = simd.scale_real(inv, Aj0);
                            Aj1 = simd.scale_real(inv, Aj1);

                            slice.rb_mut().get_unchecked_mut(stride * j + 0).set(Aj0);
                            slice.rb_mut().get_unchecked_mut(stride * j + 1).set(Aj1);

                            length -= 1;
                            j += 1;
                        }
                    }

                    while j < n {
                        let mut slice = slice.rb_mut().subslice_unchecked(stride - 1..stride * n);

                        let mut Aj0 = slice.rb().get_unchecked(stride * j + 0).get();
                        for k in 0..j {
                            let Ak0 = slice.rb().get_unchecked(stride * k + 0).get();

                            let factor = simd.rotate_left(Ak0, j);
                            let factor = simd.splat(-P::mul_diag(
                                *diag_slice.get_unchecked(k),
                                first_elem::<E, S>(factor).faer_conj(),
                            ));

                            Aj0 = simd.mul_add_e(factor, Ak0, Aj0);
                        }

                        let diag = simd.rotate_left(Aj0, j);
                        let diag = first_elem::<E, S>(diag);
                        let diag = P::process(process, diag.faer_real(), j, &mut count)?;
                        *diag_slice.get_unchecked_mut(j) = diag;

                        let inv = simd_real.splat(diag.faer_inv());

                        Aj0 = simd.scale_real(inv, Aj0);

                        slice.rb_mut().get_unchecked_mut(stride * j + 0).set(Aj0);

                        j += 1;
                    }

                    Ok(count)
                }

                if A.row_stride() == 1 {
                    let mut length = lanes * stride;
                    let mut j = 0usize;

                    if length > lanes {
                        while length > 3 * lanes {
                            let (head, tail) =
                                as_simd::<E, S>(A.rb().col(j).try_as_slice().unwrap());
                            let head = SliceGroup::<'_, E, _>::new(head)
                                .subslice_unchecked(stride - 4..stride - 1);
                            let tail = E::faer_partial_load(simd, tail);

                            let mut dst = slice
                                .rb_mut()
                                .subslice_unchecked(j * stride..(j + 1) * stride)
                                .subslice_unchecked(stride - 4..stride);

                            dst.rb_mut()
                                .get_unchecked_mut(0)
                                .set(head.get_unchecked(0).get());
                            dst.rb_mut()
                                .get_unchecked_mut(1)
                                .set(head.get_unchecked(1).get());
                            dst.rb_mut()
                                .get_unchecked_mut(2)
                                .set(head.get_unchecked(2).get());
                            dst.rb_mut().get_unchecked_mut(3).set(tail);

                            length -= 1;
                            j += 1;
                        }
                        while length > 2 * lanes {
                            let (head, tail) =
                                as_simd::<E, S>(A.rb().col(j).try_as_slice().unwrap());
                            let head = SliceGroup::<'_, E, _>::new(head)
                                .subslice_unchecked(stride - 3..stride - 1);
                            let tail = E::faer_partial_load(simd, tail);

                            let mut dst = slice
                                .rb_mut()
                                .subslice_unchecked(j * stride..(j + 1) * stride)
                                .subslice_unchecked(stride - 3..stride);

                            dst.rb_mut()
                                .get_unchecked_mut(0)
                                .set(head.get_unchecked(0).get());
                            dst.rb_mut()
                                .get_unchecked_mut(1)
                                .set(head.get_unchecked(1).get());
                            dst.rb_mut().get_unchecked_mut(2).set(tail);

                            length -= 1;
                            j += 1;
                        }
                        while length > 1 * lanes {
                            let (head, tail) =
                                as_simd::<E, S>(A.rb().col(j).try_as_slice().unwrap());
                            let head = SliceGroup::<'_, E, _>::new(head)
                                .subslice_unchecked(stride - 2..stride - 1);
                            let tail = E::faer_partial_load(simd, tail);

                            let mut dst = slice
                                .rb_mut()
                                .subslice_unchecked(j * stride..(j + 1) * stride)
                                .subslice_unchecked(stride - 2..stride);

                            dst.rb_mut()
                                .get_unchecked_mut(0)
                                .set(head.get_unchecked(0).get());
                            dst.rb_mut().get_unchecked_mut(1).set(tail);

                            length -= 1;
                            j += 1;
                        }
                    }
                    while j < n {
                        let (head, tail) = as_simd::<E, S>(A.rb().col(j).try_as_slice().unwrap());
                        let _ = SliceGroup::<'_, E, _>::new(head)
                            .subslice_unchecked(stride - 1..stride - 1);
                        let tail = E::faer_partial_load(simd, tail);

                        let mut dst = slice
                            .rb_mut()
                            .subslice_unchecked(j * stride..(j + 1) * stride)
                            .subslice_unchecked(stride - 1..stride);

                        dst.rb_mut().get_unchecked_mut(0).set(tail);

                        j += 1;
                    }
                } else {
                    let slice = E::faer_map(slice.rb_mut().into_inner(), |slice| {
                        bytemuck::cast_slice_mut::<SimdUnitFor<E, S>, E::Unit>(slice)
                    });
                    crate::mat::from_column_major_slice_with_stride_mut_generic::<E, _, _>(
                        slice,
                        n,
                        n,
                        stride * lanes,
                    )
                    .copy_from(&A);
                }

                let (process, enable) = process.new();
                let result;
                if enable {
                    result = in_place_cholesky::<_, _, P>(
                        simd,
                        n,
                        stride,
                        slice.rb_mut(),
                        diag_slice,
                        (&process, enable),
                    );
                } else {
                    result = in_place_cholesky::<_, _, P>(
                        simd,
                        n,
                        stride,
                        slice.rb_mut(),
                        diag_slice,
                        (&process, enable),
                    );
                }
                if A.row_stride() == 1 {
                    let mut length = lanes * stride;
                    let mut j = 0usize;

                    if length > lanes {
                        while length > 3 * lanes {
                            let (head, tail) = as_simd_mut::<E, S>(
                                A.rb_mut().col_mut(j).try_as_slice_mut().unwrap(),
                            );
                            let mut head = SliceGroupMut::<'_, E, _>::new(head)
                                .subslice_unchecked(stride - 4..stride - 1);

                            let src = slice
                                .rb()
                                .subslice_unchecked(j * stride..(j + 1) * stride)
                                .subslice_unchecked(stride - 4..stride);

                            head.rb_mut()
                                .get_unchecked_mut(0)
                                .set(src.get_unchecked(0).get());
                            head.rb_mut()
                                .get_unchecked_mut(1)
                                .set(src.get_unchecked(1).get());
                            head.rb_mut()
                                .get_unchecked_mut(2)
                                .set(src.get_unchecked(2).get());
                            E::faer_partial_store(simd, tail, src.get_unchecked(3).get());

                            length -= 1;
                            j += 1;
                        }
                        while length > 2 * lanes {
                            let (head, tail) = as_simd_mut::<E, S>(
                                A.rb_mut().col_mut(j).try_as_slice_mut().unwrap(),
                            );
                            let mut head = SliceGroupMut::<'_, E, _>::new(head)
                                .subslice_unchecked(stride - 3..stride - 1);

                            let src = slice
                                .rb()
                                .subslice_unchecked(j * stride..(j + 1) * stride)
                                .subslice_unchecked(stride - 3..stride);

                            head.rb_mut()
                                .get_unchecked_mut(0)
                                .set(src.get_unchecked(0).get());
                            head.rb_mut()
                                .get_unchecked_mut(1)
                                .set(src.get_unchecked(1).get());
                            E::faer_partial_store(simd, tail, src.get_unchecked(2).get());

                            length -= 1;
                            j += 1;
                        }
                        while length > 1 * lanes {
                            let (head, tail) = as_simd_mut::<E, S>(
                                A.rb_mut().col_mut(j).try_as_slice_mut().unwrap(),
                            );
                            let mut head = SliceGroupMut::<'_, E, _>::new(head)
                                .subslice_unchecked(stride - 2..stride - 1);

                            let src = slice
                                .rb()
                                .subslice_unchecked(j * stride..(j + 1) * stride)
                                .subslice_unchecked(stride - 2..stride);

                            head.rb_mut()
                                .get_unchecked_mut(0)
                                .set(src.get_unchecked(0).get());
                            E::faer_partial_store(simd, tail, src.get_unchecked(1).get());

                            length -= 1;
                            j += 1;
                        }
                    }
                    while j < n {
                        let (head, tail) =
                            as_simd_mut::<E, S>(A.rb_mut().col_mut(j).try_as_slice_mut().unwrap());
                        let _ = SliceGroupMut::<'_, E, _>::new(head)
                            .subslice_unchecked(stride - 1..stride - 1);

                        let src = slice
                            .rb()
                            .subslice_unchecked(j * stride..(j + 1) * stride)
                            .subslice_unchecked(stride - 1..stride);

                        E::faer_partial_store(simd, tail, src.get_unchecked(0).get());

                        j += 1;
                    }
                } else {
                    let slice = E::faer_map(slice.rb_mut().into_inner(), |slice| {
                        bytemuck::cast_slice_mut::<SimdUnitFor<E, S>, E::Unit>(slice)
                    });
                    A.copy_from(
                        crate::mat::from_column_major_slice_with_stride_mut_generic::<E, _, _>(
                            slice,
                            n,
                            n,
                            stride * lanes,
                        ),
                    );
                }
                for (dst, src) in core::iter::zip(
                    A.rb_mut().diagonal_mut().column_vector_mut().iter_mut(),
                    &*diag_slice,
                ) {
                    RefGroupMut::<E>::new(dst).set_(E::faer_from_real(*src).faer_into_units());
                }

                result
            }
        }
    }

    E::Simd::default().dispatch(Impl { A, stack, process })
}

fn cholesky_in_place_left_looking_impl<E: ComplexField>(
    matrix: MatMut<'_, E>,
    regularization: LdltRegularization<'_, E>,
    parallelism: Parallelism,
    params: LdltDiagParams,
    stack: &mut PodStack,
) -> usize {
    _ = params;
    _ = parallelism;
    new_cholesky(matrix, &regularization, stack).unwrap()
}

/// LDLT factorization tuning parameters.
#[derive(Default, Copy, Clone)]
#[non_exhaustive]
pub struct LdltDiagParams {}

/// Computes the size and alignment of required workspace for performing a Cholesky decomposition.
pub fn raw_cholesky_in_place_req<E: Entity>(
    dim: usize,
    parallelism: Parallelism,
    params: LdltDiagParams,
) -> Result<StackReq, SizeOverflow> {
    let _ = parallelism;
    let _ = params;
    temp_mat_req::<E>(dim, dim)?.try_and(StackReq::try_new::<E>(dim)?)
}

// uses an out parameter for tail recursion
fn cholesky_in_place_impl<E: ComplexField>(
    count: &mut usize,
    matrix: MatMut<'_, E>,
    regularization: LdltRegularization<'_, E>,
    parallelism: Parallelism,
    stack: &mut PodStack,
    params: LdltDiagParams,
) {
    // right looking cholesky

    debug_assert!(matrix.nrows() == matrix.ncols());
    let mut matrix = matrix;
    let mut stack = stack;

    struct Lanes<E> {
        __marker: PhantomData<E>,
    }
    impl<E: ComplexField> pulp::WithSimd for Lanes<E> {
        type Output = usize;
        fn with_simd<S: Simd>(self, _: S) -> Self::Output {
            core::mem::size_of::<entity::SimdUnitFor<E, S>>() / core::mem::size_of::<E::Unit>()
        }
    }

    let lanes = E::Simd::default().dispatch(Lanes {
        __marker: PhantomData::<E>,
    });
    let stride = matrix.nrows().msrv_div_ceil(lanes);

    let n = matrix.nrows();
    if stride <= 4 {
        *count +=
            cholesky_in_place_left_looking_impl(matrix, regularization, parallelism, params, stack)
    } else {
        let block_size = Ord::min(n / 2, 128);
        let rem = n - block_size;
        let (mut l00, _, mut a10, mut a11) = matrix.rb_mut().split_at_mut(block_size, block_size);

        cholesky_in_place_impl(
            count,
            l00.rb_mut(),
            regularization,
            parallelism,
            stack.rb_mut(),
            params,
        );

        let l00 = l00.into_const();
        let d0 = l00.diagonal().column_vector();

        solve::solve_unit_lower_triangular_in_place(
            l00.conjugate(),
            a10.rb_mut().transpose_mut(),
            parallelism,
        );

        {
            // reserve space for L10×D0
            let (mut l10xd0, _) = temp_mat_uninit::<E>(rem, block_size, stack);
            let mut l10xd0 = l10xd0.as_mut();

            for j in 0..block_size {
                let l10xd0_col = l10xd0.rb_mut().col_mut(j);
                let a10_col = a10.rb_mut().col_mut(j);
                let d0_elem = d0.read(j).faer_real().faer_inv();

                zipped!(l10xd0_col, a10_col).for_each(
                    |unzipped!(mut l10xd0_elem, mut a10_elem)| {
                        let a10_elem_read = a10_elem.read();
                        a10_elem.write(a10_elem_read.faer_scale_real(d0_elem));
                        l10xd0_elem.write(a10_elem_read);
                    },
                );
            }

            crate::linalg::matmul::triangular::matmul(
                a11.rb_mut(),
                BlockStructure::TriangularLower,
                a10.into_const(),
                BlockStructure::Rectangular,
                l10xd0.adjoint_mut().into_const(),
                BlockStructure::Rectangular,
                Some(E::faer_one()),
                E::faer_one().faer_neg(),
                parallelism,
            );
        }

        cholesky_in_place_impl(
            count,
            a11,
            LdltRegularization {
                dynamic_regularization_signs: regularization
                    .dynamic_regularization_signs
                    .map(|signs| &signs[block_size..]),
                dynamic_regularization_delta: regularization.dynamic_regularization_delta,
                dynamic_regularization_epsilon: regularization.dynamic_regularization_epsilon,
            },
            parallelism,
            stack,
            params,
        )
    }
}

/// Dynamic LDLT regularization.
/// Values below `epsilon` in absolute value, or with the wrong sign are set to `delta` with
/// their corrected sign.
#[derive(Copy, Clone, Debug)]
pub struct LdltRegularization<'a, E: ComplexField> {
    /// Expected signs for the diagonal at each step of the decomposition.
    pub dynamic_regularization_signs: Option<&'a [i8]>,
    /// Regularized value.
    pub dynamic_regularization_delta: E::Real,
    /// Regularization threshold.
    pub dynamic_regularization_epsilon: E::Real,
}

/// Info about the result of the LDLT factorization.
#[derive(Copy, Clone, Debug)]
pub struct LdltInfo {
    /// Number of pivots whose value or sign had to be corrected.
    pub dynamic_regularization_count: usize,
}

impl<E: ComplexField> Default for LdltRegularization<'_, E> {
    fn default() -> Self {
        Self {
            dynamic_regularization_signs: None,
            dynamic_regularization_delta: E::Real::faer_zero(),
            dynamic_regularization_epsilon: E::Real::faer_zero(),
        }
    }
}

/// Computes the Cholesky factors $L$ and $D$ of the input matrix such that $L$ is strictly lower
/// triangular, $D$ is real-valued diagonal, and
/// $$LDL^H = A.$$
///
/// The result is stored back in the same matrix.
///
/// The input matrix is interpreted as Hermitian with the values being extracted from the lower
/// part, but the entire matrix is required to be initialized.
///
/// The matrix $L$ is stored in the strictly lower triangular part of the input matrix, and the
/// diagonal elements of $D$ are stored on the diagonal.
///
/// The strictly upper triangular part of the matrix is clobbered and may be filled with garbage
/// values.
///
/// # Warning
///
/// The Cholesky decomposition with diagonal may have poor numerical stability properties when used
/// with non positive definite matrices. In the general case, it is recommended to first permute
/// (and conjugate when necessary) the rows and columns of the matrix using the permutation obtained
/// from [`faer::linalg::cholesky::compute_cholesky_permutation`](crate::linalg::cholesky::compute_cholesky_permutation).
///
/// # Panics
///
/// Panics if the input matrix is not square.
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`raw_cholesky_in_place_req`]).
#[track_caller]
#[inline]
pub fn raw_cholesky_in_place<E: ComplexField>(
    matrix: MatMut<'_, E>,
    regularization: LdltRegularization<'_, E>,
    parallelism: Parallelism,
    stack: &mut PodStack,
    params: LdltDiagParams,
) -> LdltInfo {
    assert!(matrix.ncols() == matrix.nrows());
    #[cfg(feature = "perf-warn")]
    if matrix.row_stride().unsigned_abs() != 1 && crate::__perf_warn!(CHOLESKY_WARN) {
        if matrix.col_stride().unsigned_abs() == 1 {
            log::warn!(target: "faer_perf", "LDLT prefers column-major matrix. Found row-major matrix.");
        } else {
            log::warn!(target: "faer_perf", "LDLT prefers column-major matrix. Found matrix with generic strides.");
        }
    }

    let mut count = 0;
    cholesky_in_place_impl(
        &mut count,
        matrix,
        regularization,
        parallelism,
        stack,
        params,
    );
    LdltInfo {
        dynamic_regularization_count: count,
    }
}
