//! Linear algebra module.
//!
//! Contains low level routines and the implementation of their corresponding high level
//! wrappers.
//!
//! # Memory allocation
//! Since most `faer` crates aim to expose a low level api for optimal performance, most algorithms
//! try to defer memory allocation to the user.
//!
//! However, since a lot of algorithms need some form of temporary space for intermediate
//! computations, they may ask for a slice of memory for that purpose, by taking a [`stack:
//! PodStack`](dyn_stack::PodStack) parameter. A [`PodStack`] is a thin wrapper over a slice of
//! memory bytes. This memory may come from any valid source (heap allocation, fixed-size array on
//! the stack, etc.). The functions taking a [`PodStack`] parameter have a corresponding function
//! with a similar name ending in `_req` that returns the memory requirements of the algorithm. For
//! example:
//! [`householder::apply_block_householder_on_the_left_in_place_with_conj`] and
//! [`householder::apply_block_householder_on_the_left_in_place_req`].
//!
//! The memory stack may be reused in user-code to avoid repeated allocations, and it is also
//! possible to compute the sum ([`dyn_stack::StackReq::all_of`]) or union
//! ([`dyn_stack::StackReq::any_of`]) of multiple requirements, in order to optimally combine them
//! into a single allocation.
//!
//! After computing a [`dyn_stack::StackReq`], one can query its size and alignment to allocate the
//! required memory. The simplest way to do so is through [`dyn_stack::GlobalMemBuffer::new`].
//!
//! # Entity trait
//! Matrices are built on top of the [`Entity`] trait, which describes the prefered memory
//! storage layout for a given type `E`. An entity can be decomposed into a group of units: for
//! a natively supported type ([`f32`], [`f64`], [`c32`](crate::complex_native::c32),
//! [`c64`](crate::complex_native::c64)), the unit is simply the type itself, and a group
//! contains a single element. On the other hand, for a type with a more specific preferred
//! layout, like an extended precision floating point type, or a dual number type, the unit
//! would be one of the natively supported types, and the group would be a structure holding
//! the components that build up the full value.
//!
//! To take a more specific example: [`num_complex::Complex<f64>`] has a storage memory layout
//! that differs from that of [`c64`](crate::complex_native::c64) (see
//! [`faer::complex_native`](crate::complex_native) for more details). Its real and complex
//! components are stored separately, so its unit type is `f64`, while its group type is
//! `Complex`. In practice, this means that for a `Mat<f64>`, methods such as
//! [`Mat::col_as_slice`] will return a `&[f64]`. Meanwhile, for a `Mat<Complex<f64>>`,
//! [`Mat::col_as_slice`] will return `Complex<&[f64]>`, which holds two slices, each pointing
//! respectively to a view over the real and the imaginary components.
//!
//! While the design of the entity trait is unconventional, it helps us achieve much higher
//! performance when targetting non native types, due to the design matching the typical
//! preffered CPU layout for SIMD operations. And for native types, since [`Group<T>` is just
//! `T`](Entity#impl-Entity-for-f64), the entity layer is a no-op, and the matrix layout is
//! compatible with the classic contiguous layout that's commonly used by other libraries.

use crate::{
    mat::{self, *},
    utils::DivCeil,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use entity::{ComplexField, Entity};

pub use faer_entity as entity;

pub mod zip;

pub mod householder;
pub mod matmul;
pub mod triangular_inverse;
pub mod triangular_solve;

pub mod cholesky;
pub mod lu;
pub mod qr;

pub mod evd;
pub mod svd;

/// High level linear system solvers.
pub mod solvers;

pub(crate) mod kron_impl;
mod mat_ops;
pub(crate) mod reductions;

pub use kron_impl::kron;

#[inline]
pub(crate) fn col_stride<Unit: 'static>(nrows: usize) -> usize {
    if !crate::mat::matalloc::is_vectorizable::<Unit>() || nrows >= isize::MAX as usize {
        nrows
    } else {
        nrows
            .msrv_checked_next_multiple_of(align_for::<Unit>() / core::mem::size_of::<Unit>())
            .unwrap()
    }
}

/// Returns the stack requirements for creating a temporary matrix with the given dimensions.
#[inline]
pub fn temp_mat_req<E: Entity>(nrows: usize, ncols: usize) -> Result<StackReq, SizeOverflow> {
    let col_stride = col_stride::<E::Unit>(nrows);
    let alloc_size = ncols.checked_mul(col_stride).ok_or(SizeOverflow)?;
    let additional = StackReq::try_new_aligned::<E::Unit>(alloc_size, align_for::<E::Unit>())?;

    let req = Ok(StackReq::empty());
    let (req, _) = E::faer_map_with_context(req, E::UNIT, &mut {
        #[inline(always)]
        |req, ()| {
            let req = match req {
                Ok(req) => req.try_and(additional),
                _ => Err(SizeOverflow),
            };

            (req, ())
        }
    });

    req
}

/// Creates a temporary matrix of constant values, from the given memory stack.
pub fn temp_mat_constant<E: ComplexField>(
    nrows: usize,
    ncols: usize,
    value: E,
    stack: PodStack<'_>,
) -> (MatMut<'_, E>, PodStack<'_>) {
    let (mut mat, stack) = temp_mat_uninit::<E>(nrows, ncols, stack);
    mat.as_mut().fill(value);
    (mat, stack)
}

/// Creates a temporary matrix of zero values, from the given memory stack.
pub fn temp_mat_zeroed<E: ComplexField>(
    nrows: usize,
    ncols: usize,
    stack: PodStack<'_>,
) -> (MatMut<'_, E>, PodStack<'_>) {
    let (mut mat, stack) = temp_mat_uninit::<E>(nrows, ncols, stack);
    mat.as_mut().fill_zero();
    (mat, stack)
}

/// Creates a temporary matrix of untouched values, from the given memory stack.
pub fn temp_mat_uninit<E: ComplexField>(
    nrows: usize,
    ncols: usize,
    stack: PodStack<'_>,
) -> (MatMut<'_, E>, PodStack<'_>) {
    let col_stride = col_stride::<E::Unit>(nrows);
    let alloc_size = ncols.checked_mul(col_stride).unwrap();

    let (stack, alloc) = E::faer_map_with_context(stack, E::UNIT, &mut {
        #[inline(always)]
        |stack, ()| {
            let (alloc, stack) =
                stack.make_aligned_raw::<E::Unit>(alloc_size, align_for::<E::Unit>());
            (stack, alloc)
        }
    });
    (
        unsafe {
            mat::from_raw_parts_mut(
                E::faer_map(alloc, |alloc| alloc.as_mut_ptr()),
                nrows,
                ncols,
                1,
                col_stride as isize,
            )
        },
        stack,
    )
}
