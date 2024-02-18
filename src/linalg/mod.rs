use crate::{
    mat::{self, matalloc::align_for, MatMut},
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
