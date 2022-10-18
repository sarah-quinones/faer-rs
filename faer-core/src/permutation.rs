use assert2::{assert as fancy_assert, debug_assert as fancy_debug_assert};
use dyn_stack::DynStack;
use reborrow::*;

use crate::{MatMut, MatRef};

#[derive(Clone, Copy, Debug)]
pub struct PermutationIndicesRef<'a> {
    forward: &'a [usize],
    inverse: &'a [usize],
}

impl<'a> PermutationIndicesRef<'a> {
    /// Returns the permutation as an array.
    #[inline]
    pub fn into_arrays(self) -> (&'a [usize], &'a [usize]) {
        (self.forward, self.inverse)
    }

    /// Returns the inverse permutation.
    #[inline]
    pub fn inverse(self) -> Self {
        Self {
            forward: self.inverse,
            inverse: self.forward,
        }
    }

    /// Creates a new permutation reference, without checking the validity of the inputs.
    ///
    /// # Safety
    ///
    /// `forward` and `inverse` must have the same length, be valid permutations, and be inverse
    /// permutations of each other.
    #[inline]
    pub unsafe fn new_unchecked(forward: &'a [usize], inverse: &'a [usize]) -> Self {
        fancy_debug_assert!(forward.len() == inverse.len());
        Self { forward, inverse }
    }
}

impl<'a> PermutationIndicesMut<'a> {
    /// Returns the permutation as an array.
    #[inline]
    pub unsafe fn into_arrays(self) -> (&'a mut [usize], &'a mut [usize]) {
        (self.forward, self.inverse)
    }

    /// Returns the inverse permutation.
    #[inline]
    pub fn inverse(self) -> Self {
        Self {
            forward: self.inverse,
            inverse: self.forward,
        }
    }

    /// Creates a new permutation mutable reference, without checking the validity of the inputs.
    ///
    /// # Safety
    ///
    /// `forward` and `inverse` must have the same length, be valid permutations, and be inverse
    /// permutations of each other.
    #[inline]
    pub unsafe fn new_unchecked(forward: &'a mut [usize], inverse: &'a mut [usize]) -> Self {
        fancy_debug_assert!(forward.len() == inverse.len());
        Self { forward, inverse }
    }
}

#[derive(Debug)]
pub struct PermutationIndicesMut<'a> {
    forward: &'a mut [usize],
    inverse: &'a mut [usize],
}

impl<'short, 'a> Reborrow<'short> for PermutationIndicesRef<'a> {
    type Target = PermutationIndicesRef<'short>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}

impl<'short, 'a> ReborrowMut<'short> for PermutationIndicesRef<'a> {
    type Target = PermutationIndicesRef<'short>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}

impl<'short, 'a> Reborrow<'short> for PermutationIndicesMut<'a> {
    type Target = PermutationIndicesRef<'short>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        PermutationIndicesRef {
            forward: &*self.forward,
            inverse: &*self.inverse,
        }
    }
}

impl<'short, 'a> ReborrowMut<'short> for PermutationIndicesMut<'a> {
    type Target = PermutationIndicesMut<'short>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        PermutationIndicesMut {
            forward: &mut *self.forward,
            inverse: &mut *self.inverse,
        }
    }
}

/// Computes a symmetric permutation of the source matrix using the given permutation, and stores
/// the result in the destination matrix.
///
/// Both the source and the destination are interpreted as symmetric matrices, and only their lower
/// triangular part is accessed.
pub fn permute_rows_and_cols_symmetric<T: Clone>(
    dst: MatMut<'_, T>,
    src: MatRef<'_, T>,
    perm_indices: PermutationIndicesRef<'_>,
) {
    let mut dst = dst;
    let n = src.nrows();
    fancy_assert!(src.nrows() == src.ncols(), "source matrix must be square",);
    fancy_assert!(
        dst.nrows() == dst.ncols(),
        "destination matrix must be square",
    );
    fancy_assert!(
        src.nrows() == dst.nrows(),
        "source and destination matrices must have the same shape",
    );
    fancy_assert!(
        perm_indices.into_arrays().0.len() == n,
        "permutation must have the same length as the dimension of the matrices"
    );

    let perm = perm_indices.into_arrays().0;
    let src_tril = |i, j| unsafe {
        if i > j {
            src.get_unchecked(i, j)
        } else {
            src.get_unchecked(j, i)
        }
    };
    for j in 0..n {
        for i in j..n {
            unsafe {
                *dst.rb_mut().get_unchecked(i, j) =
                    src_tril(*perm.get_unchecked(i), *perm.get_unchecked(j)).clone();
            }
        }
    }
}

#[inline]
pub unsafe fn permute_rows_unchecked<T: Clone + Send + Sync>(
    dst: MatMut<'_, T>,
    src: MatRef<'_, T>,
    perm_indices: PermutationIndicesRef<'_>,
    n_threads: usize,
) {
    let mut dst = dst;
    let m = src.nrows();
    let n = src.ncols();
    fancy_debug_assert!(
        (src.nrows(), src.ncols()) == (dst.nrows(), dst.ncols()),
        "source and destination matrices must have the same shape",
    );
    fancy_debug_assert!(
        perm_indices.into_arrays().0.len() == m,
        "permutation must have the same length as the number of rows of the matrices"
    );

    let perm = perm_indices.into_arrays().0;

    if n > 1 && n_threads > 1 {
        let (_, _, dst0, dst1) = dst.split_at_unchecked(0, n / 2);
        let (_, _, src0, src1) = src.split_at_unchecked(0, n / 2);
        crate::join(
            |n_threads, _| permute_rows_unchecked(dst0, src0, perm_indices, n_threads),
            |n_threads, _| permute_rows_unchecked(dst1, src1, perm_indices, n_threads),
            |_| dyn_stack::StackReq::default(),
            |_| 0,
            0,
            DynStack::new(&mut []),
        );
        return;
    }
    for j in 0..n {
        for i in 0..m {
            unsafe {
                *dst.rb_mut().get_unchecked(i, j) =
                    src.get_unchecked(*perm.get_unchecked(i), j).clone();
            }
        }
    }
}

#[track_caller]
#[inline]
pub fn permute_rows<T: Clone + Send + Sync>(
    dst: MatMut<'_, T>,
    src: MatRef<'_, T>,
    perm_indices: PermutationIndicesRef<'_>,
    n_threads: usize,
) {
    fancy_assert!(
        (src.nrows(), src.ncols()) == (dst.nrows(), dst.ncols()),
        "source and destination matrices must have the same shape",
    );
    fancy_assert!(
        perm_indices.into_arrays().0.len() == src.nrows(),
        "permutation must have the same length as the number of rows of the matrices"
    );

    unsafe { permute_rows_unchecked(dst, src, perm_indices, n_threads) };
}
