use core::fmt::Debug;

/// Matrix structure in column major format.
///
/// A matrix can be thought of as a 2D array of values.
/// These values are stored in memory so that the columns are contiguous.
///
/// # Note
///
/// Note that the matrix as a whole may not necessarily be contiguous.
/// The implementation may add padding at the end of each column when
/// overaligning each column can provide a performance gain.
///
/// Let us consider a 3×4 matrix
///
/// ```
/// 0 3 6  9
/// 1 4 7 10
/// 2 5 8 11
/// ```
///
/// The memory representation of such a matrix could look like the following:
/// ```
/// 0 1 2 X 3 4 5 X 6 7 8 X 9 10 11 X
/// ```
/// where `X` represents padding elements.
#[derive(Clone)]
pub struct Mat<T: 'static> {
    inner: faer_core::Mat<T>,
}

impl<T: Debug + 'static> Debug for Mat<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}
