use crate::{
    linalg::entity::GroupFor, mat, ColMut, ColRef, Entity, MatMut, MatRef, RowMut, RowRef, Shape,
};

use self::chunks::ChunkPolicy;

/// Fixed-size chunked column iterator over the elements.
#[derive(Debug, Clone)]
pub struct ColElemPartition<'a, E: Entity> {
    pub(crate) inner: ColRef<'a, E>,
    pub(crate) policy: chunks::PartitionCountPolicy,
}
/// Fixed-size chunked column iterator over the elements.
#[derive(Debug)]
pub struct ColElemPartitionMut<'a, E: Entity> {
    pub(crate) inner: ColMut<'a, E>,
    pub(crate) policy: chunks::PartitionCountPolicy,
}
/// Fixed-size chunked row iterator over the elements.
#[derive(Debug, Clone)]
pub struct RowElemPartition<'a, E: Entity> {
    pub(crate) inner: RowRef<'a, E>,
    pub(crate) policy: chunks::PartitionCountPolicy,
}
/// Fixed-size chunked row iterator over the elements.
#[derive(Debug)]
pub struct RowElemPartitionMut<'a, E: Entity> {
    pub(crate) inner: RowMut<'a, E>,
    pub(crate) policy: chunks::PartitionCountPolicy,
}

/// Fixed-size chunked matrix iterator over the columns.
#[derive(Debug, Clone)]
pub struct ColPartition<'a, E: Entity> {
    pub(crate) inner: MatRef<'a, E>,
    pub(crate) policy: chunks::PartitionCountPolicy,
}
/// Fixed-size chunked matrix iterator over the columns.
#[derive(Debug)]
pub struct ColPartitionMut<'a, E: Entity> {
    pub(crate) inner: MatMut<'a, E>,
    pub(crate) policy: chunks::PartitionCountPolicy,
}
/// Fixed-size chunked matrix iterator over the rows.
#[derive(Debug, Clone)]
pub struct RowPartition<'a, E: Entity> {
    pub(crate) inner: MatRef<'a, E>,
    pub(crate) policy: chunks::PartitionCountPolicy,
}
/// Fixed-size chunked matrix iterator over the rows.
#[derive(Debug)]
pub struct RowPartitionMut<'a, E: Entity> {
    pub(crate) inner: MatMut<'a, E>,
    pub(crate) policy: chunks::PartitionCountPolicy,
}

/// Chunked column iterator over the elements.
#[derive(Debug, Clone)]
pub struct ColElemChunks<'a, E: Entity> {
    pub(crate) inner: ColRef<'a, E>,
    pub(crate) policy: chunks::ChunkSizePolicy,
}
/// Chunked column iterator over the elements.
#[derive(Debug)]
pub struct ColElemChunksMut<'a, E: Entity> {
    pub(crate) inner: ColMut<'a, E>,
    pub(crate) policy: chunks::ChunkSizePolicy,
}
/// Chunked row iterator over the elements.
#[derive(Debug, Clone)]
pub struct RowElemChunks<'a, E: Entity> {
    pub(crate) inner: RowRef<'a, E>,
    pub(crate) policy: chunks::ChunkSizePolicy,
}
/// Chunked row iterator over the elements.
#[derive(Debug)]
pub struct RowElemChunksMut<'a, E: Entity> {
    pub(crate) inner: RowMut<'a, E>,
    pub(crate) policy: chunks::ChunkSizePolicy,
}
/// Chunked matrix iterator over the columns.
#[derive(Debug, Clone)]
pub struct ColChunks<'a, E: Entity> {
    pub(crate) inner: MatRef<'a, E>,
    pub(crate) policy: chunks::ChunkSizePolicy,
}
/// Chunked matrix iterator over the columns.
#[derive(Debug)]
pub struct ColChunksMut<'a, E: Entity> {
    pub(crate) inner: MatMut<'a, E>,
    pub(crate) policy: chunks::ChunkSizePolicy,
}
/// Chunked matrix iterator over the rows.
#[derive(Debug, Clone)]
pub struct RowChunks<'a, E: Entity> {
    pub(crate) inner: MatRef<'a, E>,
    pub(crate) policy: chunks::ChunkSizePolicy,
}
/// Chunked matrix iterator over the rows.
#[derive(Debug)]
pub struct RowChunksMut<'a, E: Entity> {
    pub(crate) inner: MatMut<'a, E>,
    pub(crate) policy: chunks::ChunkSizePolicy,
}

macro_rules! impl_chunk_iter {
    ($ty: ident, $item: ident, $dim: ident, $split: ident) => {
        impl<'a, E: Entity> Iterator for $ty<'a, E> {
            type Item = $item<'a, E>;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                let dim = self.inner.$dim();
                if dim == 0 {
                    None
                } else {
                    let size = self.policy.advance(dim);
                    let (head, tail) = core::mem::take(&mut self.inner).$split(size);
                    self.inner = tail;
                    Some(head)
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.policy.len(self.inner.$dim());
                (len, Some(len))
            }
        }
        impl<'a, E: Entity> DoubleEndedIterator for $ty<'a, E> {
            #[inline]
            fn next_back(&mut self) -> Option<Self::Item> {
                let dim = self.inner.$dim();
                if dim == 0 {
                    None
                } else {
                    let size = self.policy.advance_back(dim);
                    let (head, tail) = core::mem::take(&mut self.inner).$split(dim - size);
                    self.inner = head;
                    Some(tail)
                }
            }
        }
        impl<'a, E: Entity> ExactSizeIterator for $ty<'a, E> {}
    };
}

impl_chunk_iter!(ColChunks, MatRef, ncols, split_at_col);
impl_chunk_iter!(ColChunksMut, MatMut, ncols, split_at_col_mut);
impl_chunk_iter!(RowChunks, MatRef, nrows, split_at_row);
impl_chunk_iter!(RowChunksMut, MatMut, nrows, split_at_row_mut);
impl_chunk_iter!(ColElemChunks, ColRef, nrows, split_at);
impl_chunk_iter!(ColElemChunksMut, ColMut, nrows, split_at_mut);
impl_chunk_iter!(RowElemChunks, RowRef, ncols, split_at);
impl_chunk_iter!(RowElemChunksMut, RowMut, ncols, split_at_mut);

impl_chunk_iter!(ColPartition, MatRef, ncols, split_at_col);
impl_chunk_iter!(ColPartitionMut, MatMut, ncols, split_at_col_mut);
impl_chunk_iter!(RowPartition, MatRef, nrows, split_at_row);
impl_chunk_iter!(RowPartitionMut, MatMut, nrows, split_at_row_mut);
impl_chunk_iter!(ColElemPartition, ColRef, nrows, split_at);
impl_chunk_iter!(ColElemPartitionMut, ColMut, nrows, split_at_mut);
impl_chunk_iter!(RowElemPartition, RowRef, ncols, split_at);
impl_chunk_iter!(RowElemPartitionMut, RowMut, ncols, split_at_mut);
/// Iterator over the elements of a row or column.
#[derive(Debug, Clone)]
pub struct ElemIter<'a, E: Entity> {
    pub(crate) inner: ColRef<'a, E>,
}
/// Iterator over the elements of a row or column.
#[derive(Debug)]
pub struct ElemIterMut<'a, E: Entity> {
    pub(crate) inner: ColMut<'a, E>,
}

/// Iterator over the columns of a matrix.
#[derive(Debug, Clone)]
pub struct ColIter<'a, E: Entity, R: Shape = usize> {
    pub(crate) inner: MatRef<'a, E, R>,
}
/// Iterator over the columns of a matrix.
#[derive(Debug)]
pub struct ColIterMut<'a, E: Entity, R: Shape = usize> {
    pub(crate) inner: MatMut<'a, E, R>,
}
/// Iterator over the rows of a matrix.
#[derive(Debug, Clone)]
pub struct RowIter<'a, E: Entity, C: Shape = usize> {
    pub(crate) inner: MatRef<'a, E, usize, C>,
}
/// Iterator over the rows of a matrix.
#[derive(Debug)]
pub struct RowIterMut<'a, E: Entity, C: Shape = usize> {
    pub(crate) inner: MatMut<'a, E, usize, C>,
}

impl<'a, E: Entity> Iterator for ElemIter<'a, E> {
    type Item = GroupFor<E, &'a E::Unit>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match core::mem::take(&mut self.inner).split_first() {
            Some((head, tail)) => {
                self.inner = tail;
                Some(head)
            }
            None => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.inner.nrows(), Some(self.inner.nrows()))
    }
}

impl<'a, E: Entity> DoubleEndedIterator for ElemIter<'a, E> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        match core::mem::take(&mut self.inner).split_last() {
            Some((head, tail)) => {
                self.inner = tail;
                Some(head)
            }
            None => None,
        }
    }
}

impl<'a, E: Entity> ExactSizeIterator for ElemIter<'a, E> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.nrows()
    }
}

impl<'a, E: Entity> Iterator for ElemIterMut<'a, E> {
    type Item = GroupFor<E, &'a mut E::Unit>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match core::mem::take(&mut self.inner).split_first_mut() {
            Some((head, tail)) => {
                self.inner = tail;
                Some(head)
            }
            None => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.inner.nrows(), Some(self.inner.nrows()))
    }
}

impl<'a, E: Entity> DoubleEndedIterator for ElemIterMut<'a, E> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        match core::mem::take(&mut self.inner).split_last_mut() {
            Some((head, tail)) => {
                self.inner = tail;
                Some(head)
            }
            None => None,
        }
    }
}

impl<'a, E: Entity> ExactSizeIterator for ElemIterMut<'a, E> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.nrows()
    }
}

impl<'a, E: Entity, R: Shape> Iterator for ColIter<'a, E, R> {
    type Item = ColRef<'a, E, R>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let nrows = self.inner.nrows();
        match core::mem::replace(
            &mut self.inner,
            mat::from_column_major_slice_generic(
                E::faer_map(E::UNIT, |()| &[] as &[E::Unit]),
                nrows,
                0,
            ),
        )
        .split_first_col()
        {
            Some((head, tail)) => {
                self.inner = tail;
                Some(head)
            }
            None => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.inner.ncols(), Some(self.inner.ncols()))
    }
}
impl<'a, E: Entity, R: Shape> DoubleEndedIterator for ColIter<'a, E, R> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let nrows = self.inner.nrows();
        match core::mem::replace(
            &mut self.inner,
            mat::from_column_major_slice_generic(
                E::faer_map(E::UNIT, |()| &[] as &[E::Unit]),
                nrows,
                0,
            ),
        )
        .split_last_col()
        {
            Some((head, tail)) => {
                self.inner = tail;
                Some(head)
            }
            None => None,
        }
    }
}
impl<'a, E: Entity, R: Shape> ExactSizeIterator for ColIter<'a, E, R> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.ncols()
    }
}

impl<'a, E: Entity, R: Shape> Iterator for ColIterMut<'a, E, R> {
    type Item = ColMut<'a, E, R>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let nrows = self.inner.nrows();
        match core::mem::replace(
            &mut self.inner,
            mat::from_column_major_slice_mut_generic(
                E::faer_map(E::UNIT, |()| &mut [] as &mut [E::Unit]),
                nrows,
                0,
            ),
        )
        .split_first_col_mut()
        {
            Some((head, tail)) => {
                self.inner = tail;
                Some(head)
            }
            None => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.inner.ncols(), Some(self.inner.ncols()))
    }
}
impl<'a, E: Entity, R: Shape> DoubleEndedIterator for ColIterMut<'a, E, R> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let nrows = self.inner.nrows();
        match core::mem::replace(
            &mut self.inner,
            mat::from_column_major_slice_mut_generic(
                E::faer_map(E::UNIT, |()| &mut [] as &mut [E::Unit]),
                nrows,
                0,
            ),
        )
        .split_last_col_mut()
        {
            Some((head, tail)) => {
                self.inner = tail;
                Some(head)
            }
            None => None,
        }
    }
}
impl<'a, E: Entity, R: Shape> ExactSizeIterator for ColIterMut<'a, E, R> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.ncols()
    }
}

impl<'a, E: Entity> Iterator for RowIter<'a, E> {
    type Item = RowRef<'a, E>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match core::mem::take(&mut self.inner).split_first_row() {
            Some((head, tail)) => {
                self.inner = tail;
                Some(head)
            }
            None => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.inner.nrows(), Some(self.inner.nrows()))
    }
}
impl<'a, E: Entity> DoubleEndedIterator for RowIter<'a, E> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        match core::mem::take(&mut self.inner).split_last_row() {
            Some((head, tail)) => {
                self.inner = tail;
                Some(head)
            }
            None => None,
        }
    }
}
impl<'a, E: Entity> ExactSizeIterator for RowIter<'a, E> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.nrows()
    }
}

impl<'a, E: Entity> Iterator for RowIterMut<'a, E> {
    type Item = RowMut<'a, E>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match core::mem::take(&mut self.inner).split_first_row_mut() {
            Some((head, tail)) => {
                self.inner = tail;
                Some(head)
            }
            None => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.inner.nrows(), Some(self.inner.nrows()))
    }
}
impl<'a, E: Entity> DoubleEndedIterator for RowIterMut<'a, E> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        match core::mem::take(&mut self.inner).split_last_row_mut() {
            Some((head, tail)) => {
                self.inner = tail;
                Some(head)
            }
            None => None,
        }
    }
}
impl<'a, E: Entity> ExactSizeIterator for RowIterMut<'a, E> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.nrows()
    }
}

pub(crate) mod chunks {
    use crate::utils::DivCeil;

    #[derive(Copy, Clone, Debug)]
    pub struct ChunkSize(pub usize);

    #[derive(Copy, Clone, Debug)]
    pub struct PartitionCount(pub usize);

    #[derive(Copy, Clone, Debug)]
    pub struct ChunkSizePolicy {
        chunk_size: usize,
        rchunk_size: usize,
    }

    #[derive(Copy, Clone, Debug)]
    pub struct PartitionCountPolicy {
        count: usize,
        div: usize,
        rem: usize,
    }

    impl ChunkPolicy for ChunkSizePolicy {
        type Param = ChunkSize;

        #[inline]
        fn new(len: usize, param: Self::Param) -> Self {
            let mut rchunk_size = len % param.0;
            if rchunk_size == 0 {
                rchunk_size = param.0;
            }

            Self {
                chunk_size: param.0,
                rchunk_size,
            }
        }

        #[inline]
        fn advance(&mut self, len: usize) -> usize {
            Ord::min(len, self.chunk_size)
        }

        #[inline]
        fn advance_back(&mut self, len: usize) -> usize {
            _ = len;
            let size = self.rchunk_size;
            self.rchunk_size = self.chunk_size;
            Ord::min(len, size)
        }

        #[inline]
        fn len(&self, len: usize) -> usize {
            len.msrv_div_ceil(self.chunk_size)
        }
    }

    impl ChunkPolicy for PartitionCountPolicy {
        type Param = PartitionCount;

        #[inline]
        fn new(len: usize, param: Self::Param) -> Self {
            Self {
                count: param.0,
                div: len / param.0,
                rem: len % param.0,
            }
        }

        #[inline]
        fn advance(&mut self, len: usize) -> usize {
            _ = len;
            self.count -= 1;
            if self.rem == 0 {
                self.div
            } else {
                self.rem -= 1;
                self.div + 1
            }
        }

        #[inline]
        fn advance_back(&mut self, len: usize) -> usize {
            let count = self.count;
            self.count -= 1;
            if len == (self.div + 1) * count {
                self.rem -= 1;
                self.div + 1
            } else {
                self.div
            }
        }

        #[inline]
        fn len(&self, len: usize) -> usize {
            _ = len;
            self.count
        }
    }

    pub trait ChunkPolicy {
        type Param;

        fn new(len: usize, param: Self::Param) -> Self;
        fn advance(&mut self, len: usize) -> usize;
        fn advance_back(&mut self, len: usize) -> usize;
        fn len(&self, len: usize) -> usize;
    }
}
