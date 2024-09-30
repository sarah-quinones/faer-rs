use faer_entity::Entity;

pub trait Seal {}

impl<E: Entity> Seal for crate::mat::MatRef<'_, E> {}
impl<E: Entity> Seal for crate::mat::MatMut<'_, E> {}

impl<E: Entity> Seal for crate::col::ColRef<'_, E> {}
impl<E: Entity> Seal for crate::col::ColMut<'_, E> {}

impl<E: Entity> Seal for crate::row::RowRef<'_, E> {}
impl<E: Entity> Seal for crate::row::RowMut<'_, E> {}

impl Seal for i32 {}
impl Seal for i64 {}
impl Seal for i128 {}
impl Seal for isize {}
impl Seal for u32 {}
impl Seal for u64 {}
impl Seal for u128 {}
impl Seal for usize {}

impl Seal for crate::utils::bound::Dim<'_> {}
impl<I: crate::Index> Seal for crate::utils::bound::Idx<'_, I> {}
impl<I: crate::Index> Seal for crate::utils::bound::IdxInc<'_, I> {}
