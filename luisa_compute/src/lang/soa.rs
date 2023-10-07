use crate::prelude::*;
use luisa_compute_ir::ir::Type;
use std::sync::Arc;

use super::types::SoaValue;
/** A buffer with SOA layout.
 */
pub struct SoaBuffer<T: SoaValue> {
    storage: Arc<ByteBuffer>,
    metadata: Buffer<SoaMetadata>,
    _marker: std::marker::PhantomData<T>,
}
#[derive(Clone, Copy, Value)]
#[repr(C)]
pub struct SoaMetadata {
    /// number of elements in the global buffer
    pub count: u64,
    /// number of elements in the view
    pub view_start: u64,
    pub view_count: u64,
}
pub(crate) struct SoaStorage {
    data: Arc<ByteBuffer>,
}
pub struct SoaBufferView<'a, T: SoaValue> {
    metadata: Buffer<SoaMetadata>,
    buffer: &'a SoaBuffer<T>,
}
pub struct SoaBufferVar<T: SoaValue> {
    proxy: T::SoaBuffer,
}

fn compute_number_of_32bits_buffers(ty: &Type) -> usize {
    (ty.size() + 3) / 4
}

// impl<T> IndexRead for SoaBuffer<T>
// where
//     T: Value,
// {
//     type Element = T;
//     fn read<I: super::index::IntoIndex>(&self, i: I) -> Expr<Self::Element> {
//         let i = i.to_u64();
//         todo!()
//     }
// }
