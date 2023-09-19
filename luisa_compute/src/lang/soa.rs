use luisa_compute_ir::{ir::Type, CArc};

use crate::prelude::*;
/** A buffer with SOA layout.
 */
pub struct SoaBuffer<T: Value> {
    inner: ByteBuffer,
    _marker: std::marker::PhantomData<T>,
}
