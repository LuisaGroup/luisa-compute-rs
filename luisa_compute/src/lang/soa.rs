use crate::internal_prelude::*;
use crate::prelude::*;
use crate::runtime::submit_default_stream_and_sync;
use parking_lot::Mutex;
use std::ops::RangeBounds;
use std::sync::Arc;

use crate::runtime::Kernel;

use super::index::IntoIndex;
use super::types::SoaValue;
/** A buffer with SOA layout.
 */
pub struct SoaBuffer<T: SoaValue> {
    pub(crate) device: Device,
    pub(crate) storage: Arc<ByteBuffer>,
    pub(crate) metadata_buf: Buffer<SoaMetadata>,
    pub(crate) metadata: SoaMetadata,
    pub(crate) copy_kernel: Mutex<Option<SoaBufferCopyKernel<T>>>,
    pub(crate) _marker: std::marker::PhantomData<T>,
}
pub(crate) struct SoaBufferCopyKernel<T: SoaValue> {
    copy_to: Kernel<fn(SoaBuffer<T>, Buffer<T>, u64)>,
    copy_from: Kernel<fn(SoaBuffer<T>, Buffer<T>, u64)>,
}
impl<T: SoaValue> SoaBufferCopyKernel<T> {
    #[tracked]
    fn new(device: &Device) -> Self {
        let copy_to =
            device.create_kernel::<fn(SoaBuffer<T>, Buffer<T>, u64)>(&|soa, buf, offset| {
                let i = dispatch_id().x.as_u64() + offset;
                let v = soa.read(i);
                buf.write(i, v);
            });
        let copy_from =
            device.create_kernel::<fn(SoaBuffer<T>, Buffer<T>, u64)>(&|soa, buf, offset| {
                let i = dispatch_id().x.as_u64() + offset;
                let v = buf.read(i);
                soa.write(i, v);
            });
        Self { copy_to, copy_from }
    }
}
impl<T: SoaValue> SoaBuffer<T> {
    pub fn var(&self) -> SoaBufferVar<T> {
        self.view(..).var()
    }
    pub fn len(&self) -> usize {
        self.metadata.count as usize
    }
    pub fn len_expr(&self) -> Expr<u64> {
        self.metadata_buf.read(0).count
    }
    pub fn view<S: RangeBounds<usize>>(&self, range: S) -> SoaBufferView<T> {
        let lower = range.start_bound();
        let upper = range.end_bound();
        let lower = match lower {
            std::ops::Bound::Included(&x) => x,
            std::ops::Bound::Excluded(&x) => x + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let upper = match upper {
            std::ops::Bound::Included(&x) => x + 1,
            std::ops::Bound::Excluded(&x) => x,
            std::ops::Bound::Unbounded => self.len(),
        };
        assert!(lower <= upper);
        assert!(upper <= self.metadata.count as usize);
        let metadata = SoaMetadata {
            count: self.metadata.count,
            view_start: lower as u64,
            view_count: (upper - lower) as u64,
        };
        SoaBufferView {
            metadata_buf: self.device.create_buffer_from_slice(&[metadata]),
            metadata,
            buffer: self,
        }
    }
    pub fn copy_from_buffer_async(&self, buffer: &Buffer<T>) -> Command<'static, 'static> {
        self.view(..).copy_from_buffer_async(buffer)
    }
    pub fn copy_from_buffer(&self, buffer: &Buffer<T>) {
        self.view(..).copy_from_buffer(buffer)
    }
    pub fn copy_to_buffer_async(&self, buffer: &Buffer<T>) -> Command<'static, 'static> {
        self.view(..).copy_to_buffer_async(buffer)
    }
    pub fn copy_to_buffer(&self, buffer: &Buffer<T>) {
        self.view(..).copy_to_buffer(buffer)
    }
}
impl<'a, T: SoaValue> SoaBufferView<'a, T> {
    fn init_copy_kernel(&self) {
        let mut copy_kernel = self.buffer.copy_kernel.lock();
        if copy_kernel.is_none() {
            *copy_kernel = Some(SoaBufferCopyKernel::new(&self.buffer.device));
        }
    }
    pub fn var(&self) -> SoaBufferVar<T> {
        SoaBufferVar {
            proxy: T::SoaBuffer::from_soa_storage(
                self.buffer.storage.var(),
                self.metadata_buf.read(0),
                0,
            ),
        }
    }
    pub fn copy_from_buffer_async(&self, buffer: &Buffer<T>) -> Command<'static, 'static> {
        self.init_copy_kernel();
        let copy_kernel = self.buffer.copy_kernel.lock();
        let copy_kernel = copy_kernel.as_ref().unwrap();
        copy_kernel.copy_from.dispatch_async(
            [self.metadata.view_count as u32, 1, 1],
            self,
            buffer,
            &self.metadata.view_start,
        )
    }
    pub fn copy_from_buffer(&self, buffer: &Buffer<T>) {
        submit_default_stream_and_sync(&self.buffer.device, [self.copy_from_buffer_async(buffer)]);
    }
    pub fn copy_to_buffer_async(&self, buffer: &Buffer<T>) -> Command<'static, 'static> {
        self.init_copy_kernel();
        let copy_kernel = self.buffer.copy_kernel.lock();
        let copy_kernel = copy_kernel.as_ref().unwrap();
        copy_kernel.copy_to.dispatch_async(
            [self.metadata.view_count as u32, 1, 1],
            self,
            buffer,
            &self.metadata.view_start,
        )
    }
    pub fn copy_to_buffer(&self, buffer: &Buffer<T>) {
        submit_default_stream_and_sync(&self.buffer.device, [self.copy_to_buffer_async(buffer)]);
    }
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
pub struct SoaBufferView<'a, T: SoaValue> {
    pub(crate) metadata_buf: Buffer<SoaMetadata>,
    pub(crate) metadata: SoaMetadata,
    pub(crate) buffer: &'a SoaBuffer<T>,
}
pub struct SoaBufferVar<T: SoaValue> {
    pub(crate) proxy: T::SoaBuffer,
}
impl<T: SoaValue> IndexRead for SoaBufferVar<T> {
    type Element = T;
    fn read<I: IntoIndex>(&self, i: I) -> Expr<Self::Element> {
        self.proxy.read(i)
    }
}
impl<T: SoaValue> IndexWrite for SoaBufferVar<T> {
    fn write<I: IntoIndex, V: AsExpr<Value = Self::Element>>(&self, i: I, value: V) {
        self.proxy.write(i, value)
    }
}
