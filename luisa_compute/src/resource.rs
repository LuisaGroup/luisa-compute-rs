use std::cell::{Cell, RefCell};
use std::ops::RangeBounds;
use std::process::abort;
use std::sync::Arc;

use parking_lot::lock_api::RawMutex as RawMutexTrait;
use parking_lot::RawMutex;

use crate::internal_prelude::*;

use crate::lang::index::IntoIndex;
use crate::runtime::*;

use api::{BufferDownloadCommand, BufferUploadCommand, INVALID_RESOURCE_HANDLE};
use libc::c_void;

pub struct ByteBuffer {
    pub(crate) device: Device,
    pub(crate) handle: Arc<BufferHandle>,
    pub(crate) len: usize,
}
impl ByteBuffer {
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn handle(&self) -> api::Buffer {
        self.handle.handle
    }
    #[inline]
    pub fn native_handle(&self) -> *mut c_void {
        self.handle.native_handle
    }
    #[inline]
    pub fn copy_from(&self, data: &[u8]) {
        self.view(..).copy_from(data);
    }
    #[inline]
    pub fn copy_from_async<'a>(&self, data: &[u8]) -> Command<'_> {
        self.view(..).copy_from_async(data)
    }
    #[inline]
    pub fn copy_to(&self, data: &mut [u8]) {
        self.view(..).copy_to(data);
    }
    #[inline]
    pub fn copy_to_async<'a>(&self, data: &'a mut [u8]) -> Command<'a> {
        self.view(..).copy_to_async(data)
    }
    #[inline]
    pub fn copy_to_vec(&self) -> Vec<u8> {
        self.view(..).copy_to_vec()
    }
    #[inline]
    pub fn copy_to_buffer(&self, dst: &ByteBuffer) {
        self.view(..).copy_to_buffer(dst.view(..));
    }
    #[inline]
    pub fn copy_to_buffer_async<'a>(&'a self, dst: &'a ByteBuffer) -> Command<'a> {
        self.view(..).copy_to_buffer_async(dst.view(..))
    }
    #[inline]
    pub fn fill_fn<F: FnMut(usize) -> u8>(&self, f: F) {
        self.view(..).fill_fn(f);
    }
    #[inline]
    pub fn fill(&self, value: u8) {
        self.view(..).fill(value);
    }
    pub fn view<S: RangeBounds<usize>>(&self, range: S) -> ByteBufferView<'_> {
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
            std::ops::Bound::Unbounded => self.len,
        };
        assert!(lower <= upper);
        assert!(upper <= self.len);
        ByteBufferView {
            buffer: self,
            offset: lower,
            len: upper - lower,
        }
    }
    pub fn var(&self) -> ByteBufferVar {
        ByteBufferVar::new(&self.view(..))
    }
}
pub struct ByteBufferView<'a> {
    pub(crate) buffer: &'a ByteBuffer,
    pub(crate) offset: usize,
    pub(crate) len: usize,
}
impl<'a> ByteBufferView<'a> {
    pub fn handle(&self) -> api::Buffer {
        self.buffer.handle()
    }
    pub fn copy_to_async<'b>(&'a self, data: &'b mut [u8]) -> Command<'b> {
        assert_eq!(data.len(), self.len);
        let mut rt = ResourceTracker::new();
        rt.add(self.buffer.handle.clone());
        Command {
            inner: api::Command::BufferDownload(BufferDownloadCommand {
                buffer: self.handle(),
                offset: self.offset,
                size: data.len(),
                data: data.as_mut_ptr() as *mut u8,
            }),
            marker: PhantomData,
            resource_tracker: rt,
            callback: None,
        }
    }
    pub fn copy_to_vec(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(self.len);
        unsafe {
            let slice = std::slice::from_raw_parts_mut(data.as_mut_ptr(), self.len);
            self.copy_to(slice);
            data.set_len(self.len);
        }
        data
    }
    pub fn copy_to(&self, data: &mut [u8]) {
        unsafe {
            submit_default_stream_and_sync(&self.buffer.device, [self.copy_to_async(data)]);
        }
    }

    pub fn copy_from_async<'b>(&'a self, data: &'b [u8]) -> Command<'static> {
        assert_eq!(data.len(), self.len);
        let mut rt = ResourceTracker::new();
        rt.add(self.buffer.handle.clone());
        Command {
            inner: api::Command::BufferUpload(BufferUploadCommand {
                buffer: self.handle(),
                offset: self.offset,
                size: data.len(),
                data: data.as_ptr() as *const u8,
            }),
            marker: PhantomData,
            resource_tracker: rt,
            callback: None,
        }
    }
    pub fn copy_from(&self, data: &[u8]) {
        submit_default_stream_and_sync(&self.buffer.device, [self.copy_from_async(data)]);
    }
    pub fn fill_fn<F: FnMut(usize) -> u8>(&self, f: F) {
        self.copy_from(&(0..self.len).map(f).collect::<Vec<_>>());
    }
    pub fn fill(&self, value: u8) {
        self.fill_fn(|_| value);
    }
    pub fn copy_to_buffer_async(&self, dst: ByteBufferView<'a>) -> Command<'static> {
        assert_eq!(self.len, dst.len);
        let mut rt = ResourceTracker::new();
        rt.add(self.buffer.handle.clone());
        rt.add(dst.buffer.handle.clone());
        Command {
            inner: api::Command::BufferCopy(api::BufferCopyCommand {
                src: self.handle(),
                src_offset: self.offset,
                dst: dst.handle(),
                dst_offset: dst.offset,
                size: self.len,
            }),
            marker: PhantomData,
            resource_tracker: rt,
            callback: None,
        }
    }
    pub fn copy_to_buffer(&self, dst: ByteBufferView<'a>) {
        submit_default_stream_and_sync(&self.buffer.device, [self.copy_to_buffer_async(dst)]);
    }
}
#[derive(Clone)]
pub struct ByteBufferVar {
    #[allow(dead_code)]
    pub(crate) handle: Option<Arc<BufferHandle>>,
    pub(crate) node: NodeRef,
}
impl ByteBufferVar {
    pub fn new(buffer: &ByteBufferView<'_>) -> Self {
        let node = RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            assert!(r.lock, "BufferVar must be created from within a kernel");
            let binding = Binding::Buffer(BufferBinding {
                handle: buffer.handle().0,
                size: buffer.len,
                offset: buffer.offset as u64,
            });
            if let Some((_, node, _, _)) = r.captured_buffer.get(&binding) {
                *node
            } else {
                let node = new_node(
                    r.pools.as_ref().unwrap(),
                    Node::new(CArc::new(Instruction::Buffer), Type::void()),
                );
                let i = r.captured_buffer.len();
                r.captured_buffer
                    .insert(binding, (i, node, binding, buffer.buffer.handle.clone()));
                node
            }
        });
        Self {
            node,
            handle: Some(buffer.buffer.handle.clone()),
        }
    }
    pub fn read<T: Value>(&self, index_bytes: impl IntoIndex) -> Expr<T> {
        let i = index_bytes.to_u64();
        Expr::<T>::from_node(__current_scope(|b| {
            b.call(
                Func::ByteBufferRead,
                &[self.node, i.node],
                <T as TypeOf>::type_(),
            )
        }))
    }
    pub fn len(&self) -> Expr<u64> {
        Expr::<u64>::from_node(__current_scope(|b| {
            b.call(Func::ByteBufferSize, &[self.node], <u64 as TypeOf>::type_())
        }))
    }
    pub fn write<T: Value>(&self, index_bytes: impl IntoIndex, value: impl Into<Expr<T>>) {
        let i = index_bytes.to_u64();
        let value: Expr<T> = value.into();
        __current_scope(|b| {
            b.call(
                Func::ByteBufferWrite,
                &[self.node, i.node, value.node()],
                Type::void(),
            )
        });
    }
}
pub struct Buffer<T: Value> {
    pub(crate) device: Device,
    pub(crate) handle: Arc<BufferHandle>,
    pub(crate) len: usize,
    pub(crate) _marker: PhantomData<T>,
}
pub(crate) struct BufferHandle {
    pub(crate) device: Device,
    pub(crate) handle: api::Buffer,
    pub(crate) native_handle: *mut c_void,
}
unsafe impl Send for BufferHandle {}
unsafe impl Sync for BufferHandle {}

impl Drop for BufferHandle {
    fn drop(&mut self) {
        self.device.inner.destroy_buffer(self.handle);
    }
}
#[derive(Clone, Copy)]
pub struct BufferView<'a, T: Value> {
    pub(crate) buffer: &'a Buffer<T>,
    pub(crate) offset: usize,
    pub(crate) len: usize,
}
impl<'a, T: Value> BufferView<'a, T> {
    pub fn var(&self) -> BufferVar<T> {
        BufferVar::new(self)
    }
    pub fn handle(&self) -> api::Buffer {
        self.buffer.handle()
    }
    pub fn copy_to_async<'b>(&'a self, data: &'b mut [T]) -> Command<'b> {
        assert_eq!(data.len(), self.len);
        let mut rt = ResourceTracker::new();
        rt.add(self.buffer.handle.clone());
        Command {
            inner: api::Command::BufferDownload(BufferDownloadCommand {
                buffer: self.handle(),
                offset: self.offset * std::mem::size_of::<T>(),
                size: data.len() * std::mem::size_of::<T>(),
                data: data.as_mut_ptr() as *mut u8,
            }),
            marker: PhantomData,
            resource_tracker: rt,
            callback: None,
        }
    }
    pub fn copy_to_vec(&self) -> Vec<T> {
        let mut data = Vec::with_capacity(self.len);
        unsafe {
            let slice = std::slice::from_raw_parts_mut(data.as_mut_ptr(), self.len);
            self.copy_to(slice);
            data.set_len(self.len);
        }
        data
    }
    pub fn copy_to(&self, data: &mut [T]) {
        unsafe {
            submit_default_stream_and_sync(&self.buffer.device, [self.copy_to_async(data)]);
        }
    }

    pub fn copy_from_async<'b>(&'a self, data: &'b [T]) -> Command<'static> {
        assert_eq!(data.len(), self.len);
        let mut rt = ResourceTracker::new();
        rt.add(self.buffer.handle.clone());
        Command {
            inner: api::Command::BufferUpload(BufferUploadCommand {
                buffer: self.handle(),
                offset: self.offset * std::mem::size_of::<T>(),
                size: data.len() * std::mem::size_of::<T>(),
                data: data.as_ptr() as *const u8,
            }),
            marker: PhantomData,
            resource_tracker: rt,
            callback: None,
        }
    }
    pub fn copy_from(&self, data: &[T]) {
        submit_default_stream_and_sync(&self.buffer.device, [self.copy_from_async(data)]);
    }
    pub fn fill_fn<F: FnMut(usize) -> T>(&self, f: F) {
        self.copy_from(&(0..self.len).map(f).collect::<Vec<_>>());
    }
    pub fn fill(&self, value: T) {
        self.fill_fn(|_| value);
    }
    pub fn copy_to_buffer_async(&self, dst: BufferView<'a, T>) -> Command<'static> {
        assert_eq!(self.len, dst.len);
        let mut rt = ResourceTracker::new();
        rt.add(self.buffer.handle.clone());
        rt.add(dst.buffer.handle.clone());
        Command {
            inner: api::Command::BufferCopy(api::BufferCopyCommand {
                src: self.handle(),
                src_offset: self.offset * std::mem::size_of::<T>(),
                dst: dst.handle(),
                dst_offset: dst.offset * std::mem::size_of::<T>(),
                size: self.len * std::mem::size_of::<T>(),
            }),
            marker: PhantomData,
            resource_tracker: rt,
            callback: None,
        }
    }
    pub fn copy_to_buffer(&self, dst: BufferView<T>) {
        submit_default_stream_and_sync(&self.buffer.device, [self.copy_to_buffer_async(dst)]);
    }
}
impl<T: Value> Buffer<T> {
    #[inline]
    pub fn handle(&self) -> api::Buffer {
        self.handle.handle
    }
    #[inline]
    pub unsafe fn shallow_clone(&self) -> Buffer<T> {
        Buffer {
            device: self.device.clone(),
            handle: self.handle.clone(),
            len: self.len,
            _marker: PhantomData,
        }
    }
    #[inline]
    pub fn native_handle(&self) -> *mut c_void {
        self.handle.native_handle
    }
    #[inline]
    pub fn copy_from(&self, data: &[T]) {
        self.view(..).copy_from(data);
    }
    #[inline]
    pub fn copy_from_async<'a>(&self, data: &[T]) -> Command<'_> {
        self.view(..).copy_from_async(data)
    }
    #[inline]
    pub fn copy_to(&self, data: &mut [T]) {
        self.view(..).copy_to(data);
    }
    #[inline]
    pub fn copy_to_async<'a>(&self, data: &'a mut [T]) -> Command<'a> {
        self.view(..).copy_to_async(data)
    }
    #[inline]
    pub fn copy_to_vec(&self) -> Vec<T> {
        self.view(..).copy_to_vec()
    }
    #[inline]
    pub fn copy_to_buffer(&self, dst: &Buffer<T>) {
        self.view(..).copy_to_buffer(dst.view(..));
    }
    #[inline]
    pub fn copy_to_buffer_async<'a>(&'a self, dst: &'a Buffer<T>) -> Command<'a> {
        self.view(..).copy_to_buffer_async(dst.view(..))
    }
    #[inline]
    pub fn fill_fn<F: FnMut(usize) -> T>(&self, f: F) {
        self.view(..).fill_fn(f);
    }
    #[inline]
    pub fn fill(&self, value: T) {
        self.view(..).fill(value);
    }
    pub fn view<S: RangeBounds<usize>>(&self, range: S) -> BufferView<T> {
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
            std::ops::Bound::Unbounded => self.len,
        };
        assert!(lower <= upper);
        assert!(upper <= self.len);
        BufferView {
            buffer: self,
            offset: lower,
            len: upper - lower,
        }
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
    #[inline]
    pub fn var(&self) -> BufferVar<T> {
        BufferVar::new(&self.view(..))
    }
}
impl<T: Value> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        let cloned = self.device.create_buffer(self.len);
        self.copy_to_buffer(&cloned);
        cloned
    }
}
pub(crate) struct BindlessArrayHandle {
    pub(crate) device: Device,
    pub(crate) handle: api::BindlessArray,
    pub(crate) native_handle: *mut c_void,
}
impl Drop for BindlessArrayHandle {
    fn drop(&mut self) {
        self.device.inner.destroy_bindless_array(self.handle);
    }
}
#[derive(Clone)]
pub(crate) struct BindlessArraySlot {
    pub(crate) buffer: Option<Arc<BufferHandle>>,
    pub(crate) tex2d: Option<Arc<TextureHandle>>,
    pub(crate) tex3d: Option<Arc<TextureHandle>>,
}
pub struct BufferHeap<T: Value> {
    pub(crate) inner: BindlessArray,
    pub(crate) _marker: PhantomData<T>,
}
pub struct BufferHeapVar<T: Value> {
    inner: BindlessArrayVar,
    _marker: PhantomData<T>,
}
impl<T: Value> BufferHeap<T> {
    #[inline]
    pub fn var(&self) -> BufferHeapVar<T> {
        BufferHeapVar {
            inner: self.inner.var(),
            _marker: PhantomData,
        }
    }
    #[inline]
    pub fn handle(&self) -> api::BindlessArray {
        self.inner.handle()
    }
    #[inline]
    pub fn native_handle(&self) -> *mut std::ffi::c_void {
        self.inner.native_handle()
    }
    pub fn emplace_buffer_async(&self, index: usize, buffer: &Buffer<T>) {
        self.inner.emplace_buffer_async(index, buffer);
    }
    pub fn emplace_buffer_view_async<'a>(&self, index: usize, bufferview: &BufferView<'a, T>) {
        self.inner.emplace_buffer_view_async(index, bufferview);
    }
    pub fn remove_buffer_async(&self, index: usize) {
        self.inner.remove_buffer_async(index);
    }
    #[inline]
    pub fn emplace_buffer(&self, index: usize, buffer: &Buffer<T>) {
        self.inner.emplace_buffer(index, buffer);
    }
    #[inline]
    pub fn emplace_buffer_view<'a>(&self, index: usize, bufferview: &BufferView<'a, T>) {
        self.inner.emplace_buffer_view_async(index, bufferview);
    }
    #[inline]
    pub fn remove_buffer(&self, index: usize) {
        self.inner.remove_buffer(index);
    }
    #[inline]
    pub fn update(&self) {
        self.inner.update();
    }
    #[inline]
    pub fn buffer(&self, index: impl Into<Expr<u32>>) -> BindlessBufferVar<T> {
        self.inner.buffer(index)
    }
}
impl<T: Value> BufferHeapVar<T> {
    #[inline]
    pub fn buffer(&self, index: impl Into<Expr<u32>>) -> BindlessBufferVar<T> {
        self.inner.buffer(index)
    }
}
pub struct BindlessArray {
    pub(crate) device: Device,
    pub(crate) handle: Arc<BindlessArrayHandle>,
    pub(crate) modifications: RefCell<Vec<api::BindlessArrayUpdateModification>>,
    pub(crate) slots: RefCell<Vec<BindlessArraySlot>>,
    pub(crate) pending_slots: RefCell<Vec<BindlessArraySlot>>,
    pub(crate) lock: Arc<RawMutex>,
    pub(crate) dirty: Cell<bool>,
}
impl BindlessArray {
    #[inline]
    fn lock(&self) {
        self.lock.lock();
    }
    #[inline]
    fn unlock(&self) {
        unsafe {
            self.lock.unlock();
        }
    }
    #[inline]
    fn make_pending_slots(&self) {
        let mut pending = self.pending_slots.borrow_mut();
        if self.dirty.get() {
            let mut slots = self.slots.borrow_mut();
            *slots = pending.clone();
            self.dirty.set(false);
        }
        if pending.is_empty() {
            *pending = self.slots.borrow().clone();
        }
    }
    #[inline]
    pub fn var(&self) -> BindlessArrayVar {
        BindlessArrayVar::new(self)
    }
    #[inline]
    pub fn handle(&self) -> api::BindlessArray {
        self.handle.handle
    }
    #[inline]
    pub fn native_handle(&self) -> *mut std::ffi::c_void {
        self.handle.native_handle
    }
    pub fn emplace_byte_buffer_async(&self, index: usize, buffer: &ByteBuffer) {
        self.emplace_byte_buffer_view_async(index, &buffer.view(..))
    }
    pub fn emplace_byte_buffer_view_async<'a>(
        &self,
        index: usize,
        bufferview: &ByteBufferView<'a>,
    ) {
        self.lock();
        self.modifications
            .borrow_mut()
            .push(api::BindlessArrayUpdateModification {
                slot: index,
                buffer: api::BindlessArrayUpdateBuffer {
                    op: api::BindlessArrayUpdateOperation::Emplace,
                    handle: bufferview.handle(),
                    offset: bufferview.offset,
                },
                tex2d: api::BindlessArrayUpdateTexture::default(),
                tex3d: api::BindlessArrayUpdateTexture::default(),
            });
        self.make_pending_slots();
        let mut pending = self.pending_slots.borrow_mut();
        pending[index].buffer = Some(bufferview.buffer.handle.clone());
        self.unlock();
    }
    pub fn emplace_buffer_async<T: Value>(&self, index: usize, buffer: &Buffer<T>) {
        self.emplace_buffer_view_async(index, &buffer.view(..))
    }
    pub fn emplace_buffer_view_async<'a, T: Value>(
        &self,
        index: usize,
        bufferview: &BufferView<'a, T>,
    ) {
        self.lock();
        self.modifications
            .borrow_mut()
            .push(api::BindlessArrayUpdateModification {
                slot: index,
                buffer: api::BindlessArrayUpdateBuffer {
                    op: api::BindlessArrayUpdateOperation::Emplace,
                    handle: bufferview.handle(),
                    offset: bufferview.offset,
                },
                tex2d: api::BindlessArrayUpdateTexture::default(),
                tex3d: api::BindlessArrayUpdateTexture::default(),
            });
        self.make_pending_slots();
        let mut pending = self.pending_slots.borrow_mut();
        pending[index].buffer = Some(bufferview.buffer.handle.clone());
        self.unlock();
    }
    pub fn emplace_tex2d_async<T: IoTexel>(
        &self,
        index: usize,
        texture: &Tex2d<T>,
        sampler: Sampler,
    ) {
        self.lock();
        self.modifications
            .borrow_mut()
            .push(api::BindlessArrayUpdateModification {
                slot: index,
                buffer: api::BindlessArrayUpdateBuffer::default(),
                tex2d: api::BindlessArrayUpdateTexture {
                    op: api::BindlessArrayUpdateOperation::Emplace,
                    handle: texture.handle(),
                    sampler,
                },
                tex3d: api::BindlessArrayUpdateTexture::default(),
            });
        self.make_pending_slots();
        let mut pending = self.pending_slots.borrow_mut();
        pending[index].tex2d = Some(texture.handle.clone());
        self.unlock();
    }
    pub fn emplace_tex3d_async<T: IoTexel>(
        &self,
        index: usize,
        texture: &Tex3d<T>,
        sampler: Sampler,
    ) {
        self.lock();
        self.modifications
            .borrow_mut()
            .push(api::BindlessArrayUpdateModification {
                slot: index,
                buffer: api::BindlessArrayUpdateBuffer::default(),
                tex2d: api::BindlessArrayUpdateTexture::default(),
                tex3d: api::BindlessArrayUpdateTexture {
                    op: api::BindlessArrayUpdateOperation::Emplace,
                    handle: texture.handle(),
                    sampler,
                },
            });
        self.make_pending_slots();
        let mut pending = self.pending_slots.borrow_mut();
        pending[index].tex3d = Some(texture.handle.clone());
        self.unlock();
    }
    pub fn remove_buffer_async(&self, index: usize) {
        self.lock();
        self.modifications
            .borrow_mut()
            .push(api::BindlessArrayUpdateModification {
                slot: index,
                buffer: api::BindlessArrayUpdateBuffer {
                    op: api::BindlessArrayUpdateOperation::Remove,
                    handle: api::Buffer(INVALID_RESOURCE_HANDLE),
                    offset: 0,
                },
                tex2d: api::BindlessArrayUpdateTexture::default(),
                tex3d: api::BindlessArrayUpdateTexture::default(),
            });
        self.make_pending_slots();
        let mut pending = self.pending_slots.borrow_mut();
        pending[index].buffer = None;
        self.unlock();
    }
    pub fn remove_tex2d_async(&self, index: usize) {
        self.lock();
        self.modifications
            .borrow_mut()
            .push(api::BindlessArrayUpdateModification {
                slot: index,
                buffer: api::BindlessArrayUpdateBuffer::default(),
                tex2d: api::BindlessArrayUpdateTexture {
                    op: api::BindlessArrayUpdateOperation::Remove,
                    handle: api::Texture(INVALID_RESOURCE_HANDLE),
                    sampler: Sampler::default(),
                },
                tex3d: api::BindlessArrayUpdateTexture::default(),
            });
        self.make_pending_slots();
        let mut pending = self.pending_slots.borrow_mut();
        pending[index].tex2d = None;
        self.unlock();
    }
    pub fn remove_tex3d_async(&self, index: usize) {
        self.lock();
        self.modifications
            .borrow_mut()
            .push(api::BindlessArrayUpdateModification {
                slot: index,
                buffer: api::BindlessArrayUpdateBuffer::default(),
                tex2d: api::BindlessArrayUpdateTexture::default(),
                tex3d: api::BindlessArrayUpdateTexture {
                    op: api::BindlessArrayUpdateOperation::Remove,
                    handle: api::Texture(INVALID_RESOURCE_HANDLE),
                    sampler: Sampler::default(),
                },
            });
        self.make_pending_slots();
        let mut pending = self.pending_slots.borrow_mut();
        pending[index].tex3d = None;
        self.unlock();
    }
    #[inline]
    pub fn emplace_byte_buffer(&self, index: usize, buffer: &ByteBuffer) {
        self.emplace_byte_buffer_async(index, buffer);
        self.update();
    }
    #[inline]
    pub fn emplace_byte_buffer_view(&self, index: usize, buffer: &ByteBufferView<'_>) {
        self.emplace_byte_buffer_view_async(index, buffer);
        self.update();
    }
    #[inline]
    pub fn emplace_buffer<T: Value>(&self, index: usize, buffer: &Buffer<T>) {
        self.emplace_buffer_async(index, buffer);
        self.update();
    }
    #[inline]
    pub fn emplace_buffer_view<T: Value>(&self, index: usize, buffer: &BufferView<T>) {
        self.emplace_buffer_view_async(index, buffer);
        self.update();
    }
    #[inline]
    pub fn set_tex2d<T: IoTexel>(&self, index: usize, texture: &Tex2d<T>, sampler: Sampler) {
        self.emplace_tex2d_async(index, texture, sampler);
        self.update();
    }
    #[inline]
    pub fn set_tex3d<T: IoTexel>(&self, index: usize, texture: &Tex3d<T>, sampler: Sampler) {
        self.emplace_tex3d_async(index, texture, sampler);
        self.update();
    }
    #[inline]
    pub fn remove_buffer(&self, index: usize) {
        self.remove_buffer_async(index);
        self.update();
    }
    #[inline]
    pub fn remove_tex2d(&self, index: usize) {
        self.remove_tex2d_async(index);
        self.update();
    }
    #[inline]
    pub fn remove_tex3d(&self, index: usize) {
        self.remove_tex3d_async(index);
        self.update();
    }
    #[inline]
    pub fn update(&self) {
        submit_default_stream_and_sync(&self.device, [self.update_async()]);
    }
    pub fn update_async<'a>(&'a self) -> Command<'a> {
        self.lock();
        let mut rt = ResourceTracker::new();
        let modifications = Arc::new(std::mem::replace(
            &mut *self.modifications.borrow_mut(),
            Vec::new(),
        ));
        rt.add(modifications.clone());
        self.dirty.set(true);
        let lock = self.lock.clone();
        Command {
            inner: api::Command::BindlessArrayUpdate(api::BindlessArrayUpdateCommand {
                handle: self.handle.handle,
                modifications: modifications.as_ptr(),
                modifications_count: modifications.len(),
            }),
            marker: PhantomData,
            resource_tracker: rt,
            callback: Some(Box::new(move || unsafe {
                lock.unlock();
            })),
        }
    }
    pub fn tex2d(&self, tex2d_index: impl Into<Expr<u32>>) -> BindlessTex2dVar {
        self.var().tex2d(tex2d_index)
    }
    pub fn tex3d(&self, tex3d_index: impl Into<Expr<u32>>) -> BindlessTex3dVar {
        self.var().tex3d(tex3d_index)
    }
    pub fn buffer<T: Value>(&self, buffer_index: impl Into<Expr<u32>>) -> BindlessBufferVar<T> {
        self.var().buffer::<T>(buffer_index)
    }
}
unsafe impl Send for BindlessArray {}
unsafe impl Sync for BindlessArray {}
pub use api::{PixelFormat, PixelStorage, Sampler, SamplerAddress, SamplerFilter};
use luisa_compute_ir::context::type_hash;
use luisa_compute_ir::ir::{
    new_node, Binding, BindlessArrayBinding, BufferBinding, Func, Instruction, Node, TextureBinding,
};

pub(crate) struct TextureHandle {
    pub(crate) device: Device,
    pub(crate) handle: api::Texture,
    pub(crate) native_handle: *mut std::ffi::c_void,
    #[allow(dead_code)]
    pub(crate) format: PixelFormat,
    pub(crate) storage: PixelStorage,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) depth: u32,
    #[allow(dead_code)]
    pub(crate) levels: u32,
}
unsafe impl Send for TextureHandle {}
unsafe impl Sync for TextureHandle {}
trait GetPixelFormat {
    fn pixel_format(storage: PixelStorage) -> PixelFormat;
}
impl GetPixelFormat for f32 {
    fn pixel_format(storage: PixelStorage) -> PixelFormat {
        match storage {
            PixelStorage::Byte1 => PixelFormat::R8Unorm,
            PixelStorage::Byte2 => PixelFormat::Rg8Unorm,
            PixelStorage::Byte4 => PixelFormat::Rgba8Unorm,
            PixelStorage::Half1 => PixelFormat::Rg16f,
            PixelStorage::Half2 => PixelFormat::Rg16f,
            PixelStorage::Half4 => PixelFormat::Rgba16f,
            PixelStorage::Short1 => PixelFormat::R16Unorm,
            PixelStorage::Short2 => PixelFormat::Rg16Unorm,
            PixelStorage::Short4 => PixelFormat::Rgba16Unorm,
            PixelStorage::Float1 => PixelFormat::R32f,
            PixelStorage::Float2 => PixelFormat::Rg32f,
            PixelStorage::Float4 => PixelFormat::Rgba32f,
            _ => panic!("Invalid pixel storage for f32"),
        }
    }
}
impl GetPixelFormat for i32 {
    fn pixel_format(storage: PixelStorage) -> PixelFormat {
        match storage {
            PixelStorage::Byte1 => PixelFormat::R8Sint,
            PixelStorage::Byte2 => PixelFormat::Rg8Sint,
            PixelStorage::Byte4 => PixelFormat::Rgba8Sint,
            PixelStorage::Short1 => PixelFormat::R16Sint,
            PixelStorage::Short2 => PixelFormat::Rg16Sint,
            PixelStorage::Short4 => PixelFormat::Rgba16Sint,
            PixelStorage::Int1 => PixelFormat::R32Sint,
            PixelStorage::Int2 => PixelFormat::Rg32Sint,
            PixelStorage::Int4 => PixelFormat::Rgba32Sint,
            _ => panic!("Invalid pixel storage for i32"),
        }
    }
}
impl GetPixelFormat for u32 {
    fn pixel_format(storage: PixelStorage) -> PixelFormat {
        match storage {
            PixelStorage::Byte1 => PixelFormat::R8Uint,
            PixelStorage::Byte2 => PixelFormat::Rg8Uint,
            PixelStorage::Byte4 => PixelFormat::Rgba8Uint,
            PixelStorage::Short1 => PixelFormat::R16Uint,
            PixelStorage::Short2 => PixelFormat::Rg16Uint,
            PixelStorage::Short4 => PixelFormat::Rgba16Uint,
            PixelStorage::Int1 => PixelFormat::R32Uint,
            PixelStorage::Int2 => PixelFormat::Rg32Uint,
            PixelStorage::Int4 => PixelFormat::Rgba32Uint,
            _ => panic!("Invalid pixel storage for u32"),
        }
    }
}
// Type that can be converted from a pixel format
// This is the type that is read from/written to a texture
pub trait IoTexel: Value {
    type RwType: Value;
    fn pixel_format(storage: PixelStorage) -> PixelFormat;
    fn convert_from_read(texel: Expr<Self::RwType>) -> Expr<Self>;
    fn convert_to_write(value: Expr<Self>) -> Expr<Self::RwType>;
}

macro_rules! impl_io_texel {
    ($t:ty,$el:ty, $rw:ty, $cvt_from:expr, $cvt_to:expr) => {
        impl IoTexel for $t {
            type RwType = $rw;
            fn pixel_format(storage: PixelStorage) -> PixelFormat {
                <$el as GetPixelFormat>::pixel_format(storage)
            }
            fn convert_from_read(texel: Expr<Self::RwType>) -> Expr<Self> {
                ($cvt_from)(texel)
            }
            fn convert_to_write(value: Expr<Self>) -> Expr<Self::RwType> {
                ($cvt_to)(value)
            }
        }
    };
}
impl_io_texel!(f32, f32, Float4, |x: Expr<Float4>| x.x, |x| {
    Float4::splat_expr(x)
});
impl_io_texel!(Float2, f32, Float4, |x: Expr<Float4>| x.xy(), |x: Expr<
    Float2,
>| {
    Float4::expr(x.x, x.y, 0.0, 0.0)
});
impl_io_texel!(Float4, f32, Float4, |x: Expr<Float4>| x, |x: Expr<
    Float4,
>| x);

// impl_io_texel!(u16,);
// impl_io_texel!(i16,);
// impl_io_texel!(Ushort2,);
// impl_io_texel!(Short2,);
// impl_io_texel!(Ushort4,);
// impl_io_texel!(Short4,);
impl_io_texel!(
    u32,
    u32,
    Uint4,
    |x: Expr<Uint4>| x.x,
    |x| Uint4::splat_expr(x)
);
impl_io_texel!(i32, i32, Int4, |x: Expr<Int4>| x.x, |x| Int4::splat_expr(x));
impl_io_texel!(Uint2, u32, Uint4, |x: Expr<Uint4>| x.xy(), |x: Expr<
    Uint2,
>| {
    Uint4::expr(x.x, x.y, 0u32, 0u32)
});
impl_io_texel!(Int2, i32, Int4, |x: Expr<Int4>| x.xy(), |x: Expr<Int2>| {
    Int4::expr(x.x, x.y, 0i32, 0i32)
});
impl_io_texel!(Uint4, u32, Uint4, |x: Expr<Uint4>| x, |x| x);
impl_io_texel!(Int4, i32, Int4, |x: Expr<Int4>| x, |x| x);

// Types that is stored in a texture
pub trait StorageTexel<T: IoTexel> {
    fn pixel_storage() -> PixelStorage;
}
macro_rules! impl_storage_texel {
    ($t:ty, $st:ident, $t2:ident,) => {
        impl StorageTexel<$t2> for $t {
            fn pixel_storage() -> PixelStorage {
                PixelStorage::$st
            }
        }
    };
    ($t:ty, $st:ident, $t2:ident, $($rest:ident,) *) => {
        impl_storage_texel!($t, $st, $t2, );
        impl_storage_texel!($t, $st, $($rest, )*);
    };
}

impl_storage_texel!(u8, Byte1, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!(Ubyte2, Byte2, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!(Ubyte4, Byte4, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!([u8; 2], Byte2, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!([u8; 4], Byte4, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);

impl_storage_texel!(i8, Byte1, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!(Byte2, Byte2, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!(Byte4, Byte4, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!([i8; 2], Byte2, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!([i8; 4], Byte4, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);

impl_storage_texel!(u16, Short1, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!(Ushort2, Short2, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!(Ushort4, Short4, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!([u16; 2], Byte2, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!([u16; 4], Byte4, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);

impl_storage_texel!(i16, Short1, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!(Short2, Short2, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!(Short4, Short4, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!([i16; 2], Byte2, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!([i16; 4], Byte4, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);

impl_storage_texel!(u32, Int1, i32, u32, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!(Uint2, Int2, i32, u32, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!(Uint4, Int4, i32, u32, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!([u32; 2], Byte2, i32, u32, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!([u32; 4], Byte4, i32, u32, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);

impl_storage_texel!(i32, Int1, i32, u32, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!(Int2, Int2, i32, u32, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!(Int4, Int4, i32, u32, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!([i32; 2], Byte2, i32, u32, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!([i32; 4], Byte4, i32, u32, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);

impl_storage_texel!(f32, Float1, f32, Float2, Float4,);
impl_storage_texel!(Float2, Float2, f32, Float2, Float4,);
impl_storage_texel!(Float4, Float4, f32, Float2, Float4,);
impl_storage_texel!([f32; 2], Byte2, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!([f32; 4], Byte4, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);

impl_storage_texel!(f16, Half1, f32, Float2, Float4,);
impl_storage_texel!(Half2, Half2, f32, Float2, Float4,);
impl_storage_texel!(Half4, Half4, f32, Float2, Float4,);
impl_storage_texel!([f16; 2], Byte2, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);
impl_storage_texel!([f16; 4], Byte4, f32, Float2, Float4, Int2, Int4, Uint2, Uint4,);

// `T` is the read out type of the texture, which is not necessarily the same as
// the storage type In fact, the texture can be stored in any format as long as
// it can be converted to `T`
pub struct Tex2d<T: IoTexel> {
    #[allow(dead_code)]
    pub(crate) width: u32,
    #[allow(dead_code)]
    pub(crate) height: u32,
    pub(crate) handle: Arc<TextureHandle>,
    pub(crate) marker: PhantomData<T>,
}

// `T` is the read out type of the texture, which is not necessarily the same as
// the storage type In fact, the texture can be stored in any format as long as
// it can be converted to `T`
pub struct Tex3d<T: IoTexel> {
    #[allow(dead_code)]
    pub(crate) width: u32,
    #[allow(dead_code)]
    pub(crate) height: u32,
    #[allow(dead_code)]
    pub(crate) depth: u32,
    pub(crate) handle: Arc<TextureHandle>,
    pub(crate) marker: PhantomData<T>,
}
#[derive(Clone, Copy)]
pub struct Tex2dView<'a, T: IoTexel> {
    pub(crate) tex: &'a Tex2d<T>,
    pub(crate) level: u32,
}
#[derive(Clone, Copy)]
pub struct Tex3dView<'a, T: IoTexel> {
    pub(crate) tex: &'a Tex3d<T>,
    pub(crate) level: u32,
}
impl<T: IoTexel> Tex2d<T> {
    pub fn handle(&self) -> api::Texture {
        self.handle.handle
    }
    pub fn var(&self) -> Tex2dVar<T> {
        Tex2dVar::new(self.view(0))
    }
}
impl<T: IoTexel> Tex3d<T> {
    pub fn handle(&self) -> api::Texture {
        self.handle.handle
    }
    pub fn var(&self) -> Tex3dVar<T> {
        Tex3dVar::new(self.view(0))
    }
}
macro_rules! impl_tex_view {
    ($name:ident) => {
        impl<'a, T: IoTexel> $name<'a, T> {
            pub fn copy_to_async<U: StorageTexel<T>>(&'a self, data: &'a mut [U]) -> Command<'a> {
                assert_eq!(data.len(), self.texel_count() as usize);
                assert_eq!(self.tex.handle.storage, U::pixel_storage());
                let mut rt = ResourceTracker::new();
                rt.add(self.tex.handle.clone());
                Command {
                    inner: api::Command::TextureDownload(api::TextureDownloadCommand {
                        texture: self.handle(),
                        storage: self.tex.handle.storage,
                        level: self.level,
                        size: self.size(),
                        data: data.as_mut_ptr() as *mut u8,
                    }),
                    resource_tracker: rt,
                    marker: PhantomData,
                    callback: None,
                }
            }
            pub fn copy_to<U: StorageTexel<T>>(&'a self, data: &'a mut [U]) {
                assert_eq!(data.len(), self.texel_count() as usize);

                submit_default_stream_and_sync(&self.tex.handle.device, [self.copy_to_async(data)]);
            }
            pub fn copy_to_vec<U: StorageTexel<T>>(&'a self) -> Vec<U> {
                let mut data = Vec::with_capacity(self.texel_count() as usize);
                unsafe {
                    data.set_len(self.texel_count() as usize);
                }
                self.copy_to(&mut data);
                data
            }
            pub fn copy_from_async<'b, U: StorageTexel<T>>(&'a self, data: &'b [U]) -> Command<'b> {
                assert_eq!(data.len(), self.texel_count() as usize);
                assert_eq!(self.tex.handle.storage, U::pixel_storage());
                let mut rt = ResourceTracker::new();
                rt.add(self.tex.handle.clone());
                Command {
                    inner: api::Command::TextureUpload(api::TextureUploadCommand {
                        texture: self.handle(),
                        storage: self.tex.handle.storage,
                        level: self.level,
                        size: self.size(),
                        data: data.as_ptr() as *const u8,
                    }),
                    resource_tracker: rt,
                    marker: PhantomData,
                    callback: None,
                }
            }
            pub fn copy_from<U: StorageTexel<T>>(&'a self, data: &[U]) {
                submit_default_stream_and_sync(
                    &self.tex.handle.device,
                    [self.copy_from_async(data)],
                );
            }
            pub fn copy_to_buffer_async<'b, U: StorageTexel<T> + Value>(
                &'a self,
                buffer_view: &'b BufferView<U>,
            ) -> Command<'static> {
                let mut rt = ResourceTracker::new();
                rt.add(self.tex.handle.clone());
                rt.add(buffer_view.buffer.handle.clone());
                assert_eq!(buffer_view.len, self.texel_count() as usize);
                assert_eq!(self.tex.handle.storage, U::pixel_storage());
                Command {
                    inner: api::Command::TextureToBufferCopy(api::TextureToBufferCopyCommand {
                        texture: self.handle(),
                        storage: self.tex.handle.storage,
                        texture_level: self.level,
                        texture_size: self.size(),
                        buffer: buffer_view.handle(),
                        buffer_offset: buffer_view.offset,
                    }),
                    resource_tracker: rt,
                    marker: PhantomData,
                    callback: None,
                }
            }
            pub fn copy_to_buffer<U: StorageTexel<T> + Value>(
                &'a self,
                buffer_view: &BufferView<U>,
            ) {
                submit_default_stream_and_sync(
                    &self.tex.handle.device,
                    [self.copy_to_buffer_async(buffer_view)],
                );
            }
            pub fn copy_from_buffer_async<'b, U: StorageTexel<T> + Value>(
                &'a self,
                buffer_view: BufferView<U>,
            ) -> Command<'static> {
                let mut rt = ResourceTracker::new();
                rt.add(self.tex.handle.clone());
                rt.add(buffer_view.buffer.handle.clone());
                assert_eq!(buffer_view.len, self.texel_count() as usize);
                assert_eq!(self.tex.handle.storage, U::pixel_storage());
                Command {
                    inner: api::Command::BufferToTextureCopy(api::BufferToTextureCopyCommand {
                        texture: self.handle(),
                        storage: self.tex.handle.storage,
                        texture_level: self.level,
                        texture_size: self.size(),
                        buffer: buffer_view.handle(),
                        buffer_offset: buffer_view.offset,
                    }),
                    resource_tracker: rt,
                    marker: PhantomData,
                    callback: None,
                }
            }
            pub fn copy_from_buffer<U: StorageTexel<T> + Value>(
                &'a self,
                buffer_view: BufferView<U>,
            ) {
                submit_default_stream_and_sync(
                    &self.tex.handle.device,
                    [self.copy_from_buffer_async(buffer_view)],
                );
            }
            pub fn copy_to_texture_async(&'a self, other: $name<T>) -> Command<'static> {
                let mut rt = ResourceTracker::new();
                rt.add(self.tex.handle.clone());
                rt.add(other.tex.handle.clone());
                assert_eq!(self.size(), other.size());
                assert_eq!(self.tex.handle.storage, other.tex.handle.storage);
                assert_eq!(self.tex.handle.format, other.tex.handle.format);
                Command {
                    inner: api::Command::TextureCopy(api::TextureCopyCommand {
                        src: self.handle(),
                        storage: self.tex.handle.storage,
                        src_level: self.level,
                        size: self.size(),
                        dst: other.handle(),
                        dst_level: other.level,
                    }),
                    resource_tracker: rt,
                    marker: PhantomData,
                    callback: None,
                }
            }
            pub fn copy_to_texture(&'a self, other: $name<T>) {
                submit_default_stream_and_sync(
                    &self.tex.handle.device,
                    [self.copy_to_texture_async(other)],
                );
            }
        }
    };
}
impl<'a, T: IoTexel> Tex2dView<'a, T> {
    pub fn handle(&self) -> api::Texture {
        self.tex.handle.handle
    }
    pub fn texel_count(&self) -> u32 {
        let s = self.size();
        s[0] * s[1]
    }
    pub fn size(&self) -> [u32; 3] {
        [
            (self.tex.handle.width >> self.level).max(1),
            (self.tex.handle.height >> self.level).max(1),
            1,
        ]
    }
    pub fn var(&self) -> Tex2dVar<T> {
        Tex2dVar::new(*self)
    }
}
impl_tex_view!(Tex2dView);
impl<'a, T: IoTexel> Tex3dView<'a, T> {
    pub fn handle(&self) -> api::Texture {
        self.tex.handle.handle
    }
    pub fn texel_count(&self) -> u32 {
        let s = self.size();
        s[0] * s[1] * s[2]
    }
    pub fn size(&self) -> [u32; 3] {
        [
            (self.tex.handle.width >> self.level).max(1),
            (self.tex.handle.height >> self.level).max(1),
            (self.tex.handle.depth >> self.level).max(1),
        ]
    }
    pub fn var(&self) -> Tex3dVar<T> {
        Tex3dVar::new(*self)
    }
}
impl_tex_view!(Tex3dView);
impl Drop for TextureHandle {
    fn drop(&mut self) {
        self.device.inner.destroy_texture(self.handle);
    }
}

impl<T: IoTexel> Tex2d<T> {
    pub fn view(&self, level: u32) -> Tex2dView<T> {
        Tex2dView { tex: self, level }
    }
    pub fn native_handle(&self) -> *mut std::ffi::c_void {
        self.handle.native_handle
    }
    pub fn width(&self) -> u32 {
        self.handle.width
    }
    pub fn height(&self) -> u32 {
        self.handle.height
    }
    pub fn format(&self) -> PixelFormat {
        self.handle.format
    }
}
impl<T: IoTexel> Tex3d<T> {
    pub fn view(&self, level: u32) -> Tex3dView<T> {
        Tex3dView { tex: self, level }
    }
    pub fn native_handle(&self) -> *mut std::ffi::c_void {
        self.handle.native_handle
    }
    pub fn width(&self) -> u32 {
        self.handle.width
    }
    pub fn height(&self) -> u32 {
        self.handle.height
    }
    pub fn depth(&self) -> u32 {
        self.handle.depth
    }
    pub fn format(&self) -> PixelFormat {
        self.handle.format
    }
}
#[derive(Clone)]
pub struct BufferVar<T: Value> {
    pub(crate) marker: PhantomData<T>,
    #[allow(dead_code)]
    pub(crate) handle: Option<Arc<BufferHandle>>,
    pub(crate) node: NodeRef,
}
impl<T: Value> ToNode for BufferVar<T> {
    fn node(&self) -> NodeRef {
        self.node
    }
}
impl<T: Value> Drop for BufferVar<T> {
    fn drop(&mut self) {}
}
#[derive(Clone)]
pub struct BindlessArrayVar {
    pub(crate) node: NodeRef,
    #[allow(dead_code)]
    pub(crate) handle: Option<Arc<BindlessArrayHandle>>,
}
#[derive(Clone)]
pub struct BindlessBufferVar<T> {
    array: NodeRef,
    buffer_index: Expr<u32>,
    _marker: PhantomData<T>,
}
impl<T: Value> ToNode for BindlessBufferVar<T> {
    fn node(&self) -> NodeRef {
        self.array
    }
}

impl<T: Value> IndexRead for BindlessBufferVar<T> {
    type Element = T;
    fn read<I: IntoIndex>(&self, i: I) -> Expr<T> {
        let i = i.to_u64();
        if need_runtime_check() {
            lc_assert!(i.lt(self.len()));
        }

        Expr::<T>::from_node(__current_scope(|b| {
            b.call(
                Func::BindlessBufferRead,
                &[self.array, self.buffer_index.node(), ToNode::node(&i)],
                T::type_(),
            )
        }))
    }
}
impl<T: Value> BindlessBufferVar<T> {
    pub fn len(&self) -> Expr<u64> {
        let stride = (T::type_().size() as u64).expr();
        Expr::<u64>::from_node(__current_scope(|b| {
            b.call(
                Func::BindlessBufferSize,
                &[self.array, self.buffer_index.node(), stride.node()],
                u32::type_(),
            )
        }))
    }
    pub fn __type(&self) -> Expr<u64> {
        Expr::<u64>::from_node(__current_scope(|b| {
            b.call(
                Func::BindlessBufferType,
                &[self.array, self.buffer_index.node()],
                u64::type_(),
            )
        }))
    }
}
#[derive(Clone)]
pub struct BindlessByteBufferVar {
    array: NodeRef,
    buffer_index: Expr<u32>,
}
impl ToNode for BindlessByteBufferVar {
    fn node(&self) -> NodeRef {
        self.array
    }
}
impl BindlessByteBufferVar {
    pub fn read<T: Value>(&self, index_bytes: impl IntoIndex) -> Expr<T> {
        let i = index_bytes.to_u64();
        Expr::<T>::from_node(__current_scope(|b| {
            b.call(
                Func::BindlessByteBufferRead,
                &[self.array, self.buffer_index.node(), i.node],
                <T as TypeOf>::type_(),
            )
        }))
    }
    pub fn len(&self) -> Expr<u64> {
        let s = (1u64).expr();
        Expr::<u64>::from_node(__current_scope(|b| {
            b.call(
                Func::BindlessBufferSize,
                &[self.array, self.buffer_index.node(), s.node()],
                <u64 as TypeOf>::type_(),
            )
        }))
    }
}
#[derive(Clone)]
pub struct BindlessTex2dVar {
    array: NodeRef,
    tex2d_index: Expr<u32>,
}

impl BindlessTex2dVar {
    pub fn sample(&self, uv: Expr<Float2>) -> Expr<Float4> {
        Expr::<Float4>::from_node(__current_scope(|b| {
            b.call(
                Func::BindlessTexture2dSample,
                &[self.array, self.tex2d_index.node(), uv.node()],
                Float4::type_(),
            )
        }))
    }
    pub fn sample_level(&self, uv: Expr<Float2>, level: Expr<f32>) -> Expr<Float4> {
        Expr::<Float4>::from_node(__current_scope(|b| {
            b.call(
                Func::BindlessTexture2dSampleLevel,
                &[self.array, self.tex2d_index.node(), uv.node(), level.node()],
                Float4::type_(),
            )
        }))
    }
    pub fn sample_grad(
        &self,
        uv: Expr<Float2>,
        ddx: Expr<Float2>,
        ddy: Expr<Float2>,
    ) -> Expr<Float4> {
        Expr::<Float4>::from_node(__current_scope(|b| {
            b.call(
                Func::BindlessTexture2dSampleLevel,
                &[
                    self.array,
                    self.tex2d_index.node(),
                    uv.node(),
                    ddx.node(),
                    ddy.node(),
                ],
                Float4::type_(),
            )
        }))
    }
    pub fn read(&self, coord: Expr<Uint2>) -> Expr<Float4> {
        Expr::<Float4>::from_node(__current_scope(|b| {
            b.call(
                Func::BindlessTexture2dRead,
                &[self.array, self.tex2d_index.node(), coord.node()],
                Float4::type_(),
            )
        }))
    }
    pub fn read_level(&self, coord: Expr<Uint2>, level: Expr<u32>) -> Expr<Float4> {
        Expr::<Float4>::from_node(__current_scope(|b| {
            b.call(
                Func::BindlessTexture2dReadLevel,
                &[
                    self.array,
                    self.tex2d_index.node(),
                    coord.node(),
                    level.node(),
                ],
                Float4::type_(),
            )
        }))
    }
    pub fn size(&self) -> Expr<Uint2> {
        Expr::<Uint2>::from_node(__current_scope(|b| {
            b.call(
                Func::BindlessTexture2dSize,
                &[self.array, self.tex2d_index.node()],
                Uint2::type_(),
            )
        }))
    }
    pub fn size_level(&self, level: Expr<u32>) -> Expr<Uint2> {
        Expr::<Uint2>::from_node(__current_scope(|b| {
            b.call(
                Func::BindlessTexture2dSizeLevel,
                &[self.array, self.tex2d_index.node(), level.node()],
                Uint2::type_(),
            )
        }))
    }
}
#[derive(Clone)]
pub struct BindlessTex3dVar {
    array: NodeRef,
    tex3d_index: Expr<u32>,
}

impl BindlessTex3dVar {
    pub fn sample(&self, uv: Expr<Float3>) -> Expr<Float4> {
        Expr::<Float4>::from_node(__current_scope(|b| {
            b.call(
                Func::BindlessTexture3dSample,
                &[self.array, self.tex3d_index.node(), uv.node()],
                Float4::type_(),
            )
        }))
    }
    pub fn sample_level(&self, uv: Expr<Float3>, level: Expr<f32>) -> Expr<Float4> {
        Expr::<Float4>::from_node(__current_scope(|b| {
            b.call(
                Func::BindlessTexture3dSampleLevel,
                &[self.array, self.tex3d_index.node(), uv.node(), level.node()],
                Float4::type_(),
            )
        }))
    }
    pub fn sample_grad(
        &self,
        uv: Expr<Float3>,
        ddx: Expr<Float3>,
        ddy: Expr<Float3>,
    ) -> Expr<Float4> {
        Expr::<Float4>::from_node(__current_scope(|b| {
            b.call(
                Func::BindlessTexture3dSampleLevel,
                &[
                    self.array,
                    self.tex3d_index.node(),
                    uv.node(),
                    ddx.node(),
                    ddy.node(),
                ],
                Float4::type_(),
            )
        }))
    }
    pub fn read(&self, coord: Expr<Uint3>) -> Expr<Float4> {
        Expr::<Float4>::from_node(__current_scope(|b| {
            b.call(
                Func::BindlessTexture3dRead,
                &[self.array, self.tex3d_index.node(), coord.node()],
                Float4::type_(),
            )
        }))
    }
    pub fn read_level(&self, coord: Expr<Uint3>, level: Expr<u32>) -> Expr<Float4> {
        Expr::<Float4>::from_node(__current_scope(|b| {
            b.call(
                Func::BindlessTexture3dReadLevel,
                &[
                    self.array,
                    self.tex3d_index.node(),
                    coord.node(),
                    level.node(),
                ],
                Float4::type_(),
            )
        }))
    }
    pub fn size(&self) -> Expr<Uint3> {
        Expr::<Uint3>::from_node(__current_scope(|b| {
            b.call(
                Func::BindlessTexture3dSize,
                &[self.array, self.tex3d_index.node()],
                Uint3::type_(),
            )
        }))
    }
    pub fn size_level(&self, level: Expr<u32>) -> Expr<Uint3> {
        Expr::<Uint3>::from_node(__current_scope(|b| {
            b.call(
                Func::BindlessTexture3dSizeLevel,
                &[self.array, self.tex3d_index.node(), level.node()],
                Uint3::type_(),
            )
        }))
    }
}

impl BindlessArrayVar {
    pub fn tex2d(&self, tex2d_index: impl Into<Expr<u32>>) -> BindlessTex2dVar {
        let v = BindlessTex2dVar {
            array: self.node,
            tex2d_index: tex2d_index.into(),
        };
        v
    }
    pub fn tex3d(&self, tex3d_index: impl Into<Expr<u32>>) -> BindlessTex3dVar {
        let v = BindlessTex3dVar {
            array: self.node,
            tex3d_index: tex3d_index.into(),
        };
        v
    }
    pub fn byte_address_buffer(&self, buffer_index: impl Into<Expr<u32>>) -> BindlessByteBufferVar {
        let v = BindlessByteBufferVar {
            array: self.node,
            buffer_index: buffer_index.into(),
        };
        v
    }
    pub fn buffer<T: Value>(&self, buffer_index: impl Into<Expr<u32>>) -> BindlessBufferVar<T> {
        let v = BindlessBufferVar {
            array: self.node,
            buffer_index: buffer_index.into(),
            _marker: PhantomData,
        };
        if __env_need_backtrace() && is_cpu_backend() {
            let vt = v.__type();
            let expected = type_hash(&T::type_());
            let backtrace = get_backtrace();
            let check_type = CpuFn::new(move |t: &mut u64| {
                if *t != expected {
                    {
                        let mut stderr = std::io::stderr().lock();
                        use std::io::Write;
                        writeln!(stderr,
                                 "Bindless buffer type mismatch: expected hash {:?}, got {:?}; host backtrace:\n {:?}",
                                 expected, t, backtrace
                        ).unwrap();
                    }
                    abort();
                }
            });
            let _ = check_type.call(vt);
        } else if is_cpu_backend() {
            if need_runtime_check() {
                let expected = type_hash(&T::type_());
                lc_assert!(v.__type().eq(expected));
            }
        }
        v
    }

    pub fn new(array: &BindlessArray) -> Self {
        let node = RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            assert!(
                r.lock,
                "BindlessArrayVar must be created from within a kernel"
            );
            let handle: u64 = array.handle().0;
            let binding = Binding::BindlessArray(BindlessArrayBinding { handle });

            if let Some((_, node, _, _)) = r.captured_buffer.get(&binding) {
                *node
            } else {
                let node = new_node(
                    r.pools.as_ref().unwrap(),
                    Node::new(CArc::new(Instruction::Bindless), Type::void()),
                );
                let i = r.captured_buffer.len();
                r.captured_buffer
                    .insert(binding, (i, node, binding, array.handle.clone()));
                node
            }
        });
        Self {
            node,
            handle: Some(array.handle.clone()),
        }
    }
}
impl<T: Value> ToNode for Buffer<T> {
    fn node(&self) -> NodeRef {
        self.var().node()
    }
}
impl<T: Value> IndexRead for Buffer<T> {
    type Element = T;
    fn read<I: IntoIndex>(&self, i: I) -> Expr<T> {
        self.var().read(i)
    }
}
impl<T: Value> IndexWrite for Buffer<T> {
    fn write<I: IntoIndex, V: AsExpr<Value = T>>(&self, i: I, v: V) {
        self.var().write(i, v)
    }
}
impl<T: Value> IndexRead for BufferVar<T> {
    type Element = T;
    fn read<I: IntoIndex>(&self, i: I) -> Expr<T> {
        let i = i.to_u64();
        if need_runtime_check() {
            lc_assert!(i.lt(self.len()));
        }
        __current_scope(|b| {
            FromNode::from_node(b.call(
                Func::BufferRead,
                &[self.node, ToNode::node(&i)],
                T::type_(),
            ))
        })
    }
}
impl<T: Value> IndexWrite for BufferVar<T> {
    fn write<I: IntoIndex, V: AsExpr<Value = T>>(&self, i: I, v: V) {
        let i = i.to_u64();
        let v = v.as_expr();
        if need_runtime_check() {
            lc_assert!(i.lt(self.len()));
        }
        __current_scope(|b| {
            b.call(
                Func::BufferWrite,
                &[self.node, ToNode::node(&i), v.node()],
                Type::void(),
            )
        });
    }
}
impl<T: Value> BufferVar<T> {
    pub fn new(buffer: &BufferView<'_, T>) -> Self {
        let node = RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            assert!(r.lock, "BufferVar must be created from within a kernel");
            let binding = Binding::Buffer(BufferBinding {
                handle: buffer.handle().0,
                size: buffer.len * std::mem::size_of::<T>(),
                offset: (buffer.offset * std::mem::size_of::<T>()) as u64,
            });
            if let Some((_, node, _, _)) = r.captured_buffer.get(&binding) {
                *node
            } else {
                let node = new_node(
                    r.pools.as_ref().unwrap(),
                    Node::new(CArc::new(Instruction::Buffer), T::type_()),
                );
                let i = r.captured_buffer.len();
                r.captured_buffer
                    .insert(binding, (i, node, binding, buffer.buffer.handle.clone()));
                node
            }
        });
        Self {
            node,
            marker: PhantomData,
            handle: Some(buffer.buffer.handle.clone()),
        }
    }
    pub fn len(&self) -> Expr<u64> {
        FromNode::from_node(
            __current_scope(|b| b.call(Func::BufferSize, &[self.node], u64::type_())).into(),
        )
    }
}

macro_rules! impl_atomic {
    ($t:ty) => {
        impl BufferVar<$t> {
            pub fn atomic_exchange<I: IntoIndex, V: AsExpr<Value = $t>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.to_u64();
                let v = v.as_expr();
                if need_runtime_check() {
                    lc_assert!(i.lt(self.len()));
                }
                Expr::<$t>::from_node(__current_scope(|b| {
                    b.call(
                        Func::AtomicExchange,
                        &[self.node, ToNode::node(&i), v.node()],
                        <$t>::type_(),
                    )
                }))
            }
            pub fn atomic_compare_exchange<
                I: IntoIndex,
                V0: AsExpr<Value = $t>,
                V1: AsExpr<Value = $t>,
            >(
                &self,
                i: I,
                expected: V0,
                desired: V1,
            ) -> Expr<$t> {
                let i = i.to_u64();
                let expected = expected.as_expr();
                let desired = desired.as_expr();
                if need_runtime_check() {
                    lc_assert!(i.lt(self.len()));
                }
                Expr::<$t>::from_node(__current_scope(|b| {
                    b.call(
                        Func::AtomicCompareExchange,
                        &[self.node, ToNode::node(&i), expected.node(), desired.node()],
                        <$t>::type_(),
                    )
                }))
            }
            pub fn atomic_fetch_add<I: IntoIndex, V: AsExpr<Value = $t>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.to_u64();
                let v = v.as_expr();
                if need_runtime_check() {
                    lc_assert!(i.lt(self.len()));
                }
                Expr::<$t>::from_node(__current_scope(|b| {
                    b.call(
                        Func::AtomicFetchAdd,
                        &[self.node, ToNode::node(&i), v.node()],
                        <$t>::type_(),
                    )
                }))
            }
            pub fn atomic_fetch_sub<I: IntoIndex, V: AsExpr<Value = $t>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.to_u64();
                let v = v.as_expr();
                if need_runtime_check() {
                    lc_assert!(i.lt(self.len()));
                }
                Expr::<$t>::from_node(__current_scope(|b| {
                    b.call(
                        Func::AtomicFetchSub,
                        &[self.node, ToNode::node(&i), v.node()],
                        <$t>::type_(),
                    )
                }))
            }
            pub fn atomic_fetch_min<I: IntoIndex, V: AsExpr<Value = $t>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.to_u64();
                let v = v.as_expr();
                if need_runtime_check() {
                    lc_assert!(i.lt(self.len()));
                }
                Expr::<$t>::from_node(__current_scope(|b| {
                    b.call(
                        Func::AtomicFetchMin,
                        &[self.node, ToNode::node(&i), v.node()],
                        <$t>::type_(),
                    )
                }))
            }
            pub fn atomic_fetch_max<I: IntoIndex, V: AsExpr<Value = $t>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.to_u64();
                let v = v.as_expr();
                if need_runtime_check() {
                    lc_assert!(i.lt(self.len()));
                }
                Expr::<$t>::from_node(__current_scope(|b| {
                    b.call(
                        Func::AtomicFetchMax,
                        &[self.node, ToNode::node(&i), v.node()],
                        <$t>::type_(),
                    )
                }))
            }
        }
    };
}
macro_rules! impl_atomic_bit {
    ($t:ty) => {
        impl BufferVar<$t> {
            pub fn atomic_fetch_and<I: IntoIndex, V: AsExpr<Value = $t>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.to_u64();
                let v = v.as_expr();
                if need_runtime_check() {
                    lc_assert!(i.lt(self.len()));
                }
                Expr::<$t>::from_node(__current_scope(|b| {
                    b.call(
                        Func::AtomicFetchAnd,
                        &[self.node, ToNode::node(&i), v.node()],
                        <$t>::type_(),
                    )
                }))
            }
            pub fn atomic_fetch_or<I: IntoIndex, V: AsExpr<Value = $t>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.to_u64();
                let v = v.as_expr();
                if need_runtime_check() {
                    lc_assert!(i.lt(self.len()));
                }
                Expr::<$t>::from_node(__current_scope(|b| {
                    b.call(
                        Func::AtomicFetchOr,
                        &[self.node, ToNode::node(&i), v.node()],
                        <$t>::type_(),
                    )
                }))
            }
            pub fn atomic_fetch_xor<I: IntoIndex, V: AsExpr<Value = $t>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.to_u64();
                let v = v.as_expr();
                if need_runtime_check() {
                    lc_assert!(i.lt(self.len()));
                }
                Expr::<$t>::from_node(__current_scope(|b| {
                    b.call(
                        Func::AtomicFetchXor,
                        &[self.node, ToNode::node(&i), v.node()],
                        <$t>::type_(),
                    )
                }))
            }
        }
    };
}
impl_atomic!(i32);
impl_atomic!(u32);
impl_atomic!(i64);
impl_atomic!(u64);
impl_atomic!(f32);
impl_atomic_bit!(u32);
impl_atomic_bit!(u64);
impl_atomic_bit!(i32);
impl_atomic_bit!(i64);
#[derive(Clone)]
pub struct Tex2dVar<T: IoTexel> {
    pub(crate) node: NodeRef,
    #[allow(dead_code)]
    pub(crate) handle: Option<Arc<TextureHandle>>,
    pub(crate) marker: PhantomData<T>,
    #[allow(dead_code)]
    pub(crate) level: Option<u32>,
}

impl<T: IoTexel> Tex2dVar<T> {
    pub fn new(view: Tex2dView<'_, T>) -> Self {
        let node = RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            assert!(r.lock, "Tex2dVar<T> must be created from within a kernel");
            let handle: u64 = view.tex.handle().0;
            let binding = Binding::Texture(TextureBinding {
                handle,
                level: view.level,
            });
            if let Some((_, node, _, _)) = r.captured_buffer.get(&binding) {
                *node
            } else {
                let node = new_node(
                    r.pools.as_ref().unwrap(),
                    Node::new(CArc::new(Instruction::Texture2D), T::RwType::type_()),
                );
                let i = r.captured_buffer.len();
                r.captured_buffer
                    .insert(binding, (i, node, binding, view.tex.handle.clone()));
                node
            }
        });
        Self {
            node,
            handle: Some(view.tex.handle.clone()),
            level: Some(view.level),
            marker: PhantomData,
        }
    }
    pub fn read(&self, uv: impl Into<Expr<Uint2>>) -> Expr<T> {
        let uv = uv.into();
        T::convert_from_read(Expr::<T::RwType>::from_node(__current_scope(|b| {
            b.call(
                Func::Texture2dRead,
                &[self.node, uv.node()],
                T::RwType::type_(),
            )
        })))
    }
    pub fn write(&self, uv: impl Into<Expr<Uint2>>, v: impl Into<Expr<T>>) {
        let uv = uv.into();
        let v = v.into();
        let v = T::convert_to_write(v);
        __current_scope(|b| {
            b.call(
                Func::Texture2dWrite,
                &[self.node, uv.node(), v.node()],
                Type::void(),
            );
        })
    }
}

impl<T: IoTexel> Tex3dVar<T> {
    pub fn new(view: Tex3dView<'_, T>) -> Self {
        let node = RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            assert!(r.lock, "Tex3dVar<T> must be created from within a kernel");
            let handle: u64 = view.tex.handle().0;
            let binding = Binding::Texture(TextureBinding {
                handle,
                level: view.level,
            });
            if let Some((_, node, _, _)) = r.captured_buffer.get(&binding) {
                *node
            } else {
                let node = new_node(
                    r.pools.as_ref().unwrap(),
                    Node::new(CArc::new(Instruction::Texture3D), T::RwType::type_()),
                );
                let i = r.captured_buffer.len();
                r.captured_buffer
                    .insert(binding, (i, node, binding, view.tex.handle.clone()));
                node
            }
        });
        Self {
            node,
            handle: Some(view.tex.handle.clone()),
            level: Some(view.level),
            marker: PhantomData,
        }
    }
    pub fn read(&self, uv: impl Into<Expr<Uint3>>) -> Expr<T> {
        let uv = uv.into();
        T::convert_from_read(Expr::<T::RwType>::from_node(__current_scope(|b| {
            b.call(
                Func::Texture3dRead,
                &[self.node, uv.node()],
                T::RwType::type_(),
            )
        })))
    }
    pub fn write(&self, uv: impl Into<Expr<Uint3>>, v: impl Into<Expr<T>>) {
        let uv = uv.into();
        let v = v.into();
        let v = T::convert_to_write(v);
        __current_scope(|b| {
            b.call(
                Func::Texture3dWrite,
                &[self.node, uv.node(), v.node()],
                Type::void(),
            );
        })
    }
}
#[derive(Clone)]
pub struct Tex3dVar<T: IoTexel> {
    pub(crate) node: NodeRef,
    #[allow(dead_code)]
    pub(crate) handle: Option<Arc<TextureHandle>>,
    pub(crate) marker: PhantomData<T>,
    #[allow(dead_code)]
    pub(crate) level: Option<u32>,
}
