use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::ops::RangeBounds;
use std::process::abort;
use std::sync::{Arc, Weak};

use parking_lot::lock_api::RawMutex as RawMutexTrait;
use parking_lot::RawMutex;

use crate::internal_prelude::*;

use crate::lang::index::IntoIndex;
use crate::lang::types::AtomicRef;
use crate::runtime::*;

use api::{BufferDownloadCommand, BufferUploadCommand, INVALID_RESOURCE_HANDLE};
use std::ffi::c_void;

pub type ByteBuffer = Buffer<u8>;
pub type ByteBufferView = BufferView<u8>;
pub type ByteBufferVar = BufferVar<u8>;

// Uncomment if the alias blowup again...
// pub struct ByteBuffer {
//     pub(crate) device: Device,
//     pub(crate) handle: Arc<BufferHandle>,
//     pub(crate) len: usize,
// }
// impl ByteBuffer {
//     pub fn len(&self) -> usize {
//         self.len
//     }
//     #[inline]
//     pub fn handle(&self) -> api::Buffer {
//         self.handle.handle
//     }
//     #[inline]
//     pub fn native_handle(&self) -> *mut c_void {
//         self.handle.native_handle
//     }
//     #[inline]
//     pub fn copy_from(&self, data: &[u8]) {
//         self.view(..).copy_from(data);
//     }
//     #[inline]
//     pub fn copy_from_async<'a>(&self, data: &[u8]) -> Command<'_> {
//         self.view(..).copy_from_async(data)
//     }
//     #[inline]
//     pub fn copy_to(&self, data: &mut [u8]) {
//         self.view(..).copy_to(data);
//     }
//     #[inline]
//     pub fn copy_to_async<'a>(&self, data: &'a mut [u8]) -> Command<'a> {
//         self.view(..).copy_to_async(data)
//     }
//     #[inline]
//     pub fn copy_to_vec(&self) -> Vec<u8> {
//         self.view(..).copy_to_vec()
//     }
//     #[inline]
//     pub fn copy_to_buffer(&self, dst: &ByteBuffer) {
//         self.view(..).copy_to_buffer(dst.view(..));
//     }
//     #[inline]
//     pub fn copy_to_buffer_async<'a>(&'a self, dst: &'a ByteBuffer) -> Command<'a> {
//         self.view(..).copy_to_buffer_async(dst.view(..))
//     }
//     #[inline]
//     pub fn fill_fn<F: FnMut(usize) -> u8>(&self, f: F) {
//         self.view(..).fill_fn(f);
//     }
//     #[inline]
//     pub fn fill(&self, value: u8) {
//         self.view(..).fill(value);
//     }
//     pub fn view<S: RangeBounds<usize>>(&self, range: S) -> ByteBufferView<'_> {
//         let lower = range.start_bound();
//         let upper = range.end_bound();
//         let lower = match lower {
//             std::ops::Bound::Included(&x) => x,
//             std::ops::Bound::Excluded(&x) => x + 1,
//             std::ops::Bound::Unbounded => 0,
//         };
//         let upper = match upper {
//             std::ops::Bound::Included(&x) => x + 1,
//             std::ops::Bound::Excluded(&x) => x,
//             std::ops::Bound::Unbounded => self.len,
//         };
//         assert!(lower <= upper);
//         assert!(upper <= self.len);
//         ByteBufferView {
//             buffer: self,
//             offset: lower,
//             len: upper - lower,
//         }
//     }
//     pub fn var(&self) -> ByteBufferVar {
//         ByteBufferVar::new(&self.view(..))
//     }
// }
// pub struct ByteBufferView<'a> {
//     pub(crate) buffer: &'a ByteBuffer,
//     pub(crate) offset: usize,
//     pub(crate) len: usize,
// }
// impl<'a> ByteBufferView<'a> {
//     pub fn handle(&self) -> api::Buffer {
//         self.buffer.handle()
//     }
//     pub fn copy_to_async<'b>(&'a self, data: &'b mut [u8]) -> Command<'b> {
//         assert_eq!(data.len(), self.len);
//         let mut rt = ResourceTracker::new();
//         rt.add(self.buffer.handle.clone());
//         Command {
//             inner: api::Command::BufferDownload(BufferDownloadCommand {
//                 buffer: self.handle(),
//                 offset: self.offset,
//                 size: data.len(),
//                 data: data.as_mut_ptr() as *mut u8,
//             }),
//             marker: PhantomData,
//             resource_tracker: rt,
//             callback: None,
//         }
//     }
//     pub fn copy_to_vec(&self) -> Vec<u8> {
//         let mut data = Vec::with_capacity(self.len);
//         unsafe {
//             let slice = std::slice::from_raw_parts_mut(data.as_mut_ptr(), self.len);
//             self.copy_to(slice);
//             data.set_len(self.len);
//         }
//         data
//     }
//     pub fn copy_to(&self, data: &mut [u8]) {
//         unsafe {
//             submit_default_stream_and_sync(&self.buffer.device, [self.copy_to_async(data)]);
//         }
//     }

//     pub fn copy_from_async<'b>(&'a self, data: &'b [u8]) -> Command<'static> {
//         assert_eq!(data.len(), self.len);
//         let mut rt = ResourceTracker::new();
//         rt.add(self.buffer.handle.clone());
//         Command {
//             inner: api::Command::BufferUpload(BufferUploadCommand {
//                 buffer: self.handle(),
//                 offset: self.offset,
//                 size: data.len(),
//                 data: data.as_ptr() as *const u8,
//             }),
//             marker: PhantomData,
//             resource_tracker: rt,
//             callback: None,
//         }
//     }
//     pub fn copy_from(&self, data: &[u8]) {
//         submit_default_stream_and_sync(&self.buffer.device, [self.copy_from_async(data)]);
//     }
//     pub fn fill_fn<F: FnMut(usize) -> u8>(&self, f: F) {
//         self.copy_from(&(0..self.len).map(f).collect::<Vec<_>>());
//     }
//     pub fn fill(&self, value: u8) {
//         self.fill_fn(|_| value);
//     }
//     pub fn copy_to_buffer_async(&self, dst: ByteBufferView<'a>) -> Command<'static> {
//         assert_eq!(self.len, dst.len);
//         let mut rt = ResourceTracker::new();
//         rt.add(self.buffer.handle.clone());
//         rt.add(dst.buffer.handle.clone());
//         Command {
//             inner: api::Command::BufferCopy(api::BufferCopyCommand {
//                 src: self.handle(),
//                 src_offset: self.offset,
//                 dst: dst.handle(),
//                 dst_offset: dst.offset,
//                 size: self.len,
//             }),
//             marker: PhantomData,
//             resource_tracker: rt,
//             callback: None,
//         }
//     }
//     pub fn copy_to_buffer(&self, dst: ByteBufferView<'a>) {
//         submit_default_stream_and_sync(&self.buffer.device, [self.copy_to_buffer_async(dst)]);
//     }
// }
// #[derive(Clone)]
// pub struct ByteBufferVar {
//     #[allow(dead_code)]
//     pub(crate) handle: Option<Arc<BufferHandle>>,
//     pub(crate) node: NodeRef,
// }
// impl ByteBufferVar {
//     pub fn new(buffer: &ByteBufferView<'_>) -> Self {
//         let node = RECORDER.with(|r| {
//             let mut r = r.borrow_mut();
//             assert!(r.lock, "BufferVar must be created from within a kernel");
//             let binding = Binding::Buffer(BufferBinding {
//                 handle: buffer.handle().0,
//                 size: buffer.len,
//                 offset: buffer.offset as u64,
//             });
//             r.capture_or_get(binding, &buffer.buffer.handle, || {
//                 Node::new(CArc::new(Instruction::Buffer), Type::void())
//             })
//         });
//         Self {
//             node,
//             handle: Some(buffer.buffer.handle.clone()),
//         }
//     }
//     pub unsafe fn read_as<T: Value>(&self, index_bytes: impl IntoIndex) -> Expr<T> {
//         let i = index_bytes.to_u64();
//         Expr::<T>::from_node(__current_scope(|b| {
//             b.call(
//                 Func::ByteBufferRead,
//                 &[self.node, i.node],
//                 <T as TypeOf>::type_(),
//             )
//         }))
//     }
//     pub fn len_bytes_expr(&self) -> Expr<u64> {
//         Expr::<u64>::from_node(__current_scope(|b| {
//             b.call(Func::ByteBufferSize, &[self.node], <u64 as TypeOf>::type_())
//         }))
//     }
//     pub unsafe fn write_as<T: Value>(
//         &self,
//         index_bytes: impl IntoIndex,
//         value: impl Into<Expr<T>>,
//     ) {
//         let i = index_bytes.to_u64();
//         let value: Expr<T> = value.into();
//         __current_scope(|b| {
//             b.call(
//                 Func::ByteBufferWrite,
//                 &[self.node, i.node, value.node()],
//                 Type::void(),
//             )
//         });
//     }
// }
impl BufferVar<u8> {
    pub unsafe fn read_as<T: Value>(&self, index_bytes: impl IntoIndex) -> Expr<T> {
        let i = index_bytes.to_u64();
        let self_node = self.node.get();
        let i = i.node().get();
        Expr::<T>::from_node(
            __current_scope(|b| {
                b.call(
                    Func::ByteBufferRead,
                    &[self_node, i],
                    <T as TypeOf>::type_(),
                )
            })
            .into(),
        )
    }
    pub fn len_bytes_expr(&self) -> Expr<u64> {
        let self_node = self.node.get();
        Expr::<u64>::from_node(
            __current_scope(|b| {
                b.call(Func::ByteBufferSize, &[self_node], <u64 as TypeOf>::type_())
            })
            .into(),
        )
    }
    pub unsafe fn write_as<T: Value>(
        &self,
        index_bytes: impl IntoIndex,
        value: impl AsExpr<Value = T>,
    ) {
        let i = index_bytes.to_u64().node().get();
        let value = value.as_expr().node().get();
        let self_node = self.node.get();
        __current_scope(|b| b.call(Func::ByteBufferWrite, &[self_node, i, value], Type::void()));
    }
}
pub struct Buffer<T: Value> {
    pub(crate) handle: Arc<BufferHandle>,
    pub(crate) full_view: BufferView<T>,
}
impl<T: Value> BufferView<T> {
    pub fn copy_async<'a>(&self, s: &'a Scope<'a>) -> Buffer<T> {
        let copy = self.device.create_buffer(self.len);
        s.submit([self.copy_to_buffer_async(&copy)]);
        copy
    }
    pub fn copy(&self) -> Buffer<T> {
        let default_stream = self.device.default_stream();
        default_stream.with_scope(|s| self.copy_async(s))
    }
}
impl<T: Value + fmt::Debug> fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct DebugEllipsis;
        impl fmt::Debug for DebugEllipsis {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("..")
            }
        }

        write!(f, "Buffer<{}>({})", std::any::type_name::<T>(), self.len())?;
        // if self.len() <= 16 || f.precision().is_some() {
        //     let count = f.precision().unwrap_or(16);
        //     if count >= self.len() {
        //         f.debug_list().entries(self.copy_to_vec().iter()).finish()?;
        //     } else {
        //         let values = self.view(0..count).copy_to_vec();
        //
        //         f.debug_list()
        //             .entries(values.iter())
        //             .entry(&DebugEllipsis)
        //             .finish()?;
        //     }
        // }
        Ok(())
    }
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
#[derive(Clone)]
pub struct BufferView<T: Value> {
    pub(crate) device: Device,
    pub(crate) handle: Weak<BufferHandle>,
    /// offset in #elements
    pub(crate) offset: usize,
    /// length in #elements
    pub(crate) len: usize,
    pub(crate) total_size_bytes: usize,
    pub(crate) _marker: PhantomData<fn() -> T>,
}
#[macro_export]
macro_rules! impl_resource_deref_to_var {
    ($r:ident, $v:ident [T: $tr:ident]) => {
        impl<T: $tr> std::ops::Deref for $r<T> {
            type Target = $v<T>;
            fn deref(&self) -> &Self::Target {
                let v = self.var();
                with_recorder(|r| {
                    let v = r.arena.alloc(v);
                    r.dtors.push((v as *mut _ as *mut u8, |v| unsafe {
                        std::ptr::drop_in_place(v as *mut $v<T>)
                    }));
                    unsafe { std::mem::transmute(v) }
                })
            }
        }
        
    };
    ($r:ident, $v:ident) => {
        impl std::ops::Deref for $r {
            type Target = $v;
            fn deref(&self) -> &Self::Target {
                let v = self.var();
                with_recorder(|r| {
                    let v = r.arena.alloc(v);
                    r.dtors.push((v as *mut _ as *mut u8, |v| unsafe {
                        std::ptr::drop_in_place(v as *mut $v)
                    }));
                    unsafe { std::mem::transmute(v) }
                })
            }
        }
        
    };
}
impl_resource_deref_to_var!(BufferView, BufferVar [T: Value]);
impl_resource_deref_to_var!(Tex2dView, Tex2dVar [T: IoTexel]);
impl_resource_deref_to_var!(Tex3dView, Tex3dVar [T: IoTexel]);
impl_resource_deref_to_var!(Tex2d, Tex2dVar [T: IoTexel]);
impl_resource_deref_to_var!(Tex3d, Tex3dVar [T: IoTexel]);
impl_resource_deref_to_var!(BindlessArray, BindlessArrayVar);

impl<T: Value> BufferView<T> {
    /// reinterpret the buffer as a different type
    /// must satisfy `std::mem::size_of::<T>() * self.len() % std::mem::size_of::<U>() == 0`
    pub unsafe fn transmute<U: Value>(&self) -> BufferView<U> {
        assert_eq!(
            std::mem::size_of::<T>() * self.len() % std::mem::size_of::<U>(),
            0
        );
        BufferView {
            device: self.device.clone(),
            handle: self.handle.clone(),
            offset: self.offset,
            len: self.len * std::mem::size_of::<T>() / std::mem::size_of::<U>(),
            total_size_bytes: self.total_size_bytes,
            _marker: PhantomData,
        }
    }
    #[inline]
    pub fn var(&self) -> BufferVar<T> {
        BufferVar::new(self)
    }
    pub(crate) fn _handle(&self) -> Arc<BufferHandle> {
        Weak::upgrade(&self.handle).unwrap_or_else(|| {
            panic!("BufferView was created from a Buffer that has already been dropped.")
        })
    }
    #[inline]
    pub fn handle(&self) -> api::Buffer {
        self._handle().handle
    }
    #[inline]
    pub fn native_handle(&self) -> *mut c_void {
        self._handle().native_handle
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
    pub fn copy_to_async<'a>(&self, data: &'a mut [T]) -> Command<'a, 'a> {
        assert_eq!(data.len(), self.len);
        let mut rt = ResourceTracker::new();
        rt.add(self._handle());
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
            submit_default_stream_and_sync(&self.device, [self.copy_to_async(data)]);
        }
    }

    pub fn copy_from_async<'a>(&self, data: &'a [T]) -> Command<'a, 'static> {
        assert_eq!(data.len(), self.len);
        let mut rt = ResourceTracker::new();
        rt.add(self._handle());
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
        submit_default_stream_and_sync(&self.device, [self.copy_from_async(data)]);
    }
    pub fn fill_fn<F: FnMut(usize) -> T>(&self, f: F) {
        self.copy_from(&(0..self.len).map(f).collect::<Vec<_>>());
    }
    pub fn fill(&self, value: T) {
        self.fill_fn(|_| value);
    }
    pub fn copy_to_buffer_async(&self, dst: &BufferView<T>) -> Command<'static, 'static> {
        assert_eq!(self.len, dst.len);
        let mut rt = ResourceTracker::new();
        rt.add(self._handle());
        rt.add(dst._handle());
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
    pub fn copy_to_buffer(&self, dst: &BufferView<T>) {
        submit_default_stream_and_sync(&self.device, [self.copy_to_buffer_async(dst)]);
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
            device: self.device.clone(),
            handle: self.handle.clone(),
            offset: lower,
            len: upper - lower,
            total_size_bytes: self.total_size_bytes,
            _marker: PhantomData,
        }
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

pub struct BindlessArray {
    pub(crate) device: Device,
    pub(crate) handle: Arc<BindlessArrayHandle>,
    pub(crate) modifications: RefCell<HashMap<usize, api::BindlessArrayUpdateModification>>,
    pub(crate) slots: RefCell<Vec<BindlessArraySlot>>,
    pub(crate) lock: Arc<RawMutex>,
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
    pub fn var(&self) -> BindlessArrayVar {
        self.lock();
        assert!(
            self.modifications.borrow().is_empty(),
            "Did not call update() after last modification"
        );
        let var = BindlessArrayVar::new(self);
        self.unlock();
        var
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
    pub fn emplace_byte_buffer_view_async(&self, index: usize, bufferview: &ByteBufferView) {
        self.emplace_buffer_view_async(index, bufferview)
    }
    pub fn emplace_buffer_async<T: Value>(&self, index: usize, buffer: &Buffer<T>) {
        self.emplace_buffer_view_async(index, &buffer.view(..))
    }
    pub fn emplace_buffer_view_async<T: Value>(&self, index: usize, bufferview: &BufferView<T>) {
        self.lock();
        self.modifications
            .borrow_mut()
            .entry(index)
            .or_insert(api::BindlessArrayUpdateModification {
                slot: index,
                buffer: Default::default(),
                tex2d: Default::default(),
                tex3d: Default::default(),
            })
            .buffer = api::BindlessArrayUpdateBuffer {
            op: api::BindlessArrayUpdateOperation::Emplace,
            handle: bufferview.handle(),
            offset: bufferview.offset,
        };
        let mut slots = self.slots.borrow_mut();
        slots[index].buffer = Some(bufferview._handle());
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
            .entry(index)
            .or_insert(api::BindlessArrayUpdateModification {
                slot: index,
                buffer: Default::default(),
                tex2d: Default::default(),
                tex3d: Default::default(),
            })
            .tex2d = api::BindlessArrayUpdateTexture {
            op: api::BindlessArrayUpdateOperation::Emplace,
            handle: texture.handle(),
            sampler,
        };
        let mut slots = self.slots.borrow_mut();
        slots[index].tex2d = Some(texture.handle.clone());
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
            .entry(index)
            .or_insert(api::BindlessArrayUpdateModification {
                slot: index,
                buffer: Default::default(),
                tex2d: Default::default(),
                tex3d: Default::default(),
            })
            .tex3d = api::BindlessArrayUpdateTexture {
            op: api::BindlessArrayUpdateOperation::Emplace,
            handle: texture.handle(),
            sampler,
        };
        let mut slots = self.slots.borrow_mut();
        slots[index].tex3d = Some(texture.handle.clone());
        self.unlock();
    }
    pub fn remove_buffer_async(&self, index: usize) {
        self.lock();
        self.modifications
            .borrow_mut()
            .entry(index)
            .or_insert(api::BindlessArrayUpdateModification {
                slot: index,
                buffer: Default::default(),
                tex2d: Default::default(),
                tex3d: Default::default(),
            })
            .buffer = api::BindlessArrayUpdateBuffer {
            op: api::BindlessArrayUpdateOperation::Remove,
            handle: api::Buffer(INVALID_RESOURCE_HANDLE),
            offset: 0,
        };
        let mut slots = self.slots.borrow_mut();
        slots[index].buffer = None;
        self.unlock();
    }
    pub fn remove_tex2d_async(&self, index: usize) {
        self.lock();
        self.modifications
            .borrow_mut()
            .entry(index)
            .or_insert(api::BindlessArrayUpdateModification {
                slot: index,
                buffer: Default::default(),
                tex2d: Default::default(),
                tex3d: Default::default(),
            })
            .tex2d = api::BindlessArrayUpdateTexture {
            op: api::BindlessArrayUpdateOperation::Remove,
            handle: api::Texture(INVALID_RESOURCE_HANDLE),
            sampler: Sampler::default(),
        };
        let mut slots = self.slots.borrow_mut();
        slots[index].tex2d = None;
        self.unlock();
    }
    pub fn remove_tex3d_async(&self, index: usize) {
        self.lock();
        self.modifications
            .borrow_mut()
            .entry(index)
            .or_insert(api::BindlessArrayUpdateModification {
                slot: index,
                buffer: Default::default(),
                tex2d: Default::default(),
                tex3d: Default::default(),
            })
            .tex3d = api::BindlessArrayUpdateTexture {
            op: api::BindlessArrayUpdateOperation::Remove,
            handle: api::Texture(INVALID_RESOURCE_HANDLE),
            sampler: Sampler::default(),
        };
        let mut slots = self.slots.borrow_mut();
        slots[index].tex3d = None;
        self.unlock();
    }
    #[inline]
    pub fn emplace_byte_buffer(&self, index: usize, buffer: &ByteBuffer) {
        self.emplace_byte_buffer_async(index, buffer);
        self.update();
    }
    #[inline]
    pub fn emplace_byte_buffer_view(&self, index: usize, buffer: &ByteBufferView) {
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
    pub fn update_async<'a>(&'a self) -> Command<'a, 'a> {
        // What lifetime should this be?
        self.lock();
        let mut rt = ResourceTracker::new();
        let mut modifications = self.modifications.borrow_mut();
        let modifications = Arc::new(modifications.drain().map(|(_k, v)| v).collect::<Vec<_>>());
        rt.add(modifications.clone());
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
    pub fn tex2d(&self, tex2d_index: impl AsExpr<Value = u32>) -> BindlessTex2dVar {
        self.var().tex2d(tex2d_index)
    }
    pub fn tex3d(&self, tex3d_index: impl AsExpr<Value = u32>) -> BindlessTex3dVar {
        self.var().tex3d(tex3d_index)
    }
    pub fn buffer<T: Value>(&self, buffer_index: impl AsExpr<Value = u32>) -> BindlessBufferVar<T> {
        self.var().buffer::<T>(buffer_index)
    }
}
unsafe impl Send for BindlessArray {}
unsafe impl Sync for BindlessArray {}
pub use api::{PixelFormat, PixelStorage, Sampler, SamplerAddress, SamplerFilter};
use luisa_compute_ir::context::type_hash;
use luisa_compute_ir::ir::{
    Binding, BindlessArrayBinding, BufferBinding, Func, Instruction, Node, TextureBinding,
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
    pub(crate) views: Vec<Tex2dView<T>>,
}

impl<T: IoTexel> Tex2d<T> {
    /// Create a new texture with the same dimensions and storage as `self`
    /// and copy the contents of `self` to it asynchronously
    pub fn copy_async<'a>(&self, s: &Scope<'a>) -> Self {
        let h = self.handle.as_ref();
        let width = self.width;
        let height = self.height;
        let storage = h.storage;
        let mips = h.levels;
        let device = &h.device;
        let copy = device.create_tex2d::<T>(storage, width, height, mips);
        s.submit((0..mips).map(|level| self.view(level).copy_to_texture_async(&copy.view(level))));
        copy
    }

    /// Create a new texture with the same dimensions and storage as `self`
    /// and copy the contents of `self` to it.
    pub fn copy(&self) -> Self {
        let default_stream = self.handle.device.default_stream();
        default_stream.with_scope(|s| self.copy_async(s))
    }
}

impl<T: IoTexel + fmt::Debug> fmt::Debug for Tex2d<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tex2d<{}>({}, {})",
            std::any::type_name::<T>(),
            self.width(),
            self.height(),
        )
    }
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
    pub(crate) views: Vec<Tex3dView<T>>,
}
impl<T: IoTexel> Tex3d<T> {
    /// Create a new texture with the same dimensions and storage as `self`
    /// and copy the contents of `self` to it asynchronously
    pub fn copy_async(&self, s: &Scope) -> Self {
        let h = self.handle.as_ref();
        let width = self.width;
        let height = self.height;
        let depth = self.depth;
        let storage = h.storage;
        let mips = h.levels;
        let device = &h.device;
        let copy = device.create_tex3d::<T>(storage, width, height, depth, mips);
        s.submit((0..mips).map(|level| self.view(level).copy_to_texture_async(&copy.view(level))));
        copy
    }

    /// Create a new texture with the same dimensions and storage as `self`
    /// and copy the contents of `self` to it.
    pub fn copy(&self) -> Self {
        let default_stream = self.handle.device.default_stream();
        default_stream.with_scope(|s| self.copy_async(s))
    }
}
impl<T: IoTexel + fmt::Debug> fmt::Debug for Tex3d<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tex3d<{}>({}, {}, {})",
            std::any::type_name::<T>(),
            self.width(),
            self.height(),
            self.depth(),
        )
    }
}

#[derive(Clone)]
pub struct Tex2dView<T: IoTexel> {
    pub(crate) device: Device,
    #[allow(dead_code)]
    pub(crate) width: u32,
    #[allow(dead_code)]
    pub(crate) height: u32,
    pub(crate) storage: PixelStorage,
    pub(crate) format: PixelFormat,
    pub(crate) handle: Weak<TextureHandle>,
    pub(crate) level: u32,
    pub(crate) marker: PhantomData<fn() -> T>,
}
#[derive(Clone)]
pub struct Tex3dView<T: IoTexel> {
    pub(crate) device: Device,
    #[allow(dead_code)]
    pub(crate) width: u32,
    #[allow(dead_code)]
    pub(crate) height: u32,
    #[allow(dead_code)]
    pub(crate) depth: u32,
    pub(crate) storage: PixelStorage,
    pub(crate) format: PixelFormat,
    pub(crate) handle: Weak<TextureHandle>,
    pub(crate) level: u32,
    pub(crate) marker: PhantomData<fn() -> T>,
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
        impl<T: IoTexel> $name<T> {
            pub(crate) fn _handle(&self) -> Arc<TextureHandle> {
                self.handle.upgrade().unwrap()
            }
            pub fn copy_to_async<'a, U: StorageTexel<T>>(
                &self,
                data: &'a mut [U],
            ) -> Command<'a, 'a> {
                assert_eq!(data.len(), self.texel_count() as usize);
                assert_eq!(self.storage, U::pixel_storage());
                let mut rt = ResourceTracker::new();
                rt.add(self._handle());
                Command {
                    inner: api::Command::TextureDownload(api::TextureDownloadCommand {
                        texture: self.handle(),
                        storage: self.storage,
                        level: self.level,
                        size: self.size(),
                        data: data.as_mut_ptr() as *mut u8,
                    }),
                    resource_tracker: rt,
                    marker: PhantomData,
                    callback: None,
                }
            }
            pub fn copy_to<U: StorageTexel<T>>(&self, data: &mut [U]) {
                assert_eq!(data.len(), self.texel_count() as usize);

                submit_default_stream_and_sync(&self.device, [self.copy_to_async(data)]);
            }
            pub fn copy_to_vec<U: StorageTexel<T>>(&self) -> Vec<U> {
                let mut data = Vec::with_capacity(self.texel_count() as usize);
                unsafe {
                    data.set_len(self.texel_count() as usize);
                }
                self.copy_to(&mut data);
                data
            }
            pub fn copy_from_async<U: StorageTexel<T>>(
                &self,
                data: &[U],
            ) -> Command<'static, 'static> {
                assert_eq!(data.len(), self.texel_count() as usize);
                assert_eq!(self.storage, U::pixel_storage());
                let mut rt = ResourceTracker::new();
                rt.add(self._handle());
                Command {
                    inner: api::Command::TextureUpload(api::TextureUploadCommand {
                        texture: self.handle(),
                        storage: self.storage,
                        level: self.level,
                        size: self.size(),
                        data: data.as_ptr() as *const u8,
                    }),
                    resource_tracker: rt,
                    marker: PhantomData,
                    callback: None,
                }
            }
            pub fn copy_from<U: StorageTexel<T>>(&self, data: &[U]) {
                submit_default_stream_and_sync(&self.device, [self.copy_from_async(data)]);
            }
            pub fn copy_to_buffer_async<U: StorageTexel<T> + Value>(
                &self,
                buffer_view: &BufferView<U>,
            ) -> Command<'static, 'static> {
                let mut rt = ResourceTracker::new();
                rt.add(self._handle());
                rt.add(buffer_view._handle());
                assert_eq!(buffer_view.len, self.texel_count() as usize);
                assert_eq!(self.storage, U::pixel_storage());
                Command {
                    inner: api::Command::TextureToBufferCopy(api::TextureToBufferCopyCommand {
                        texture: self.handle(),
                        storage: self.storage,
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
            pub fn copy_to_buffer<U: StorageTexel<T> + Value>(&self, buffer_view: &BufferView<U>) {
                submit_default_stream_and_sync(
                    &self.device,
                    [self.copy_to_buffer_async(buffer_view)],
                );
            }
            pub fn copy_from_buffer_async<U: StorageTexel<T> + Value>(
                &self,
                buffer_view: &BufferView<U>,
            ) -> Command<'static, 'static> {
                let mut rt = ResourceTracker::new();
                rt.add(self._handle());
                rt.add(buffer_view._handle());
                assert_eq!(buffer_view.len, self.texel_count() as usize);
                assert_eq!(self.storage, U::pixel_storage());
                Command {
                    inner: api::Command::BufferToTextureCopy(api::BufferToTextureCopyCommand {
                        texture: self.handle(),
                        storage: self.storage,
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
                &self,
                buffer_view: &BufferView<U>,
            ) {
                submit_default_stream_and_sync(
                    &self.device,
                    [self.copy_from_buffer_async(buffer_view)],
                );
            }
            pub fn copy_to_texture_async(&self, other: &$name<T>) -> Command<'static, 'static> {
                let mut rt = ResourceTracker::new();
                rt.add(self._handle());
                rt.add(other._handle());
                assert_eq!(self.size(), other.size());
                assert_eq!(self.storage, other.storage);
                assert_eq!(self.format, other.format);
                Command {
                    inner: api::Command::TextureCopy(api::TextureCopyCommand {
                        src: self.handle(),
                        storage: self.storage,
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
            pub fn copy_to_texture(&self, other: &$name<T>) {
                submit_default_stream_and_sync(&self.device, [self.copy_to_texture_async(other)]);
            }
        }
    };
}
impl<T: IoTexel> Tex2dView<T> {
    pub fn handle(&self) -> api::Texture {
        self._handle().handle
    }
    pub fn texel_count(&self) -> u32 {
        let s = self.size();
        s[0] * s[1]
    }
    pub fn size(&self) -> [u32; 3] {
        [
            (self.width >> self.level).max(1),
            (self.height >> self.level).max(1),
            1,
        ]
    }
    pub fn var(&self) -> Tex2dVar<T> {
        Tex2dVar::new(self.clone())
    }
}
impl_tex_view!(Tex2dView);
impl<T: IoTexel> Tex3dView<T> {
    pub fn handle(&self) -> api::Texture {
        self._handle().handle
    }
    pub fn texel_count(&self) -> u32 {
        let s = self.size();
        s[0] * s[1] * s[2]
    }
    pub fn size(&self) -> [u32; 3] {
        [
            (self.width >> self.level).max(1),
            (self.height >> self.level).max(1),
            (self.depth >> self.level).max(1),
        ]
    }
    pub fn var(&self) -> Tex3dVar<T> {
        Tex3dVar::new(self.clone())
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
        self.views[level as usize].clone()
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
    // pub fn read(&self, uv: impl AsExpr<Value = Uint2>) -> Expr<T> {
    //     self.var().read(uv)
    // }
    // pub fn write(&self, uv: impl AsExpr<Value = Uint2>, v: impl AsExpr<Value = T>) {
    //     self.var().write(uv, v)
    // }
}
impl<T: IoTexel> Tex3d<T> {
    pub fn view(&self, level: u32) -> Tex3dView<T> {
        self.views[level as usize].clone()
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
    // pub fn read(&self, uv: impl AsExpr<Value = Uint3>) -> Expr<T> {
    //     self.var().read(uv)
    // }
    // pub fn write(&self, uv: impl AsExpr<Value = Uint3>, v: impl AsExpr<Value = T>) {
    //     self.var().write(uv, v)
    // }
}
#[derive(Clone)]
pub struct BufferVar<T: Value> {
    pub(crate) marker: PhantomData<T>,
    #[allow(dead_code)]
    pub(crate) handle: Option<Arc<BufferHandle>>,
    pub(crate) node: SafeNodeRef,
}
impl<T: Value> ToNode for BufferVar<T> {
    fn node(&self) -> SafeNodeRef {
        self.node
    }
}
impl<T: Value> Drop for BufferVar<T> {
    fn drop(&mut self) {}
}
#[derive(Clone)]
pub struct BindlessArrayVar {
    pub(crate) node: SafeNodeRef,
    #[allow(dead_code)]
    pub(crate) handle: Option<Arc<BindlessArrayHandle>>,
}
#[derive(Clone)]
pub struct BindlessBufferVar<T> {
    array: SafeNodeRef,
    buffer_index: Expr<u32>,
    _marker: PhantomData<T>,
}
impl<T: Value> ToNode for BindlessBufferVar<T> {
    fn node(&self) -> SafeNodeRef {
        self.array
    }
}

impl<T: Value> IndexRead for BindlessBufferVar<T> {
    type Element = T;
    fn read<I: IntoIndex>(&self, i: I) -> Expr<T> {
        let i = i.to_u64();
        if need_runtime_check() {
            lc_assert!(i.lt(self.len_expr()));
        }
        let array = self.array.get();
        let buffer_index = self.buffer_index.node().get();
        let i = i.node().get();
        Expr::<T>::from_node(
            __current_scope(|b| {
                b.call(
                    Func::BindlessBufferRead,
                    &[array, buffer_index, i],
                    T::type_(),
                )
            })
            .into(),
        )
    }
}
impl<T: Value> IndexWrite for BindlessBufferVar<T> {
    fn write<I: IntoIndex, V: AsExpr<Value = Self::Element>>(&self, i: I, value: V) {
        let i = i.to_u64();
        if need_runtime_check() {
            lc_assert!(i.lt(self.len_expr()));
        }
        let array = self.array.get();
        let buffer_index = self.buffer_index.node().get();
        let i = i.node().get();
        let value = value.as_expr().node().get();
        __current_scope(|b| {
            b.call(
                Func::BindlessBufferWrite,
                &[array, buffer_index, i, value],
                Type::void(),
            )
        });
    }
}
impl<T: Value> BindlessBufferVar<T> {
    pub fn len_expr(&self) -> Expr<u64> {
        let stride = (T::type_().size() as u64).expr().node().get();
        let array = self.array.get();
        let buffer_index = self.buffer_index.node().get();
        Expr::<u64>::from_node(
            __current_scope(|b| {
                b.call(
                    Func::BindlessBufferSize,
                    &[array, buffer_index, stride],
                    u32::type_(),
                )
            })
            .into(),
        )
    }
    pub fn __type(&self) -> Expr<u64> {
        let array = self.array.get();
        let buffer_index = self.buffer_index.node().get();
        Expr::<u64>::from_node(
            __current_scope(|b| {
                b.call(
                    Func::BindlessBufferType,
                    &[array, buffer_index],
                    u64::type_(),
                )
            })
            .into(),
        )
    }
}
#[derive(Clone)]
pub struct BindlessByteBufferVar {
    array: SafeNodeRef,
    buffer_index: Expr<u32>,
}
impl ToNode for BindlessByteBufferVar {
    fn node(&self) -> SafeNodeRef {
        self.array
    }
}
impl BindlessByteBufferVar {
    pub unsafe fn read_as<T: Value>(&self, index_bytes: impl IntoIndex) -> Expr<T> {
        let i = index_bytes.to_u64().node().get();
        let array = self.array.get();
        let buffer_index = self.buffer_index.node().get();
        Expr::<T>::from_node(
            __current_scope(|b| {
                b.call(
                    Func::BindlessByteBufferRead,
                    &[array, buffer_index, i],
                    <T as TypeOf>::type_(),
                )
            })
            .into(),
        )
    }
    pub fn len(&self) -> Expr<u64> {
        let s = (1u64).expr().node().get();
        let array = self.array.get();
        let buffer_index = self.buffer_index.node().get();
        Expr::<u64>::from_node(
            __current_scope(|b| {
                b.call(
                    Func::BindlessBufferSize,
                    &[array, buffer_index, s],
                    <u64 as TypeOf>::type_(),
                )
            })
            .into(),
        )
    }
}
#[derive(Clone)]
pub struct BindlessTex2dVar {
    array: SafeNodeRef,
    tex2d_index: Expr<u32>,
}

impl BindlessTex2dVar {
    pub fn sample(&self, uv: impl AsExpr<Value = Float2>) -> Expr<Float4> {
        let array = self.array.get();
        let tex2d_index = self.tex2d_index.node().get();
        let uv = uv.as_expr().node().get();
        Expr::<Float4>::from_node(
            __current_scope(|b| {
                b.call(
                    Func::BindlessTexture2dSample,
                    &[array, tex2d_index, uv],
                    Float4::type_(),
                )
            })
            .into(),
        )
    }
    pub fn sample_level(
        &self,
        uv: impl AsExpr<Value = Float2>,
        level: impl AsExpr<Value = u32>,
    ) -> Expr<Float4> {
        let array = self.array.get();
        let tex2d_index = self.tex2d_index.node().get();
        let uv = uv.as_expr().node().get();
        let level = level.as_expr().node().get();
        Expr::<Float4>::from_node(
            __current_scope(|b| {
                b.call(
                    Func::BindlessTexture2dSampleLevel,
                    &[array, tex2d_index, uv, level],
                    Float4::type_(),
                )
            })
            .into(),
        )
    }
    pub fn sample_grad(
        &self,
        uv: impl AsExpr<Value = Float2>,
        ddx: impl AsExpr<Value = Float2>,
        ddy: impl AsExpr<Value = Float2>,
    ) -> Expr<Float4> {
        let array = self.array.get();
        let tex2d_index = self.tex2d_index.node().get();
        let uv = uv.as_expr().node().get();
        let ddx = ddx.as_expr().node().get();
        let ddy = ddy.as_expr().node().get();
        Expr::<Float4>::from_node(
            __current_scope(|b| {
                b.call(
                    Func::BindlessTexture2dSampleLevel,
                    &[array, tex2d_index, uv, ddx, ddy],
                    Float4::type_(),
                )
            })
            .into(),
        )
    }
    pub fn read(&self, coord: impl AsExpr<Value = Uint2>) -> Expr<Float4> {
        let array = self.array.get();
        let tex2d_index = self.tex2d_index.node().get();
        let coord = coord.as_expr().node().get();
        Expr::<Float4>::from_node(
            __current_scope(|b| {
                b.call(
                    Func::BindlessTexture2dRead,
                    &[array, tex2d_index, coord],
                    Float4::type_(),
                )
            })
            .into(),
        )
    }
    pub fn read_level(
        &self,
        coord: impl AsExpr<Value = Uint2>,
        level: impl AsExpr<Value = u32>,
    ) -> Expr<Float4> {
        let array = self.array.get();
        let tex2d_index = self.tex2d_index.node().get();
        let coord = coord.as_expr().node().get();
        let level = level.as_expr().node().get();
        Expr::<Float4>::from_node(
            __current_scope(|b| {
                b.call(
                    Func::BindlessTexture2dReadLevel,
                    &[array, tex2d_index, coord, level],
                    Float4::type_(),
                )
            })
            .into(),
        )
    }
    pub fn size(&self) -> Expr<Uint2> {
        Expr::<Uint2>::from_node(
            __current_scope(|b| {
                let array = self.array.get();
                let tex2d_index = self.tex2d_index.node().get();
                b.call(
                    Func::BindlessTexture2dSize,
                    &[array, tex2d_index],
                    Uint2::type_(),
                )
            })
            .into(),
        )
    }
    pub fn size_level(&self, level: impl AsExpr<Value = u32>) -> Expr<Uint2> {
        let array = self.array.get();
        let tex2d_index = self.tex2d_index.node().get();
        let level = level.as_expr().node().get();
        Expr::<Uint2>::from_node(
            __current_scope(|b| {
                b.call(
                    Func::BindlessTexture2dSizeLevel,
                    &[array, tex2d_index, level],
                    Uint2::type_(),
                )
            })
            .into(),
        )
    }
}
#[derive(Clone)]
pub struct BindlessTex3dVar {
    array: SafeNodeRef,
    tex3d_index: Expr<u32>,
}

impl BindlessTex3dVar {
    pub fn sample(&self, uv: impl AsExpr<Value = Float3>) -> Expr<Float4> {
        let array = self.array.get();
        let tex3d_index = self.tex3d_index.node().get();
        let uv = uv.as_expr().node().get();
        Expr::<Float4>::from_node(
            __current_scope(|b| {
                b.call(
                    Func::BindlessTexture3dSample,
                    &[array, tex3d_index, uv],
                    Float4::type_(),
                )
            })
            .into(),
        )
    }
    pub fn sample_level(
        &self,
        uv: impl AsExpr<Value = Float3>,
        level: impl AsExpr<Value = f32>,
    ) -> Expr<Float4> {
        let array = self.array.get();
        let tex3d_index = self.tex3d_index.node().get();
        let uv = uv.as_expr().node().get();
        let level = level.as_expr().node().get();
        Expr::<Float4>::from_node(
            __current_scope(|b| {
                b.call(
                    Func::BindlessTexture3dSampleLevel,
                    &[array, tex3d_index, uv, level],
                    Float4::type_(),
                )
            })
            .into(),
        )
    }
    pub fn sample_grad(
        &self,
        uv: impl AsExpr<Value = Float3>,
        ddx: impl AsExpr<Value = Float3>,
        ddy: impl AsExpr<Value = Float3>,
    ) -> Expr<Float4> {
        let array = self.array.get();
        let tex3d_index = self.tex3d_index.node().get();
        let uv = uv.as_expr().node().get();
        let ddx = ddx.as_expr().node().get();
        let ddy = ddy.as_expr().node().get();
        Expr::<Float4>::from_node(
            __current_scope(|b| {
                b.call(
                    Func::BindlessTexture3dSampleLevel,
                    &[array, tex3d_index, uv, ddx, ddy],
                    Float4::type_(),
                )
            })
            .into(),
        )
    }
    pub fn read(&self, coord: impl AsExpr<Value = Uint3>) -> Expr<Float4> {
        let array = self.array.get();
        let tex3d_index = self.tex3d_index.node().get();
        let coord = coord.as_expr().node().get();
        Expr::<Float4>::from_node(
            __current_scope(|b| {
                b.call(
                    Func::BindlessTexture3dRead,
                    &[array, tex3d_index, coord],
                    Float4::type_(),
                )
            })
            .into(),
        )
    }
    pub fn read_level(
        &self,
        coord: impl AsExpr<Value = Uint3>,
        level: impl AsExpr<Value = f32>,
    ) -> Expr<Float4> {
        let array = self.array.get();
        let tex3d_index = self.tex3d_index.node().get();
        let coord = coord.as_expr().node().get();
        let level = level.as_expr().node().get();
        Expr::<Float4>::from_node(
            __current_scope(|b| {
                b.call(
                    Func::BindlessTexture3dReadLevel,
                    &[array, tex3d_index, coord, level],
                    Float4::type_(),
                )
            })
            .into(),
        )
    }
    pub fn size(&self) -> Expr<Uint3> {
        let array = self.array.get();
        let tex3d_index = self.tex3d_index.node().get();
        Expr::<Uint3>::from_node(
            __current_scope(|b| {
                b.call(
                    Func::BindlessTexture3dSize,
                    &[array, tex3d_index],
                    Uint3::type_(),
                )
            })
            .into(),
        )
    }
    pub fn size_level(&self, level: impl AsExpr<Value = f32>) -> Expr<Uint3> {
        let array = self.array.get();
        let tex3d_index = self.tex3d_index.node().get();
        let level = level.as_expr().node().get();
        Expr::<Uint3>::from_node(
            __current_scope(|b| {
                b.call(
                    Func::BindlessTexture3dSizeLevel,
                    &[array, tex3d_index, level],
                    Uint3::type_(),
                )
            })
            .into(),
        )
    }
}

impl BindlessArrayVar {
    pub fn tex2d(&self, tex2d_index: impl AsExpr<Value = u32>) -> BindlessTex2dVar {
        let v = BindlessTex2dVar {
            array: self.node,
            tex2d_index: tex2d_index.as_expr(),
        };
        v
    }
    pub fn tex3d(&self, tex3d_index: impl AsExpr<Value = u32>) -> BindlessTex3dVar {
        let v = BindlessTex3dVar {
            array: self.node,
            tex3d_index: tex3d_index.as_expr(),
        };
        v
    }
    pub fn byte_address_buffer(
        &self,
        buffer_index: impl AsExpr<Value = u32>,
    ) -> BindlessByteBufferVar {
        let v = BindlessByteBufferVar {
            array: self.node,
            buffer_index: buffer_index.as_expr(),
        };
        v
    }
    pub fn buffer<T: Value>(&self, buffer_index: impl AsExpr<Value = u32>) -> BindlessBufferVar<T> {
        let v = BindlessBufferVar {
            array: self.node,
            buffer_index: buffer_index.as_expr(),
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
        let node = with_recorder(|r| {
            let handle: u64 = array.handle().0;
            let binding = Binding::BindlessArray(BindlessArrayBinding { handle });
            if let Some((a, b)) = r.check_on_same_device(&array.device) {
                panic!(
                    "BindlessArray created for a device: `{:?}` but used in `{:?}`",
                    b, a
                );
            }
            r.capture_or_get(binding, &Arc::downgrade(&array.handle), || {
                Node::new(CArc::new(Instruction::Bindless), Type::void())
            })
        })
        .into();
        Self {
            node,
            handle: Some(array.handle.clone()),
        }
    }
}

impl<T: Value> std::ops::Deref for Buffer<T> {
    type Target = BufferView<T>;
    fn deref(&self) -> &Self::Target {
        &self.full_view
    }
}
impl<T: Value> ToNode for Buffer<T> {
    fn node(&self) -> SafeNodeRef {
        self.var().node()
    }
}
// impl<T: Value> IndexRead for BufferView<T> {
//     type Element = T;
//     fn read<I: IntoIndex>(&self, i: I) -> Expr<T> {
//         self.var().read(i)
//     }
// }
// impl<T: Value> IndexWrite for BufferView<T> {
//     fn write<I: IntoIndex, V: AsExpr<Value = T>>(&self, i: I, v: V) {
//         self.var().write(i, v)
//     }
// }
// impl<T: Value> IndexRead for Buffer<T> {
//     type Element = T;
//     fn read<I: IntoIndex>(&self, i: I) -> Expr<T> {
//         self.var().read(i)
//     }
// }
// impl<T: Value> IndexWrite for Buffer<T> {
//     fn write<I: IntoIndex, V: AsExpr<Value = T>>(&self, i: I, v: V) {
//         self.var().write(i, v)
//     }
// }
impl<T: Value> IndexRead for BufferVar<T> {
    type Element = T;
    fn read<I: IntoIndex>(&self, i: I) -> Expr<T> {
        let i = i.to_u64();
        if need_runtime_check() {
            lc_assert!(i.lt(self.len_expr()));
        }
        let self_node = self.node.get();
        let i = i.node.get();
        Expr::<T>::from_node(
            __current_scope(|b| b.call(Func::BufferRead, &[self_node, i], T::type_())).into(),
        )
    }
}
impl<T: Value> IndexWrite for BufferVar<T> {
    fn write<I: IntoIndex, V: AsExpr<Value = T>>(&self, i: I, v: V) {
        let i = i.to_u64();
        let v = v.as_expr().node().get();
        if need_runtime_check() {
            lc_assert!(i.lt(self.len_expr()));
        }
        let i = i.node().get();
        let self_node = self.node.get();
        __current_scope(|b| b.call(Func::BufferWrite, &[self_node, i, v], Type::void()));
    }
}
impl<T: Value> BufferVar<T> {
    pub fn new(buffer: &BufferView<T>) -> Self {
        let node = with_recorder(|r| {
            let binding = Binding::Buffer(BufferBinding {
                handle: buffer.handle().0,
                size: buffer.len * std::mem::size_of::<T>(),
                offset: (buffer.offset * std::mem::size_of::<T>()) as u64,
            });
            if let Some((a, b)) = r.check_on_same_device(&buffer.device) {
                panic!(
                    "Buffer created for a device: `{:?}` but used in `{:?}`",
                    b, a
                );
            }
            r.capture_or_get(binding, &buffer.handle, || {
                Node::new(CArc::new(Instruction::Buffer), T::type_())
            })
        })
        .into();
        Self {
            node,
            marker: PhantomData,
            handle: Some(buffer._handle()),
        }
    }
    pub fn atomic_ref(&self, i: impl IntoIndex) -> AtomicRef<T> {
        let i = i.to_u64();
        if need_runtime_check() {
            lc_assert!(i.lt(self.len_expr()));
        }
        let i = i.node().get();
        let self_node = self.node.get();
        AtomicRef::<T>::from_node(
            __current_scope(|b| b.call_no_append(Func::AtomicRef, &[self_node, i], T::type_()))
                .into(),
        )
    }
    pub fn len_expr(&self) -> Expr<u64> {
        let self_node = self.node.get();
        FromNode::from_node(
            __current_scope(|b| b.call(Func::BufferSize, &[self_node], u64::type_())).into(),
        )
    }
    pub fn len_expr_u32(&self) -> Expr<u32> {
        let self_node = self.node.get();
        FromNode::from_node(
            __current_scope(|b| b.call(Func::BufferSize, &[self_node], u32::type_())).into(),
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
                    lc_assert!(i.lt(self.len_expr()));
                }
                let self_node = self.node.get();
                let i = i.node().get();
                let v = v.node().get();
                Expr::<$t>::from_node(
                    __current_scope(|b| {
                        b.call(Func::AtomicExchange, &[self_node, i, v], <$t>::type_())
                    })
                    .into(),
                )
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
                    lc_assert!(i.lt(self.len_expr()));
                }
                let self_node = self.node.get();
                let i = i.node().get();
                let expected = expected.node().get();
                let desired = desired.node().get();
                Expr::<$t>::from_node(
                    __current_scope(|b| {
                        b.call(
                            Func::AtomicCompareExchange,
                            &[self_node, i, expected, desired],
                            <$t>::type_(),
                        )
                    })
                    .into(),
                )
            }
            pub fn atomic_fetch_add<I: IntoIndex, V: AsExpr<Value = $t>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.to_u64();
                let v = v.as_expr();
                if need_runtime_check() {
                    lc_assert!(i.lt(self.len_expr()));
                }
                let self_node = self.node.get();
                let i = i.node().get();
                let v = v.node().get();
                Expr::<$t>::from_node(
                    __current_scope(|b| {
                        b.call(Func::AtomicFetchAdd, &[self_node, i, v], <$t>::type_())
                    })
                    .into(),
                )
            }
            pub fn atomic_fetch_sub<I: IntoIndex, V: AsExpr<Value = $t>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.to_u64();
                let v = v.as_expr();
                if need_runtime_check() {
                    lc_assert!(i.lt(self.len_expr()));
                }
                let self_node = self.node.get();
                let i = i.node().get();
                let v = v.node().get();
                Expr::<$t>::from_node(
                    __current_scope(|b| {
                        b.call(Func::AtomicFetchSub, &[self_node, i, v], <$t>::type_())
                    })
                    .into(),
                )
            }
            pub fn atomic_fetch_min<I: IntoIndex, V: AsExpr<Value = $t>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.to_u64();
                let v = v.as_expr();
                if need_runtime_check() {
                    lc_assert!(i.lt(self.len_expr()));
                }
                let self_node = self.node.get();
                let i = i.node().get();
                let v = v.node().get();
                Expr::<$t>::from_node(
                    __current_scope(|b| {
                        b.call(Func::AtomicFetchMin, &[self_node, i, v], <$t>::type_())
                    })
                    .into(),
                )
            }
            pub fn atomic_fetch_max<I: IntoIndex, V: AsExpr<Value = $t>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.to_u64();
                let v = v.as_expr();
                if need_runtime_check() {
                    lc_assert!(i.lt(self.len_expr()));
                }
                let self_node = self.node.get();
                let i = i.node().get();
                let v = v.node().get();
                Expr::<$t>::from_node(
                    __current_scope(|b| {
                        b.call(Func::AtomicFetchMax, &[self_node, i, v], <$t>::type_())
                    })
                    .into(),
                )
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
                    lc_assert!(i.lt(self.len_expr()));
                }
                let self_node = self.node.get();
                let i = i.node().get();
                let v = v.node().get();
                Expr::<$t>::from_node(
                    __current_scope(|b| {
                        b.call(Func::AtomicFetchAnd, &[self_node, i, v], <$t>::type_())
                    })
                    .into(),
                )
            }
            pub fn atomic_fetch_or<I: IntoIndex, V: AsExpr<Value = $t>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.to_u64();
                let v = v.as_expr();
                if need_runtime_check() {
                    lc_assert!(i.lt(self.len_expr()));
                }
                let self_node = self.node.get();
                let i = i.node().get();
                let v = v.node().get();
                Expr::<$t>::from_node(
                    __current_scope(|b| {
                        b.call(Func::AtomicFetchOr, &[self_node, i, v], <$t>::type_())
                    })
                    .into(),
                )
            }
            pub fn atomic_fetch_xor<I: IntoIndex, V: AsExpr<Value = $t>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.to_u64();
                let v = v.as_expr();
                if need_runtime_check() {
                    lc_assert!(i.lt(self.len_expr()));
                }
                let self_node = self.node.get();
                let i = i.node().get();
                let v = v.node().get();
                Expr::<$t>::from_node(
                    __current_scope(|b| {
                        b.call(Func::AtomicFetchXor, &[self_node, i, v], <$t>::type_())
                    })
                    .into(),
                )
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
    pub(crate) node: SafeNodeRef,
    #[allow(dead_code)]
    pub(crate) handle: Option<Arc<TextureHandle>>,
    pub(crate) marker: PhantomData<T>,
    #[allow(dead_code)]
    pub(crate) level: Option<u32>,
}

impl<T: IoTexel> Tex2dVar<T> {
    pub fn new(view: Tex2dView<T>) -> Self {
        let node = with_recorder(|r| {
            let handle: u64 = view.handle().0;
            let binding = Binding::Texture(TextureBinding {
                handle,
                level: view.level,
            });
            if let Some((a, b)) = r.check_on_same_device(&view.device) {
                panic!(
                    "Tex2d created for a device: `{:?}` but used in `{:?}`",
                    b, a
                );
            }
            r.capture_or_get(binding, &view.handle, || {
                Node::new(CArc::new(Instruction::Texture2D), T::RwType::type_())
            })
        })
        .into();
        Self {
            node,
            handle: Some(view._handle()),
            level: Some(view.level),
            marker: PhantomData,
        }
    }
    pub fn read(&self, uv: impl AsExpr<Value = Uint2>) -> Expr<T> {
        let uv = uv.as_expr().node().get();
        let self_node = self.node.get();
        T::convert_from_read(Expr::<T::RwType>::from_node(
            __current_scope(|b| b.call(Func::Texture2dRead, &[self_node, uv], T::RwType::type_()))
                .into(),
        ))
    }
    pub fn write(&self, uv: impl AsExpr<Value = Uint2>, v: impl AsExpr<Value = T>) {
        let uv = uv.as_expr().node().get();
        let v = v.as_expr();
        let v = T::convert_to_write(v).node().get();
        let self_node = self.node.get();
        __current_scope(|b| {
            b.call(Func::Texture2dWrite, &[self_node, uv, v], Type::void());
        })
    }
    pub fn size(&self) -> Expr<Uint2> {
        let self_node = self.node.get();
        Expr::<Uint2>::from_node(
            __current_scope(|b| b.call(Func::Texture2dSize, &[self_node], Uint2::type_())).into(),
        )
    }
}

impl<T: IoTexel> Tex3dVar<T> {
    pub fn new(view: Tex3dView<T>) -> Self {
        let node = with_recorder(|r| {
            let handle: u64 = view.handle().0;
            let binding = Binding::Texture(TextureBinding {
                handle,
                level: view.level,
            });
            if let Some((a, b)) = r.check_on_same_device(&view.device) {
                panic!(
                    "Tex3d created for a device: `{:?}` but used in `{:?}`",
                    b, a
                );
            }
            r.capture_or_get(binding, &view.handle, || {
                Node::new(CArc::new(Instruction::Texture3D), T::RwType::type_())
            })
        })
        .into();
        Self {
            node,
            handle: Some(view._handle()),
            level: Some(view.level),
            marker: PhantomData,
        }
    }
    pub fn read(&self, uv: impl AsExpr<Value = Uint3>) -> Expr<T> {
        let uv = uv.as_expr().node().get();
        let self_node = self.node.get();
        T::convert_from_read(Expr::<T::RwType>::from_node(
            __current_scope(|b| b.call(Func::Texture3dRead, &[self_node, uv], T::RwType::type_()))
                .into(),
        ))
    }
    pub fn write(&self, uv: impl AsExpr<Value = Uint3>, v: impl AsExpr<Value = T>) {
        let uv = uv.as_expr().node().get();
        let v = v.as_expr();
        let v = T::convert_to_write(v).node().get();
        let self_node = self.node.get();
        __current_scope(|b| {
            b.call(Func::Texture3dWrite, &[self_node, uv, v], Type::void());
        })
    }
    pub fn size(&self) -> Expr<Uint3> {
        let self_node = self.node.get();
        Expr::<Uint3>::from_node(
            __current_scope(|b| b.call(Func::Texture3dSize, &[self_node], Uint3::type_())).into(),
        )
    }
}
#[derive(Clone)]
pub struct Tex3dVar<T: IoTexel> {
    pub(crate) node: SafeNodeRef,
    #[allow(dead_code)]
    pub(crate) handle: Option<Arc<TextureHandle>>,
    pub(crate) marker: PhantomData<T>,
    #[allow(dead_code)]
    pub(crate) level: Option<u32>,
}
