use std::any::Any;
use std::cell::RefCell;
use std::ops::RangeBounds;
use std::sync::Arc;

use crate::*;
use api::BufferDownloadCommand;
use api::BufferUploadCommand;
use lang::BindlessArrayVar;
use lang::BufferVar;
use lang::Value;
use libc::c_void;
use runtime::*;
pub struct Buffer<T: Value> {
    pub(crate) device: Device,
    pub(crate) handle: Arc<BufferHandle>,
    pub(crate) len: usize,
    pub(crate) _marker: std::marker::PhantomData<T>,
}
pub(crate) struct BufferHandle {
    pub(crate) device: Device,
    pub(crate) handle: api::Buffer,
}

impl Drop for BufferHandle {
    fn drop(&mut self) {
        self.device.inner.destroy_buffer(self.handle);
    }
}
pub struct BufferView<'a, T: Value> {
    buffer: &'a Buffer<T>,
    offset: usize,
    len: usize,
}
impl<'a, T: Value> BufferView<'a, T> {
    pub fn copy_to_async(&'a self, data: &'a mut [T]) -> Command<'a> {
        assert_eq!(data.len(), self.len);
        let mut rt = ResourceTracker::new();
        rt.add(self.buffer.handle.clone());
        Command {
            inner: api::Command::BufferDownload(BufferDownloadCommand {
                buffer: self.buffer.handle.handle,
                offset: self.offset * std::mem::size_of::<T>(),
                size: data.len() * std::mem::size_of::<T>(),
                data: data.as_mut_ptr() as *mut u8,
            }),
            marker: std::marker::PhantomData,
            resource_tracker: rt,
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
            submit_default_stream_and_sync(&self.buffer.device, [self.copy_to_async(data)])
                .unwrap();
        }
    }
    pub fn copy_from_async(&'a self, data: &'a [T]) -> Command<'a> {
        assert_eq!(data.len(), self.len);
        let mut rt = ResourceTracker::new();
        rt.add(self.buffer.handle.clone());
        Command {
            inner: api::Command::BufferUpload(BufferUploadCommand {
                buffer: self.buffer.handle.handle,
                offset: self.offset * std::mem::size_of::<T>(),
                size: data.len() * std::mem::size_of::<T>(),
                data: data.as_ptr() as *const u8,
            }),
            marker: std::marker::PhantomData,
            resource_tracker: rt,
        }
    }
    pub fn copy_from(&self, data: &[T]) {
        submit_default_stream_and_sync(&self.buffer.device, [self.copy_from_async(data)]).unwrap();
    }
    pub fn fill_fn<F: FnMut(usize) -> T>(&self, f: F) {
        self.copy_from(&(0..self.len).map(f).collect::<Vec<_>>());
    }
    pub fn fill(&self, value: T) {
        self.fill_fn(|_| value);
    }
    pub fn copy_to_buffer_async(&self, dst: &BufferView<'a, T>) -> Command<'a> {
        assert_eq!(self.len, dst.len);
        let mut rt = ResourceTracker::new();
        rt.add(self.buffer.handle.clone());
        rt.add(dst.buffer.handle.clone());
        Command {
            inner: api::Command::BufferCopy(api::BufferCopyCommand {
                src: self.buffer.handle.handle,
                src_offset: self.offset * std::mem::size_of::<T>(),
                dst: dst.buffer.handle.handle,
                dst_offset: dst.offset * std::mem::size_of::<T>(),
                size: self.len * std::mem::size_of::<T>(),
            }),
            marker: std::marker::PhantomData,
            resource_tracker: rt,
        }
    }
    pub fn copy_to_buffer(&self, dst: &BufferView<'a, T>) {
        submit_default_stream_and_sync(&self.buffer.device, [self.copy_to_buffer_async(dst)])
            .unwrap();
    }
}
impl<T: Value> Buffer<T> {
    pub(crate) fn handle(&self) -> api::Buffer {
        self.handle.handle
    }
    pub fn native_handle(&self) -> *mut c_void {
        self.device.inner.buffer_native_handle(self.handle())
    }
    pub unsafe fn shallow_clone(&self) -> Buffer<T> {
        Buffer {
            device: self.device.clone(),
            handle: self.handle.clone(),
            len: self.len,
            _marker: std::marker::PhantomData,
        }
    }
    pub fn copy_from(&self, data: &[T]) {
        self.view(..).copy_from(data);
    }
    pub fn copy_to(&self, data: &mut [T]) {
        self.view(..).copy_to(data);
    }
    pub fn copy_to_vec(&self) -> Vec<T> {
        self.view(..).copy_to_vec()
    }
    pub fn copy_to_buffer(&self, dst: &Buffer<T>) {
        self.view(..).copy_to_buffer(&dst.view(..));
    }
    pub fn fill_fn<F: FnMut(usize) -> T>(&self, f: F) {
        self.view(..).fill_fn(f);
    }
    pub fn fill(&self, value: T) {
        self.view(..).fill(value);
    }
    pub fn view<'a, S: RangeBounds<u64>>(&'a self, range: S) -> BufferView<'a, T> {
        let lower = range.start_bound();
        let upper = range.end_bound();
        let lower = match lower {
            std::ops::Bound::Included(&x) => x as usize,
            std::ops::Bound::Excluded(&x) => x as usize + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let upper = match upper {
            std::ops::Bound::Included(&x) => x as usize + 1,
            std::ops::Bound::Excluded(&x) => x as usize,
            std::ops::Bound::Unbounded => self.len,
        };
        assert!(lower <= upper);
        assert!(upper <= self.len);
        BufferView {
            buffer: self,
            offset: lower,
            len: (upper - lower) as usize,
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
    pub fn var(&self) -> BufferVar<T> {
        BufferVar::new(self)
    }
}
impl<T: Value> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        let cloned = self.device.create_buffer(self.len).unwrap();
        self.copy_to_buffer(&cloned);
        cloned
    }
}
pub(crate) struct BindlessArrayHandle {
    pub(crate) device: Device,
    pub(crate) handle: api::BindlessArray,
}
impl Drop for BindlessArrayHandle {
    fn drop(&mut self) {
        self.device.inner.destroy_bindless_array(self.handle);
    }
}
pub struct BindlessArray {
    pub(crate) device: Device,
    pub(crate) handle: Arc<BindlessArrayHandle>,
    pub(crate) buffers: RefCell<Vec<Option<Box<dyn Any>>>>,
    pub(crate) tex_2ds: RefCell<Vec<Option<Box<dyn Any>>>>,
    pub(crate) tex_3ds: RefCell<Vec<Option<Box<dyn Any>>>>,
}
impl BindlessArray {
    // pub fn buffer<T:Value>(&self, index: usize)->BufferVar<T> {
    //     todo!()
    // }
    pub fn var(&self) -> BindlessArrayVar {
        BindlessArrayVar::new(self)
    }
    pub fn handle(&self) -> api::BindlessArray {
        self.handle.handle
    }
    pub unsafe fn set_buffer_async<T: Value>(&self, index: usize, buffer: &Buffer<T>) {
        self.device.inner.emplace_buffer_in_bindless_array(
            self.handle.handle,
            index,
            buffer.handle(),
            0,
        );
        self.buffers.borrow_mut()[index] = Some(Box::new(buffer.handle.clone()));
    }
    pub unsafe fn set_tex2d_async<T: Texel>(
        &self,
        index: usize,
        texture: &Tex2d<T>,
        sampler: Sampler,
    ) {
        self.device.inner.emplace_tex2d_in_bindless_array(
            self.handle.handle,
            index,
            texture.handle(),
            sampler,
        );
        self.tex_2ds.borrow_mut()[index] = Some(Box::new(texture.handle.clone()));
    }
    pub unsafe fn set_tex3d_async<T: Texel>(
        &self,
        index: usize,
        texture: &Tex3d<T>,
        sampler: Sampler,
    ) {
        self.device.inner.emplace_tex3d_in_bindless_array(
            self.handle.handle,
            index,
            texture.handle(),
            sampler,
        );
        self.tex_3ds.borrow_mut()[index] = Some(Box::new(texture.handle.clone()));
    }
    pub unsafe fn remove_buffer_async(&self, index: usize) {
        self.device
            .inner
            .remove_buffer_from_bindless_array(self.handle.handle, index);
        self.buffers.borrow_mut()[index] = None;
    }
    pub unsafe fn remove_tex2d_async(&self, index: usize) {
        self.device
            .inner
            .remove_tex2d_from_bindless_array(self.handle.handle, index);
        self.tex_2ds.borrow_mut()[index] = None;
    }
    pub unsafe fn remove_tex3d_async(&self, index: usize) {
        self.device
            .inner
            .remove_tex3d_from_bindless_array(self.handle.handle, index);
        self.tex_3ds.borrow_mut()[index] = None;
    }
    pub fn set_buffer<T: Value>(&self, index: usize, buffer: &Buffer<T>) {
        unsafe {
            self.set_buffer_async(index, buffer);
            submit_default_stream_and_sync(&self.device, [self.update_async()]).unwrap();
        }
    }
    pub fn set_tex2d<T: Texel>(&self, index: usize, texture: &Tex2d<T>, sampler: Sampler) {
        unsafe {
            self.set_tex2d_async(index, texture, sampler);
            submit_default_stream_and_sync(&self.device, [self.update_async()]).unwrap();
        }
    }
    pub fn set_tex3d<T: Texel>(&self, index: usize, texture: &Tex3d<T>, sampler: Sampler) {
        unsafe {
            self.set_tex3d_async(index, texture, sampler);
            submit_default_stream_and_sync(&self.device, [self.update_async()]).unwrap();
        }
    }
    pub fn remove_buffer(&self, index: usize) {
        unsafe {
            self.remove_buffer_async(index);
            submit_default_stream_and_sync(&self.device, [self.update_async()]).unwrap();
        }
    }
    pub fn remove_tex2d(&self, index: usize) {
        unsafe {
            self.remove_tex2d_async(index);
            submit_default_stream_and_sync(&self.device, [self.update_async()]).unwrap();
        }
    }
    pub fn remove_tex3d(&self, index: usize) {
        unsafe {
            self.remove_tex3d_async(index);
            submit_default_stream_and_sync(&self.device, [self.update_async()]).unwrap();
        }
    }
    pub fn update_async<'a>(&'a self) -> Command<'a> {
        let mut rt = ResourceTracker::new();
        rt.add(self.handle.clone());
        Command {
            inner: api::Command::BindlessArrayUpdate(self.handle.handle),
            marker: std::marker::PhantomData,
            resource_tracker: rt,
        }
    }
}
pub use api::{PixelFormat, PixelStorage, Sampler, SamplerAddress, SamplerFilter};
pub(crate) struct TextureHandle {
    pub(crate) device: Device,
    pub(crate) handle: api::Texture,
    #[allow(dead_code)]
    pub(crate) format: PixelFormat,
    pub(crate) storage: PixelStorage,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) depth: u32,
    pub(crate) levels: u32,
}

pub trait Texel: Value {
    // acceptable pixel format
    fn pixel_formats() -> &'static [api::PixelFormat];
}
pub struct Tex2d<T: Texel> {
    pub(crate) handle: Arc<TextureHandle>,
    pub(crate) marker: std::marker::PhantomData<T>,
}
pub struct Tex3d<T: Texel> {
    pub(crate) handle: Arc<TextureHandle>,
    pub(crate) marker: std::marker::PhantomData<T>,
}
pub struct Tex2dView<T: Texel> {
    pub(crate) handle: Arc<TextureHandle>,
    pub(crate) marker: std::marker::PhantomData<T>,
    pub(crate) level: u32,
}
pub struct Tex3dView<T: Texel> {
    pub(crate) handle: Arc<TextureHandle>,
    pub(crate) marker: std::marker::PhantomData<T>,
    pub(crate) level: u32,
}
impl<T: Texel> Tex2d<T> {
    pub(crate) fn handle(&self) -> api::Texture {
        self.handle.handle
    }
}
impl<T: Texel> Tex3d<T> {
    pub(crate) fn handle(&self) -> api::Texture {
        self.handle.handle
    }
}
macro_rules! impl_tex_view {
    ($name:ident) => {
        impl<T: Texel> $name<T> {
            pub fn copy_to_async<'a>(&'a self, data: &'a mut [T]) -> Command<'a> {
                assert_eq!(data.len(), self.texel_count() as usize);
                let mut rt = ResourceTracker::new();
                rt.add(self.handle.clone());
                Command {
                    inner: api::Command::TextureDownload(api::TextureDownloadCommand {
                        texture: self.handle(),
                        storage: self.handle.storage,
                        level: self.level,
                        size: self.size(),
                        data: data.as_mut_ptr() as *mut u8,
                    }),
                    resource_tracker: rt,
                    marker: std::marker::PhantomData,
                }
            }
            pub fn copy_to<'a>(&'a self, data: &'a mut [T]) {
                assert_eq!(data.len(), self.texel_count() as usize);

                submit_default_stream_and_sync(&self.handle.device, [self.copy_to_async(data)])
                    .unwrap();
            }
            pub fn copy_to_vec<'a>(&'a self) -> Vec<T> {
                let mut data = Vec::with_capacity(self.texel_count() as usize);
                self.copy_to(&mut data);
                unsafe {
                    data.set_len(self.texel_count() as usize);
                }
                data
            }
            pub fn copy_from_async<'a>(&'a self, data: &'a [T]) -> Command<'a> {
                assert_eq!(data.len(), self.texel_count() as usize);
                let mut rt = ResourceTracker::new();
                rt.add(self.handle.clone());
                Command {
                    inner: api::Command::TextureUpload(api::TextureUploadCommand {
                        texture: self.handle(),
                        storage: self.handle.storage,
                        level: self.level,
                        size: self.size(),
                        data: data.as_ptr() as *const u8,
                    }),
                    resource_tracker: rt,
                    marker: std::marker::PhantomData,
                }
            }
            pub fn copy_from<'a>(&'a self, data: &[T]) {
                submit_default_stream_and_sync(
                    &self.handle.device,
                    [self.copy_from_async(data)],
                )
                .unwrap();
            }
            pub fn copy_to_buffer_async<'a>(
                &'a self,
                buffer_view: BufferView<'a, T>,
            ) -> Command<'a> {
                let mut rt = ResourceTracker::new();
                rt.add(self.handle.clone());
                rt.add(buffer_view.buffer.handle.clone());
                assert_eq!(buffer_view.len, self.texel_count() as usize);
                Command {
                    inner: api::Command::TextureToBufferCopy(api::TextureToBufferCopyCommand {
                        texture: self.handle(),
                        storage: self.handle.storage,
                        texture_level: self.level,
                        texture_size: self.size(),
                        buffer: buffer_view.buffer.handle(),
                        buffer_offset: buffer_view.offset,
                    }),
                    resource_tracker: rt,
                    marker: std::marker::PhantomData,
                }
            }
            pub fn copy_to_buffer<'a>(&'a self, buffer_view: BufferView<'a, T>) {
                submit_default_stream_and_sync(
                    &self.handle.device,
                    [self.copy_to_buffer_async(buffer_view)],
                )
                .unwrap();
            }
            pub fn copy_from_buffer_async<'a>(
                &'a self,
                buffer_view: BufferView<'a, T>,
            ) -> Command<'a> {
                let mut rt = ResourceTracker::new();
                rt.add(self.handle.clone());
                rt.add(buffer_view.buffer.handle.clone());
                assert_eq!(buffer_view.len, self.texel_count() as usize);
                Command {
                    inner: api::Command::BufferToTextureCopy(api::BufferToTextureCopyCommand {
                        texture: self.handle(),
                        storage: self.handle.storage,
                        texture_level: self.level,
                        texture_size: self.size(),
                        buffer: buffer_view.buffer.handle(),
                        buffer_offset: buffer_view.offset,
                    }),
                    resource_tracker: rt,
                    marker: std::marker::PhantomData,
                }
            }
            pub fn copy_from_buffer<'a>(&'a self, buffer_view: BufferView<'a, T>) {
                submit_default_stream_and_sync(
                    &self.handle.device,
                    [self.copy_from_buffer_async(buffer_view)],
                )
                .unwrap();
            }
            pub fn copy_to_texture_async<'a>(&'a self, other: $name<T>) -> Command<'a> {
                let mut rt = ResourceTracker::new();
                rt.add(self.handle.clone());
                rt.add(other.handle.clone());
                assert_eq!(self.size(), other.size());
                assert_eq!(self.handle.storage, other.handle.storage);
                Command {
                    inner: api::Command::TextureCopy(api::TextureCopyCommand {
                        src: self.handle(),
                        storage: self.handle.storage,
                        src_level: self.level,
                        size: self.size(),
                        dst: other.handle(),
                        dst_level: other.level,
                    }),
                    resource_tracker: rt,
                    marker: std::marker::PhantomData,
                }
            }
            pub fn copy_to_texture<'a>(&'a self, other: $name<T>) {
                submit_default_stream_and_sync(
                    &self.handle.device,
                    [self.copy_to_texture_async(other)],
                )
                .unwrap();
            }
        }
    };
}
impl<T: Texel> Tex2dView<T> {
    pub(crate) fn handle(&self) -> api::Texture {
        self.handle.handle
    }
    pub fn texel_count(&self) -> u32 {
        let s = self.size();
        s[0] * s[1]
    }
    pub fn size(&self) -> [u32; 3] {
        [
            (self.handle.width >> self.level).max(1),
            (self.handle.height >> self.level).max(1),
            1,
        ]
    }
}
impl_tex_view!(Tex2dView);
impl<T: Texel> Tex3dView<T> {
    pub(crate) fn handle(&self) -> api::Texture {
        self.handle.handle
    }
    pub fn texel_count(&self) -> u32 {
        let s = self.size();
        s[0] * s[1] * s[2]
    }
    pub fn size(&self) -> [u32; 3] {
        [
            (self.handle.width >> self.level).max(1),
            (self.handle.height >> self.level).max(1),
            (self.handle.depth >> self.level).max(1),
        ]
    }
}
impl_tex_view!(Tex3dView);
impl Drop for TextureHandle {
    fn drop(&mut self) {
        self.device.inner.destroy_texture(self.handle);
    }
}
