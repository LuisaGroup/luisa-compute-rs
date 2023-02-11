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
    pub unsafe fn copy_to_async(&'a self, data: &'a mut [T]) -> Command<'a> {
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
    pub unsafe fn copy_from_async(&'a self, data: &'a [T]) -> Command<'a> {
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
        unsafe {
            submit_default_stream_and_sync(&self.buffer.device, [self.copy_from_async(data)])
                .unwrap();
        }
    }
    pub fn fill_fn<F: FnMut(usize) -> T>(&self, f: F) {
        self.copy_from(&(0..self.len).map(f).collect::<Vec<_>>());
    }
    pub fn fill(&self, value: T) {
        self.fill_fn(|_| value);
    }
}
impl<T: Value> Buffer<T> {
    pub(crate) fn handle(&self) -> api::Buffer {
        self.handle.handle
    }

    // pub unsafe fn copy_to_async<'a>(&'a self, data: &'a mut [T]) -> Command<'a> {
    //     self.view(..).copy_to_async(data)
    // }

    // pub fn copy_to_sync(&self, data: &[T]) {
    //     self.view(..).copy_to_sync(data)
    // }
    // pub fn copy_from<'a>(&'a self, data: &'a [T]) -> Command<'a> {
    //     self.view(..).copy_from(data)
    // }
    pub fn native_handle(&self) -> *mut c_void {
        self.device.inner.buffer_native_handle(self.handle())
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
        texture: &Tex2D<T>,
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
        texture: &Tex3D<T>,
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
    pub fn set_tex2d<T: Texel>(&self, index: usize, texture: &Tex2D<T>, sampler: Sampler) {
        unsafe {
            self.set_tex2d_async(index, texture, sampler);
            submit_default_stream_and_sync(&self.device, [self.update_async()]).unwrap();
        }
    }
    pub fn set_tex3d<T: Texel>(&self, index: usize, texture: &Tex3D<T>, sampler: Sampler) {
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
    pub unsafe fn update_async<'a>(&'a self) -> Command<'a> {
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
    pub(crate) level: u32,
}

pub trait Texel: Value {
    // acceptable pixel format
    fn pixel_formats() -> &'static [api::PixelFormat];
}
pub struct Image<T: Texel> {
    pub(crate) handle: Arc<TextureHandle>,
    pub(crate) marker: std::marker::PhantomData<T>,
}
pub struct Volume<T: Texel> {
    pub(crate) handle: Arc<TextureHandle>,
    pub(crate) marker: std::marker::PhantomData<T>,
}
impl<T: Texel> Image<T> {
    pub(crate) fn handle(&self) -> api::Texture {
        self.handle.handle
    }
}
impl<T: Texel> Volume<T> {
    pub(crate) fn handle(&self) -> api::Texture {
        self.handle.handle
    }
}
pub type Tex2D<T> = Image<T>;
pub type Tex3D<T> = Volume<T>;
impl Drop for TextureHandle {
    fn drop(&mut self) {
        self.device.inner.destroy_texture(self.handle);
    }
}
