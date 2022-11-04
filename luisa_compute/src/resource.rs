use std::ops::Range;
use std::ops::RangeBounds;
use std::sync::Arc;

use crate::*;
use api::BufferDownloadCommand;
use api::BufferUploadCommand;
use lang::Value;
use runtime::*;
use sys::{LCPixelFormat, LCPixelStorage};
pub struct Buffer<T: Value> {
    pub(crate) device: Device,
    pub(crate) handle: Arc<BufferHandle>,
    pub(crate) len: usize,
    pub(crate) _marker: std::marker::PhantomData<T>,
}
pub(crate) struct BufferHandle {
    pub(crate) device: Device,
    pub(crate) handle: sys::LCBuffer,
}

impl Drop for BufferHandle {
    fn drop(&mut self) {
        catch_abort! {{
            sys::luisa_compute_buffer_destroy(self.device.handle(), self.handle);
        }}
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
        Command {
            inner: api::Command::BufferDownload(BufferDownloadCommand {
                buffer: api::Buffer(self.buffer.handle.handle._0),
                offset: self.offset * std::mem::size_of::<T>(),
                size: data.len() * std::mem::size_of::<T>(),
                data: data.as_mut_ptr() as *mut u8,
            }),
            marker: std::marker::PhantomData,
            resource_tracker: vec![Box::new(self.buffer.handle.clone())],
        }
    }

    pub fn copy_to(&self, data: &mut [T]) {
        unsafe {
            submit_default_stream_and_sync(&self.buffer.device, [self.copy_to_async(data)]);
        }
    }
    pub unsafe fn copy_from_async(&'a self, data: &'a [T]) -> Command<'a> {
        assert_eq!(data.len(), self.len);
        Command {
            inner: api::Command::BufferUpload(BufferUploadCommand {
                buffer: api::Buffer(self.buffer.handle.handle._0),
                offset: self.offset * std::mem::size_of::<T>(),
                size: data.len() * std::mem::size_of::<T>(),
                data: data.as_ptr() as *const u8,
            }),
            marker: std::marker::PhantomData,
            resource_tracker: vec![Box::new(self.buffer.handle.clone())],
        }
    }
    pub fn copy_from(&self, data: &[T]) {
        unsafe {
            submit_default_stream_and_sync(&self.buffer.device, [self.copy_from_async(data)]);
        }
    }
}
impl<T: Value> Buffer<T> {
    pub(crate) fn handle(&self) -> sys::LCBuffer {
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
}
pub(crate) struct BindlessArrayHandle {
    pub(crate) device: Device,
    pub(crate) handle: sys::LCBindlessArray,
}
impl Drop for BindlessArrayHandle {
    fn drop(&mut self) {
        catch_abort! {{
            sys::luisa_compute_bindless_array_destroy(self.device.handle(), self.handle);
        }}
    }
}
pub struct BindlessArray {
    pub(crate) device: Device,
    pub(crate) handle: Arc<BindlessArrayHandle>,
}
impl BindlessArray {
    // pub fn buffer<T:Value>(&self, index: usize)->BufferVar<T> {
    //     todo!()
    // }
    pub unsafe fn set_buffer_async<T: Value>(&self, index: usize, buffer: &Buffer<T>) {
        catch_abort! {{
            sys::luisa_compute_bindless_array_emplace_buffer(
                self.device.handle(),
                self.handle.handle,
                index as u64,
                buffer.handle(),
            );
        }}
    }
    pub unsafe fn set_tex2d_async<T: Texel>(
        &self,
        index: usize,
        texture: &Tex2D<T>,
        sampler: Sampler,
    ) {
        catch_abort! {{
            sys::luisa_compute_bindless_array_emplace_tex2d(
                self.device.handle(),
                self.handle.handle,
                index as u64,
                texture.handle(),
                unsafe{std::mem::transmute(sampler)},
            );
        }}
    }
    pub unsafe fn set_tex3d_async<T: Texel>(
        &self,
        index: usize,
        texture: &Tex3D<T>,
        sampler: Sampler,
    ) {
        catch_abort! {{
            sys::luisa_compute_bindless_array_emplace_tex3d(
                self.device.handle(),
                self.handle.handle,
                index as u64,
                texture.handle(),
                unsafe{std::mem::transmute(sampler)},
            );
        }}
    }
    pub unsafe fn remove_buffer_async(&self, index: usize) {
        catch_abort! {{
            sys::luisa_compute_bindless_array_remove_buffer(
                self.device.handle(),
                self.handle.handle,
                index as u64,
            );
        }}
    }
    pub unsafe fn remove_tex2d_async(&self, index: usize) {
        catch_abort! {{
            sys::luisa_compute_bindless_array_remove_tex2d(
                self.device.handle(),
                self.handle.handle,
                index as u64,
            );
        }}
    }
    pub unsafe fn remove_tex3d_async(&self, index: usize) {
        catch_abort! {{
            sys::luisa_compute_bindless_array_remove_tex3d(
                self.device.handle(),
                self.handle.handle,
                index as u64,
            );
        }}
    }
    pub fn set_buffer<T: Value>(&self, index: usize, buffer: &Buffer<T>) {
        unsafe {
            self.set_buffer_async(index, buffer);
            submit_default_stream_and_sync(&self.device, [self.update_async()]);
        }
    }
    pub fn set_tex2d<T: Texel>(&self, index: usize, texture: &Tex2D<T>, sampler: Sampler) {
        unsafe {
            self.set_tex2d_async(index, texture, sampler);
            submit_default_stream_and_sync(&self.device, [self.update_async()]);
        }
    }
    pub fn set_tex3d<T: Texel>(&self, index: usize, texture: &Tex3D<T>, sampler: Sampler) {
        unsafe {
            self.set_tex3d_async(index, texture, sampler);
            submit_default_stream_and_sync(&self.device, [self.update_async()]);
        }
    }
    pub fn remove_buffer(&self, index: usize) {
        unsafe {
            self.remove_buffer_async(index);
            submit_default_stream_and_sync(&self.device, [self.update_async()]);
        }
    }
    pub fn remove_tex2d(&self, index: usize) {
        unsafe {
            self.remove_tex2d_async(index);
            submit_default_stream_and_sync(&self.device, [self.update_async()]);
        }
    }
    pub fn remove_tex3d(&self, index: usize) {
        unsafe {
            self.remove_tex3d_async(index);
            submit_default_stream_and_sync(&self.device, [self.update_async()]);
        }
    }
    pub unsafe fn update_async<'a>(&'a self) -> Command<'a> {
        catch_abort! {{
            Command {
                inner: api::Command::BindlessArrayUpdate(api::BindlessArray(self.handle.handle._0)),
                marker: std::marker::PhantomData,
                resource_tracker: vec![Box::new(self.handle.clone())],
            }
        }}
    }
}
pub use api::{PixelFormat, PixelStorage, Sampler, SamplerAddress, SamplerFilter};
pub(crate) struct TextureHandle {
    pub(crate) device: Device,
    pub(crate) handle: sys::LCTexture,
    pub(crate) format: PixelFormat,
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
    pub(crate) fn handle(&self) -> sys::LCTexture {
        self.handle.handle
    }
}
impl<T: Texel> Volume<T> {
    pub(crate) fn handle(&self) -> sys::LCTexture {
        self.handle.handle
    }
}
pub type Tex2D<T> = Image<T>;
pub type Tex3D<T> = Volume<T>;
impl Drop for TextureHandle {
    fn drop(&mut self) {
        catch_abort! {{
            sys::luisa_compute_texture_destroy(self.device.handle(), self.handle);
        }}
    }
}
