use std::any::Any;
use std::cell::RefCell;
use std::collections::HashSet;
use std::ops::RangeBounds;
use std::sync::Arc;

use crate::prelude::*;
use crate::*;
use api::BufferDownloadCommand;
use api::BufferUploadCommand;
use lang::BindlessArrayVar;
use lang::BufferVar;
use lang::Value;
use lazy_static::lazy_static;
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
pub struct BufferView<T: Value> {
    pub(crate) handle: Arc<BufferHandle>,
    pub(crate) device: Device,
    pub(crate) offset: usize,
    pub(crate) len: usize,
    _marker: std::marker::PhantomData<T>,
}
impl<T: Value> BufferView<T> {
    pub(crate) fn handle(&self) -> api::Buffer {
        self.handle.handle
    }
    pub fn copy_to_async<'a>(&'a self, data: &'a mut [T]) -> Command<'a> {
        assert_eq!(data.len(), self.len);
        let mut rt = ResourceTracker::new();
        rt.add(self.handle.clone());
        Command {
            inner: api::Command::BufferDownload(BufferDownloadCommand {
                buffer: self.handle.handle,
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
            submit_default_stream_and_sync(&self.device, [self.copy_to_async(data)]).unwrap();
        }
    }
    pub fn copy_from_async<'a>(&'a self, data: &'a [T]) -> Command<'a> {
        assert_eq!(data.len(), self.len);
        let mut rt = ResourceTracker::new();
        rt.add(self.handle.clone());
        Command {
            inner: api::Command::BufferUpload(BufferUploadCommand {
                buffer: self.handle.handle,
                offset: self.offset * std::mem::size_of::<T>(),
                size: data.len() * std::mem::size_of::<T>(),
                data: data.as_ptr() as *const u8,
            }),
            marker: std::marker::PhantomData,
            resource_tracker: rt,
        }
    }
    pub fn copy_from(&self, data: &[T]) {
        submit_default_stream_and_sync(&self.device, [self.copy_from_async(data)]).unwrap();
    }
    pub fn fill_fn<F: FnMut(usize) -> T>(&self, f: F) {
        self.copy_from(&(0..self.len).map(f).collect::<Vec<_>>());
    }
    pub fn fill(&self, value: T) {
        self.fill_fn(|_| value);
    }
    pub fn copy_to_buffer_async<'a>(&self, dst: &'a BufferView<T>) -> Command<'a> {
        assert_eq!(self.len, dst.len);
        let mut rt = ResourceTracker::new();
        rt.add(self.handle.clone());
        rt.add(dst.handle.clone());
        Command {
            inner: api::Command::BufferCopy(api::BufferCopyCommand {
                src: self.handle.handle,
                src_offset: self.offset * std::mem::size_of::<T>(),
                dst: dst.handle.handle,
                dst_offset: dst.offset * std::mem::size_of::<T>(),
                size: self.len * std::mem::size_of::<T>(),
            }),
            marker: std::marker::PhantomData,
            resource_tracker: rt,
        }
    }
    pub fn copy_to_buffer(&self, dst: &BufferView<T>) {
        submit_default_stream_and_sync(&self.device, [self.copy_to_buffer_async(dst)]).unwrap();
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
    pub fn view<S: RangeBounds<u64>>(&self, range: S) -> BufferView<T> {
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
            device: self.device.clone(),
            handle: self.handle.clone(),
            offset: lower,
            len: (upper - lower) as usize,
            _marker: std::marker::PhantomData,
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
    pub unsafe fn set_tex2d_async<T: IoTexel>(
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
    pub unsafe fn set_tex3d_async<T: IoTexel>(
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
    pub fn set_tex2d<T: IoTexel>(&self, index: usize, texture: &Tex2d<T>, sampler: Sampler) {
        unsafe {
            self.set_tex2d_async(index, texture, sampler);
            submit_default_stream_and_sync(&self.device, [self.update_async()]).unwrap();
        }
    }
    pub fn set_tex3d<T: IoTexel>(&self, index: usize, texture: &Tex3d<T>, sampler: Sampler) {
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
    #[allow(dead_code)]
    pub(crate) levels: u32,
}
// Type that can be converted from a pixel format
// This is the type that is read from/written to a texture
pub trait IoTexel: Value {
    // acceptable pixel format
    fn pixel_formats() -> &'static HashSet<api::PixelFormat>;
    fn pixel_storage() -> &'static HashSet<api::PixelStorage>;
}

macro_rules! impl_io_texel {
    ($t:ty, $($formats:ident),*) => {
        impl IoTexel for $t {
            fn pixel_formats() -> &'static HashSet<api::PixelFormat> {
                lazy_static!{
                    static ref FORMATS: HashSet<api::PixelFormat> = {
                        let mut set = HashSet::new();
                        $(
                            set.insert(api::PixelFormat::$formats);
                        )*
                        set = set.union(<$t as IoTexel>::pixel_formats()).cloned().collect();
                        set
                    };
                }
                &FORMATS
            }
            fn pixel_storage() -> &'static HashSet<api::PixelStorage> {
                lazy_static! {
                    static ref STORAGES: HashSet<api::PixelStorage> = {
                       <$t as IoTexel>::pixel_formats().iter().map(|f| f.storage()).collect()
                    };
                }
                &STORAGES
            }
        }
    };
}
impl_io_texel!(
    f32, R8Sint, R8Uint, R8Unorm, R16Sint, R16Uint, R16Unorm, R32Sint, R32Uint, R16f, R32f
);
impl_io_texel!(Float2, Rg16Sint, Rg16Uint, Rg16Unorm, Rg32Sint, Rg32Uint, Rg16f, Rg32f);
impl_io_texel!(
    Float4,
    Rgba8Sint,
    Rgba8Uint,
    Rgba8Unorm,
    Rgba16Sint,
    Rgba16Uint,
    Rgba16Unorm,
    Rgba32Sint,
    Rgba32Uint,
    Rgba16f,
    Rgba32f
);

impl_io_texel!(u16,);
impl_io_texel!(i16,);
impl_io_texel!(Ushort2,);
impl_io_texel!(Short2,);
impl_io_texel!(Ushort4,);
impl_io_texel!(Short4,);
impl_io_texel!(u32,);
impl_io_texel!(i32,);
impl_io_texel!(Uint2,);
impl_io_texel!(Int2,);
impl_io_texel!(Uint4,);
impl_io_texel!(Int4,);

// Types that is stored in a texture
pub trait StorageTexel {
    fn pixel_formats() -> &'static HashSet<api::PixelFormat>;
    fn pixel_storage() -> &'static HashSet<api::PixelStorage>;
}
macro_rules! impl_storage_texel {
    ($t:ty, $($formats:ident),*) => {
        impl StorageTexel for $t {
            fn pixel_formats() -> &'static HashSet<api::PixelFormat> {
                lazy_static!{
                    static ref FORMATS: HashSet<api::PixelFormat> = {
                        let mut set = HashSet::new();
                        $(
                            set.insert(api::PixelFormat::$formats);
                        )*
                        set
                    };
                }
                &FORMATS
            }
            fn pixel_storage() -> &'static HashSet<api::PixelStorage> {
                lazy_static! {
                    static ref STORAGES: HashSet<api::PixelStorage> = {
                       <$t as StorageTexel>::pixel_formats().iter().map(|f| f.storage()).collect()
                    };
                }
                &STORAGES
            }
        }
    };
}

impl_storage_texel!(u8, R8Uint, R8Unorm);
impl_storage_texel!(i8, R8Sint);
impl_storage_texel!(Ubyte3, Rgb8Uint, Rgb8Unorm);
impl_storage_texel!(Byte3, Rgb8Sint);
impl_storage_texel!(Ubyte4, Rgba8Uint, Rgba8Unorm);
impl_storage_texel!(Byte4, Rgba8Sint);

impl_storage_texel!(u16, R16Uint, R16Unorm);
impl_storage_texel!(i16, R16Sint);

impl_storage_texel!(Ushort2, Rg16Uint, Rg16Unorm);
impl_storage_texel!(Short2, Rg16Sint);

impl_storage_texel!(Ushort4, Rgba16Uint, Rgba16Unorm);
impl_storage_texel!(Short4, Rgba16Sint);

impl_storage_texel!(u32, R32Uint);
impl_storage_texel!(i32, R32Sint);

impl_storage_texel!(Uint2, Rg32Uint);
impl_storage_texel!(Int2, Rg32Sint);

impl_storage_texel!(Uint4, Rgba32Uint);
impl_storage_texel!(Int4, Rgba32Sint);

impl_storage_texel!(f16, R16f);
impl_storage_texel!(Half2, Rg16f);
impl_storage_texel!(Half4, Rgba16f);

impl_storage_texel!(f32, R32f);
impl_storage_texel!(Float2, Rg32f);
impl_storage_texel!(Float4, Rgba32f);

// `T` is the read out type of the texture, which is not necessarily the same as the storage type
// In fact, the texture can be stored in any format as long as it can be converted to `T`
pub struct Tex2d<T: IoTexel> {
    pub(crate) handle: Arc<TextureHandle>,
    pub(crate) marker: std::marker::PhantomData<T>,
}

// `T` is the read out type of the texture, which is not necessarily the same as the storage type
// In fact, the texture can be stored in any format as long as it can be converted to `T`
pub struct Tex3d<T: IoTexel> {
    pub(crate) handle: Arc<TextureHandle>,
    pub(crate) marker: std::marker::PhantomData<T>,
}
pub struct Tex2dView<T: IoTexel> {
    pub(crate) handle: Arc<TextureHandle>,
    pub(crate) marker: std::marker::PhantomData<T>,
    pub(crate) level: u32,
}
pub struct Tex3dView<T: IoTexel> {
    pub(crate) handle: Arc<TextureHandle>,
    pub(crate) marker: std::marker::PhantomData<T>,
    pub(crate) level: u32,
}
impl<T: IoTexel> Tex2d<T> {
    pub(crate) fn handle(&self) -> api::Texture {
        self.handle.handle
    }
}
impl<T: IoTexel> Tex3d<T> {
    pub(crate) fn handle(&self) -> api::Texture {
        self.handle.handle
    }
}
macro_rules! impl_tex_view {
    ($name:ident) => {
        impl<T: IoTexel> $name<T> {
            pub fn copy_to_async<'a, U: StorageTexel>(&'a self, data: &'a mut [U]) -> Command<'a> {
                assert_eq!(data.len(), self.texel_count() as usize);
                assert!(U::pixel_storage().contains(&self.handle.storage));
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
            pub fn copy_to<'a, U: StorageTexel>(&'a self, data: &'a mut [U]) {
                assert_eq!(data.len(), self.texel_count() as usize);

                submit_default_stream_and_sync(&self.handle.device, [self.copy_to_async(data)])
                    .unwrap();
            }
            pub fn copy_to_vec<'a, U: StorageTexel>(&'a self) -> Vec<U> {
                let mut data = Vec::with_capacity(self.texel_count() as usize);
                self.copy_to(&mut data);
                unsafe {
                    data.set_len(self.texel_count() as usize);
                }
                data
            }
            pub fn copy_from_async<'a, U: StorageTexel>(&'a self, data: &'a [U]) -> Command<'a> {
                assert_eq!(data.len(), self.texel_count() as usize);
                assert!(U::pixel_storage().contains(&self.handle.storage));
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
            pub fn copy_from<'a, U: StorageTexel>(&'a self, data: &[U]) {
                submit_default_stream_and_sync(&self.handle.device, [self.copy_from_async(data)])
                    .unwrap();
            }
            pub fn copy_to_buffer_async<'a, U: StorageTexel + Value>(
                &'a self,
                buffer_view: &'a BufferView<U>,
            ) -> Command<'a> {
                let mut rt = ResourceTracker::new();
                rt.add(self.handle.clone());
                rt.add(buffer_view.handle.clone());
                assert_eq!(buffer_view.len, self.texel_count() as usize);
                assert!(U::pixel_storage().contains(&self.handle.storage));
                Command {
                    inner: api::Command::TextureToBufferCopy(api::TextureToBufferCopyCommand {
                        texture: self.handle(),
                        storage: self.handle.storage,
                        texture_level: self.level,
                        texture_size: self.size(),
                        buffer: buffer_view.handle(),
                        buffer_offset: buffer_view.offset,
                    }),
                    resource_tracker: rt,
                    marker: std::marker::PhantomData,
                }
            }
            pub fn copy_to_buffer<'a, U: StorageTexel + Value>(
                &'a self,
                buffer_view: &BufferView<U>,
            ) {
                submit_default_stream_and_sync(
                    &self.handle.device,
                    [self.copy_to_buffer_async(buffer_view)],
                )
                .unwrap();
            }
            pub fn copy_from_buffer_async<'a, U: StorageTexel + Value>(
                &'a self,
                buffer_view: &BufferView<U>,
            ) -> Command<'a> {
                let mut rt = ResourceTracker::new();
                rt.add(self.handle.clone());
                rt.add(buffer_view.handle.clone());
                assert_eq!(buffer_view.len, self.texel_count() as usize);
                assert!(U::pixel_storage().contains(&self.handle.storage));
                Command {
                    inner: api::Command::BufferToTextureCopy(api::BufferToTextureCopyCommand {
                        texture: self.handle(),
                        storage: self.handle.storage,
                        texture_level: self.level,
                        texture_size: self.size(),
                        buffer: buffer_view.handle(),
                        buffer_offset: buffer_view.offset,
                    }),
                    resource_tracker: rt,
                    marker: std::marker::PhantomData,
                }
            }
            pub fn copy_from_buffer<'a, U: StorageTexel + Value>(
                &'a self,
                buffer_view: &BufferView<U>,
            ) {
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
impl<T: IoTexel> Tex2dView<T> {
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
impl<T: IoTexel> Tex3dView<T> {
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
