use std::{ffi::CString, path::PathBuf};

use crate::resource::*;
use crate::*;
use serde_json::{json, Value};
pub struct Device {
    pub(crate) inner: sys::LCDevice,
}
unsafe impl Send for Device {}
unsafe impl Sync for Device {}
pub struct Context {
    pub(crate) inner: sys::LCContext,
}
unsafe impl Send for Context {}
unsafe impl Sync for Context {}
impl Context {
    pub fn new() -> Self {
        let exe_path = std::env::current_exe().unwrap();
        catch_abort! {{
            let exe_path = CString::new(exe_path.to_str().unwrap()).unwrap();
            let ctx = sys::luisa_compute_context_create(exe_path.as_ptr());
            Self { inner: ctx }
        }}
    }
    pub fn create_device(&self, device: &str, properties: Value) -> Device {
        catch_abort! {{
            let device = CString::new(device).unwrap();
            let properties = CString::new(properties.to_string()).unwrap();
            let device =
                sys::luisa_compute_device_create(self.inner, device.as_ptr(), properties.as_ptr());
            Device { inner: device }
        }}
    }
    pub fn runtime_dir(&self) -> PathBuf {
        catch_abort! {{
            let path = sys::luisa_compute_context_runtime_directory(self.inner);
            let path = std::ffi::CStr::from_ptr(path).to_str().unwrap().to_string();
            PathBuf::from(path)
        }}
    }
    pub fn cache_dir(&self) -> PathBuf {
        catch_abort! {{
            let path = sys::luisa_compute_context_cache_directory(self.inner);
            let path = std::ffi::CStr::from_ptr(path).to_str().unwrap().to_string();
            PathBuf::from(path)
        }}
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        catch_abort! {{
            sys::luisa_compute_context_destroy(self.inner);
        }}
    }
}
impl Clone for Device {
    fn clone(&self) -> Self {
        catch_abort! {{
            sys::luisa_compute_device_retain(self.inner);
        }}
        Self { inner: self.inner }
    }
}
impl Drop for Device {
    fn drop(&mut self) {
        catch_abort! {{
            sys::luisa_compute_device_release(self.inner);
        }}
    }
}
impl Device {
    pub fn create_buffer<T: Copy>(&self, count: usize) -> Buffer<T> {
        catch_abort! {{
            let buffer = sys::luisa_compute_buffer_create(
                self.inner,
                std::mem::size_of::<T>() as u64 * count as u64,
            );
            Buffer {
                device: self.clone(),
                handle: buffer,
                _marker: std::marker::PhantomData {},
            }
        }}
    }
    pub fn create_texture(
        &self,
        format: PixelFormat,
        dim: u32,
        width: u32,
        height: u32,
        depth: u32,
        mips: u32,
    ) -> Texture {
        catch_abort! {{
            let texture = sys::luisa_compute_texture_create(
                self.inner,
                format.0 as u32,
                dim,
                width,
                height,
                depth,
                mips,
            );
            Texture {
                device: self.clone(),
                handle: texture,
                format,
            }
        }}
    }
    pub fn create_stream(&self) -> Stream {
        catch_abort! {{
            let stream = sys::luisa_compute_stream_create(self.inner);
            Stream {
                device: self.clone(),
                handle: stream,
            }
        }}
    }
}
pub struct Stream {
    pub(crate) device: Device,
    pub(crate) handle: sys::LCStream,
}
impl Drop for Stream {
    fn drop(&mut self) {
        catch_abort! {{
            sys::luisa_compute_stream_destroy(self.device.inner, self.handle);
        }}
    }
}
impl Stream {
    pub fn synchronize(&self) {
        catch_abort! {{
            sys::luisa_compute_stream_synchronize(self.device.inner, self.handle);
        }}
    }
}
pub struct CommandList {
    pub(crate) handle: sys::LCCommandList,
}
pub struct Command {
    pub(crate) handle: sys::LCCommand,
}
impl Command {
    pub fn upload_buffer<T: Copy>(buffer: &Buffer<T>, offset: usize, data: &[T]) -> Self {
        catch_abort! {{
            let count = data.len();
            let cmd = sys::luisa_compute_command_upload_buffer(
                buffer.handle(),
                offset as u64 * std::mem::size_of::<T>() as u64,
                count as u64 * std::mem::size_of::<T>() as u64,
                data.as_ptr() as *const std::ffi::c_void,
            );
            Self { handle: cmd }
        }}
    }
    pub fn download_buffer<T: Copy>(buffer: &Buffer<T>, offset: usize, data: &mut [T]) -> Self {
        catch_abort! {{
            let count = data.len();
            let cmd = sys::luisa_compute_command_download_buffer(
                buffer.handle(),
                offset as u64 * std::mem::size_of::<T>() as u64,
                count as u64 * std::mem::size_of::<T>() as u64,
                data.as_mut_ptr() as *mut std::ffi::c_void,
            );
            Self { handle: cmd }
        }}
    }
    pub fn copy_buffer_to_buffer<T: Copy>(
        src: &Buffer<T>,
        src_offset: usize,
        dst: &Buffer<T>,
        dst_offset: usize,
        count: usize,
    ) -> Self {
        catch_abort! {{
            let cmd = sys::luisa_compute_command_copy_buffer_to_buffer(
                src.handle(),
                src_offset as u64 * std::mem::size_of::<T>() as u64,
                dst.handle(),
                dst_offset as u64 * std::mem::size_of::<T>() as u64,
                std::mem::size_of::<T>() as u64 * count as u64,
            );
            Self { handle: cmd }
        }}
    }
    pub fn copy_buffer_to_texture<T: Copy>(
        buffer: &Buffer<T>,
        buffer_offset: usize,
        texture: &Texture,
        storage: PixelStorage,
        level: u32,
        size: [u32; 3],
    ) -> Self {
        catch_abort! {{
            let cmd = sys::luisa_compute_command_copy_buffer_to_texture(
                buffer.handle(),
                buffer_offset as u64 * std::mem::size_of::<T>() as u64,
                texture.handle(),
                storage,
                level,
                sys::lc_uint3 {
                    x: size[0],
                    y: size[1],
                    z: size[2],
                },
            );
            Self { handle: cmd }
        }}
    }
    pub fn copy_texture_to_buffer<T: Copy>(
        buffer: &Buffer<T>,
        buffer_offset: usize,
        texture: &Texture,
        storage: PixelStorage,
        level: u32,
        size: [u32; 3],
    ) -> Self {
        catch_abort! {{
            let cmd = sys::luisa_compute_command_copy_texture_to_buffer(
                buffer.handle(),
                buffer_offset as u64 * std::mem::size_of::<T>() as u64,
                texture.handle(),
                storage,
                level,
                sys::lc_uint3 {
                    x: size[0],
                    y: size[1],
                    z: size[2],
                },
            );
            Self { handle: cmd }
        }}
    }
    pub fn copy_texture_to_texture(
        src: &Texture,
        src_level: u32,
        dst: &Texture,
        dst_level: u32,
        storage: PixelStorage,
        size: [u32; 3],
    ) -> Self {
        catch_abort! {{
            let cmd = sys::luisa_compute_command_copy_texture_to_texture(
                src.handle(),
                src_level,
                dst.handle(),
                dst_level,
                storage,
                sys::lc_uint3 {
                    x: size[0],
                    y: size[1],
                    z: size[2],
                },
            );
            Self { handle: cmd }
        }}
    }
}
impl CommandList {
    pub fn new() -> Self {
        catch_abort! {{
            let list = sys::luisa_compute_command_list_create();
            Self { handle: list }
        }}
    }
    pub fn append(&self, command: Command) {
        catch_abort! {{
            sys::luisa_compute_command_list_append(self.handle, command.handle);
        }}
    }
    pub fn empty(&self) -> bool {
        catch_abort! {{ sys::luisa_compute_command_list_empty(self.handle) != 0 }}
    }
    pub fn clear(&self) {
        catch_abort! {{
            sys::luisa_compute_command_list_clear(self.handle);
        }}
    }
}
impl Drop for CommandList {
    fn drop(&mut self) {
        catch_abort! {{
            sys::luisa_compute_command_list_destroy(self.handle);
        }}
    }
}
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_device() {
        let ctx = super::Context::new();
        let device = ctx.create_device("cuda", json!({}));
    }
}
