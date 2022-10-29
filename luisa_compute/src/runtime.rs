use crate::*;
use crate::{lang::Value, resource::*};
pub use luisa_compute_api_types as api;
use std::any::Any;
use std::sync::Arc;
use std::{ffi::CString, path::PathBuf};
#[derive(Clone)]
pub struct Device {
    pub(crate) inner: Arc<DeviceHandle>,
}
pub(crate) struct DeviceHandle {
    pub(crate) handle: sys::LCDevice,
    pub(crate) default_stream: sys::LCStream,
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
    pub fn create_device(&self, device: &str, properties: serde_json::Value) -> Device {
        catch_abort! {{
            let device = CString::new(device).unwrap();
            let properties = CString::new(properties.to_string()).unwrap();
            let device =
                sys::luisa_compute_device_create(self.inner, device.as_ptr(), properties.as_ptr());
            let default_stream = sys::luisa_compute_stream_create(device);
            Device {
                inner: Arc::new(DeviceHandle{
                    handle:device,
                    default_stream
                })
        }
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

impl Drop for DeviceHandle {
    fn drop(&mut self) {
        catch_abort! {{
            sys::luisa_compute_stream_destroy(self.handle, self.default_stream);
            sys::luisa_compute_device_release(self.handle);
        }}
    }
}
impl Device {
    pub fn handle(&self) -> sys::LCDevice {
        self.inner.handle
    }
    pub fn create_buffer<T: Value>(&self, count: usize) -> Buffer<T> {
        catch_abort! {{
            let buffer = sys::luisa_compute_buffer_create(
                self.handle(),
                std::mem::size_of::<T>() as u64 * count as u64,
            );
            Buffer {
                device: self.clone(),
                handle: Arc::new(BufferHandle{
                    device: self.clone(),
                    handle: buffer,
                }),
                _marker: std::marker::PhantomData {},
                len: count,
            }
        }}
    }
    pub fn create_bindless_buffer(&self, slots: usize) -> BindlessArray {
        catch_abort! {{
            let buffer = sys::luisa_compute_bindless_array_create(
                self.handle(),
                slots as u64,
            );
            BindlessArray {
                device: self.clone(),
                handle: Arc::new(BindlessArrayHandle{
                    device: self.clone(),
                    handle: buffer,
                }),
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
                self.handle(),
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
    pub fn default_stream(&self) -> Stream {
        Stream {
            device: self.clone(),
            handle: Arc::new(StreamHandle::Default(
                self.inner.handle,
                self.inner.default_stream,
            )),
        }
    }
    pub fn create_stream(&self) -> Stream {
        catch_abort! {{
            let stream = sys::luisa_compute_stream_create(self.handle());
            Stream {
                device: self.clone(),
                handle: Arc::new(StreamHandle::NonDefault{ device: self.clone(), handle:stream }),
            }
        }}
    }
}
pub(crate) enum StreamHandle {
    Default(sys::LCDevice, sys::LCStream),
    NonDefault {
        device: Device,
        handle: sys::LCStream,
    },
}
pub struct Stream {
    pub(crate) device: Device,
    pub(crate) handle: Arc<StreamHandle>,
}
impl StreamHandle {
    pub(crate) fn device_handle(&self) -> sys::LCDevice {
        match self {
            StreamHandle::Default(device, _) => *device,
            StreamHandle::NonDefault { device, .. } => device.handle(),
        }
    }
    pub(crate) fn handle(&self) -> sys::LCStream {
        match self {
            StreamHandle::Default(_, stream) => *stream,
            StreamHandle::NonDefault { handle, .. } => *handle,
        }
    }
}
impl Drop for StreamHandle {
    fn drop(&mut self) {
        catch_abort! {{
            match self{
                StreamHandle::Default(_, _) => {},
                StreamHandle::NonDefault{device, handle} => {
                    sys::luisa_compute_stream_destroy(device.handle(), *handle);
                }
            }
        }}
    }
}
impl Stream {
    pub fn handle(&self) -> sys::LCStream {
        match &*self.handle {
            StreamHandle::Default(_, handle) => *handle,
            StreamHandle::NonDefault { handle, .. } => *handle,
        }
    }
    pub fn synchronize(&self) {
        catch_abort! {{
            sys::luisa_compute_stream_synchronize(self.device.handle(), self.handle());
        }}
    }
    pub fn command_buffer<'a>(&self) -> CommandBuffer<'a> {
        CommandBuffer::<'a> {
            marker: std::marker::PhantomData {},
            stream: self.handle.clone(),
            commands: Vec::new(),
        }
    }
}
pub struct CommandBuffer<'a> {
    stream: Arc<StreamHandle>,
    marker: std::marker::PhantomData<&'a ()>,
    commands: Vec<Command<'a>>,
}
impl<'a> CommandBuffer<'a> {
    pub fn extend<I: IntoIterator<Item = Command<'a>>>(&mut self, commands: I) {
        self.commands.extend(commands);
    }
    pub fn push(&mut self, command: Command<'a>) {
        self.commands.push(command);
    }
    pub fn commit(self) {
        catch_abort!({
            {
                let command_list = sys::LCCommandList {
                    commands: self.commands.as_ptr() as *const sys::LCCommand,
                    commands_count: self.commands.len() as u64,
                };
                sys::luisa_compute_stream_dispatch(
                    self.stream.device_handle(),
                    self.stream.handle(),
                    command_list,
                );
            }
        })
    }
}

pub fn submit_default_stream_and_sync<'a, I: IntoIterator<Item = Command<'a>>>(
    device: &Device,
    commands: I,
) {
    let default_stream = device.default_stream();
    let mut cmd_buffer = default_stream.command_buffer();

    cmd_buffer.extend(commands);

    cmd_buffer.commit();
    default_stream.synchronize();
}
pub struct Command<'a> {
    pub(crate) inner: api::Command,
    pub(crate) marker: std::marker::PhantomData<&'a ()>,
    pub(crate) resource_tracker: Vec<Box<dyn Any>>,
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_layout() {
        assert_eq!(
            std::mem::size_of::<api::Command>(),
            std::mem::size_of::<sys::LCCommand>()
        );
    }
    #[test]
    fn test_device() {
        let ctx = super::Context::new();
        let device = ctx.create_device("cuda", serde_json::json!({}));
    }
}
