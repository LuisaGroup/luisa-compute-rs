use crate::*;
use lang::Value;
use runtime::*;
pub use sys::LCPixelFormat as PixelFormat;
pub use sys::LCPixelStorage as PixelStorage;
pub struct Buffer<T: Value> {
    pub(crate) device: Device,
    pub(crate) handle: sys::LCBuffer,
    pub(crate) _marker: std::marker::PhantomData<T>,
}
impl<T: Value> Buffer<T> {
    pub(crate) fn handle(&self) -> sys::LCBuffer {
        self.handle
    }
}
impl<T: Value> Drop for Buffer<T> {
    fn drop(&mut self) {
        catch_abort! {{
            sys::luisa_compute_buffer_destroy(self.device.inner, self.handle);
        }}
    }
}
pub struct BindlessBuffer<T: Value> {
    pub(crate) device: Device,
    pub(crate) handle: sys::LCBindlessArray,
    pub(crate) _marker: std::marker::PhantomData<T>,
}
pub struct Texture {
    pub(crate) device: Device,
    pub(crate) handle: sys::LCTexture,
    pub(crate) format: PixelFormat,
}
pub struct BindlessTexture {
    pub(crate) device: Device,
    pub(crate) handle: sys::LCTexture,
}
impl Texture {
    pub(crate) fn handle(&self) -> sys::LCTexture {
        self.handle
    }
}
impl Drop for Texture {
    fn drop(&mut self) {
        catch_abort! {{
            sys::luisa_compute_texture_destroy(self.device.inner, self.handle);
        }}
    }
}
