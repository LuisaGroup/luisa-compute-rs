use crate::*;
use runtime::*;
pub use sys::LCPixelFormat as PixelFormat;
pub use sys::LCPixelStorage as PixelStorage;
pub struct Buffer<T: Copy> {
    pub(crate) device: Device,
    pub(crate) handle: sys::LCBuffer,
    pub(crate) _marker: std::marker::PhantomData<T>,
}
impl <T:Copy>Buffer<T>{
    pub(crate) fn handle(&self) -> sys::LCBuffer {
        self.handle
    }
}
impl<T: Copy> Drop for Buffer<T> {
    fn drop(&mut self) {
        catch_abort! {{
            sys::luisa_compute_buffer_destroy(self.device.inner, self.handle);
        }}
    }
}
pub struct Texture {
    pub(crate) device: Device,
    pub(crate) handle: sys::LCTexture,
    pub(crate) format: PixelFormat,
}
impl Texture{
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
