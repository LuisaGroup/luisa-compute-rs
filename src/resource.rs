use crate::*;
use runtime::*;
pub use sys::LCPixelFormat as PixelFormat;
pub use sys::LCPixelStorage as PixelStorage;
pub struct Buffer<T: Copy> {
    device: Device,
    handle: sys::LCBuffer,
    _marker: std::marker::PhantomData<T>,
}

pub struct Texture {
    device: Device,
    handle: sys::LCTexture,
    format: PixelFormat,
    storage: PixelStorage,
}
