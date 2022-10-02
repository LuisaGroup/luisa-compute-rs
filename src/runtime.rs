use std::ffi::CString;

use serde_json::{Value, json};

use crate::*;
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
        unsafe {
            let exe_path = CString::new(exe_path.to_str().unwrap()).unwrap();
            let ctx = sys::luisa_compute_context_create(exe_path.as_ptr());
            Self { inner: ctx }
        }
    }
    pub fn create_device(&self, device: &str, properties: Value) -> Device {
        unsafe {
            let device = CString::new(device).unwrap();
            let properties = CString::new(properties.to_string()).unwrap();
            let device =
                sys::luisa_compute_device_create(self.inner, device.as_ptr(), properties.as_ptr());
            Device { inner: device }
        }
    }
}
impl Drop for Context{
    fn drop(&mut self) {
        unsafe {
            sys::luisa_compute_context_destroy(self.inner);
        }
    }
}
impl Clone for Device {
    fn clone(&self) -> Self {
        unsafe {
            sys::luisa_compute_device_retain(self.inner);
        }
        Self { inner: self.inner }
    }
}
impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            sys::luisa_compute_device_release(self.inner);
        }
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
