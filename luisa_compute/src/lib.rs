#![allow(unused_unsafe)]
use std::sync::Arc;

#[cfg(feature = "_cpp")]
pub(crate) use luisa_compute_sys as sys;
pub mod lang;
pub mod resource;
pub mod runtime;
pub use luisa_compute_ir::Gc;
pub use luisa_compute_backend as backend;
use luisa_compute_backend::Backend;
pub mod prelude {
    pub use crate::*;
    pub use lang::math::*;
    pub use lang::traits::*;
    pub use lang::*;
    pub use luisa_compute_derive::*;
    pub use runtime::*;
}
use libc;
use prelude::{Device, DeviceHandle};
pub fn init() {
    let gc_ctx = luisa_compute_ir::ir::luisa_compute_gc_create_context();
    luisa_compute_ir::ir::luisa_compute_gc_set_context(gc_ctx);
    let ctx = luisa_compute_ir::context::luisa_compute_ir_new_context();
    luisa_compute_ir::context::luisa_compute_ir_set_context(ctx);
}
pub fn create_cpu_device() -> backend::Result<Device> {
    let backend = backend::rust::RustBackend::new();
    let default_stream = backend.create_stream()?;
        Ok(Device {
            inner: Arc::new(DeviceHandle {
                backend,
                default_stream,
            }),
        })
}
pub(crate) fn _signal_handler(signal: libc::c_int) {
    if signal == libc::SIGABRT {
        panic!("std::abort() called inside LuisaCompute");
    }
}
#[macro_export]
macro_rules! catch_abort {
    ($stmts:expr) => {
        unsafe {
            let old = libc::signal(libc::SIGABRT, _signal_handler as libc::sighandler_t);
            let ret = $stmts;
            libc::signal(libc::SIGABRT, old);
            ret
        }
    };
}
