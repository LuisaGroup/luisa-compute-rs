#![allow(unused_unsafe)]
use std::sync::Arc;

#[cfg(feature = "_cpp")]
pub(crate) use luisa_compute_sys as sys;
pub mod lang;
pub mod resource;
pub mod runtime;
pub use luisa_compute_backend as backend;
use luisa_compute_backend::Backend;
pub use luisa_compute_ir::Gc;
pub mod prelude {
    pub use crate::*;
    pub use glam;
    pub use lang::math::*;
    pub use lang::traits::*;
    pub use lang::*;
    pub use luisa_compute_derive::*;
    pub use resource::*;
    pub use runtime::*;
}
use prelude::{Device, DeviceHandle};
use std::sync::Once;
static INIT: Once = Once::new();
pub fn init() {
    INIT.call_once(|| unsafe {
        let gc_ctx = luisa_compute_ir::ir::luisa_compute_gc_create_context();
        luisa_compute_ir::ir::luisa_compute_gc_set_context(gc_ctx);
        let ctx = luisa_compute_ir::context::luisa_compute_ir_new_context();
        luisa_compute_ir::context::luisa_compute_ir_set_context(ctx);
    });
}
pub fn init_logger() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .init();
}

pub fn create_cpu_device() -> backend::Result<Device> {
    create_device("cpu")
}

pub fn create_device(device: &str) -> backend::Result<Device> {
    let backend: Arc<dyn Backend> = match device {
        "cpu" => backend::rust::RustBackend::new(),
        "cuda" => {
            #[cfg(feature = "_cpp")]
            {
                sys::cpp_proxy_backend::CppProxyBackend::new(device)
            }
            #[cfg(not(feature = "_cpp"))]
            {
                panic!("{} backend is not enabled", device)
            }
        }
        _ => panic!("unsupported device: {}", device),
    };
    let default_stream = backend.create_stream()?;
    Ok(Device {
        inner: Arc::new(DeviceHandle {
            backend,
            default_stream,
        }),
    })
}
