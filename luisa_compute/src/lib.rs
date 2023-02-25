#![allow(unused_unsafe)]
use std::{any::Any, sync::Arc};

pub mod lang;
pub mod resource;
pub mod rtx;
pub mod runtime;
pub use luisa_compute_backend as backend;
use luisa_compute_backend::Backend;
pub mod prelude {
    pub use crate::*;
    pub use api::{AccelBuildModificationFlags, AccelBuildRequest, AccelUsageHint, MeshType};
    pub use glam;
    pub use lang::math::*;
    pub use lang::poly::*;
    pub use lang::traits::*;
    pub use lang::*;
    pub use luisa_compute_derive::*;
    pub use luisa_compute_ir::ir::UserNodeData;
    pub use resource::*;
    pub use rtx::*;
    pub use runtime::*;
}
pub use luisa_compute_sys as sys;
use prelude::{Device, DeviceHandle};
use std::sync::Once;
static INIT: Once = Once::new();
pub fn init() {
    INIT.call_once(|| {
        // do nothing?
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
            #[cfg(feature = "cuda")]
            {
                sys::cpp_proxy_backend::CppProxyBackend::new(device)
            }
            #[cfg(not(feature = "cuda"))]
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
pub struct ResourceTracker {
    resources: Vec<Arc<dyn Any>>,
}
impl ResourceTracker {
    pub fn add<T: Any>(&mut self, ptr: Arc<T>) -> &mut Self {
        self.resources.push(ptr);
        self
    }
    pub fn add_any(&mut self, ptr: Arc<dyn Any>) -> &mut Self {
        self.resources.push(ptr);
        self
    }
    pub fn new() -> Self {
        Self { resources: vec![] }
    }
}
unsafe impl Send for ResourceTracker {}
unsafe impl Sync for ResourceTracker {}
