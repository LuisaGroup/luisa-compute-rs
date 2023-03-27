#![allow(unused_unsafe)]
use std::path::{Path, PathBuf};
use std::{any::Any, sync::Arc};

pub mod lang;
pub mod resource;
pub mod rtx;
pub mod runtime;
pub use half::f16;
use luisa_compute_api_types as api;
pub use luisa_compute_backend as backend;
use luisa_compute_backend::Backend;
pub mod prelude {
    pub use crate::lang::poly::PolymorphicImpl;
    pub use crate::lang::traits::VarTrait;
    pub use crate::lang::traits::{CommonVarOp, FloatVarTrait, IntVarTrait, VarCmp, VarCmpEq};
    pub use crate::lang::{
        Aggregate, ExprProxy, FromNode, KernelBuildFn, KernelParameter, KernelSignature, Value,
        VarProxy, _Mask,
    };
    pub use crate::lang::{
        __compose, __cpu_dbg, __current_scope, __env_need_backtrace, __extract, __insert,
        __module_pools, __new_user_node, __pop_scope,
    };
    pub use crate::resource::{IoTexel, StorageTexel};
    pub use crate::runtime::KernelArg;
    pub use luisa_compute_ir::TypeOf;
}
pub use api::{
    AccelBuildModificationFlags, AccelBuildRequest, AccelOption, AccelUsageHint, MeshType,
    PixelFormat, PixelStorage,
};
pub use glam;
pub use lang::math;
pub use lang::math::*;
pub use lang::poly;
pub use lang::poly::*;
pub use lang::traits::*;
pub use lang::*;
pub use luisa_compute_derive as derive;
pub use luisa_compute_derive::*;
pub use luisa_compute_ir::ir::UserNodeData;
pub use resource::*;
pub use runtime::*;
pub mod macros {
    pub use crate::{cpu_dbg, if_, impl_polymorphic, var, while_};
}

pub use luisa_compute_sys as sys;
pub use runtime::{CommandList, Device, Stream};
use std::sync::Once;
pub struct Context {
    inner: backend::Context,
}
pub fn init_logger() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .init();
}

impl Context {
    // path to libluisa-*
    // if the current_exe() is in the same directory as libluisa-*, then passing current_exe() is enough
    pub fn new(lib_path: impl AsRef<Path>) -> Self {
        let mut lib_path = lib_path.as_ref().to_path_buf();
        if lib_path.is_file() {
            lib_path = lib_path.parent().unwrap().to_path_buf();
        }
        Self {
            inner: backend::Context::new(lib_path),
        }
    }
    #[inline]
    pub fn create_cpu_device(&self) -> backend::Result<Device> {
        self.create_device("cpu")
    }

    pub fn create_device(&self, device: &str) -> backend::Result<Device> {
        use luisa_compute_backend::SwapChainForCpuContext;
        let backend = self.inner.create_device(device)?;
        let default_stream = api::Stream(backend.create_stream(api::StreamTag::Graphics)?.handle);
        Ok(Device {
            inner: Arc::new(DeviceHandle {
                backend,
                default_stream,
            }),
        })
    }
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
