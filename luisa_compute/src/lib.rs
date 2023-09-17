#![allow(unused_unsafe)]

use std::any::Any;
use std::backtrace::Backtrace;
use std::path::Path;
use std::sync::Arc;

pub mod lang;
pub mod resource;
// pub mod rtx;
pub mod runtime;

pub use half::f16;
// use luisa_compute_api_types as api;
pub use luisa_compute_backend as backend;

pub mod prelude {
    pub use crate::lang::poly::PolymorphicImpl;
    pub use crate::lang::traits::{
        CommonVarOp, FloatVarTrait, IntVarTrait, VarCmp, VarCmpEq, VarTrait,
    };
    pub use crate::lang::{
        Aggregate, ExprProxy, FromNode, IndexRead, IndexWrite, KernelBuildFn, KernelParameter,
        KernelSignature, Mask, Value, VarProxy,
    };
    // pub use crate::lang::{
    //     __compose, __cpu_dbg, __current_scope, __env_need_backtrace, __extract, __insert,
    //     __module_pools, __new_user_node, __pop_scope,
    // };
    pub use crate::lang::math::*;
    pub use crate::lang::swizzle::*;
    pub use crate::lang::{
        dispatch_id, dispatch_size, Bool, F16, F32, F64, I16, I32, I64, U16, U32, U64,
    };
    pub use crate::resource::{IoTexel, StorageTexel};
    pub use crate::runtime::KernelArg;
    pub use crate::{cpu_dbg, if_, lc_assert, lc_unreachable, loop_, while_};
    pub use luisa_compute_ir::TypeOf;
    pub use luisa_compute_track::track;
}

pub use api::{
    AccelBuildModificationFlags, AccelBuildRequest, AccelOption, AccelUsageHint, MeshType,
    PixelFormat, PixelStorage,
};
pub use lang::math::*;
pub use lang::poly::*;
pub use lang::traits::*;
pub use lang::{math, poly, *};
pub use luisa_compute_derive::*;
pub use luisa_compute_ir::ir::UserNodeData;
pub use resource::*;
pub use runtime::*;
pub use {glam, log, luisa_compute_derive as derive};

pub mod macros {
    pub use crate::{
        cpu_dbg, if_, impl_new_poly_array, impl_polymorphic, lc_assert, lc_dbg, lc_debug, lc_error,
        lc_info, lc_log, lc_unreachable, lc_warn, loop_, struct_, var, while_,
    };
}

use lazy_static::lazy_static;
use luisa_compute_backend::Backend;
pub use luisa_compute_sys as sys;
use parking_lot::lock_api::RawMutex as RawMutexTrait;
use parking_lot::{Mutex, RawMutex};
pub use runtime::{Device, Scope, Stream};
use std::collections::HashMap;
use std::sync::Weak;

pub struct Context {
    inner: Arc<backend::Context>,
}

pub fn init_logger() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .init();
}
pub fn init_logger_verbose() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
        .format_timestamp_secs()
        .init();
}
lazy_static! {
    static ref CTX_CACHE: Mutex<HashMap<String, Weak<backend::Context>>> =
        Mutex::new(HashMap::new());
}
impl Context {
    /// path to libluisa-*
    /// if the current_exe() is in the same directory as libluisa-*, then passing current_exe() is enough
    pub fn new(lib_path: impl AsRef<Path>) -> Self {
        let mut lib_path = lib_path.as_ref().to_path_buf();
        lib_path = lib_path.canonicalize().unwrap();
        if lib_path.is_file() {
            lib_path = lib_path.parent().unwrap().to_path_buf();
        }
        let inner = {
            let mut cache = CTX_CACHE.lock();
            if let Some(ctx) = cache.get(lib_path.to_str().unwrap()) {
                if let Some(ctx) = ctx.upgrade() {
                    return Self { inner: ctx.clone() };
                }
            }
            let ctx = Arc::new(backend::Context::new(lib_path.clone()));
            cache.insert(lib_path.to_str().unwrap().to_string(), Arc::downgrade(&ctx));
            ctx
        };
        Self { inner }
    }
    #[inline]
    pub fn create_cpu_device(&self) -> Device {
        self.create_device("cpu")
    }
    pub fn create_device(&self, device: &str) -> Device {
        self.create_device_with_config(device, serde_json::json!({}))
    }
    pub fn create_device_with_config(&self, device: &str, config: serde_json::Value) -> Device {
        let backend = self.inner.create_device(device, config);
        let default_stream = backend.create_stream(api::StreamTag::Graphics);
        Device {
            inner: Arc::new_cyclic(|weak| DeviceHandle {
                backend,
                default_stream: Some(Arc::new(StreamHandle::Default {
                    handle: api::Stream(default_stream.handle),
                    native_handle: default_stream.native_handle,
                    device: weak.clone(),
                    mutex: RawMutex::INIT,
                })),
                ctx: self.inner.clone(),
            }),
        }
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

pub(crate) fn get_backtrace() -> Backtrace {
    Backtrace::force_capture()
}
