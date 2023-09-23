#![allow(unused_unsafe)]

extern crate self as luisa_compute;

use std::any::Any;
use std::backtrace::Backtrace;
use std::path::Path;
use std::sync::Arc;

pub mod lang;
pub mod printer;
pub mod resource;
pub mod rtx;
pub mod runtime;

pub use crate::lang::ops::{max, min};

pub mod prelude {
    pub use half::f16;

    pub use crate::lang::control_flow::{
        break_, continue_, for_range, return_, return_v, select, switch,
    };
    pub use crate::lang::functions::{block_size, dispatch_id, dispatch_size, set_block_size};
    pub use crate::lang::index::{IndexRead, IndexWrite};
    pub use crate::lang::ops::{
        AbsExpr, ActivateMaybeExpr, AddAssignExpr, AddExpr, ArrayNewExpr, BitAndAssignExpr,
        BitAndExpr, BitOrAssignExpr, BitOrExpr, BitXorAssignExpr, BitXorExpr, ClampExpr, CmpExpr,
        CrossExpr, DivAssignExpr, DivExpr, DotExpr, EqExpr, FloatArcTan2Expr, FloatCopySignExpr,
        FloatExpr, FloatLerpExpr, FloatLogExpr, FloatMulAddExpr, FloatPowfExpr, FloatPowiExpr,
        FloatSmoothStepExpr, FloatStepExpr, IntExpr, LazyBoolMaybeExpr, LoopMaybeExpr, MatExpr,
        MinMaxExpr, MulAssignExpr, MulExpr, NormExpr, OuterProductExpr, ReduceExpr, RemAssignExpr,
        RemExpr, SelectMaybeExpr, ShlAssignExpr, ShlExpr, ShrAssignExpr, ShrExpr, SubAssignExpr,
        SubExpr,
    };
    pub use crate::lang::types::vector::swizzle::*;
    pub use crate::lang::types::vector::VectorExprProxy;
    pub use crate::lang::types::{AsExpr, Expr, Value, Var};
    pub use crate::lang::Aggregate;
    pub use crate::resource::{IoTexel, StorageTexel, *};
    pub use crate::runtime::api::StreamTag;
    pub use crate::runtime::{
        create_static_callable, Command, Device, KernelBuildOptions, Scope, Stream,
    };
    pub use crate::{cpu_dbg, if_, lc_assert, lc_unreachable, loop_, while_, Context};

    pub use luisa_compute_derive::*;
    pub use luisa_compute_track::{track, tracked};
}

mod internal_prelude {
    pub(crate) use crate::lang::debug::{__env_need_backtrace, is_cpu_backend, CpuFn};
    pub(crate) use crate::lang::ir::ffi::*;
    pub(crate) use crate::lang::ir::{
        new_node, register_type, BasicBlock, Const, Func, Instruction, IrBuilder, Node,
        PhiIncoming, Pooled, Type, TypeOf, INVALID_REF,
    };
    pub(crate) use crate::lang::ops::Linear;
    pub(crate) use crate::lang::types::vector::alias::*;
    pub(crate) use crate::lang::types::vector::*;
    #[allow(unused_imports)]
    pub(crate) use crate::lang::{
        check_index_lt_usize, ir, CallFuncTrait, Recorder, __compose, __extract, __insert,
        __module_pools, need_runtime_check, FromNode, NodeLike, NodeRef, ToNode, __current_scope,
        __pop_scope, RECORDER,
    };
    pub(crate) use crate::prelude::*;
    pub(crate) use crate::runtime::{
        CallableArgEncoder, CallableParameter, CallableRet, KernelBuilder,
    };
    pub(crate) use crate::{
        get_backtrace, impl_simple_atomic_ref_proxy, impl_simple_expr_proxy, impl_simple_var_proxy,
        ResourceTracker,
    };
    pub(crate) use luisa_compute_backend::Backend;
    pub(crate) use std::marker::PhantomData;
}

pub use luisa_compute_derive::*;

use luisa_compute_api_types as api;
pub use {luisa_compute_backend as backend, luisa_compute_sys as sys};

use lazy_static::lazy_static;
use luisa_compute_backend::Backend;
use parking_lot::lock_api::RawMutex as RawMutexTrait;
use parking_lot::{Mutex, RawMutex};
use runtime::{Device, DeviceHandle, StreamHandle};
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
    /// if the current_exe() is in the same directory as libluisa-*, then
    /// passing current_exe() is enough
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
