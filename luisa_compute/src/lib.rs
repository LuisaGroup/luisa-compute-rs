#![allow(unused_unsafe)]
#[cfg(feature = "_cpp")]
pub(crate) use luisa_compute_sys as sys;
pub mod backend;
pub mod lang;
pub mod resource;
pub mod runtime;
pub use luisa_compute_ir::Gc;
pub mod prelude {
    pub use crate::*;
    pub use lang::math::*;
    // pub use lang::math_impl::*;
    pub use lang::traits::*;
    pub use lang::traits_impl::*;
    pub use lang::*;
    pub use luisa_compute_derive::*;
    pub use runtime::*;
}
use libc;
pub fn init() {
    let gc_ctx = luisa_compute_ir::ir::luisa_compute_gc_create_context();
    luisa_compute_ir::ir::luisa_compute_gc_set_context(gc_ctx);
    let ctx = luisa_compute_ir::context::luisa_compute_ir_new_context();
    luisa_compute_ir::context::luisa_compute_ir_set_context(ctx);
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
