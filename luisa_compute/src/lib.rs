#![allow(unused_unsafe)]
pub(crate) use luisa_compute_sys as sys;
pub mod backend;
pub mod lang;
pub mod resource;
pub mod runtime;
pub use luisa_compute_ir::Gc;
pub mod prelude {
    pub use crate::*;
    pub use luisa_compute_derive::*;
    pub use lang::traits::*;
    pub use lang::traits_impl::*;
    pub use lang::math::*;
    pub use lang::math_impl::*;
    pub use lang::*;
}
use libc;
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
