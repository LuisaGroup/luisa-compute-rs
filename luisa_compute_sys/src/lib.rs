#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
pub mod binding;
pub use binding::*;
pub mod cpp_proxy_backend;
// pub use cpp_proxy_backend::init_cpp;