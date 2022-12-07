#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#[cfg(feature = "_cpp")]
pub mod binding;
#[cfg(feature = "_cpp")]
pub use binding::*;