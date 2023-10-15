use std::ffi::CString;
use std::fmt::Debug;

use crate::internal_prelude::*;

use super::{with_recorder, recording_started};

#[macro_export]
macro_rules! cpu_dbg {
    ($arg:expr) => {{
        $crate::lang::debug::__cpu_dbg($arg, file!(), line!())
    }};
}
#[macro_export]
macro_rules! lc_dbg {
    ($arg:expr) => {{
        $crate::lang::debug::__cpu_dbg($arg, file!(), line!())
    }};
}
#[macro_export]
macro_rules! lc_unreachable {
    () => {
        $crate::lang::debug::__unreachable(file!(), line!(), column!())
    };
}
#[macro_export]
macro_rules! lc_assert {
    ($arg:expr) => {
        $crate::lang::debug::__assert($arg, stringify!($arg), file!(), line!(), column!())
    };
    ($arg:expr, $msg:expr) => {
        $crate::lang::debug::__assert($arg, $msg, file!(), line!(), column!())
    };
}
pub fn __cpu_dbg<V: Value + Debug>(arg: Expr<V>, file: &'static str, line: u32) {
    if !is_cpu_backend() {
        return;
    }
    let f = CpuFn::new(move |x: &mut V| {
        println!("[{}:{}] {:?}", file, line, x);
    });
    let _ = f.call(arg);
}

pub fn is_cpu_backend() -> bool {
    with_recorder(|r| {
        if r.device.is_none() {
            return false;
        }
        r.device
            .as_ref()
            .unwrap()
            .upgrade()
            .unwrap()
            .inner
            .query("device_name")
            .map(|s| s == "cpu")
            .unwrap_or(false)
    })
}

pub fn __env_need_backtrace() -> bool {
    match std::env::var("LUISA_BACKTRACE") {
        Ok(s) => s == "1" || s == "ON",
        Err(_) => false,
    }
}

pub fn __unreachable(file: &str, line: u32, col: u32) {
    let path = std::path::Path::new(file);
    let pretty_filename: String;
    if path.exists() {
        pretty_filename = std::fs::canonicalize(path)
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
    } else {
        pretty_filename = file.to_string();
    }
    let msg = if __env_need_backtrace() {
        let backtrace = get_backtrace();
        format!(
            "unreachable code at {}:{}:{} \nbacktrace: {}",
            pretty_filename, line, col, backtrace
        )
    } else {
        format!(
            "unreachable code at {}:{}:{} \n",
            pretty_filename, line, col
        )
    };
    __current_scope(|b| {
        b.call(
            Func::Unreachable(CBoxedSlice::new(
                CString::new(msg).unwrap().into_bytes_with_nul(),
            )),
            &[],
            Type::void(),
        );
    });
}

pub fn __assert(cond: impl Into<Expr<bool>>, msg: &str, file: &str, line: u32, col: u32) {
    let cond = cond.into();
    let path = std::path::Path::new(file);
    let pretty_filename: String;
    if path.exists() {
        pretty_filename = std::fs::canonicalize(path)
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
    } else {
        pretty_filename = file.to_string();
    }
    let msg = if __env_need_backtrace() {
        let backtrace = get_backtrace();
        format!(
            "assertion failed: {} at {}:{}:{} \nbacktrace:\n{}",
            msg, pretty_filename, line, col, backtrace
        )
    } else {
        format!(
            "assertion failed: {} at {}:{}:{} \n",
            msg, pretty_filename, line, col
        )
    };
    let cond = cond.node().get();
    __current_scope(|b| {
        b.call(
            Func::Assert(CBoxedSlice::new(
                CString::new(msg).unwrap().into_bytes_with_nul(),
            )),
            &[cond],
            Type::void(),
        );
    });
}

pub fn comment(msg: &str) {
    if !recording_started() {
        return;
    }
    __current_scope(|b| {
        b.comment(CBoxedSlice::new(
            CString::new(msg).unwrap().into_bytes_with_nul(),
        ))
    });
}

#[macro_export]
macro_rules! lc_comment_lineno {
    () => {
        $crate::lang::debug::comment(&format!("{}:{}:{}", file!(), line!(), column!()))
    };
    ($msg:literal) => {
        $crate::lang::debug::comment(&format!(
            "`{}` at {}:{}:{}",
            $msg,
            file!(),
            line!(),
            column!()
        ))
    };
}
