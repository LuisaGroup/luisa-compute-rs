use ir::CpuCustomOp;
use std::ffi::CString;
use std::fmt::Debug;
use std::sync::Arc;

use crate::internal_prelude::*;

pub struct CpuFn<T: Value> {
    op: CArc<CpuCustomOp>,
    _marker: PhantomData<T>,
}

/*
Interestingly, Box::into_raw(Box<Closure>) does not give a valid pointer.
*/
struct ClosureContainer<T> {
    f: Arc<dyn Fn(&mut T) + 'static + Send + Sync>,
}

impl<T: Value> CpuFn<T> {
    pub fn new<F: Fn(&mut T) + 'static + Send + Sync>(f: F) -> Self {
        let f_ptr = Box::into_raw(Box::new(ClosureContainer::<T> { f: Arc::new(f) }));
        let op = CpuCustomOp {
            data: f_ptr as *mut u8,
            func: _trampoline::<T, F>,
            destructor: _drop::<F>,
            arg_type: T::type_(),
        };
        Self {
            op: CArc::new(op),
            _marker: PhantomData,
        }
    }
    pub fn call(&self, arg: impl AsExpr<Value = T>) -> Expr<T> {
        RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            assert!(r.lock);
            assert_eq!(
                r.device
                    .as_ref()
                    .unwrap()
                    .upgrade()
                    .unwrap()
                    .inner
                    .query("device_name")
                    .unwrap(),
                "cpu",
                "CpuFn can only be used in cpu backend"
            );
            let addr = CArc::as_ptr(&self.op) as u64;
            if let Some((_, op)) = r.cpu_custom_ops.get(&addr) {
                assert_eq!(CArc::as_ptr(op), CArc::as_ptr(&self.op));
            } else {
                let i = r.cpu_custom_ops.len();
                r.cpu_custom_ops.insert(addr, (i, self.op.clone()));
            }
        });
        Expr::<T>::from_node(__current_scope(|b| {
            b.call(
                Func::CpuCustomOp(self.op.clone()),
                &[arg.as_expr().node()],
                T::type_(),
            )
        }))
    }
}

extern "C" fn _trampoline<T, F: FnMut(&mut T)>(data: *mut u8, args: *mut u8) {
    unsafe {
        let container = &*(data as *const ClosureContainer<T>);
        let f = &container.f;
        let args = &mut *(args as *mut T);
        f(args);
    }
}

extern "C" fn _drop<T>(data: *mut u8) {
    unsafe {
        let _ = Box::from_raw(data as *mut T);
    }
}

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
    RECORDER.with(|r| {
        let r = r.borrow();
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
    __current_scope(|b| {
        b.call(
            Func::Assert(CBoxedSlice::new(
                CString::new(msg).unwrap().into_bytes_with_nul(),
            )),
            &[cond.node()],
            Type::void(),
        );
    });
}

pub fn comment(msg: &str) {
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
        $crate::lang::debug::comment(&format!("`{}` at {}:{}:{}", $msg, file!(), line!(), column!()))
    };
}
