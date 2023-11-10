use std::ffi::CString;

use crate::internal_prelude::*;
/// TODO: support custom print format
pub struct DevicePrintFormatter {
    pub(crate) fmt: String,
    pub(crate) args: Vec<NodeRef>,
}
impl DevicePrintFormatter {
    pub fn new() -> Self {
        Self {
            fmt: String::new(),
            args: Vec::new(),
        }
    }
    pub fn push_str(&mut self, s: &str) {
        self.fmt.push_str(s);
    }
    pub fn push_arg(&mut self, node: SafeNodeRef) {
        self.args.push(node.get());
    }
    pub fn print(self) {
        let Self { fmt, args } = self;
        __current_scope(|b| {
            b.print(
                CBoxedSlice::new(CString::new(fmt).unwrap().into_bytes_with_nul()),
                &args,
            );
        })
    }
}
pub trait DevicePrint {
    fn fmt(&self, fmt: &mut DevicePrintFormatter);
}

impl<T: Value> DevicePrint for Expr<T> {
    fn fmt(&self, fmt: &mut DevicePrintFormatter) {
        fmt.push_arg(self.node());
    }
}
impl<T: Value> DevicePrint for Var<T> {
    fn fmt(&self, fmt: &mut DevicePrintFormatter) {
        DevicePrint::fmt(&self.load(), fmt);
    }
}

#[macro_export]
macro_rules! device_log {
    ($fmt:literal, $($arg:expr),*) => {{
        let mut fmt = $crate::lang::print::DevicePrintFormatter::new();
        fmt.push_str($fmt);
        $(
            $crate::lang::print::DevicePrint::fmt(&$arg, &mut fmt);
        )*
        fmt.print();
    }};
}
