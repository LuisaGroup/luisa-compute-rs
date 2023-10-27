use crate::internal_prelude::*;

use super::types::core::Primitive;

pub struct DebugFormatter {
    pub(crate) fmt: String,
    pub(crate) args: Vec<NodeRef>,
}
impl DebugFormatter {
    pub fn new() -> Self {
        Self {
            fmt: String::new(),
            args: Vec::new(),
        }
    }
    pub fn push_str(&mut self, s: &str) {
        assert!(
            s.find("{}").is_none(),
            "DebugFormatter::push_str cannot contain {{}}"
        );
        self.fmt.push_str(s);
    }
    pub fn push_arg(&mut self, node: NodeRef) {
        assert!(
            node.type_().is_primitive(),
            "DebugFormatter::push_arg must be primitive"
        );
        self.fmt.push_str("{}");
        self.args.push(node);
    }
}
pub trait DebugPrintValue: Value {
    fn fmt_args(e: Expr<Self>, fmt: &mut DebugFormatter);
}

impl<T: DebugPrintValue + Primitive> DebugPrintValue for T {
    fn fmt_args(e: Expr<Self>, fmt: &mut DebugFormatter) {
        fmt.push_arg(e.node().get());
    }
}

pub trait DebugPrint {
    fn fmt_args(&self, fmt: &mut DebugFormatter);
}

impl<T: DebugPrintValue> DebugPrint for Expr<T> {
    fn fmt_args(&self, fmt: &mut DebugFormatter) {
        T::fmt_args(*self, fmt);
    }
}
impl<T: DebugPrintValue> DebugPrint for Var<T> {
    fn fmt_args(&self, fmt: &mut DebugFormatter) {
        T::fmt_args(self.load(), fmt);
    }
}
