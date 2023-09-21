use crate::internal_prelude::*;

pub trait IntoIndex {
    fn to_u64(&self) -> Expr<u64>;
}
impl IntoIndex for i32 {
    fn to_u64(&self) -> Expr<u64> {
        (*self as u64).expr()
    }
}
impl IntoIndex for i64 {
    fn to_u64(&self) -> Expr<u64> {
        (*self as u64).expr()
    }
}
impl IntoIndex for u32 {
    fn to_u64(&self) -> Expr<u64> {
        (*self as u64).expr()
    }
}
impl IntoIndex for u64 {
    fn to_u64(&self) -> Expr<u64> {
        (*self).expr()
    }
}
impl IntoIndex for Expr<u32> {
    fn to_u64(&self) -> Expr<u64> {
        self.cast::<u64>()
    }
}
impl IntoIndex for Expr<u64> {
    fn to_u64(&self) -> Expr<u64> {
        *self
    }
}

pub trait IndexRead: ToNode {
    type Element: Value;
    fn read<I: IntoIndex>(&self, i: I) -> Expr<Self::Element>;
}

pub trait IndexWrite: IndexRead {
    fn write<I: IntoIndex, V: AsExpr<Value = Self::Element>>(&self, i: I, value: V);
}
