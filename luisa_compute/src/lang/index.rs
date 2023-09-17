use super::*;

pub trait IntoIndex {
    fn to_u64(&self) -> Expr<u64>;
}
impl IntoIndex for i32 {
    fn to_u64(&self) -> Expr<u64> {
        const_(*self as u64)
    }
}
impl IntoIndex for i64 {
    fn to_u64(&self) -> Expr<u64> {
        const_(*self as u64)
    }
}
impl IntoIndex for u32 {
    fn to_u64(&self) -> Expr<u64> {
        const_(*self as u64)
    }
}
impl IntoIndex for u64 {
    fn to_u64(&self) -> Expr<u64> {
        const_(*self)
    }
}
impl IntoIndex for Expr<u32> {
    fn to_u64(&self) -> Expr<u64> {
        self.ulong()
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
    fn write<I: IntoIndex, V: Into<Expr<Self::Element>>>(&self, i: I, value: V);
}
