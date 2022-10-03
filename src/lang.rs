use crate::*;
pub struct Var<T: Copy> {
    pub(crate) expr: sys::LCExpression,
    pub(crate) _marker: std::marker::PhantomData<T>,
}
