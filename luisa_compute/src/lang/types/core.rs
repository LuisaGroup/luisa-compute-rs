use super::*;
use std::ops::Deref;

pub(crate) trait Primitive: Copy + TypeOf + 'static {
    fn const_(&self) -> Const;
}
impl<T: Primitive> Value for T {
    type Expr = PrimitiveExpr<T>;
    type Var = PrimitiveVar<T>;
    type ExprData = ();
    type VarData = ();

    fn expr(&self) -> Expr<Self> {
        let node = __current_scope(|s| -> NodeRef { s.const_(self.const_()) });
        Expr::<Self>::from_node(node)
    }
}

impl_simple_expr_proxy!([T: Primitive] PrimitiveExpr[T] for T);
impl_simple_var_proxy!([T: Primitive] PrimitiveVar[T] for T);

impl Primitive for bool {
    fn const_(&self) -> Const {
        Const::Bool(*self)
    }
}

impl Primitive for f16 {
    fn const_(&self) -> Const {
        Const::F16(*self)
    }
}
impl Primitive for f32 {
    fn const_(&self) -> Const {
        Const::F32(*self)
    }
}
impl Primitive for f64 {
    fn const_(&self) -> Const {
        Const::F64(*self)
    }
}

// impl Primitive for i8 {
//     fn const_(&self) -> Const {
//         Const::I8(*self)
//     }
// }
impl Primitive for i16 {
    fn const_(&self) -> Const {
        Const::I16(*self)
    }
}
impl Primitive for i32 {
    fn const_(&self) -> Const {
        Const::I32(*self)
    }
}
impl Primitive for i64 {
    fn const_(&self) -> Const {
        Const::I64(*self)
    }
}

// impl Primitive for u8 {
//     fn const_(&self) -> Const {
//         Const::U8(*self)
//     }
// }
impl Primitive for u16 {
    fn const_(&self) -> Const {
        Const::U16(*self)
    }
}
impl Primitive for u32 {
    fn const_(&self) -> Const {
        Const::U32(*self)
    }
}
impl Primitive for u64 {
    fn const_(&self) -> Const {
        Const::U64(*self)
    }
}

#[deprecated]
pub type Bool = Expr<bool>;
#[deprecated]
pub type F16 = Expr<f16>;
#[deprecated]
pub type F32 = Expr<f32>;
#[deprecated]
pub type F64 = Expr<f64>;
#[deprecated]
pub type I16 = Expr<i16>;
#[deprecated]
pub type I32 = Expr<i32>;
#[deprecated]
pub type I64 = Expr<i64>;
#[deprecated]
pub type U16 = Expr<u16>;
#[deprecated]
pub type U32 = Expr<u32>;
#[deprecated]
pub type U64 = Expr<u64>;

#[deprecated]
pub type F16Var = Var<f16>;
#[deprecated]
pub type F32Var = Var<f32>;
#[deprecated]
pub type F64Var = Var<f64>;
#[deprecated]
pub type I16Var = Var<i16>;
#[deprecated]
pub type I32Var = Var<i32>;
#[deprecated]
pub type I64Var = Var<i64>;
#[deprecated]
pub type U16Var = Var<u16>;
#[deprecated]
pub type U32Var = Var<u32>;
#[deprecated]
pub type U64Var = Var<u64>;

#[deprecated]
pub type Half = Expr<f16>;
#[deprecated]
pub type Float = Expr<f32>;
#[deprecated]
pub type Double = Expr<f64>;
#[deprecated]
pub type Int = Expr<i32>;
#[deprecated]
pub type Long = Expr<i64>;
#[deprecated]
pub type Uint = Expr<u32>;
#[deprecated]
pub type Ulong = Expr<u64>;
#[deprecated]
pub type Short = Expr<i16>;
#[deprecated]
pub type Ushort = Expr<u16>;

#[deprecated]
pub type BoolVar = Var<bool>;
#[deprecated]
pub type HalfVar = Var<f16>;
#[deprecated]
pub type FloatVar = Var<f32>;
#[deprecated]
pub type DoubleVar = Var<f64>;
#[deprecated]
pub type IntVar = Var<i32>;
#[deprecated]
pub type LongVar = Var<i64>;
#[deprecated]
pub type UintVar = Var<u32>;
#[deprecated]
pub type UlongVar = Var<u64>;
#[deprecated]
pub type ShortVar = Var<i16>;
#[deprecated]
pub type UshortVar = Var<u16>;
