use super::*;
use crate::lang::types::core::*;
use crate::lang::types::VarDerefProxy;

macro_rules! impl_var_trait {
    ($t:ty) => {
        impl VarTrait for prim::Expr<$t> {
            type Value = $t;
            type Short = prim::Expr<i16>;
            type Ushort = prim::Expr<u16>;
            type Int = prim::Expr<i32>;
            type Uint = prim::Expr<u32>;
            type Long = prim::Expr<i64>;
            type Ulong = prim::Expr<u64>;
            type Half = prim::Expr<f16>;
            type Float = prim::Expr<f32>;
            type Double = prim::Expr<f64>;
            type Bool = prim::Expr<bool>;
        }
        impl ScalarVarTrait for prim::Expr<$t> {}
        impl ScalarOrVector for prim::Expr<$t> {
            type Element = prim::Expr<$t>;
            type ElementHost = $t;
        }
        impl BuiltinVarTrait for prim::Expr<$t> {}
    };
}
impl_var_trait!(f16);
impl_var_trait!(f32);
impl_var_trait!(f64);
impl_var_trait!(i16);
impl_var_trait!(u16);
impl_var_trait!(i32);
impl_var_trait!(u32);
impl_var_trait!(i64);
impl_var_trait!(u64);
impl_var_trait!(bool);

impl VarCmpEq for prim::Expr<f16> {}
impl VarCmpEq for prim::Expr<f32> {}
impl VarCmpEq for prim::Expr<f64> {}
impl VarCmpEq for prim::Expr<i16> {}
impl VarCmpEq for prim::Expr<i32> {}
impl VarCmpEq for prim::Expr<i64> {}
impl VarCmpEq for prim::Expr<u16> {}
impl VarCmpEq for prim::Expr<u32> {}
impl VarCmpEq for prim::Expr<u64> {}

impl VarCmpEq for prim::Expr<bool> {}

impl VarCmp for prim::Expr<f16> {}
impl VarCmp for prim::Expr<f32> {}
impl VarCmp for prim::Expr<f64> {}
impl VarCmp for prim::Expr<i16> {}
impl VarCmp for prim::Expr<i32> {}
impl VarCmp for prim::Expr<i64> {}
impl VarCmp for prim::Expr<u16> {}
impl VarCmp for prim::Expr<u32> {}
impl VarCmp for prim::Expr<u64> {}

impl CommonVarOp for prim::Expr<f16> {}
impl CommonVarOp for prim::Expr<f32> {}
impl CommonVarOp for prim::Expr<f64> {}
impl CommonVarOp for prim::Expr<i16> {}
impl CommonVarOp for prim::Expr<i32> {}
impl CommonVarOp for prim::Expr<i64> {}
impl CommonVarOp for prim::Expr<u16> {}
impl CommonVarOp for prim::Expr<u32> {}
impl CommonVarOp for prim::Expr<u64> {}

impl CommonVarOp for prim::Expr<bool> {}

impl FloatVarTrait for prim::Expr<f16> {}
impl FloatVarTrait for prim::Expr<f32> {}
impl FloatVarTrait for prim::Expr<f64> {}

impl IntVarTrait for prim::Expr<i16> {}
impl IntVarTrait for prim::Expr<i32> {}
impl IntVarTrait for prim::Expr<i64> {}
impl IntVarTrait for prim::Expr<u16> {}
impl IntVarTrait for prim::Expr<u32> {}
impl IntVarTrait for prim::Expr<u64> {}

macro_rules! impl_from {
    ($from:ty, $to:ty) => {
        impl From<$from> for prim::Expr<$to> {
            fn from(x: $from) -> Self {
                let y: $to = (x.try_into().unwrap());
                y.expr()
            }
        }
    };
}

impl_from!(i16, u16);
impl_from!(i16, i32);
impl_from!(i16, u32);
impl_from!(i16, i64);
impl_from!(i16, u64);

impl_from!(u16, i16);
impl_from!(u16, i32);
impl_from!(u16, u32);
impl_from!(u16, i64);
impl_from!(u16, u64);

impl_from!(i32, u16);
impl_from!(i32, i16);
impl_from!(i32, u32);
impl_from!(i32, i64);
impl_from!(i32, u64);

impl_from!(i64, u16);
impl_from!(i64, i16);
impl_from!(i64, u64);
impl_from!(i64, i32);
impl_from!(i64, u32);

impl_from!(u32, u16);
impl_from!(u32, i16);
impl_from!(u32, i32);
impl_from!(u32, i64);
impl_from!(u32, u64);

impl_from!(u64, u16);
impl_from!(u64, i16);
impl_from!(u64, i64);
impl_from!(u64, i32);
impl_from!(u64, u32);

impl From<f64> for prim::Expr<f32> {
    fn from(x: f64) -> Self {
        (x as f32).into()
    }
}
impl From<f32> for prim::Expr<f64> {
    fn from(x: f32) -> Self {
        (x as f64).into()
    }
}
impl From<f64> for prim::Expr<f16> {
    fn from(x: f64) -> Self {
        f16::from_f64(x).into()
    }
}
impl From<f32> for prim::Expr<f16> {
    fn from(x: f32) -> Self {
        f16::from_f32(x).into()
    }
}

macro_rules! impl_binop {
    ($t:ty, $proxy:ty, $tr_assign:ident, $method_assign:ident, $tr:ident, $method:ident) => {
        impl $tr_assign<prim::Expr<$t>> for $proxy {
            fn $method_assign(&mut self, rhs: prim::Expr<$t>) {
                *self = self.clone().$method(rhs);
            }
        }
        impl $tr_assign<$t> for $proxy {
            fn $method_assign(&mut self, rhs: $t) {
                *self = self.clone().$method(rhs);
            }
        }
        impl $tr<prim::Expr<$t>> for $proxy {
            type Output = prim::Expr<$t>;
            fn $method(self, rhs: prim::Expr<$t>) -> Self::Output {
                __current_scope(|s| {
                    let lhs = ToNode::node(&self);
                    let rhs = ToNode::node(&rhs);
                    let ret = s.call(Func::$tr, &[lhs, rhs], Self::Output::type_());
                    Expr::<$t>::from_node(ret)
                })
            }
        }

        impl $tr<$t> for $proxy {
            type Output = prim::Expr<$t>;
            fn $method(self, rhs: $t) -> Self::Output {
                $tr::$method(self, rhs.expr())
            }
        }
        impl $tr<$proxy> for $t {
            type Output = prim::Expr<$t>;
            fn $method(self, rhs: $proxy) -> Self::Output {
                $tr::$method(self.expr(), rhs)
            }
        }
    };
}
macro_rules! impl_common_binop {
    ($t:ty,$proxy:ty) => {
        impl_binop!($t, $proxy, AddAssign, add_assign, Add, add);
        impl_binop!($t, $proxy, SubAssign, sub_assign, Sub, sub);
        impl_binop!($t, $proxy, MulAssign, mul_assign, Mul, mul);
        impl_binop!($t, $proxy, DivAssign, div_assign, Div, div);
        impl_binop!($t, $proxy, RemAssign, rem_assign, Rem, rem);
    };
}
macro_rules! impl_int_binop {
    ($t:ty,$proxy:ty) => {
        impl_binop!($t, $proxy, ShlAssign, shl_assign, Shl, shl);
        impl_binop!($t, $proxy, ShrAssign, shr_assign, Shr, shr);
        impl_binop!($t, $proxy, BitAndAssign, bitand_assign, BitAnd, bitand);
        impl_binop!($t, $proxy, BitOrAssign, bitor_assign, BitOr, bitor);
        impl_binop!($t, $proxy, BitXorAssign, bitxor_assign, BitXor, bitxor);
    };
}

macro_rules! impl_not {
    ($t:ty,$proxy:ty) => {
        impl Not for $proxy {
            type Output = prim::Expr<$t>;
            fn not(self) -> Self::Output {
                __current_scope(|s| {
                    let ret = s.call(Func::BitNot, &[ToNode::node(&self)], Self::Output::type_());
                    Expr::<$t>::from_node(ret)
                })
            }
        }
    };
}
macro_rules! impl_neg {
    ($t:ty,$proxy:ty) => {
        impl Neg for $proxy {
            type Output = prim::Expr<$t>;
            fn neg(self) -> Self::Output {
                __current_scope(|s| {
                    let ret = s.call(Func::Neg, &[ToNode::node(&self)], Self::Output::type_());
                    Expr::<$t>::from_node(ret)
                })
            }
        }
    };
}
macro_rules! impl_fneg {
    ($t:ty, $proxy:ty) => {
        impl Neg for $proxy {
            type Output = prim::Expr<$t>;
            fn neg(self) -> Self::Output {
                __current_scope(|s| {
                    let ret = s.call(Func::Neg, &[ToNode::node(&self)], Self::Output::type_());
                    Expr::<$t>::from_node(ret)
                })
            }
        }
    };
}
impl Not for prim::Expr<bool> {
    type Output = prim::Expr<bool>;
    fn not(self) -> Self::Output {
        __current_scope(|s| {
            let ret = s.call(Func::BitNot, &[ToNode::node(&self)], Self::Output::type_());
            FromNode::from_node(ret)
        })
    }
}
impl_common_binop!(f16, prim::Expr<f16>);
impl_common_binop!(f32, prim::Expr<f32>);
impl_common_binop!(f64, prim::Expr<f64>);
impl_common_binop!(i16, prim::Expr<i16>);
impl_common_binop!(i32, prim::Expr<i32>);
impl_common_binop!(i64, prim::Expr<i64>);
impl_common_binop!(u16, prim::Expr<u16>);
impl_common_binop!(u32, prim::Expr<u32>);
impl_common_binop!(u64, prim::Expr<u64>);

impl_binop!(
    bool,
    prim::Expr<bool>,
    BitAndAssign,
    bitand_assign,
    BitAnd,
    bitand
);
impl_binop!(
    bool,
    prim::Expr<bool>,
    BitOrAssign,
    bitor_assign,
    BitOr,
    bitor
);
impl_binop!(
    bool,
    prim::Expr<bool>,
    BitXorAssign,
    bitxor_assign,
    BitXor,
    bitxor
);
impl_int_binop!(i16, prim::Expr<i16>);
impl_int_binop!(i32, prim::Expr<i32>);
impl_int_binop!(i64, prim::Expr<i64>);
impl_int_binop!(u16, prim::Expr<u16>);
impl_int_binop!(u32, prim::Expr<u32>);
impl_int_binop!(u64, prim::Expr<u64>);

impl_not!(i16, prim::Expr<i16>);
impl_not!(i32, prim::Expr<i32>);
impl_not!(i64, prim::Expr<i64>);
impl_not!(u16, prim::Expr<u16>);
impl_not!(u32, prim::Expr<u32>);
impl_not!(u64, prim::Expr<u64>);

impl_neg!(i16, prim::Expr<i16>);
impl_neg!(i32, prim::Expr<i32>);
impl_neg!(i64, prim::Expr<i64>);
impl_neg!(u16, prim::Expr<u16>);
impl_neg!(u32, prim::Expr<u32>);
impl_neg!(u64, prim::Expr<u64>);

impl_fneg!(f16, prim::Expr<f16>);
impl_fneg!(f32, prim::Expr<f32>);
impl_fneg!(f64, prim::Expr<f64>);

macro_rules! impl_assign_ops {
    ($ass:ident, $ass_m:ident, $o:ident, $o_m:ident) => {
        impl<P, T: Value, Rhs> std::ops::$ass<Rhs> for VarDerefProxy<P, T>
        where
            P: VarProxy<Value = T>,
            Expr<T>: std::ops::$o<Rhs, Output = Expr<T>>,
        {
            fn $ass_m(&mut self, rhs: Rhs) {
                *self.deref_mut() = std::ops::$o::$o_m(**self, rhs);
            }
        }
    };
}
impl_assign_ops!(AddAssign, add_assign, Add, add);
impl_assign_ops!(SubAssign, sub_assign, Sub, sub);
impl_assign_ops!(MulAssign, mul_assign, Mul, mul);
impl_assign_ops!(DivAssign, div_assign, Div, div);
impl_assign_ops!(RemAssign, rem_assign, Rem, rem);
impl_assign_ops!(BitAndAssign, bitand_assign, BitAnd, bitand);
impl_assign_ops!(BitOrAssign, bitor_assign, BitOr, bitor);
impl_assign_ops!(BitXorAssign, bitxor_assign, BitXor, bitxor);
impl_assign_ops!(ShlAssign, shl_assign, Shl, shl);
impl_assign_ops!(ShrAssign, shr_assign, Shr, shr);
