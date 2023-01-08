use crate::prelude::*;
use luisa_compute_ir::{ir::Func, ir::Type, TypeOf};
use std::any::Any;
use std::ops::*;

use super::Expr;
pub trait VarTrait: Copy + Clone + 'static + FromNode {
    type Value: Value;
    type Int: VarTrait;
    type Uint: VarTrait;
    type Long: VarTrait;
    type Ulong: VarTrait;
    type Float: VarTrait;
    // type Double: VarTrait;
    type Bool: VarTrait + Not<Output = Self::Bool>;
    fn type_() -> Gc<Type> {
        <Self::Value as TypeOf>::type_()
    }
}
macro_rules! impl_var_trait {
    ($t:ty) => {
        impl VarTrait for PrimExpr<$t> {
            type Value = $t;
            type Int = Expr<i32>;
            type Uint = Expr<u32>;
            type Long = Expr<i64>;
            type Ulong = Expr<u64>;
            type Float = Expr<f32>;
            // type Double = Expr<f64>;
            type Bool = Expr<bool>;
        }
    };
}
impl_var_trait!(f32);
impl_var_trait!(f64);
impl_var_trait!(i32);
impl_var_trait!(u32);
impl_var_trait!(i64);
impl_var_trait!(u64);
impl_var_trait!(bool);

impl<T: Copy + 'static + Value> FromNode for PrimExpr<T> {
    fn from_node(node: NodeRef) -> Self {
        Self {
            node,
            _phantom: std::marker::PhantomData,
        }
    }
    fn node(&self) -> NodeRef {
        self.node
    }
}
pub trait CommonVarOp: VarTrait {
    fn max<A: Into<Self>>(&self, other: A) -> Self {
        let lhs = self.node();
        let rhs = other.into().node();
        current_scope(|s| {
            let ret = s.call(Func::Max, &[lhs, rhs], Self::type_());
            Self::from_node(ret)
        })
    }
    fn min<A: Into<Self>>(&self, other: A) -> Self {
        let lhs = self.node();
        let rhs = other.into().node();
        current_scope(|s| {
            let ret = s.call(Func::Min, &[lhs, rhs], Self::type_());
            Self::from_node(ret)
        })
    }
    fn clamp<A: Into<Self>, B: Into<Self>>(&self, min: A, max: B) -> Self {
        let min = min.into().node();
        let max = max.into().node();
        current_scope(|s| {
            let ret = s.call(Func::Clamp, &[self.node(), min, max], Self::type_());
            Self::from_node(ret)
        })
    }
    fn abs(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Abs, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn _cast<A: VarTrait>(&self) -> A {
        let ty = <A::Value>::type_();
        let node = current_scope(|s| s.cast(self.node(), ty));
        FromNode::from_node(node)
    }
    fn bitcast<T: Value>(&self) -> Expr<T> {
        assert_eq!(std::mem::size_of::<Self::Value>(), std::mem::size_of::<T>());
        let ty = <T>::type_();
        let node = current_scope(|s| s.bitcast(self.node(), ty));
        Expr::<T>::from_node(node)
    }
    fn uint(&self) -> Self::Uint {
        self._cast()
    }
    fn int(&self) -> Self::Int {
        self._cast()
    }
    fn ulong(&self) -> Self::Ulong {
        self._cast()
    }
    fn long(&self) -> Self::Long {
        self._cast()
    }
    fn float(&self) -> Self::Float {
        self._cast()
    }
    // fn double(&self) -> Self::Double {
    //     self._cast()
    // }
    fn bool_(&self) -> Self::Bool {
        self._cast()
    }
}
pub trait VarCmpEq: VarTrait {
    fn cmpeq<A: Into<Self>>(&self, other: A) -> Self::Bool {
        let lhs = self.node();
        let rhs = other.into().node();
        current_scope(|s| {
            let ret = s.call(Func::Eq, &[lhs, rhs],  Self::Bool::type_());
            FromNode::from_node(ret)
        })
    }
    fn cmpne<A: Into<Self>>(&self, other: A) -> Self::Bool {
        let lhs = self.node();
        let rhs = other.into().node();
        current_scope(|s| {
            let ret = s.call(Func::Ne, &[lhs, rhs],  Self::Bool::type_());
            FromNode::from_node(ret)
        })
    }
}
pub trait VarCmp: VarTrait + VarCmpEq{
    fn cmplt<A: Into<Self>>(&self, other: A) -> Self::Bool {
        let lhs = self.node();
        let rhs = other.into().node();
        current_scope(|s| {
            let ret = s.call(Func::Lt, &[lhs, rhs],  Self::Bool::type_());
            FromNode::from_node(ret)
        })
    }
    fn cmple<A: Into<Self>>(&self, other: A) -> Self::Bool {
        let lhs = self.node();
        let rhs = other.into().node();
        current_scope(|s| {
            let ret = s.call(Func::Le, &[lhs, rhs],  Self::Bool::type_());
            FromNode::from_node(ret)
        })
    }
    fn cmpgt<A: Into<Self>>(&self, other: A) -> Self::Bool {
        let lhs = self.node();
        let rhs = other.into().node();
        current_scope(|s| {
            let ret = s.call(Func::Gt, &[lhs, rhs],  Self::Bool::type_());
            FromNode::from_node(ret)
        })
    }
    fn cmpge<A: Into<Self>>(&self, other: A) -> Self::Bool {
        let lhs = self.node();
        let rhs = other.into().node();
        current_scope(|s| {
            let ret = s.call(Func::Ge, &[lhs, rhs],  Self::Bool::type_());
            FromNode::from_node(ret)
        })
    }
}
pub trait IntVarTrait:
    VarTrait
    + CommonVarOp
    + VarCmp
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + Shl<Output = Self>
    + Shr<Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + RemAssign
    + ShlAssign
    + ShrAssign
    + BitAndAssign
    + BitOrAssign
    + BitXorAssign
    + Neg<Output = Self>
    + Clone
    + Not<Output = Self>
    + From<Self::Value>
    + From<i64>
{
    fn one() -> Self {
        Self::from(1i64)
    }
    fn zero() -> Self {
        Self::from(0i64)
    }
    fn rotate_right(&self, n: Expr<u32>) -> Self {
        let lhs = self.node();
        let rhs = Expr::<u32>::node(&n);
        current_scope(|s| {
            let ret = s.call(Func::RotRight, &[lhs, rhs], Self::type_());
            Self::from_node(ret)
        })
    }
    fn rotate_left(&self, n: Expr<u32>) -> Self {
        let lhs = self.node();
        let rhs = Expr::<u32>::node(&n);
        current_scope(|s| {
            let ret = s.call(Func::RotLeft, &[lhs, rhs], Self::type_());
            Self::from_node(ret)
        })
    }
}
pub trait FloatVarTrait:
    VarTrait
    + CommonVarOp
    + VarCmp
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + RemAssign
    + Clone
    + From<Self::Value>
    + From<f32>
{
    fn one() -> Self {
        Self::from(1.0f32)
    }
    fn zero() -> Self {
        Self::from(0.0f32)
    }
    fn mul_add<A: Into<Self>, B: Into<Self>>(&self, a: A, b: B) -> Self {
        let a: Self = a.into();
        let b: Self = b.into();
        let node =
            current_scope(|s| s.call(Func::Fma, &[self.node(), a.node(), b.node()], Self::type_()));
        Self::from_node(node)
    }
    fn ceil(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Ceil, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn floor(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Floor, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn round(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Round, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn trunc(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Trunc, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn copysign<A: Into<Self>>(&self, other: A) -> Self {
        current_scope(|s| {
            let ret = s.call(
                Func::Copysign,
                &[self.node(), other.into().node()],
                Self::type_(),
            );
            Self::from_node(ret)
        })
    }
    fn sqrt(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Sqrt, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn rsqrt(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Rsqrt, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn fract(&self) -> Self {
        self.clone() - self.clone().floor()
    }
    fn sin(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Sin, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
        // crate::math::approx_sin_cos(self.clone(), true, false).0
    }
    fn cos(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Cos, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
        // crate::math::approx_sin_cos(self.clone(), false, true).1
    }
    fn tan(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Tan, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn asin(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Asin, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn acos(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Acos, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn atan(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Atan, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn atan2(&self, other: Self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Atan2, &[self.node(), other.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn sinh(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Sinh, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn cosh(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Cosh, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn tanh(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Tanh, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn asinh(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Asinh, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn acosh(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Acosh, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn atanh(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Atanh, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn exp(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Exp, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn exp2(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Exp2, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn is_finite(&self) -> Self::Bool {
        !self.is_infinite()
    }
    fn is_infinite(&self) -> Self::Bool {
        current_scope(|s| {
            let ret = s.call(Func::IsInf, &[self.node()], Self::type_());
            FromNode::from_node(ret)
        })
    }
    fn is_nan(&self) -> Self::Bool {
        // let any = self as &dyn Any;
        // if let Some(a) = any.downcast_ref::<Expr<f32>>() {
        //     let u: Expr<u32> = a.bitcast::<u32>();
        //     (u & 0x7f800000u32).cmpeq(0x7f800000u32) & (u & 0x007fffffu32).cmpne(0u32)
        // } else {
        //     panic!("expect Expr<f32>")
        // }
        current_scope(|s| {
            let ret = s.call(Func::IsNan, &[self.node()], Self::type_());
            FromNode::from_node(ret)
        })
    }
    fn ln(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Log, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn log(&self, base: Self) -> Self {
        self.ln() / base.ln()
    }
    fn log2(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Log2, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn log10(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Log10, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn powf<A: Into<Self>>(&self, exp: A) -> Self {
        (self.ln() * exp.into()).exp()
    }
    fn powi(&self, exp: i32) -> Self {
        let mut n = exp.abs();
        let mut result = Self::one();
        let mut x = self.clone();
        while n > 0 {
            if n & 1 == 1 {
                result *= x.clone();
            }
            x *= x.clone();
            n >>= 1;
        }
        if exp < 0 {
            result.recip()
        } else {
            result
        }
    }
    fn recip(&self) -> Self {
        Self::one() / self.clone()
    }
    fn sin_cos(&self) -> (Self, Self) {
        (self.sin(), self.cos())
    }
}
macro_rules! impl_binop {
    ($t:ty, $proxy:ty, $tr_assign:ident, $method_assign:ident, $tr:ident, $method:ident) => {
        impl $tr_assign<Expr<$t>> for $proxy {
            fn $method_assign(&mut self, rhs: Expr<$t>) {
                *self = self.clone().$method(rhs);
            }
        }
        impl $tr_assign<$t> for $proxy {
            fn $method_assign(&mut self, rhs: $t) {
                *self = self.clone().$method(rhs);
            }
        }
        impl $tr<Expr<$t>> for $proxy {
            type Output = Expr<$t>;
            fn $method(self, rhs: Expr<$t>) -> Self::Output {
                current_scope(|s| {
                    let lhs = FromNode::node(&self);
                    let rhs = FromNode::node(&rhs);
                    let ret = s.call(Func::$tr, &[lhs, rhs], Self::Output::type_());
                    Expr::<$t>::from_node(ret)
                })
            }
        }

        impl $tr<$t> for $proxy {
            type Output = Expr<$t>;
            fn $method(self, rhs: $t) -> Self::Output {
                $tr::$method(self, const_(rhs))
            }
        }
        impl $tr<$proxy> for $t {
            type Output = Expr<$t>;
            fn $method(self, rhs: $proxy) -> Self::Output {
                $tr::$method(const_(self), rhs)
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
            type Output = Expr<$t>;
            fn not(self) -> Self::Output {
                current_scope(|s| {
                    let ret = s.call(Func::BitNot, &[FromNode::node(&self)], Self::Output::type_());
                    Expr::<$t>::from_node(ret)
                })
            }
        }
    };
}
macro_rules! impl_neg {
    ($t:ty,$proxy:ty) => {
        impl Neg for $proxy {
            type Output = Expr<$t>;
            fn neg(self) -> Self::Output {
                current_scope(|s| {
                    let ret = s.call(Func::Neg, &[FromNode::node(&self)], Self::Output::type_());
                    Expr::<$t>::from_node(ret)
                })
            }
        }
    };
}
macro_rules! impl_fneg {
    ($t:ty, $proxy:ty) => {
        impl Neg for $proxy {
            type Output = Expr<$t>;
            fn neg(self) -> Self::Output {
                current_scope(|s| {
                    let ret = s.call(Func::Neg, &[FromNode::node(&self)], Self::Output::type_());
                    Expr::<$t>::from_node(ret)
                })
            }
        }
    };
}
impl Not for PrimExpr<bool> {
    type Output = Expr<bool>;
    fn not(self) -> Self::Output {
        current_scope(|s| {
            let ret = s.call(Func::BitNot, &[FromNode::node(&self)], Self::Output::type_());
            FromNode::from_node(ret)
        })
    }
}
impl_common_binop!(f32, PrimExpr<f32>);
impl_common_binop!(f64, PrimExpr<f64>);
impl_common_binop!(i32, PrimExpr<i32>);
impl_common_binop!(i64, PrimExpr<i64>);
impl_common_binop!(u32, PrimExpr<u32>);
impl_common_binop!(u64, PrimExpr<u64>);

impl_binop!(
    bool,
    PrimExpr<bool>,
    BitAndAssign,
    bitand_assign,
    BitAnd,
    bitand
);
impl_binop!(
    bool,
    PrimExpr<bool>,
    BitOrAssign,
    bitor_assign,
    BitOr,
    bitor
);
impl_binop!(
    bool,
    PrimExpr<bool>,
    BitXorAssign,
    bitxor_assign,
    BitXor,
    bitxor
);
impl_int_binop!(i32, PrimExpr<i32>);
impl_int_binop!(i64, PrimExpr<i64>);
impl_int_binop!(u32, PrimExpr<u32>);
impl_int_binop!(u64, PrimExpr<u64>);

impl_not!(i32, PrimExpr<i32>);
impl_not!(i64, PrimExpr<i64>);
impl_not!(u32, PrimExpr<u32>);
impl_not!(u64, PrimExpr<u64>);

impl_neg!(i32, PrimExpr<i32>);
impl_neg!(i64, PrimExpr<i64>);
impl_neg!(u32, PrimExpr<u32>);
impl_neg!(u64, PrimExpr<u64>);

impl_fneg!(f32, PrimExpr<f32>);
impl VarCmpEq for PrimExpr<f32> {}
// impl VarCmp for PrimExpr<f64> {}
impl VarCmpEq for PrimExpr<i32> {}
impl VarCmpEq for PrimExpr<i64> {}
impl VarCmpEq for PrimExpr<u32> {}
impl VarCmpEq for PrimExpr<u64> {}

// impl_fneg!(f64, PrimExpr<f64>);
impl VarCmp for PrimExpr<f32> {}
// impl VarCmp for PrimExpr<f64> {}
impl VarCmp for PrimExpr<i32> {}
impl VarCmp for PrimExpr<i64> {}
impl VarCmp for PrimExpr<u32> {}
impl VarCmp for PrimExpr<u64> {}
impl VarCmpEq for PrimExpr<bool> {}
impl CommonVarOp for PrimExpr<f32> {}
// impl CommonVarOp for PrimExpr<f64> {}
impl CommonVarOp for PrimExpr<i32> {}
impl CommonVarOp for PrimExpr<i64> {}
impl CommonVarOp for PrimExpr<u32> {}
impl CommonVarOp for PrimExpr<u64> {}
impl CommonVarOp for PrimExpr<bool> {}

impl From<f64> for Float32 {
    fn from(x: f64) -> Self {
        (x as f32).into()
    }
}

impl FloatVarTrait for PrimExpr<f32> {}
impl IntVarTrait for PrimExpr<i32> {}
impl IntVarTrait for PrimExpr<i64> {}
impl IntVarTrait for PrimExpr<u32> {}
impl IntVarTrait for PrimExpr<u64> {}

macro_rules! impl_from {
    ($from:ty, $to:ty) => {
        impl From<$from> for PrimExpr<$to> {
            fn from(x: $from) -> Self {
                const_::<$to>(x.try_into().unwrap())
            }
        }
    };
}
impl_from!(i32, u32);
impl_from!(i32, i64);
impl_from!(i32, u64);

impl_from!(i64, u64);
impl_from!(i64, i32);
impl_from!(i64, u32);

impl_from!(u32, i32);
impl_from!(u32, i64);
impl_from!(u32, u64);

impl_from!(u64, i64);
impl_from!(u64, i32);
impl_from!(u64, u32);
