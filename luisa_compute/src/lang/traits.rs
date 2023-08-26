use crate::prelude::*;
use crate::*;
use luisa_compute_ir::ir::new_user_node;
use luisa_compute_ir::CArc;
use luisa_compute_ir::{ir::Func, ir::Type, TypeOf};
use std::cell::{Cell, RefCell};
use std::ops::*;

use super::Expr;
pub trait VarTrait: Copy + Clone + 'static + FromNode {
    type Value: Value;
    type Short: VarTrait;
    type Ushort: VarTrait;
    type Int: VarTrait;
    type Uint: VarTrait;
    type Long: VarTrait;
    type Ulong: VarTrait;
    type Half: VarTrait;
    type Float: VarTrait;
    // type Double: VarTrait;
    type Bool: VarTrait + Not<Output = Self::Bool> + BitAnd<Output = Self::Bool>;
    fn type_() -> CArc<Type> {
        <Self::Value as TypeOf>::type_()
    }
}
macro_rules! impl_var_trait {
    ($t:ty) => {
        impl VarTrait for PrimExpr<$t> {
            type Value = $t;
            type Short = Expr<i16>;
            type Ushort = Expr<u16>;
            type Int = Expr<i32>;
            type Uint = Expr<u32>;
            type Long = Expr<i64>;
            type Ulong = Expr<u64>;
            type Half = Expr<f16>;
            type Float = Expr<f32>;
            // type Double = Expr<f64>;
            type Bool = Expr<bool>;
        }
        impl ScalarVarTrait for PrimExpr<$t> {}
        impl ScalarOrVector for PrimExpr<$t> {
            type Element = PrimExpr<$t>;
            type ElementHost = $t;
        }
        impl BuiltinVarTrait for PrimExpr<$t> {}
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

impl<T: Copy + 'static + Value> FromNode for PrimExpr<T> {
    fn from_node(node: NodeRef) -> Self {
        Self {
            node,
            _phantom: std::marker::PhantomData,
        }
    }
}
impl<T: Copy + 'static + Value> ToNode for PrimExpr<T> {
    fn node(&self) -> NodeRef {
        self.node
    }
}
fn _cast<T: VarTrait, U: VarTrait>(expr: T) -> U {
    let node = expr.node();
    __current_scope(|s| {
        let ret = s.call(Func::Cast, &[node], U::type_());
        U::from_node(ret)
    })
}
pub trait CommonVarOp: VarTrait {
    fn max<A: Into<Self>>(&self, other: A) -> Self {
        let lhs = self.node();
        let rhs = other.into().node();
        __current_scope(|s| {
            let ret = s.call(Func::Max, &[lhs, rhs], Self::type_());
            Self::from_node(ret)
        })
    }
    fn min<A: Into<Self>>(&self, other: A) -> Self {
        let lhs = self.node();
        let rhs = other.into().node();
        __current_scope(|s| {
            let ret = s.call(Func::Min, &[lhs, rhs], Self::type_());
            Self::from_node(ret)
        })
    }
    fn clamp<A: Into<Self>, B: Into<Self>>(&self, min: A, max: B) -> Self {
        let min = min.into().node();
        let max = max.into().node();
        __current_scope(|s| {
            let ret = s.call(Func::Clamp, &[self.node(), min, max], Self::type_());
            Self::from_node(ret)
        })
    }
    fn abs(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Abs, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn bitcast<T: Value>(&self) -> Expr<T> {
        assert_eq!(std::mem::size_of::<Self::Value>(), std::mem::size_of::<T>());
        let ty = <T>::type_();
        let node = __current_scope(|s| s.bitcast(self.node(), ty));
        Expr::<T>::from_node(node)
    }
    fn uint(&self) -> Self::Uint {
        _cast(*self)
    }
    fn int(&self) -> Self::Int {
        _cast(*self)
    }
    fn ulong(&self) -> Self::Ulong {
        _cast(*self)
    }
    fn long(&self) -> Self::Long {
        _cast(*self)
    }
    fn float(&self) -> Self::Float {
        _cast(*self)
    }
    fn short(&self) -> Self::Short {
        _cast(*self)
    }
    fn ushort(&self) -> Self::Ushort {
        _cast(*self)
    }
    fn half(&self) -> Self::Half {
        _cast(*self)
    }
    // fn double(&self) -> Self::Double {
    //     _cast(*self)
    // }
    fn bool_(&self) -> Self::Bool {
        _cast(*self)
    }
}
pub trait VarCmpEq: VarTrait {
    fn cmpeq<A: Into<Self>>(&self, other: A) -> Self::Bool {
        let lhs = self.node();
        let rhs = other.into().node();
        __current_scope(|s| {
            let ret = s.call(Func::Eq, &[lhs, rhs], Self::Bool::type_());
            FromNode::from_node(ret)
        })
    }
    fn cmpne<A: Into<Self>>(&self, other: A) -> Self::Bool {
        let lhs = self.node();
        let rhs = other.into().node();
        __current_scope(|s| {
            let ret = s.call(Func::Ne, &[lhs, rhs], Self::Bool::type_());
            FromNode::from_node(ret)
        })
    }
}
pub trait VarCmp: VarTrait + VarCmpEq {
    fn cmplt<A: Into<Self>>(&self, other: A) -> Self::Bool {
        let lhs = self.node();
        let rhs = other.into().node();
        __current_scope(|s| {
            let ret = s.call(Func::Lt, &[lhs, rhs], Self::Bool::type_());
            FromNode::from_node(ret)
        })
    }
    fn cmple<A: Into<Self>>(&self, other: A) -> Self::Bool {
        let lhs = self.node();
        let rhs = other.into().node();
        __current_scope(|s| {
            let ret = s.call(Func::Le, &[lhs, rhs], Self::Bool::type_());
            FromNode::from_node(ret)
        })
    }
    fn cmpgt<A: Into<Self>>(&self, other: A) -> Self::Bool {
        let lhs = self.node();
        let rhs = other.into().node();
        __current_scope(|s| {
            let ret = s.call(Func::Gt, &[lhs, rhs], Self::Bool::type_());
            FromNode::from_node(ret)
        })
    }
    fn cmpge<A: Into<Self>>(&self, other: A) -> Self::Bool {
        let lhs = self.node();
        let rhs = other.into().node();
        __current_scope(|s| {
            let ret = s.call(Func::Ge, &[lhs, rhs], Self::Bool::type_());
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
        __current_scope(|s| {
            let ret = s.call(Func::RotRight, &[lhs, rhs], Self::type_());
            Self::from_node(ret)
        })
    }
    fn rotate_left(&self, n: Expr<u32>) -> Self {
        let lhs = self.node();
        let rhs = Expr::<u32>::node(&n);
        __current_scope(|s| {
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
        let node = __current_scope(|s| {
            s.call(Func::Fma, &[self.node(), a.node(), b.node()], Self::type_())
        });
        Self::from_node(node)
    }
    fn ceil(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Ceil, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn floor(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Floor, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn round(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Round, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn trunc(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Trunc, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn copysign<A: Into<Self>>(&self, other: A) -> Self {
        __current_scope(|s| {
            let ret = s.call(
                Func::Copysign,
                &[self.node(), other.into().node()],
                Self::type_(),
            );
            Self::from_node(ret)
        })
    }
    fn sqrt(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Sqrt, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn rsqrt(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Rsqrt, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn fract(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Fract, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }

    // x.step(edge)
    fn step(&self, edge: Self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Step, &[edge.node(), self.node()], Self::type_());
            Self::from_node(ret)
        })
    }

    fn smooth_step(&self, edge0: Self, edge1: Self) -> Self {
        __current_scope(|s| {
            let ret = s.call(
                Func::SmoothStep,
                &[edge0.node(), edge1.node(), self.node()],
                Self::type_(),
            );
            Self::from_node(ret)
        })
    }

    fn saturate(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Saturate, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }

    fn sin(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Sin, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
        // crate::math::approx_sin_cos(self.clone(), true, false).0
    }
    fn cos(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Cos, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
        // crate::math::approx_sin_cos(self.clone(), false, true).1
    }
    fn tan(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Tan, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn asin(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Asin, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn acos(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Acos, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn atan(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Atan, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn atan2(&self, other: Self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Atan2, &[self.node(), other.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn sinh(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Sinh, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn cosh(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Cosh, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn tanh(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Tanh, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn asinh(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Asinh, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn acosh(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Acosh, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn atanh(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Atanh, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn exp(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Exp, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn exp2(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Exp2, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn is_finite(&self) -> Self::Bool {
        !self.is_infinite() & !self.is_nan()
    }
    fn is_infinite(&self) -> Self::Bool {
        __current_scope(|s| {
            let ret = s.call(Func::IsInf, &[self.node()], <Self::Bool>::type_());
            FromNode::from_node(ret)
        })
    }
    fn is_nan(&self) -> Self::Bool {
        __current_scope(|s| {
            let ret = s.call(Func::IsNan, &[self.node()], <Self::Bool>::type_());
            FromNode::from_node(ret)
        })
    }
    fn ln(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Log, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn log(&self, base: impl Into<Self>) -> Self {
        self.ln() / base.into().ln()
    }
    fn log2(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Log2, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn log10(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Log10, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn powf(&self, exp: impl Into<Self>) -> Self {
        let exp = exp.into();
        __current_scope(|s| {
            let ret = s.call(Func::Powf, &[self.node(), exp.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn sqr(&self) -> Self {
        *self * *self
    }
    fn cube(&self) -> Self {
        *self * *self * *self
    }
    fn powi(&self, exp: impl Into<Self::Int>) -> Self {
        let exp = exp.into();
        __current_scope(|s| {
            let ret = s.call(Func::Powi, &[self.node(), exp.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn lerp(&self, other: impl Into<Self>, frac: impl Into<Self>) -> Self {
        let other = other.into();
        let frac = frac.into();
        __current_scope(|s| {
            let ret = s.call(
                Func::Lerp,
                &[self.node(), other.node(), frac.node()],
                Self::type_(),
            );
            Self::from_node(ret)
        })
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
                __current_scope(|s| {
                    let lhs = ToNode::node(&self);
                    let rhs = ToNode::node(&rhs);
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
            type Output = Expr<$t>;
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
            type Output = Expr<$t>;
            fn neg(self) -> Self::Output {
                __current_scope(|s| {
                    let ret = s.call(Func::Neg, &[ToNode::node(&self)], Self::Output::type_());
                    Expr::<$t>::from_node(ret)
                })
            }
        }
    };
}
impl Not for PrimExpr<bool> {
    type Output = Expr<bool>;
    fn not(self) -> Self::Output {
        __current_scope(|s| {
            let ret = s.call(Func::BitNot, &[ToNode::node(&self)], Self::Output::type_());
            FromNode::from_node(ret)
        })
    }
}
impl_common_binop!(f16, PrimExpr<f16>);
impl_common_binop!(f32, PrimExpr<f32>);
impl_common_binop!(f64, PrimExpr<f64>);
impl_common_binop!(i16, PrimExpr<i16>);
impl_common_binop!(i32, PrimExpr<i32>);
impl_common_binop!(i64, PrimExpr<i64>);
impl_common_binop!(u16, PrimExpr<u16>);
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
impl_int_binop!(i16, PrimExpr<i16>);
impl_int_binop!(i32, PrimExpr<i32>);
impl_int_binop!(i64, PrimExpr<i64>);
impl_int_binop!(u16, PrimExpr<u16>);
impl_int_binop!(u32, PrimExpr<u32>);
impl_int_binop!(u64, PrimExpr<u64>);

impl_not!(i16, PrimExpr<i16>);
impl_not!(i32, PrimExpr<i32>);
impl_not!(i64, PrimExpr<i64>);
impl_not!(u16, PrimExpr<u16>);
impl_not!(u32, PrimExpr<u32>);
impl_not!(u64, PrimExpr<u64>);

impl_neg!(i16, PrimExpr<i16>);
impl_neg!(i32, PrimExpr<i32>);
impl_neg!(i64, PrimExpr<i64>);
impl_neg!(u16, PrimExpr<u16>);
impl_neg!(u32, PrimExpr<u32>);
impl_neg!(u64, PrimExpr<u64>);

impl_fneg!(f16, PrimExpr<f16>);
impl_fneg!(f32, PrimExpr<f32>);
impl_fneg!(f64, PrimExpr<f64>);

impl VarCmpEq for PrimExpr<f16> {}
impl VarCmpEq for PrimExpr<f32> {}
impl VarCmpEq for PrimExpr<f64> {}
impl VarCmpEq for PrimExpr<i16> {}
impl VarCmpEq for PrimExpr<i32> {}
impl VarCmpEq for PrimExpr<i64> {}
impl VarCmpEq for PrimExpr<u16> {}
impl VarCmpEq for PrimExpr<u32> {}
impl VarCmpEq for PrimExpr<u64> {}

impl VarCmpEq for PrimExpr<bool> {}

impl VarCmp for PrimExpr<f16> {}
impl VarCmp for PrimExpr<f32> {}
impl VarCmp for PrimExpr<f64> {}
impl VarCmp for PrimExpr<i16> {}
impl VarCmp for PrimExpr<i32> {}
impl VarCmp for PrimExpr<i64> {}
impl VarCmp for PrimExpr<u16> {}
impl VarCmp for PrimExpr<u32> {}
impl VarCmp for PrimExpr<u64> {}

impl CommonVarOp for PrimExpr<f16> {}
impl CommonVarOp for PrimExpr<f32> {}
impl CommonVarOp for PrimExpr<f64> {}
impl CommonVarOp for PrimExpr<i16> {}
impl CommonVarOp for PrimExpr<i32> {}
impl CommonVarOp for PrimExpr<i64> {}
impl CommonVarOp for PrimExpr<u16> {}
impl CommonVarOp for PrimExpr<u32> {}
impl CommonVarOp for PrimExpr<u64> {}

impl CommonVarOp for PrimExpr<bool> {}

impl From<f64> for Float {
    fn from(x: f64) -> Self {
        (x as f32).into()
    }
}
impl From<f32> for Double {
    fn from(x: f32) -> Self {
        (x as f64).into()
    }
}
impl From<f64> for Half {
    fn from(x: f64) -> Self {
        f16::from_f64(x).into()
    }
}
impl From<f32> for Half {
    fn from(x: f32) -> Self {
        f16::from_f32(x).into()
    }
}

impl FloatVarTrait for PrimExpr<f16> {}
impl FloatVarTrait for PrimExpr<f32> {}
impl FloatVarTrait for PrimExpr<f64> {}

impl IntVarTrait for PrimExpr<i16> {}
impl IntVarTrait for PrimExpr<i32> {}
impl IntVarTrait for PrimExpr<i64> {}
impl IntVarTrait for PrimExpr<u16> {}
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

impl<T: Aggregate> Aggregate for Vec<T> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        let len_node = new_user_node(__module_pools(), nodes.len());
        nodes.push(len_node);
        for item in self {
            item.to_nodes(nodes);
        }
    }

    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        let len_node = iter.next().unwrap();
        let len = len_node.unwrap_user_data::<usize>();
        let mut ret = Vec::with_capacity(*len);
        for _ in 0..*len {
            ret.push(T::from_nodes(iter));
        }
        ret
    }
}

impl<T: Aggregate> Aggregate for RefCell<T> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        self.borrow().to_nodes(nodes);
    }

    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        RefCell::new(T::from_nodes(iter))
    }
}
impl<T: Aggregate + Copy> Aggregate for Cell<T> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        self.get().to_nodes(nodes);
    }

    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        Cell::new(T::from_nodes(iter))
    }
}
impl<T: Aggregate> Aggregate for Option<T> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        match self {
            Some(x) => {
                let node = new_user_node(__module_pools(), 1);
                nodes.push(node);
                x.to_nodes(nodes);
            }
            None => {
                let node = new_user_node(__module_pools(), 0);
                nodes.push(node);
            }
        }
    }

    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        let node = iter.next().unwrap();
        let tag = node.unwrap_user_data::<usize>();
        match *tag {
            0 => None,
            1 => Some(T::from_nodes(iter)),
            _ => unreachable!(),
        }
    }
}
pub trait ScalarVarTrait: ToNode + FromNode {}
pub trait VectorVarTrait: ToNode + FromNode {}
pub trait MatrixVarTrait: ToNode + FromNode {}
pub trait ScalarOrVector: ToNode + FromNode {
    type Element: ScalarVarTrait;
    type ElementHost: Value;
}
pub trait BuiltinVarTrait: ToNode + FromNode {}
pub trait Int32 {}
impl Int32 for i32 {}
impl Int32 for u32 {}