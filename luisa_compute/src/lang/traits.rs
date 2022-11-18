use crate::prelude::*;
use luisa_compute_ir::{ir::Func, ir::Type, TypeOf};
use std::any::Any;
use std::ops::*;
pub trait VarTrait: Copy + Clone + 'static {
    type Scalar: Value;
    fn from_node(node: NodeRef) -> Self;
    fn node(&self) -> NodeRef;
    fn type_() -> Gc<Type> {
        <Self::Scalar as TypeOf>::type_()
    }
}
impl<T: Value + 'static> VarTrait for Expr<T> {
    type Scalar = T;
    fn from_node(node: NodeRef) -> Self {
        Expr::from_node(node)
    }
    fn node(&self) -> NodeRef {
        self.proxy.node()
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
        self.min(max).max(min)
    }
    fn abs(&self) -> Self {
        current_scope(|s| {
            let ret = s.call(Func::Abs, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn cast<A: VarTrait>(&self) -> A {
        let ty = <A as VarTrait>::type_();
        let node = current_scope(|s| s.cast(self.node(), ty));
        A::from_node(node)
    }
    fn bitcast<A: VarTrait>(&self) -> A {
        assert_eq!(
            std::mem::size_of::<Self::Scalar>(),
            std::mem::size_of::<A::Scalar>()
        );
        let ty = <A as VarTrait>::type_();
        let node = current_scope(|s| s.bitcast(self.node(), ty));
        A::from_node(node)
    }
    fn uint(&self) -> Expr<u32> {
        self.cast()
    }
    fn int(&self) -> Expr<i32> {
        self.cast()
    }
    fn ulong(&self) -> Expr<u64> {
        self.cast()
    }
    fn long(&self) -> Expr<i64> {
        self.cast()
    }
    fn float(&self) -> Expr<f32> {
        self.cast()
    }
    fn double(&self) -> Expr<f64> {
        self.cast()
    }
    fn bool_(&self) -> Expr<bool> {
        self.cast()
    }
}
pub trait VarCmp: VarTrait {
    fn lt<A: Into<Self>>(&self, other: A) -> Expr<bool> {
        let lhs = self.node();
        let rhs = other.into().node();
        current_scope(|s| {
            let ret = s.call(Func::Lt, &[lhs, rhs], Expr::<bool>::type_());
            Expr::<bool>::from_node(ret)
        })
    }
    fn le<A: Into<Self>>(&self, other: A) -> Expr<bool> {
        let lhs = self.node();
        let rhs = other.into().node();
        current_scope(|s| {
            let ret = s.call(Func::Le, &[lhs, rhs], Expr::<bool>::type_());
            Expr::<bool>::from_node(ret)
        })
    }
    fn gt<A: Into<Self>>(&self, other: A) -> Expr<bool> {
        let lhs = self.node();
        let rhs = other.into().node();
        current_scope(|s| {
            let ret = s.call(Func::Gt, &[lhs, rhs], Expr::<bool>::type_());
            Expr::<bool>::from_node(ret)
        })
    }
    fn ge<A: Into<Self>>(&self, other: A) -> Expr<bool> {
        let lhs = self.node();
        let rhs = other.into().node();
        current_scope(|s| {
            let ret = s.call(Func::Ge, &[lhs, rhs], Expr::<bool>::type_());
            Expr::<bool>::from_node(ret)
        })
    }
    fn eq<A: Into<Self>>(&self, other: A) -> Expr<bool> {
        let lhs = self.node();
        let rhs = other.into().node();
        current_scope(|s| {
            let ret = s.call(Func::Eq, &[lhs, rhs], Expr::<bool>::type_());
            Expr::<bool>::from_node(ret)
        })
    }
    fn ne<A: Into<Self>>(&self, other: A) -> Expr<bool> {
        let lhs = self.node();
        let rhs = other.into().node();
        current_scope(|s| {
            let ret = s.call(Func::Ne, &[lhs, rhs], Expr::<bool>::type_());
            Expr::<bool>::from_node(ret)
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
    + From<Self::Scalar>
{
    fn from_i64(i: i64) -> Self;
    fn one() -> Self {
        Self::from_i64(1)
    }
    fn zero() -> Self {
        Self::from_i64(0)
    }
    fn rotate_right(&self, n: Expr<u32>) -> Self {
        let lhs = self.node();
        let rhs = n.node();
        current_scope(|s| {
            let ret = s.call(Func::RotRight, &[lhs, rhs], Self::type_());
            Self::from_node(ret)
        })
    }
    fn rotate_left(&self, n: Expr<u32>) -> Self {
        let lhs = self.node();
        let rhs = n.node();
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
    + From<Self::Scalar>
{
    fn from_f64(x: f64) -> Self;
    fn one() -> Self {
        Self::from_f64(1.0)
    }
    fn zero() -> Self {
        Self::from_f64(0.0)
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
    fn is_finite(&self) -> Mask {
        unimplemented!()
    }
    fn is_infinite(&self) -> Mask {
        unimplemented!()
    }
    fn is_nan(&self) -> Mask {
        let any = self as &dyn Any;
        if let Some(a) = any.downcast_ref::<Expr<f32>>() {
            let u: Expr<u32> = a.bitcast();
            (&u & 0x7f800000u32).eq(0x7f800000u32) & (&u & 0x007fffffu32).ne(0u32)
        } else {
            panic!("expect Expr<f32>")
        }
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
    ($t:ty, $tr_assign:ident, $method_assign:ident, $tr:ident, $method:ident) => {
        impl $tr_assign for Expr<$t> {
            fn $method_assign(&mut self, rhs: Self) {
                *self = self.clone().$method(rhs);
            }
        }
        impl $tr_assign<&Expr<$t>> for Expr<$t> {
            fn $method_assign(&mut self, rhs: &Self) {
                *self = self.clone().$method(rhs);
            }
        }
        impl $tr_assign<$t> for Expr<$t> {
            fn $method_assign(&mut self, rhs: $t) {
                *self = self.clone().$method(rhs);
            }
        }
        impl $tr for &Expr<$t> {
            type Output = Expr<$t>;
            fn $method(self, rhs: &Expr<$t>) -> Self::Output {
                current_scope(|s| {
                    let lhs = self.node();
                    let rhs = rhs.node();
                    let ret = s.call(Func::$tr, &[lhs, rhs], Self::Output::type_());
                    Expr::<$t>::from_node(ret)
                })
            }
        }
        impl $tr<$t> for &Expr<$t> {
            type Output = Expr<$t>;
            fn $method(self, rhs: $t) -> Self::Output {
                $tr::$method(self, &const_(rhs))
            }
        }
        impl $tr<$t> for Expr<$t> {
            type Output = Expr<$t>;
            fn $method(self, rhs: $t) -> Self::Output {
                $tr::$method(&self, &const_(rhs))
            }
        }
        impl $tr<&Expr<$t>> for $t {
            type Output = Expr<$t>;
            fn $method(self, rhs: &Expr<$t>) -> Self::Output {
                $tr::$method(&const_(self), rhs)
            }
        }
        impl $tr<&Expr<$t>> for Expr<$t> {
            type Output = Expr<$t>;
            fn $method(self, rhs: &Expr<$t>) -> Self::Output {
                $tr::$method(&self, rhs)
            }
        }
        impl $tr<Expr<$t>> for &Expr<$t> {
            type Output = Expr<$t>;
            fn $method(self, rhs: Expr<$t>) -> Self::Output {
                $tr::$method(self, &rhs)
            }
        }
        impl $tr<Expr<$t>> for $t {
            type Output = Expr<$t>;
            fn $method(self, rhs: Expr<$t>) -> Self::Output {
                $tr::$method(&const_(self), &rhs)
            }
        }
        impl $tr for Expr<$t> {
            type Output = Expr<$t>;
            fn $method(self, rhs: Expr<$t>) -> Self::Output {
                $tr::$method(&self, &rhs)
            }
        }
    };
}
macro_rules! impl_common_binop {
    ($t:ty) => {
        impl_binop!($t, AddAssign, add_assign, Add, add);
        impl_binop!($t, SubAssign, sub_assign, Sub, sub);
        impl_binop!($t, MulAssign, mul_assign, Mul, mul);
        impl_binop!($t, DivAssign, div_assign, Div, div);
        impl_binop!($t, RemAssign, rem_assign, Rem, rem);
    };
}
macro_rules! impl_int_binop {
    ($t:ty) => {
        impl_binop!($t, ShlAssign, shl_assign, Shl, shl);
        impl_binop!($t, ShrAssign, shr_assign, Shr, shr);
        impl_binop!($t, BitAndAssign, bitand_assign, BitAnd, bitand);
        impl_binop!($t, BitOrAssign, bitor_assign, BitOr, bitor);
        impl_binop!($t, BitXorAssign, bitxor_assign, BitXor, bitxor);
    };
}

macro_rules! impl_not {
    ($t:ty) => {
        impl Not for Expr<$t> {
            type Output = Expr<$t>;
            fn not(self) -> Self::Output {
                &self ^ &const_(!0)
            }
        }
        impl Not for &Expr<$t> {
            type Output = Expr<$t>;
            fn not(self) -> Self::Output {
                self ^ &const_(!0)
            }
        }
    };
}
macro_rules! impl_neg {
    ($t:ty) => {
        impl Neg for Expr<$t> {
            type Output = Expr<$t>;
            fn neg(self) -> Self::Output {
                const_(0) - &self
            }
        }
        impl Neg for &Expr<$t> {
            type Output = Expr<$t>;
            fn neg(self) -> Self::Output {
                const_(0) - self
            }
        }
    };
}
macro_rules! impl_fneg {
    ($t:ty) => {
        impl Neg for Expr<$t> {
            type Output = Expr<$t>;
            fn neg(self) -> Self::Output {
                current_scope(|s| {
                    let ret = s.call(Func::Neg, &[self.node()], Self::Output::type_());
                    Expr::<$t>::from_node(ret)
                })
            }
        }
    };
}
impl Not for Expr<bool> {
    type Output = Expr<bool>;
    fn not(self) -> Self::Output {
        &self ^ &const_(true)
    }
}
impl_common_binop!(f32);
impl_common_binop!(f64);
impl_common_binop!(i32);
impl_common_binop!(i64);
impl_common_binop!(u32);
impl_common_binop!(u64);

impl_binop!(bool, BitAndAssign, bitand_assign, BitAnd, bitand);
impl_binop!(bool, BitOrAssign, bitor_assign, BitOr, bitor);
impl_binop!(bool, BitXorAssign, bitxor_assign, BitXor, bitxor);
impl_int_binop!(i32);
impl_int_binop!(i64);
impl_int_binop!(u32);
impl_int_binop!(u64);

impl_not!(i32);
impl_not!(i64);
impl_not!(u32);
impl_not!(u64);

impl_neg!(i32);
impl_neg!(i64);
impl_neg!(u32);
impl_neg!(u64);

impl_fneg!(f32);
impl_fneg!(f64);
impl VarCmp for Expr<f32> {}
impl VarCmp for Expr<f64> {}
impl VarCmp for Expr<i32> {}
impl VarCmp for Expr<i64> {}
impl VarCmp for Expr<u32> {}
impl VarCmp for Expr<u64> {}
impl VarCmp for Expr<bool> {}
impl CommonVarOp for Expr<f32> {}
impl CommonVarOp for Expr<f64> {}
impl CommonVarOp for Expr<i32> {}
impl CommonVarOp for Expr<i64> {}
impl CommonVarOp for Expr<u32> {}
impl CommonVarOp for Expr<u64> {}
impl CommonVarOp for Expr<bool> {}

impl From<f64> for Expr<f32> {
    fn from(x: f64) -> Self {
        (x as f32).into()
    }
}

// impl Expr<f32>VarTrait for Expr<f32> {
//     fn from_f64(x: f64) -> Self {
//         const_(x as f32)
//     }
// }

impl IntVarTrait for Expr<i32> {
    fn from_i64(x: i64) -> Self {
        const_(x as i32)
    }
}
impl IntVarTrait for Expr<i64> {
    fn from_i64(x: i64) -> Self {
        const_(x)
    }
}
impl IntVarTrait for Expr<u32> {
    fn from_i64(x: i64) -> Self {
        const_(x as u32)
    }
}
