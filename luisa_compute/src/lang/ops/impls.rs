use crate::lang::types::{ExprType, ValueType};

use super::*;

impl<X: Linear> Expr<X> {
    pub fn as_<Y: Linear>(self) -> Expr<Y>
    where
        Y::Scalar: CastFrom<X::Scalar>,
    {
        assert_eq!(
            X::N,
            Y::N,
            "Cannot cast between scalars/vectors of different dimensions."
        );
        Func::Cast.call(self)
    }
    pub fn cast<S: VectorElement>(self) -> Expr<X::WithScalar<S>>
    where
        S: CastFrom<X::Scalar>,
    {
        self.as_::<<X as Linear>::WithScalar<S>>()
    }
}

macro_rules! impl_ops_trait {
    (
        [$($bounds:tt)*] $TraitExpr:ident [$TraitThis:ident] for $T:ty where [$($where:tt)*] {
            $(
                fn $fn:ident [$fn_this:ident] ($sl:ident, $($arg:ident),*) { $body:expr }
            )*
        }
    ) => {
        impl<$($bounds)*> $TraitThis for $T where $($where)* {
            $(
                fn $fn_this($sl, $($arg: Self),*) -> Self {
                   $body
                }
            )*
        }
    };
    (
        [$($bounds:tt)*] $TraitExpr:ident [$TraitThis:ident] for $T:ty where [$($where:tt)*] {
            type Output = $Output:ty;
            $(
                fn $fn:ident [$fn_this:ident] ($sl:ident, $($arg:ident),*) { $body:expr }
            )*
        }
    ) => {
        impl<$($bounds)*> $TraitThis for $T where $($where)* {
            type Output = $Output;
            $(
                fn $fn_this($sl, $($arg: Self),*) -> Self::Output {
                   $body
                }
            )*
        }
    }
}
macro_rules! impl_simple_binop {
    (
        [$($bounds:tt)*] $TraitExpr:ident [$TraitThis:ident] for $T:ty where [$($where:tt)*]: $fn:ident [$fn_this:ident] ($func:ident)
    ) => {
        impl_ops_trait!([$($bounds)*] $TraitExpr [$TraitThis] for $T where [$($where)*] {
            fn $fn[$fn_this](self, other) { Func::$func.call2(self, other) }
        });
    }
}

impl_ops_trait!([X: Linear] MinMaxExpr[MinMaxThis] for Expr<X> where [X::Scalar: Numeric] {
    type Output = Expr<X::WithScalar<X::Scalar>>;

    fn max_expr[_max_expr](self, other) { Func::Max.call2(self, other) }
    fn min_expr[_min_expr](self, other) { Func::Min.call2(self, other) }
});

impl_ops_trait!([X: Linear] ClampExpr[ClampThis] for Expr<X> where [X::Scalar: Numeric] {
    fn clamp[_clamp](self, min, max) { Func::Clamp.call3(self, min, max) }
});

impl<X: Linear> AbsExpr for Expr<X>
where
    X::Scalar: Signed,
{
    fn abs(&self) -> Self {
        Func::Abs.call(self.clone())
    }
}

impl_ops_trait!([X: Linear] EqExpr[EqThis] for Expr<X> where [X::Scalar: VectorElement] {
    type Output = Expr<X::WithScalar<bool>>;

    fn eq[_eq](self, other) { Func::Eq.call2(self, other) }
    fn ne[_ne](self, other) { Func::Ne.call2(self, other) }
});

impl_ops_trait!([X: Linear] CmpExpr[CmpThis] for Expr<X> where [X::Scalar: Numeric] {
    type Output = Expr<X::WithScalar<bool>>;

    fn lt[_lt](self, other) { Func::Lt.call2(self, other) }
    fn le[_le](self, other) { Func::Le.call2(self, other) }
    fn gt[_gt](self, other) { Func::Gt.call2(self, other) }
    fn ge[_ge](self, other) { Func::Ge.call2(self, other) }
});

impl_simple_binop!([X: Linear] AddExpr[AddThis] for Expr<X> where [X::Scalar: Numeric]: add[_add](Add));
impl_simple_binop!([X: Linear] SubExpr[SubThis] for Expr<X> where [X::Scalar: Numeric]: sub[_sub](Sub));
impl_simple_binop!([X: Linear] MulExpr[MulThis] for Expr<X> where [X::Scalar: Numeric]: mul[_mul](Mul));
impl_simple_binop!([X: Linear] DivExpr[DivThis] for Expr<X> where [X::Scalar: Numeric]: div[_div](Div));
impl_simple_binop!([X: Linear] RemExpr[RemThis] for Expr<X> where [X::Scalar: Numeric]: rem[_rem](Rem));
impl_simple_binop!([X: Linear] BitAndExpr[BitAndThis] for Expr<X> where [X::Scalar: Integral]: bitand[_bitand](BitAnd));
impl_simple_binop!([X: Linear] BitOrExpr[BitOrThis] for Expr<X> where [X::Scalar: Integral]: bitor[_bitor](BitOr));
impl_simple_binop!([X: Linear] BitXorExpr[BitXorThis] for Expr<X> where [X::Scalar: Integral]: bitxor[_bitxor](BitXor));
impl_simple_binop!([X: Linear] ShlExpr[ShlThis] for Expr<X> where [X::Scalar: Integral]: shl[_shl](Shl));
impl_simple_binop!([X: Linear] ShrExpr[ShrThis] for Expr<X> where [X::Scalar: Integral]: shr[_shr](Shr));

impl<X: Linear> Neg for Expr<X>
where
    X::Scalar: Signed,
{
    type Output = Self;
    fn neg(self) -> Self {
        Func::Neg.call(self)
    }
}
impl<X: Linear> Not for Expr<X>
where
    X::Scalar: Integral,
{
    type Output = Self;
    fn not(self) -> Self {
        Func::BitNot.call(self)
    }
}

impl<X: Linear> IntExpr for Expr<X>
where
    X::Scalar: Integral + Numeric,
{
    fn rotate_left(&self, n: Expr<u32>) -> Self {
        Func::RotRight.call2(self.clone(), n)
    }
    fn rotate_right(&self, n: Expr<u32>) -> Self {
        Func::RotLeft.call2(self.clone(), n)
    }
}

macro_rules! impl_simple_fns {
    ($($fname:ident => $func:ident),+) => {$(
        fn $fname(&self) -> Self {
            Func::$func.call(self.clone())
        }
    )+};
}

impl<X: Linear> FloatExpr for Expr<X>
where
    X::Scalar: Floating,
{
    type Bool = Expr<X::WithScalar<bool>>;
    impl_simple_fns! {
        ceil => Ceil,
        floor => Floor,
        round => Round,
        trunc => Trunc,
        sqrt => Sqrt,
        rsqrt => Rsqrt,
        fract => Fract,
        saturate => Saturate,
        sin => Sin,
        cos => Cos,
        tan => Tan,
        asin => Asin,
        acos => Acos,
        atan => Atan,
        sinh => Sinh,
        cosh => Cosh,
        tanh => Tanh,
        asinh => Asinh,
        acosh => Acosh,
        atanh => Atanh,
        exp => Exp,
        exp2 => Exp2,
        ln => Log,
        log2 => Log2,
        log10 => Log10
    }
    fn is_finite(&self) -> Self::Bool {
        !self.is_infinite().bitand(!self.is_nan())
    }
    fn is_infinite(&self) -> Self::Bool {
        Func::IsInf.call(self.clone())
    }
    fn is_nan(&self) -> Self::Bool {
        Func::IsNan.call(self.clone())
    }
    fn sqr(&self) -> Self {
        self.clone().mul(self.clone())
    }
    fn cube(&self) -> Self {
        self.clone().mul(self.clone()).mul(self.clone())
    }
    fn recip(&self) -> Self {
        todo!()
        // 1.0 / self.clone()
    }
    fn sin_cos(&self) -> (Self, Self) {
        (self.sin(), self.cos())
    }
}
impl_ops_trait!([X: Linear] FloatMulAddExpr[FloatMulAddThis] for Expr<X> where [X::Scalar: Floating] {
    fn mul_add[_mul_add](self, a, b) { Func::Fma.call3(self, a, b) }
});

impl_ops_trait!([X: Linear] FloatCopySignExpr[FloatCopySignThis] for Expr<X> where [X::Scalar: Floating] {
    fn copy_sign[_copy_sign](self, sign) { Func::Copysign.call2(self, sign) }
});

impl_ops_trait!([X: Linear] FloatStepExpr[FloatStepThis] for Expr<X> where [X::Scalar: Floating] {
    fn step[_step](self, edge) { Func::Step.call2(edge, self) }
});

impl_ops_trait!([X: Linear] FloatSmoothStepExpr[FloatSmoothStepThis] for Expr<X> where [X::Scalar: Floating] {
    fn smooth_step[_smooth_step](self, edge0, edge1) { Func::SmoothStep.call3(edge0, edge1, self) }
});

impl_ops_trait!([X: Linear] FloatArcTan2Expr[FloatArcTan2This] for Expr<X> where [X::Scalar: Floating] {
    fn atan2[_atan2](self, other) { Func::Atan2.call2(self, other) }
});

impl_ops_trait!([X: Linear] FloatLogExpr[FloatLogThis] for Expr<X> where [X::Scalar: Floating] {
    fn log[_log](self, base) { self.ln().div(base.ln()) }
});

impl_ops_trait!([X: Linear] FloatPowfExpr[FloatPowfThis] for Expr<X> where [X::Scalar: Floating] {
    fn powf[_powf](self, exponent) { Func::Powf.call2(self, exponent) }
});

impl<X: Linear, Y: Linear<Scalar = i32>> FloatPowiExpr<Expr<Y>> for Expr<X>
where
    X::Scalar: Floating,
{
    type Output = Self;

    fn powi(self, exponent: Expr<Y>) -> Self::Output {
        Func::Powi.call2(self, exponent)
    }
}

impl_ops_trait!([X: Linear] FloatLerpExpr[FloatLerpThis] for Expr<X> where [X::Scalar: Floating] {
    fn lerp[_lerp](self, other, frac) { Func::Lerp.call3(self, other, frac) }
});

// Traits for `track!`.

impl<T: Sized> StoreMaybeExpr<T> for &mut T {
    fn store(self, value: T) {
        *self = value;
    }
}
impl<V: Value, E: AsExpr<Value = V>> StoreMaybeExpr<E> for &Var<V> {
    fn store(self, value: E) {
        crate::lang::_store(self, &value.as_expr());
    }
}
impl<V: Value, E: AsExpr<Value = V>> StoreMaybeExpr<E> for Var<V> {
    fn store(self, value: E) {
        crate::lang::_store(&self, &value.as_expr());
    }
}

impl<R> SelectMaybeExpr<R> for bool {
    fn if_then_else(self, on: impl FnOnce() -> R, off: impl FnOnce() -> R) -> R {
        if self {
            on()
        } else {
            off()
        }
    }
    fn select(self, on: R, off: R) -> R {
        if self {
            on
        } else {
            off
        }
    }
}
impl<R: Aggregate> SelectMaybeExpr<R> for Expr<bool> {
    fn if_then_else(self, on: impl FnOnce() -> R, off: impl FnOnce() -> R) -> R {
        crate::lang::control_flow::if_then_else(self, on, off)
    }
    fn select(self, on: R, off: R) -> R {
        crate::lang::control_flow::select(self, on, off)
    }
}

impl ActivateMaybeExpr for bool {
    fn activate(self, then: impl FnOnce()) {
        if self {
            then()
        }
    }
}
impl ActivateMaybeExpr for Expr<bool> {
    fn activate(self, then: impl FnOnce()) {
        crate::lang::control_flow::if_then_else(self, then, || {})
    }
}

impl LoopMaybeExpr for bool {
    fn while_loop(mut cond: impl FnMut() -> Self, mut body: impl FnMut()) {
        while cond() {
            body()
        }
    }
}

impl LoopMaybeExpr for Expr<bool> {
    fn while_loop(cond: impl FnMut() -> Self, body: impl FnMut()) {
        crate::lang::control_flow::generic_loop(cond, body, || {})
    }
}

impl LazyBoolMaybeExpr<bool, ValueType> for bool {
    type Bool = bool;
    fn and(self, other: impl FnOnce() -> bool) -> bool {
        self && other()
    }
    fn or(self, other: impl FnOnce() -> bool) -> bool {
        self || other()
    }
}
impl LazyBoolMaybeExpr<Expr<bool>, ExprType> for bool {
    type Bool = Expr<bool>;
    fn and(self, other: impl FnOnce() -> Expr<bool>) -> Self::Bool {
        if self {
            other()
        } else {
            false.expr()
        }
    }
    fn or(self, other: impl FnOnce() -> Expr<bool>) -> Self::Bool {
        if self {
            true.expr()
        } else {
            other()
        }
    }
}
impl LazyBoolMaybeExpr<bool, ExprType> for Expr<bool> {
    type Bool = Expr<bool>;
    fn and(self, other: impl FnOnce() -> bool) -> Self::Bool {
        if other() {
            self
        } else {
            false.expr()
        }
    }
    fn or(self, other: impl FnOnce() -> bool) -> Self::Bool {
        if other() {
            true.expr()
        } else {
            self
        }
    }
}
impl LazyBoolMaybeExpr<Expr<bool>, ExprType> for Expr<bool> {
    type Bool = Expr<bool>;
    fn and(self, other: impl FnOnce() -> Expr<bool>) -> Self::Bool {
        crate::lang::control_flow::if_then_else(self, other, || false.expr())
    }
    fn or(self, other: impl FnOnce() -> Expr<bool>) -> Self::Bool {
        crate::lang::control_flow::if_then_else(self, || true.expr(), other)
    }
}
