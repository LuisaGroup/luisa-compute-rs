use super::*;

pub trait MinMaxExpr<T = Self> {
    type Output;

    fn max(self, other: T) -> Self::Output;
    fn min(self, other: T) -> Self::Output;
}

pub trait ClampExpr<A = Self, B = Self> {
    type Output;

    fn clamp(self, min: A, max: B) -> Self::Output;
}

pub trait AbsExpr {
    fn abs(&self) -> Self;
}

pub trait EqExpr<T = Self> {
    type Output;

    fn eq(self, other: T) -> Self::Output;
    fn ne(self, other: T) -> Self::Output;
}

pub trait CmpExpr<T = Self>: EqExpr<T> {
    fn lt(self, other: T) -> Self::Output;
    fn le(self, other: T) -> Self::Output;
    fn gt(self, other: T) -> Self::Output;
    fn ge(self, other: T) -> Self::Output;
}

pub trait IntExpr {
    fn rotate_right(&self, n: Expr<u32>) -> Self;
    fn rotate_left(&self, n: Expr<u32>) -> Self;
}

pub trait FloatExpr: Sized {
    type Bool;

    fn ceil(&self) -> Self;
    fn floor(&self) -> Self;
    fn round(&self) -> Self;
    fn trunc(&self) -> Self;
    fn sqrt(&self) -> Self;
    fn rsqrt(&self) -> Self;
    fn fract(&self) -> Self;
    fn saturate(&self) -> Self;
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
    fn tan(&self) -> Self;
    fn asin(&self) -> Self;
    fn acos(&self) -> Self;
    fn atan(&self) -> Self;
    fn sinh(&self) -> Self;
    fn cosh(&self) -> Self;
    fn tanh(&self) -> Self;
    fn asinh(&self) -> Self;
    fn acosh(&self) -> Self;
    fn atanh(&self) -> Self;
    fn exp(&self) -> Self;
    fn exp2(&self) -> Self;
    fn is_finite(&self) -> Self::Bool;
    fn is_infinite(&self) -> Self::Bool;
    fn is_nan(&self) -> Self::Bool;
    fn ln(&self) -> Self;
    fn log2(&self) -> Self;
    fn log10(&self) -> Self;
    fn sqr(&self) -> Self;
    fn cube(&self) -> Self;
    fn recip(&self) -> Self;
    fn sin_cos(&self) -> (Self, Self);
}
pub trait FloatMulAddExpr<A = Self, B = Self> {
    type Output;

    fn mul_add(self, a: A, b: B) -> Self::Output;
}
pub trait FloatCopySignExpr<T = Self> {
    type Output;

    fn copy_sign(self, sign: T) -> Self::Output;
}
pub trait FloatStepExpr<T = Self> {
    type Output;

    fn step(self, edge: T) -> Self::Output;
}
pub trait FloatSmoothStepExpr<T = Self, S = Self> {
    type Output;

    fn smooth_step(self, edge0: T, edge1: S) -> Self::Output;
}
pub trait FloatArcTan2Expr<T = Self> {
    type Output;

    fn atan2(self, other: T) -> Self::Output;
}
pub trait FloatLogExpr<T = Self> {
    type Output;

    fn log(self, base: T) -> Self::Output;
}
pub trait FloatPowfExpr<T = Self> {
    type Output;

    fn powf(self, exponent: T) -> Self::Output;
}
pub trait FloatPowiExpr<T> {
    type Output;

    fn powi(self, exponent: T) -> Self::Output;
}
pub trait FloatLerpExpr<T = Self, S = Self> {
    type Output;

    fn lerp(self, other: T, frac: S) -> Self::Output;
}

pub trait StoreExpr<V> {
    fn store(self, value: V);
}

pub trait SwitchExpr<R> {
    fn switch(self, on: impl FnOnce() -> R, off: impl FnOnce() -> R) -> R;
}

pub trait ActivateExpr {
    fn activate(self, then: impl FnOnce());
}

pub trait LoopExpr {
    fn while_loop(cond: impl FnMut() -> Self, body: impl FnMut());
}

pub trait LazyBoolExpr<T = Self> {
    type Bool;
    fn and(self, other: impl FnOnce() -> T) -> Self::Bool;
    fn or(self, other: impl FnOnce() -> T) -> Self::Bool;
}
