use super::*;

macro_rules! ops_trait {
    (
        $TraitExpr:ident<$($T:ident),*> [ $TraitThis:ident] {
            $(
                fn $fn:ident [$fn_this:ident] (self, $($arg:ident: $S:ident),*);
            )+
        }
    ) => {
        pub(crate) trait $TraitThis {
            $(
                fn $fn_this(self, $($arg: Self),*) -> Self;
            )*
        }
        pub trait $TraitExpr<$($T = Self),*> {
            type Output;

            $(
                fn $fn(self, $($arg: $S),*) -> Self::Output;
            )*
        }
    };
    (
        $TraitExpr:ident<$($T:ident),*> [ $TraitThis:ident] {
            type Output;
            $(
                fn $fn:ident [$fn_this:ident] (self, $($arg:ident: $S:ident),*);
            )+
        }
    ) => {
        pub(crate) trait $TraitThis {
            type Output;
            $(
                fn $fn_this(self, $($arg: Self),*) -> Self::Output;
            )*
        }
        pub trait $TraitExpr<$($T = Self),*> {
            type Output;

            $(
                fn $fn(self, $($arg: $S),*) -> Self::Output;
            )*
        }
    }
}

ops_trait!(MinMaxExpr<T>[MinMaxThis] {
    fn max[_max](self, other: T);
    fn min[_min](self, other: T);
});

ops_trait!(ClampExpr<A, B>[ClampThis] {
    fn clamp[_clamp](self, min: A, max: B);
});

pub trait AbsExpr {
    fn abs(&self) -> Self;
}

ops_trait!(EqExpr<T>[EqThis] {
    type Output;

    fn eq[_eq](self, other: T);
    fn ne[_ne](self, other: T);
});

ops_trait!(CmpExpr<T>[CmpThis] {
    type Output;

    fn lt[_lt](self, other: T);
    fn le[_le](self, other: T);
    fn gt[_gt](self, other: T);
    fn ge[_ge](self, other: T);
});

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

ops_trait!(FloatMulAddExpr<A, B>[FloatMulAddThis] {
    fn mul_add[_mul_add](self, a: A, b: B);
});

ops_trait!(FloatCopySignExpr<T>[FloatCopySignThis] {
    fn copy_sign[_copy_sign](self, sign: T);
});

ops_trait!(FloatStepExpr<T>[FloatStepThis] {
    fn step[_step](self, edge: T);
});

ops_trait!(FloatSmoothStepExpr<T, S>[FloatSmoothStepThis] {
    fn smooth_step[_smooth_step](self, edge0: T, edge1: S);
});

ops_trait!(FloatArcTan2Expr<T>[FloatArcTan2This] {
    fn atan2[_atan2](self, other: T);
});

ops_trait!(FloatLogExpr<T>[FloatLogThis] {
    fn log[_log](self, base: T);
});

ops_trait!(FloatPowfExpr<T>[FloatPowfThis] {
    fn powf[_powf](self, exponent: T);
});

pub trait FloatPowiExpr<T> {
    type Output;

    fn powi(self, exponent: T) -> Self::Output;
}

ops_trait!(FloatLerpExpr<A, B>[FloatLerpThis] {
    fn lerp[_lerp](self, other: A, frac: B);
});

pub trait StoreMaybeExpr<V> {
    fn store(self, value: V);
}

pub trait SelectMaybeExpr<R> {
    fn if_then_else(self, on: impl FnOnce() -> R, off: impl FnOnce() -> R) -> R;
    fn select(self, on: R, off: R) -> R;
}

pub trait ActivateMaybeExpr {
    fn activate(self, then: impl FnOnce());
}

pub trait LoopMaybeExpr {
    fn while_loop(cond: impl FnMut() -> Self, body: impl FnMut());
}

pub trait LazyBoolMaybeExpr<T = Self> {
    type Bool;
    fn and(self, other: impl FnOnce() -> T) -> Self::Bool;
    fn or(self, other: impl FnOnce() -> T) -> Self::Bool;
}

pub trait EqMaybeExpr<T, const EXPR: bool> {
    type Bool;
    fn __eq(self, other: T) -> Self::Bool;
    fn __ne(self, other: T) -> Self::Bool;
}

pub trait CmpMaybeExpr<T, const EXPR: bool> {
    type Bool;
    fn __lt(self, other: T) -> Self::Bool;
    fn __le(self, other: T) -> Self::Bool;
    fn __gt(self, other: T) -> Self::Bool;
    fn __ge(self, other: T) -> Self::Bool;
}
