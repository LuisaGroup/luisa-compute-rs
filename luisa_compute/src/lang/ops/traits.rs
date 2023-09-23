use crate::lang::types::{ExprType, TrackingType, ValueType};

use super::*;

// The double trait implementation is necessary as the compiler infinite loops
// when trying to resolve the Expr<T>: SpreadOps<Expr<Vector<T, N>>> bound.
macro_rules! ops_trait {
    (
        $TraitExpr:ident<$($T:ident),*> [ $TraitThis:ident, $TraitOrig:ident$(($OrigOutput:path))? => $TraitMaybe:ident ] {
            $(type $o:ident;)?
            $(
                fn $fn:ident [$fn_this:ident, $orig_fn:expr => $fn_maybe:ident] ($self:ident, $($arg:ident: $S:ident),*);
            )+
        }
    ) => {
        ops_trait!(
            @XVARS(X, $TraitExpr<$($T),*>, $TraitOrig<$($T),*>)
            $TraitExpr<$($T),*> [ $TraitThis, $TraitOrig$(($OrigOutput))? => $TraitMaybe ] {
                $(type $o;)?
                $(
                    fn $fn [$fn_this, $orig_fn => $fn_maybe] ($self, $($arg: $S),*);
                )+
            }
        );
    };
    (
        @XVARS($X:ident, $EXPANDED_EXPR:path, $EXPANDED_ORIG:path)

        $TraitExpr:ident<$($T:ident),*> [ $TraitThis:ident, $TraitOrig:ident => $TraitMaybe:ident ] {
            $(type $o:ident;)?
            $(
                fn $fn:ident [$fn_this:ident, $orig_fn:expr => $fn_maybe:ident] ($self:ident, $($arg:ident: $S:ident),*);
            )+
        }
    ) => {
        ops_trait!(
            @XVARS($X, $EXPANDED_EXPR, $EXPANDED_ORIG)
            $TraitExpr<$($T),*> [ $TraitThis, $TraitOrig(<$X as $EXPANDED_ORIG>::Output) => $TraitMaybe ] {
                $(type $o;)?
                $(
                    fn $fn [$fn_this, $orig_fn => $fn_maybe] ($self, $($arg: $S),*);
                )+
            }
        );
    };
    (
        @XVARS($X:ident, $EXPANDED_EXPR:path, $EXPANDED_ORIG:path)

        $TraitExpr:ident<$($T:ident),*> [ $TraitThis:ident, $TraitOrig:ident($($OrigOutput:tt)*) => $TraitMaybe:ident ] {
            $(type $o:ident;)?
            $(
                fn $fn:ident [$fn_this:ident, $orig_fn:expr => $fn_maybe:ident] ($self:ident, $($arg:ident: $S:ident),*);
            )+
        }
    ) => {
        ops_trait!($TraitExpr <$($T),*> [ $TraitThis ] {
            $(type $o;)?
            $(
                fn $fn [$fn_this] ($self, $($arg: $S),*);
            )+
        });
        pub trait $TraitMaybe<$($T,)* Ty: TrackingType> {
            type Output;
            $(
                fn $fn_maybe($self, $($arg: $S),*) -> Self::Output;
            )*
        }
        impl<$X $(,$T)*> $TraitMaybe<$($T,)* ExprType> for $X where $X: $EXPANDED_EXPR {
            type Output = <$X as $EXPANDED_EXPR>::Output;
            $(
                fn $fn_maybe($self, $($arg: $S),*) -> Self::Output {
                    <$X as $EXPANDED_EXPR>::$fn($self, $($arg),*)
                }
            )*
        }
        impl<$X $(,$T)*> $TraitMaybe<$($T,)* ValueType> for $X where $X: $EXPANDED_ORIG {
            type Output = $($OrigOutput)*;
            $(
                fn $fn_maybe($self, $($arg: $S),*) -> Self::Output {
                    $orig_fn
                }
            )*
        }
    };
    (
        $TraitExpr:ident<$($T:ident),*> [ $TraitThis:ident ] {
            $(
                fn $fn:ident [$fn_this:ident] (self, $($arg:ident: $S:ident),*);
            )+
        }
    ) => {
        pub trait $TraitThis {
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
        $TraitExpr:ident<$($T:ident),*> [ $TraitThis:ident ] {
            type Output;
            $(
                fn $fn:ident [$fn_this:ident] (self, $($arg:ident: $S:ident),*);
            )+
        }
    ) => {
        pub trait $TraitThis {
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

macro_rules! simple_binop_trait {
    ($TraitExpr:ident [$TraitThis:ident, $TraitOrig:ident => $TraitMaybe:ident]: $fn:ident [$fn_this: ident, $fn_orig:ident => $fn_maybe:ident]) => {
        ops_trait!(
            $TraitExpr<T>[$TraitThis, $TraitOrig => $TraitMaybe] {
                fn $fn[$fn_this, <Self as $TraitOrig<_>>::$fn_orig(self, rhs) => $fn_maybe](self, rhs: T);
            }
        );
    }
}

macro_rules! assignop_trait {
    ($TraitExpr:ident [$TraitOrig:ident => $TraitMaybe:ident]: $fn:ident [$fn_orig:ident => $fn_maybe:ident]) => {
        pub trait $TraitExpr<T = Self> {
            fn $fn(self, other: T);
        }
        pub trait $TraitMaybe<T, Ty: TrackingType> {
            fn $fn_maybe(self, other: T);
        }
        impl<X, T> $TraitMaybe<T, ExprType> for X
        where
            X: $TraitExpr<T>,
        {
            fn $fn_maybe(self, other: T) {
                <X as $TraitExpr<T>>::$fn(self, other)
            }
        }
        impl<X: DerefMut, T> $TraitMaybe<T, ValueType> for &mut X
        where
            X::Target: $TraitOrig<T>,
        {
            fn $fn_maybe(self, other: T) {
                <X::Target as $TraitOrig<T>>::$fn_orig(self.deref_mut(), other)
            }
        }
    };
}

ops_trait!(MinMaxExpr<T>[MinMaxThis] {
    type Output;

    fn max_[_max_](self, other: T);
    fn min_[_min_](self, other: T);
});

ops_trait!(ClampExpr<A, B>[ClampThis] {
    fn clamp[_clamp](self, min: A, max: B);
});

pub trait AbsExpr {
    fn abs(&self) -> Self;
}

ops_trait!(EqExpr<T>[EqThis, PartialEq(bool) => EqMaybeExpr] {
    type Output;

    fn eq[_eq, self == other => __eq](self, other: T);
    fn ne[_ne, self != other => __ne](self, other: T);
});

ops_trait!(CmpExpr<T>[CmpThis, PartialOrd(bool) => CmpMaybeExpr] {
    type Output;

    fn lt[_lt, self < other => __lt](self, other: T);
    fn le[_le, self <= other => __le](self, other: T);
    fn gt[_gt, self > other => __gt](self, other: T);
    fn ge[_ge, self >= other => __ge](self, other: T);
});

simple_binop_trait!(AddExpr[AddThis, Add => AddMaybeExpr]: add[_add, add => __add]);
simple_binop_trait!(SubExpr[SubThis, Sub => SubMaybeExpr]: sub[_sub, sub => __sub]);
simple_binop_trait!(MulExpr[MulThis, Mul => MulMaybeExpr]: mul[_mul, mul => __mul]);
simple_binop_trait!(DivExpr[DivThis, Div => DivMaybeExpr]: div[_div, div => __div]);
simple_binop_trait!(RemExpr[RemThis, Rem => RemMaybeExpr]: rem[_rem, rem => __rem]);
simple_binop_trait!(BitAndExpr[BitAndThis, BitAnd => BitAndMaybeExpr]: bitand[_bitand, bitand => __bitand]);
simple_binop_trait!(BitOrExpr[BitOrThis, BitOr => BitOrMaybeExpr]: bitor[_bitor, bitor => __bitor]);
simple_binop_trait!(BitXorExpr[BitXorThis, BitXor => BitXorMaybeExpr]: bitxor[_bitxor, bitxor => __bitxor]);
simple_binop_trait!(ShlExpr[ShlThis, Shl => ShlMaybeExpr]: shl[_shl, shl => __shl]);
simple_binop_trait!(ShrExpr[ShrThis, Shr => ShrMaybeExpr]: shr[_shr, shr => __shr]);
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

pub trait ReduceExpr: Sized {
    type Output;
    fn reduce_sum(&self) -> Self::Output;
    fn reduce_prod(&self) -> Self::Output;
    fn reduce_min(&self) -> Self::Output;
    fn reduce_max(&self) -> Self::Output;
}
pub trait NormExpr: Sized {
    type Output;
    fn norm(&self) -> Self::Output;
    fn norm_squared(&self) -> Self::Output;
    fn normalize(&self) -> Self;
    fn length(&self) -> Self::Output {
        self.norm()
    }
    fn length_squared(&self) -> Self::Output {
        self.norm_squared()
    }
}
pub trait OuterProductExpr: Sized {
    type Value;
    type Output;
    fn outer_product(&self, other: impl AsExpr<Value = Self::Value>) -> Self::Output;
}
pub trait DotExpr: Sized {
    type Value;
    type Output;
    fn dot(&self, other: impl AsExpr<Value = Self::Value>) -> Self::Output;
}
pub trait CrossExpr: Sized {
    type Value;
    type Output;
    fn cross(&self, other: impl AsExpr<Value = Self::Value>) -> Self::Output;
}
pub trait MatExpr: Sized {
    type Scalar;
    type Value;
    fn comp_mul(&self, other: impl AsExpr<Value = Self::Value>) -> Self;
    fn transpose(&self) -> Self;
    fn determinant(&self) -> Self::Scalar;
    fn inverse(&self) -> Self;
}
pub trait VectorSelectExpr: Sized {
    fn select<X: Linear>(self, on: impl AsExpr<Value = X>, off: impl AsExpr<Value = X>) -> Self;
}

pub trait ArrayNewExpr<T: Value, const N: usize>: Value {
    fn from_elems_expr(elems: [Expr<T>; N]) -> Expr<Self>;
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

assignop_trait!(AddAssignExpr[AddAssign => AddAssignMaybeExpr]: add_assign[add_assign => __add_assign]);
assignop_trait!(SubAssignExpr[SubAssign => SubAssignMaybeExpr]: sub_assign[sub_assign => __sub_assign]);
assignop_trait!(MulAssignExpr[MulAssign => MulAssignMaybeExpr]: mul_assign[mul_assign => __mul_assign]);
assignop_trait!(DivAssignExpr[DivAssign => DivAssignMaybeExpr]: div_assign[div_assign => __div_assign]);
assignop_trait!(RemAssignExpr[RemAssign => RemAssignMaybeExpr]: rem_assign[rem_assign => __rem_assign]);
assignop_trait!(BitAndAssignExpr[BitAndAssign => BitAndAssignMaybeExpr]: bitand_assign[bitand_assign => __bitand_assign]);
assignop_trait!(BitOrAssignExpr[BitOrAssign => BitOrAssignMaybeExpr]: bitor_assign[bitor_assign => __bitor_assign]);
assignop_trait!(BitXorAssignExpr[BitXorAssign => BitXorAssignMaybeExpr]: bitxor_assign[bitxor_assign => __bitxor_assign]);
assignop_trait!(ShlAssignExpr[ShlAssign => ShlAssignMaybeExpr]: shl_assign[shl_assign => __shl_assign]);
assignop_trait!(ShrAssignExpr[ShrAssign => ShrAssignMaybeExpr]: shr_assign[shr_assign => __shr_assign]);

// Traits for track!.

pub trait StoreMaybeExpr<V> {
    fn __store(self, value: V);
}

pub trait SelectMaybeExpr<R> {
    fn if_then_else(self, on: impl Fn() -> R, off: impl Fn() -> R) -> R;
    fn select(self, on: R, off: R) -> R;
}

pub trait ActivateMaybeExpr {
    fn activate(self, then: impl Fn());
}

pub trait LoopMaybeExpr {
    fn while_loop(cond: impl FnMut() -> Self, body: impl FnMut());
}

pub trait LazyBoolMaybeExpr<T, Ty: TrackingType> {
    type Bool;
    fn and(self, other: impl Fn() -> T) -> Self::Bool;
    fn or(self, other: impl Fn() -> T) -> Self::Bool;
}
