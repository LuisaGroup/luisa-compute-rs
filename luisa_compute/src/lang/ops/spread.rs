use crate::lang::types::core::PrimitiveVar;
use crate::lang::types::{ExprProxy, VarProxy};

use super::*;
use traits::*;

pub trait SpreadOps<Other> {
    type Join: Value;
    fn lift_self(x: Self) -> Expr<Self::Join>;
    fn lift_other(x: Other) -> Expr<Self::Join>;
}

macro_rules! impl_spread {
    (@sym [$($bounds:tt)*] $T:ty : |$x:ident| $f:expr, $S:ty : |$y:ident| $g:expr => Expr<$J:ty>) => {
        impl<$($bounds)*> SpreadOps<$S> for $T {
            type Join = $J;
            fn lift_self($x: $T) -> Expr<Self::Join> {
                $f
            }
            fn lift_other($y: $S) -> Expr<Self::Join> {
                $g
            }
        }
    };
    ([$($bounds:tt)*] $T:ty : |$x:ident| $f:expr, $S:ty : |$y:ident| $g:expr => Expr<$J:ty>) => {
        impl<$($bounds)*> SpreadOps<$S> for $T {
            type Join = $J;
            fn lift_self($x: $T) -> Expr<Self::Join> {
                $f
            }
            fn lift_other($x: $S) -> Expr<Self::Join> {
                $g
            }
        }
        impl<$($bounds)*> SpreadOps<$T> for $S {
            type Join = $J;
            fn lift_self($y: $S) -> Expr<Self::Join> {
                $g
            }
            fn lift_other($y: $T) -> Expr<Self::Join> {
                $f
            }
        }
    };
}

macro_rules! call_linear_fn_spread {
    ($f:ident [$($bounds:tt)*]($T:ty)) => {
        $f!([$($bounds)*] $T: |x| x.expr(), Expr<$T>: |x| x => Expr<$T>);
        $f!(['a, $($bounds)*] &'a $T: |x| x.expr(), Expr<$T>: |x| x => Expr<$T>);
        $f!(['b, $($bounds)*] $T: |x| x.expr(), &'b Expr<$T>: |x| x.clone() => Expr<$T>);
        $f!(['a, 'b, $($bounds)*] &'a $T: |x| x.expr(), &'b Expr<$T>: |x| x.clone() => Expr<$T>);

        $f!(['b, $($bounds)*] Expr<$T>: |x| x, &'b Expr<$T>: |x| x.clone() => Expr<$T>);
        $f!(['b, $($bounds)*] Var<$T>: |x| x.load(), &'b Var<$T>: |x| x.load() => Expr<$T>);
        $f!(@sym ['a, 'b, $($bounds)*] &'a Expr<$T>: |x| x.clone(), &'b Expr<$T>: |x| x.clone() => Expr<$T>);
        $f!(@sym [$($bounds)*] Var<$T>: |x| x.load(), Var<$T>: |x| x.load() => Expr<$T>);
        $f!(@sym ['a, 'b, $($bounds)*] &'a Var<$T>: |x| x.load(), &'b Var<$T>: |x| x.load() => Expr<$T>);

        $f!([$($bounds)*] $T: |x| x.expr(), Var<$T>: |x| x.load() => Expr<$T>);
        $f!(['a, $($bounds)*] &'a $T: |x| x.expr(), Var<$T>: |x| x.load() => Expr<$T>);
        $f!(['b, $($bounds)*] $T: |x| x.expr(), &'b Var<$T>: |x| x.load() => Expr<$T>);
        $f!(['a, 'b, $($bounds)*] &'a $T: |x| x.expr(), &'b Var<$T>: |x| x.load() => Expr<$T>);

        $f!(['a, $($bounds)*] &'a Expr<$T>: |x| x.clone(), Var<$T>: |x| x.load() => Expr<$T>);
        $f!(['a, 'b, $($bounds)*] &'a Expr<$T>: |x| x.clone(), &'b Var<$T>: |x| x.load() => Expr<$T>);
        $f!([$($bounds)*] Expr<$T>: |x| x, Var<$T>: |x| x.load() => Expr<$T>);
        $f!(['b, $($bounds)*] Expr<$T>: |x| x, &'b Var<$T>: |x| x.load() => Expr<$T>);
    };
    ($f:ident [$T:ident]) => {
        call_linear_fn_spread!($f [$T: Linear]($T));
    }
}

call_linear_fn_spread!(impl_spread[T]);

macro_rules! call_vector_fn_spread {
    ($f:ident [$($bounds:tt)*]($N:tt, $T:ty) $Vt:ty, $Vsplat:path) => {
        $f!([$($bounds)*] $T: |x| $Vsplat(x), Expr<$Vt>: |x| x => Expr<$Vt>);
        $f!(['a, $($bounds)*] &'a $T: |x| $Vsplat(*x), Expr<$Vt>: |x| x => Expr<$Vt>);
        $f!([$($bounds)*] Expr<$T>: |x| $Vsplat(x), Expr<$Vt>: |x| x => Expr<$Vt>);
        $f!(['a, $($bounds)*] &'a Expr<$T>: |x| $Vsplat(x), Expr<$Vt>: |x| x => Expr<$Vt>);
        $f!(['b, $($bounds)*] $T: |x| $Vsplat(x), &'b Expr<$Vt>: |x| x.clone() => Expr<$Vt>);
        $f!(['a, 'b, $($bounds)*] &'a $T: |x| $Vsplat(*x), &'b Expr<$Vt>: |x| x.clone() => Expr<$Vt>);
        $f!(['b, $($bounds)*] Expr<$T>: |x| $Vsplat(x), &'b Expr<$Vt>: |x| x.clone() => Expr<$Vt>);
        $f!(['a, 'b, $($bounds)*] &'a Expr<$T>: |x| $Vsplat(x), &'b Expr<$Vt>: |x| x.clone() => Expr<$Vt>);

        $f!([$($bounds)*] Expr<$T>: |x| $Vsplat(x), $Vt: |x| x.expr() => Expr<$Vt>);
        $f!(['a, $($bounds)*] &'a Expr<$T>: |x| $Vsplat(x), $Vt: |x| x.expr() => Expr<$Vt>);
        $f!(['b, $($bounds)*] Expr<$T>: |x| $Vsplat(x), &'b $Vt: |x| x.expr() => Expr<$Vt>);
        $f!(['a, 'b, $($bounds)*] &'a Expr<$T>: |x| $Vsplat(x), &'b $Vt: |x| x.expr() => Expr<$Vt>);

        $f!([$($bounds)*] $T: |x| $Vsplat(x), Var<$Vt>: |x| x.load() => Expr<$Vt>);
        $f!(['a, $($bounds)*] &'a $T: |x| $Vsplat(*x), Var<$Vt>: |x| x.load() => Expr<$Vt>);
        $f!([$($bounds)*] Expr<$T>: |x| $Vsplat(x), Var<$Vt>: |x| x.load() => Expr<$Vt>);
        $f!(['a, $($bounds)*] &'a Expr<$T>: |x| $Vsplat(x), Var<$Vt>: |x| x.load() => Expr<$Vt>);
        $f!(['b, $($bounds)*] $T: |x| $Vsplat(x), &'b Var<$Vt>: |x| x.load() => Expr<$Vt>);
        $f!(['a, 'b, $($bounds)*] &'a $T: |x| $Vsplat(*x), &'b Var<$Vt>: |x| x.load() => Expr<$Vt>);
        $f!(['b, $($bounds)*] Expr<$T>: |x| $Vsplat(x), &'b Var<$Vt>: |x| x.load() => Expr<$Vt>);
        $f!(['a, 'b, $($bounds)*] &'a Expr<$T>: |x| $Vsplat(x), &'b Var<$Vt>: |x| x.load() => Expr<$Vt>);

        $f!([$($bounds)*] Var<$T>: |x| $Vsplat(x), Var<$Vt>: |x| x.load() => Expr<$Vt>);
        $f!(['a, $($bounds)*] &'a Var<$T>: |x| $Vsplat(x), Var<$Vt>: |x| x.load() => Expr<$Vt>);
        $f!(['b, $($bounds)*] Var<$T>: |x| $Vsplat(x), &'b Var<$Vt>: |x| x.load() => Expr<$Vt>);
        $f!(['a, 'b, $($bounds)*] &'a Var<$T>: |x| $Vsplat(x), &'b Var<$Vt>: |x| x.load() => Expr<$Vt>);
    };
    ($f:ident [$($bounds:tt)*]($N:tt, $T:ty)) => {
        call_vector_fn_spread!($f[$($bounds)*]($N, $T) Vector<$T, $N>, Vector::<$T, $N>::splat_expr);
    };
    ($f:ident[$N:ident, $T:ident]) => {
        call_vector_fn_spread!($f[const $N: usize, $T: VectorAlign<$N>]($N, $T));
    }
}

call_vector_fn_spread!(impl_spread[N, T]);

macro_rules! impl_simple_binop_spread {
    ($TraitExpr:ident [$TraitThis:ident]: $fn:ident[$fn_this:ident]) => {
        impl<T, S> $TraitExpr<S> for T
        where
            T: SpreadOps<S>,
            Expr<T::Join>: $TraitThis,
        {
            type Output = Expr<T::Join>;
            fn $fn(self, other: S) -> Self::Output {
                Expr::<T::Join>::$fn_this(Self::lift_self(self), Self::lift_other(other))
            }
        }
    };
}

macro_rules! impl_assignop_spread {
    ([$($bounds:tt)*] $TraitExpr:ident [$TraitOrigExpr:ident] for $X:ty[$V:ty]: $assign_fn:ident [$fn:ident]) => {
        impl<_Other, $($bounds)*> $TraitExpr<_Other> for &$X
        where
            Expr<$V>: $TraitOrigExpr,
            Expr<$V>: SpreadOps<_Other, Join = $V> + Sized,
        {
            fn $assign_fn(self, other: _Other) {
                self.as_var_from_proxy().store(
                    <Expr::<$V> as $TraitOrigExpr>::$fn(
                        self.deref().clone(),
                        <Expr::<$V> as SpreadOps<_Other>>::lift_other(other)
                    )
                );
            }
        }
    }
}

macro_rules! impl_assignops {
    ([$($bounds:tt)*] $X:ty[$V:ty]) => {
        impl_assignop_spread!([$($bounds)*] AddAssignExpr[AddThis] for $X[$V]: add_assign[_add]);
        impl_assignop_spread!([$($bounds)*] SubAssignExpr[SubThis] for $X[$V]: sub_assign[_sub]);
        impl_assignop_spread!([$($bounds)*] MulAssignExpr[MulThis] for $X[$V]: mul_assign[_mul]);
        impl_assignop_spread!([$($bounds)*] DivAssignExpr[DivThis] for $X[$V]: div_assign[_div]);
        impl_assignop_spread!([$($bounds)*] RemAssignExpr[RemThis] for $X[$V]: rem_assign[_rem]);
        impl_assignop_spread!([$($bounds)*] BitAndAssignExpr[BitAndThis] for $X[$V]: bitand_assign[_bitand]);
        impl_assignop_spread!([$($bounds)*] BitOrAssignExpr[BitOrThis] for $X[$V]: bitor_assign[_bitor]);
        impl_assignop_spread!([$($bounds)*] BitXorAssignExpr[BitXorThis] for $X[$V]: bitxor_assign[_bitxor]);
        impl_assignop_spread!([$($bounds)*] ShlAssignExpr[ShlThis] for $X[$V]: shl_assign[_shl]);
        impl_assignop_spread!([$($bounds)*] ShrAssignExpr[ShrThis] for $X[$V]: shr_assign[_shr]);
    }
}
impl_assignops!([T: Primitive] PrimitiveVar<T>[T]);
impl_assignops!([T: VectorAlign<2, VectorVar = VectorVarProxy2<T>>] VectorVarProxy2<T>[Vector<T, 2>]);
impl_assignops!([T: VectorAlign<3, VectorVar = VectorVarProxy3<T>>] VectorVarProxy3<T>[Vector<T, 3>]);
impl_assignops!([T: VectorAlign<4, VectorVar = VectorVarProxy4<T>>] VectorVarProxy4<T>[Vector<T, 4>]);

impl_simple_binop_spread!(AddExpr[AddThis]: add[_add]);
impl_simple_binop_spread!(SubExpr[SubThis]: sub[_sub]);
impl_simple_binop_spread!(MulExpr[MulThis]: mul[_mul]);
impl_simple_binop_spread!(DivExpr[DivThis]: div[_div]);
impl_simple_binop_spread!(RemExpr[RemThis]: rem[_rem]);
impl_simple_binop_spread!(BitAndExpr[BitAndThis]: bitand[_bitand]);
impl_simple_binop_spread!(BitOrExpr[BitOrThis]: bitor[_bitor]);
impl_simple_binop_spread!(BitXorExpr[BitXorThis]: bitxor[_bitxor]);
impl_simple_binop_spread!(ShlExpr[ShlThis]: shl[_shl]);
impl_simple_binop_spread!(ShrExpr[ShrThis]: shr[_shr]);

impl<T, S> MinMaxExpr<S> for T
where
    T: SpreadOps<S>,
    Expr<T::Join>: MinMaxThis,
{
    type Output = <Expr<T::Join> as MinMaxThis>::Output;
    fn min_expr(self, other: S) -> Self::Output {
        Expr::<T::Join>::_min_expr(Self::lift_self(self), Self::lift_other(other))
    }
    fn max_expr(self, other: S) -> Self::Output {
        Expr::<T::Join>::_max_expr(Self::lift_self(self), Self::lift_other(other))
    }
}

pub fn min<T, S>(x: T, y: S) -> <T as MinMaxExpr<S>>::Output
where
    T: MinMaxExpr<S>,
{
    x.min_expr(y)
}
pub fn max<T, S>(x: T, y: S) -> <T as MinMaxExpr<S>>::Output
where
    T: MinMaxExpr<S>,
{
    x.max_expr(y)
}

impl<T: Value, S, U> ClampExpr<S, U> for Expr<T>
where
    S: SpreadOps<U, Join = T>,
    Expr<T>: ClampThis,
{
    type Output = Expr<T>;
    fn clamp(self, min: S, max: U) -> Self::Output {
        Expr::<T>::_clamp(self, S::lift_self(min), S::lift_other(max))
    }
}
impl<T, S> EqExpr<S> for T
where
    T: SpreadOps<S>,
    Expr<T::Join>: EqThis,
{
    type Output = <Expr<T::Join> as EqThis>::Output;
    fn eq(self, other: S) -> Self::Output {
        Expr::<T::Join>::_eq(Self::lift_self(self), Self::lift_other(other))
    }
    fn ne(self, other: S) -> Self::Output {
        Expr::<T::Join>::_ne(Self::lift_self(self), Self::lift_other(other))
    }
}
impl<T, S> CmpExpr<S> for T
where
    T: SpreadOps<S>,
    Expr<T::Join>: CmpThis,
{
    type Output = <Expr<T::Join> as CmpThis>::Output;
    fn lt(self, other: S) -> Self::Output {
        Expr::<T::Join>::_lt(Self::lift_self(self), Self::lift_other(other))
    }
    fn le(self, other: S) -> Self::Output {
        Expr::<T::Join>::_le(Self::lift_self(self), Self::lift_other(other))
    }
    fn gt(self, other: S) -> Self::Output {
        Expr::<T::Join>::_gt(Self::lift_self(self), Self::lift_other(other))
    }
    fn ge(self, other: S) -> Self::Output {
        Expr::<T::Join>::_ge(Self::lift_self(self), Self::lift_other(other))
    }
}
impl<T: Value, S, U> FloatMulAddExpr<S, U> for Expr<T>
where
    S: SpreadOps<U, Join = T>,
    Expr<T>: FloatMulAddThis,
{
    type Output = Expr<T>;
    fn mul_add(self, mul: S, add: U) -> Self::Output {
        Expr::<T>::_mul_add(self, S::lift_self(mul), S::lift_other(add))
    }
}
impl<T, S> FloatCopySignExpr<S> for T
where
    T: SpreadOps<S>,
    Expr<T::Join>: FloatCopySignThis,
{
    type Output = Expr<T::Join>;
    fn copy_sign(self, sign: S) -> Self::Output {
        Expr::<T::Join>::_copy_sign(Self::lift_self(self), Self::lift_other(sign))
    }
}
impl<T, S> FloatStepExpr<S> for T
where
    T: SpreadOps<S>,
    Expr<T::Join>: FloatStepThis,
{
    type Output = Expr<T::Join>;
    fn step(self, edge: S) -> Self::Output {
        Expr::<T::Join>::_step(Self::lift_self(self), Self::lift_other(edge))
    }
}
impl<T: Value, S, U> FloatSmoothStepExpr<S, U> for Expr<T>
where
    S: SpreadOps<U, Join = T>,
    Expr<T>: FloatSmoothStepThis,
{
    type Output = Expr<T>;
    fn smooth_step(self, edge0: S, edge1: U) -> Self::Output {
        Expr::<T>::_smooth_step(self, S::lift_self(edge0), S::lift_other(edge1))
    }
}
impl<T, S> FloatArcTan2Expr<S> for T
where
    T: SpreadOps<S>,
    Expr<T::Join>: FloatArcTan2This,
{
    type Output = Expr<T::Join>;
    fn atan2(self, other: S) -> Self::Output {
        Expr::<T::Join>::_atan2(Self::lift_self(self), Self::lift_other(other))
    }
}
impl<T, S> FloatLogExpr<S> for T
where
    T: SpreadOps<S>,
    Expr<T::Join>: FloatLogThis,
{
    type Output = Expr<T::Join>;
    fn log(self, base: S) -> Self::Output {
        Expr::<T::Join>::_log(Self::lift_self(self), Self::lift_other(base))
    }
}
impl<T, S> FloatPowfExpr<S> for T
where
    T: SpreadOps<S>,
    Expr<T::Join>: FloatPowfThis,
{
    type Output = Expr<T::Join>;
    fn powf(self, exponent: S) -> Self::Output {
        Expr::<T::Join>::_powf(Self::lift_self(self), Self::lift_other(exponent))
    }
}
impl<T: Value, S, U> FloatLerpExpr<S, U> for Expr<T>
where
    S: SpreadOps<U, Join = T>,
    Expr<T>: FloatLerpThis,
{
    type Output = Expr<T>;
    fn lerp(self, other: S, frac: U) -> Self::Output {
        Expr::<T>::_lerp(self, S::lift_self(other), S::lift_other(frac))
    }
}
