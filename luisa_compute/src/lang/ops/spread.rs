use super::*;
use traits::*;

trait SpreadOps<Other> {
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

mod trait_impls {
    use super::*;
    impl<T, S> MinMaxExpr<S> for T
    where
        T: SpreadOps<S>,
        Expr<T::Join>: MinMaxThis,
    {
        type Output = Expr<T::Join>;
        fn max(self, other: S) -> Self::Output {
            Expr::<T::Join>::_max(Self::lift_self(self), Self::lift_other(other))
        }
        fn min(self, other: S) -> Self::Output {
            Expr::<T::Join>::_min(Self::lift_self(self), Self::lift_other(other))
        }
    }
    impl<T, S, U> ClampExpr<S, U> for T
    where
        S: SpreadOps<U>,
        T: SpreadOps<Expr<S::Join>>,
        Expr<T::Join>: ClampThis,
    {
        ///           T::Join
        ///          /        \
        ///         /          \
        ///        /            \
        ///       /              \
        ///      /            S::Join
        ///     /            /       \
        ///    /            /         \
        ///   /            /           \
        ///  /            /             \
        /// T            S               U

        type Output = Expr<T::Join>;
        fn clamp(self, min: S, max: U) -> Self::Output {
            Expr::<T::Join>::_clamp(
                Self::lift_self(self),
                Self::lift_other(S::lift_self(min)),
                Self::lift_other(S::lift_other(max)),
            )
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
    impl<T, S, U> FloatMulAddExpr<S, U> for T
    where
        S: SpreadOps<U>,
        T: SpreadOps<Expr<S::Join>>,
        Expr<T::Join>: FloatMulAddThis,
    {
        type Output = Expr<T::Join>;
        fn mul_add(self, mul: S, add: U) -> Self::Output {
            Expr::<T::Join>::_mul_add(
                Self::lift_self(self),
                Self::lift_other(S::lift_self(mul)),
                Self::lift_other(S::lift_other(add)),
            )
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
    impl<T, S, U> FloatSmoothStepExpr<S, U> for T
    where
        S: SpreadOps<U>,
        T: SpreadOps<Expr<S::Join>>,
        Expr<T::Join>: FloatSmoothStepThis,
    {
        type Output = Expr<T::Join>;
        fn smooth_step(self, edge0: S, edge1: U) -> Self::Output {
            Expr::<T::Join>::_smooth_step(
                Self::lift_self(self),
                Self::lift_other(S::lift_self(edge0)),
                Self::lift_other(S::lift_other(edge1)),
            )
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
    impl<T, S, U> FloatLerpExpr<S, U> for T
    where
        S: SpreadOps<U>,
        T: SpreadOps<Expr<S::Join>>,
        Expr<T::Join>: FloatLerpThis,
    {
        type Output = Expr<T::Join>;
        fn lerp(self, other: S, frac: U) -> Self::Output {
            Expr::<T::Join>::_lerp(
                Self::lift_self(self),
                Self::lift_other(S::lift_self(other)),
                Self::lift_other(S::lift_other(frac)),
            )
        }
    }
}
macro_rules! impl_spread_op {
    ([ $($bounds:tt)* ]: $Op:ident::$op_fn:ident for $T:ty, $S:ty) => {
        impl<$($bounds)*> $Op <$S> for $T where $T: SpreadOps<$S>, Expr<<$T as SpreadOps<$S>>::Join>: $Op {
            type Output = <Expr<<$T as SpreadOps<$S>>::Join> as $Op>::Output;
            fn $op_fn (self, other: $S) -> Self::Output {
                <Expr<<$T as SpreadOps<$S>>::Join> as $Op>::$op_fn (<$T as SpreadOps<$S>>::lift_self(self), <$T as SpreadOps<$S>>::lift_other(other))
            }
        }
    }
}

macro_rules! impl_num_spread_single {
    ([ $($bounds:tt)* ] $T:ty, $S:ty) => {
        impl_spread_op!( [ $($bounds)* ]: Add::add for $T, $S);
        impl_spread_op!( [ $($bounds)* ]: Sub::sub for $T, $S);
        impl_spread_op!( [ $($bounds)* ]: Mul::mul for $T, $S);
        impl_spread_op!( [ $($bounds)* ]: Div::div for $T, $S);
        impl_spread_op!( [ $($bounds)* ]: Rem::rem for $T, $S);
    }
}
macro_rules! impl_int_spread_single {
    ([ $($bounds:tt)* ] $T:ty, $S:ty) => {
        impl_spread_op!([ $($bounds)* ]: BitAnd::bitand for $T, $S);
        impl_spread_op!([ $($bounds)* ]: BitOr::bitor for $T, $S);
        impl_spread_op!([ $($bounds)* ]: BitXor::bitxor for $T, $S);
        impl_spread_op!([ $($bounds)* ]: Shl::shl for $T, $S);
        impl_spread_op!([ $($bounds)* ]: Shr::shr for $T, $S);
    }
}

macro_rules! impl_num_spread {
    (@sym [$($bounds:tt)*] $T:ty : |$x:ident| $f:expr, $S:ty : |$y:ident| $g:expr => Expr<$J:ty>) => {
        impl_num_spread_single!([$($bounds)*] $T, $S);
    };
    ([$($bounds:tt)*] $T:ty : |$x:ident| $f:expr, $S:ty : |$y:ident| $g:expr => Expr<$J:ty>) => {
        impl_num_spread_single!([$($bounds)*] $T, $S);
        impl_num_spread_single!([$($bounds)*] $S, $T);
    }
}
macro_rules! impl_int_spread {
    (@sym [$($bounds:tt)*] $T:ty : |$x:ident| $f:expr, $S:ty : |$y:ident| $g:expr => Expr<$J:ty>) => {
        impl_int_spread_single!([$($bounds)*] $T, $S);
    };
    ([$($bounds:tt)*] $T:ty : |$x:ident| $f:expr, $S:ty : |$y:ident| $g:expr => Expr<$J:ty>) => {
        impl_int_spread_single!([$($bounds)*] $T, $S);
        impl_int_spread_single!([$($bounds)*] $S, $T);
    }
}
macro_rules! call_spreads {
    ($f:ident: $($T:ty),+) => {
        $(
        call_linear_fn_spread!($f []($T));
        call_vector_fn_spread!($f [](2, $T));
        call_vector_fn_spread!($f [](3, $T));
        call_vector_fn_spread!($f [](4, $T));
        )+
    };
}
call_spreads!(impl_num_spread: f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);
call_spreads!(impl_int_spread: bool, i8, i16, i32, i64, u8, u16, u32, u64);

#[allow(dead_code)]
mod tests {
    use super::*;
    fn test() {
        let x = 10.0f32;
        let y = Vector::<_, 2>::splat(20.0f32);
        let x = x.expr();

        let w = (&x.var()).min(&0.0_f32.expr());
        println!("{:?}", w);
    }
}
