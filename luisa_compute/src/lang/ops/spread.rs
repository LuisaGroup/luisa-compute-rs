use super::*;
use traits::*;

trait SpreadOps<Other> {
    type Join: Value;
    fn lift_self(x: Self) -> Expr<Self::Join>;
    fn lift_other(x: Other) -> Expr<Self::Join>;
}

macro_rules! impl_spread_single {
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
    };
}
macro_rules! impl_spread {
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
impl_spread!([T: Linear] T: |x| x.expr(), Expr<T>: |x| x => Expr<T>);
impl_spread!([T: Linear] &T: |x| x.expr(), Expr<T>: |x| x => Expr<T>);
impl_spread!([T: Linear] T: |x| x.expr(), &Expr<T>: |x| x.clone() => Expr<T>);
impl_spread!([T: Linear] &T: |x| x.expr(), &Expr<T>: |x| x.clone() => Expr<T>);

impl_spread!([T: Linear] Expr<T>: |x| x, &Expr<T>: |x| x.clone() => Expr<T>);
impl_spread_single!([T: Linear] &Expr<T>: |x| x.clone(), &Expr<T>: |x| x.clone() => Expr<T>);

impl_spread!([T: Linear] T: |x| x.expr(), Var<T>: |x| x.load() => Expr<T>);
impl_spread!([T: Linear] &T: |x| x.expr(), Var<T>: |x| x.load() => Expr<T>);
impl_spread!([T: Linear] T: |x| x.expr(), &Var<T>: |x| x.load() => Expr<T>);
impl_spread!([T: Linear] &T: |x| x.expr(), &Var<T>: |x| x.load() => Expr<T>);

// Other way is unneded because of the deref impl.
impl_spread_single!([T: Linear] &Expr<T>: |x| x.clone(), Var<T>: |x| x.load() => Expr<T>);
impl_spread_single!([T: Linear] &Expr<T>: |x| x.clone(), &Var<T>: |x| x.load() => Expr<T>);

impl_spread!([const N: usize, T: VectorAlign<N>] T: |x| Vector::<T, N>::splat_expr(x), Expr<Vector<T, N>>: |x| x => Expr<Vector<T, N>>);
impl_spread!([const N: usize, T: VectorAlign<N>] &T: |x| Vector::<T, N>::splat_expr(x), Expr<Vector<T, N>>: |x| x => Expr<Vector<T, N>>);
impl_spread!([const N: usize, T: VectorAlign<N>] Expr<T>: |x| Vector::<T, N>::splat_expr(x), Expr<Vector<T, N>>: |x| x => Expr<Vector<T, N>>);
impl_spread!([const N: usize, T: VectorAlign<N>] &Expr<T>: |x| Vector::<T, N>::splat_expr(x), Expr<Vector<T, N>>: |x| x => Expr<Vector<T, N>>);
impl_spread!([const N: usize, T: VectorAlign<N>] T: |x| Vector::<T, N>::splat_expr(x), &Expr<Vector<T, N>>: |x| x.clone() => Expr<Vector<T, N>>);
impl_spread!([const N: usize, T: VectorAlign<N>] &T: |x| Vector::<T, N>::splat_expr(x), &Expr<Vector<T, N>>: |x| x.clone() => Expr<Vector<T, N>>);
impl_spread!([const N: usize, T: VectorAlign<N>] Expr<T>: |x| Vector::<T, N>::splat_expr(x), &Expr<Vector<T, N>>: |x| x.clone() => Expr<Vector<T, N>>);
impl_spread!([const N: usize, T: VectorAlign<N>] &Expr<T>: |x| Vector::<T, N>::splat_expr(x), &Expr<Vector<T, N>>: |x| x.clone() => Expr<Vector<T, N>>);

impl_spread!([const N: usize, T: VectorAlign<N>] Expr<T>: |x| Vector::<T, N>::splat_expr(x), Vector<T, N>: |x| x.expr() => Expr<Vector<T, N>>);
impl_spread!([const N: usize, T: VectorAlign<N>] &Expr<T>: |x| Vector::<T, N>::splat_expr(x), Vector<T, N>: |x| x.expr() => Expr<Vector<T, N>>);
impl_spread!([const N: usize, T: VectorAlign<N>] Expr<T>: |x| Vector::<T, N>::splat_expr(x), &Vector<T, N>: |x| x.expr() => Expr<Vector<T, N>>);
impl_spread!([const N: usize, T: VectorAlign<N>] &Expr<T>: |x| Vector::<T, N>::splat_expr(x), &Vector<T, N>: |x| x.expr() => Expr<Vector<T, N>>);

mod impls {
    use super::*;
    impl<T, S> MinMaxExpr<S> for T
    where
        T: SpreadOps<S>,
        Expr<T::Join>: MinMaxExpr,
    {
        type Output = <Expr<T::Join> as MinMaxExpr>::Output;
        fn max(self, other: S) -> Self::Output {
            Expr::<T::Join>::max(Self::lift_self(self), Self::lift_other(other))
        }
        fn min(self, other: S) -> Self::Output {
            Expr::<T::Join>::min(Self::lift_self(self), Self::lift_other(other))
        }
    }
    impl<T, S, U> ClampExpr<S, U> for T
    where
        S: SpreadOps<U>,
        T: SpreadOps<S::Join>,
        Expr<T::Join>: ClampExpr,
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

        type Output = <Expr<T::Join> as ClampExpr>::Output;
        fn clamp(&self, min: S, max: U) -> Self::Output {
            Expr::<T::Join>::clamp(
                Self::lift_self(self),
                Self::lift_other(S::lift_self(min)),
                Self::lift_other(S::lift_other(max)),
            )
        }
    }
    impl<T, S> EqExpr<S> for T
    where
        T: SpreadOps<S>,
        Expr<T::Join>: EqExpr,
    {
        type Output = <Expr<T::Join> as EqExpr>::Output;
        fn eq(self, other: S) -> Self::Output {
            Expr::<T::Join>::eq(Self::lift_self(self), Self::lift_other(other))
        }
        fn ne(self, other: S) -> Self::Output {
            Expr::<T::Join>::ne(Self::lift_self(self), Self::lift_other(other))
        }
    }
    impl<T, S> CmpExpr<S> for T
    where
        T: SpreadOps<S>,
        Expr<T::Join>: CmpExpr,
    {
        fn lt(self, other: S) -> Self::Output {
            Expr::<T::Join>::lt(Self::lift_self(self), Self::lift_other(other))
        }
        fn le(self, other: S) -> Self::Output {
            Expr::<T::Join>::le(Self::lift_self(self), Self::lift_other(other))
        }
        fn gt(self, other: S) -> Self::Output {
            Expr::<T::Join>::gt(Self::lift_self(self), Self::lift_other(other))
        }
        fn ge(self, other: S) -> Self::Output {
            Expr::<T::Join>::ge(Self::lift_self(self), Self::lift_other(other))
        }
    }
    impl<T, S, U> FloatMulAddExpr<S, U> for T
    where
        S: SpreadOps<U>,
        T: SpreadOps<S::Join>,
        Expr<T::Join>: FloatMulAddExpr,
    {
        type Output = <Expr<T::Join> as FloatMulAddExpr>::Output;
        fn mul_add(self, mul: S, add: U) -> Self::Output {
            Expr::<T::Join>::mul_add(
                Self::lift_self(self),
                Self::lift_other(S::lift_self(mul)),
                Self::lift_other(S::lift_other(add)),
            )
        }
    }
    impl<T, S> FloatCopySignExpr<S> for T
    where
        T: SpreadOps<S>,
        Expr<T::Join>: FloatCopySignExpr,
    {
        type Output = <Expr<T::Join> as FloatCopySignExpr>::Output;
        fn copy_sign(self, sign: S) -> Self::Output {
            Expr::<T::Join>::copy_sign(Self::lift_self(self), Self::lift_other(sign))
        }
    }
    impl<T, S> FloatStepExpr<S> for T
    where
        T: SpreadOps<S>,
        Expr<T::Join>: FloatStepExpr,
    {
        type Output = <Expr<T::Join> as FloatStepExpr>::Output;
        fn step(self, edge: S) -> Self::Output {
            Expr::<T::Join>::step(Self::lift_self(self), Self::lift_other(edge))
        }
    }
    impl<T, S, U> FloatSmoothStepExpr<S, U> for T
    where
        S: SpreadOps<U>,
        T: SpreadOps<S::Join>,
        Expr<T::Join>: FloatSmoothStepExpr,
    {
        type Output = <Expr<T::Join> as FloatSmoothStepExpr>::Output;
        fn smooth_step(self, edge0: S, edge1: U) -> Self::Output {
            Expr::<T::Join>::smooth_step(
                Self::lift_self(self),
                Self::lift_other(S::lift_self(edge0)),
                Self::lift_other(S::lift_other(edge1)),
            )
        }
    }
    impl<T, S> FloatArcTan2Expr<S> for T
    where
        T: SpreadOps<S>,
        Expr<T::Join>: FloatArcTan2Expr,
    {
        type Output = <Expr<T::Join> as FloatArcTan2Expr>::Output;
        fn atan2(self, other: S) -> Self::Output {
            Expr::<T::Join>::atan2(Self::lift_self(self), Self::lift_other(other))
        }
    }
    impl<T, S> FloatLogExpr<S> for T
    where
        T: SpreadOps<S>,
        Expr<T::Join>: FloatLogExpr,
    {
        type Output = <Expr<T::Join> as FloatLogExpr>::Output;
        fn log(self, base: S) -> Self::Output {
            Expr::<T::Join>::log(Self::lift_self(self), Self::lift_other(base))
        }
    }
    impl<T, S> FloatPowfExpr<S> for T
    where
        T: SpreadOps<S>,
        Expr<T::Join>: FloatPowfExpr,
    {
        type Output = <Expr<T::Join> as FloatPowfExpr>::Output;
        fn powf(self, exponent: S) -> Self::Output {
            Expr::<T::Join>::powf(Self::lift_self(self), Self::lift_other(exponent))
        }
    }
    impl<T, S, U> FloatLerpExpr<S, U> for T
    where
        S: SpreadOps<U>,
        T: SpreadOps<S::Join>,
        Expr<T::Join>: FloatLerpExpr,
    {
        type Output = <Expr<T::Join> as FloatLerpExpr>::Output;
        fn lerp(self, other: S, frac: U) -> Self::Output {
            Expr::<T::Join>::lerp(
                Self::lift_self(self),
                Self::lift_other(S::lift_self(other)),
                Self::lift_other(S::lift_other(frac)),
            )
        }
    }
}
macro_rules! impl_spread_op {
    ([ $($bounds:tt)* ]: $Op:ident::$op_fn:ident for $T:ty, $S:ty) => {
        impl<$($bounds)*> $Op <$S> for $T where $T: SpreadOps<$T>, Expr<$T::Join>: $Op {
            type Output = <Expr<$T::Join> as $Op>::Output;
            fn $op_fn (self, other: $S) -> Self::Output {
                <Expr<$T::Join> as $Op>::$op_fn (Self::lift_self(self), Self::lift_other(other))
            }
        }
    }
}

macro_rules! impl_num_spread {
    ([ $($bounds:tt)* ]: $T:ty, $S:ty) => {
        impl_spread_op!( [ $($bounds)* ]: Add::add for $T, $S);
        impl_spread_op!( [ $($bounds)* ]: Sub::sub for $T, $S);
        impl_spread_op!( [ $($bounds)* ]: Mul::mul for $T, $S);
        impl_spread_op!( [ $($bounds)* ]: Div::div for $T, $S);
        impl_spread_op!( [ $($bounds)* ]: Rem::rem for $T, $S);
    };
}

mod tests {
    fn test() {
        let x = 10.0f32;
        let y = 20.0f32;
        let z = x.min(y);

        let w = x.expr().min(y);
        println!("{:?}", w);
    }
}
