use crate::lang::types::{vector, ExprType, ValueType};

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

    fn max_[_max_](self, other) { Func::Max.call2(self, other) }
    fn min_[_min_](self, other) { Func::Min.call2(self, other) }
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
impl<X: Linear> Neg for Var<X>
where
    X::Scalar: Signed,
{
    type Output = Expr<X>;
    fn neg(self) -> Expr<X> {
        Func::Neg.call(self.load())
    }
}
impl<X: Linear> Not for Var<X>
where
    X::Scalar: Integral,
{
    type Output = Expr<X>;
    fn not(self) -> Expr<X> {
        Func::BitNot.call(self.load())
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

    ($out:ty, $($fname:ident => $func:ident),+) => {$(
        fn $fname(&self) -> $out {
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
        (!self.is_infinite()).bitand(!self.is_nan())
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
        let self_node = self.node().get();
        <Self as FromNode>::from_node(
            __current_scope(|b| {
                let one = b.const_(Const::One(<X as TypeOf>::type_()));
                b.call(Func::Div, &[one, self_node], <X as TypeOf>::type_())
            })
            .into(),
        )
    }
    fn sin_cos(&self) -> (Self, Self) {
        (self.sin(), self.cos())
    }
}
impl<const N: usize, X: Floating> NormExpr for Expr<Vector<X, N>>
where
    X: vector::VectorAlign<N>,
{
    type Output = Expr<X>;
    impl_simple_fns! {
        Self::Output,
        norm => Length,
        norm_squared => LengthSquared
    }
    impl_simple_fns! {
        normalize=>Normalize
    }
}
impl OuterProductExpr for Expr<Float2> {
    type Output = Expr<Mat2>;
    type Value = Float2;
    fn outer_product(&self, other: impl AsExpr<Value = Self::Value>) -> Self::Output {
        Func::OuterProduct.call2(self.clone(), other.as_expr())
    }
}
impl OuterProductExpr for Expr<Float3> {
    type Output = Expr<Mat3>;
    type Value = Float3;
    fn outer_product(&self, other: impl AsExpr<Value = Self::Value>) -> Self::Output {
        Func::OuterProduct.call2(self.clone(), other.as_expr())
    }
}
impl OuterProductExpr for Expr<Float4> {
    type Output = Expr<Mat4>;
    type Value = Float4;
    fn outer_product(&self, other: impl AsExpr<Value = Self::Value>) -> Self::Output {
        Func::OuterProduct.call2(self.clone(), other.as_expr())
    }
}
impl<const N: usize, X: VectorElement> ReduceExpr for Expr<Vector<X, N>>
where
    X: vector::VectorAlign<N>,
{
    type Output = Expr<X>;
    impl_simple_fns! {
        Self::Output,
        reduce_max=>ReduceMax,
        reduce_min=>ReduceMin,
        reduce_prod=>ReduceProd,
        reduce_sum=>ReduceSum
    }
}
impl<const N: usize, X: Floating> DotExpr for Expr<Vector<X, N>>
where
    X: vector::VectorAlign<N>,
{
    type Value = Vector<X, N>;
    type Output = Expr<X>;
    fn dot(&self, other: impl AsExpr<Value = Self::Value>) -> Self::Output {
        Func::Dot.call2(self.clone(), other.as_expr())
    }
}
impl<X: Floating> CrossExpr for Expr<Vec3<X>>
where
    Vec3<X>: Linear,
    X: VectorAlign<3>,
{
    type Value = Vec3<X>;
    type Output = Expr<Vec3<X>>;
    fn cross(&self, other: impl AsExpr<Value = Self::Value>) -> Self::Output {
        Func::Cross.call2(self.clone(), other.as_expr())
    }
}

impl<const N: usize> Expr<Vector<bool, N>>
where
    Vector<bool, N>: Linear + Value,
    bool: vector::VectorAlign<N>,
{
    pub fn any(&self) -> Expr<bool> {
        Func::Any.call(self.clone())
    }
    pub fn all(&self) -> Expr<bool> {
        Func::All.call(self.clone())
    }
    pub fn select<X: Linear>(&self, t: Expr<X>, f: Expr<X>) -> Expr<X> {
        assert_eq!(
            X::N,
            N,
            "Cannot select between vectors of different dimensions."
        );
        Func::Select.call3(self.clone(), t, f)
    }
}

impl_ops_trait!([X: Linear] FloatMulAddExpr[FloatMulAddThis] for Expr<X> where [X::Scalar: Floating] {
    fn mul_add[_mul_add](self, a, b) { Func::Fma.call3(self, a, b) }
});

impl_ops_trait!([X: Linear] FloatCopySignExpr[FloatCopySignThis] for Expr<X> where [X::Scalar: Floating] {
    fn copysign[_copysign](self, sign) { Func::Copysign.call2(self, sign) }
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

impl<T: DerefMut> StoreMaybeExpr<T::Target> for &mut T
where
    T::Target: Sized,
{
    fn __store(self, value: T::Target) {
        *self.deref_mut() = value;
    }
}
impl<V: Value, E: AsExpr<Value = V>> StoreMaybeExpr<E> for &Var<V> {
    fn __store(self, value: E) {
        crate::lang::_store(self, &value.as_expr());
    }
}
impl<V: Value, E: AsExpr<Value = V>> StoreMaybeExpr<E> for Var<V> {
    fn __store(self, value: E) {
        crate::lang::_store(&self, &value.as_expr());
    }
}

impl<R> SelectMaybeExpr<R> for bool {
    fn if_then_else(self, on: impl Fn() -> R, off: impl Fn() -> R) -> R {
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
    fn if_then_else(self, on: impl Fn() -> R, off: impl Fn() -> R) -> R {
        crate::lang::control_flow::if_then_else(self, on, off)
    }
    fn select(self, on: R, off: R) -> R {
        crate::lang::control_flow::select(self, on, off)
    }
}

impl ActivateMaybeExpr for bool {
    fn activate(self, then: impl Fn()) {
        if self {
            then()
        }
    }
}
impl ActivateMaybeExpr for Expr<bool> {
    fn activate(self, then: impl Fn()) {
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
    fn and(self, other: impl Fn() -> bool) -> bool {
        self && other()
    }
    fn or(self, other: impl Fn() -> bool) -> bool {
        self || other()
    }
}
impl LazyBoolMaybeExpr<Expr<bool>, ExprType> for bool {
    type Bool = Expr<bool>;
    fn and(self, other: impl Fn() -> Expr<bool>) -> Self::Bool {
        if self {
            other()
        } else {
            false.expr()
        }
    }
    fn or(self, other: impl Fn() -> Expr<bool>) -> Self::Bool {
        if self {
            true.expr()
        } else {
            other()
        }
    }
}
impl LazyBoolMaybeExpr<bool, ExprType> for Expr<bool> {
    type Bool = Expr<bool>;
    fn and(self, other: impl Fn() -> bool) -> Self::Bool {
        let other = other().expr();
        select(self, other, false.expr())
    }
    fn or(self, other: impl Fn() -> bool) -> Self::Bool {
        let other = other().expr();
        select(self, true.expr(), other)
    }
}
impl LazyBoolMaybeExpr<Expr<bool>, ExprType> for Expr<bool> {
    type Bool = Expr<bool>;
    fn and(self, other: impl Fn() -> Expr<bool>) -> Self::Bool {
        crate::lang::control_flow::if_then_else(self, other, || false.expr())
    }
    fn or(self, other: impl Fn() -> Expr<bool>) -> Self::Bool {
        crate::lang::control_flow::if_then_else(self, || true.expr(), other)
    }
}
