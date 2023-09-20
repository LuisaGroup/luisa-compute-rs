use super::*;
use traits::*;

impl<const N: usize, X: Linear<N>> Expr<X> {
    fn as_<Y: Linear<N, Scalar = X::Scalar>>(self) -> Y
    where
        Y::Scalar: CastFrom<X::Scalar>,
    {
        Func::Cast.call(self)
    }
    fn cast<S: VectorElement<N>>(self) -> Expr<X::WithScalar<S>>
    where
        S: CastFrom<X::Scalar>,
    {
        self.as_::<Self::WithScalar<S>>()
    }
}

impl<const N: usize, X: Linear<N>> MinMaxExpr for Expr<X>
where
    X::Scalar: Numeric,
{
    type Output = Self;

    fn max(self, other: Self) -> Self {
        Func::Max.call2(self, other)
    }
    fn min(self, other: Self) -> Self {
        Func::Min.call2(self, other)
    }
}

impl<const N: usize, X: Linear<N>> ClampExpr for Expr<X>
where
    X::Scalar: Numeric,
{
    type Output = Self;

    fn clamp(self, min: Self, max: Self) -> Self {
        Func::Clamp.call3(self, min, max)
    }
}

impl<const N: usize, X: Linear<N>> AbsExpr for Expr<X>
where
    X::Scalar: Signed,
{
    fn abs(&self) -> Self {
        Func::Abs.call(self)
    }
}

impl<const N: usize, X: Linear<N>> EqExpr for Expr<X> {
    type Output = Expr<X::WithBool>;
    fn eq(self, other: Self) -> Self::Output {
        Func::Eq.call2(self, other)
    }
    fn ne(self, other: Self) -> Self::Output {
        Func::Ne.call2(self, other)
    }
}
impl<const N: usize, X: Linear<N>> CmpExpr for Expr<X> {
    fn lt(self, other: Self) -> Self::Output {
        Func::Lt.call2(self, other)
    }
    fn le(self, other: Self) -> Self::Output {
        Func::Le.call2(self, other)
    }
    fn gt(self, other: Self) -> Self::Output {
        Func::Gt.call2(self, other)
    }
    fn ge(self, other: Self) -> Self::Output {
        Func::Ge.call2(self, other)
    }
}

impl<const N: usize, X: Linear<N>> Add for Expr<X>
where
    X::Scalar: Numeric,
{
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Func::Add.call2(self, other)
    }
}
impl<const N: usize, X: Linear<N>> Sub for Expr<X>
where
    X::Scalar: Numeric,
{
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Func::Sub.call2(self, other)
    }
}
impl<const N: usize, X: Linear<N>> Mul for Expr<X>
where
    X::Scalar: Numeric,
{
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Func::Mul.call2(self, other)
    }
}
impl<const N: usize, X: Linear<N>> Div for Expr<X>
where
    X::Scalar: Numeric,
{
    type Output = Self;
    fn div(self, other: Self) -> Self {
        Func::Div.call2(self, other)
    }
}
impl<const N: usize, X: Linear<N>> Rem for Expr<X>
where
    X::Scalar: Numeric,
{
    type Output = Self;
    fn rem(self, other: Self) -> Self {
        Func::Rem.call2(self, other)
    }
}

impl<const N: usize, X: Linear<N>> BitAnd for Expr<X>
where
    X::Scalar: Integral,
{
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        Func::BitAnd.call2(self, other)
    }
}
impl<const N: usize, X: Linear<N>> BitOr for Expr<X>
where
    X::Scalar: Integral,
{
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        Func::BitOr.call2(self, other)
    }
}
impl<const N: usize, X: Linear<N>> BitXor for Expr<X>
where
    X::Scalar: Integral,
{
    type Output = Self;
    fn bitxor(self, other: Self) -> Self {
        Func::BitXor.call2(self, other)
    }
}
impl<const N: usize, X: Linear<N>> Shl for Expr<X>
where
    X::Scalar: Integral,
{
    type Output = Self;
    fn shl(self, other: Self) -> Self {
        Func::Shl.call2(self, other)
    }
}
impl<const N: usize, X: Linear<N>> Shr for Expr<X>
where
    X::Scalar: Integral,
{
    type Output = Self;
    fn shr(self, other: Self) -> Self {
        Func::Shr.call2(self, other)
    }
}

impl<const N: usize, X: Linear<N>> Neg for Expr<X>
where
    X::Scalar: Signed,
{
    type Output = Self;
    fn neg(self) -> Self {
        Func::Neg.call(self)
    }
}
impl<const N: usize, X: Linear<N>> Not for Expr<X>
where
    X::Scalar: Integral,
{
    type Output = Self;
    fn not(self) -> Self {
        Func::BitNot.call(self)
    }
}

impl<const N: usize, X: Linear<N>> IntExpr for Expr<X>
where
    X::Scalar: Integral + Numeric,
{
    fn rotate_left(&self, n: Expr<u32>) -> Self {
        Func::RotRight.call2(self, n)
    }
    fn rotate_right(&self, n: Expr<u32>) -> Self {
        Func::RotLeft.call2(self, n)
    }
}

macro_rules! impl_simple_fns {
    ($($fname:ident => $func:ident),+) => {$(
        fn $fname(&self) -> Self {
            Func::$func.call(self)
        }
    )+};
}

impl<const N: usize, X: Linear<N>> FloatExpr for Expr<X>
where
    X::Scalar: Floating,
{
    type Bool = Self::WithBool;
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
        is_infinite => IsInf,
        is_nan => IsNan,
        ln => Log,
        log2 => Log2,
        log10 => Log10
    }
    fn is_finite(&self) -> Self::Bool {
        !self.is_infinite() & !self.is_nan()
    }
    fn sqr(&self) -> Self {
        *self * *self
    }
    fn cube(&self) -> Self {
        *self * *self * *self
    }
    fn recip(&self) -> Self {
        1.0 / *self
    }
    fn sin_cos(&self) -> (Self, Self) {
        (self.sin(), self.cos())
    }
}
impl<const N: usize, X: Linear<N>> FloatMulAddExpr for Expr<X>
where
    X::Scalar: Floating,
{
    type Output = Self;

    fn mul_add(self, a: Self, b: Self) -> Self::Output {
        Func::Fma.call3(self, a, b)
    }
}
impl<const N: usize, X: Linear<N>> FloatCopySignExpr for Expr<X>
where
    X::Scalar: Floating,
{
    type Output = Self;

    fn copy_sign(self, sign: Self) -> Self::Output {
        Func::Copysign.call2(self, sign)
    }
}
impl<const N: usize, X: Linear<N>> FloatStepExpr for Expr<X>
where
    X::Scalar: Floating,
{
    type Output = Self;

    fn step(self, edge: Self) -> Self::Output {
        Func::Step.call2(edge, self)
    }
}
impl<const N: usize, X: Linear<N>> FloatSmoothStepExpr for Expr<X>
where
    X::Scalar: Floating,
{
    type Output = Self;

    fn smooth_step(self, edge0: Self, edge1: Self) -> Self::Output {
        Func::SmoothStep.call3(edge0, edge1, self)
    }
}
impl<const N: usize, X: Linear<N>> FloatArcTan2Expr for Expr<X>
where
    X::Scalar: Floating,
{
    type Output = Self;

    fn atan2(self, other: Self) -> Self::Output {
        Func::Atan2.call2(self, other)
    }
}
impl<const N: usize, X: Linear<N>> FloatLogExpr for Expr<X>
where
    X::Scalar: Floating,
{
    type Output = Self;

    fn log(self, base: Self) -> Self::Output {
        self.ln() / base.ln()
    }
}
impl<const N: usize, X: Linear<N>> FloatPowfExpr for Expr<X>
where
    X::Scalar: Floating,
{
    type Output = Self;

    fn powf(self, exponent: Self) -> Self::Output {
        Func::Powf.call2(self, exponent)
    }
}
impl<const N: usize, X: Linear<N>, Y: Linear<N, Scalar = i32>> FloatPowiExpr<Expr<Y>> for Expr<X>
where
    X::Scalar: Floating,
{
    type Output = Self;

    fn powi(self, exponent: Expr<Y>) -> Self::Output {
        Func::Powi.call2(self, exponent)
    }
}
impl<const N: usize, X: Linear<N>> FloatLerpExpr for Expr<X>
where
    X::Scalar: Floating,
{
    type Output = Self;

    fn lerp(self, other: Self, frac: Self) -> Self::Output {
        Func::Lerp.call3(self, other, frac)
    }
}
