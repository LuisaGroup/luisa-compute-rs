use super::*;
use traits::*;

impl<X: Linear> Expr<X> {
    fn as_<Y: Linear<Scalar = X::Scalar>>(self) -> Y
    where
        Y::Scalar: CastFrom<X::Scalar>,
    {
        Func::Cast.call(self)
    }
    fn cast<S: VectorElement>(self) -> Expr<X::WithScalar<S>>
    where
        S: CastFrom<X::Scalar>,
    {
        self.as_::<Self::WithScalar<S>>()
    }
}

impl<X: Linear> MinMaxExpr for Expr<X>
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

impl<X: Linear> ClampExpr for Expr<X>
where
    X::Scalar: Numeric,
{
    type Output = Self;

    fn clamp(self, min: Self, max: Self) -> Self {
        Func::Clamp.call3(self, min, max)
    }
}

impl<X: Linear> AbsExpr for Expr<X>
where
    X::Scalar: Signed,
{
    fn abs(&self) -> Self {
        Func::Abs.call(self)
    }
}

impl<X: Linear> EqExpr for Expr<X> {
    type Output = Expr<X::WithScalar<bool>>;
    fn eq(self, other: Self) -> Self::Output {
        Func::Eq.call2(self, other)
    }
    fn ne(self, other: Self) -> Self::Output {
        Func::Ne.call2(self, other)
    }
}
impl<X: Linear> CmpExpr for Expr<X> {
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

impl<X: Linear> Add for Expr<X>
where
    X::Scalar: Numeric,
{
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Func::Add.call2(self, other)
    }
}
impl<X: Linear> Sub for Expr<X>
where
    X::Scalar: Numeric,
{
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Func::Sub.call2(self, other)
    }
}
impl<X: Linear> Mul for Expr<X>
where
    X::Scalar: Numeric,
{
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Func::Mul.call2(self, other)
    }
}
impl<X: Linear> Div for Expr<X>
where
    X::Scalar: Numeric,
{
    type Output = Self;
    fn div(self, other: Self) -> Self {
        Func::Div.call2(self, other)
    }
}
impl<X: Linear> Rem for Expr<X>
where
    X::Scalar: Numeric,
{
    type Output = Self;
    fn rem(self, other: Self) -> Self {
        Func::Rem.call2(self, other)
    }
}

impl<X: Linear> BitAnd for Expr<X>
where
    X::Scalar: Integral,
{
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        Func::BitAnd.call2(self, other)
    }
}
impl<X: Linear> BitOr for Expr<X>
where
    X::Scalar: Integral,
{
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        Func::BitOr.call2(self, other)
    }
}
impl<X: Linear> BitXor for Expr<X>
where
    X::Scalar: Integral,
{
    type Output = Self;
    fn bitxor(self, other: Self) -> Self {
        Func::BitXor.call2(self, other)
    }
}
impl<X: Linear> Shl for Expr<X>
where
    X::Scalar: Integral,
{
    type Output = Self;
    fn shl(self, other: Self) -> Self {
        Func::Shl.call2(self, other)
    }
}
impl<X: Linear> Shr for Expr<X>
where
    X::Scalar: Integral,
{
    type Output = Self;
    fn shr(self, other: Self) -> Self {
        Func::Shr.call2(self, other)
    }
}

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
impl<X: Linear> FloatMulAddExpr for Expr<X>
where
    X::Scalar: Floating,
{
    type Output = Self;

    fn mul_add(self, a: Self, b: Self) -> Self::Output {
        Func::Fma.call3(self, a, b)
    }
}
impl<X: Linear> FloatCopySignExpr for Expr<X>
where
    X::Scalar: Floating,
{
    type Output = Self;

    fn copy_sign(self, sign: Self) -> Self::Output {
        Func::Copysign.call2(self, sign)
    }
}
impl<X: Linear> FloatStepExpr for Expr<X>
where
    X::Scalar: Floating,
{
    type Output = Self;

    fn step(self, edge: Self) -> Self::Output {
        Func::Step.call2(edge, self)
    }
}
impl<X: Linear> FloatSmoothStepExpr for Expr<X>
where
    X::Scalar: Floating,
{
    type Output = Self;

    fn smooth_step(self, edge0: Self, edge1: Self) -> Self::Output {
        Func::SmoothStep.call3(edge0, edge1, self)
    }
}
impl<X: Linear> FloatArcTan2Expr for Expr<X>
where
    X::Scalar: Floating,
{
    type Output = Self;

    fn atan2(self, other: Self) -> Self::Output {
        Func::Atan2.call2(self, other)
    }
}
impl<X: Linear> FloatLogExpr for Expr<X>
where
    X::Scalar: Floating,
{
    type Output = Self;

    fn log(self, base: Self) -> Self::Output {
        self.ln() / base.ln()
    }
}
impl<X: Linear> FloatPowfExpr for Expr<X>
where
    X::Scalar: Floating,
{
    type Output = Self;

    fn powf(self, exponent: Self) -> Self::Output {
        Func::Powf.call2(self, exponent)
    }
}
impl<X: Linear, Y: Linear<Scalar = i32>> FloatPowiExpr<Expr<Y>> for Expr<X>
where
    X::Scalar: Floating,
{
    type Output = Self;

    fn powi(self, exponent: Expr<Y>) -> Self::Output {
        Func::Powi.call2(self, exponent)
    }
}
impl<X: Linear> FloatLerpExpr for Expr<X>
where
    X::Scalar: Floating,
{
    type Output = Self;

    fn lerp(self, other: Self, frac: Self) -> Self::Output {
        Func::Lerp.call3(self, other, frac)
    }
}
