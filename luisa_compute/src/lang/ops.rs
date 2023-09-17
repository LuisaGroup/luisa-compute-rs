use super::*;

// pub mod impls;

pub trait VarTrait: Copy + Clone + 'static + FromNode {
    type Value: Value;
    type Short: VarTrait;
    type Ushort: VarTrait;
    type Int: VarTrait;
    type Uint: VarTrait;
    type Long: VarTrait;
    type Ulong: VarTrait;
    type Half: VarTrait;
    type Float: VarTrait;
    type Double: VarTrait;
    type Bool: VarTrait + Not<Output = Self::Bool> + BitAnd<Output = Self::Bool>;
    fn type_() -> CArc<Type> {
        <Self::Value as TypeOf>::type_()
    }
}

fn _cast<T: VarTrait, U: VarTrait>(expr: T) -> U {
    let node = expr.node();
    __current_scope(|s| {
        let ret = s.call(Func::Cast, &[node], U::type_());
        U::from_node(ret)
    })
}
pub trait CommonVarOp: VarTrait {
    fn max<A: Into<Self>>(&self, other: A) -> Self {
        let lhs = self.node();
        let rhs = other.into().node();
        __current_scope(|s| {
            let ret = s.call(Func::Max, &[lhs, rhs], Self::type_());
            Self::from_node(ret)
        })
    }
    fn min<A: Into<Self>>(&self, other: A) -> Self {
        let lhs = self.node();
        let rhs = other.into().node();
        __current_scope(|s| {
            let ret = s.call(Func::Min, &[lhs, rhs], Self::type_());
            Self::from_node(ret)
        })
    }
    fn clamp<A: Into<Self>, B: Into<Self>>(&self, min: A, max: B) -> Self {
        let min = min.into().node();
        let max = max.into().node();
        __current_scope(|s| {
            let ret = s.call(Func::Clamp, &[self.node(), min, max], Self::type_());
            Self::from_node(ret)
        })
    }
    fn abs(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Abs, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn bitcast<T: Value>(&self) -> Expr<T> {
        assert_eq!(std::mem::size_of::<Self::Value>(), std::mem::size_of::<T>());
        let ty = <T>::type_();
        let node = __current_scope(|s| s.bitcast(self.node(), ty));
        Expr::<T>::from_node(node)
    }
    fn uint(&self) -> Self::Uint {
        _cast(*self)
    }
    fn int(&self) -> Self::Int {
        _cast(*self)
    }
    fn ulong(&self) -> Self::Ulong {
        _cast(*self)
    }
    fn long(&self) -> Self::Long {
        _cast(*self)
    }
    fn float(&self) -> Self::Float {
        _cast(*self)
    }
    fn short(&self) -> Self::Short {
        _cast(*self)
    }
    fn ushort(&self) -> Self::Ushort {
        _cast(*self)
    }
    fn half(&self) -> Self::Half {
        _cast(*self)
    }
    fn double(&self) -> Self::Double {
        _cast(*self)
    }
    fn bool_(&self) -> Self::Bool {
        _cast(*self)
    }
}
pub trait VarCmpEq: VarTrait {
    fn cmpeq<A: Into<Self>>(&self, other: A) -> Self::Bool {
        let lhs = self.node();
        let rhs = other.into().node();
        __current_scope(|s| {
            let ret = s.call(Func::Eq, &[lhs, rhs], Self::Bool::type_());
            FromNode::from_node(ret)
        })
    }
    fn cmpne<A: Into<Self>>(&self, other: A) -> Self::Bool {
        let lhs = self.node();
        let rhs = other.into().node();
        __current_scope(|s| {
            let ret = s.call(Func::Ne, &[lhs, rhs], Self::Bool::type_());
            FromNode::from_node(ret)
        })
    }
}
pub trait VarCmp: VarTrait + VarCmpEq {
    fn cmplt<A: Into<Self>>(&self, other: A) -> Self::Bool {
        let lhs = self.node();
        let rhs = other.into().node();
        __current_scope(|s| {
            let ret = s.call(Func::Lt, &[lhs, rhs], Self::Bool::type_());
            FromNode::from_node(ret)
        })
    }
    fn cmple<A: Into<Self>>(&self, other: A) -> Self::Bool {
        let lhs = self.node();
        let rhs = other.into().node();
        __current_scope(|s| {
            let ret = s.call(Func::Le, &[lhs, rhs], Self::Bool::type_());
            FromNode::from_node(ret)
        })
    }
    fn cmpgt<A: Into<Self>>(&self, other: A) -> Self::Bool {
        let lhs = self.node();
        let rhs = other.into().node();
        __current_scope(|s| {
            let ret = s.call(Func::Gt, &[lhs, rhs], Self::Bool::type_());
            FromNode::from_node(ret)
        })
    }
    fn cmpge<A: Into<Self>>(&self, other: A) -> Self::Bool {
        let lhs = self.node();
        let rhs = other.into().node();
        __current_scope(|s| {
            let ret = s.call(Func::Ge, &[lhs, rhs], Self::Bool::type_());
            FromNode::from_node(ret)
        })
    }
}
pub trait IntVarTrait:
    VarTrait
    + CommonVarOp
    + VarCmp
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + Shl<Output = Self>
    + Shr<Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + RemAssign
    + ShlAssign
    + ShrAssign
    + BitAndAssign
    + BitOrAssign
    + BitXorAssign
    + Neg<Output = Self>
    + Clone
    + Not<Output = Self>
    + From<Self::Value>
    + From<i64>
{
    fn one() -> Self {
        Self::from(1i64)
    }
    fn zero() -> Self {
        Self::from(0i64)
    }
    fn rotate_right(&self, n: Expr<u32>) -> Self {
        let lhs = self.node();
        let rhs = Expr::<u32>::node(&n);
        __current_scope(|s| {
            let ret = s.call(Func::RotRight, &[lhs, rhs], Self::type_());
            Self::from_node(ret)
        })
    }
    fn rotate_left(&self, n: Expr<u32>) -> Self {
        let lhs = self.node();
        let rhs = Expr::<u32>::node(&n);
        __current_scope(|s| {
            let ret = s.call(Func::RotLeft, &[lhs, rhs], Self::type_());
            Self::from_node(ret)
        })
    }
}
pub trait FloatVarTrait:
    VarTrait
    + CommonVarOp
    + VarCmp
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + RemAssign
    + Clone
    + From<Self::Value>
    + From<f32>
{
    fn one() -> Self {
        Self::from(1.0f32)
    }
    fn zero() -> Self {
        Self::from(0.0f32)
    }
    fn mul_add<A: Into<Self>, B: Into<Self>>(&self, a: A, b: B) -> Self {
        let a: Self = a.into();
        let b: Self = b.into();
        let node = __current_scope(|s| {
            s.call(Func::Fma, &[self.node(), a.node(), b.node()], Self::type_())
        });
        Self::from_node(node)
    }
    fn ceil(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Ceil, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn floor(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Floor, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn round(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Round, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn trunc(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Trunc, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn copysign<A: Into<Self>>(&self, other: A) -> Self {
        __current_scope(|s| {
            let ret = s.call(
                Func::Copysign,
                &[self.node(), other.into().node()],
                Self::type_(),
            );
            Self::from_node(ret)
        })
    }
    fn sqrt(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Sqrt, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn rsqrt(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Rsqrt, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn fract(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Fract, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }

    // x.step(edge)
    fn step(&self, edge: Self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Step, &[edge.node(), self.node()], Self::type_());
            Self::from_node(ret)
        })
    }

    fn smooth_step(&self, edge0: Self, edge1: Self) -> Self {
        __current_scope(|s| {
            let ret = s.call(
                Func::SmoothStep,
                &[edge0.node(), edge1.node(), self.node()],
                Self::type_(),
            );
            Self::from_node(ret)
        })
    }

    fn saturate(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Saturate, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }

    fn sin(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Sin, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
        // crate::math::approx_sin_cos(self.clone(), true, false).0
    }
    fn cos(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Cos, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
        // crate::math::approx_sin_cos(self.clone(), false, true).1
    }
    fn tan(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Tan, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn asin(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Asin, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn acos(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Acos, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn atan(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Atan, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn atan2(&self, other: Self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Atan2, &[self.node(), other.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn sinh(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Sinh, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn cosh(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Cosh, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn tanh(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Tanh, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn asinh(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Asinh, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn acosh(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Acosh, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn atanh(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Atanh, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn exp(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Exp, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn exp2(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Exp2, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn is_finite(&self) -> Self::Bool {
        !self.is_infinite() & !self.is_nan()
    }
    fn is_infinite(&self) -> Self::Bool {
        __current_scope(|s| {
            let ret = s.call(Func::IsInf, &[self.node()], <Self::Bool>::type_());
            FromNode::from_node(ret)
        })
    }
    fn is_nan(&self) -> Self::Bool {
        __current_scope(|s| {
            let ret = s.call(Func::IsNan, &[self.node()], <Self::Bool>::type_());
            FromNode::from_node(ret)
        })
    }
    fn ln(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Log, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn log(&self, base: impl Into<Self>) -> Self {
        self.ln() / base.into().ln()
    }
    fn log2(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Log2, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn log10(&self) -> Self {
        __current_scope(|s| {
            let ret = s.call(Func::Log10, &[self.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn powf(&self, exp: impl Into<Self>) -> Self {
        let exp = exp.into();
        __current_scope(|s| {
            let ret = s.call(Func::Powf, &[self.node(), exp.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn sqr(&self) -> Self {
        *self * *self
    }
    fn cube(&self) -> Self {
        *self * *self * *self
    }
    fn powi(&self, exp: impl Into<Self::Int>) -> Self {
        let exp = exp.into();
        __current_scope(|s| {
            let ret = s.call(Func::Powi, &[self.node(), exp.node()], Self::type_());
            Self::from_node(ret)
        })
    }
    fn lerp(&self, other: impl Into<Self>, frac: impl Into<Self>) -> Self {
        let other = other.into();
        let frac = frac.into();
        __current_scope(|s| {
            let ret = s.call(
                Func::Lerp,
                &[self.node(), other.node(), frac.node()],
                Self::type_(),
            );
            Self::from_node(ret)
        })
    }
    fn recip(&self) -> Self {
        Self::one() / self.clone()
    }
    fn sin_cos(&self) -> (Self, Self) {
        (self.sin(), self.cos())
    }
}

pub trait ScalarVarTrait: ToNode + FromNode {}
pub trait VectorVarTrait: ToNode + FromNode {}
pub trait MatrixVarTrait: ToNode + FromNode {}
pub trait ScalarOrVector: ToNode + FromNode {
    type Element: ScalarVarTrait;
    type ElementHost: Value;
}
pub trait BuiltinVarTrait: ToNode + FromNode {}
