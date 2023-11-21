use crate::internal_prelude::*;

pub trait CurveEvaluator {
    fn position(&self, u: impl AsExpr<Value = f32>) -> Expr<Float4>;
    fn derivative(&self, u: impl AsExpr<Value = f32>) -> Expr<Float4>;
    fn second_derivative(&self, u: impl AsExpr<Value = f32>) -> Expr<Float4>;
    /// u: curve parameter
    /// ps: hit point on the surface
    #[tracked]
    fn surface_position_and_normal(
        &self,
        u: impl AsExpr<Value = f32>,
        ps: impl AsExpr<Value = Float3>,
    ) -> (Expr<Float3>, Expr<Float3>) {
        let ps: Var<Float3> = ps.as_expr().var();
        let normal = Var::<Float3>::zeroed();
        let u: Expr<f32> = u.as_expr();
        outline(|| {
            if u == 0.0f32 {
                *normal = -self.derivative(0.0f32).xyz();
            } else if u == 1.0 {
                *normal = self.derivative(1.0f32).xyz();
            } else {
                let p4 = self.position(u);
                let p: Expr<Float3> = p4.xyz();
                let r: Expr<f32> = p4.w;
                let d4 = self.derivative(u);
                let d: Expr<Float3> = d4.xyz();
                let dr = d4.w;
                let dd = d.dot(d).var();
                let o1 = (ps - p).var();
                *o1 -= o1.dot(d) / dd * d;
                *o1 *= r / o1.length();
                *ps = p + o1;
                *dd -= self.second_derivative(u).xyz().dot(o1);
                *normal = dd * o1 - (dr * r) * d;
            }
        });
        (ps.load(), normal.normalize())
    }
}

pub struct PiecewiseLinearCurve {
    p0: Expr<Float4>,
    p1: Expr<Float4>,
}
impl CurveEvaluator for PiecewiseLinearCurve {
    #[tracked]
    fn position(&self, u: impl AsExpr<Value = f32>) -> Expr<Float4> {
        self.p0 + u.as_expr() * self.p1
    }
    #[tracked]
    fn derivative(&self, _u: impl AsExpr<Value = f32>) -> Expr<Float4> {
        self.p1
    }
    #[tracked]
    fn second_derivative(&self, _u: impl AsExpr<Value = f32>) -> Expr<Float4> {
        Expr::zeroed()
    }
}

pub struct CubicCurve {
    p0: Expr<Float4>,
    p1: Expr<Float4>,
    p2: Expr<Float4>,
    p3: Expr<Float4>,
}
impl CurveEvaluator for CubicCurve {
    #[tracked]
    fn position(&self, u: impl AsExpr<Value = f32>) -> Expr<Float4> {
        let u = u.as_expr();
        (((self.p0 * u) + self.p1) * u + self.p2) * u + self.p3
    }
    #[tracked]
    fn derivative(&self, u: impl AsExpr<Value = f32>) -> Expr<Float4> {
        let u = u.as_expr();
        let u = u.clamp(1e-6f32.expr(), (1.0f32 - 1e-6f32).expr());
        ((3.0 * self.p0 * u) + 2.0 * self.p1) * u + self.p2
    }
    #[tracked]
    fn second_derivative(&self, u: impl AsExpr<Value = f32>) -> Expr<Float4> {
        let u = u.as_expr();
        let u = u.clamp(1e-6f32.expr(), (1.0f32 - 1e-6f32).expr());
        6.0 * self.p0 * u + 2.0 * self.p1
    }
}
impl CubicCurve {
    #[tracked]
    pub fn bspline(
        p0: impl AsExpr<Value = Float4>,
        p1: impl AsExpr<Value = Float4>,
        p2: impl AsExpr<Value = Float4>,
        p3: impl AsExpr<Value = Float4>,
    ) -> Self {
        let (q0, q1, q2, q3) = (p0.as_expr(), p1.as_expr(), p2.as_expr(), p3.as_expr());
        let p1 = (q0 * (-1.0) + q1 * (3.0) + q2 * (-3.0) + q3) / 6.0;
        let p2 = (q0 * (3.0) + q1 * (-6.0) + q2 * (3.0)) / 6.0;
        let p3 = (q0 * (-3.0) + q2 * (3.0)) / 6.0;
        let p4 = (q0 * (1.0) + q1 * (4.0) + q2 * (1.0)) / 6.0;
        Self {
            p0: p1,
            p1: p2,
            p2: p3,
            p3: p4,
        }
    }
    #[tracked]
    pub fn catmull_rom(
        p0: impl AsExpr<Value = Float4>,
        p1: impl AsExpr<Value = Float4>,
        p2: impl AsExpr<Value = Float4>,
        p3: impl AsExpr<Value = Float4>,
    ) -> Self {
        let (q0, q1, q2, q3) = (p0.as_expr(), p1.as_expr(), p2.as_expr(), p3.as_expr());
        let p0 = (-1.0 * q0 + (3.0) * q1 + (-3.0) * q2 + (1.0) * q3) / 2.0;
        let p1 = (2.0 * q0 + (-5.0) * q1 + (4.0) * q2 + (-1.0) * q3) / 2.0;
        let p2 = (-1.0 * q0 + (1.0) * q2) / 2.0;
        let p3 = ((2.0) * q1) / 2.0;
        Self { p0, p1, p2, p3 }
    }
    #[tracked]
    pub fn bezier(
        p0: impl AsExpr<Value = Float4>,
        p1: impl AsExpr<Value = Float4>,
        p2: impl AsExpr<Value = Float4>,
        p3: impl AsExpr<Value = Float4>,
    ) -> Self {
        let (q0, q1, q2, q3) = (p0.as_expr(), p1.as_expr(), p2.as_expr(), p3.as_expr());
        let p0 = -q0 + 3.0 * q1 - 3.0 * q2 + q3;
        let p1 = 3.0 * q0 - 6.0 * q1 + 3.0 * q2;
        let p2 = -3.0 * q0 + 3.0 * q1;
        let p3 = q0;
        Self { p0, p1, p2, p3 }
    }
}

pub struct CubicBSplineCurve {
    base: CubicCurve,
}
impl CubicBSplineCurve {
    #[tracked]
    pub fn new(
        p0: impl AsExpr<Value = Float4>,
        p1: impl AsExpr<Value = Float4>,
        p2: impl AsExpr<Value = Float4>,
        p3: impl AsExpr<Value = Float4>,
    ) -> Self {
        Self {
            base: CubicCurve::bspline(p0, p1, p2, p3),
        }
    }
}
pub struct CatmullRomCurve {
    base: CubicCurve,
}
impl CatmullRomCurve {
    #[tracked]
    pub fn new(
        p0: impl AsExpr<Value = Float4>,
        p1: impl AsExpr<Value = Float4>,
        p2: impl AsExpr<Value = Float4>,
        p3: impl AsExpr<Value = Float4>,
    ) -> Self {
        Self {
            base: CubicCurve::catmull_rom(p0, p1, p2, p3),
        }
    }
}
pub struct BezierCurve {
    base: CubicCurve,
}
impl BezierCurve {
    #[tracked]
    pub fn new(
        p0: impl AsExpr<Value = Float4>,
        p1: impl AsExpr<Value = Float4>,
        p2: impl AsExpr<Value = Float4>,
        p3: impl AsExpr<Value = Float4>,
    ) -> Self {
        Self {
            base: CubicCurve::bezier(p0, p1, p2, p3),
        }
    }
}
macro_rules! impl_curve {
    ($name:ident, $base:ident) => {
        impl CurveEvaluator for $name {
            #[tracked]
            fn position(&self, u: impl AsExpr<Value = f32>) -> Expr<Float4> {
                self.base.position(u)
            }
            #[tracked]
            fn derivative(&self, u: impl AsExpr<Value = f32>) -> Expr<Float4> {
                self.base.derivative(u)
            }
            #[tracked]
            fn second_derivative(&self, u: impl AsExpr<Value = f32>) -> Expr<Float4> {
                self.base.second_derivative(u)
            }
        }
    };
}
impl_curve!(CubicBSplineCurve, CubicCurve);
impl_curve!(CatmullRomCurve, CubicCurve);
impl_curve!(BezierCurve, CubicCurve);
