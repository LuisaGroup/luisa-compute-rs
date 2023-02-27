use std::ops::Mul;

pub use super::swizzle::*;
use super::{Aggregate, ExprProxy, Value, VarProxy, __extract, traits::*, Float};
use crate::prelude::FromNode;
use crate::prelude::{__compose, __insert, const_, __current_scope, Expr, PrimExpr, Var};
use luisa_compute_ir::{
    context::register_type,
    ir::{Func, MatrixType, NodeRef, Primitive, Type, VectorElementType, VectorType},
    TypeOf,
};
macro_rules! def_vec {
    ($name:ident, $scalar:ty, $align:literal, $($comp:ident), *) => {
        #[repr(C, align($align))]
        #[derive(Copy, Clone, Debug, Default)]
        pub struct $name {
            $(pub $comp: $scalar), *
        }
        impl $name {
            #[inline]
            pub fn new($($comp: $scalar), *) -> Self {
                Self { $($comp), * }
            }
            #[inline]
            pub fn splat(scalar: $scalar) -> Self {
                Self { $($comp: scalar), * }
            }
        }
        impl From<$name> for glam::$name {
            #[inline]
            fn from(v: $name) -> Self {
                Self::new($(v.$comp), *)
            }
        }
        impl From<glam::$name> for $name {
            #[inline]
            fn from(v: glam::$name) -> Self {
                Self::new($(v.$comp), *)
            }
        }
    };
}
macro_rules! def_vec_long {
    ($name:ident, $scalar:ty, $align:literal, $($comp:ident), *) => {
        #[repr(C, align($align))]
        #[derive(Copy, Clone, Debug, Default)]
        pub struct $name {
            $(pub $comp: $scalar), *
        }
        impl $name {
            #[inline]
            pub fn new($($comp: $scalar), *) -> Self {
                Self { $($comp), * }
            }
            #[inline]
            pub fn splat(scalar: $scalar) -> Self {
                Self { $($comp: scalar), * }
            }
        }
    };
}
def_vec!(Vec2, f32, 8, x, y);
def_vec!(Vec3, f32, 16, x, y, z);
def_vec!(Vec4, f32, 16, x, y, z, w);

def_vec!(UVec2, u32, 8, x, y);
def_vec!(UVec3, u32, 16, x, y, z);
def_vec!(UVec4, u32, 16, x, y, z, w);

def_vec!(IVec2, i32, 8, x, y);
def_vec!(IVec3, i32, 16, x, y, z);
def_vec!(IVec4, i32, 16, x, y, z, w);

def_vec!(DVec2, f64, 16, x, y);
def_vec!(DVec3, f64, 32, x, y, z);
def_vec!(DVec4, f64, 32, x, y, z, w);

def_vec!(BVec2, bool, 2, x, y);
def_vec!(BVec3, bool, 4, x, y, z);
def_vec!(BVec4, bool, 4, x, y, z, w);

def_vec_long!(ULVec2, u64, 16, x, y);
def_vec_long!(ULVec3, u64, 32, x, y, z);
def_vec_long!(ULVec4, u64, 32, x, y, z, w);

def_vec_long!(LVec2, i64, 16, x, y);
def_vec_long!(LVec3, i64, 32, x, y, z);
def_vec_long!(LVec4, i64, 32, x, y, z, w);

#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(8))]
pub struct Mat2 {
    pub cols: [Vec2; 2],
}
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
pub struct Mat3 {
    pub cols: [Vec3; 3],
}
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
pub struct Mat4 {
    pub cols: [Vec4; 4],
}
impl Mat4 {
    pub fn into_affine3x4(&self)->[f32;12] {
        [
            self.cols[0].x, self.cols[0].y, self.cols[0].z,
            self.cols[1].x, self.cols[1].y, self.cols[1].z,
            self.cols[2].x, self.cols[2].y, self.cols[2].z,
            self.cols[3].x, self.cols[3].y, self.cols[3].z,
        ]
    }
}
impl From<Mat2> for glam::Mat2 {
    #[inline]
    fn from(m: Mat2) -> Self {
        Self::from_cols(m.cols[0].into(), m.cols[1].into())
    }
}
impl From<Mat3> for glam::Mat3 {
    #[inline]
    fn from(m: Mat3) -> Self {
        Self::from_cols(m.cols[0].into(), m.cols[1].into(), m.cols[2].into())
    }
}
impl From<Mat4> for glam::Mat4 {
    #[inline]
    fn from(m: Mat4) -> Self {
        Self::from_cols(
            m.cols[0].into(),
            m.cols[1].into(),
            m.cols[2].into(),
            m.cols[3].into(),
        )
    }
}
impl From<glam::Mat2> for Mat2 {
    #[inline]
    fn from(m: glam::Mat2) -> Self {
        Self {
            cols: [m.x_axis.into(), m.y_axis.into()],
        }
    }
}
impl From<glam::Mat3> for Mat3 {
    #[inline]
    fn from(m: glam::Mat3) -> Self {
        Self {
            cols: [m.x_axis.into(), m.y_axis.into(), m.z_axis.into()],
        }
    }
}
impl From<glam::Mat4> for Mat4 {
    #[inline]
    fn from(m: glam::Mat4) -> Self {
        Self {
            cols: [
                m.x_axis.into(),
                m.y_axis.into(),
                m.z_axis.into(),
                m.w_axis.into(),
            ],
        }
    }
}

macro_rules! impl_proxy_fields {
    ($vec:ident, $proxy:ident, $scalar:ty, x) => {
        impl $proxy {
            #[inline]
            pub fn x(&self) -> Expr<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, 0))
            }
            #[inline]
            pub fn set_x(&self, value: Expr<$scalar>) -> Self {
                Self::from_node(__insert::<$vec>(self.node, 0, FromNode::node(&value)))
            }
        }
    };
    ($vec:ident,$proxy:ident, $scalar:ty, y) => {
        impl $proxy {
            #[inline]
            pub fn y(&self) -> Expr<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, 1))
            }
            #[inline]
            pub fn set_y(&self, value: Expr<$scalar>) -> Self {
                Self::from_node(__insert::<$vec>(self.node, 1, FromNode::node(&value)))
            }
        }
    };
    ($vec:ident,$proxy:ident, $scalar:ty, z) => {
        impl $proxy {
            #[inline]
            pub fn z(&self) -> Expr<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, 2))
            }
            #[inline]
            pub fn set_z(&self, value: Expr<$scalar>) -> Self {
                Self::from_node(__insert::<$vec>(self.node, 2, FromNode::node(&value)))
            }
        }
    };
    ($vec:ident,$proxy:ident, $scalar:ty, w) => {
        impl $proxy {
            #[inline]
            pub fn w(&self) -> Expr<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, 3))
            }
            #[inline]
            pub fn set_w(&self, value: Expr<$scalar>) -> Self {
                Self::from_node(__insert::<$vec>(self.node, 3, FromNode::node(&value)))
            }
        }
    };
}
macro_rules! impl_var_proxy_fields {
    ($proxy:ident, $scalar:ty, x) => {
        impl $proxy {
            #[inline]
            pub fn x(&self) -> Var<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, 0))
            }
        }
    };
    ($proxy:ident, $scalar:ty, y) => {
        impl $proxy {
            #[inline]
            pub fn y(&self) -> Var<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, 1))
            }
        }
    };
    ($proxy:ident, $scalar:ty, z) => {
        impl $proxy {
            #[inline]
            pub fn z(&self) -> Var<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, 2))
            }
        }
    };
    ($proxy:ident, $scalar:ty, w) => {
        impl $proxy {
            #[inline]
            pub fn w(&self) -> Var<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, 3))
            }
        }
    };
}
macro_rules! impl_vec_proxy {
    ($vec:ident, $expr_proxy:ident, $var_proxy:ident, $scalar:ty, $scalar_ty:ident, $length:literal, $($comp:ident), *) => {
        #[derive(Clone, Copy)]
        pub struct $expr_proxy {
            node: NodeRef,
        }
        #[derive(Clone, Copy)]
        pub struct $var_proxy {
            node: NodeRef,
        }
        impl Value for $vec {
            type Expr = $expr_proxy;
            type Var = $var_proxy;
            fn fields() -> Vec<String> {
                vec![$(stringify!($comp).to_string()),*]
            }
        }
        impl TypeOf for $vec {
            fn type_() -> luisa_compute_ir::CArc<luisa_compute_ir::ir::Type> {
                let type_ = Type::Vector(VectorType {
                    element: VectorElementType::Scalar(Primitive::$scalar_ty),
                    length: $length,
                });
                register_type(type_)
            }
        }
        impl Aggregate for $expr_proxy {
            fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
                nodes.push(self.node);
            }
            fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
                Self {
                    node: iter.next().unwrap(),
                }
            }
        }
        impl Aggregate for $var_proxy {
            fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
                nodes.push(self.node);
            }
            fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
                Self {
                    node: iter.next().unwrap(),
                }
            }
        }
        impl FromNode for $expr_proxy {
            fn from_node(node: NodeRef) -> Self {
                Self { node }
            }
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl FromNode for $var_proxy {
            fn from_node(node: NodeRef) -> Self {
                Self { node }
            }
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl ExprProxy<$vec> for $expr_proxy {
            type Elem = $vec;
        }
        impl VarProxy<$vec> for $var_proxy {
            type Elem = $vec;
        }
        impl From<$var_proxy> for $expr_proxy {
            fn from(var: $var_proxy) -> Self {
                var.load()
            }
        }
        $(impl_proxy_fields!($vec, $expr_proxy, $scalar, $comp);)*
        $(impl_var_proxy_fields!($var_proxy, $scalar, $comp);)*
        impl $expr_proxy {
            #[inline]
            pub fn new($($comp: Expr<$scalar>), *) -> Self {
                Self {
                    node: __compose::<$vec>(&[$(FromNode::node(&$comp)), *]),
                }
            }
        }
    };
}

macro_rules! impl_mat_proxy {
    ($mat:ident, $expr_proxy:ident, $var_proxy:ident, $vec:ty, $scalar_ty:ident, $length:literal, $($comp:ident), *) => {
        #[derive(Clone, Copy)]
        pub struct $expr_proxy {
            node: NodeRef,
        }
        #[derive(Clone, Copy)]
        pub struct $var_proxy {
            node: NodeRef,
        }
        impl Value for $mat {
            type Expr = $expr_proxy;
            type Var = $var_proxy;
            fn fields() -> Vec<String> {
                vec![$(stringify!($comp).to_string()),*]
            }
        }
        impl TypeOf for $mat {
            fn type_() -> luisa_compute_ir::CArc<luisa_compute_ir::ir::Type> {
                let type_ = Type::Matrix(MatrixType {
                    element: VectorElementType::Scalar(Primitive::$scalar_ty),
                    dimension: $length,
                });
                register_type(type_)
            }
        }
        impl Aggregate for $expr_proxy {
            fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
                nodes.push(self.node);
            }
            fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
                Self {
                    node: iter.next().unwrap(),
                }
            }
        }
        impl Aggregate for $var_proxy {
            fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
                nodes.push(self.node);
            }
            fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
                Self {
                    node: iter.next().unwrap(),
                }
            }
        }
        impl FromNode for $expr_proxy {
            fn from_node(node: NodeRef) -> Self {
                Self { node }
            }
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl ExprProxy<$mat> for $expr_proxy {
            type Elem = $mat;
        }
        impl FromNode for $var_proxy {
            fn from_node(node: NodeRef) -> Self {
                Self { node }
            }
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl VarProxy<$mat> for $var_proxy {
            type Elem = $mat;
        }
        impl From<$var_proxy> for $expr_proxy {
            fn from(var: $var_proxy) -> Self {
                var.load()
            }
        }
        impl $expr_proxy {
            #[inline]
            pub fn new($($comp: Expr<$vec>), *) -> Self {
                Self {
                    node: __compose::<$mat>(&[$(FromNode::node(&$comp)), *]),
                }
            }
            pub fn col(&self, index: usize) -> Expr<$vec> {
                Expr::<$vec>::from_node(__extract::<$vec>(self.node, index))
            }
        }
    };
}

impl_vec_proxy!(BVec2, BVec2Expr, BVec2Var, bool, Bool, 2, x, y);
impl_vec_proxy!(BVec3, BVec3Expr, BVec3Var, bool, Bool, 3, x, y, z);
impl_vec_proxy!(BVec4, BVec4Expr, BVec4Var, bool, Bool, 4, x, y, z, w);

impl_vec_proxy!(Vec2, Vec2Expr, Vec2Var, f32, Float32, 2, x, y);
impl_vec_proxy!(Vec3, Vec3Expr, Vec3Var, f32, Float32, 3, x, y, z);
impl_vec_proxy!(Vec4, Vec4Expr, Vec4Var, f32, Float32, 4, x, y, z, w);

impl_vec_proxy!(UVec2, UVec2Expr, UVec2Var, u32, Uint32, 2, x, y);
impl_vec_proxy!(UVec3, UVec3Expr, UVec3Var, u32, Uint32, 3, x, y, z);
impl_vec_proxy!(UVec4, UVec4Expr, UVec4Var, u32, Uint32, 4, x, y, z, w);

impl_vec_proxy!(IVec2, IVec2Expr, IVec2Var, i32, Int32, 2, x, y);
impl_vec_proxy!(IVec3, IVec3Expr, IVec3Var, i32, Int32, 3, x, y, z);
impl_vec_proxy!(IVec4, IVec4Expr, IVec4Var, i32, Int32, 4, x, y, z, w);

impl_vec_proxy!(ULVec2, ULVec2Expr, ULVec2Var, u64, Uint64, 2, x, y);
impl_vec_proxy!(ULVec3, ULVec3Expr, ULVec3Var, u64, Uint64, 3, x, y, z);
impl_vec_proxy!(ULVec4, ULVec4Expr, ULVec4Var, u64, Uint64, 4, x, y, z, w);

impl_vec_proxy!(LVec2, LVec2Expr, LVec2Var, i64, Int64, 2, x, y);
impl_vec_proxy!(LVec3, LVec3Expr, LVec3Var, i64, Int64, 3, x, y, z);
impl_vec_proxy!(LVec4, LVec4Expr, LVec4Var, i64, Int64, 4, x, y, z, w);

impl_mat_proxy!(Mat2, Mat2Expr, Mat2Var, Vec2, Float32, 2, x, y);
impl_mat_proxy!(Mat3, Mat3Expr, Mat3Var, Vec3, Float32, 3, x, y, z);
impl_mat_proxy!(Mat4, Mat4Expr, Mat4Var, Vec4, Float32, 4, x, y, z, w);

macro_rules! impl_binop {
    ($t:ty, $scalar:ty, $proxy:ty, $tr:ident, $m:ident, $tr_assign:ident, $m_assign:ident) => {
        impl std::ops::$tr_assign<$proxy> for $proxy {
            fn $m_assign(&mut self, rhs: $proxy) {
                use std::ops::$tr;
                *self = (*self).$m(rhs);
            }
        }
        impl std::ops::$tr for $proxy {
            type Output = $proxy;
            fn $m(self, rhs: $proxy) -> Self::Output {
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::$tr, &[self.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::$tr<$scalar> for $proxy {
            type Output = $proxy;
            fn $m(self, rhs: $scalar) -> Self::Output {
                let rhs = Self::splat(rhs);
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::$tr, &[self.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::$tr<$proxy> for $scalar {
            type Output = $proxy;
            fn $m(self, rhs: $proxy) -> Self::Output {
                let lhs = <$proxy>::splat(self);
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::$tr, &[lhs.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::$tr<PrimExpr<$scalar>> for $proxy {
            type Output = $proxy;
            fn $m(self, rhs: PrimExpr<$scalar>) -> Self::Output {
                let rhs = Self::splat(rhs);
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::$tr, &[self.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::$tr<$proxy> for PrimExpr<$scalar> {
            type Output = $proxy;
            fn $m(self, rhs: $proxy) -> Self::Output {
                let lhs = <$proxy>::splat(self);
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::$tr, &[lhs.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
    };
}
macro_rules! impl_arith_binop {
    ($t:ty, $scalar:ty, $proxy:ty) => {
        impl_common_op!($t, $scalar, $proxy);
        impl_binop!($t, $scalar, $proxy, Add, add, AddAssign, add_assign);
        impl_binop!($t, $scalar, $proxy, Sub, sub, SubAssign, sub_assign);
        impl_binop!($t, $scalar, $proxy, Mul, mul, MulAssign, mul_assign);
        impl_binop!($t, $scalar, $proxy, Div, div, DivAssign, div_assign);
        impl_binop!($t, $scalar, $proxy, Rem, rem, RemAssign, rem_assign);
        impl_reduce!($t, $scalar, $proxy);
        impl std::ops::Neg for $proxy {
            type Output = $proxy;
            fn neg(self) -> Self::Output {
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::Neg, &[self.node], <$t as TypeOf>::type_())
                }))
            }
        }
    };
}
macro_rules! impl_int_binop {
    ($t:ty, $scalar:ty, $proxy:ty) => {
        impl_binop!(
            $t,
            $scalar,
            $proxy,
            BitAnd,
            bitand,
            BitAndAssign,
            bitand_assign
        );
        impl_binop!($t, $scalar, $proxy, BitOr, bitor, BitOrAssign, bitor_assign);
        impl_binop!(
            $t,
            $scalar,
            $proxy,
            BitXor,
            bitxor,
            BitXorAssign,
            bitxor_assign
        );
        impl_binop!($t, $scalar, $proxy, Shl, shl, ShlAssign, shl_assign);
        impl_binop!($t, $scalar, $proxy, Shr, shr, ShrAssign, shr_assign);
        impl std::ops::Not for $proxy {
            type Output = Expr<$t>;
            fn not(self) -> Self::Output {
                __current_scope(|s| {
                    let ret = s.call(
                        Func::BitNot,
                        &[FromNode::node(&self)],
                        Self::Output::type_(),
                    );
                    Expr::<$t>::from_node(ret)
                })
            }
        }
    };
}
macro_rules! impl_bool_binop {
    ($t:ty, $proxy:ty) => {
        impl_binop!(
            $t,
            bool,
            $proxy,
            BitAnd,
            bitand,
            BitAndAssign,
            bitand_assign
        );
        impl_binop!($t, bool, $proxy, BitOr, bitor, BitOrAssign, bitor_assign);
        impl_binop!(
            $t,
            bool,
            $proxy,
            BitXor,
            bitxor,
            BitXorAssign,
            bitxor_assign
        );
        impl $proxy {
            pub fn splat<V: Into<PrimExpr<bool>>>(value: V) -> Self {
                let value = value.into();
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::Vec, &[value.node], <$t as TypeOf>::type_())
                }))
            }
            pub fn zero() -> Self {
                Self::splat(false)
            }
            pub fn one() -> Self {
                Self::splat(true)
            }
            pub fn all(&self) -> Expr<bool> {
                Expr::<bool>::from_node(__current_scope(|s| {
                    s.call(Func::All, &[self.node], <bool as TypeOf>::type_())
                }))
            }
            pub fn any(&self) -> Expr<bool> {
                Expr::<bool>::from_node(__current_scope(|s| {
                    s.call(Func::Any, &[self.node], <bool as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::Not for $proxy {
            type Output = Expr<$t>;
            fn not(self) -> Self::Output {
                self ^ Self::splat(true)
            }
        }
    };
}
macro_rules! impl_reduce {
    ($t:ty, $scalar:ty, $proxy:ty) => {
        impl $proxy {
            #[inline]
            pub fn reduce_sum(&self) -> Expr<$scalar> {
                FromNode::from_node(__current_scope(|s| {
                    s.call(Func::ReduceSum, &[self.node], <$scalar as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn reduce_prod(&self) -> Expr<$scalar> {
                FromNode::from_node(__current_scope(|s| {
                    s.call(Func::ReduceProd, &[self.node], <$scalar as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn reduce_min(&self) -> Expr<$scalar> {
                FromNode::from_node(__current_scope(|s| {
                    s.call(Func::ReduceMin, &[self.node], <$scalar as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn reduce_max(&self) -> Expr<$scalar> {
                FromNode::from_node(__current_scope(|s| {
                    s.call(Func::ReduceMax, &[self.node], <$scalar as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn dot(&self, rhs: $proxy) -> Expr<$scalar> {
                FromNode::from_node(__current_scope(|s| {
                    s.call(
                        Func::Dot,
                        &[self.node, rhs.node],
                        <$scalar as TypeOf>::type_(),
                    )
                }))
            }
        }
    };
}
macro_rules! impl_common_op {
    ($t:ty, $scalar:ty, $proxy:ty) => {
        impl $proxy {
            pub fn splat<V: Into<PrimExpr<$scalar>>>(value: V) -> Self {
                let value = value.into();
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::Vec, &[value.node], <$t as TypeOf>::type_())
                }))
            }
            pub fn zero() -> Self {
                Self::splat(0.0 as $scalar)
            }
            pub fn one() -> Self {
                Self::splat(1.0 as $scalar)
            }
        }
    };
}
macro_rules! impl_vec_op {
    ($t:ty, $scalar:ty, $proxy:ty, $mat:ty) => {
        impl $proxy {
            #[inline]
            pub fn length(&self) -> Expr<$scalar> {
                FromNode::from_node(__current_scope(|s| {
                    s.call(Func::Length, &[self.node], <$scalar as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn normalize(&self) -> Self {
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::Normalize, &[self.node], <$t as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn length_squared(&self) -> Expr<$scalar> {
                FromNode::from_node(__current_scope(|s| {
                    s.call(
                        Func::LengthSquared,
                        &[self.node],
                        <$scalar as TypeOf>::type_(),
                    )
                }))
            }
            #[inline]
            pub fn distance(&self, rhs: $proxy) -> Expr<$scalar> {
                (*self - rhs).length()
            }
            #[inline]
            pub fn distance_squared(&self, rhs: $proxy) -> Expr<$scalar> {
                (*self - rhs).length_squared()
            }
            #[inline]
            pub fn fma(&self, a: $proxy, b: $proxy) -> Self {
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(
                        Func::Fma,
                        &[self.node, a.node, b.node],
                        <$t as TypeOf>::type_(),
                    )
                }))
            }
            #[inline]
            pub fn outer_product(&self, rhs: $proxy) -> Expr<$mat> {
                Expr::<$mat>::from_node(__current_scope(|s| {
                    s.call(
                        Func::OuterProduct,
                        &[self.node, rhs.node],
                        <$mat as TypeOf>::type_(),
                    )
                }))
            }
        }
    };
}
impl_arith_binop!(Vec2, f32, Vec2Expr);
impl_arith_binop!(Vec3, f32, Vec3Expr);
impl_arith_binop!(Vec4, f32, Vec4Expr);

impl_arith_binop!(IVec2, i32, IVec2Expr);
impl_arith_binop!(IVec3, i32, IVec3Expr);
impl_arith_binop!(IVec4, i32, IVec4Expr);

impl_arith_binop!(UVec2, u32, UVec2Expr);
impl_arith_binop!(UVec3, u32, UVec3Expr);
impl_arith_binop!(UVec4, u32, UVec4Expr);

impl_arith_binop!(LVec2, i64, LVec2Expr);
impl_arith_binop!(LVec3, i64, LVec3Expr);
impl_arith_binop!(LVec4, i64, LVec4Expr);

impl_arith_binop!(ULVec2, u64, ULVec2Expr);
impl_arith_binop!(ULVec3, u64, ULVec3Expr);
impl_arith_binop!(ULVec4, u64, ULVec4Expr);

impl_int_binop!(IVec2, i32, IVec2Expr);
impl_int_binop!(IVec3, i32, IVec3Expr);
impl_int_binop!(IVec4, i32, IVec4Expr);

impl_int_binop!(UVec2, u32, UVec2Expr);
impl_int_binop!(UVec3, u32, UVec3Expr);
impl_int_binop!(UVec4, u32, UVec4Expr);

impl_int_binop!(LVec2, i64, LVec2Expr);
impl_int_binop!(LVec3, i64, LVec3Expr);
impl_int_binop!(LVec4, i64, LVec4Expr);

impl_int_binop!(ULVec2, u64, ULVec2Expr);
impl_int_binop!(ULVec3, u64, ULVec3Expr);
impl_int_binop!(ULVec4, u64, ULVec4Expr);

impl_bool_binop!(BVec2, BVec2Expr);
impl_bool_binop!(BVec3, BVec3Expr);
impl_bool_binop!(BVec4, BVec4Expr);
macro_rules! impl_select {
    ($bvec:ty, $vec:ty, $proxy:ty) => {
        impl $proxy {
            pub fn select(mask: Expr<$bvec>, a: Expr<$vec>, b: Expr<$vec>) -> Expr<$vec> {
                Expr::<$vec>::from_node(__current_scope(|s| {
                    s.call(
                        Func::Select,
                        &[mask.node(), a.node(), b.node()],
                        <$vec as TypeOf>::type_(),
                    )
                }))
            }
        }
    };
}

impl_select!(BVec2, BVec2, BVec2Expr);
impl_select!(BVec3, BVec3, BVec3Expr);
impl_select!(BVec4, BVec4, BVec4Expr);

impl_select!(BVec2, Vec2, Vec2Expr);
impl_select!(BVec3, Vec3, Vec3Expr);
impl_select!(BVec4, Vec4, Vec4Expr);

impl_select!(BVec2, IVec2, IVec2Expr);
impl_select!(BVec3, IVec3, IVec3Expr);
impl_select!(BVec4, IVec4, IVec4Expr);

impl_select!(BVec2, UVec2, UVec2Expr);
impl_select!(BVec3, UVec3, UVec3Expr);
impl_select!(BVec4, UVec4, UVec4Expr);

macro_rules! impl_cast {
    ($proxy:ty, $to:ty, $m:ident) => {
        impl $proxy {
            pub fn $m(&self) -> Expr<$to> {
                Expr::<$to>::from_node(__current_scope(|s| {
                    s.call(Func::Cast, &[self.node], <$to as TypeOf>::type_())
                }))
            }
        }
    };
}
impl_cast!(Vec2Expr, IVec2, as_ivec2);
impl_cast!(Vec2Expr, UVec2, as_uvec2);
impl_cast!(Vec3Expr, IVec3, as_ivec3);
impl_cast!(Vec3Expr, UVec3, as_uvec3);
impl_cast!(Vec4Expr, IVec4, as_ivec4);
impl_cast!(Vec4Expr, UVec4, as_uvec4);

impl_cast!(IVec2Expr, Vec2, as_vec2);
impl_cast!(IVec2Expr, UVec2, as_uvec2);
impl_cast!(IVec3Expr, Vec3, as_vec3);
impl_cast!(IVec3Expr, UVec3, as_uvec3);
impl_cast!(IVec4Expr, Vec4, as_vec4);
impl_cast!(IVec4Expr, UVec4, as_uvec4);

impl_cast!(UVec2Expr, Vec2, as_vec2);
impl_cast!(UVec2Expr, IVec2, as_ivec2);
impl_cast!(UVec3Expr, Vec3, as_vec3);
impl_cast!(UVec3Expr, IVec3, as_ivec3);
impl_cast!(UVec4Expr, Vec4, as_vec4);
impl_cast!(UVec4Expr, IVec4, as_ivec4);
macro_rules! impl_permute {
    ($tr:ident, $proxy:ty,$len:expr, $v2:ty, $v3:ty, $v4:ty) => {
        impl $tr for $proxy {
            type Vec2 = Expr<$v2>;
            type Vec3 = Expr<$v3>;
            type Vec4 = Expr<$v4>;
            fn permute2(&self, x: i32, y: i32) -> Self::Vec2 {
                assert!(x < $len);
                assert!(y < $len);
                let x: Expr<i32> = x.into();
                let y: Expr<i32> = y.into();
                Expr::<$v2>::from_node(__current_scope(|s| {
                    s.call(
                        Func::Permute,
                        &[self.node, FromNode::node(&x), FromNode::node(&y)],
                        <$v2 as TypeOf>::type_(),
                    )
                }))
            }
            fn permute3(&self, x: i32, y: i32, z: i32) -> Self::Vec3 {
                assert!(x < $len);
                assert!(y < $len);
                assert!(z < $len);
                let x: Expr<i32> = x.into();
                let y: Expr<i32> = y.into();
                let z: Expr<i32> = z.into();
                Expr::<$v3>::from_node(__current_scope(|s| {
                    s.call(
                        Func::Permute,
                        &[
                            self.node,
                            FromNode::node(&x),
                            FromNode::node(&y),
                            FromNode::node(&z),
                        ],
                        <$v3 as TypeOf>::type_(),
                    )
                }))
            }
            fn permute4(&self, x: i32, y: i32, z: i32, w: i32) -> Self::Vec4 {
                assert!(x < $len);
                assert!(y < $len);
                assert!(z < $len);
                assert!(w < $len);
                let x: Expr<i32> = x.into();
                let y: Expr<i32> = y.into();
                let z: Expr<i32> = z.into();
                let w: Expr<i32> = w.into();
                Expr::<$v4>::from_node(__current_scope(|s| {
                    s.call(
                        Func::Permute,
                        &[
                            self.node,
                            FromNode::node(&x),
                            FromNode::node(&y),
                            FromNode::node(&z),
                            FromNode::node(&w),
                        ],
                        <$v4 as TypeOf>::type_(),
                    )
                }))
            }
        }
    };
}
impl_permute!(Vec2Swizzle, Vec2Expr, 2, Vec2, Vec3, Vec4);
impl_permute!(Vec3Swizzle, Vec3Expr, 3, Vec2, Vec3, Vec4);
impl_permute!(Vec4Swizzle, Vec4Expr, 4, Vec2, Vec3, Vec4);

impl_permute!(Vec2Swizzle, IVec2Expr, 2, IVec2, IVec3, IVec4);
impl_permute!(Vec3Swizzle, IVec3Expr, 3, IVec2, IVec3, IVec4);
impl_permute!(Vec4Swizzle, IVec4Expr, 4, IVec2, IVec3, IVec4);

impl_permute!(Vec2Swizzle, UVec2Expr, 2, UVec2, UVec3, UVec4);
impl_permute!(Vec3Swizzle, UVec3Expr, 3, UVec2, UVec3, UVec4);
impl_permute!(Vec4Swizzle, UVec4Expr, 4, UVec2, UVec3, UVec4);

impl_permute!(Vec2Swizzle, LVec2Expr, 2, LVec2, LVec3, LVec4);
impl_permute!(Vec3Swizzle, LVec3Expr, 3, LVec2, LVec3, LVec4);
impl_permute!(Vec4Swizzle, LVec4Expr, 4, LVec2, LVec3, LVec4);

impl_permute!(Vec2Swizzle, ULVec2Expr, 2, ULVec2, ULVec3, ULVec4);
impl_permute!(Vec3Swizzle, ULVec3Expr, 3, ULVec2, ULVec3, ULVec4);
impl_permute!(Vec4Swizzle, ULVec4Expr, 4, ULVec2, ULVec3, ULVec4);

impl Vec3Expr {
    #[inline]
    pub fn cross(&self, rhs: Vec3Expr) -> Self {
        Vec3Expr::from_node(__current_scope(|s| {
            s.call(
                Func::Cross,
                &[self.node, rhs.node],
                <Vec3 as TypeOf>::type_(),
            )
        }))
    }
}
impl_vec_op!(Vec2, f32, Vec2Expr, Mat2);
impl_vec_op!(Vec3, f32, Vec3Expr, Mat3);
impl_vec_op!(Vec4, f32, Vec4Expr, Mat4);

macro_rules! impl_var_trait2 {
    ($t:ty, $v:ty) => {
        impl VarTrait for $t {
            type Value = $v;
            type Int = IVec2Expr;
            type Uint = UVec2Expr;
            type Float = Vec2Expr;
            type Bool = BVec2Expr;
            // type Double = DVec2Expr;
            type Long = LVec2Expr;
            type Ulong = ULVec2Expr;
        }
        impl CommonVarOp for $t {}
        impl VarCmp for $t {}
        impl VarCmpEq for $t {}
        impl From<$v> for $t {
            fn from(v: $v) -> Self {
                Self::new(const_(v.x), const_(v.y))
            }
        }
    };
}
macro_rules! impl_var_trait3 {
    ($t:ty, $v:ty) => {
        impl VarTrait for $t {
            type Value = $v;
            type Int = IVec3Expr;
            type Uint = UVec3Expr;
            type Float = Vec3Expr;
            type Bool = BVec3Expr;
            type Long = LVec3Expr;
            type Ulong = ULVec3Expr;
        }
        impl CommonVarOp for $t {}
        impl VarCmp for $t {}
        impl VarCmpEq for $t {}
        impl From<$v> for $t {
            fn from(v: $v) -> Self {
                Self::new(const_(v.x), const_(v.y), const_(v.z))
            }
        }
    };
}
macro_rules! impl_var_trait4 {
    ($t:ty, $v:ty) => {
        impl VarTrait for $t {
            type Value = $v;
            type Int = IVec2Expr;
            type Uint = UVec2Expr;
            type Float = Vec2Expr;
            type Bool = BVec2Expr;
            type Long = LVec2Expr;
            type Ulong = ULVec2Expr;
        }
        impl CommonVarOp for $t {}
        impl VarCmp for $t {}
        impl VarCmpEq for $t {}
        impl From<$v> for $t {
            fn from(v: $v) -> Self {
                Self::new(const_(v.x), const_(v.y), const_(v.z), const_(v.w))
            }
        }
    };
}
impl_var_trait2!(Vec2Expr, Vec2);
impl_var_trait2!(IVec2Expr, IVec2);
impl_var_trait2!(UVec2Expr, UVec2);
impl_var_trait2!(BVec2Expr, BVec2);
impl_var_trait2!(LVec2Expr, LVec2);
impl_var_trait2!(ULVec2Expr, ULVec2);

impl_var_trait3!(Vec3Expr, Vec3);
impl_var_trait3!(IVec3Expr, IVec3);
impl_var_trait3!(UVec3Expr, UVec3);
impl_var_trait3!(BVec3Expr, BVec3);
impl_var_trait3!(LVec3Expr, LVec3);
impl_var_trait3!(ULVec3Expr, ULVec3);

impl_var_trait4!(Vec4Expr, Vec4);
impl_var_trait4!(IVec4Expr, IVec4);
impl_var_trait4!(UVec4Expr, UVec4);
impl_var_trait4!(BVec4Expr, BVec4);
impl_var_trait4!(LVec4Expr, LVec4);
impl_var_trait4!(ULVec4Expr, ULVec4);

macro_rules! impl_float_trait {
    ($t:ty) => {
        impl From<f32> for $t {
            fn from(v: f32) -> Self {
                Self::splat(v)
            }
        }
        impl FloatVarTrait for $t {}
    };
}
impl_float_trait!(Vec2Expr);
impl_float_trait!(Vec3Expr);
impl_float_trait!(Vec4Expr);
macro_rules! impl_int_trait {
    ($t:ty) => {
        impl From<i64> for $t {
            fn from(v: i64) -> Self {
                Self::splat(v)
            }
        }
        impl IntVarTrait for $t {}
    };
}
impl_int_trait!(IVec2Expr);
impl_int_trait!(IVec3Expr);
impl_int_trait!(IVec4Expr);
impl_int_trait!(LVec2Expr);
impl_int_trait!(LVec3Expr);
impl_int_trait!(LVec4Expr);
impl_int_trait!(UVec2Expr);
impl_int_trait!(UVec3Expr);
impl_int_trait!(UVec4Expr);
impl_int_trait!(ULVec2Expr);
impl_int_trait!(ULVec3Expr);
impl_int_trait!(ULVec4Expr);

impl Mul<Vec2Expr> for Mat2Expr {
    type Output = Vec2Expr;
    #[inline]
    fn mul(self, rhs: Vec2Expr) -> Self::Output {
        Vec2Expr::from_node(__current_scope(|s| {
            s.call(Func::Mul, &[self.node, rhs.node], <Vec2 as TypeOf>::type_())
        }))
    }
}
impl Mat2Expr {
    pub fn inverse(&self) -> Self {
        Mat2Expr::from_node(__current_scope(|s| {
            s.call(Func::Inverse, &[self.node], <Mat2 as TypeOf>::type_())
        }))
    }
    pub fn transpose(&self) -> Self {
        Mat2Expr::from_node(__current_scope(|s| {
            s.call(Func::Transpose, &[self.node], <Mat2 as TypeOf>::type_())
        }))
    }
    pub fn determinant(&self) -> Float {
        FromNode::from_node(__current_scope(|s| {
            s.call(Func::Determinant, &[self.node], <f32 as TypeOf>::type_())
        }))
    }
}
impl Mul<Vec3Expr> for Mat3Expr {
    type Output = Vec3Expr;
    #[inline]
    fn mul(self, rhs: Vec3Expr) -> Self::Output {
        Vec3Expr::from_node(__current_scope(|s| {
            s.call(Func::Mul, &[self.node, rhs.node], <Vec3 as TypeOf>::type_())
        }))
    }
}
impl Mat3Expr {
    pub fn inverse(&self) -> Self {
        Self::from_node(__current_scope(|s| {
            s.call(Func::Inverse, &[self.node], <Mat3 as TypeOf>::type_())
        }))
    }
    pub fn transpose(&self) -> Self {
        Self::from_node(__current_scope(|s| {
            s.call(Func::Transpose, &[self.node], <Mat3 as TypeOf>::type_())
        }))
    }
    pub fn determinant(&self) -> Float {
        FromNode::from_node(__current_scope(|s| {
            s.call(Func::Determinant, &[self.node], <f32 as TypeOf>::type_())
        }))
    }
}
impl Mul<Vec4Expr> for Mat4Expr {
    type Output = Vec4Expr;
    #[inline]
    fn mul(self, rhs: Vec4Expr) -> Self::Output {
        Vec4Expr::from_node(__current_scope(|s| {
            s.call(Func::Mul, &[self.node, rhs.node], <Vec4 as TypeOf>::type_())
        }))
    }
}
impl Mul for Mat2Expr {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_node(__current_scope(|s| {
            s.call(Func::Mul, &[self.node, rhs.node], <Mat2 as TypeOf>::type_())
        }))
    }
}
impl Mul for Mat3Expr {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_node(__current_scope(|s| {
            s.call(Func::Mul, &[self.node, rhs.node], <Mat3 as TypeOf>::type_())
        }))
    }
}
impl Mul for Mat4Expr {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_node(__current_scope(|s| {
            s.call(Func::Mul, &[self.node, rhs.node], <Mat4 as TypeOf>::type_())
        }))
    }
}
impl Mat4Expr {
    pub fn inverse(&self) -> Self {
        Self::from_node(__current_scope(|s| {
            s.call(Func::Inverse, &[self.node], <Mat4 as TypeOf>::type_())
        }))
    }
    pub fn transpose(&self) -> Self {
        Self::from_node(__current_scope(|s| {
            s.call(Func::Transpose, &[self.node], <Mat4 as TypeOf>::type_())
        }))
    }
    pub fn determinant(&self) -> Float {
        FromNode::from_node(__current_scope(|s| {
            s.call(Func::Determinant, &[self.node], <f32 as TypeOf>::type_())
        }))
    }
}
#[inline]
pub fn make_float2<X: Into<PrimExpr<f32>>, Y: Into<PrimExpr<f32>>>(x: X, y: Y) -> Expr<Vec2> {
    Expr::<Vec2>::new(x.into(), y.into())
}
#[inline]
pub fn make_float3<X: Into<PrimExpr<f32>>, Y: Into<PrimExpr<f32>>, Z: Into<PrimExpr<f32>>>(
    x: X,
    y: Y,
    z: Z,
) -> Expr<Vec3> {
    Expr::<Vec3>::new(x.into(), y.into(), z.into())
}
#[inline]
pub fn make_float4<
    X: Into<PrimExpr<f32>>,
    Y: Into<PrimExpr<f32>>,
    Z: Into<PrimExpr<f32>>,
    W: Into<PrimExpr<f32>>,
>(
    x: X,
    y: Y,
    z: Z,
    w: W,
) -> Expr<Vec4> {
    Expr::<Vec4>::new(x.into(), y.into(), z.into(), w.into())
}

#[inline]
pub fn make_int2<X: Into<PrimExpr<i32>>, Y: Into<PrimExpr<i32>>>(x: X, y: Y) -> Expr<IVec2> {
    Expr::<IVec2>::new(x.into(), y.into())
}
#[inline]
pub fn make_int3<X: Into<PrimExpr<i32>>, Y: Into<PrimExpr<i32>>, Z: Into<PrimExpr<i32>>>(
    x: X,
    y: Y,
    z: Z,
) -> Expr<IVec3> {
    Expr::<IVec3>::new(x.into(), y.into(), z.into())
}
#[inline]
pub fn make_int4<
    X: Into<PrimExpr<i32>>,
    Y: Into<PrimExpr<i32>>,
    Z: Into<PrimExpr<i32>>,
    W: Into<PrimExpr<i32>>,
>(
    x: X,
    y: Y,
    z: Z,
    w: W,
) -> Expr<IVec4> {
    Expr::<IVec4>::new(x.into(), y.into(), z.into(), w.into())
}
#[inline]
pub fn make_uint2<X: Into<PrimExpr<u32>>, Y: Into<PrimExpr<u32>>>(x: X, y: Y) -> Expr<UVec2> {
    Expr::<UVec2>::new(x.into(), y.into())
}
#[inline]
pub fn make_uint3<X: Into<PrimExpr<u32>>, Y: Into<PrimExpr<u32>>, Z: Into<PrimExpr<u32>>>(
    x: X,
    y: Y,
    z: Z,
) -> Expr<UVec3> {
    Expr::<UVec3>::new(x.into(), y.into(), z.into())
}
#[inline]
pub fn make_uint4<
    X: Into<PrimExpr<u32>>,
    Y: Into<PrimExpr<u32>>,
    Z: Into<PrimExpr<u32>>,
    W: Into<PrimExpr<u32>>,
>(
    x: X,
    y: Y,
    z: Z,
    w: W,
) -> Expr<UVec4> {
    Expr::<UVec4>::new(x.into(), y.into(), z.into(), w.into())
}
