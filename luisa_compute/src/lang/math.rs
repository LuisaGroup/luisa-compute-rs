pub use super::math_impl::*;
use super::{Aggregate, ExprProxy, Value, VarProxy, __extract, traits::*};
use crate::prelude::{__compose, const_, current_scope, Expr, PrimProxy, Var};
use luisa_compute_ir::{
    context::register_type,
    ir::{Func, MatrixType, NodeRef, Primitive, Type, VectorElementType, VectorType},
    TypeOf,
};
macro_rules! impl_proxy_fields {
    ($proxy:ident, $scalar:ty, x) => {
        impl $proxy {
            #[inline]
            pub fn x(&self) -> Expr<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(__extract::<$scalar>(self.node, 0))
            }
        }
    };
    ($proxy:ident, $scalar:ty, y) => {
        impl $proxy {
            #[inline]
            pub fn y(&self) -> Expr<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(__extract::<$scalar>(self.node, 1))
            }
        }
    };
    ($proxy:ident, $scalar:ty, z) => {
        impl $proxy {
            #[inline]
            pub fn z(&self) -> Expr<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(__extract::<$scalar>(self.node, 2))
            }
        }
    };
    ($proxy:ident, $scalar:ty, w) => {
        impl $proxy {
            #[inline]
            pub fn w(&self) -> Expr<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(__extract::<$scalar>(self.node, 3))
            }
        }
    };
}
macro_rules! impl_var_proxy_fields {
    ($proxy:ident, $scalar:ty, x) => {
        impl $proxy {
            #[inline]
            pub fn x(&self) -> Var<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(__extract::<$scalar>(self.node, 0))
            }
        }
    };
    ($proxy:ident, $scalar:ty, y) => {
        impl $proxy {
            #[inline]
            pub fn y(&self) -> Var<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(__extract::<$scalar>(self.node, 1))
            }
        }
    };
    ($proxy:ident, $scalar:ty, z) => {
        impl $proxy {
            #[inline]
            pub fn z(&self) -> Var<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(__extract::<$scalar>(self.node, 2))
            }
        }
    };
    ($proxy:ident, $scalar:ty, w) => {
        impl $proxy {
            #[inline]
            pub fn w(&self) -> Var<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(__extract::<$scalar>(self.node, 3))
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
        }
        impl TypeOf for $vec {
            fn type_() -> luisa_compute_ir::Gc<luisa_compute_ir::ir::Type> {
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
        impl ExprProxy<$vec> for $expr_proxy {
            fn from_node(node: NodeRef) -> Self {
                Self { node }
            }
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl VarProxy<$vec> for $var_proxy {
            fn from_node(node: NodeRef) -> Self {
                Self { node }
            }
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl From<$var_proxy> for $expr_proxy {
            fn from(var: $var_proxy) -> Self {
                var.load()
            }
        }
        $(impl_proxy_fields!($expr_proxy, $scalar, $comp);)*
        $(impl_var_proxy_fields!($var_proxy, $scalar, $comp);)*
        impl $expr_proxy {
            #[inline]
            pub fn new($($comp: Expr<$scalar>), *) -> Self {
                Self {
                    node: __compose::<$vec>(&[$(VarTrait::node(&$comp)), *]),
                }
            }
        }
    };
}

macro_rules! impl_mat_proxy {
    ($mat:ident, $expr_proxy:ident, $var_proxy:ident, $vec:ty, $scalar_ty:ident, $length:literal) => {
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
        }
        impl TypeOf for $mat {
            fn type_() -> luisa_compute_ir::Gc<luisa_compute_ir::ir::Type> {
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
        impl ExprProxy<$mat> for $expr_proxy {
            fn from_node(node: NodeRef) -> Self {
                Self { node }
            }
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl VarProxy<$mat> for $var_proxy {
            fn from_node(node: NodeRef) -> Self {
                Self { node }
            }
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl From<$var_proxy> for $expr_proxy {
            fn from(var: $var_proxy) -> Self {
                var.load()
            }
        }
        impl $expr_proxy {
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

impl_mat_proxy!(Mat2, Mat2Expr, Mat2Var, Vec2, Float32, 2);
impl_mat_proxy!(Mat3, Mat3Expr, Mat3Var, Vec3, Float32, 3);
impl_mat_proxy!(Mat4, Mat4Expr, Mat4Var, Vec4, Float32, 4);

macro_rules! impl_binop {
    ($t:ty, $scalar:ty, $proxy:ty, $tr:ident, $m:ident) => {
        impl std::ops::$tr for $proxy {
            type Output = $proxy;
            fn $m(self, rhs: $proxy) -> Self::Output {
                <$proxy>::from_node(current_scope(|s| {
                    s.call(Func::$tr, &[self.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::$tr<$scalar> for $proxy {
            type Output = $proxy;
            fn $m(self, rhs: $scalar) -> Self::Output {
                let rhs = Self::splat(rhs);
                <$proxy>::from_node(current_scope(|s| {
                    s.call(Func::$tr, &[self.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::$tr<$proxy> for $scalar {
            type Output = $proxy;
            fn $m(self, rhs: $proxy) -> Self::Output {
                let lhs = <$proxy>::splat(self);
                <$proxy>::from_node(current_scope(|s| {
                    s.call(Func::$tr, &[lhs.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::$tr<PrimProxy<$scalar>> for $proxy {
            type Output = $proxy;
            fn $m(self, rhs: PrimProxy<$scalar>) -> Self::Output {
                let rhs = Self::splat(rhs);
                <$proxy>::from_node(current_scope(|s| {
                    s.call(Func::$tr, &[self.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::$tr<$proxy> for PrimProxy<$scalar> {
            type Output = $proxy;
            fn $m(self, rhs: $proxy) -> Self::Output {
                let lhs = <$proxy>::splat(self);
                <$proxy>::from_node(current_scope(|s| {
                    s.call(Func::$tr, &[lhs.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
    };
}
macro_rules! impl_arith_binop {
    ($t:ty, $scalar:ty, $proxy:ty) => {
        impl_common_op!($t, $scalar, $proxy);
        impl_binop!($t, $scalar, $proxy, Add, add);
        impl_binop!($t, $scalar, $proxy, Sub, sub);
        impl_binop!($t, $scalar, $proxy, Mul, mul);
        impl_binop!($t, $scalar, $proxy, Div, div);
        impl_binop!($t, $scalar, $proxy, Rem, rem);
        impl_reduce!($t, $scalar, $proxy);
    };
}
macro_rules! impl_int_binop {
    ($t:ty, $scalar:ty, $proxy:ty) => {
        impl_binop!($t, $scalar, $proxy, BitAnd, bitand);
        impl_binop!($t, $scalar, $proxy, BitOr, bitor);
        impl_binop!($t, $scalar, $proxy, BitXor, bitxor);
        impl_binop!($t, $scalar, $proxy, Shl, shl);
        impl_binop!($t, $scalar, $proxy, Shr, shr);
    };
}
macro_rules! impl_bool_binop {
    ($t:ty, $proxy:ty) => {
        impl_binop!($t, bool, $proxy, BitAnd, bitand);
        impl_binop!($t, bool, $proxy, BitOr, bitor);
        impl_binop!($t, bool, $proxy, BitXor, bitxor);
        impl $proxy {
            pub fn splat<V: Into<PrimProxy<bool>>>(value: V) -> Self {
                let value = value.into();
                <$proxy>::from_node(current_scope(|s| {
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
                <PrimProxy<bool> as VarTrait>::from_node(current_scope(|s| {
                    s.call(Func::All, &[self.node], <bool as TypeOf>::type_())
                }))
            }
            pub fn any(&self) -> Expr<bool> {
                <PrimProxy<bool> as VarTrait>::from_node(current_scope(|s| {
                    s.call(Func::Any, &[self.node], <bool as TypeOf>::type_())
                }))
            }
        }
    };
}
macro_rules! impl_reduce {
    ($t:ty, $scalar:ty, $proxy:ty) => {
        impl $proxy {
            #[inline]
            pub fn reduce_sum(&self) -> Expr<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(current_scope(|s| {
                    s.call(Func::ReduceSum, &[self.node], <$scalar as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn reduce_prod(&self) -> Expr<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(current_scope(|s| {
                    s.call(Func::ReduceProd, &[self.node], <$scalar as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn reduce_min(&self) -> Expr<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(current_scope(|s| {
                    s.call(Func::ReduceMin, &[self.node], <$scalar as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn reduce_max(&self) -> Expr<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(current_scope(|s| {
                    s.call(Func::ReduceMax, &[self.node], <$scalar as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn dot(&self, rhs: $proxy) -> Expr<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(current_scope(|s| {
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
macro_rules! impl_cmp {
    ($t:ty, $scalar:ty, $proxy:ty, $mask:ty) => {
        impl ArithCmp for $proxy {
            type Output = Expr<$mask>;
            fn cmplt<T: Into<Self>>(self, rhs: T) -> Self::Output {
                let rhs = rhs.into();
                Expr::<$mask>::from_node(current_scope(|s| {
                    s.call(Func::Lt, &[self.node, rhs.node], <$mask as TypeOf>::type_())
                }))
            }
            fn cmple<T: Into<Self>>(self, rhs: T) -> Self::Output {
                let rhs = rhs.into();
                Expr::<$mask>::from_node(current_scope(|s| {
                    s.call(Func::Le, &[self.node, rhs.node], <$mask as TypeOf>::type_())
                }))
            }
            fn cmpgt<T: Into<Self>>(self, rhs: T) -> Self::Output {
                let rhs = rhs.into();
                Expr::<$mask>::from_node(current_scope(|s| {
                    s.call(Func::Gt, &[self.node, rhs.node], <$mask as TypeOf>::type_())
                }))
            }
            fn cmpge<T: Into<Self>>(self, rhs: T) -> Self::Output {
                let rhs = rhs.into();
                Expr::<$mask>::from_node(current_scope(|s| {
                    s.call(Func::Ge, &[self.node, rhs.node], <$mask as TypeOf>::type_())
                }))
            }
            fn cmpne<T: Into<Self>>(self, rhs: T) -> Self::Output {
                let rhs = rhs.into();
                Expr::<$mask>::from_node(current_scope(|s| {
                    s.call(Func::Ne, &[self.node, rhs.node], <$mask as TypeOf>::type_())
                }))
            }
            fn cmpeq<T: Into<Self>>(self, rhs: T) -> Self::Output {
                let rhs = rhs.into();
                Expr::<$mask>::from_node(current_scope(|s| {
                    s.call(Func::Eq, &[self.node, rhs.node], <$mask as TypeOf>::type_())
                }))
            }
        }
    };
}
macro_rules! impl_common_op {
    ($t:ty, $scalar:ty, $proxy:ty) => {
        impl $proxy {
            pub fn splat<V: Into<PrimProxy<$scalar>>>(value: V) -> Self {
                let value = value.into();
                <$proxy>::from_node(current_scope(|s| {
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
                <PrimProxy<$scalar> as VarTrait>::from_node(current_scope(|s| {
                    s.call(Func::Length, &[self.node], <$scalar as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn normalize(&self) -> Self {
                <$proxy>::from_node(current_scope(|s| {
                    s.call(Func::Normalize, &[self.node], <$t as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn length_squared(&self) -> Expr<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(current_scope(|s| {
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
                <$proxy>::from_node(current_scope(|s| {
                    s.call(
                        Func::Fma,
                        &[self.node, a.node, b.node],
                        <$t as TypeOf>::type_(),
                    )
                }))
            }
            #[inline]
            pub fn outer_product(&self, rhs: $proxy) -> Expr<$mat> {
                Expr::<$mat>::from_node(current_scope(|s| {
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

impl_int_binop!(IVec2, i32, IVec2Expr);
impl_int_binop!(IVec3, i32, IVec3Expr);
impl_int_binop!(IVec4, i32, IVec4Expr);

impl_int_binop!(UVec2, u32, UVec2Expr);
impl_int_binop!(UVec3, u32, UVec3Expr);
impl_int_binop!(UVec4, u32, UVec4Expr);

impl_cmp!(Vec2, f32, Vec2Expr, BVec2);
impl_cmp!(Vec3, f32, Vec3Expr, BVec3);
impl_cmp!(Vec4, f32, Vec4Expr, BVec4);

impl_cmp!(IVec2, i32, IVec2Expr, BVec2);
impl_cmp!(IVec3, i32, IVec3Expr, BVec3);
impl_cmp!(IVec4, i32, IVec4Expr, BVec4);

impl_cmp!(UVec2, u32, UVec2Expr, BVec2);
impl_cmp!(UVec3, u32, UVec3Expr, BVec3);
impl_cmp!(UVec4, u32, UVec4Expr, BVec4);

impl_bool_binop!(BVec2, BVec2Expr);
impl_bool_binop!(BVec3, BVec3Expr);
impl_bool_binop!(BVec4, BVec4Expr);

impl Vec3Expr {
    #[inline]
    pub fn cross(&self, rhs: Vec3Expr) -> Self {
        Vec3Expr::from_node(current_scope(|s| {
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

pub type Float2 = Vec2;
pub type Float3 = Vec3;
pub type Float4 = Vec4;

pub type Int2 = IVec2;
pub type Int3 = IVec3;
pub type Int4 = IVec4;

pub type Uint2 = UVec2;
pub type Uint3 = UVec3;
pub type Uint4 = UVec4;

#[inline]
pub fn make_float2<X: Into<PrimProxy<f32>>, Y: Into<PrimProxy<f32>>>(x: X, y: Y) -> Expr<Float2> {
    Expr::<Float2>::new(x.into(), y.into())
}
#[inline]
pub fn make_float3<X: Into<PrimProxy<f32>>, Y: Into<PrimProxy<f32>>, Z: Into<PrimProxy<f32>>>(
    x: X,
    y: Y,
    z: Z,
) -> Expr<Float3> {
    Expr::<Float3>::new(x.into(), y.into(), z.into())
}
#[inline]
pub fn make_float4<
    X: Into<PrimProxy<f32>>,
    Y: Into<PrimProxy<f32>>,
    Z: Into<PrimProxy<f32>>,
    W: Into<PrimProxy<f32>>,
>(
    x: X,
    y: Y,
    z: Z,
    w: W,
) -> Expr<Float4> {
    Expr::<Float4>::new(x.into(), y.into(), z.into(), w.into())
}

#[inline]
pub fn make_int2<X: Into<PrimProxy<i32>>, Y: Into<PrimProxy<i32>>>(x: X, y: Y) -> Expr<Int2> {
    Expr::<Int2>::new(x.into(), y.into())
}
#[inline]
pub fn make_int3<X: Into<PrimProxy<i32>>, Y: Into<PrimProxy<i32>>, Z: Into<PrimProxy<i32>>>(
    x: X,
    y: Y,
    z: Z,
) -> Expr<Int3> {
    Expr::<Int3>::new(x.into(), y.into(), z.into())
}
#[inline]
pub fn make_int4<
    X: Into<PrimProxy<i32>>,
    Y: Into<PrimProxy<i32>>,
    Z: Into<PrimProxy<i32>>,
    W: Into<PrimProxy<i32>>,
>(
    x: X,
    y: Y,
    z: Z,
    w: W,
) -> Expr<Int4> {
    Expr::<Int4>::new(x.into(), y.into(), z.into(), w.into())
}
#[inline]
pub fn make_uint2<X: Into<PrimProxy<u32>>, Y: Into<PrimProxy<u32>>>(x: X, y: Y) -> Expr<Uint2> {
    Expr::<Uint2>::new(x.into(), y.into())
}
#[inline]
pub fn make_uint3<X: Into<PrimProxy<u32>>, Y: Into<PrimProxy<u32>>, Z: Into<PrimProxy<u32>>>(
    x: X,
    y: Y,
    z: Z,
) -> Expr<Uint3> {
    Expr::<Uint3>::new(x.into(), y.into(), z.into())
}
#[inline]
pub fn make_uint4<
    X: Into<PrimProxy<u32>>,
    Y: Into<PrimProxy<u32>>,
    Z: Into<PrimProxy<u32>>,
    W: Into<PrimProxy<u32>>,
>(
    x: X,
    y: Y,
    z: Z,
    w: W,
) -> Expr<Uint4> {
    Expr::<Uint4>::new(x.into(), y.into(), z.into(), w.into())
}
